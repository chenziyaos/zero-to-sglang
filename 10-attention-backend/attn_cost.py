"""注意力这一层的账：为什么 decode 注意力永远吃带宽、GQA 组大小怎么决定走不走 Tensor Core、
Flash-Decoding 的 split-K 为什么在小 batch 长上下文时救命、MLA 为什么反而会变成算力瓶颈。

只用标准库。模型结构和 GPU 规格取自 ../common/specs.py。

  python3 attn_cost.py                     # decode 注意力的算术强度：MHA / GQA / MLA
  python3 attn_cost.py --scenario tc       # FlashInfer 的 Tensor Core 开关：组大小 ≥ 4
  python3 attn_cost.py --scenario split    # split-K：小 batch 长上下文时 SM 占满了没有
  python3 attn_cost.py --scenario prefill  # prefill 时注意力 FLOPs 什么时候追上线性层
  python3 attn_cost.py --scenario share    # decode 一步里注意力占多少时间

口径：
  decode 时每个请求自己读自己的 KV，batch 再大也不能复用别人的 KV，所以注意力的
  FLOP/字节只取决于一个 KV 头被多少个 Q 头共享（GQA 组大小 G）和 KV 精度。
  线性层不同：权重被整个 batch 共享，batch 越大强度越高。这是两者在 Roofline 上分家的根源。
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common.specs import GPUS, MODELS, weight_bytes  # noqa: E402

SM_COUNT = {'a100-80g': 108, 'h100-sxm': 132, 'h200': 132, 'l40s': 142, 'rtx4090': 128}


# ---------------------------------------------------------------- decode 强度

def decode_attn(m, ctx: int, tp: int = 1, kv_bytes: float = 2, s_q: int = 1):
    """一层、一个请求、一步 decode 的注意力：返回 (FLOPs, 读 KV 的字节, FLOP/字节)。

    MHA/GQA：每个 Q 头做 QK^T（2·d·ctx）和 PV（2·d·ctx），共 4·H·d·ctx；
             KV 读 2·Hkv·d·ctx 个元素。强度 = 2·G·s_q / kv_bytes，G = H / Hkv。
    MLA（吸收后）：所有 Q 头共享一份 latent（rank + rope 维），K 用 576 维、V 用 512 维；
             FLOPs = 2·H·s_q·ctx·(dk + dv)，读 ctx·dk 个元素。强度约 2·H·s_q·(dk+dv)/dk/kv_bytes/2，
             和 FlashMLA 深度解析里的 2·h_q·s_q 一致（bf16）。
    """
    h = m.heads // tp
    if m.mla_kv_rank:
        dk = m.mla_kv_rank + m.mla_rope_dim
        dv = m.mla_kv_rank
        flops = 2 * h * s_q * ctx * (dk + dv)
        byts = ctx * dk * kv_bytes
    else:
        hkv = max(1, m.kv_heads // tp)
        flops = 4 * h * s_q * m.head_dim * ctx
        byts = 2 * hkv * m.head_dim * ctx * kv_bytes
    return flops, byts, flops / byts


def linear_intensity(m, bs: int, tp: int = 1) -> float:
    """decode 时线性层的 FLOP/字节：权重被 bs 个请求共享。"""
    w = (weight_bytes(m, 'bf16') - m.embed_b * 2e9) / tp
    return 2 * m.active_b * 1e9 / tp * bs / w


def scenario_intensity():
    g = GPUS['h100-sxm']
    print(f'  H100 SXM 的 BF16 拐点 {g.ridge:.0f} FLOP/字节；低于它就是带宽瓶颈')
    print(f'  {"模型":<22}{"Q/KV 头":>9}{"组 G":>6}{"bf16 KV":>9}{"fp8 KV":>8}{"MTP s_q=2":>11}')
    for key in ['llama3.1-8b', 'qwen3-8b', 'llama3.1-70b', 'qwen3-235b-a22b', 'gpt-oss-120b', 'deepseek-v3']:
        m = MODELS[key]
        if m.mla_kv_rank:
            heads = f'{m.heads}/latent'
            grp = '-'
        else:
            heads = f'{m.heads}/{m.kv_heads}'
            grp = str(m.heads // m.kv_heads)
        a = decode_attn(m, 4096)[2]
        b = decode_attn(m, 4096, kv_bytes=1)[2]
        c = decode_attn(m, 4096, s_q=2)[2]
        print(f'  {m.name:<22}{heads:>9}{grp:>6}{a:>9.1f}{b:>8.1f}{c:>11.1f}')
    print()
    m = MODELS['qwen3-8b']
    print('  对比：Qwen3-8B 线性层在 decode 时的强度随 batch 涨，注意力不动')
    for bs in (1, 16, 64, 256):
        print(f'    bs={bs:<4} 线性层 {linear_intensity(m, bs):6.0f} FLOP/字节   注意力 {decode_attn(m, 4096)[2]:.0f}')
    print()
    print('  读法：GQA 模型 decode 注意力的强度就是 G（bf16），离拐点差一两个数量级，换什么后端都是读 KV 的速度；')
    print('  后端之间比的是能吃到多少 HBM 带宽，以及元数据准备和 launch 有多轻。')
    print('  DeepSeek-V3 不切 TP 时 128 个头共享 latent，强度约 240，贴近 H100 拐点：')
    print('  FlashMLA 解析里按 H800 降频后的 865 TFLOPS 算，h_q·s_q ≥ 128 就转成算力瓶颈。')
    print('  TP=8 时每卡 16 个头，强度掉到 30 左右，又回到带宽瓶颈。这也是 DeepSeek 推理 decode 不开 TP 的原因之一。')


# ---------------------------------------------------------------- Tensor Core

def tc_tile_fill(group: int, s_q: int = 1, tile_m: int = 16) -> float:
    """FlashInfer 的 Tensor Core decode 把同一个 KV 头下的 G 个 Q 头和 s_q 个 token 摊在 MMA 的 M 维。
    MMA 的 M 维至少 16 行（mma.m16n8k16），填不满的行算了也白算。"""
    rows = group * s_q
    return min(rows, math.ceil(rows / tile_m) * tile_m) / (math.ceil(rows / tile_m) * tile_m)


def scenario_tc(tc_over_cuda: float = 7.0):
    """tc_over_cuda：同精度下 Tensor Core 峰值是 CUDA core 的几倍。H100 上 BF16 稠密 989 TFLOPS，
    非 Tensor 的 FP16/BF16 只有百来 TFLOPS，7 倍是量级假设，换卡自己改。"""
    print('  FlashInfer 的规则（flashinfer_backend.py:2670 should_use_tensor_core）：')
    print('    环境变量 SGLANG_FLASHINFER_USE_TENSOR_CORE 优先；KV 是 fp8 一律开；fp16/bf16 时组大小 ≥ 4 才开')
    print(f'  玩具模型：Tensor Core 有效算力 = {tc_over_cuda:.0f}× CUDA core × M 维填充率')
    print(f'  {"G":>4}{"s_q":>5}{"填充率":>8}{"相对 CUDA core":>16}  默认')
    for g, sq in [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (2, 4), (1, 8)]:
        fill = tc_tile_fill(g, sq)
        on = 'prefill wrapper（TC）' if sq > 1 else ('TC' if g >= 4 else 'CUDA core')
        print(f'  {g:>4}{sq:>5}{fill:>8.0%}{tc_over_cuda * fill:>15.2f}x  {on}')
    print()
    print('  读法：decode 注意力反正是带宽瓶颈，开不开 Tensor Core 改变的是「算」那一半会不会把带宽拖慢。')
    print('  G 小时 MMA 的 16 行大半是空的，还要多做一次 Q 的重排；G ≥ 4 后 Tensor Core 才稳赚。')
    print('  这条规则只管 decode wrapper。投机验证（s_q > 1）在 FlashInfer 里走 prefill wrapper（flashinfer_backend.py:1025），')
    print('  本来就用 Tensor Core；FP8 KV 要先反量化，CUDA core 更忙，所以一律开。')
    print('  组大小按每张卡上的头数算（attn_tp_size 切过之后）；KV 头少于 TP 时 KV 头被复制，G 不会小于原值。')


# ---------------------------------------------------------------- split-K

def decode_attn_us(m, gpu: str, bs: int, ctx: int, splits: int, tp: int = 1,
                   bw_eff: float = 0.8, reduce_us: float = 3.0) -> dict:
    """一层 decode 注意力的时间，假设每个 CTA 处理 (请求, KV 头, 一段 KV)。

    能吃到的带宽 ≈ 峰值 × min(1, CTA 数 / SM 数)：一个 SM 自己拉不满整张卡的 HBM，
    CTA 数少于 SM 数时，空着的 SM 对应的那份带宽就浪费了。这是 Flash-Decoding 博客里
    「batch=1 时 FlashAttention 用不到 1% 的 GPU」的来源。
    split > 1 时多一个 reduce kernel，外加把 splits 份部分结果（fp32 O 和 LSE）写回再读出。
    """
    g = GPUS[gpu]
    sms = SM_COUNT[gpu]
    hkv = 1 if m.mla_kv_rank else max(1, m.kv_heads // tp)
    ctas = bs * hkv * splits
    occ = min(1.0, ctas / sms)
    _, byts, _ = decode_attn(m, ctx, tp)
    h = m.heads // tp
    dv = m.mla_kv_rank or m.head_dim
    partial = bs * h * splits * (dv + 1) * 4 * 2 if splits > 1 else 0
    t_mem = (bs * byts + partial) / (g.bw_tbs * 1e12 * bw_eff * occ) * 1e6
    t = t_mem + (reduce_us if splits > 1 else 0)
    return dict(ctas=ctas, occ=occ, us=t, partial_mb=partial / 1e6)


def best_splits(m, gpu: str, bs: int, ctx: int, tp: int = 1, max_splits: int = 64,
                min_tokens: int = 256) -> int:
    """在 [1, max_splits] 里挑最快的 split 数，每段至少 min_tokens 个 token。"""
    cands = [s for s in range(1, max_splits + 1) if ctx / s >= min_tokens] or [1]
    return min(cands, key=lambda s: decode_attn_us(m, gpu, bs, ctx, s, tp)['us'])


def scenario_split():
    m, gpu = MODELS['qwen3-8b'], 'h100-sxm'
    print(f'  Qwen3-8B，H100（{SM_COUNT[gpu]} 个 SM），一层 decode 注意力，KV 头 {m.kv_heads}')
    print(f'  {"bs":>4}{"ctx":>8}{"splits":>8}{"CTA":>6}{"占用":>7}{"耗时 us":>10}{"相对不切":>10}')
    for bs, ctx in [(1, 4096), (1, 32768), (1, 131072), (8, 32768), (32, 8192), (64, 4096)]:
        base = decode_attn_us(m, gpu, bs, ctx, 1)['us']
        for s in sorted({1, 8, best_splits(m, gpu, bs, ctx)}):
            r = decode_attn_us(m, gpu, bs, ctx, s)
            print(f'  {bs:>4}{ctx:>8}{s:>8}{r["ctas"]:>6}{r["occ"]:>7.0%}{r["us"]:>10.1f}{base / r["us"]:>9.1f}x')
        print()
    print('  读法：bs × KV 头数 < SM 数时，不切 KV 就有一大半 SM 闲着；切 8 份是 sglang Triton 后端的默认上限')
    print('  （--triton-attention-num-kv-splits 8，帮助里写着「上下文越长越大越好」）。')
    print('  bs 一大，CTA 自己就够填满 SM，再切只多出 reduce 和部分结果的读写，所以 split 数要随 batch 动态取。')
    print('  FlashInfer 在 plan 阶段按 batch 和长度定 split；CUDA Graph 下它预先按最大 split 开好缓冲。')


# ---------------------------------------------------------------- prefill

def scenario_prefill():
    print('  prefill 时注意力 FLOPs（因果，平均每个 token 看前面一半）和线性层 FLOPs 的比')
    print(f'  {"模型":<18}{"平衡点 ctx":>12}{"8K 时占比":>11}{"32K 时占比":>12}')
    for key in ['qwen3-8b', 'llama3.1-70b', 'qwen3-235b-a22b', 'deepseek-v3']:
        m = MODELS[key]
        if m.mla_kv_rank:
            # 训练式（非吸收）prefill：Q·K 用 192 维（128 nope + 64 rope），V 128 维
            per_pos = 2 * m.heads * (192 + 128)
        else:
            per_pos = 4 * m.heads * m.head_dim
        attn_per_tok = lambda L: per_pos * (L / 2) * m.layers  # noqa: E731
        lin = 2 * m.active_b * 1e9
        bal = lin / (per_pos / 2 * m.layers)
        f8 = attn_per_tok(8192) / (attn_per_tok(8192) + lin)
        f32 = attn_per_tok(32768) / (attn_per_tok(32768) + lin)
        print(f'  {m.name:<18}{bal:>12,.0f}{f8:>11.0%}{f32:>12.0%}')
    print()
    print('  读法：短 prompt 的 prefill 主要是 GEMM，换注意力后端几乎看不出差别；')
    print('  上下文到几万 token，注意力 FLOPs 和线性层同量级，prefill 后端（FA3 / FA4 / trtllm-gen）的 MFU 才开始决定 TTFT。')
    print('  MoE 模型激活参数少，平衡点更早到来：Qwen3-235B-A22B 和 DeepSeek-V3 在 3 万 token 左右就五五开，Llama-70B 要到 10 万。')


# ---------------------------------------------------------------- 一步里的占比

def scenario_share():
    m, g = MODELS['qwen3-8b'], GPUS['h100-sxm']
    w = weight_bytes(m, 'bf16') - m.embed_b * 2e9
    per_tok = m.kv_bytes_per_token()
    cap = (g.mem_gb * 1e9 * 0.85 - weight_bytes(m, 'bf16')) / per_tok
    print('  Qwen3-8B，H100，decode 一步：权重读取 vs KV 读取（都按 80% 带宽算）')
    print(f'  每 token KV {per_tok / 1024:.0f} KiB，按 85% 显存估 KV 池约 {cap / 1e3:.0f}K token，bs×ctx 不能超过它')
    print(f'  {"bs":>4}{"ctx":>8}{"bs×ctx":>9}{"权重 ms":>9}{"KV ms":>8}{"注意力占比":>11}')
    for bs, ctx in [(1, 4096), (8, 4096), (32, 4096), (64, 4096), (1, 131072), (8, 32768), (12, 32768)]:
        n = bs * ctx
        tw = w / (g.bw_tbs * 1e12 * 0.8) * 1e3
        tk = n * per_tok / (g.bw_tbs * 1e12 * 0.8) * 1e3
        flag = '' if n <= cap else '  放不下'
        print(f'  {bs:>4}{ctx:>8}{n:>9,}{tw:>9.2f}{tk:>8.2f}{tk / (tw + tk):>11.0%}{flag}')
    print('  读法：bs×ctx 到十万 token 量级，KV 读取就追上权重读取；KV 池塞满时注意力占一步的七成以上。')
    print('  单卡 8B 模型的 KV 池只有三四十万 token，大 batch 长上下文要靠 FP8 KV、TP 分摊或 HiCache。')
    print('  KV 占大头时，注意力后端的带宽利用率、FP8 KV、page 布局才值得花时间；小 batch 短上下文先去看 SGL-09 的 CPU 开销。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='intensity',
                    choices=['intensity', 'tc', 'split', 'prefill', 'share'])
    s = ap.parse_args().scenario
    {'intensity': scenario_intensity, 'tc': scenario_tc, 'split': scenario_split,
     'prefill': scenario_prefill, 'share': scenario_share}[s]()


if __name__ == '__main__':
    main()
