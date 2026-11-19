"""量化省下来的显存和带宽，各自换成了什么：权重体积、KV 池容量、decode 每步耗时。

三个场景：
  size     各量化格式的真实每参数位宽（含 scale / 零点），按"只量化 Linear、embedding 和
           lm_head 保持 BF16"的口径算 checkpoint 体积，和 Hugging Face 上的实际文件大小对账
  kv       权重格式 x KV 格式 两两组合，在一张 / 两张 H100 上 KV 池能放多少 token
           （复用 02-ledger 的切池逻辑；FP4 KV 按 0.5625 字节/元素另算）
  decode   Roofline 估 decode 一步 Linear 的耗时，看 W4A16 / W8A16 / W8A8 相对 BF16 的加速比
           怎么随 batch 衰减，再把 KV 读取算进去看端到端还剩多少

对应源码（sglang 8fabf5e，路径相对 python/sglang/srt/）：
  layers/quantization/marlin_utils.py        Marlin W4A16 / W8A16，group 128 的 scale 按 FP16 存
  layers/quantization/fp8.py                 FP8 块量化 scale 是 float32，[N/128, K/128]
  mem_cache/kv_cache_dtype.py                --kv-cache-dtype fp8_e4m3 / nvfp4 映射到的 torch dtype
  model_executor/pool_configurator.py        KV 每 token 字节数 = 层数 x KV 头 x head_dim x 2 x 元素字节

用法：
    python3 mem_sim.py                     # 默认 size
    python3 mem_sim.py --scenario kv
    python3 mem_sim.py --scenario decode
"""
import argparse
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
from common.specs import GPUS, MODELS  # noqa: E402
from sim_ledger import ledger  # noqa: E402

GiB = 1 << 30

# 每个权重元素的平均位宽，scale 和零点摊进去。group 128 的 scale 是 FP16（16 bit / 128 元素），
# AWQ 的零点按 4 bit 打包；MXFP4 每 32 个元素一个 E8M0；NVFP4 每 16 个元素一个 E4M3，
# 外加一个可忽略的全局 FP32；FP8 块量化每 128x128 一个 FP32
FORMAT_BITS = {
    'bf16': 16.0,
    'fp8-channel': 8.0,                       # 每输出通道一个 FP32，摊到 K 个元素上可以忽略
    'fp8-block': 8.0 + 32 / (128 * 128),
    'gptq-int4-g128': 4.0 + 16 / 128,         # 对称，零点固定 8，不存
    'awq-int4-g128': 4.0 + 16 / 128 + 4 / 128,
    'mxfp4': 4.0 + 8 / 32,
    'nvfp4': 4.0 + 8 / 16,
}

# Hugging Face 上实际 checkpoint 的 safetensors 总大小（GB，1e9），用来校准上面的口径
HF_SIZES = [
    ('llama3.1-70b', 'bf16', 141.11, 'meta-llama/Llama-3.1-70B-Instruct'),
    ('llama3.1-70b', 'awq-int4-g128', 39.77, 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'),
    ('qwen3-8b', 'fp8-block', 9.44, 'Qwen/Qwen3-8B-FP8'),
    ('qwen3-8b', 'awq-int4-g128', 6.10, 'Qwen/Qwen3-8B-AWQ'),
]

# KV 每元素字节：fp4 是 4 bit 数据 + 每 16 个元素一个 FP8 scale = 0.5 + 1/16
KV_BYTES = {'bf16': 2.0, 'fp8': 1.0, 'fp4': 0.5625}


def unquantized_b(m):
    """不量化的部分（十亿参数）：输入 embedding + lm_head。tie 的模型 embed_b 填 0，lm_head 照算一次。"""
    return m.embed_b + m.vocab * m.hidden / 1e9


def checkpoint_gb(key, fmt):
    """按"只量化 Linear"口径算 checkpoint 体积（GB，1e9）。norm、bias、router 量级太小，忽略。"""
    m = MODELS[key]
    keep = unquantized_b(m)
    quant = m.params_b - keep
    return keep * 2 + quant * FORMAT_BITS[fmt] / 8


def scenario_size():
    print('每参数平均位宽（含 scale / 零点）')
    for f, b in FORMAT_BITS.items():
        print(f'  {f:16s} {b:6.3f} bit   相对 BF16 压缩 {16 / b:4.2f}x')
    print('\n和 Hugging Face 实际文件对账（embedding / lm_head 保持 BF16）')
    rows = []
    for key, fmt, real, repo in HF_SIZES:
        est = checkpoint_gb(key, fmt)
        rows.append((key, fmt, est, real))
        print(f'  {repo:52s} 估算 {est:7.2f} GB  实际 {real:7.2f} GB  偏差 {100 * (est / real - 1):+5.1f}%')
    print('\n常见模型各格式体积（GB）')
    keys = ['qwen3-8b', 'llama3.1-70b', 'qwen3-235b-a22b']
    fmts = ['bf16', 'fp8-block', 'awq-int4-g128', 'nvfp4']
    print('  ' + ' ' * 18 + ''.join(f'{f:>16s}' for f in fmts))
    for k in keys:
        print(f'  {MODELS[k].name:18s}' + ''.join(f'{checkpoint_gb(k, f):16.1f}' for f in fmts))
    print("""
  公式的口径就三条：Linear 按格式位宽算，scale 摊进位宽，embedding 和 lm_head 留 BF16。
  四个真实 checkpoint 的偏差都在 0.5% 以内。INT4 的名义压缩比是 4x，实际 70B 只有 3.5x，
  差的部分是 group scale（每 128 个权重多 16 bit）和 2.1B 参数的 embedding / lm_head。
  小模型更吃亏：Qwen3-8B 的 embedding + lm_head 占了 15% 参数，AWQ 只压到 BF16 的 37%。""")
    return rows


def kv_tokens(key, gpu, tp, wfmt, kvfmt):
    """KV 池 token 数。ledger 负责 reserved / mem_fraction 那套算法，这里只替换权重体积和 KV 元素字节。"""
    L = ledger(key, gpu, tp=tp, weight_dtype='bf16', kv_dtype='bf16')
    w = checkpoint_gb(key, wfmt) * 1e9 / tp / GiB
    rest = L['rest'] + L['w'] - w
    if rest <= 0:
        return 0, w, rest
    cell = L['cell'] * KV_BYTES[kvfmt] / KV_BYTES['bf16']
    return int(rest * GiB // cell), w, rest


def scenario_kv():
    out = {}
    for key, gpu, tp in [('llama3.1-70b', 'h100-sxm', 1), ('llama3.1-70b', 'h100-sxm', 2),
                         ('qwen3-8b', 'h100-sxm', 1), ('qwen3-8b', 'l40s', 1)]:
        print(f'\n{MODELS[key].name} 在 {tp} 张 {GPUS[gpu].name} 上，KV 池 token 数')
        print(f'  {"权重格式":14s} {"每卡权重":>8s} {"留给 KV":>8s}' + ''.join(f'{"KV " + k:>12s}' for k in KV_BYTES))
        for wf in ['bf16', 'fp8-block', 'awq-int4-g128']:
            cells = []
            for kf in KV_BYTES:
                t, w, rest = kv_tokens(key, gpu, tp, wf, kf)
                out[(key, gpu, tp, wf, kf)] = t
                cells.append(f'{t:12,d}' if t else f'{"放不下":>9s}')
            print(f'  {wf:16s} {w:6.1f}G {max(rest, 0):7.1f}G' + ''.join(cells))
    print("""
  权重量化先决定"能不能放下"，再决定"KV 剩多少"。70B 单卡只有 INT4 放得下；两卡时 BF16
  每卡 65.8G 权重，扣掉激活和 CUDA Graph 预留后一个 token 都放不下，换 FP8 权重才能起，
  而且直接有 20 万 token 的 KV 池。KV 从 BF16 换 FP8 是严格的
  2 倍 token，FP4 是 3.56 倍，代价是精度（见 quant_sim.py --scenario kv 和正文里的 AIME 表）。
  小模型权重本来就小，量化权重带来的 KV 增量有限，KV 量化才是扩容主力。""")
    return out


@dataclass(frozen=True)
class Card:
    name: str
    bw_tbs: float
    bf16_tflops: float
    fp8_tflops: float


# A10 是 Marlin 论文的主测卡：600 GB/s，BF16 稠密 125 TFLOPS，没有 FP8
CARDS = {
    'a10': Card('A10', 0.6, 125, 0),
    'h100-sxm': Card('H100 SXM', GPUS['h100-sxm'].bw_tbs, GPUS['h100-sxm'].bf16_tflops,
                     GPUS['h100-sxm'].fp8_tflops),
}

# 每种方案：(权重位宽, 计算用的峰值字段)
SCHEMES = {
    'bf16': (16.0, 'bf16'),
    'w8a16': (8.0, 'bf16'),                   # FP8 Marlin：权重 1 字节，kernel 里反量化成 BF16 算
    'w8a8-fp8': (8.0, 'fp8'),                 # 激活也量化，用 FP8 Tensor Core
    'w4a16': (FORMAT_BITS['gptq-int4-g128'], 'bf16'),
}


def linear_time_us(card, n, k, batch, scheme):
    """一个 [batch, k] x [k, n] 的 GEMM，按 Roofline 取访存和计算的较大者。激活读写计入。"""
    bits, peak = SCHEMES[scheme]
    tflops = card.fp8_tflops if peak == 'fp8' else card.bf16_tflops
    if tflops == 0:
        return None
    act = batch * (k + n) * (1 if peak == 'fp8' else 2)
    byts = n * k * bits / 8 + act
    flops = 2 * batch * n * k
    return max(byts / (card.bw_tbs * 1e12), flops / (tflops * 1e12)) * 1e6


def knee_batch(card, scheme):
    """Roofline 拐点：加速比开始下降的 batch。"""
    bits, peak = SCHEMES[scheme]
    tflops = card.fp8_tflops if peak == 'fp8' else card.bf16_tflops
    return tflops / card.bw_tbs * bits / 8 / 2


def decode_step_ms(card, key, batch, ctx, scheme, kvfmt='bf16'):
    """整模型 decode 一步：所有 Linear 的权重读一遍 + 每个请求的 KV 读一遍 + 计算。"""
    m = MODELS[key]
    bits, peak = SCHEMES[scheme]
    tflops = card.fp8_tflops if peak == 'fp8' else card.bf16_tflops
    quant = (m.params_b - unquantized_b(m)) * 1e9
    keep = unquantized_b(m) * 1e9 - m.embed_b * 1e9   # lm_head 每步都要读，输入 embedding 只取几行
    wbytes = quant * bits / 8 + keep * 2
    kvbytes = batch * ctx * m.kv_bytes_per_token(KV_BYTES[kvfmt])
    flops = 2 * batch * (quant + keep) + 4 * batch * ctx * m.layers * m.heads * m.head_dim
    t_mem = (wbytes + kvbytes) / (card.bw_tbs * 1e12)
    t_cmp = flops / (tflops * 1e12)
    return max(t_mem, t_cmp) * 1e3


def scenario_decode():
    n, k = 4096 * 3, 4096          # 8B 模型 gate_up 的一个典型形状
    batches = [1, 8, 16, 32, 64, 128, 256, 512]
    res = {}
    for ck, card in CARDS.items():
        print(f'\n{card.name}：单个 Linear [{n} x {k}]，相对 BF16 的加速比（Roofline 理想值）')
        print('  batch     ' + ''.join(f'{b:>7d}' for b in batches) + '   拐点 batch')
        for s in ['w4a16', 'w8a16', 'w8a8-fp8']:
            row = []
            for b in batches:
                base = linear_time_us(card, n, k, b, 'bf16')
                t = linear_time_us(card, n, k, b, s)
                sp = base / t if t else None
                res[(ck, s, b)] = sp
                row.append(f'{sp:7.2f}' if sp else '      -')
            kb = knee_batch(card, s) if SCHEMES[s][1] == 'bf16' or card.fp8_tflops else None
            print(f'  {s:10s}' + ''.join(row) + (f'   {kb:6.0f}' if kb else '   无 FP8'))
    print('\nLlama-3.1-8B 在 A10 上 decode 一步（上下文 2048，KV BF16）')
    print('  batch     ' + ''.join(f'{b:>7d}' for b in [1, 8, 16, 32, 64]))
    for s in ['bf16', 'w4a16', 'w8a16']:
        row = [decode_step_ms(CARDS['a10'], 'llama3.1-8b', b, 2048, s) for b in [1, 8, 16, 32, 64]]
        print(f'  {s:8s}ms' + ''.join(f'{t:7.1f}' for t in row))
    e2e = [decode_step_ms(CARDS['a10'], 'llama3.1-8b', b, 2048, 'bf16') /
           decode_step_ms(CARDS['a10'], 'llama3.1-8b', b, 2048, 'w4a16') for b in [1, 8, 16, 32, 64]]
    res['e2e_w4a16_a10'] = e2e
    print('  W4A16 端到端加速' + ''.join(f'{x:7.2f}' for x in e2e))
    print("""
  单个 Linear 上，W4A16 的理想加速是 16 / 4.125 = 3.9x，一直保持到权重访存不再是瓶颈。
  拐点 batch = 峰值算力 / 带宽 x 权重字节 / 2：A10 的 BF16 拐点是 208，INT4 权重把它压到
  54 左右，过了这个 batch 加速比一路降到 1。Marlin 论文实测在 batch 16 到 32 之前接近 4x，
  64 到 128 逐步回落，比理想拐点早，原因是反量化指令和 scale 读取吃掉了一部分算力。
  W8A8 和 W8A16 在小 batch 一样快（都只省一半带宽），区别在大 batch：W8A8 用 FP8 Tensor Core，
  计算峰值翻倍，prefill 和大 batch decode 才看得出差距。
  端到端还要再打折：权重量化不压 KV，batch 越大 KV 读取占比越高，加速比就越接近 1。
  论文里 vLLM 端到端最高 2.8x，就是这个折扣的结果。""")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='size', choices=['size', 'kv', 'decode'])
    a = ap.parse_args()
    {'size': scenario_size, 'kv': scenario_kv, 'decode': scenario_decode}[a.scenario]()


if __name__ == '__main__':
    main()
