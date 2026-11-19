"""SGL-23：给一张卡算出 Qwen3-32B 该用哪种精度、要几张卡、KV 能放多少 token，并给出启动命令。

只用标准库。数字和文档第 2、3、8 章一致：
  Qwen3-32B：64 层，8 个 KV 头，head_dim 128，KV 每 token BF16 256 KiB、FP8 128 KiB
  checkpoint：BF16 65.52 GB、FP8 34.32 GB、AWQ 19.33 GB（Hugging Face 文件体积）
  默认 mem_fraction_static 按 sglang arg_groups/memory_hook.py 的公式复算
  KV 预算按 kv_cache_configurator.py 的 _profile_available_bytes 近似：
    KV = 加载前空闲显存 × mem_fraction_static − 每卡权重
这些是估算值，真实容量以启动日志里的 max_total_num_tokens 为准。

用法：
  python3 plan.py --gpu h20-96
  python3 plan.py --gpu h20-96 --n 2 --precision bf16
  python3 plan.py --gpu rtx4090
  python3 plan.py --gpu a100-80 --target-tokens 131072
  python3 plan.py --qwen38 --gpu h200 --avg-len 9216 --spec38 eagle
"""
import argparse
import sys
from dataclasses import dataclass

GIB = 1 << 30
MIB = 1 << 20

# Qwen3-32B config.json
LAYERS = 64
KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 5120
INTERMEDIATE = 25600
Q_HEADS = 64
MAX_POS = 40960

# Hugging Face 上的 checkpoint 体积，单位 GB（10^9 字节）
CKPT_GB = {'bf16': 65.52, 'fp8': 34.32, 'awq': 19.33}
REPO = {'bf16': 'Qwen/Qwen3-32B', 'fp8': 'Qwen/Qwen3-32B-FP8', 'awq': 'Qwen/Qwen3-32B-AWQ'}
EAGLE3_DRAFT = 'AngelSlim/Qwen3-32B_eagle3'
EAGLE3_GB = 1.46
CONTEXT_OVERHEAD_MIB = 800     # CUDA context、NCCL 等加载前就占掉的显存，估值
FP8_BLOCK = 128                # Qwen3-32B-FP8 的权重分块 [128, 128]


def kv_bytes_per_token(kv_dtype='bf16', tp=1):
    """每卡每 token 的 KV 字节数。KV 头按 TP 切分，TP 超过 KV 头数时每卡复制一个头。"""
    heads = max(1, KV_HEADS // tp)
    elem = 1 if kv_dtype == 'fp8' else 2
    return 2 * LAYERS * heads * HEAD_DIM * elem


@dataclass(frozen=True)
class GPU:
    name: str
    sm: int          # 计算能力，89 表示 8.9
    mib: int         # nvidia-smi 报的显存总量，MiB，取常见值
    bw: float        # 显存带宽 TB/s


GPUS = {
    'b200': GPU('B200', 100, 183359, 8.0),
    'h200': GPU('H200', 90, 143771, 4.8),
    'h20-141': GPU('H20 141G', 90, 143771, 4.8),
    'h20-96': GPU('H20 96G', 90, 97871, 4.0),
    'h100': GPU('H100 80G', 90, 81559, 3.35),
    'a100-80': GPU('A100 80G', 80, 81920, 2.0),
    'a100-40': GPU('A100 40G', 80, 40960, 1.56),
    'l40s': GPU('L40S', 89, 46068, 0.86),
    'rtx-pro-6000': GPU('RTX PRO 6000 96G', 120, 97887, 1.79),
    'rtx5090': GPU('RTX 5090', 120, 32607, 1.79),
    'rtx4090': GPU('RTX 4090', 89, 24564, 1.0),
    'rtx3090': GPU('RTX 3090', 86, 24576, 0.94),
}

# Qwen 官方速度基准：H20 96G，SGLang 0.4.6.post1，batch 1，生成 2048 token。
# 值为 (输入 + 输出) / 秒，输入 1 时约等于单流 decode 速度。
QWEN_SPEED_H20 = {
    1: {'bf16': 20.72, 'fp8': 46.17, 'awq': 47.67},
    6144: {'bf16': 77.82, 'fp8': 165.71, 'awq': 159.99},
    14336: {'bf16': 143.08, 'fp8': 287.60, 'awq': 260.44},
    30720: {'bf16': 240.75, 'fp8': 436.59, 'awq': 366.84},
    63488: {'bf16': 342.96, 'fp8': 532.18, 'awq': 425.23},
}


# ---------------- sglang 默认值复算（arg_groups/memory_hook.py、cuda_graph_hook.py）----------------

def capacity_defaults(mib, tp=1):
    """按显存大小给出默认 chunked_prefill_size 和 decode CUDA Graph max_bs。"""
    if mib < 20 * 1024:
        return 2048, 8
    if mib < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if mib < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if mib < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def prefill_graph_bs(max_bs):
    sizes = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
             + list(range(576, 1025, 64)) + list(range(1280, 4097, 256))
             + list(range(4608, max_bs + 1, 512)))
    return [s for s in sizes if s <= max_bs]


def decode_graph_bs(max_bs, spec=False):
    if spec:
        sizes = (list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 65, 4))
                 + list(range(72, 257, 8)) + list(range(272, max_bs + 1, 16)))
    else:
        sizes = ([1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16))
                 + list(range(512, max_bs + 1, 32)))
    sizes = [s for s in sizes if s <= max_bs]
    if max_bs not in sizes:
        sizes.append(max_bs)
    return sizes


def default_mem_fraction(mib, tp=1, chunk=None, max_bs=None, prefill_graph=True):
    """复算默认 mem_fraction_static，返回 (mem_fraction, reserved_mib, chunk, max_bs, 预填图档数)。"""
    d_chunk, d_bs = capacity_defaults(mib, tp)
    chunk = chunk or d_chunk
    max_bs = max_bs or d_bs
    n_prefill = len(prefill_graph_bs(chunk)) if prefill_graph else 0
    reserved = 512 + max(chunk, 2048) * 1.5 + tp / 8 * 1024 + max_bs * 2 + n_prefill * 8
    if mib > 60 * 1024:
        reserved = max(reserved, 10 * 1024)
    return round((mib - reserved) / mib, 3), reserved, chunk, max_bs, n_prefill


def weights_mib(precision, tp=1, spec=False):
    w = CKPT_GB[precision] * 1e9 / tp
    if spec:
        w += EAGLE3_GB * 1e9          # 草稿模型不切分，每卡一份
    return w / MIB


def kv_budget_mib(gpu, precision, tp=1, mem_frac=None, spec=False, **kw):
    if mem_frac is None:
        mem_frac = default_mem_fraction(gpu.mib, tp, **kw)[0]
    pre = gpu.mib - CONTEXT_OVERHEAD_MIB
    return pre * mem_frac - weights_mib(precision, tp, spec), mem_frac


def kv_tokens(gpu, precision, tp=1, kv_dtype='bf16', mem_frac=None, spec=False, **kw):
    kv, mf = kv_budget_mib(gpu, precision, tp, mem_frac, spec, **kw)
    return max(0, int(kv * MIB / kv_bytes_per_token(kv_dtype, tp))), mf


def fp8_tp_ok(tp, intermediate=INTERMEDIATE):
    """FP8 分块权重要求按 TP 切开后每片仍是 128 的整数倍。
    列并行切 gate/up 和 Q，行并行切 down 和 o_proj 的输入维，都落在 intermediate 和 Q_HEADS×HEAD_DIM 上；
    KV 头不够分时每卡复制整头，不会切出半块。"""
    dims = [intermediate, Q_HEADS * HEAD_DIM]
    return all(d % tp == 0 and (d // tp) % FP8_BLOCK == 0 for d in dims)


def decode_ceiling(gpu, precision, tp=1):
    """batch 1 的 decode 理论上限：每步把本卡权重读一遍。"""
    return gpu.bw * 1e12 / (CKPT_GB[precision] * 1e9 / tp)


def fp8_kernel(sm):
    if sm >= 90 and sm != 120:
        return 'w8a8 分块 FP8（Hopper 走 DeepGEMM，Blackwell 数据中心卡走 FlashInfer）'
    if sm == 120:
        return 'w8a8 分块 FP8（SM120 走 CUTLASS）'
    if sm == 89:
        return 'w8a8 分块 FP8，Ada 走 Triton 实现'
    return 'w8a16，FP8 Marlin 只量化权重'


TUNED_CHUNK = 2048   # 小卡把 chunked_prefill_size 压到 2048，sglang 会按它重算默认 mem_fraction_static


def kv_tokens_profile(gpu, precision, tp, kv_dtype='bf16', spec=False, tuned=False):
    """tuned=False 用全默认；tuned=True 只显式给 --chunked-prefill-size 2048，mem_fraction 仍由 sglang 推。"""
    return kv_tokens(gpu, precision, tp, kv_dtype, spec=spec, chunk=TUNED_CHUNK if tuned else None)


def choose(gpu, target_tokens, kv_dtype='bf16', precision=None, spec=False):
    """先少卡，再按架构偏好的精度，最后才动 chunked_prefill_size。返回 (精度, tp, token, mem_frac, tuned)。"""
    if precision:
        order = [precision]
    elif gpu.sm >= 89:
        order = ['fp8', 'awq']
    else:
        order = ['awq', 'fp8']
    for tp in (1, 2, 4, 8):
        for p in order:
            if p == 'fp8' and not fp8_tp_ok(tp):
                continue
            for tuned in (False, True):
                t, mf = kv_tokens_profile(gpu, p, tp, kv_dtype, spec, tuned)
                if t >= target_tokens:
                    return p, tp, t, mf, tuned
    p = order[-1]
    t, mf = kv_tokens_profile(gpu, p, 8, kv_dtype, spec, True)
    return p, 8, t, mf, True


def fit_context(tokens, cap=32768):
    """KV 至少要放下一条完整请求，取不超过 token 数的 2 的幂，下限 4096。"""
    c = cap
    while c > 4096 and c > tokens:
        c //= 2
    return c


def command(precision, tp, context=32768, chunk=None, spec=False, kv_dtype='bf16', mem_frac=None):
    lines = [f'sglang serve {REPO[precision]}']
    if tp > 1:
        lines.append(f'--tp {tp}')
    lines.append('--reasoning-parser qwen3 --tool-call-parser qwen25')
    lines.append(f'--context-length {context}')
    if context > MAX_POS:
        lines.append('--json-model-override-args \'{"rope_scaling":{"rope_type":"yarn","factor":%.1f,'
                     '"original_max_position_embeddings":32768}}\'' % (context / 32768))
    if mem_frac:
        lines.append(f'--mem-fraction-static {mem_frac}')
    if chunk:
        lines.append(f'--chunked-prefill-size {chunk}')
    if kv_dtype == 'fp8':
        lines.append('--kv-cache-dtype fp8_e4m3')
    if spec:
        lines.append(f'--speculative-algorithm EAGLE3 --speculative-draft-model-path {EAGLE3_DRAFT}')
        lines.append('--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4')
    lines.append('--host 0.0.0.0 --port 30000')
    return ' \\\n  '.join(lines)


# ---------------- Qwen3.8-27B：GDN 状态槽和 KV 的配比 ----------------

Q38_GDN_LAYERS = 48
Q38_ATTN_LAYERS = 16
Q38_V_HEADS = 48
Q38_QK_HEADS = 16
Q38_HEAD_DIM = 128
Q38_ATTN_KV_HEADS = 4
Q38_ATTN_HEAD_DIM = 256
Q38_CONV_KERNEL = 4
Q38_WEIGHTS_GB = {'fp8': 28.5, 'nvfp4': 16.5}
SLOTS = {'extra_buffer': 5, 'extra_buffer_lazy': 4, 'no_buffer': 3, 'radix_off': 1}
SPEC_SLOTS = {'none': 0, 'eagle': 4, 'dflash': 8, 'dspark': 8}


def q38_state_bytes(ssm_dtype='float32'):
    ssm = Q38_GDN_LAYERS * Q38_V_HEADS * Q38_HEAD_DIM * Q38_HEAD_DIM * (4 if ssm_dtype == 'float32' else 2)
    conv_dim = 2 * Q38_QK_HEADS * Q38_HEAD_DIM + Q38_V_HEADS * Q38_HEAD_DIM
    conv = Q38_GDN_LAYERS * (Q38_CONV_KERNEL - 1) * conv_dim * 2
    return ssm + conv


def q38_kv_bytes(kv_dtype='fp8'):
    return Q38_ATTN_LAYERS * Q38_ATTN_KV_HEADS * Q38_ATTN_HEAD_DIM * 2 * (1 if kv_dtype == 'fp8' else 2)


def q38_ratio(avg_len, strategy='extra_buffer', spec='none', ssm_dtype='float32', kv_dtype='fp8'):
    """--mamba-full-memory-ratio 的平衡值：(S + D) × 状态字节 / (L × 每 token KV 字节)。"""
    s, dd = SLOTS[strategy], SPEC_SLOTS[spec]
    return (s + dd) * q38_state_bytes(ssm_dtype) / (avg_len * q38_kv_bytes(kv_dtype))


def q38_capacity(rest_gib, ratio, strategy='extra_buffer', spec='none', ssm_dtype='float32', kv_dtype='fp8'):
    """给定剩余显存和配比，估算能并发的请求数和 KV token 数。"""
    mamba = rest_gib * GIB * ratio / (1 + ratio)
    kv = rest_gib * GIB / (1 + ratio)
    slots = mamba / q38_state_bytes(ssm_dtype)
    per_req = SLOTS[strategy] + SPEC_SLOTS[spec]
    return int(slots / per_req), int(kv / q38_kv_bytes(kv_dtype))


# ---------------- 报告 ----------------

def report_32b(a):
    gpu = GPUS[a.gpu]
    if a.n:
        p = a.precision or choose(gpu, 0, a.kv, spec=a.spec)[0]
        tp = a.n
        t, mf = kv_tokens_profile(gpu, p, tp, a.kv, a.spec, False)
        tuned = False
        if t < a.target_tokens:
            t2, mf2 = kv_tokens_profile(gpu, p, tp, a.kv, a.spec, True)
            if t2 > t:
                t, mf, tuned = t2, mf2, True
    else:
        p, tp, t, mf, tuned = choose(gpu, a.target_tokens, a.kv, a.precision, a.spec)
    chunk_arg = TUNED_CHUNK if tuned else None
    if a.mem_frac:
        t, mf = kv_tokens(gpu, p, tp, a.kv, mem_frac=a.mem_frac, spec=a.spec, chunk=chunk_arg)
    _, reserved, chunk, max_bs, n_pf = default_mem_fraction(gpu.mib, tp, chunk=chunk_arg)
    kv_mib, _ = kv_budget_mib(gpu, p, tp, mem_frac=a.mem_frac, spec=a.spec, chunk=chunk_arg)
    print(f'卡型：{gpu.name}，SM{gpu.sm}，{gpu.mib} MiB，带宽 {gpu.bw} TB/s')
    print(f'选择：{p.upper()}，TP{tp}，FP8 内核：{fp8_kernel(gpu.sm) if p == "fp8" else "不涉及"}')
    tag = '（显式压小）' if tuned else '（默认）'
    print(f'chunked_prefill_size {chunk}{tag}，decode 图 max_bs {max_bs}，预填图 {n_pf} 档')
    if a.mem_frac:
        print(f'显式 mem_fraction_static {mf}（sglang 默认会推 {default_mem_fraction(gpu.mib, tp, chunk=chunk_arg)[0]}）')
    else:
        print(f'sglang 推出的 mem_fraction_static {mf}，预留 {reserved / 1024:.1f} GiB 给激活和 CUDA Graph')
    print(f'每卡权重 {weights_mib(p, tp, a.spec) / 1024:.1f} GiB，KV 预算 {kv_mib / 1024:.1f} GiB')
    per_tok = kv_bytes_per_token(a.kv, tp)
    print(f'KV 每 token 每卡 {per_tok // 1024} KiB，可放约 {t:,} token（{EST_TAG}）')
    if t <= 0:
        print('提示：KV 预算为负，启动会报 Not enough memory，换精度或加卡')
    elif t < a.target_tokens:
        more = '加卡' if a.kv == 'fp8' else '加 --kv fp8 或加卡'
        print(f'提示：低于目标 {a.target_tokens:,} token，可{more}')
    print(f'batch 1 decode 理论上限 {decode_ceiling(gpu, p, tp):.0f} tok/s')
    if a.gpu == 'h20-96' and tp == 1:
        r = QWEN_SPEED_H20[1].get(p)
        if r:
            print(f'Qwen 官方 H20 实测（0.4.6.post1，batch 1）：{r} tok/s')
    print()
    ctx = a.context or fit_context(t)
    if ctx > t > 0:
        print(f'提示：--context-length {ctx} 大于 KV 容量，长请求会排队或被拒')
    print(command(p, tp, ctx, chunk_arg, a.spec, a.kv, a.mem_frac))


EST_TAG = '估算'


def report_38(a):
    gpu = GPUS[a.gpu]
    ratio = q38_ratio(a.avg_len, a.strategy, a.spec38, a.ssm_dtype, a.kv)
    print(f'Qwen3.8-27B，平均请求长度 {a.avg_len}，{a.strategy}，投机 {a.spec38}，状态 {a.ssm_dtype}，KV {a.kv}')
    print(f'每请求状态 {q38_state_bytes(a.ssm_dtype) / 1e6:.1f} MB，KV 每 token {q38_kv_bytes(a.kv) / 1024:.0f} KiB，'
          f'一个状态槽折合 {q38_state_bytes(a.ssm_dtype) / q38_kv_bytes(a.kv):.0f} token')
    print(f'平衡配比 --mamba-full-memory-ratio {ratio:.2f}（默认 0.9）')
    w = Q38_WEIGHTS_GB[a.q38_precision] * 1e9 / GIB
    rest = (gpu.mib - CONTEXT_OVERHEAD_MIB) / 1024 * a.mem_frac - w
    for name, r in (('默认 0.9', 0.9), ('平衡值', ratio)):
        req, tok = q38_capacity(rest, r, a.strategy, a.spec38, a.ssm_dtype, a.kv)
        print(f'{name}：剩余 {rest:.1f} GiB，约 {req} 个并发请求，KV 约 {tok:,} token（{EST_TAG}）')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True, choices=sorted(GPUS))
    p.add_argument('--n', type=int, default=0, help='卡数，不给就自动找最少卡数')
    p.add_argument('--precision', choices=sorted(CKPT_GB))
    p.add_argument('--kv', choices=['bf16', 'fp8'], help='KV cache 精度，Qwen3-32B 默认 bf16，Qwen3.8 默认 fp8')
    p.add_argument('--context', type=int, default=0, help='--context-length，不给就按 KV 容量取')
    p.add_argument('--target-tokens', type=int, default=65536, help='每个 TP 组至少要放下的 KV token 数')
    p.add_argument('--spec', action='store_true', help='加 EAGLE3 草稿模型')
    p.add_argument('--qwen38', action='store_true', help='改算 Qwen3.8-27B 的状态槽配比')
    p.add_argument('--avg-len', type=int, default=9216, help='平均请求总长度（输入加输出）')
    p.add_argument('--strategy', choices=sorted(SLOTS), default='extra_buffer')
    p.add_argument('--spec38', choices=sorted(SPEC_SLOTS), default='none')
    p.add_argument('--ssm-dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--q38-precision', choices=sorted(Q38_WEIGHTS_GB), default='fp8')
    p.add_argument('--mem-frac', type=float, help='显式 --mem-fraction-static；Qwen3.8 不给时按 0.85')
    a = p.parse_args(argv)
    if a.qwen38:
        a.kv = a.kv or 'fp8'
        a.mem_frac = a.mem_frac or 0.85
        report_38(a)
    else:
        a.kv = a.kv or 'bf16'
        report_32b(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
