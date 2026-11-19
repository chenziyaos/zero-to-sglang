"""按 SGLang 主干的默认启发式，预测启动日志里的关键数字（CPU 即可运行）。

复现的逻辑（sglang 主干提交 8fabf5e）：
  arg_groups/memory_hook.py      handle_gpu_memory_settings：按显存档位定 chunked_prefill_size、
                                 decode CUDA Graph max_bs，再算 reserved_mem 和 mem_fraction_static
  arg_groups/memory_hook.py      reserve_for_graph_mb：CUDA Graph 预留
  arg_groups/cuda_graph_hook.py  prefill / decode 捕获档位
  mem_cache/kv_cache_configurator.py  _profile_available_bytes、resolve_max_num_reqs

只覆盖单机、非 MLA、无投机解码、无 DP Attention、非 VLM 的情况，够第一次上手对账。
其他情况看 SGL-02 的完整账本。

用法：
    python3 estimate_startup.py --model qwen3-8b --gpu h100-sxm
    python3 estimate_startup.py --model qwen3-8b --gpu rtx4090
    python3 estimate_startup.py --model llama3.1-70b --gpu h100-sxm --tp 4
    python3 estimate_startup.py --model qwen3-8b --gpu h100-sxm --mem-fraction-static 0.88
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.specs import GPUS, MODELS  # noqa: E402

GiB = 1 << 30

# nvidia-smi --query-gpu=memory.total 的实际读数（MiB），比标称容量略小
SMI_MIB = {'a100-80g': 81920, 'h100-sxm': 81559, 'h200': 143771, 'l40s': 46068, 'rtx4090': 24564}
# 模型的 max_position_embeddings，用于 resolve_max_num_reqs
CONTEXT = {'qwen3-0.6b': 40960, 'qwen3-8b': 40960, 'llama3.1-8b': 131072, 'llama3.1-70b': 131072}


def tier(gpu_mib, tp):
    """memory_hook.py 的显存档位：返回 (chunked_prefill_size, decode max_bs)。"""
    if gpu_mib < 20 * 1024:
        return 2048, 8
    if gpu_mib < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if gpu_mib < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if gpu_mib < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def prefill_bs(max_bs):
    s = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
         + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, max_bs + 1, 512)))
    return [x for x in s if x <= max_bs]


def decode_bs(max_bs):
    s = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, max_bs + 1, 32))
    s = [x for x in s if x <= max_bs]
    if max_bs not in s:
        s.append(max_bs)
    return s


def estimate(model, gpu_key, tp=1, mem_frac=None, chunk=None, max_bs=None, max_running=None,
             cuda_ctx_gib=0.6, kv_bytes=2):
    m = MODELS[model]
    gpu_mib = SMI_MIB[gpu_key]
    c0, b0 = tier(gpu_mib, tp)
    chunk = chunk or c0
    max_bs = max_bs or b0
    if max_running is not None and max_bs > max_running:
        max_bs = max_running                      # decode max_bs 不超过 max_running_requests
    pbs, dbs = prefill_bs(chunk), decode_bs(max_bs)

    graph_mb = max_bs * 2 + len(pbs) * 8          # reserve_for_graph_mb，非 MLA
    reserved = 512 + max(chunk, 2048) * 1.5 + tp / 8 * 1024 + graph_mb
    if gpu_mib > 60 * 1024:
        reserved = max(reserved, 10 * 1024)
    auto_frac = round((gpu_mib - reserved) / gpu_mib, 3)
    frac = mem_frac if mem_frac is not None else auto_frac

    total = gpu_mib / 1024
    pre_load = total - cuda_ctx_gib               # 加载权重前的空闲显存（Load weight begin 的 avail mem）
    weights = m.params_b * 1e9 * 2 / tp / GiB     # BF16，TP 均分（embedding、lm_head 也切）
    after = pre_load - weights                    # Load weight end 的 avail mem
    slack = pre_load * (1 - frac)                 # 留给激活和 CUDA Graph 的部分
    rest = after - slack

    kv_heads_local = max(1, m.kv_heads // tp)
    cell = 2 * m.layers * kv_heads_local * m.head_dim * kv_bytes
    tokens = int(rest * GiB // cell) if rest > 0 else 0

    ctx = CONTEXT.get(model, 32768)
    est = max(min(int(tokens / ctx * 512), 4096), 2048)
    running = min(max_running, tokens // 2) if max_running else min(est, tokens // 2)
    return dict(gpu_mib=gpu_mib, chunk=chunk, max_bs=max_bs, n_prefill_bs=len(pbs), n_decode_bs=len(dbs),
                graph_mb=graph_mb, reserved_mb=reserved, auto_frac=auto_frac, frac=frac,
                pre_load=pre_load, weights=weights, after=after, slack=slack, rest=rest,
                cell=cell, tokens=tokens, k_gib=tokens * cell / 2 / GiB, ctx=ctx, running=running)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen3-8b', choices=[k for k in MODELS if k in CONTEXT])
    ap.add_argument('--gpu', default='h100-sxm', choices=SMI_MIB)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--mem-fraction-static', type=float)
    ap.add_argument('--chunked-prefill-size', type=int)
    ap.add_argument('--cuda-graph-max-bs-decode', '--cuda-graph-max-bs', dest='cuda_graph_max_bs', type=int)
    ap.add_argument('--max-running-requests', type=int)
    ap.add_argument('--kv-cache-dtype', default='bf16', choices=['bf16', 'fp8'])
    ap.add_argument('--cuda-ctx-gib', type=float, default=0.6, help='CUDA 上下文、NCCL 等加载前已占用的显存')
    a = ap.parse_args()
    r = estimate(a.model, a.gpu, a.tp, a.mem_fraction_static, a.chunked_prefill_size, a.cuda_graph_max_bs,
                 a.max_running_requests, a.cuda_ctx_gib, 1 if a.kv_cache_dtype == 'fp8' else 2)
    if r['weights'] >= r['pre_load']:
        print('权重放不下这张卡，增大 --tp')
        return
    print(f'{MODELS[a.model].name} on {a.tp} x {GPUS[a.gpu].name}（nvidia-smi {r["gpu_mib"]} MiB）')
    print('[参数推导]')
    print(f'  chunked_prefill_size      {r["chunk"]}')
    print(f'  decode cuda graph max_bs  {r["max_bs"]}（{r["n_decode_bs"]} 个捕获档）')
    print(f'  prefill cuda graph 档数   {r["n_prefill_bs"]}')
    print(f'  graph 预留                {r["graph_mb"]:.0f} MB')
    print(f'  reserved_mem              {r["reserved_mb"]:.0f} MB')
    print(f'  mem_fraction_static       {r["frac"]}' + ('' if r['frac'] == r['auto_frac'] else f'（自动值 {r["auto_frac"]}）'))
    print('[预测日志]')
    print(f'  Load weight begin. avail mem={r["pre_load"]:.2f} GB')
    print(f'  Load weight end. ... avail mem={r["after"]:.2f} GB, mem usage={r["weights"]:.2f} GB.')
    print(f'  KV Cache is allocated. #tokens: {r["tokens"]}, K size: {r["k_gib"]:.2f} GB, V size: {r["k_gib"]:.2f} GB')
    print(f'  max_total_num_tokens={r["tokens"]}, chunked_prefill_size={r["chunk"]}, '
          f'max_running_requests={r["running"]}, context_len={r["ctx"]}')
    print('[对账提示]')
    print(f'  每 token KV {r["cell"] / 1024:.0f} KB/卡；留给激活和 Graph 的 slack {r["slack"]:.2f} GB')
    print(f'  一条满上下文请求要 {r["ctx"]} token，池子能装 {r["tokens"] / r["ctx"]:.1f} 条')


if __name__ == '__main__':
    main()
