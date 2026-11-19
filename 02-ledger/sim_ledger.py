"""推理账本：按 SGLang 主干的切池逻辑，算一张卡上权重、KV 池、预留各占多少，KV 池能放多少 token。

复现的逻辑（sglang 主干提交 8fabf5e）：
  arg_groups/memory_hook.py                 reserved_mem 与 mem_fraction_static（同 01-first-server）
  mem_cache/kv_cache_configurator.py        _profile_available_bytes：rest = 加载后空闲 - 加载前空闲 x (1 - f)
  model_executor/pool_configurator.py       DefaultPoolConfigurator：MHA/GQA 每卡 KV 头数 max(1, kv_heads // tp)，
                                            MLA 每卡存完整 latent
                                            HybridSWAPoolConfigurator：cell = full + ratio x swa，ratio 默认 0.8
  mem_cache/kv_cache_configurator.py        resolve_max_num_reqs：clamp(tokens / context x 512, 2048, 4096)
  mem_cache/memory_pool.py                  ReqToTokenPool：int32 [(max_running + 1), context + 4]，不计入 KV 预算

不覆盖：投机解码的草稿层、DP Attention、PP、VLM、HiCache、--max-total-tokens 上限。

用法：
    python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
    python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm --kv-dtype fp8
    python3 sim_ledger.py --model qwen3-235b-a22b --gpu h100-sxm --tp 8 --weight-dtype fp8
    python3 sim_ledger.py --model gpt-oss-120b --gpu h100-sxm
    python3 sim_ledger.py --model gpt-oss-120b --gpu h100-sxm --swa-ratio 0.25
    python3 sim_ledger.py --model deepseek-v3 --gpu h200 --tp 8
    python3 sim_ledger.py --compare          # 打印正文里的对照表
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
from common.specs import GPUS, KV_DTYPE, MODELS, SMI_MIB, WEIGHT_DTYPE, weight_bytes  # noqa: E402
from estimate_startup import prefill_bs, tier  # noqa: E402

GiB = 1 << 30
EVICTION_INTERVAL = 128   # environ.py: SGLANG_SWA_EVICTION_INTERVAL


def reserved_mb(model, gpu_mib, tp, chunk, max_bs):
    """memory_hook.py：激活 + 并行 + CUDA Graph 预留，80 GB 档以上有 10 GB 下限。"""
    graph = max_bs * 2
    graph += 1.5 * 1024 if model.mla_kv_rank else len(prefill_bs(chunk)) * 8
    r = 512 + max(chunk, 2048) * 1.5 + tp / 8 * 1024 + graph
    if gpu_mib > 60 * 1024:
        r = max(r, 10 * 1024)
    return r


def cell_bytes(model, tp, kv_dtype, swa_ratio):
    """返回 (每 token 计价字节, 全注意力部分, SWA 部分, 每卡 KV 头数)。"""
    e = KV_DTYPE[kv_dtype]
    if model.mla_kv_rank:
        full = model.layers * (model.mla_kv_rank + model.mla_rope_dim) * e
        return full, full, 0, 0
    heads = max(1, model.kv_heads // tp)
    per_layer = heads * 2 * model.head_dim * e
    full = per_layer * model.full_layers
    swa = per_layer * model.swa_layers
    return full + swa_ratio * swa, full, swa, heads


def ledger(key, gpu_key, tp=1, weight_dtype=None, kv_dtype='bf16', mem_frac=None, swa_ratio=0.8,
           avg_ctx=4096, cuda_ctx_gib=0.6, max_running=None):
    m = MODELS[key]
    weight_dtype = weight_dtype or m.native_dtype
    gpu_mib = SMI_MIB[gpu_key]
    chunk, max_bs = tier(gpu_mib, tp)
    auto = round((gpu_mib - reserved_mb(m, gpu_mib, tp, chunk, max_bs)) / gpu_mib, 3)
    f = mem_frac if mem_frac is not None else auto

    total = gpu_mib / 1024
    pre = total - cuda_ctx_gib
    w = weight_bytes(m, weight_dtype) / tp / GiB
    after = pre - w
    slack = pre * (1 - f)
    rest = after - slack

    cell, full, swa, heads = cell_bytes(m, tp, kv_dtype, swa_ratio if m.swa_layers else 0)
    tokens = int(rest * GiB // cell) if rest > 0 else 0
    swa_tokens = int(tokens * swa_ratio) if m.swa_layers else 0

    est = max(min(int(tokens / m.context * 512), 4096), 2048)
    running = min(max_running, tokens // 2) if max_running else min(est, tokens // 2)
    r2t = (running + 1) * (m.context + 4) * 4 / GiB

    # 全注意力池决定了实际能同时驻留多少请求；每个驻留请求在 SWA 池里只占窗口 + 驱逐间隔 + 两页，
    # 再加上 overlap 调度下两个在途的 prefill chunk（同 compute_swa_request_cap 的口径）
    resident = min(running, max(tokens // avg_ctx, 1))
    swa_need = 2 * chunk + resident * (m.window + EVICTION_INTERVAL + 2) if m.swa_layers else 0
    return dict(model=m, gpu=GPUS[gpu_key], tp=tp, weight_dtype=weight_dtype, kv_dtype=kv_dtype,
                chunk=chunk, max_bs=max_bs, auto=auto, frac=f, total=total, pre=pre, w=w, after=after,
                slack=slack, rest=rest, cell=cell, full=full, swa=swa, heads=heads, tokens=tokens,
                swa_tokens=swa_tokens, swa_ratio=swa_ratio, running=running, r2t=r2t,
                swa_need=swa_need, avg_ctx=avg_ctx)


def show(L):
    m = L['model']
    print(f'{m.name} on {L["tp"]} x {L["gpu"].name}  weight={L["weight_dtype"]} kv={L["kv_dtype"]}')
    print(f'  mem_fraction_static    {L["frac"]}' + ('' if L['frac'] == L['auto'] else f'（自动值 {L["auto"]}）'))
    print('[每卡账本，GiB]')
    print(f'  可见显存               {L["total"]:7.2f}')
    print(f'  CUDA 上下文等          {L["total"] - L["pre"]:7.2f}')
    print(f'  权重                   {L["w"]:7.2f}')
    print(f'  激活与 Graph 预留      {L["slack"]:7.2f}   其中 req_to_token 表 {L["r2t"]:.2f}')
    print(f'  KV 池                  {max(L["rest"], 0):7.2f}')
    if L['rest'] <= 0:
        need = 1 - L['after'] / L['pre']
        print(f'  KV 池为负：启动会报错，mem_fraction_static 至少要 {need:.3f}，或者加卡、降权重精度')
        return
    print('[KV 池]')
    if m.mla_kv_rank:
        print(f'  MLA latent {m.mla_kv_rank}+{m.mla_rope_dim} 维 x {m.layers} 层，每卡完整一份')
    else:
        rep = max(1, L['tp'] // m.kv_heads)
        note = f'，KV 头在 {rep} 张卡上重复存' if rep > 1 else ''
        print(f'  每卡 KV 头 {L["heads"]}（共 {m.kv_heads}{note}）')
    if m.swa_layers:
        print(f'  全注意力层 {m.full_layers} 层 {L["full"] / 1024:.1f} KB/token，'
              f'滑窗层 {m.swa_layers} 层 {L["swa"] / 1024:.1f} KB/token，ratio {L["swa_ratio"]}')
    print(f'  每 token 计价          {L["cell"] / 1024:7.1f} KB')
    print(f'  max_total_num_tokens   {L["tokens"]:>9,}')
    if m.swa_layers:
        use = L['swa_need'] / max(L['swa_tokens'], 1)
        print(f'  swa_layer_tokens       {L["swa_tokens"]:>9,}   按平均 {L["avg_ctx"]} token 驻留时实际要用约 '
              f'{L["swa_need"]:,}（{use:.0%}）' + ('，偏小' if use > 1 else ''))
    print(f'  max_running_requests   {L["running"]:>9,}')
    full_req = L['tokens'] / m.context
    print(f'  满上下文 {m.context} 的请求能装 {full_req:.1f} 条'
          + ('，单条长请求会被截断或拒绝' if full_req < 1 else ''))
    print(f'  平均 {L["avg_ctx"]} token 的请求能同时驻留 {L["tokens"] // L["avg_ctx"]} 条')


CASES = [
    # (说明, model, gpu, tp, weight_dtype, kv_dtype, swa_ratio)
    ('稠密 GQA 基线', 'qwen3-8b', 'h100-sxm', 1, 'bf16', 'bf16', 0.8),
    ('KV 换 FP8', 'qwen3-8b', 'h100-sxm', 1, 'bf16', 'fp8', 0.8),
    ('70B 四卡', 'llama3.1-70b', 'h100-sxm', 4, 'bf16', 'bf16', 0.8),
    ('70B 八卡', 'llama3.1-70b', 'h100-sxm', 8, 'bf16', 'bf16', 0.8),
    ('MoE BF16 八卡', 'qwen3-235b-a22b', 'h100-sxm', 8, 'bf16', 'bf16', 0.8),
    ('MoE FP8 八卡', 'qwen3-235b-a22b', 'h100-sxm', 8, 'fp8', 'bf16', 0.8),
    ('MoE FP8 四卡', 'qwen3-235b-a22b', 'h100-sxm', 4, 'fp8', 'bf16', 0.8),
    ('SWA 默认比例', 'gpt-oss-120b', 'h100-sxm', 1, 'mxfp4', 'bf16', 0.8),
    ('SWA 比例 0.25', 'gpt-oss-120b', 'h100-sxm', 1, 'mxfp4', 'bf16', 0.25),
    ('MLA 八卡 H200', 'deepseek-v3', 'h200', 8, 'fp8', 'bf16', 0.8),
]


def compare():
    print(f'{"场景":<14}{"模型":<17}{"卡":>9}{"权重":>7}{"KV":>5}{"f":>7}{"权重GiB":>9}{"KV池GiB":>9}'
          f'{"KB/tok":>8}{"tokens":>11}{"满上下文条数":>10}')
    for name, mk, gk, tp, wd, kd, ratio in CASES:
        L = ledger(mk, gk, tp, wd, kd, swa_ratio=ratio)
        gpu = f'{tp}x{gk.split("-")[0].upper()}'
        print(f'{name:<14}{L["model"].name:<17}{gpu:>9}{wd:>7}{kd:>5}{L["frac"]:>7.3f}{L["w"]:>9.2f}'
              f'{max(L["rest"], 0):>9.2f}{L["cell"] / 1024:>8.1f}{L["tokens"]:>11,}'
              f'{L["tokens"] / L["model"].context:>10.1f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--gpu', default='h100-sxm', choices=SMI_MIB)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--weight-dtype', choices=WEIGHT_DTYPE, help='默认取官方 checkpoint 的精度')
    ap.add_argument('--kv-dtype', default='bf16', choices=KV_DTYPE)
    ap.add_argument('--mem-fraction-static', type=float)
    ap.add_argument('--swa-ratio', type=float, default=0.8, help='--swa-full-tokens-ratio')
    ap.add_argument('--max-running-requests', type=int)
    ap.add_argument('--avg-ctx', type=int, default=4096, help='每个请求平均占用的上下文长度')
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    if a.compare:
        compare()
        return
    show(ledger(a.model, a.gpu, a.tp, a.weight_dtype, a.kv_dtype, a.mem_fraction_static, a.swa_ratio,
                a.avg_ctx, max_running=a.max_running_requests))


if __name__ == '__main__':
    main()
