"""并行这一层的账：一张卡上放多少权重、能存多少 token 的 KV、每层要搬多少字节。

只用标准库。模型结构和 GPU 规格取自 ../common/specs.py，切分规则按 sglang 8fabf5e 的
deepseek_v2.py、dp_attention.py、runtime_context.py 重写。

  python3 sim_parallel.py                      # 每卡权重、每卡 KV、整组能存多少 token
  python3 sim_parallel.py --scenario layout    # rank 编号：(dp, cp, tp) 三维展开和参数校验
  python3 sim_parallel.py --scenario shard     # 切完之后的维度能不能被 FP8 块大小 128 整除
  python3 sim_parallel.py --scenario comm      # 每层通信字节：TP all-reduce / DP gather / DeepEP
  python3 sim_parallel.py --scenario padding   # DP gather 选 MAX_LEN 还是 SUM_LEN
  python3 sim_parallel.py --scenario deepep    # DeepEP low_latency 的 RDMA buffer 有多大

口径：
  tp_size 在 sglang 里是整个模型并行组的宽度（一个节点组里的 GPU 数），attention 和 FFN
  在这个组里各自再切：attention 用 attn_tp = tp / attn_dp / attn_cp，路由专家用 ep 和
  moe_tp = tp / ep，稠密 MLP 用 moe_dense_tp（只能是 1 或 tp）。
  KV 预算 = 显存 × mem_frac − 本卡权重，粗估，不扣激活、CUDA Graph 和通信 buffer，
  实测以启动日志里的 max_total_num_tokens 为准。
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common.specs import GPUS, KV_DTYPE, MODELS, WEIGHT_DTYPE  # noqa: E402

# config.json 里 specs.py 没收的几个维度
ARCH = {
    # deepseek-ai/DeepSeek-V3：q_lora_rank 1536，qk_nope 128，qk_rope 64，v_head 128，
    # 稠密层 intermediate 18432，专家 intermediate 2048，1 个共享专家，前 3 层稠密，n_group 8
    'deepseek-v3': dict(q_lora=1536, qk_nope=128, qk_rope=64, v_head=128, dense_inter=18432,
                        moe_inter=2048, shared=1, groups=8),
    'qwen3-235b-a22b': dict(moe_inter=1536, shared=0, dense_inter=0, groups=0),
    'llama3.1-70b': dict(dense_inter=28672, moe_inter=0, shared=0, groups=0),
    'qwen3-8b': dict(dense_inter=12288, moe_inter=0, shared=0, groups=0),
}

# tie_word_embeddings=True：lm_head 和输入 embedding 共用一张表
TIED = {'qwen3-0.6b'}

A2A_SPANNING = {'deepep', 'deepep_v2', 'mooncake', 'nixl', 'mori', 'pplx', 'flashinfer', 'megamoe'}


@dataclass(frozen=True)
class Par:
    tp: int
    attn_dp: int = 1
    attn_cp: int = 1
    ep: int = 1
    dp: int = 1                      # 整份模型的副本数，和 attn_dp 互斥
    moe_dense_tp: Optional[int] = None
    a2a: str = 'none'
    dp_lm_head: bool = False
    redundant: int = 0               # ep_num_redundant_experts
    embed_rep: bool = False          # SGLANG_ENABLE_EMBED_REPLICATION

    @property
    def attn_tp(self) -> int:
        return self.tp // self.attn_dp // self.attn_cp

    @property
    def moe_tp(self) -> int:
        return self.tp // self.ep


class ConfigError(ValueError):
    pass


def resolve(m, p: Par) -> Par:
    """按启动时的校验顺序走一遍，返回改写后的配置；不合法时抛 ConfigError。

    parallel_hook.py:162 / :287、overrides.py:1612、validation_hook.py:207、
    eplb/expert_location.py:256（物理专家数要能被 ep 整除）的组合。
    """
    if p.attn_dp > 1 and p.dp > 1:
        raise ConfigError('--dp-size 和 --attn-dp-size 不能同时大于 1')
    if p.tp % (p.attn_dp * p.attn_cp):
        raise ConfigError(f'tp {p.tp} 不能被 attn_dp×attn_cp = {p.attn_dp * p.attn_cp} 整除')
    if p.a2a in A2A_SPANNING and p.ep != p.tp:
        p = replace(p, ep=p.tp)            # 日志：expert parallel size is adjusted from ...
    if p.moe_dense_tp not in (None, 1, p.tp):
        raise ConfigError(f'moe_dense_tp 只能是 1 或 tp，给的是 {p.moe_dense_tp}')
    if p.tp % p.ep:
        raise ConfigError(f'tp {p.tp} 不能被 ep {p.ep} 整除')
    if m.heads % p.attn_tp:
        raise ConfigError(f'{m.heads} 个注意力头分不到 attn_tp={p.attn_tp} 张卡上')
    if m.experts and (m.experts + p.redundant) % p.ep:
        raise ConfigError(f'{m.experts}+{p.redundant} 个物理专家不能均分到 ep={p.ep}')
    return p


# ---------------------------------------------------------------- 权重切分

def components(m):
    """把模型参数拆成按不同规则切分的几块，单位：个参数。"""
    key = next(k for k, v in MODELS.items() if v is m)
    a = ARCH.get(key, {})
    H, L = m.hidden, m.layers
    moe_layers = m.moe_layers
    dense_layers = L - moe_layers
    out = {}
    if m.mla_kv_rank:
        qk = a['qk_nope'] + a['qk_rope']
        # fused_qkv_a_proj_with_mqa 是 ReplicatedLinear；q_b / kv_b / o 按 attn_tp 切
        out['attn_rep'] = L * H * (a['q_lora'] + m.mla_kv_rank + m.mla_rope_dim)
        out['attn_shard'] = L * (a['q_lora'] * m.heads * qk
                                 + m.mla_kv_rank * m.heads * (a['qk_nope'] + a['v_head'])
                                 + m.heads * a['v_head'] * H)
        out['attn_kv'] = 0
    else:
        out['attn_rep'] = 0
        out['attn_shard'] = L * 2 * H * m.heads * m.head_dim          # q_proj + o_proj
        out['attn_kv'] = L * 2 * H * m.kv_heads * m.head_dim          # k_proj + v_proj
    out['dense'] = dense_layers * 3 * H * a.get('dense_inter', 0)
    out['shared'] = moe_layers * 3 * H * a.get('moe_inter', 0) * a.get('shared', 0)
    out['routed'] = m.routed_b * 1e9
    out['embed'] = m.vocab * H
    out['lm_head'] = 0 if key in TIED else m.vocab * H
    return out


def weights_per_gpu(m, p: Par, dtype: str) -> dict:
    """每张卡上各块权重的字节数。embedding 和 lm_head 按 BF16 计（FP8 checkpoint 也不量化它们）。"""
    c = components(m)
    dense_b, expert_b = WEIGHT_DTYPE[dtype]
    kv_local = max(1, m.kv_heads // p.attn_tp) if m.kv_heads else 0
    w = {}
    w['attention'] = (c['attn_rep'] + c['attn_shard'] / p.attn_tp
                      + c['attn_kv'] * kv_local / max(1, m.kv_heads)) * dense_b
    dense_tp = p.tp if p.moe_dense_tp in (None, p.tp) else 1
    w['dense_mlp'] = c['dense'] / dense_tp * dense_b
    # 共享专家：a2a 后端下 parallel_group="replicated"（deepseek_v2.py:874-897）
    w['shared_exp'] = c['shared'] / (1 if p.a2a != 'none' else p.tp) * dense_b
    if m.experts:
        per_expert = c['routed'] / m.experts
        w['routed_exp'] = per_expert * (m.experts + p.redundant) / p.ep / p.moe_tp * expert_b
    else:
        w['routed_exp'] = 0
    # 输入 embedding：get_embedding_tp_kwargs，DP attention 下只在 attn_tp 组里切
    emb_div = 1 if p.embed_rep else (p.attn_tp if p.attn_dp > 1 else p.tp)
    w['embedding'] = c['embed'] / emb_div * 2
    lm_div = p.attn_tp if p.dp_lm_head else p.tp
    w['lm_head'] = c['lm_head'] / lm_div * 2
    return w


def kv_per_gpu_per_token(m, p: Par, kv_bytes: float = 2) -> float:
    """一个 token 在本卡上占多少 KV 字节。

    GQA：本卡 KV 头数 = max(1, kv_heads // attn_tp)（model_config.py:1718），头数少于卡数时复制。
    MLA：latent 不分头，attn_tp 组里每张卡都存一整份。
    """
    if m.mla_kv_rank:
        return m.layers * (m.mla_kv_rank + m.mla_rope_dim) * kv_bytes
    return 2 * m.layers * max(1, m.kv_heads // p.attn_tp) * m.head_dim * kv_bytes


def kv_dup(m, p: Par) -> float:
    """同一份 KV 在 attn_tp 组里被存了几遍。"""
    full = m.kv_bytes_per_token(1)
    return kv_per_gpu_per_token(m, p, 1) * p.attn_tp / full


def capacity(m, p: Par, gpu: str, dtype: str, kv: str = 'bf16', mem_frac: float = 0.85):
    g = GPUS[gpu]
    w = sum(weights_per_gpu(m, p, dtype).values())
    budget = g.mem_gb * 1e9 * mem_frac - w
    per_tok = kv_per_gpu_per_token(m, p, KV_DTYPE[kv])
    tok_gpu = max(0, budget) / per_tok
    groups = p.attn_dp * p.dp
    return dict(weight_gb=w / 1e9, budget_gb=budget / 1e9, kv_kb=per_tok / 1024,
                tok_gpu=tok_gpu, tok_total=tok_gpu * groups, dup=kv_dup(m, p))


# ---------------------------------------------------------------- rank 布局

def rank_table(tp: int, attn_dp: int, attn_cp: int = 1, ep: int = 1):
    """runtime_context.py:79 derive_attention_ranks 与 :106 derive_spawn_ranks。"""
    attn_tp = tp // attn_dp // attn_cp
    rows = []
    for r in range(tp):
        rows.append(dict(tp_rank=r, attn_dp_rank=r // (attn_tp * attn_cp),
                         attn_cp_rank=(r // attn_tp) % attn_cp, attn_tp_rank=r % attn_tp,
                         moe_ep_rank=r // (tp // ep), moe_tp_rank=r % (tp // ep)))
    return rows


# ---------------------------------------------------------------- 通信

def ring_allreduce(nbytes: float, n: int) -> float:
    """ring all-reduce 每卡发送的字节数。"""
    return 2 * (n - 1) / n * nbytes if n > 1 else 0


def ring_allgather(total_bytes: float, n: int) -> float:
    """all-gather / reduce-scatter：总 buffer 为 total_bytes 时每卡发送的字节数。"""
    return (n - 1) / n * total_bytes if n > 1 else 0


def deepep_msg_bytes(hidden: int):
    """DeepEP v1.2.1 csrc/config.hpp LowLatencyLayout：一条 dispatch / combine 消息的字节数。"""
    scales = hidden // 128
    dispatch = 16 + max(hidden * 2, hidden + scales * 4)     # FP8 数据 + 每 128 通道一个 float 缩放
    combine = scales * 4 + hidden * 2                         # BF16 数据 + 每 128 通道 min/max
    return dispatch, combine


def padding_mode(tokens: List[int], extend: bool = False, a2a: str = 'none', spec: bool = False,
                 car_max_bytes: float = 16 * 1024 * 1024, hidden: int = 7168,
                 dtype_bytes: int = 2) -> str:
    """dp_attention.py:165 get_dp_padding_mode 的判断顺序。"""
    width = len(tokens)
    if a2a == 'pplx':
        return 'MAX_LEN'                       # 对称 collective，空闲 rank 不发信号会死锁
    if extend and width > 1:
        return 'SUM_LEN'
    fits = (width > 1 and min(tokens) > 0 and a2a == 'none' and not spec
            and sum(tokens) * hidden * dtype_bytes <= car_max_bytes)
    if fits:
        return 'SUM_LEN'                       # 一次 custom all-reduce 装得下
    return 'MAX_LEN' if sum(tokens) * 2 >= max(tokens) * width else 'SUM_LEN'


def dp_gather_bytes(tokens: List[int], hidden: int, mode: str, dtype_bytes: int = 2) -> dict:
    """DP attention 进出 FFN 的两次集合通信，每卡发送字节。

    MAX_LEN：每个 DP 槽位补齐到 max，all_gather_into_tensor 进，reduce-scatter 出
             （layer_boundary/exit.py，满足 can_use_dp_reduce_scatter 时）。
    SUM_LEN：按 sum 开 buffer，每个 rank 只填自己那段，用 all-reduce 当 gather；
             出去也是 all-reduce，再 dp_scatter 取回自己的行。
    """
    n = len(tokens)
    if mode == 'MAX_LEN':
        buf = max(tokens) * n * hidden * dtype_bytes
        return dict(gather=ring_allgather(buf, n), scatter=ring_allgather(buf, n))
    buf = sum(tokens) * hidden * dtype_bytes
    return dict(gather=ring_allreduce(buf, n), scatter=ring_allreduce(buf, n))


# DeepEP v1.2.1 README，H800 + CX7 400G。low_latency：128 token/rank，top-8，FP8 dispatch / BF16 combine
LL_MEASURED = {8: (77, 114), 16: (118, 195), 32: (155, 273), 64: (173, 314), 128: (192, 369),
               256: (194, 360)}
LL_BW = {8: (98, 127), 16: (63, 74), 32: (48, 53), 64: (43, 46), 128: (39, 39), 256: (39, 40)}
# normal：4096 token，瓶颈带宽（GB/s）
NORMAL_BW = {8: (153, 158), 16: (43, 43), 32: (58, 57), 64: (51, 50)}


def nearest(table: dict, ep: int):
    k = min(table, key=lambda x: (abs(math.log2(x) - math.log2(ep)), x))
    return table[k]


def deepep_wire_bytes(hidden: int):
    """线上真正搬的字节：dispatch 是 FP8 数据 + 每 128 通道一个 float 缩放，combine 是 BF16。
    deepep_msg_bytes 是 buffer 槽位大小（要能装下 BF16 dispatch），比这个大。"""
    return hidden + hidden // 128 * 4, hidden * 2


def deepep_bytes(tokens_per_rank: int, hidden: int, topk: int):
    """README 的带宽口径：每个 token 的 topk 份拷贝都算（含发给本卡的）。"""
    d, c = deepep_wire_bytes(hidden)
    return tokens_per_rank * topk * d, tokens_per_rank * topk * c


def ll_buffer_bytes(n_tokens: int, hidden: int, num_experts: int) -> int:
    """DeepEP v1.2.1 get_low_latency_rdma_size_hint：和 rank 数无关，只看 E × N。"""
    d, c = deepep_msg_bytes(hidden)
    send = max(n_tokens * d, num_experts * n_tokens * c)
    recv = max(num_experts * n_tokens * d, num_experts * n_tokens * c)
    sig = math.ceil(num_experts * 4 / 128) * 128
    total = 2 * send + 2 * recv + 2 * sig
    return (total + 128) // 128 * 128


# ---------------------------------------------------------------- 场景

def fmt_gb(x):
    return f'{x:7.1f}'


def scenario_memory():
    cases = [
        ('deepseek-v3', 'h200', 'fp8', [
            ('TP8', Par(tp=8)),
            ('TP8 + attn_dp8', Par(tp=8, attn_dp=8, dp_lm_head=True)),
            ('TP8 + attn_dp8 + dense DP', Par(tp=8, attn_dp=8, moe_dense_tp=1, dp_lm_head=True)),
            ('DP8 + DeepEP EP8', Par(tp=8, attn_dp=8, a2a='deepep', moe_dense_tp=1,
                                     dp_lm_head=True)),
        ]),
        ('deepseek-v3', 'h100-sxm', 'fp8', [
            ('TP16（2 节点）', Par(tp=16)),
            ('DP16 + DeepEP EP16', Par(tp=16, attn_dp=16, a2a='deepep', moe_dense_tp=1,
                                       dp_lm_head=True)),
            ('DP32 + EP32 + 32 冗余', Par(tp=32, attn_dp=32, a2a='deepep', moe_dense_tp=1,
                                         dp_lm_head=True, redundant=32)),
        ]),
        ('qwen3-235b-a22b', 'h100-sxm', 'bf16', [
            ('TP8', Par(tp=8)),
            ('TP8 + attn_dp2', Par(tp=8, attn_dp=2)),
            ('TP8 + EP8', Par(tp=8, ep=8)),
        ]),
        ('llama3.1-70b', 'h100-sxm', 'bf16', [
            ('TP8', Par(tp=8)),
            ('TP8 + attn_dp2', Par(tp=8, attn_dp=2)),
            ('DP2 × TP4（两份副本）', Par(tp=4, dp=2)),
        ]),
    ]
    for key, gpu, dt, rows in cases:
        m = MODELS[key]
        tot = sum(components(m).values()) / 1e9
        print(f'\n{m.name}  {dt} 权重  {GPUS[gpu].name}  （按结构累加 {tot:.1f}B 参数）')
        print(f'  {"配置":<28}{"attn":>7}{"dense":>7}{"共享":>7}{"路由":>7}{"emb+lm":>8}'
              f'{"合计GB":>8}{"KV/tok":>9}{"复制":>6}{"每卡 tok":>11}{"整组 tok":>11}')
        for name, p in rows:
            p = resolve(m, p)
            w = weights_per_gpu(m, p, dt)
            c = capacity(m, p, gpu, dt)
            print(f'  {name:<28}{w["attention"] / 1e9:7.1f}{w["dense_mlp"] / 1e9:7.2f}'
                  f'{w["shared_exp"] / 1e9:7.2f}{w["routed_exp"] / 1e9:7.1f}'
                  f'{(w["embedding"] + w["lm_head"]) / 1e9:8.2f}{c["weight_gb"]:8.1f}'
                  f'{c["kv_kb"]:8.1f}K{c["dup"]:5.0f}x{c["tok_gpu"] / 1e3:10.0f}k'
                  f'{c["tok_total"] / 1e3:10.0f}k')
    print('\n  KV/tok 是本卡每 token 的 KV（BF16），复制 = 同一份 KV 在 attn_tp 组里存了几遍；')
    print('  整组 tok = 每卡 tok × DP 组数，是整个部署同时能挂住的 token 上限（mem_frac=0.85 粗估）。')


def scenario_layout():
    print('tp=8, attn_dp=4, attn_cp=1, ep=8：tp_rank = (dp·cp + cp_rank)·attn_tp + attn_tp_rank')
    print(f'  {"tp_rank":>8}{"attn_dp":>9}{"attn_tp":>9}{"moe_ep":>8}{"moe_tp":>8}')
    for r in rank_table(8, 4, 1, 8):
        print(f'  {r["tp_rank"]:>8}{r["attn_dp_rank"]:>9}{r["attn_tp_rank"]:>9}'
              f'{r["moe_ep_rank"]:>8}{r["moe_tp_rank"]:>8}')
    print('\ntp=16, attn_dp=2, attn_cp=2（attn_tp=4）：')
    print('  ' + '  '.join(f'{r["tp_rank"]}:{r["attn_dp_rank"]}{r["attn_cp_rank"]}{r["attn_tp_rank"]}'
                           for r in rank_table(16, 2, 2)))
    print('  （rank:dp cp tp，tp 变化最快）')
    print('\n参数校验：')
    m = MODELS['deepseek-v3']
    tries = [
        ('--tp 8 --dp 2 --attn-dp-size 2', Par(tp=8, dp=2, attn_dp=2)),
        ('--tp 8 --attn-dp-size 3', Par(tp=8, attn_dp=3)),
        ('--tp 8 --moe-dense-tp-size 2', Par(tp=8, moe_dense_tp=2)),
        ('--tp 16 --moe-a2a-backend deepep --ep 4', Par(tp=16, attn_dp=16, a2a='deepep', ep=4)),
        ('--tp 24 --moe-a2a-backend deepep', Par(tp=24, attn_dp=24, a2a='deepep')),
        ('--tp 24 --moe-a2a-backend deepep --ep-num-redundant-experts 32',
         Par(tp=24, attn_dp=24, a2a='deepep', redundant=32)),
        ('--tp 6（DeepSeek 128 头）', Par(tp=6)),
    ]
    for flag, p in tries:
        try:
            q = resolve(m, p)
            note = f'通过，ep={q.ep}，attn_tp={q.attn_tp}，moe_tp={q.moe_tp}'
        except ConfigError as e:
            note = f'拒绝：{e}'
        print(f'  {flag:<66}{note}')


def scenario_shard():
    print('FP8 块量化（128×128）要求：列切后的输出维、行切后的输入维都是 128 的倍数（fp8.py:593-609）')
    dims = [
        ('DeepSeek-V3 稠密 MLP', 18432), ('DeepSeek-V3 专家', 2048),
        ('Qwen3-235B 专家', 1536), ('Llama-3.1-70B MLP', 28672),
    ]
    tps = [2, 4, 8, 16, 32]
    print(f'  {"矩阵":<22}' + ''.join(f'{"tp" + str(t):>10}' for t in tps))
    for name, d in dims:
        cells = []
        for t in tps:
            x = d / t
            ok = x == int(x) and int(x) % 128 == 0
            cells.append(f'{int(x) if x == int(x) else x:>8}{"" if ok else "✗":>2}')
        print(f'  {name:<22}' + ''.join(cells))
    print('\n  ✗ 的组合启动时报 not divisible by weight quantization block_n = 128。')
    print('  出路：专家改走 EP（moe_tp=1，专家不切），稠密层 --moe-dense-tp-size 1，或者降 TP。')
    print('\n稠密 FFN 的显存最优 TP（LMSYS 大规模 EP 博客的模型）：')
    print('  Memory = N_param / TP + (1+k)·N_hidden·DP，取 DP = TP 时最优 TP = sqrt(N_param / ((1+k)·N_hidden))')
    n_param = 18432 * 7168
    for phase, tok, k in (('prefill', 2048, 0), ('prefill', 8192, 0), ('decode', 128, 3),
                          ('decode', 32, 3)):
        opt = math.sqrt(n_param / ((1 + k) * tok * 7168))
        print(f'  {phase:<8} 每卡 {tok:>5} token，k={k}：最优 TP ≈ {opt:4.1f}')


def scenario_comm(tok_rank: int = 128, nvlink_gbs: float = 150, rdma_gbs: float = 45):
    m = MODELS['deepseek-v3']
    H, topk = m.hidden, m.topk
    print(f'DeepSeek-V3 一个 MoE 层，每张卡 {tok_rank} 个 token（decode），BF16 激活')
    print(f'  时间口径：TP / DP gather 按节点内 {nvlink_gbs:.0f} GB/s 算；DeepEP 行用 README 在 H800 + CX7 上的')
    print('  low_latency 实测带宽（EP72 取 EP64 那一档），EP8 走 NVLink，EP32 起跨节点走 RDMA\n')
    rows = []
    # 1. 纯 TP8：8 张卡看到同一批 8×tok_rank 个 token
    S = 8 * tok_rank * H * 2
    rows.append(('TP8', 'attn 后 all-reduce + MoE 后 all-reduce',
                 2 * ring_allreduce(S, 8), 2 * ring_allreduce(S, 8) / (nvlink_gbs * 1e9) * 1e6))
    # 2. DP attention + MoE TP，均匀
    for label, toks in (('DP-attn8 + MoE TP8 均匀', [tok_rank] * 8),
                        ('DP-attn8 + MoE TP8 不均', [tok_rank * 4, tok_rank * 2, tok_rank, tok_rank // 2,
                                                    tok_rank // 2, 0, 0, 0])):
        mode = padding_mode(toks, a2a='none', car_max_bytes=0)   # 关掉 custom AR 捷径看代价规则
        b = dp_gather_bytes(toks, H, mode)
        tot = b['gather'] + b['scatter']
        rows.append((label, f'{mode}：gather + scatter', tot, tot / (nvlink_gbs * 1e9) * 1e6))
    # 3. DeepEP
    d, c = deepep_bytes(tok_rank, H, topk)
    for ep in (8, 32, 72):
        bd, bc = nearest(LL_BW, ep)
        us = d / (bd * 1e9) * 1e6 + c / (bc * 1e9) * 1e6
        rows.append((f'DP{ep} + DeepEP EP{ep}', 'dispatch(FP8) + combine(BF16)', d + c, us))
    print(f'  {"配置":<26}{"做法":<34}{"每卡发送 MB":>12}{"估算 μs":>10}')
    for name, how, b, us in rows:
        print(f'  {name:<26}{how:<34}{b / 1e6:12.2f}{us:10.0f}')
    print('\n  同样 8 卡、同样总 token：TP8 两次 all-reduce，DP attention 均匀时换成')
    print('  all-gather + reduce-scatter，字节减半；负载严重不均时 MAX_LEN 的补零变贵，自动切 SUM_LEN。')
    print('\n对账 DeepEP README 的 low_latency 实测（128 token/rank，top-8，hidden 7168）：')
    d, c = deepep_bytes(128, 7168, 8)
    print(f'  每卡 dispatch 128×8×{deepep_wire_bytes(7168)[0]} B = {d / 1e6:.2f} MB，combine 128×8×{deepep_wire_bytes(7168)[1]} B = {c / 1e6:.2f} MB')
    print(f'  {"EP":>5}{"实测 dispatch":>14}{"按带宽反推":>12}{"实测 combine":>14}{"按带宽反推":>12}')
    for ep, (td, tc) in LL_MEASURED.items():
        bd, bc = LL_BW[ep]
        print(f'  {ep:>5}{td:>12}μs{d / (bd * 1e9) * 1e6:>10.0f}μs{tc:>12}μs{c / (bc * 1e9) * 1e6:>10.0f}μs')
    print('  反推和实测吻合，说明 README 的带宽就是「每卡 token × topk × 消息字节 ÷ 时延」。')


def scenario_padding():
    cases = [
        ('decode 均匀', [64] * 8, dict()),
        ('decode 均匀但 buffer 超过 custom AR', [512] * 8, dict()),
        ('decode 一卡独大', [512, 8, 8, 8, 8, 8, 8, 8], dict()),
        ('decode 有空闲 rank', [256, 256, 256, 0, 256, 256, 256, 256], dict()),
        ('prefill 混在批里', [4096, 100, 2000, 0, 0, 0, 0, 0], dict(extend=True)),
        ('pplx 后端', [64, 1, 1, 1, 1, 1, 1, 1], dict(a2a='pplx')),
        ('开了投机解码', [64] * 8, dict(spec=True)),
    ]
    print(f'  {"情形":<34}{"每 rank token":<42}{"模式":<9}{"每卡发送 MB":>12}')
    for name, toks, kw in cases:
        mode = padding_mode(toks, **kw)
        b = dp_gather_bytes(toks, 7168, mode)
        print(f'  {name:<34}{str(toks):<42}{mode:<9}{(b["gather"] + b["scatter"]) / 1e6:12.2f}')
    print('\n  规则：pplx 强制 MAX_LEN；带 prefill 用 SUM_LEN；decode 的 sum buffer 塞得进')
    print('  custom all-reduce（16 MB，且无空闲 rank、无 a2a、无投机）用 SUM_LEN；')
    print('  其余按代价比：sum×2 ≥ max×width 选 MAX_LEN，打平也选 MAX_LEN（对称内存）。')


def scenario_deepep():
    print('DeepEP low_latency 的 RDMA buffer（v1.2.1 LowLatencyLayout，奇偶双缓冲）')
    print(f'  {"num_max_dispatch_tokens":>24}{"专家数":>8}{"GB/卡":>9}')
    for n in (64, 128, 256, 512):
        for e in (256, 288):
            print(f'  {n:>24}{e:>8}{ll_buffer_bytes(n, 7168, e) / 1e9:9.2f}')
    print('\n  只和专家数 × 每 rank 最大 token 数有关，和 EP 规模无关。')
    print('  sglang 默认 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128，上限 1024；')
    print('  decode 每卡的 batch（含 CUDA Graph 的最大 bs）不能超过它。')
    d, c = deepep_msg_bytes(7168)
    print(f'\n  消息字节：dispatch {d}（16 + 7168 FP8 + 56 个 float 缩放，和 BF16 的 14336 取大），'
          f'combine {c}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='memory',
                    choices=['memory', 'layout', 'shard', 'comm', 'padding', 'deepep'])
    ap.add_argument('--tokens', type=int, default=128, help='comm 场景每卡 token 数')
    a = ap.parse_args()
    if a.scenario == 'comm':
        scenario_comm(a.tokens)
    else:
        globals()['scenario_' + a.scenario]()


if __name__ == '__main__':
    main()
