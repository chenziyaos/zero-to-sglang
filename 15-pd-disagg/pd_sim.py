"""PD 分离：KV 搬多少、走几个描述符、P:D 怎么配、什么时候拆反而更慢、decode 侧怎么准入。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  disaggregation/common/conn.py:1432        _resolve_rank_mapping：P、D 的 attn TP 相等 / D 大 / D 小三种映射，
                                            MLA 在 D 小时只从一个 P rank 取、其余发 dummy
  disaggregation/common/utils.py:132        group_concurrent_contiguous：src、dst 两边同时连续才能合成一段
  disaggregation/mooncake/conn.py:1028      _send_kvcache_generic：每段每层一个描述符（MLA L 层，MHA K、V 共 2L）
  disaggregation/mooncake/conn.py:1656      send_kvcache_slice：TP 不等时每个 token 槽、每层一个描述符
  disaggregation/common/staging_buffer.py:773  compute_head_slice_params：异构 TP 下每个 rank 发哪几个 KV 头
  disaggregation/common/staging_buffer.py:809  compute_staging_layout：P 大 D 小时几个 writer、各写多少字节
  disaggregation/common/staging_buffer.py:860  staging_grid_tokens / compute_grid_segments：按 chunk 宽度切格子
  disaggregation/prefill.py:1460            maybe_send_cached_prefix_chunk：显存里的前缀页对齐后先发
  disaggregation/prefill.py:1528            _send_kv_chunk：非最后一块扣掉不满的尾页，staging 下再对齐到格子
  disaggregation/decode.py:511              _required_admission_tokens：max(分配 + 预留, 输入 - 前缀 + 输出 - 可回收)
  disaggregation/decode.py:1336             pop_preallocated：FIFO 预分配，放不下就 break（队头阻塞）
  disaggregation/decode.py:2030-2140        _need_space_for_single_req / _active_req_count / _allocatable_token_budgets
  disaggregation/utils.py:300               _all_reduce_polls：各 rank 的 KVPoll 取 MIN
  mem_cache/allocator/token.py              need_sort=True：free 先进 release_pages，不够时合并排序（PD 打开）

  python3 pd_sim.py                          # transfer：每请求 KV 字节、各链路耗时、chunk 流水暴露的尾巴、描述符数
  python3 pd_sim.py --scenario ratio         # ratio：按 TTFT / TPOT SLO 算 P:D 配比和每卡 goodput
  python3 pd_sim.py --scenario colo          # colo：两卡合并部署 vs 1P1D，长 prompt、短请求、TCP 三种负载
  python3 pd_sim.py --scenario admission     # admission：decode 准入预算、512 × 活跃请求、队头阻塞
  python3 pd_sim.py --scenario timeline      # timeline：一条请求的 KVPoll 状态、分块发送边界、两种分配策略占槽时长
"""
import argparse
import math
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
sys.path.insert(0, os.path.join(HERE, '..', '06-chunked-prefill'))

from common.specs import KV_DTYPE, MODELS  # noqa: E402
from sim_ledger import ledger  # noqa: E402
from sim_serve import Server  # noqa: E402
import sim_stats  # noqa: E402

CLIP = 4096                  # CLIP_MAX_NEW_TOKEN
RESERVED = 512               # --num-reserved-decode-tokens 默认
GB = 1e9

# 每个 rank 实际拿到的单向带宽（GB/s），估计值，换成自己机器的实测
LINKS = OrderedDict([
    ('NVLink 机内', 150.0),        # Mooncake NVLink 传输，H100 机内 NVLink 理论 450 GB/s 单向
    ('RDMA 400G/卡', 40.0),        # 每卡一张 400Gbps 网卡，扣协议开销
    ('RDMA 200G/卡', 20.0),
    ('TCP 100GbE÷8', 1.2),         # 一张 100GbE 网卡被 8 个 rank 分，mooncake_tcp 调试用
])

CONFIGS = [
    ('llama3.1-8b', 'h100-sxm', 1),
    ('llama3.1-70b', 'h100-sxm', 4),
    ('qwen3-235b-a22b', 'h100-sxm', 8),
    ('deepseek-v3', 'h200', 8),
]


def name_of(model, gpu, tp):
    return f'{MODELS[model].name} {tp}×{gpu.split("-")[0].upper()}'


# ---------------------------------------------------------------- 搬运原语（照源码）

def group_contiguous(src, dst):
    """common/utils.py:132。返回 [(src 起点, dst 起点, 长度)]。"""
    if not src or not dst:
        return []
    if len(src) != len(dst):
        raise ValueError('group_concurrent_contiguous requires equal-length src/dst index arrays, '
                         f'got {len(src)} and {len(dst)}')
    out, s0, d0, n = [], src[0], dst[0], 1
    for i in range(1, len(src)):
        if src[i] - src[i - 1] != 1 or dst[i] - dst[i - 1] != 1:
            out.append((s0, d0, n))
            s0, d0, n = src[i], dst[i], 0
        n += 1
    out.append((s0, d0, n))
    return out


def rank_mapping(d_rank, d_tp, p_tp, mla=False, mla_sharding=False):
    """common/conn.py:1432 的 TP 部分。返回 (target_tp_ranks, target_tp_rank, required_dst_info_num,
    required_prefill_response_num, warn)。"""
    warn = None
    if mla and mla_sharding:
        assert d_tp == p_tp, 'MLA KV cache sharding requires equal prefill/decode attn TP sizes'
        return list(range(p_tp)), None, d_tp, p_tp, warn
    if d_tp == p_tp:
        t = d_rank % d_tp
        return [t], t, 1, 1, warn
    if not mla:
        warn = 'Performance is NOT guaranteed when using different TP sizes for non-MLA models. '
    if d_tp > p_tp:
        t = (d_rank % d_tp) // (d_tp // p_tp)
        return [t], t, d_tp // p_tp, 1, warn
    k = p_tp // d_tp
    r = d_rank % d_tp
    ranks = list(range(r * k, (r + 1) * k))
    return ranks, ranks[0], 1, (1 if mla else k), warn


def head_slice(src_tp, dst_tp, src_rank, dst_rank, total):
    """staging_buffer.py:773。返回 (src_head_start, n, dst_head_start, n)。"""
    src_hpr, dst_hpr = max(1, total // src_tp), max(1, total // dst_tp)
    local, dg = src_rank % src_tp, dst_rank % dst_tp
    if src_tp > dst_tp:
        uniq = local // max(1, src_tp // total)
        return 0, src_hpr, (uniq * src_hpr) % dst_hpr, src_hpr
    uniq = dg // max(1, dst_tp // total)
    return (uniq * dst_hpr) % src_hpr, dst_hpr, 0, dst_hpr


def staging_layout(src_tp, dst_tp, dst_rank, total, tokens, bph_token, layers):
    """staging_buffer.py:809。bph_token = 每头每 token 一层 K（或 V）的字节。"""
    n = src_tp // max(1, dst_tp) if src_tp > dst_tp else 1
    wb = [tokens * head_slice(src_tp, dst_tp, wr, dst_rank, total)[1] * bph_token * layers * 2
          for wr in range(n)]
    return n, wb, sum(wb)


def grid_tokens(chunk, page):
    return max(1, (chunk or 8192) // page) * page


def grid_segments(start, end, base, grid):
    segs, s = [], start
    while s < end:
        nb = base + ((s - base) // grid + 1) * grid
        segs.append((s, min(nb, end)))
        s = min(nb, end)
    return segs or [(start, end)]


def send_end(start, extend_end, input_len, page, last, staging=False, chunk=None, base=0, end=None):
    """prefill.py:1528 _send_kv_chunk 算出的发送终点；返回 None 表示这次跳过。"""
    end = end if end is not None else min(extend_end, input_len)
    if not last:
        end -= end % page
        if staging:
            g = grid_tokens(chunk, page)
            end = base + ((end - base) // g) * g
    return None if end < start else end


def early_send_end(prefix_len, host_hit, start_send, page, pending_bootstrap=False, allow=True):
    """prefill.py:1460 maybe_send_cached_prefix_chunk（非 staging）：返回提前发送的终点或 None。"""
    if pending_bootstrap or not allow:
        return None
    cached = prefix_len - host_hit
    if cached <= start_send or cached % page:
        return None
    return cached


def kvpoll_min(polls):
    """utils.py:300：Failed 0 ＜ Bootstrapping 1 ＜ WaitingForInput 2 ＜ Transferring 3 ＜ Success 4，取最小。"""
    return min(polls)


POLL = {0: 'Failed', 1: 'Bootstrapping', 2: 'WaitingForInput', 3: 'Transferring', 4: 'Success'}


# ---------------------------------------------------------------- 分配器碎片（token.py，page_size 1）

class TokenAllocator:
    def __init__(self, size, need_sort=True):
        self.size, self.need_sort = size, need_sort
        self.free_pages = list(range(1, size + 1))      # 槽位 0 留给 padding
        self.release = []

    def available(self):
        return len(self.free_pages) + len(self.release)

    def merge_and_sort(self):
        if self.release:
            self.free_pages = sorted(self.free_pages + self.release)
            self.release = []

    def alloc(self, n):
        if self.need_sort and n > len(self.free_pages):
            self.merge_and_sort()
        if n > len(self.free_pages):
            return None
        out, self.free_pages = self.free_pages[:n], self.free_pages[n:]
        return out

    def free(self, idx):
        if self.need_sort:
            self.release.extend(idx)
        else:
            self.free_pages.extend(idx)


def churn(alloc, steps, rng, lo=200, hi=6000, target=0.85):
    """把池子用到 target 左右来回分配、释放，模拟跑了一阵的线上实例。"""
    live = []
    for _ in range(steps):
        if alloc.available() > alloc.size * (1 - target):
            got = alloc.alloc(rng.randint(lo, hi))
            if got is not None:
                live.append(got)
                continue
        if live:
            alloc.free(live.pop(rng.randrange(len(live))))
    return live


def runs_after_churn(need_sort, size=200_000, n=4096, trials=20, seed=0):
    """P、D 两边各跑一阵后，新来一条 n token 的请求，两边索引配对能合成几段。"""
    rng = random.Random(seed)
    P, D = TokenAllocator(size, need_sort), TokenAllocator(size, need_sort)
    churn(P, 3000, rng)
    churn(D, 3000, rng)
    res = []
    for _ in range(trials):
        s, d = P.alloc(n), D.alloc(n)
        if s is None or d is None:
            churn(P, 50, rng)
            churn(D, 50, rng)
            continue
        res.append(len(group_contiguous(s, d)))
        P.free(s)
        D.free(d)
        churn(P, 60, rng)
        churn(D, 60, rng)
    return sim_stats.mean(res), max(res)


# ---------------------------------------------------------------- 字节和时间

def kv_bytes_per_rank(model, gpu, tp, tokens):
    return tokens * ledger(model, gpu, tp)['cell']


def descriptors(model, tp, runs=1, tokens=0, mode='equal'):
    """每个 rank 一条请求要提交的描述符数。equal：每段每层（MLA L，MHA 2L）；slice：每 token 每层。"""
    m = MODELS[model]
    per_layer = m.layers if m.mla_kv_rank else 2 * m.layers
    if mode == 'equal':
        return per_layer * runs
    if mode == 'slice':
        return per_layer * tokens
    raise ValueError(mode)


def chunk_pipeline(srv, tokens, cell, gbs, chunk=None):
    """P 侧按 chunk 算，每块算完把这块 KV 交给传输线程；传输串行占用链路。
    返回 (prefill 总耗时, 最后一块 KV 落地的时刻, 暴露在 TTFT 上的尾巴)，单位秒。"""
    chunk = chunk or srv.chunk
    t = link = 0.0
    done = 0
    while done < tokens:
        n = min(chunk, tokens - done)
        t += srv.t_prefill([(n, done)])
        link = max(link, t) + n * cell / (gbs * GB)
        done += n
    return t, link, link - t


def scenario_transfer(inp=32768):
    res = {}
    print(f'一条 {inp} token 的请求在每个 rank 上要搬的 KV（P、D 同 TP，单卡单网卡）')
    print(f'  {"模型 × 卡":<26}{"cell B":>8}{"每rank MB":>10}{"prefill ms":>11}'
          + ''.join(f'{k:>16}' for k in LINKS))
    for model, gpu, tp in CONFIGS:
        srv = Server(model, gpu, tp)
        cell = ledger(model, gpu, tp)['cell']
        mb = inp * cell / 1e6
        t_p, _, _ = chunk_pipeline(srv, inp, cell, 1e9)
        row = {}
        for k, g in LINKS.items():
            _, _, tail = chunk_pipeline(srv, inp, cell, g)
            row[k] = (inp * cell / (g * GB) * 1e3, tail * 1e3)
        res[model] = dict(cell=cell, mb=mb, prefill_ms=t_p * 1e3, links=row)
        print(f'  {name_of(model, gpu, tp):<26}{cell:>8}{mb:>10.0f}{t_p * 1e3:>11.0f}'
              + ''.join(f'{row[k][0]:>8.1f}/{row[k][1]:<7.1f}' for k in LINKS))
    print('  每格「整段搬运 ms / 按 chunk 流水后暴露在 TTFT 上的 ms」。chunk 取 02 账本给的默认值。')
    print('  MLA 在 attn TP8 下每个 rank 存一整份 latent，同一份 KV 要发 8 遍；DeepSeek 部署用 DP attention 避开。')
    print()

    print('长 prompt 的尾巴（Llama-3.1-70B TP4，RDMA 400G/卡）：')
    srv = Server('llama3.1-70b', 'h100-sxm', 4)
    cell = ledger('llama3.1-70b', 'h100-sxm', 4)['cell']
    tails = {}
    for n in (2048, 8192, 32768, 131072):
        t, _, tail = chunk_pipeline(srv, n, cell, LINKS['RDMA 400G/卡'])
        tails[n] = (t * 1e3, tail * 1e3)
        print(f'  {n:>7} token  prefill {t * 1e3:>7.0f} ms  KV {n * cell / 1e6:>6.0f} MB  暴露尾巴 {tail * 1e3:>5.1f} ms')
    print('  流水下只剩最后一块的搬运露在外面，长 prompt 的尾巴不随长度涨；链路慢到跟不上计算时才会堆积。')
    res['tails'] = tails
    print()

    print('描述符数（每个 rank、每条 4096 token 的请求）：')
    n = 4096
    fresh = descriptors('llama3.1-8b', 1, runs=1)
    sorted_runs = runs_after_churn(True)
    unsorted_runs = runs_after_churn(False)
    print(f'  Llama-3.1-8B 同 TP，两边池子都是新的：1 段 × 64 = {fresh}')
    print(f'  跑了一阵，need_sort=True（PD 的设置）：平均 {sorted_runs[0]:.0f} 段、最多 {sorted_runs[1]} 段'
          f' -> {descriptors("llama3.1-8b", 1, runs=round(sorted_runs[0]))} 个')
    print(f'  跑了一阵，need_sort=False：平均 {unsorted_runs[0]:.0f} 段、最多 {unsorted_runs[1]} 段'
          f' -> {descriptors("llama3.1-8b", 1, runs=round(unsorted_runs[0]))} 个')
    sl = descriptors('llama3.1-8b', 2, tokens=n, mode='slice')
    print(f'  P TP2 -> D TP1 走 send_kvcache_slice：每个 P rank 4096 × 64 = {sl:,} 个，每个只有 '
          f'{head_slice(2, 1, 0, 0, 8)[1] * 128 * 2} B')
    w, wb, tot = staging_layout(2, 1, 0, 8, n, 128 * 2, 32)
    print(f'  打开 staging：{w} 个 writer，各写 {wb[0] / 1e6:.0f} MB，每个 writer 每个 chunk 一次大块传输')
    res['desc'] = dict(fresh=fresh, sorted=sorted_runs, unsorted=unsorted_runs, slice=sl, staging=(w, wb, tot))
    print()
    print('  单个描述符的固定开销（提交 + 完成）不同，slice 路径多出的时间（8B，4096 token，单 writer）：')
    data_ms = n * 32 * 2 * head_slice(2, 1, 0, 0, 8)[1] * 128 * 2 / (LINKS['RDMA 400G/卡'] * GB) * 1e3
    over = {}
    for us in (0.05, 0.2, 1.0):
        over[us] = sl * us / 1e3
        print(f'    {us:>4} µs/个  描述符 {over[us]:>6.1f} ms  vs 字节本身 {data_ms:.1f} ms')
    res['slice_over'] = (data_ms, over)
    return res


# ---------------------------------------------------------------- P:D 配比

WORKLOADS = [
    # (名字, 输入, 输出, TTFT SLO s, TPOT SLO s)
    ('聊天', 1024, 256, 1.0, 0.05),
    ('RAG 摘要', 8192, 256, 2.0, 0.05),
    ('代码补全长输出', 512, 1024, 0.5, 0.03),
    ('长上下文 agent', 32768, 512, 4.0, 0.05),
]


def prefill_capacity(srv, inp, cell, gbs, ttft_slo):
    """一个 P 实例在平均 TTFT ≤ SLO 下能接的最大请求率。
    吞吐按一个 chunk 里拼满几条算，单条耗时按自己单独算，排队按 M/D/1。"""
    k = max(1, srv.chunk // inp)
    if k > 1:
        t_step = srv.t_prefill([(inp, 0)] * k)
        svc, tail = t_step / k, inp * cell / (gbs * GB)
        t_one = srv.t_prefill([(inp, 0)])            # 低负载时自己一条单独算
    else:
        t_one, _, tail = chunk_pipeline(srv, inp, cell, gbs)
        svc = t_one
    mu = 1 / svc
    if t_one + tail > ttft_slo:
        return 0.0, t_one, tail
    lo, hi = 0.0, mu * 0.999
    for _ in range(60):
        lam = (lo + hi) / 2
        rho = lam / mu
        w = rho / (2 * mu * (1 - rho))
        if t_one + w + tail <= ttft_slo:
            lo = lam
        else:
            hi = lam
    return lo, t_one, tail


def decode_capacity(srv, inp, out, tpot_slo, cap):
    """一个 D 实例在 TPOT ≤ SLO、显存装得下时的最大并发和请求率。"""
    ctx = inp + out // 2
    best = 0
    for b in range(1, 4097):
        need = b * (inp + min(out, CLIP)) + RESERVED * b
        if need > cap or srv.t_decode([ctx] * b) > tpot_slo:
            break
        best = b
    if best == 0:
        return 0, 0.0, None
    t = srv.t_decode([ctx] * best)
    return best, best / (out * t), t


def colo_capacity(srv, inp, out, ttft_slo, tpot_slo, cap):
    """合并部署：一部分时间做 prefill，剩下做 decode；平均 TPOT = decode 一步 ÷ (1 - prefill 占比)。"""
    t_p, _, _ = chunk_pipeline(srv, inp, 0, 1e9)
    if t_p > ttft_slo:
        return 0.0
    best, lam = 0.0, 0.01
    while lam < 1000:
        f = lam * t_p
        if f >= 0.95:
            break
        # Little：并发 = 到达率 × 输出长度 × TPOT，迭代求不动点
        tpot = srv.t_decode([inp + out // 2])
        for _ in range(30):
            b = max(1, lam * out * tpot)
            if b * (inp + out) > cap:
                tpot = math.inf
                break
            tpot = srv.t_decode([inp + out // 2] * int(math.ceil(b))) / (1 - f)
        w = f / (2 * (1 / t_p) * (1 - f)) if f < 1 else math.inf
        if tpot <= tpot_slo and t_p + w <= ttft_slo:
            best = lam
        lam *= 1.05
    return best


def scenario_ratio(model='llama3.1-70b', gpu='h100-sxm', tp=4, link='RDMA 400G/卡'):
    srv = Server(model, gpu, tp)
    cell = ledger(model, gpu, tp)['cell']
    cap = srv.cap
    gbs = LINKS[link]
    print(f'{name_of(model, gpu, tp)}，KV 池 {cap:,} token，链路 {link}。P、D 各是一个 TP{tp} 实例')
    print(f'  {"负载":<14}{"输入/输出":>12}{"SLO":>14}{"P 上限 rps":>11}{"D 并发":>8}{"D 上限 rps":>11}'
          f'{"P:D":>7}{"PD rps/卡":>10}{"合并 rps/卡":>11}{"PD/合并":>8}')
    res = {}
    for name, inp, out, ttft, tpot in WORKLOADS:
        lp, t_one, tail = prefill_capacity(srv, inp, cell, gbs, ttft)
        b, ld, step = decode_capacity(srv, inp, out, tpot, cap)
        lc = colo_capacity(srv, inp, out, ttft, tpot, cap)
        if lp <= 0 or ld <= 0:
            res[name] = None
            print(f'  {name:<14}{inp:>6}/{out:<5}  P 或 D 单独都达不到 SLO')
            continue
        ratio = ld / lp
        pd = ld / ((ratio + 1) * tp)
        colo = lc / tp
        res[name] = dict(lp=lp, b=b, ld=ld, ratio=ratio, pd=pd, colo=colo, t_one=t_one, tail=tail, step=step)
        print(f'  {name:<14}{inp:>6}/{out:<5}{f"{ttft:g}s/{tpot * 1e3:.0f}ms":>14}{lp:>11.1f}{b:>8}{ld:>11.1f}'
              f'{ratio:>7.2f}{pd:>10.2f}{colo:>11.2f}{pd / colo if colo else math.inf:>8.2f}')
    print('  P:D = 一个 D 吃得下的请求率 ÷ 一个 P 撑得住的请求率，即每个 D 配几个 P。')
    print('  合并部署按平均 TPOT 算，没算 chunk 打断造成的 ITL 尾部；真看 P99 时合并部署的数字还要往下打。')
    return res


# ---------------------------------------------------------------- 合并 vs 分离：离散事件模拟

class PDReq:
    __slots__ = ('i', 'kind', 'inp', 'out', 'arr', 'done', 'kv_at', 'first', 'itl', 'last', 'end', 'gen')

    def __init__(self, i, kind, inp, out, arr):
        self.i, self.kind, self.inp, self.out, self.arr = i, kind, inp, out, arr
        self.done = 0
        self.kv_at = self.first = self.last = self.end = None
        self.itl = []
        self.gen = 0


def run_pd(reqs, model, gpu, tp, gbs, chunk=None):
    """1 个 P + 1 个 D。P 按 chunk 拼批 FIFO 地算，每块算完交给链路；D 连续批处理，
    KV 到齐且显存装得下才入批（FIFO，放不下就等，和 pop_preallocated 一样会队头阻塞）。"""
    sp, sd = Server(model, gpu, tp), Server(model, gpu, tp)
    cell = ledger(model, gpu, tp)['cell']
    chunk = chunk or sp.chunk
    pend = sorted(reqs, key=lambda r: r.arr)
    # ---- P 侧
    t = link = busy_p = 0.0
    q, k = [], 0
    while k < len(pend) or q:
        if not q:
            t = max(t, pend[k].arr)
        while k < len(pend) and pend[k].arr <= t:
            q.append(pend[k])
            k += 1
        parts, left = [], chunk
        for r in q:
            if left <= 0:
                break
            n = min(left, r.inp - r.done)
            parts.append((r, n))
            left -= n
        dt = sp.t_prefill([(n, r.done) for r, n in parts])
        t += dt
        busy_p += dt
        for r, n in parts:
            link = max(link, t) + n * cell / (gbs * GB)
            r.done += n
            if r.done == r.inp:
                r.kv_at = link
        q = [r for r in q if r.done < r.inp]
    # ---- D 侧
    arrived = sorted(reqs, key=lambda r: r.kv_at)
    t, j, run, used, busy_d = 0.0, 0, [], 0, 0.0
    cap = sd.cap
    waiting = []
    while j < len(arrived) or waiting or run:
        while j < len(arrived) and arrived[j].kv_at <= t:
            waiting.append(arrived[j])
            j += 1
        while waiting:
            r = waiting[0]
            need = r.inp + min(r.out, CLIP)
            if used + need + RESERVED * (len(run) + 1) > cap:
                break
            used += need
            waiting.pop(0)
            r.first = t
            r.last = t
            r.gen = 1                       # 第一个 token 由 P 算出，随 KV 元数据一起到
            run.append(r)
        if not run:
            if waiting:                     # 空实例都放不下队头，永远进不来
                w = waiting[0]
                raise ValueError(f'请求 {w.i} 要 {w.inp + min(w.out, CLIP) + RESERVED:,} token，'
                                 f'D 的 KV 池只有 {cap:,}')
            t = arrived[j].kv_at
            continue
        dt = sd.t_decode([r.inp + r.gen for r in run])
        t += dt
        busy_d += dt
        keep = []
        for r in run:
            r.itl.append(t - r.last)
            r.last = t
            r.gen += 1
            if r.gen >= r.out:
                r.end = t
                used -= r.inp + min(r.out, CLIP)
            else:
                keep.append(r)
        run = keep
    dur = max(r.end for r in reqs) - min(r.arr for r in reqs)
    return dict(busy_p=busy_p / dur, busy_d=busy_d / dur)


def run_colo(reqs, model, gpu, n_inst=2, chunk=None, mixed=False):
    """n_inst 个合并部署实例，轮询分发，复用 06 的调度器模拟。"""
    import sim_chunked as SC
    out = []
    busy = 0.0
    for k in range(n_inst):
        mine = [SC.Req(len(out) + i, r.kind, r.inp, r.out, r.arr) for i, r in enumerate(reqs[k::n_inst])]
        sim, _ = SC.build(model, gpu, chunk, mixed)
        done = sim.run(mine)
        busy += sim.st['prefill_time']
        out.extend(done)
    dur = max(r.end for r in out) - min(r.arr for r in out)
    return out, busy / (n_inst * dur)          # 每张卡花在 prefill 上的时间占比


def summarize(reqs, ttft_of):
    sh = [r for r in reqs if r.kind == 'short']
    itl = [x * 1e3 for r in reqs for x in r.itl]
    dur = max(r.end for r in reqs) - min(r.arr for r in reqs)
    return dict(
        ttft_p50=sim_stats.pct([ttft_of(r) * 1e3 for r in sh], 50),
        ttft_p99=sim_stats.pct([ttft_of(r) * 1e3 for r in sh], 99),
        long_ttft=sim_stats.mean([ttft_of(r) * 1e3 for r in reqs if r.kind == 'long']) if any(
            r.kind == 'long' for r in reqs) else float('nan'),
        itl_p50=sim_stats.pct(itl, 50), itl_p99=sim_stats.pct(itl, 99), itl_max=max(itl),
        tps=sum(r.gen for r in reqs) / dur)


def workload(kind, seed=42):
    import sim_chunked as SC
    if kind == 'long':
        base = SC.wl_poisson(rate=6.0, dur=60, short_in=1024, short_out=256, long_in=32768, long_out=128,
                             long_every=6.0, seed=seed)
    elif kind == 'short':
        base = SC.wl_poisson(rate=40.0, dur=30, short_in=256, short_out=512, long_every=1e9, seed=seed)
    else:
        raise ValueError(kind)
    return [(r.kind, r.inp, r.max_new, r.arr) for r in base if kind == 'long' or r.kind == 'short']


def compare(kind, model='llama3.1-8b', gpu='h100-sxm', link='RDMA 400G/卡', seed=42):
    spec = workload(kind, seed)
    pd_reqs = [PDReq(i, *s) for i, s in enumerate(spec)]
    util = run_pd(pd_reqs, model, gpu, 1, LINKS[link])
    m_pd = summarize(pd_reqs, lambda r: r.first - r.arr)
    m_pd.update(util)
    colo_spec = [PDReq(i, *s) for i, s in enumerate(spec)]
    done, busy = run_colo(colo_spec, model, gpu, 2, chunk=None)
    m_co = summarize(done, lambda r: r.ttft)
    m_co['busy'] = busy
    return m_co, m_pd


def scenario_colo(model='llama3.1-8b', gpu='h100-sxm'):
    hdr = (f'  {"部署":<18}{"短TTFT p50":>11}{"短TTFT p99":>11}{"长TTFT":>9}{"ITL p50":>9}{"ITL p99":>9}'
           f'{"ITL max":>9}{"out tok/s":>10}{"prefill 占时":>14}')

    def line(tag, m, busy):
        print(f'  {tag:<18}{m["ttft_p50"]:>11.0f}{m["ttft_p99"]:>11.0f}{m["long_ttft"]:>9.0f}{m["itl_p50"]:>9.1f}'
              f'{m["itl_p99"]:>9.1f}{m["itl_max"]:>9.1f}{m["tps"]:>10.0f}{busy:>14}')

    res = {}
    cases = [('long', 'RDMA 400G/卡', '短请求 6 req/s（1024/256）+ 每 6 s 一条 32768 token 长 prompt，60 s'),
             ('short', 'RDMA 400G/卡', '短请求 40 req/s（256/512），30 s'),
             ('long', 'TCP 100GbE÷8', '同第一种负载，链路换成 TCP')]
    for kind, link, title in cases:
        co, pd = compare(kind, model, gpu, link)
        res[(kind, link)] = (co, pd)
        print(f'{MODELS[model].name}，两张 H100：{title}')
        print(hdr)
        line('合并部署 ×2', co, f'每卡 {co["busy"]:.0%}')
        line(f'1P1D {link}', pd, f'P 卡 {pd["busy_p"]:.0%}')
        print()
    print('合并部署用 06 的调度器（默认 chunk、不开 mixed），时间单位 ms。短请求 TTFT 只统计短请求，')
    print('PD 的 TTFT 算到 D 侧拿到首 token 为止（含 KV 搬运和 D 侧排队）。')
    return res


# ---------------------------------------------------------------- decode 准入（decode.py 的预算）

class DecodeAdmission:
    """decode.py:2086 _allocatable_token_budgets 和 :1336 pop_preallocated 的 page_size=1、无 radix、无 SWA 版本。"""

    def __init__(self, pool, reserved=RESERVED, req_slots=10 ** 9):
        self.free = pool
        self.reserved = reserved
        self.running = []           # (origin, output_len, max_new)
        self.transfer = 0           # 已预分配、在等 KV 的请求数（都在显存里收）
        self.waiting = 0            # KV 已到、还没进 running 的请求数
        self.retracted = []         # 每条被 retract 的请求恢复时要的 token
        self.prebuilt = 0           # 上一个 batch 是 prebuilt extend 时的请求数
        self.req_slots = req_slots

    def need_single(self, retractable):
        if not self.running:
            return 0
        return max(min(mx, CLIP) + o - retractable for o, _, mx in self.running)

    def active(self, extra=0):
        return len(self.running) + self.transfer + self.waiting + extra

    def budget(self, retractable, count_retracted=True, extra=0, hicache=0):
        reserved = self.reserved * self.active(extra)
        b = self.free - max(reserved, self.need_single(retractable))
        b -= self.reserved * self.prebuilt
        if count_retracted:
            b -= sum(self.retracted)
        return b - hicache

    def required(self, alloc, inp, prefix, max_new, retractable):
        """decode.py:511。"""
        return max(alloc + self.reserved, inp - prefix + min(max_new, CLIP) - retractable)

    def pop_preallocated(self, queue):
        """queue：[(输入长度, max_new)]，按 FIFO。返回 (放进来的条数, 卡住的原因, 每步日志)。"""
        retractable = sum(o + out for o, out, _ in self.running)
        b = self.budget(retractable)
        n_dev, log = 0, []
        for inp, mx in queue:
            if self.req_slots - n_dev <= 0:
                return n_dev, 'req_to_token_pool 没有空槽', log
            alloc = inp                                 # page 1、无前缀：fill_len - prefix_len
            need = self.required(alloc, inp, 0, mx, retractable)
            if need > b:
                log.append((inp, need, b, False))
                return n_dev, f'队头 {inp} token 要 {need:,}，预算只有 {b:,}', log
            log.append((inp, need, b, True))
            self.free -= alloc
            n_dev += 1
            b = self.budget(retractable, extra=n_dev)
        return n_dev, None, log


def max_admit(pool, inp, max_new=512, running=(), transfer=0, waiting=0, reserved=RESERVED):
    a = DecodeAdmission(pool, reserved)
    a.running = list(running)
    a.free -= sum(o + out for o, out, _ in running)
    a.transfer, a.waiting = transfer, waiting
    n, why, _ = a.pop_preallocated([(inp, max_new)] * 100000)
    return n, why


def scenario_admission(model='llama3.1-8b', gpu='h100-sxm'):
    pool = Server(model, gpu, 1).cap
    res = {}
    print(f'{MODELS[model].name} 的 decode 实例，KV 池 {pool:,} token，num_reserved_decode_tokens={RESERVED}')
    print('1）空实例一次 pop_preallocated 能放进几条（max_new 128）：')
    for inp in (1024, 4096, 16384, 65536):
        n, _ = max_admit(pool, inp, 128)
        naive = pool // (inp + 128)
        res[('empty', inp)] = (n, naive)
        print(f'  输入 {inp:>6}：放进 {n:>4} 条，按 池子 ÷ (输入 + 输出) 估是 {naive:>4} 条')
    print('  输出短于 512 时，每条按 512 预留算，短输出的请求在 D 上比想象的占地方。')
    print()
    print('2）已有 100 条在 decode（每条输入 1500、已出 500、max_new 2048），再看排在 waiting_queue 里的请求：')
    running = [(1500, 500, 2048)] * 100
    for waiting in (0, 100, 300):
        n, why = max_admit(pool, 4096, 512, running, transfer=0, waiting=waiting)
        res[('busy', waiting)] = n
        print(f'  waiting_queue {waiting:>3} 条：能再预分配 {n:>3} 条 4096 token 的请求'
              f'（起步预留 512 × {100 + waiting} = {512 * (100 + waiting):,} token）')
    print('  这里只算预留这一项，排队请求自己已经占着的 KV 没扣，真实情况更紧。')
    print('  waiting_queue 和 transfer_queue 里的请求都按活跃请求计预留，排队越长，新请求越难进来。')
    print()
    print('3）队头阻塞：池子只剩 90,000 token，队列是 [100000, 2000 × 10]：')
    a = DecodeAdmission(90_000)
    n, why, log = a.pop_preallocated([(100_000, 256)] + [(2000, 256)] * 10)
    res['hol'] = (n, why)
    print(f'  放进 {n} 条，{why}；后面 10 条 2000 token 的请求本来放得下，也在等。')
    a = DecodeAdmission(90_000)
    n2, _, _ = a.pop_preallocated([(2000, 256)] * 10 + [(100_000, 256)])
    res['hol_swapped'] = n2
    print(f'  同样的请求换个顺序（大请求排最后）：放进 {n2} 条。')
    print()
    print('4）retract 回来的请求优先：retracted_queue 里压着 3 条各要 30,000 token 的请求：')
    a = DecodeAdmission(100_000)
    a.retracted = [30_000] * 3
    n3, why3, _ = a.pop_preallocated([(4096, 512)] * 20)
    res['retracted'] = n3
    print(f'  放进 {n3} 条新请求：{why3}')
    return res


# ---------------------------------------------------------------- 一条请求的时间线

def scenario_timeline(model='llama3.1-70b', gpu='h100-sxm', tp=4):
    srv = Server(model, gpu, tp)
    cell = ledger(model, gpu, tp)['cell']
    gbs = LINKS['RDMA 400G/卡']
    page, chunk, inp, prefix = 64, 8192, 20000, 2000
    res = {}
    print(f'{name_of(model, gpu, tp)}，输入 {inp} token，P 侧 radix 命中 {prefix}，page_size {page}，chunk {chunk}')
    print('1）P 侧每次发送的区间（start_send_idx 从 decode 承诺的前缀 0 开始）：')
    e = early_send_end(prefix, 0, 0, page)
    print(f'  提前发缓存前缀：{prefix} % {page} = {prefix % page}，不对齐，{"跳过" if e is None else e}')
    e2 = early_send_end(1984, 0, 0, page)
    print(f'  如果命中 1984：直接发 [0, {e2})，不用等 prefill')
    sends, start, done = [], 0, prefix
    while done < inp:
        ext = min(done + chunk, inp)
        last = ext == inp
        end = send_end(start, ext, inp, page, last)
        sends.append((start, end, ext, last))
        start = end if end is not None else start
        done = ext
    for s, en, ext, last in sends:
        tag = '最后一块' if last else '中间块'
        print(f'  {tag}  extend_end {ext:>6}  发 [{s:>6}, {en:>6})  扣下 {ext - en:>2} 个不满一页的 token')
    res['sends'] = sends
    st = [send_end(0, prefix + chunk, inp, page, False, staging=True, chunk=chunk)]
    print(f'  打开 staging：第一块终点再对齐到 {grid_tokens(chunk, page)} 的格子 -> {st[0]}，'
          f'格子 {grid_segments(0, inp, 0, grid_tokens(chunk, page))}')
    res['staging_end'] = st[0]
    print()
    print('2）KVPoll：每个 rank 自己 poll，再在 TP 组里取 MIN')
    for polls in ([4, 4, 4, 4], [4, 4, 3, 4], [4, 0, 4, 4]):
        v = kvpoll_min(polls)
        print(f'  {polls} -> {v} {POLL[v]}')
    print('  一个 rank 没收完，整组都等；一个 rank 失败，整组按失败处理。')
    print()
    print('3）early 和 prefill_complete：decode 槽位被这条请求占住多久（P 排队时间不同）')
    t_p, kv_at, tail = chunk_pipeline(srv, inp, cell, gbs, chunk)
    x = inp * cell / (gbs * GB)
    rows = {}
    for wait in (0.0, 2.0, 10.0):
        early = wait + kv_at
        late = x
        rows[wait] = (early, late)
        print(f'  P 排队 {wait:>4.0f} s：early 占槽 {early:>6.2f} s（{inp * early / 1e3:>7.0f} k token·s）'
              f'  prefill_complete 占槽 {late:>5.2f} s')
    print(f'  prefill {t_p:.2f} s，整段搬运 {x * 1e3:.0f} ms；prefill_complete 失去了 chunk 和传输的重叠，'
          f'TTFT 多出约 {(x - tail) * 1e3:.0f} ms')
    res['policy'] = rows
    res['t_p'], res['x'], res['tail'] = t_p, x, tail
    print()
    print('4）超时：P 侧 bootstrap 300 s、D 侧 waiting 300 s、心跳 5 s × 2 次。P 长时间排队时 D 侧')
    print('   预分配的槽位最多被占 300 s，之后请求被判失败，客户端看到的是报错而不是慢。')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='transfer',
                    choices=['transfer', 'ratio', 'colo', 'admission', 'timeline'])
    ap.add_argument('--model', default=None)
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--tp', type=int, default=None)
    ap.add_argument('--input', type=int, default=32768)
    a = ap.parse_args()
    if a.scenario == 'transfer':
        scenario_transfer(a.input)
    elif a.scenario == 'ratio':
        scenario_ratio(a.model or 'llama3.1-70b', a.gpu or 'h100-sxm', a.tp or 4)
    elif a.scenario == 'colo':
        scenario_colo(a.model or 'llama3.1-8b', a.gpu or 'h100-sxm')
    elif a.scenario == 'admission':
        scenario_admission(a.model or 'llama3.1-8b', a.gpu or 'h100-sxm')
    else:
        scenario_timeline(a.model or 'llama3.1-70b', a.gpu or 'h100-sxm', a.tp or 4)


if __name__ == '__main__':
    main()
