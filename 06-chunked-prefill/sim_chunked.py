"""Chunked prefill 的 CPU 模拟：一条长 prompt 插进正在 decode 的批里，切块大小、mixed chunk、
prefill-decode interval 分别怎么改变 TTFT、ITL 和显存账本。

目的：不用 GPU 也能看清三件事，并且每一步都能和源码对上。
  1. 默认调度下，长 prompt 的各个 chunk 是背靠背跑的：调度器每轮先续跑 chunked_req，
     decode 要等最后一块跑完。切块大小几乎不改变 decode 被卡住的总时长；
  2. 打开 --enable-mixed-chunk 后，在跑请求的 decode 行搭在每个 chunk 上一起前向，
     decode 的最长停顿约等于一个 chunk 的耗时，这时切块大小才直接决定 ITL 尾部；
  3. chunked_prefill_size 同时决定启动时给激活预留多少显存，切得越大，KV 池越小。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - 主循环：event_loop_normal，没有 overlap。每轮先试 prefill，凑不出来才 decode（get_next_batch_to_run）；
            chunked_req 不进 running，下一轮先续跑；它绕过「队列为空 / 批已满」的提前返回
  - 准入：PrefillAdder + PrefillBudget。rem_input = max_prefill_tokens - num_mixed，
          rem_chunk = chunked_prefill_size - num_mixed，total_offset 先扣 num_mixed 和在跑请求的
          min(剩余输出, 4096)（ignore_eos 时 ratio 取 1.0）；续跑块取 available_chunk_tokens，截断时不预留输出；
          新请求 输入 + 输出预留 + 1 页 ≥ remaining_total 返回 NO_TOKEN；超过 rem_chunk 的被截断成新的 chunked_req；
          关掉切块时第一条请求可以超过 max_prefill_tokens，后面的输入 ≥ rem_input 就返回 OTHER
  - mixed：is_mixed_chunk = 切块开启且 --enable-mixed-chunk；在跑请求不空、decode 显存够时每条出一个 token，
           forward_mode 记为 MIXED；显存不够时这一步不带 decode 行，也不 retract
  - interval：每跑一个 extend（含 MIXED）批，接下来 N 轮不收 prefill；在跑请求少于 min_bs 时立即清零
  - 显存：memory_hook.py 的 reserved_mem = 512 + 1.5 x max(chunk, 2048) + tp/8 x 1024 + Graph 预留，
          80 GB 档以上有 10 GB 下限；关掉切块时激活按 max(max_prefill_tokens, 2048) 算，prefill Graph 档位为空
  - 日志：Prefill batch 那一行的字段和顺序与 metrics_reporter.py 一致；中间块不出 token，最后一块才算 TTFT
不对齐的部分：
  - 时间：单步耗时取 ../03-bench/sim_serve.py 的 Roofline 下限 / 利用率模型，只用来排序和对比
  - 没有 overlap、前缀缓存、retract、page_size > 1、SWA / Mamba、PP 的 dynamic chunking、多 DP
  - 排序固定 fcfs，不模拟 shortest-prefill-first 的 chunked_req_limit

用法：
    python3 sim_chunked.py                                  # 一条 32k prompt 插进 64 路 decode，默认 / mixed x 各种 chunk
    python3 sim_chunked.py --scenario interval              # prefill-decode interval 的取舍
    python3 sim_chunked.py --scenario poisson               # 泊松短请求 + 周期性长 prompt，看分位数
    python3 sim_chunked.py --scenario memory                # 各档 GPU 上 chunk 大小换来的 KV 池
    python3 sim_chunked.py --scenario trace --chunked-prefill-size 8192 --enable-mixed-chunk --log-head 14
"""
import argparse
import math
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
from common.specs import MODELS, SMI_MIB  # noqa: E402
from estimate_startup import prefill_bs, tier  # noqa: E402
from sim_ledger import ledger  # noqa: E402
from sim_serve import Server  # noqa: E402
import sim_stats  # noqa: E402

CLIP_MAX_NEW_TOKENS = 4096
MAX_PREFILL_TOKENS = 16384
DECODE_LOG_INTERVAL = 40
PAGE = 1
CONTINUE, NO_TOKEN, OTHER = 'CONTINUE', 'NO_TOKEN', 'OTHER'


# ---------------------------------------------------------------- 显存账本
def reserved_mb(gpu_mib, chunk, tp=1, max_prefill=MAX_PREFILL_TOKENS):
    """memory_hook.py handle_gpu_memory_settings + reserve_for_graph_mb（非 MLA、非 DP attention）。"""
    _, max_bs = tier(gpu_mib, tp)
    if chunk > 0:
        act = max(chunk, 2048)
        n_prefill_graph = len(prefill_bs(chunk))
    else:
        act = max(max_prefill, 2048)
        n_prefill_graph = 0                     # prefill Graph 的 max_bs 跟着 chunk 变成 -1，档位列表为空
    graph = max_bs * 2 + n_prefill_graph * 8
    r = 512 + act * 1.5 + tp / 8 * 1024 + graph
    floor = gpu_mib > 60 * 1024 and r < 10 * 1024
    if gpu_mib > 60 * 1024:
        r = max(r, 10 * 1024)
    return r, act, graph, floor


def kv_tokens(model, gpu, chunk, tp=1, kv_dtype='bf16'):
    gpu_mib = SMI_MIB[gpu]
    r, act, graph, floor = reserved_mb(gpu_mib, chunk, tp)
    frac = round((gpu_mib - r) / gpu_mib, 3)
    L = ledger(model, gpu, tp, kv_dtype=kv_dtype, mem_frac=frac)
    return dict(reserved=r, act=act, graph=graph, floor=floor, frac=frac, tokens=L['tokens'])


# ---------------------------------------------------------------- 请求与调度器
class Req:
    __slots__ = ('i', 'kind', 'inp', 'max_new', 'arr', 'prefix_len', 'extend_end', 'kv', 'gen',
                 'ttft', 'itl', 'last', 'end', 'chunks')

    def __init__(self, i, kind, inp, max_new, arr):
        self.i, self.kind, self.inp, self.max_new, self.arr = i, kind, inp, max_new, arr
        self.prefix_len = 0       # 已经算完、写进 KV 池的输入 token
        self.extend_end = 0
        self.kv = 0               # 占用的 KV token 数
        self.gen = 0
        self.ttft = None
        self.itl = []
        self.last = 0.0
        self.end = None
        self.chunks = 0

    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len


class Adder:
    """PrefillAdder + PrefillBudget，page_size=1、无前缀缓存时的子集。"""

    def __init__(self, free, running, chunk, num_mixed):
        self.free = free
        self.rem_input = MAX_PREFILL_TOKENS - num_mixed
        self.rem_chunk = None if chunk is None else chunk - num_mixed
        self.total_offset = num_mixed
        self.current_offset = num_mixed
        for r in running:                                   # _get_running_request_total_token_offset
            self.total_offset += min(r.max_new - r.gen, CLIP_MAX_NEW_TOKENS)   # ignore_eos：ratio 1.0
        self.can_run = []
        self.new_chunked = None
        self.log_input = 0

    @property
    def remaining_total(self):
        return self.free - self.total_offset

    @property
    def remaining_current(self):
        return self.free - self.current_offset

    def budget_state(self):
        if not (self.remaining_total > 0 and self.remaining_current > 0):
            return NO_TOKEN
        if self.rem_input <= 0 or (self.rem_chunk is not None and self.rem_chunk <= 0):
            return OTHER
        return CONTINUE

    def _charge(self, ext, max_new):
        """_update_prefill_budget → PrefillBudget.reserve。"""
        self.total_offset += ext + PAGE + max_new
        self.current_offset += ext + PAGE
        self.rem_input -= ext
        if self.rem_chunk is not None:
            self.rem_chunk -= ext
        self.log_input += ext

    def add_chunked_req(self, r):
        avail = min(self.rem_chunk, self.remaining_total)
        rem = avail if avail > 0 else self.rem_chunk       # 续跑必须有进展，才能释放它的 KV
        cand = r.inp - r.prefix_len
        truncated = cand > rem
        new_len = min(cand, rem)
        assert new_len > 0, 'mixed 的 decode 行吃光了 chunk 预算，调大 chunk 或减少并发'
        r.extend_end = r.prefix_len + new_len
        self.can_run.append(r)
        self._charge(new_len, 0 if truncated else min(r.max_new, CLIP_MAX_NEW_TOKENS))
        return r if truncated else None

    def add_one_req(self, r):
        ext = r.inp - r.prefix_len
        max_new = min(r.max_new - r.gen, CLIP_MAX_NEW_TOKENS)
        total = ext + max_new + PAGE
        if not total < self.remaining_total:                # check_prefill
            return NO_TOKEN
        if self.rem_chunk is None and self.can_run and ext >= self.rem_input:
            return OTHER                                    # 不切块时只有第一条能超过 max_prefill_tokens
        if self.rem_chunk is not None and ext > self.rem_chunk:
            ext = self.rem_chunk
            max_new = 0
            self.new_chunked = r
        r.extend_end = r.prefix_len + ext
        self.can_run.append(r)
        self._charge(ext, max_new)
        return self.budget_state()


class Sim:
    def __init__(self, srv, cap, chunk, mixed=False, interval=0, interval_min_bs=0,
                 max_running=4096, verbose=False):
        self.srv, self.cap = srv, cap
        self.chunk = chunk if chunk and chunk > 0 else None
        self.mixed = mixed and self.chunk is not None        # scheduler.py is_mixed_chunk
        self.interval, self.min_bs = interval, interval_min_bs
        self.remaining = 0
        self.max_running = max_running
        self.verbose = verbose
        self.waiting, self.running = [], []
        self.chunked = None
        self.batch_is_full = False
        self.free = cap
        self.now = 0.0
        self.last_prefill_t = 0.0
        self.last_decode_t, self.gen_since, self.decode_ct = 0.0, 0, 0
        self.lines = []
        self.st = dict(prefill_steps=0, mixed_steps=0, decode_steps=0, deferred=0, sat_out=0,
                       prefill_time=0.0, decode_time=0.0)

    def log(self, msg):
        line = f'[t={self.now:8.3f}s] {msg}'
        self.lines.append(line)
        if self.verbose:
            print(line)

    def usage(self):
        return (self.cap - self.free) / self.cap

    # ---- get_next_batch_to_run ----
    def step(self):
        before = len(self.running)
        self.running = [r for r in self.running if r.end is None]
        if len(self.running) < before or not self.running:
            self.batch_is_full = False
        if self.remaining and len(self.running) < self.min_bs:     # _should_defer_prefill
            self.remaining = 0
        if self.remaining:
            self.remaining -= 1
            self.st['deferred'] += 1
            did = None
        else:
            did = self.prefill()
        if did is None:
            if not self.running:
                return False
            self.decode()
        elif self.interval:                                       # _arm_prefill_decode_interval
            self.remaining = self.interval
        return True

    # ---- _get_new_batch_prefill_raw ----
    def prefill(self):
        if (self.batch_is_full or not self.waiting) and self.chunked is None:
            return None
        if len(self.running) >= self.max_running and self.chunked is None:
            self.batch_is_full = True
            return None
        num_mixed = len(self.running) if self.mixed else 0
        ad = Adder(self.free, self.running, self.chunk, num_mixed)
        cont = self.chunked
        if cont is not None:
            self.chunked = ad.add_chunked_req(cont)
        for r in self.waiting:
            if ad.budget_state() == OTHER:
                break
            held = 1 if cont is not None else 0
            if len(ad.can_run) >= self.max_running - len(self.running) + held:
                self.batch_is_full = True
                break
            res = ad.add_one_req(r)
            if res != CONTINUE:
                if res == NO_TOKEN:
                    self.batch_is_full = bool(ad.can_run) or bool(self.running)
                break
        if not ad.can_run:
            return None
        picked = set(id(r) for r in ad.can_run)
        self.waiting = [r for r in self.waiting if id(r) not in picked]
        if ad.new_chunked is not None:
            assert self.chunked is None
            self.chunked = ad.new_chunked

        pending = sum(r.inp for r in self.waiting)
        if self.chunked is not None:
            pending += self.chunked.inp - self.chunked.extend_end
        parts = []
        for r in ad.can_run:
            ext = r.extend_len
            self.free -= ext
            r.kv += ext
            r.chunks += 1
            parts.append((ext, r.prefix_len))

        # Mixed-style chunked prefill：decode 行搭车，显存不够就这一步不带
        rows = []
        if self.mixed and self.running:
            if self.free >= len(self.running) * PAGE:            # check_decode_mem
                rows = self.running
            else:
                self.st['sat_out'] += 1
        dt = self.srv.t_prefill(parts, [r.inp + r.gen for r in rows])
        n_tok = ad.log_input + len(rows)
        graph = self.chunk is not None and n_tok <= self.chunk  # prefill Graph 最大档 = chunk
        gap = self.now + dt - self.last_prefill_t
        self.last_prefill_t = self.now + dt
        self.log(f'Prefill batch, #new-seq: {len(ad.can_run)}, #new-token: {ad.log_input}, '
                 f'#cached-token: 0, token usage: {self.usage():.2f}, '
                 f'#running-req: {len(self.running)}, #queue-req: {len(self.waiting)}, '
                 f'#pending-token: {pending}, cuda graph: {graph}, '
                 f'input throughput (token/s): {ad.log_input / gap if gap > 0 else 0:.2f}'
                 + (f'  [MIXED +{len(rows)} decode]' if rows else ''))
        self.now += dt
        self.st['prefill_steps'] += 1
        self.st['prefill_time'] += dt
        if rows:
            self.st['mixed_steps'] += 1
            self._tick(rows)
        for r in ad.can_run:
            r.prefix_len = r.extend_end
            if r is self.chunked:
                continue                                           # 中间块不出 token
            r.gen = 1
            r.ttft = self.now - r.arr
            r.last = self.now
            self.free -= PAGE
            r.kv += PAGE
            if r.gen >= r.max_new:
                self._finish(r)
            else:
                self.running.append(r)
        return ad

    def decode(self):
        batch = self.running
        assert self.free >= len(batch), 'decode 显存不够：本模拟不做 retract，调小负载'
        dt = self.srv.t_decode([r.inp + r.gen for r in batch])
        self.now += dt
        self.st['decode_steps'] += 1
        self.st['decode_time'] += dt
        self._tick(batch)
        self.decode_ct += 1
        self.gen_since += len(batch)
        if self.decode_ct % DECODE_LOG_INTERVAL == 0:
            gap = self.now - self.last_decode_t
            self.log(f'Decode batch, #running-req: {len(batch)}, #token: {self.cap - self.free}, '
                     f'token usage: {self.usage():.2f}, cuda graph: True, '
                     f'gen throughput (token/s): {self.gen_since / gap if gap > 0 else 0:.2f}, '
                     f'#queue-req: {len(self.waiting)}')
            self.last_decode_t, self.gen_since = self.now, 0

    def _tick(self, rows):
        for r in rows:
            self.free -= PAGE
            r.kv += PAGE
            r.itl.append(self.now - r.last)
            r.last = self.now
            r.gen += 1
            if r.gen >= r.max_new:
                self._finish(r)

    def _finish(self, r):
        r.end = self.now
        self.free += r.kv
        r.kv = 0

    def run(self, reqs):
        pend = deque(sorted(reqs, key=lambda r: r.arr))
        while True:
            while pend and pend[0].arr <= self.now + 1e-12:
                self.waiting.append(pend.popleft())
            busy = self.step()
            if not busy and self.chunked is None:
                if not pend:
                    break
                self.now = max(self.now, pend[0].arr)
        return reqs


# ---------------------------------------------------------------- 负载与指标
def wl_one(n_short=64, short_in=1024, short_out=512, long_in=32768, long_out=64, long_at=3.0):
    """64 路短请求同时进来、进入稳定 decode 后，t=3s 来一条 32k 的长 prompt。"""
    reqs = [Req(i, 'short', short_in, short_out, 0.0) for i in range(n_short)]
    reqs.append(Req(n_short, 'long', long_in, long_out, long_at))
    return reqs


def wl_poisson(rate=8.0, dur=60.0, short_in=1024, short_out=256, long_in=32768, long_out=128,
               long_every=6.0, seed=42):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t > dur:
            break
        reqs.append(Req(len(reqs), 'short', short_in, short_out, t))
    t = 3.0
    while t < dur:
        reqs.append(Req(len(reqs), 'long', long_in, long_out, t))
        t += long_every
    return reqs


def metrics(reqs, sim, window=None):
    """window=(t0, t1)：只统计落在这段时间里的 decode 间隔。"""
    shorts = [r for r in reqs if r.kind == 'short']
    longs = [r for r in reqs if r.kind == 'long']
    itl = []
    for r in reqs:
        t = r.arr + r.ttft
        for x in r.itl:
            t += x
            if window is None or window[0] < t <= window[1] + 1e-9:
                itl.append(x * 1e3)
    med = sim_stats.pct(itl, 50)
    big = [x for x in itl if x > 3 * med]
    dur = max(r.end for r in reqs) - min(r.arr for r in reqs)
    out = sum(r.gen for r in reqs)
    return dict(
        short_ttft_p50=sim_stats.pct([r.ttft * 1e3 for r in shorts], 50),
        short_ttft_p99=sim_stats.pct([r.ttft * 1e3 for r in shorts], 99),
        long_ttft=sim_stats.mean([r.ttft * 1e3 for r in longs]),
        itl_p50=med, itl_p99=sim_stats.pct(itl, 99), itl_max=max(itl) if itl else float('nan'),
        stall_frac=sum(big) / sum(itl) if itl else 0.0,
        out_tps=out / dur, dur=dur,
        long_chunks=sim_stats.mean([r.chunks for r in longs]), **sim.st)


def build(model, gpu, chunk, mixed, interval=0, min_bs=0, kv_dtype='bf16', verbose=False):
    srv = Server(model, gpu, 1, kv_dtype, chunk=chunk if chunk and chunk > 0 else None)
    cap = kv_tokens(model, gpu, chunk if chunk else 8192, kv_dtype=kv_dtype)['tokens']
    return Sim(srv, cap, chunk, mixed, interval, min_bs, verbose=verbose), cap


# ---------------------------------------------------------------- 场景
HDR = (f'{"模式":>6} {"chunk":>6} {"长请求块数":>8} {"长TTFT ms":>10} {"短TTFT p99":>10} '
       f'{"ITL p50":>8} {"ITL p99":>8} {"ITL max":>8} {"停顿占比":>7} {"prefill步":>8} {"decode步":>8}')


def row(mode, chunk, m):
    c = 'off' if chunk == -1 else str(chunk)
    return (f'{mode:>6} {c:>6} {m["long_chunks"]:>8.0f} {m["long_ttft"]:>10.0f} {m["short_ttft_p99"]:>10.0f} '
            f'{m["itl_p50"]:>8.1f} {m["itl_p99"]:>8.1f} {m["itl_max"]:>8.1f} {m["stall_frac"]:>7.1%} '
            f'{m["prefill_steps"]:>8d} {m["decode_steps"]:>8d}')


def scenario_one(a):
    print('# 64 条短请求（输入 1024、输出 512）t=0 进来，t=3s 来一条 32768 token 的长 prompt（输出 64）')
    print('# ITL 只统计 t=3s 之后、短请求全部结束之前的 decode 间隔；长请求的块数 = 它被切成几次 prefill')
    print(HDR)
    for mixed in (False, True):
        for chunk in a.chunks:
            if mixed and chunk == -1:
                continue
            sim, _ = build(a.model, a.gpu, chunk, mixed)
            reqs = sim.run(wl_one())
            t1 = max(r.end for r in reqs if r.kind == 'short')
            print(row('mixed' if mixed else '默认', chunk, metrics(reqs, sim, (3.0, t1))))
    srv = Server(a.model, a.gpu)
    print(f'# 参考：64 路 decode 一步 {srv.t_decode([1024 + 256] * 64) * 1e3:.1f} ms；'
          f'32768 token 整段 prefill {srv.t_prefill([(32768, 0)]) * 1e3:.0f} ms')


def scenario_interval(a):
    print('# 同一负载，打开 --prefill-decode-interval N：每跑一个 prefill 批，后面 N 轮只做 decode')
    print(f'{"chunk":>6} {"N":>4} {"min_bs":>6} {"长TTFT ms":>10} {"ITL p99":>8} {"ITL max":>8} {"停顿占比":>7} '
          f'{"被推迟轮数":>8}')
    for chunk in (2048, 8192):
        for n, mb in ((0, 0), (2, 0), (8, 0), (32, 0), (8, 80)):
            sim, _ = build(a.model, a.gpu, chunk, False, n, mb)
            reqs = sim.run(wl_one())
            t1 = max(r.end for r in reqs if r.kind == 'short')
            m = metrics(reqs, sim, (3.0, t1))
            print(f'{chunk:>6} {n:>4} {mb:>6} {m["long_ttft"]:>10.0f} {m["itl_p99"]:>8.1f} {m["itl_max"]:>8.1f} '
                  f'{m["stall_frac"]:>7.1%} {m["deferred"]:>8d}')
    print('# min_bs=80 大于在跑的 64 条，interval 每次都被清零，等于没开')


def scenario_poisson(a):
    print(f'# 短请求泊松到达 {a.rate:g} req/s（输入 1024、输出 256），每 6 s 一条 32768 token 的长 prompt（输出 128），共 60 s')
    print(f'{"模式":>6} {"chunk":>6} {"短TTFT p50":>10} {"短TTFT p99":>10} {"长TTFT ms":>10} {"ITL p50":>8} '
          f'{"ITL p99":>8} {"ITL max":>8} {"停顿占比":>7} {"out tok/s":>9}')
    for mixed in (False, True):
        for chunk in a.chunks:
            if mixed and chunk == -1:
                continue
            sim, _ = build(a.model, a.gpu, chunk, mixed)
            reqs = sim.run(wl_poisson(a.rate, seed=a.seed))
            m = metrics(reqs, sim)
            c = 'off' if chunk == -1 else str(chunk)
            print(f'{"mixed" if mixed else "默认":>6} {c:>6} {m["short_ttft_p50"]:>10.0f} {m["short_ttft_p99"]:>10.0f} '
                  f'{m["long_ttft"]:>10.0f} {m["itl_p50"]:>8.1f} {m["itl_p99"]:>8.1f} {m["itl_max"]:>8.1f} '
                  f'{m["stall_frac"]:>7.1%} {m["out_tps"]:>9.0f}')


def scenario_memory(a):
    print(f'# {MODELS[a.model].name}，mem_fraction_static 自动推导，chunk 对 KV 池的影响（off = --chunked-prefill-size -1）')
    print(f'{"GPU":>9} {"chunk":>6} {"激活token":>9} {"Graph MB":>9} {"预留 MB":>8} {"10G下限":>7} {"frac":>6} '
          f'{"KV tokens":>10} {"相对默认":>9}')
    for gpu in a.gpus:
        default_chunk, _ = tier(SMI_MIB[gpu], 1)
        base = kv_tokens(a.model, gpu, default_chunk)['tokens']
        for chunk in (2048, 4096, 8192, 16384, -1):
            k = kv_tokens(a.model, gpu, chunk)
            mark = '*' if chunk == default_chunk else ' '
            c = ('off' if chunk == -1 else str(chunk)) + mark
            print(f'{gpu:>9} {c:>6} {k["act"]:>9} {k["graph"]:>9.0f} {k["reserved"]:>8.0f} '
                  f'{"触发" if k["floor"] else "":>7} {k["frac"]:>6.3f} {k["tokens"]:>10,} '
                  f'{(k["tokens"] - base) / base:>+9.1%}')
    print('# * 是该档 GPU 的默认 chunk；KV tokens 为 0 表示权重放不下或池子为负')


def scenario_trace(a):
    sim, cap = build(a.model, a.gpu, a.chunked_prefill_size, a.enable_mixed_chunk,
                     a.prefill_decode_interval, a.prefill_decode_interval_min_bs)
    reqs = sim.run(wl_one(n_short=a.n_short))
    lines = [ln for ln in sim.lines if float(ln[3:11]) >= 2.9]
    print(f'# KV 池 {cap:,} token；chunk {a.chunked_prefill_size}，mixed {sim.mixed}，'
          f'interval {a.prefill_decode_interval}；只看 t ≥ 2.9s 的日志')
    for ln in lines[:a.log_head]:
        print(ln)
    long = [r for r in reqs if r.kind == 'long'][0]
    probe = reqs[0]
    t, gaps = probe.ttft, []
    for x in probe.itl:
        t += x
        if 2.95 < t < 3.0 + long.ttft + 0.1:
            gaps.append(x * 1e3)
    print(f'# 长请求 TTFT {long.ttft * 1e3:.0f} ms，切成 {long.chunks} 块；'
          f'短请求 #0 在这段时间里的 decode 间隔（ms）：{" ".join(f"{g:.0f}" for g in gaps[:24])}'
          + (' …' if len(gaps) > 24 else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='one', choices=['one', 'interval', 'poisson', 'memory', 'trace'])
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--gpu', default='h100-sxm', choices=SMI_MIB)
    ap.add_argument('--gpus', nargs='+', default=['rtx4090', 'l40s', 'h100-sxm', 'h200'], choices=SMI_MIB)
    ap.add_argument('--chunks', type=int, nargs='+', default=[-1, 2048, 4096, 8192, 16384])
    ap.add_argument('--chunked-prefill-size', type=int, default=8192)
    ap.add_argument('--enable-mixed-chunk', action='store_true')
    ap.add_argument('--prefill-decode-interval', type=int, default=0)
    ap.add_argument('--prefill-decode-interval-min-bs', type=int, default=0)
    ap.add_argument('--n-short', type=int, default=64)
    ap.add_argument('--rate', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-head', type=int, default=14)
    a = ap.parse_args()
    {'one': scenario_one, 'interval': scenario_interval, 'poisson': scenario_poisson,
     'memory': scenario_memory, 'trace': scenario_trace}[a.scenario](a)


if __name__ == '__main__':
    main()
