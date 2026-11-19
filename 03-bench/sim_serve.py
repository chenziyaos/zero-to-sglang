"""一台连续批处理服务的离散事件模拟：给定到达过程，算出每条请求的 TTFT、逐 token 间隔和端到端延迟。

目的不是预测绝对数字，而是在没有 GPU 的情况下看清三件事：
  1. 请求率逼近容量时，排队让 TTFT 先爆，吞吐却只是变平；
  2. 默认的 prefill 优先调度下，新请求的 prefill 会插进 decode 之间，ITL 出现尖峰，TPOT 却看不出来；
  3. 开环（按到达率发）和闭环（固定并发）压同一台机器，得到的延迟分布不一样。

服务端模型（对齐 sglang 主干 8fabf5e 的默认行为，做了简化）：
  - 容量：KV 池 token 数和请求上限取自 ../02-ledger/sim_ledger.py
  - 调度：每轮先看等待队列，能凑出 prefill 就跑 prefill（最多 chunked_prefill_size 个 token），
          否则对所有在跑请求做一步 decode（scheduler.get_next_batch_to_run）。
          --enable-mixed-chunk 打开后，prefill 那一步顺带给在跑请求各出一个 token
  - 准入：在跑请求的 输入+输出 之和不超过 KV 池（比 sglang 的 new_token_ratio 估计更保守）
  - 单步耗时：../02-ledger/sim_roofline.py 的下限除以利用率（decode 看 MBU，prefill 看 MFU），
              再加一个固定开销
客户端模型（对齐 sglang.benchmark.serving）：
  - --request-rate 有限时按泊松过程到达，inf 时全部在 0 时刻发出（get_request）
  - --max-concurrency 用信号量限流；拿到信号量之后才开始计时，所以客户端排队不计入 TTFT
  - 流式、每个 token 一个 chunk（stream_interval=1），不计网络

输出和 `python -m sglang.benchmark.serving --output-details` 同格式的 JSONL，可直接交给 sim_stats.py。
额外多两个字段：start_times（相对压测开始的发出时刻）和 arrival_times（到达时刻），真实压测结果里没有。

用法：
    python3 sim_serve.py --request-rate 8
    python3 sim_serve.py --sweep 2 4 6 8 10 12 14 --output-file sweep.jsonl
    python3 sim_serve.py --max-concurrency 64 --num-prompts 600
    python3 sim_serve.py --request-rate 10 --enable-mixed-chunk
"""
import argparse
import json
import math
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
from common.specs import GPUS, KV_DTYPE, MODELS  # noqa: E402
from sim_ledger import ledger  # noqa: E402
from sim_roofline import Step  # noqa: E402
import sim_stats  # noqa: E402


class Server:
    def __init__(self, model='qwen3-8b', gpu='h100-sxm', tp=1, kv_dtype='bf16', chunk=None,
                 max_running=None, mbu=0.7, mfu=0.55, overhead_ms=0.3, mixed=False):
        self.s = Step(model, gpu, tp, None, kv_dtype)
        L = ledger(model, gpu, tp, kv_dtype=kv_dtype)
        self.cap = L['tokens']
        self.chunk = chunk or L['chunk']
        self.max_running = max_running or L['running']
        self.mbu, self.mfu, self.oh, self.mixed = mbu, mfu, overhead_ms / 1e3, mixed

    def _t(self, fl, by):
        return max(fl / (self.s.peak * self.mfu), by / (self.s.bw * self.mbu)) + self.oh

    def t_decode(self, ctxs):
        d = self.s.decode(len(ctxs), sum(ctxs) / len(ctxs))
        return self._t(d['flops'], d['bytes'])

    def t_prefill(self, parts, decode_ctxs=()):
        """parts：[(本步新算的 token 数, 已有前缀长度)]；decode_ctxs：混合批里顺带 decode 的请求上下文。"""
        s = self.s
        n = sum(p for p, _ in parts) + len(decode_ctxs)
        p = s.prefill(n)
        fl = p['flops'] - s.attn_flops(n, n) / 2
        fl += sum(s.attn_flops(new, pre + new / 2) for new, pre in parts)
        fl += sum(s.attn_flops(1, c) for c in decode_ctxs)
        by = p['bytes'] + sum(s.kv_read(c) for c in decode_ctxs)
        return self._t(fl, by)


class Req:
    __slots__ = ('i', 'inp', 'out', 'arr', 'send', 'fwd', 'done_in', 'gen', 'ttft', 'itl', 'last', 'end')

    def __init__(self, i, inp, out, arr):
        self.i, self.inp, self.out, self.arr = i, inp, out, arr
        self.send = None
        self.fwd = None            # 第一次进 forward 的时刻（第一块 prefill 开始），对应 sglang 的 forward_entry_time
        self.done_in = 0
        self.gen = 0
        self.ttft = 0.0
        self.itl = []
        self.last = 0.0
        self.end = None


def random_lens(full, ratio, n, rng):
    """sglang benchmark/datasets/common.py compute_random_lens：[max(int(full*ratio),1), full] 上均匀取整。"""
    lo = max(int(full * ratio), 1)
    return [rng.randint(lo, full) for _ in range(n)]


def simulate(srv, n, inp, out, ratio=1.0, rate=math.inf, max_conc=None, seed=42, arrivals=None, steps=None):
    """arrivals：可选，直接给定 n 个到达时刻（变速率负载用）；steps：可选列表，逐步追加
    (kind 'prefill'/'decode', 开始, 结束, 本步 token 数, 本步 decode 请求数, 本步首块 prefill 的请求数)。两者都不给时行为不变。"""
    rng = random.Random(seed)
    ins, outs = random_lens(inp, ratio, n, rng), random_lens(out, ratio, n, rng)
    t, arr = 0.0, []
    for _ in range(n):
        arr.append(t)
        if rate != math.inf:
            t += rng.expovariate(rate)
    if arrivals is not None:
        arr = list(arrivals)
    reqs = [Req(i, ins[i], outs[i], arr[i]) for i in range(n)]

    pending = deque(reqs)          # 还没到达客户端发出点的请求
    client_q = deque()             # 已到达、在等信号量的请求（只在 --max-concurrency 时出现）
    waiting = deque()              # 服务端等待队列
    running = []
    inflight = 0
    now = 0.0
    finished = 0

    def send(r):
        nonlocal inflight
        r.send = now
        inflight += 1
        waiting.append(r)

    def admit_arrivals():
        while pending and pending[0].arr <= now + 1e-12:
            client_q.append(pending.popleft())
        while client_q and (max_conc is None or inflight < max_conc):
            send(client_q.popleft())

    def finish(r):
        nonlocal inflight, finished
        r.end = now
        inflight -= 1
        finished += 1

    while finished < n:
        admit_arrivals()
        if not waiting and not running:
            now = max(now, pending[0].arr)
            continue

        # ---- 调度：prefill 优先 ----
        parts, done = [], []
        budget = srv.chunk
        reserved = sum(r.inp + r.out for r in running)
        new_cnt = 0
        for r in waiting:
            if len(running) + new_cnt >= srv.max_running or budget <= 0:
                break
            if reserved + r.inp + r.out > srv.cap:
                break
            take = min(budget, r.inp - r.done_in)
            if r.fwd is None:
                r.fwd = now
            parts.append((take, r.done_in))
            r.done_in += take
            budget -= take
            reserved += r.inp + r.out
            new_cnt += 1
            if r.done_in == r.inp:
                done.append(r)
            else:
                break                       # 一次只有一条被切块的请求
        if parts:
            ctxs = [r.inp + r.gen for r in running] if srv.mixed else []
            t0 = now
            now += srv.t_prefill(parts, ctxs)
            if steps is not None:
                steps.append(('prefill', t0, now, sum(p for p, _ in parts) + len(ctxs), len(ctxs),
                              sum(1 for _, pre in parts if pre == 0)))
            for r in done:
                waiting.popleft()
            if srv.mixed:
                _decode_tick(running, now, finish)
            for r in done:
                r.ttft = now - r.send
                r.gen = 1
                r.last = now
                if r.out == 1:
                    finish(r)
                else:
                    running.append(r)
        else:
            t0 = now
            now += srv.t_decode([r.inp + r.gen for r in running])
            if steps is not None:
                steps.append(('decode', t0, now, len(running), len(running), 0))
            _decode_tick(running, now, finish)
        running[:] = [r for r in running if r.end is None]

    return reqs


def _decode_tick(running, now, finish):
    for r in running:
        r.itl.append(now - r.last)
        r.last = now
        r.gen += 1
        if r.gen == r.out:
            finish(r)


def to_result(reqs, rate, max_conc, args_dict):
    t0 = min(r.send for r in reqs)
    dur = max(r.end for r in reqs) - t0
    det = {
        'input_lens': [r.inp for r in reqs],
        'output_lens': [r.out for r in reqs],
        'ttfts': [round(r.ttft, 6) for r in reqs],
        'itls': [[round(x, 6) for x in r.itl] for r in reqs],
        'errors': ['' for _ in reqs],
        'start_times': [round(r.send - t0, 6) for r in reqs],
        'arrival_times': [round(r.arr - t0, 6) for r in reqs],
        'e2els': [round(r.end - r.send, 6) for r in reqs],
    }
    res = {'backend': 'sim', 'request_rate': 'inf' if rate == math.inf else rate,
           'max_concurrency': max_conc, 'duration': dur, **args_dict}
    res.update(sim_stats.summarize(det, dur))
    res.update(det)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--gpu', default='h100-sxm', choices=GPUS)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--kv-dtype', default='bf16', choices=KV_DTYPE)
    ap.add_argument('--chunked-prefill-size', type=int, help='默认按显存档位取，H100 是 8192')
    ap.add_argument('--max-running-requests', type=int)
    ap.add_argument('--enable-mixed-chunk', action='store_true')
    ap.add_argument('--mbu', type=float, default=0.7, help='decode 达到的带宽利用率')
    ap.add_argument('--mfu', type=float, default=0.55, help='prefill 达到的算力利用率')
    ap.add_argument('--overhead-ms', type=float, default=0.3, help='每步固定开销')
    ap.add_argument('--num-prompts', type=int, default=1000)
    ap.add_argument('--random-input-len', type=int, default=1024)
    ap.add_argument('--random-output-len', type=int, default=256)
    ap.add_argument('--random-range-ratio', type=float, default=1.0,
                    help='注意 sglang 的默认值是 0.0，长度在 1..full 均匀分布；这里默认 1.0 即固定长度')
    ap.add_argument('--request-rate', type=float, default=math.inf)
    ap.add_argument('--max-concurrency', type=int)
    ap.add_argument('--sweep', type=float, nargs='+', help='依次跑多个请求率')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-file', default='sim.jsonl')
    ap.add_argument('--goodput', nargs='+', metavar='KEY:VALUE', default=['ttft:2000', 'tpot:50'])
    a = ap.parse_args()

    srv = Server(a.model, a.gpu, a.tp, a.kv_dtype, a.chunked_prefill_size, a.max_running_requests,
                 a.mbu, a.mfu, a.overhead_ms, a.enable_mixed_chunk)
    print(f'# {MODELS[a.model].name} on {a.tp} x {GPUS[a.gpu].name}: KV 池 {srv.cap:,} token, '
          f'chunk {srv.chunk}, 请求上限 {srv.max_running}, MBU {a.mbu:.0%}, MFU {a.mfu:.0%}'
          + ('，mixed chunk' if a.enable_mixed_chunk else ''))
    print(f'# 单步耗时：decode b=1 {srv.t_decode([a.random_input_len]) * 1e3:.2f} ms，'
          f'b=64 {srv.t_decode([a.random_input_len + a.random_output_len // 2] * 64) * 1e3:.2f} ms，'
          f'prefill {srv.chunk} token {srv.t_prefill([(srv.chunk, 0)]) * 1e3:.1f} ms')
    meta = {'random_input_len': a.random_input_len, 'random_output_len': a.random_output_len,
            'random_range_ratio': a.random_range_ratio, 'num_prompts': a.num_prompts,
            'mixed_chunk': a.enable_mixed_chunk}
    rates = a.sweep or [a.request_rate]
    slos = sim_stats.parse_slos(a.goodput)
    with open(a.output_file, 'w') as f:
        rows = []
        for rate in rates:
            reqs = simulate(srv, a.num_prompts, a.random_input_len, a.random_output_len,
                            a.random_range_ratio, rate, a.max_concurrency, a.seed)
            res = to_result(reqs, rate, a.max_concurrency, meta)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            rows.append(sim_stats.Run(res))
    sim_stats.print_table(rows, slos)
    print(f'# 已写入 {a.output_file}，用 python3 sim_stats.py {a.output_file} 看分位数和 goodput')


if __name__ == '__main__':
    main()
