"""造本章的测试夹具：两次 /metrics 抓取、四份 torch profiler trace（0.6B 与 8B 各一对 eager / graph）、一份带 ReqTimeStats 的服务日志。

只用标准库。负载和单步耗时来自前面章节的模拟器，输出格式按 sglang 主干 8fabf5e 与 prometheus_client 0.21.1，
路径相对 python/sglang/srt/：
  - 指标名、类型、标签、桶边界：observability/metrics_collector.py:365-1290（调度器）、:1803-2075（tokenizer）
  - exposition 文本：prometheus_client exposition.generate_latest + multiprocess.MultiProcessCollector
    （多进程模式下 Counter 不出 _created；没设 multiprocess_mode 的 Gauge 带 pid 标签）
  - Prefill / Decode batch 日志：managers/scheduler_components/metrics_reporter.py:753、:962
  - ReqTimeStats 日志：managers/schedule_batch.py:2206 log_time_stats，正文见 observability/req_time_stats.py:1099
  - step[...] 与 scheduler.* 跨度：utils/profile_utils.py:462、utils/nvtx_utils.py:60、
    observability/scheduler_stage_metrics.py:128

负载：Qwen3-8B / 1×H100，输入 1024-2048、输出 128-256 均匀分布；0-45 s 每秒 8 条，45-90 s 每秒 14 条（泊松）。
夹具假设启动时给了 --cuda-graph-max-bs-decode 160，decode batch 超过 160 条就走 eager。

用法：
    python3 make_fixtures.py            # 在本目录写出 7 个夹具文件
"""
import bisect
import calendar
import gzip
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
sys.path.insert(0, os.path.join(HERE, '..', '09-overlap-cudagraph'))
import sim_serve  # noqa: E402
from cost import PRESETS, gpu_ms, n_kernels, proc_ms, sched_ms, with_  # noqa: E402
from common.specs import MODELS  # noqa: E402

MODEL = 'Qwen/Qwen3-8B'
SNAP = (45.0, 90.0)            # 两次抓取的时刻
GRAPH_MAX_BS = 160
DECODE_LOG_INTERVAL = 40
EPOCH = calendar.timegm((2026, 10, 28, 6, 0, 0))   # 日志里 asctime 按 UTC 写，测试不依赖时区
PID = 2817

# --------------------------------------------------------------------------- 桶边界（metrics_collector.py）
B_QUEUE = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
           100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000]
B_TTFT = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400]
B_ITL = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080, 0.100,
         0.200, 0.400, 0.600, 0.800, 1.0, 2.0, 4.0, 6.0, 8.0]
B_E2E = B_TTFT + [600, 1200, 1800, 2400]
B_STAGE = [0.001 * 1.62 ** i for i in range(30)]      # exponential_buckets(0.001, 1.62, 30)

STAGES = ('other', 'recv_requests', 'process_input_requests', 'get_next_batch_to_run', 'process_queue',
          'run_batch', 'process_batch_result', 'sanity_check_cache', 'idle')


# --------------------------------------------------------------------------- exposition
def go(v):
    """prometheus_client.utils.floatToGoString。"""
    v = float(v)
    if v == math.inf:
        return '+Inf'
    if v == -math.inf:
        return '-Inf'
    if math.isnan(v):
        return 'NaN'
    s = repr(v)
    dot = s.find('.')
    if v > 0 and dot > 6:
        mant = f'{s[0]}.{s[1:dot]}{s[dot + 1:]}'.rstrip('0.')
        return f'{mant}e+0{dot - 1}'
    return s


def lbl(d):
    if not d:
        return ''
    return '{' + ','.join(f'{k}="{d[k]}"' for k in sorted(d)) + '}'


class Registry:
    def __init__(self):
        self.fam = {}     # name -> [type, help, {labels_tuple: value or hist}]

    def _get(self, name, typ, doc):
        if name not in self.fam:
            self.fam[name] = [typ, doc, {}]
        return self.fam[name][2]

    def set(self, name, typ, doc, labels, v):
        self._get(name, typ, doc)[tuple(sorted(labels.items()))] = v

    def inc(self, name, doc, labels, v):
        d = self._get(name, 'counter', doc)
        k = tuple(sorted(labels.items()))
        d[k] = d.get(k, 0.0) + v

    def touch(self, name, doc, labels):
        self.inc(name, doc, labels, 0.0)

    def observe(self, name, doc, labels, buckets, v, n=1):
        d = self._get(name, 'histogram', doc)
        k = tuple(sorted(labels.items()))
        h = d.setdefault(k, {'b': list(buckets) + [math.inf], 'c': [0] * (len(buckets) + 1), 'sum': 0.0})
        h['sum'] += v * n
        for i, ub in enumerate(h['b']):
            if v <= ub:
                h['c'][i] += n
                break

    def text(self):
        out = []
        for name in sorted(self.fam):
            typ, doc, d = self.fam[name]
            out.append(f'# HELP {name} {doc}')
            out.append(f'# TYPE {name} {typ}')
            for k in sorted(d):
                labels = dict(k)
                v = d[k]
                if typ == 'histogram':
                    out.append(f'{name}_sum{lbl(labels)} {go(v["sum"])}')
                    cum = 0
                    for ub, c in zip(v['b'], v['c']):
                        cum += c
                        out.append(f'{name}_bucket{lbl({**labels, "le": go(ub)})} {go(cum)}')
                    out.append(f'{name}_count{lbl(labels)} {go(cum)}')
                else:
                    out.append(f'{name}{lbl(labels)} {go(v)}')
        return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------- 服务端模拟
def arrivals(seed=3):
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        rate = 8.0 if t < SNAP[0] else 14.0
        t += rng.expovariate(rate)
        if t >= SNAP[1]:
            return out
        out.append(t)


class _Req(sim_serve.Req):
    """sim_serve.Req 带 __slots__，子类不声明 __slots__ 才能挂 toks（每个 token 的时刻）。"""


def run_server():
    sim_serve.Req = _Req
    srv = sim_serve.Server('qwen3-8b', 'h100-sxm')
    arr = arrivals()
    steps = []
    reqs = sim_serve.simulate(srv, len(arr), 2048, 256, 0.5, arrivals=arr, steps=steps)
    for r in reqs:
        first = r.send + r.ttft
        r.toks = [first]
        for x in r.itl:
            r.toks.append(r.toks[-1] + x)
    return srv, reqs, steps


class State:
    """按时刻查询 running / queue / 已用 token。"""

    def __init__(self, reqs, cap):
        self.reqs = sorted(reqs, key=lambda r: r.send)
        self.sends = [r.send for r in self.reqs]
        self.cap = cap

    def at(self, t):
        hi = bisect.bisect_right(self.sends, t + 1e-12)
        run = que = used = pend = 0
        for r in self.reqs[:hi]:
            if r.end <= t:
                continue
            if r.toks[0] <= t:
                run += 1
                used += r.inp + bisect.bisect_right(r.toks, t)
            elif r.fwd is None or r.fwd > t:
                que += 1
                pend += r.inp
            else:
                run += 1               # 正在 prefill
                used += r.inp
        return run, que, used, pend


def stage_split(kind, bs, new_reqs, dur):
    """把一步的墙钟时间按 scheduler 阶段拆开（overlap 循环，见 scheduler.py:2064）。
    run_batch 只发 kernel；process_batch_result 里 copy_done.synchronize() 阻塞，等 GPU 的时间算在它头上。"""
    c = PRESETS['qwen3-8b']
    eager = kind == 'prefill' or bs > GRAPH_MAX_BS
    s = {'recv_requests': 0.00002, 'process_input_requests': 0.00005 * new_reqs,
         'get_next_batch_to_run': sched_ms(c, bs) / 1e3,
         'run_batch': (n_kernels(c) * c.op_us / 1e6) if eager else c.graph_us / 1e6}
    s['process_batch_result'] = max(proc_ms(c, bs) / 1e3, dur - sum(s.values()))
    return s


def build_metrics(srv, reqs, steps, T):
    R = Registry()
    sl = {'engine_type': 'unified', 'model_name': MODEL, 'moe_ep_rank': '0', 'pp_rank': '0', 'tp_rank': '0'}
    tl = {'engine_type': 'unified', 'model_name': MODEL}
    st = dict(tl, is_streaming='true')

    # tokenizer 侧：按事件发生的时刻计入
    for r in reqs:
        if r.toks[0] <= T:
            R.observe('sglang:time_to_first_token_seconds', 'Histogram of time to first token in seconds.',
                      st, B_TTFT, r.ttft)
        for a, b in zip(r.toks, r.toks[1:]):
            if b <= T:
                R.observe('sglang:inter_token_latency_seconds', 'Histogram of inter-token latency in seconds.',
                          tl, B_ITL, b - a)
        if r.end <= T:
            e2e = r.end - r.send
            R.observe('sglang:e2e_request_latency_seconds', 'Histogram of End-to-end request latency in seconds',
                      st, B_E2E, e2e)
            if r.out > 1:
                R.observe('sglang:request_time_per_output_token_seconds',
                          'Per-request TPOT in seconds: (finished_time - first_token_time) / (completion_tokens - 1).',
                          st, B_ITL, (e2e - r.ttft) / (r.out - 1))
            R.inc('sglang:prompt_tokens_total', 'Number of prefill tokens processed.', st, r.inp)
            R.inc('sglang:generation_tokens_total', 'Number of generation tokens processed.', st, r.out)
            R.inc('sglang:num_requests_total', 'Number of requests processed.', st, 1)
        # 调度器侧
        if r.fwd is not None and r.fwd <= T:
            R.observe('sglang:queue_time_seconds', 'Histogram of queueing time in seconds.', sl, B_QUEUE,
                      r.fwd - r.send)
        if r.toks[0] <= T:
            R.observe('sglang:per_stage_req_latency_seconds', 'The latency of each stage of requests.',
                      dict(sl, stage='prefill_forward'), B_STAGE, r.toks[0] - r.fwd)

    eff = 'Effective prefill tokens with retracted-request re-counts excluded, updated on each log interval. ' \
          'mode: device_hit, host_hit, storage_hit, input. Windowed prefix cache hit rate = rate(sum of *_hit) ' \
          '/ rate(sum of all modes); per-tier rate uses a single *_hit mode in the numerator.'
    for m in ('input', 'device_hit', 'host_hit', 'storage_hit'):
        R.touch('sglang:prefill_effective_tokens_total', eff, dict(sl, mode=m))
    stage_doc = 'Total scheduler-loop wall time exclusively attributed to each stage. forward_overlap (full, ' \
                'partial, none): full=both endpoints active, partial=one, none=no observed overlap (including ' \
                'unavailable timing). Not exact overlapped/exposed seconds.'
    for s in STAGES:
        R.touch('sglang:scheduler_stage_seconds_total', stage_doc, dict(sl, category=s, forward_overlap='none'))
    R.touch('sglang:scheduler_idle_seconds_total',
            'Total wall time while the scheduler has no runnable or queued work.', sl)
    R.touch('sglang:num_retracted_requests_total', 'Total number of retracted requests.', sl)

    rt_doc = 'Total number of tokens processed (updated on each log interval). mode: prefill_compute, ' \
             'prefill_cache, decode.'
    gp_doc = 'Total number of forward passes categorized by CUDA graph.'
    sends = sorted(r.send for r in reqs)
    prev_end, idle, busy, last_bs = 0.0, 0.0, 0.0, 0
    for kind, t0, t1, toks, ndec, nfirst in steps:
        if t1 > T:
            break
        if t0 > prev_end:
            idle += t0 - prev_end
        prev_end = t1
        busy += t1 - t0
        new = bisect.bisect_right(sends, t0) - bisect.bisect_right(sends, t0 - (t1 - t0))
        split = stage_split(kind, ndec if kind == 'decode' else nfirst, new, t1 - t0)
        for k, v in split.items():
            R.inc('sglang:scheduler_stage_seconds_total', stage_doc, dict(sl, category=k, forward_overlap='none'), v)
        if kind == 'prefill':
            R.inc('sglang:realtime_tokens_total', rt_doc, dict(sl, mode='prefill_compute'), toks - ndec)
            R.inc('sglang:prefill_effective_tokens_total', eff, dict(sl, mode='input'), toks - ndec)
            R.inc('sglang:cuda_graph_passes_total', gp_doc, dict(sl, mode='prefill_cuda_graph'), 1)
        else:
            R.inc('sglang:realtime_tokens_total', rt_doc, dict(sl, mode='decode'), ndec)
            g = ndec <= GRAPH_MAX_BS
            R.inc('sglang:cuda_graph_passes_total', gp_doc,
                  dict(sl, mode='decode_cuda_graph' if g else 'decode_none'), 1)
            last_bs = ndec
    idle += max(0.0, T - prev_end)
    R.inc('sglang:scheduler_stage_seconds_total', stage_doc, dict(sl, category='idle', forward_overlap='none'), idle)
    R.inc('sglang:scheduler_idle_seconds_total', 'Total wall time while the scheduler has no runnable or queued work.',
          sl, idle)
    # 同步等待是自旋，CPU 时间接近非空闲墙钟；空闲时阻塞在 zmq 收包上，几乎不占 CPU（夹具假设）
    R.inc('sglang:scheduler_process_cpu_seconds_total',
          'Total CPU time consumed by the scheduler process across all threads.', sl, 0.98 * busy + 0.01 * idle)

    run, que, used, _ = State(reqs, srv.cap).at(T)
    win = [s for s in steps if s[0] == 'decode' and s[2] <= T][-DECODE_LOG_INTERVAL:]
    gen = sum(s[4] for s in win) / (win[-1][2] - win[0][1]) if win else 0.0
    gauges = [
        ('sglang:num_running_reqs', 'The number of running requests.', run),
        ('sglang:num_queue_reqs', 'The number of requests in the waiting queue.', que),
        ('sglang:gen_throughput', 'The generation throughput (token/s).', round(gen, 2)),
        ('sglang:cache_hit_rate', 'The prefix cache hit rate.', 0.0),
        ('sglang:token_usage', 'The token usage.', round(used / srv.cap, 2)),
        ('sglang:num_used_tokens', 'The number of used tokens.', used),
        ('sglang:utilization', 'The utilization.', 0.0),
        ('sglang:fwd_occupancy', 'Forward pass GPU occupancy percentage.', math.nan),
        ('sglang:is_cuda_graph', 'Whether the batch is using CUDA graph.', 1.0 if last_bs <= GRAPH_MAX_BS else 0.0),
        ('sglang:max_total_num_tokens', 'Maximum total number of tokens in the KV cache pool.', srv.cap),
    ]
    for name, doc, v in gauges:
        R.set(name, 'gauge', doc, sl, v)
    R.set('sglang:num_retracted_reqs', 'gauge', 'The number of retracted requests.', dict(sl, pid=str(PID)), 0.0)
    return R.text()


# --------------------------------------------------------------------------- 服务日志
def stamp(t):
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(EPOCH + t))


def build_log(srv, reqs, steps):
    S = State(reqs, srv.cap)
    lines = []
    last_pre = last_dec = 0.0
    ndec, dec_tokens = 0, 0
    for kind, t0, t1, toks, nd, nfirst in steps:
        if kind == 'prefill':
            run, que, used, pend = S.at(t1)
            thr = toks / (t1 - last_pre) if t1 > last_pre else 0.0
            last_pre = t1
            lines.append((t1, f'Prefill batch, #new-seq: {max(nfirst, 1)}, #new-token: {toks - nd}, #cached-token: 0, '
                              f'token usage: {used / srv.cap:.2f}, #running-req: {run}, #queue-req: {que}, '
                              f'#pending-token: {pend}, cuda graph: True, input throughput (token/s): {thr:.2f}'))
        else:
            ndec += 1
            dec_tokens += nd
            if ndec % DECODE_LOG_INTERVAL == 0:
                run, que, used, _ = S.at(t1)
                thr = dec_tokens / (t1 - last_dec)
                last_dec, dec_tokens = t1, 0
                lines.append((t1, f'Decode batch, #running-req: {nd}, #token: {used}, token usage: {used / srv.cap:.2f}, '
                                  f'cuda graph: {nd <= GRAPH_MAX_BS}, gen throughput (token/s): {thr:.2f}, '
                                  f'#queue-req: {que}'))
    for r in reqs:
        body = (f'queue_duration={(r.fwd - r.send) * 1e3:.2f}ms, forward_duration={(r.end - r.fwd) * 1e3:.2f}ms, '
                f'entry_time={EPOCH + r.send:.3f}')
        rid = f'{random.Random(r.i).getrandbits(128):032x}'
        lines.append((r.end, f'ReqTimeStats(rid={rid}, input_len={r.inp}, cached_input_len=0, '
                             f'output_len={r.out}, attempts=0, type=unified): {body}'))
    lines.sort(key=lambda x: x[0])
    args = {'model_path': MODEL, 'tp_size': 1, 'chunked_prefill_size': srv.chunk,
            'cuda_graph_max_bs_decode': GRAPH_MAX_BS, 'enable_metrics': True,
            'enable_request_time_stats_logging': True, 'decode_log_interval': DECODE_LOG_INTERVAL}
    head = [f'[{stamp(0)}] server_args={args!r}']
    return '\n'.join(head + [f'[{stamp(t)} TP0] {m}' for t, m in lines]) + '\n'


# --------------------------------------------------------------------------- torch profiler trace
def layer_kernels(m, bs, ctx):
    """一层 decode 的 14 个 kernel 和各自读的字节数（小 kernel 记 0，按固定 1 µs 算）。"""
    q, kv = m.heads * m.head_dim, m.kv_heads * m.head_dim
    inter = {'Qwen3-0.6B': 3072, 'Qwen3-8B': 12288}[m.name]
    w = 2
    return [('input_rmsnorm', 0), ('qkv_proj_gemm', m.hidden * (q + 2 * kv) * w), ('q_rmsnorm', 0),
            ('k_rmsnorm', 0), ('rotary_emb', 0), ('kv_cache_store', 0),
            ('attention_decode', bs * ctx * 2 * kv * w), ('o_proj_gemm', q * m.hidden * w),
            ('fused_add_rmsnorm', 0), ('gate_up_proj_gemm', m.hidden * 2 * inter * w), ('silu_and_mul', 0),
            ('down_proj_gemm', inter * m.hidden * w), ('residual_add', 0), ('cast_copy', 0)]


def step_kernels(c, bs):
    m = MODELS[c.model]
    ks = [('embedding', 0)]
    for _ in range(m.layers):
        ks += layer_kernels(m, bs, c.ctx)
    ks += [('final_rmsnorm', 0), ('lm_head_gemm', m.hidden * m.vocab * 2), ('logits_softmax', 0),
           ('top_k_top_p_sampling', 0), ('next_token_copy', 0)]
    assert len(ks) == n_kernels(c)
    g = gpu_ms(c, bs) * 1e3            # µs
    floor = 1.0
    small = sum(1 for _, b in ks if b == 0)
    tot = sum(b for _, b in ks)
    return [(n, floor if b == 0 else (g - small * floor) * b / tot) for n, b in ks]


def build_trace(preset, bs, graph, steps=10):
    """overlap 循环：S(N) → L(N) → 等 N-1 的 D2H → P(N-1)，和 sim_overlap.simulate 一致；kernel 时长按读的字节数分。"""
    c = PRESETS[preset]
    m = MODELS[c.model]
    ks = step_kernels(c, bs)
    ev = []
    cpu = dict(pid=PID, tid=PID)
    gpu = dict(pid=0, tid=7)
    corr = 1000
    t, gpu_free, pending = 0.0, 0.0, None
    sk = bs * c.ctx

    def X(name, cat, a, b, where, args=None):
        e = dict(ph='X', cat=cat, name=name, ts=round(a, 3), dur=round(b - a, 3), **where)
        if args:
            e['args'] = args
        ev.append(e)

    for s in range(steps):
        X('scheduler.recv_requests', 'user_annotation', t, t + 20, cpu)
        t += 20
        a = t
        t += sched_ms(c, bs) * 1e3 - 20
        X('scheduler.get_next_batch_to_run', 'user_annotation', a, t, cpu)
        name = f'step[DECODE bs={bs} g_sq={bs} g_sqsq={bs} g_sqsk={sk} g_sk={sk} forward_id={s}]'
        rb0 = t
        kstart = []
        if graph:
            corr += 1
            X('cudaGraphLaunch', 'cuda_runtime', t + c.graph_us - 10, t + c.graph_us, cpu, {'correlation': corr})
            t += c.graph_us
            end = max(t, gpu_free)
            for n, d in ks:
                kstart.append((n, end, end + d, corr))
                end += d
        else:
            end = gpu_free
            for k, (n, d) in enumerate(ks):
                corr += 1
                issued = t + (k + 1) * c.op_us
                X('cudaLaunchKernel', 'cuda_runtime', issued - 3.0, issued, cpu, {'correlation': corr})
                st = max(issued, end)
                kstart.append((n, st, st + d, corr))
                end = st + d
            t += len(ks) * c.op_us
        X(name, 'user_annotation', rb0, t, cpu)
        X('scheduler.run_batch', 'user_annotation', rb0, t, cpu)
        for n, a2, b2, cr in kstart:
            X(n, 'kernel', a2, b2, gpu, {'device': 0, 'stream': 7, 'correlation': cr})
        X(name, 'gpu_user_annotation', kstart[0][1], kstart[-1][2], gpu)
        gpu_free = kstart[-1][2]
        d2h = gpu_free + c.d2h_ms * 1e3
        if pending is not None:
            a = t
            if pending > t:
                X('cudaEventSynchronize', 'cuda_runtime', t, pending, cpu)
            t = max(t, pending) + proc_ms(c, bs) * 1e3
            X('scheduler.process_batch_result', 'user_annotation', a, t, cpu)
        pending = d2h
    meta = [dict(ph='M', name='process_name', pid=PID, args={'name': 'python3 (sglang::scheduler_TP0)'}),
            dict(ph='M', name='process_name', pid=0, args={'name': 'GPU 0 (H100 SXM)'}),
            dict(ph='M', name='thread_name', pid=0, tid=7, args={'name': 'stream 7'})]
    return {'schemaVersion': 1, 'displayTimeUnit': 'ms', 'traceName': f'{m.name} bs={bs} '
            f'{"cuda graph" if graph else "eager"}', 'traceEvents': meta + ev}


def write_gz(path, obj):
    raw = json.dumps(obj, separators=(',', ':')).encode()
    with open(path, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as g:
            g.write(raw)


def main():
    srv, reqs, steps = run_server()
    for i, T in enumerate(SNAP):
        with open(os.path.join(HERE, f'metrics_t{i}.txt'), 'w') as f:
            f.write(build_metrics(srv, reqs, steps, T))
    with open(os.path.join(HERE, 'server_reqstats.log'), 'w') as f:
        f.write(build_log(srv, reqs, steps))
    write_gz(os.path.join(HERE, 'trace_eager.json.gz'), build_trace('qwen3-0.6b', 16, False))
    write_gz(os.path.join(HERE, 'trace_graph.json.gz'), build_trace('qwen3-0.6b', 16, True))
    # 夹具负载本身那台 8B：峰值 running 206 时 eager 和 graph 各抓一段
    write_gz(os.path.join(HERE, 'trace_8b_eager.json.gz'), build_trace('qwen3-8b', 206, False))
    write_gz(os.path.join(HERE, 'trace_8b_graph.json.gz'), build_trace('qwen3-8b', 206, True))
    print(f'{len(reqs)} 条请求，{len(steps)} 步；写出 metrics_t0.txt metrics_t1.txt server_reqstats.log '
          f'trace_eager.json.gz trace_graph.json.gz trace_8b_eager.json.gz trace_8b_graph.json.gz')


if __name__ == '__main__':
    main()
