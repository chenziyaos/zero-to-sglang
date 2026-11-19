"""读 sglang 的 /metrics：一次抓取看累计，两次抓取看窗口；直方图按 Prometheus histogram_quantile 的规则估分位数。

服务端要带 --enable-metrics 启动。指标定义见 python/sglang/srt/observability/metrics_collector.py（主干 8fabf5e）。

两次抓取之间的时长不用自己记：sglang:scheduler_stage_seconds_total 把调度循环的墙钟时间
按阶段（含 idle）互斥地记账，所有 category 加起来就是这段时间的墙钟。

用法：
    python3 scrape_metrics.py metrics_t1.txt                     # 单次抓取：从启动到现在的累计
    python3 scrape_metrics.py metrics_t0.txt metrics_t1.txt      # 两次抓取：只看中间这段
    python3 scrape_metrics.py http://127.0.0.1:30000/metrics --interval 60   # 现抓两次，间隔 60 s
    python3 scrape_metrics.py metrics_t0.txt metrics_t1.txt --json
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\d+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

HIST = {
    'ttft': ('sglang:time_to_first_token_seconds', None),
    'itl': ('sglang:inter_token_latency_seconds', None),
    'tpot': ('sglang:request_time_per_output_token_seconds', None),
    'e2e': ('sglang:e2e_request_latency_seconds', None),
    'queue': ('sglang:queue_time_seconds', None),
    'prefill_fwd': ('sglang:per_stage_req_latency_seconds', ('stage', 'prefill_forward')),
}
HIST_CN = {'ttft': 'TTFT', 'itl': 'ITL', 'tpot': 'TPOT（逐请求）', 'e2e': 'E2E', 'queue': '排队',
           'prefill_fwd': 'prefill 前向'}


def parse_value(s):
    if s in ('+Inf', 'Inf'):
        return math.inf
    if s == '-Inf':
        return -math.inf
    if s == 'NaN':
        return math.nan
    return float(s)


def parse(text):
    """返回 {(name, frozenset(labels)): value} 和 {family: type}。"""
    samples, types = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            p = line.split(None, 3)
            if len(p) >= 4 and p[1] == 'TYPE':
                types[p[2]] = p[3]
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        labels = frozenset(LABEL.findall(m.group(3) or ''))
        samples[(m.group(1), labels)] = parse_value(m.group(4))
    return samples, types


def select(samples, name, where=None, drop=()):
    """取某个指标名的样本，按 where 过滤；同名多组标签（多个 tp_rank、多个 pid）相加。
    drop 里的标签先去掉再合并，返回 {剩余标签 frozenset: value}。"""
    out = {}
    for (n, labels), v in samples.items():
        if n != name:
            continue
        d = dict(labels)
        if where and any(d.get(k) != val for k, val in where.items()):
            continue
        key = frozenset((k, val) for k, val in d.items() if k not in drop)
        out[key] = out.get(key, 0.0) + v
    return out


def total(samples, name, where=None):
    vals = select(samples, name, where)
    if not vals:
        return None
    return sum(vals.values())


def by_label(samples, name, label, where=None):
    out = {}
    for labels, v in select(samples, name, where).items():
        k = dict(labels).get(label)
        out[k] = out.get(k, 0.0) + v
    return out


def buckets(samples, base, where=None):
    """直方图的累计桶 [(le, count)]，按 le 升序；所有标签组相加（多 rank 时 Prometheus 里也要 sum by (le)）。"""
    acc = {}
    for (n, labels), v in samples.items():
        if n != base + '_bucket':
            continue
        d = dict(labels)
        if where and any(d.get(k) != val for k, val in where.items()):
            continue
        le = parse_value(d['le'])
        acc[le] = acc.get(le, 0.0) + v
    return sorted(acc.items())


def delta_buckets(b1, b0):
    d0 = dict(b0)
    return [(le, c - d0.get(le, 0.0)) for le, c in b1]


def quantile(q, bks):
    """Prometheus promql/quantile.go bucketQuantile：q 落在哪个桶，就在桶内线性插值；
    第一个桶下界当 0；落进 +Inf 桶时返回最后一个有限上界。返回 (估计值, 所在桶下界, 上界)。"""
    if not bks or bks[-1][1] <= 0:
        return math.nan, math.nan, math.nan
    if bks[-1][0] != math.inf:
        return math.nan, math.nan, math.nan
    n = bks[-1][1]
    rank = q * n
    prev_le, prev_c = 0.0, 0.0
    for i, (le, c) in enumerate(bks):
        if c >= rank:
            if le == math.inf:
                last = bks[-2][0] if len(bks) > 1 else math.nan
                return last, last, math.inf
            lo = prev_le if i > 0 else (0.0 if le > 0 else le)
            if c == prev_c:
                return le, lo, le
            return lo + (le - lo) * (rank - prev_c) / (c - prev_c), lo, le
        prev_le, prev_c = le, c
    return math.nan, math.nan, math.nan


def hist_stats(s1, s0, base, where):
    b1 = buckets(s1, base, where)
    if not b1:
        return None
    b = delta_buckets(b1, buckets(s0, base, where)) if s0 is not None else b1
    cnt = b[-1][1]
    sm = (total(s1, base + '_sum', where) or 0.0) - ((total(s0, base + '_sum', where) or 0.0) if s0 else 0.0)
    out = {'count': cnt, 'mean': sm / cnt if cnt else math.nan}
    for q in (0.5, 0.9, 0.99):
        v, lo, hi = quantile(q, b)
        out[f'p{int(q * 100)}'] = v
        out[f'p{int(q * 100)}_bucket'] = (lo, hi)
    return out


def diff(s1, s0, name, where=None):
    v1 = total(s1, name, where)
    if v1 is None:
        return None
    return v1 - ((total(s0, name, where) or 0.0) if s0 is not None else 0.0)


def analyze(text1, text0=None, window=None):
    s1, types = parse(text1)
    s0 = parse(text0)[0] if text0 is not None else None
    R = {'window_mode': 'delta' if s0 is not None else 'cumulative'}

    stages1 = by_label(s1, 'sglang:scheduler_stage_seconds_total', 'category')
    stages0 = by_label(s0, 'sglang:scheduler_stage_seconds_total', 'category') if s0 else {}
    stages = {k: v - stages0.get(k, 0.0) for k, v in stages1.items()}
    wall = sum(stages.values())
    R['window_s'] = window if window else wall
    R['window_source'] = '参数给定' if window else ('scheduler_stage_seconds_total 各阶段之和' if wall else '未知')
    W = R['window_s'] or math.nan

    R['hist'] = {}
    for k, (base, lab) in HIST.items():
        st = hist_stats(s1, s0, base, dict([lab]) if lab else None)
        if st:
            R['hist'][k] = st

    R['rate'] = {
        'requests': diff(s1, s0, 'sglang:num_requests_total'),
        'prompt_tokens': diff(s1, s0, 'sglang:prompt_tokens_total'),
        'generation_tokens': diff(s1, s0, 'sglang:generation_tokens_total'),
    }
    rt = by_label(s1, 'sglang:realtime_tokens_total', 'mode')
    rt0 = by_label(s0, 'sglang:realtime_tokens_total', 'mode') if s0 else {}
    R['realtime'] = {k: v - rt0.get(k, 0.0) for k, v in rt.items()}
    eff = by_label(s1, 'sglang:prefill_effective_tokens_total', 'mode')
    eff0 = by_label(s0, 'sglang:prefill_effective_tokens_total', 'mode') if s0 else {}
    eff = {k: v - eff0.get(k, 0.0) for k, v in eff.items()}
    eff_all = sum(eff.values())
    R['cache_hit'] = (sum(v for k, v in eff.items() if k.endswith('_hit')) / eff_all) if eff_all else None

    R['stages'] = stages
    R['stage_share'] = {k: v / wall for k, v in stages.items()} if wall else {}
    cpu = diff(s1, s0, 'sglang:scheduler_process_cpu_seconds_total')
    R['sched_cpu_util'] = cpu / wall if (cpu is not None and wall) else None

    gp = by_label(s1, 'sglang:cuda_graph_passes_total', 'mode')
    gp0 = by_label(s0, 'sglang:cuda_graph_passes_total', 'mode') if s0 else {}
    gp = {k: v - gp0.get(k, 0.0) for k, v in gp.items()}
    dec = gp.get('decode_cuda_graph', 0.0) + gp.get('decode_none', 0.0)
    R['graph'] = gp
    R['decode_eager_share'] = gp.get('decode_none', 0.0) / dec if dec else None
    R['retracted'] = diff(s1, s0, 'sglang:num_retracted_requests_total')
    R['device_timer'] = any(n == 'sglang:forward_execution_seconds_total' for n, _ in s1)

    g = {}
    for name in ('num_running_reqs', 'num_queue_reqs', 'token_usage', 'gen_throughput', 'cache_hit_rate',
                 'is_cuda_graph', 'utilization', 'fwd_occupancy', 'max_total_num_tokens', 'num_used_tokens'):
        vals = list(select(s1, 'sglang:' + name).values())
        if vals:
            g[name] = sum(vals) if name.startswith('num_') or name == 'max_total_num_tokens' else vals[0]
    R['gauges'] = g
    R['per_second'] = {k: (v / W if v is not None else None) for k, v in R['rate'].items()}
    R['findings'] = diagnose(R)
    return R


def _f(x, unit='s'):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    if unit == 's':
        return f'{x * 1e3:.0f} ms' if x < 1 else f'{x:.2f} s'
    return f'{x:.2f}'


def diagnose(R):
    F = []
    H = R['hist']
    ttft, queue, itl, tpot = H.get('ttft'), H.get('queue'), H.get('itl'), H.get('tpot')
    if ttft and queue and ttft['count'] and not math.isnan(ttft['p99']):
        share = queue['p99'] / ttft['p99'] if ttft['p99'] > 0 else 0
        if share >= 0.5 and ttft['p99'] >= 0.5:
            F.append(('[调整]', 'QUEUE_DOMINANT',
                      f'TTFT p99 {_f(ttft["p99"])}，其中排队 p99 {_f(queue["p99"])}（{share:.0%}）。慢在等调度，'
                      f'prefill 本身 p99 {_f(H["prefill_fwd"]["p99"]) if "prefill_fwd" in H else "-"}：'
                      f'先看 token usage 和 #running-req 是否到顶，到顶是容量问题（加副本、限流或按 SGL-17 放宽准入），'
                      f'没到顶再查准入和 chunk'))
        else:
            F.append(('[正常]', 'QUEUE_OK', f'排队 p99 {_f(queue["p99"])}，占 TTFT p99 的 {share:.0%}'))
    if itl and tpot and itl['count'] and tpot['count'] and not math.isnan(itl['p99']):
        ratio = itl['p99'] / tpot['p99'] if tpot['p99'] > 0 else 0
        if ratio >= 3:
            F.append(('[提示]', 'ITL_SPIKES',
                      f'ITL p99 {_f(itl["p99"])} 是逐请求 TPOT p99 {_f(tpot["p99"])} 的 {ratio:.1f} 倍：'
                      f'TPOT 把卡顿摊平了，尖峰来自插进来的 prefill。流式体验看 ITL；要压尖峰看 SGL-06 的 chunk 与 mixed'))
    eager = R['decode_eager_share']
    if eager is not None:
        if eager > 0.05:
            run = R['gauges'].get('num_running_reqs')
            F.append(('[提示]', 'EAGER_DECODE',
                      f'{eager:.0%} 的 decode 前向没走 CUDA graph（cuda_graph_passes_total mode=decode_none），'
                      f'batch 超过了 decode 图的最大 bs。吃不吃亏看 GPU 前向和 CPU 发射谁长：小模型、大 TP 发射露出来，'
                      f'8B 单卡大 batch 几乎不吃亏。抓一段 trace 用 trace_summary.py 看 eager 步的 GPU 空闲，空闲高再把 '
                      f'--cuda-graph-max-bs-decode 调到稳态峰值之上，当前 #running-req {run:.0f}（见 SGL-09、SGL-17）'
                      if run is not None else f'{eager:.0%} 的 decode 前向没走 CUDA graph'))
        else:
            F.append(('[正常]', 'GRAPH_OK', f'decode 走 eager 的比例 {eager:.1%}'))
    sh = R['stage_share']
    if sh:
        cpu_side = sum(sh.get(k, 0.0) for k in ('recv_requests', 'process_input_requests',
                                                 'get_next_batch_to_run', 'process_queue', 'other'))
        idle = sh.get('idle', 0.0)
        wait = sh.get('process_batch_result', 0.0)
        if cpu_side >= 0.25:
            F.append(('[调整]', 'CPU_BOUND',
                      f'调度器自身的 CPU 阶段（收包、组 batch、前缀匹配等）占墙钟 {cpu_side:.0%}，GPU 在等 CPU。'
                      f'抓一次 torch profiler 看 scheduler.* 跨度里谁最长'))
        elif idle >= 0.3:
            F.append(('[提示]', 'IDLE', f'调度器 {idle:.0%} 的时间无事可做，负载没压满'))
        else:
            F.append(('[正常]', 'STAGES_OK',
                      f'CPU 阶段占 {cpu_side:.0%}，idle {idle:.0%}，process_batch_result {wait:.0%}'
                      f'（overlap 调度下它包含等 GPU 的同步时间，占比高说明 GPU 是瓶颈）'))
    if R['retracted']:
        W = R['window_s'] or 1
        per_min = R['retracted'] / W * 60
        tag = '[调整]' if per_min > 1 else '[正常]'
        F.append((tag, 'RETRACT', f'窗口内 retract {R["retracted"]:.0f} 次，{per_min:.1f} 次/分（官方建议每分钟一次以内）'))
    for k in ('ttft', 'queue'):
        st = H.get(k)
        if st and st['count']:
            lo, hi = st['p99_bucket']
            if hi != math.inf and lo > 0 and hi / lo >= 2:
                F.append(('[提示]', 'BUCKET_COARSE',
                          f'{HIST_CN[k]} p99 落在 ({lo:g}, {hi:g}] 这个桶里，桶宽是下界的 {hi / lo - 1:.0%}，'
                          f'插值误差可能有半个桶；要精确值看逐请求日志（reqstats.py）' +
                          ('，或启动时用 --bucket-time-to-first-token 加密' if k == 'ttft'
                           else '，这组桶写死在 metrics_collector.py 里，改不了')))
            elif hi == math.inf:
                F.append(('[调整]', 'BUCKET_OVERFLOW',
                          f'{HIST_CN[k]} p99 落进 +Inf 桶，只能报最后一个有限上界 {lo:g} s，真实值更大'))
    g = R['gauges']
    if not R['device_timer']:
        F.append(('[提示]', 'NO_DEVICE_TIMER',
                  'sglang:forward_execution_seconds_total 不在，fwd_occupancy 是 NaN：这两项要设环境变量 '
                  'SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 才记，GPU 实际前向时间只能靠 profiler'))
    if g.get('utilization') == 0:
        F.append(('[提示]', 'UTILIZATION_ZERO',
                  'sglang:utilization 恒为 0：8fabf5e 上这个 gauge 没有赋值的地方（metrics_reporter.py:1307 的 TODO），别拿它做告警'))
    return F


def report(R):
    print(f'== 窗口 ==')
    mode = '两次抓取之差' if R['window_mode'] == 'delta' else '单次抓取（从启动累计）'
    print(f'  {mode}，时长 {R["window_s"]:.1f} s（{R["window_source"]}）')
    print('== 吞吐 ==')
    ps = R['per_second']
    rt = R['realtime']
    W = R['window_s'] or math.nan
    print(f'  完成请求 {R["rate"]["requests"] or 0:.0f} 条，{(ps["requests"] or 0):.2f} 条/s；'
          f'输出 {(ps["generation_tokens"] or 0):.0f} token/s（按完成请求计）')
    if rt:
        print(f'  实时：prefill 计算 {rt.get("prefill_compute", 0) / W:.0f} token/s，decode {rt.get("decode", 0) / W:.0f} token/s，'
              f'前缀命中 {rt.get("prefill_cache", 0) / W:.0f} token/s')
    if R['cache_hit'] is not None:
        print(f'  窗口内前缀命中率 {R["cache_hit"]:.1%}（prefill_effective_tokens_total）')
    print('== 延迟（直方图估计）==')
    print(f'  {"指标":<14}{"样本":>8}{"均值":>10}{"p50":>10}{"p90":>10}{"p99":>10}   p99 所在桶')
    for k, st in R['hist'].items():
        lo, hi = st['p99_bucket']
        bk = '-' if math.isnan(lo) else f'({lo:g}, {hi:g}]'
        name = HIST_CN[k]
        pad = 14 - sum(2 if ord(ch) > 127 else 1 for ch in name)
        print(f'  {name}{" " * max(pad, 1)}{st["count"]:>8.0f}{_f(st["mean"]):>10}{_f(st["p50"]):>10}'
              f'{_f(st["p90"]):>10}{_f(st["p99"]):>10}   {bk}')
    if R['stage_share']:
        print('== 调度循环的时间去向（scheduler_stage_seconds_total）==')
        for k, v in sorted(R['stage_share'].items(), key=lambda x: -x[1]):
            if v >= 0.0005:
                print(f'  {k:<24}{R["stages"][k]:>9.2f} s  {v:>6.1%}')
        if R['sched_cpu_util'] is not None:
            print(f'  调度进程 CPU 时间 / 墙钟 = {R["sched_cpu_util"]:.2f}')
    if R['graph']:
        gp = R['graph']
        print('== CUDA graph ==')
        print('  ' + '，'.join(f'{k} {v:.0f}' for k, v in sorted(gp.items())) +
              (f'；decode 走 eager {R["decode_eager_share"]:.1%}' if R['decode_eager_share'] is not None else ''))
    g = R['gauges']
    print('== 抓取时刻的瞬时值 ==')
    print(f'  #running-req {g.get("num_running_reqs", 0):.0f}，#queue-req {g.get("num_queue_reqs", 0):.0f}，'
          f'token usage {g.get("token_usage", math.nan):.2f}，gen throughput {g.get("gen_throughput", math.nan):.0f} token/s，'
          f'is_cuda_graph {g.get("is_cuda_graph", math.nan):.0f}')
    print('== 诊断 ==')
    for tag, code, msg in R['findings']:
        print(f'  {tag} {code}：{msg}')


def fetch(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='一个或两个 exposition 文件；或一个 http(s) 地址')
    ap.add_argument('--interval', type=float, default=60.0, help='给 URL 时两次抓取的间隔秒数')
    ap.add_argument('--window', type=float, help='手动给窗口秒数（默认从 scheduler_stage_seconds_total 推出）')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.inputs[0].startswith(('http://', 'https://')):
        t0 = fetch(a.inputs[0])
        time.sleep(a.interval)
        t1 = fetch(a.inputs[0])
        texts = [t0, t1]
    else:
        texts = []
        for p in a.inputs[:2]:
            with open(p) as f:
                texts.append(f.read())
    R = analyze(texts[-1], texts[0] if len(texts) == 2 else None, a.window)
    if a.json:
        def clean(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        json.dump(clean(R), sys.stdout, ensure_ascii=False, indent=1)
        print()
    else:
        report(R)
    return R


if __name__ == '__main__':
    main()
