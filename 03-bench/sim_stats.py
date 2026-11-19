"""读 `python -m sglang.benchmark.serving --output-details` 写出的 JSONL，重算指标、goodput，找拐点。

一行是一次压测（一个请求率或一个并发度），多行就是一次扫描。口径和 sglang 主干 8fabf5e 对齐：
  - 百分位：numpy 默认的线性插值（这里用纯 Python 实现，结果一致）
  - TPOT：(e2e - ttft) / (output_len - 1)，只统计 output_len > 1 的请求（serving.py calculate_metrics）
  - ITL：把每条请求的 itls 拼在一起算分布；客户端已经把一个 chunk 的间隔平均摊给了 chunk 里的 token
  - goodput：成功且所有 SLO 都满足才算好请求；attainment 的分母是全部发出的请求，失败按不达标计（goodput.py）
  - 单 token 响应的 TPOT 记 0
真实结果里没有逐请求的 e2e，按 ttft + sum(itls) 还原；没有发出时刻，稳态窗口只对模拟结果算。

用法：
    python3 sim_stats.py sample_sweep.jsonl
    python3 sim_stats.py sample_sweep.jsonl --goodput ttft:1000 tpot:40 --target 0.95
    python3 sim_stats.py sample_sweep.jsonl --detail 10       # 第 10 req/s 那一行的完整分位数和 ITL 尖峰
    python3 sim_stats.py my_run.jsonl --steady 0.8            # 只对带 start_times 的结果有效
"""
import argparse
import json
import math
import sys

PCTS = (50, 90, 95, 99)


def pct(xs, q):
    """numpy.percentile 的默认（linear）插值。"""
    if not xs:
        return float('nan')
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def std(xs):
    if not xs:
        return float('nan')
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def parse_slos(items):
    out = {}
    for it in items or []:
        k, v = it.split(':')
        k = k.strip().lower()
        if k not in ('ttft', 'tpot', 'e2el'):
            raise SystemExit(f'不认识的 SLO {k}，只支持 ttft / tpot / e2el（毫秒）')
        out[k] = float(v)
    return out


def per_request(det):
    """把 details 摊成逐请求记录，单位秒。"""
    n = len(det['output_lens'])
    errs = det.get('errors') or [''] * n
    e2els = det.get('e2els')
    rows = []
    for i in range(n):
        itl = det['itls'][i] or []
        ok = not errs[i] and det['output_lens'][i] > 0
        ttft = det['ttfts'][i]
        e2e = e2els[i] if e2els else ttft + sum(itl)
        out = det['output_lens'][i]
        tpot = (e2e - ttft) / (out - 1) if out > 1 else 0.0
        rows.append({'ok': ok, 'inp': det['input_lens'][i], 'out': out, 'ttft': ttft,
                     'e2e': e2e, 'tpot': tpot, 'itl': itl})
    return rows


def summarize(det, dur):
    rs = per_request(det)
    ok = [r for r in rs if r['ok']]
    ttft = [r['ttft'] for r in ok]
    e2e = [r['e2e'] for r in ok]
    tpot = [r['tpot'] for r in ok if r['out'] > 1]
    itl = [x for r in ok for x in r['itl']]
    tin, tout = sum(r['inp'] for r in ok), sum(r['out'] for r in ok)
    m = {
        'completed': len(ok),
        'total_input_tokens': tin,
        'total_output_tokens': tout,
        'request_throughput': len(ok) / dur,
        'input_throughput': tin / dur,
        'output_throughput': tout / dur,
        'total_throughput': (tin + tout) / dur,
        'concurrency': sum(e2e) / dur,
    }
    for name, xs in (('e2e_latency', e2e), ('ttft', ttft), ('tpot', tpot), ('itl', itl)):
        ms = [x * 1e3 for x in xs]
        m[f'mean_{name}_ms'] = mean(ms)
        m[f'median_{name}_ms'] = pct(ms, 50)
        m[f'std_{name}_ms'] = std(ms)
        for q in (90, 95, 99):
            m[f'p{q}_{name}_ms'] = pct(ms, q)
    m['max_itl_ms'] = max(itl) * 1e3 if itl else float('nan')
    return m


def goodput(det, slos, dur):
    rs = per_request(det)
    good, met = 0, dict.fromkeys(slos, 0)
    for r in rs:
        if not r['ok']:
            continue
        obs = {'ttft': r['ttft'] * 1e3, 'tpot': r['tpot'] * 1e3, 'e2el': r['e2e'] * 1e3}
        hit = [k for k, v in slos.items() if obs[k] <= v]
        for k in hit:
            met[k] += 1
        good += len(hit) == len(slos)
    n = len(rs)
    return {'goodput_slos_ms': slos, 'request_goodput': good / dur, 'slo_attainment': good / n,
            'slo_attainment_by_metric': {k: v / n for k, v in met.items()}}


def itl_spikes(det, k=3.0):
    """ITL 尖峰：超过中位数 k 倍的间隔占比、这些间隔在总生成时间里的占比、p99/p50。"""
    itl = [x for r in per_request(det) if r['ok'] for x in r['itl']]
    if not itl:
        return None
    med = pct(itl, 50)
    big = [x for x in itl if x > k * med]
    return {'k': k, 'median_ms': med * 1e3, 'frac': len(big) / len(itl),
            'time_frac': sum(big) / sum(itl), 'p99_over_p50': pct(itl, 99) / med,
            'max_ms': max(itl) * 1e3}


def steady_window(det, ratio=0.8):
    """sglang.benchmark.steady_state.find_steady_state_window 的同款算法，需要 start_times。"""
    st = det.get('start_times')
    if not st:
        return None
    rs = per_request(det)
    ev = {}
    for s, r in zip(st, rs):
        if r['ok'] and r['e2e'] > 0:
            ev[s] = ev.get(s, 0) + 1
            ev[s + r['e2e']] = ev.get(s + r['e2e'], 0) - 1
    ts = sorted(ev)
    c, iv = 0, []
    for a, b in zip(ts, ts[1:]):
        c += ev[a]
        iv.append((a, b, c))
    peak = max(x[2] for x in iv)
    th = max(1, math.ceil(peak * ratio))
    spans, cur = [], None
    for a, b, c in iv:
        if c >= th:
            cur = (cur[0], b) if cur else (a, b)
        elif cur:
            spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    w0, w1 = max(spans, key=lambda s: s[1] - s[0])
    # 窗口内的输出 token：逐 token 时间戳落在窗口里的才算
    tok = 0
    for s, r in zip(st, rs):
        if not r['ok']:
            continue
        t = s + r['ttft']
        tok += w0 <= t <= w1
        for x in r['itl']:
            t += x
            tok += w0 <= t <= w1
    ttft_in = [r['ttft'] * 1e3 for s, r in zip(st, rs) if r['ok'] and w0 <= s < w1]
    return {'start': w0, 'end': w1, 'dur': w1 - w0, 'peak': peak, 'threshold': th,
            'output_throughput': tok / (w1 - w0), 'p99_ttft_ms': pct(ttft_in, 99), 'n_start': len(ttft_in)}


class Run:
    def __init__(self, res, tag=''):
        self.res = res
        self.tag = tag
        self.rate = res.get('request_rate')
        self.conc = res.get('max_concurrency')
        self.dur = res['duration']
        self.has_det = 'itls' in res
        self.m = summarize(res, self.dur) if self.has_det else res

    @property
    def label(self):
        r = 'inf' if self.rate in ('inf', None) or self.rate == math.inf else f'{float(self.rate):g}'
        c = '' if self.conc is None else f' c={self.conc}'
        return (self.tag + ' ' if self.tag else '') + r + c

    def gp(self, slos):
        if self.has_det:
            return goodput(self.res, slos, self.dur)
        if 'slo_attainment' in self.res:
            return self.res
        return None


def knee(rows, slos, target):
    """满足「attainment ≥ target」的最大负载；以及输出吞吐增长不到 5% 的第一个点。"""
    best = None
    for r in rows:
        g = r.gp(slos)
        if g and g['slo_attainment'] >= target:
            best = r
    flat = None
    for a, b in zip(rows, rows[1:]):
        if b.m['output_throughput'] < a.m['output_throughput'] * 1.05:
            flat = a
            break
    return best, flat


def print_table(rows, slos, target=0.99):
    hdr = (f'{"负载":>10} {"req/s":>6} {"out tok/s":>9} {"并发":>6} {"TTFT p50":>9} {"TTFT p99":>9} '
           f'{"TPOT p50":>9} {"TPOT p99":>9} {"ITL p99":>8} {"ITL max":>8} {"goodput":>8} {"达标率":>6}')
    print(hdr)
    for r in rows:
        m, g = r.m, r.gp(slos)
        gs = f'{g["request_goodput"]:8.2f} {g["slo_attainment"]:6.1%}' if g else f'{"-":>8} {"-":>6}'
        print(f'{r.label:>10} {m["request_throughput"]:6.2f} {m["output_throughput"]:9.0f} {m["concurrency"]:6.1f} '
              f'{m["median_ttft_ms"]:9.0f} {m["p99_ttft_ms"]:9.0f} {m["median_tpot_ms"]:9.1f} {m["p99_tpot_ms"]:9.1f} '
              f'{m["p99_itl_ms"]:8.1f} {m.get("max_itl_ms", float("nan")):8.1f} {gs}')
    if len(rows) > 1 and slos:
        best, flat = knee(rows, slos, target)
        s = ' '.join(f'{k}≤{v:g}ms' for k, v in slos.items())
        print(f'# SLO {s}，达标率门槛 {target:.0%}：'
              + (f'最大可承受负载 {best.label}（goodput {best.gp(slos)["request_goodput"]:.2f} req/s）' if best else '没有一行达标'))
        if flat:
            print(f'# 输出吞吐在 {flat.label} 之后增长不到 5%，再往上加负载只会推高排队')


def print_detail(r, slos):
    m = r.m
    print(f'## 负载 {r.label}：完成 {m["completed"]}，时长 {r.dur:.1f} s，'
          f'输入 {m["input_throughput"]:.0f} tok/s，输出 {m["output_throughput"]:.0f} tok/s，平均并发 {m["concurrency"]:.1f}')
    print(f'{"指标":>8} {"mean":>9} {"p50":>9} {"p90":>9} {"p95":>9} {"p99":>9}')
    for name, lab in (('ttft', 'TTFT'), ('tpot', 'TPOT'), ('itl', 'ITL'), ('e2e_latency', 'E2E')):
        print(f'{lab:>8} {m[f"mean_{name}_ms"]:9.1f} {m[f"median_{name}_ms"]:9.1f} {m[f"p90_{name}_ms"]:9.1f} '
              f'{m[f"p95_{name}_ms"]:9.1f} {m[f"p99_{name}_ms"]:9.1f}')
    if r.has_det:
        sp = itl_spikes(r.res)
        print(f'# ITL 尖峰：超过中位数 {sp["k"]:g} 倍（{sp["median_ms"] * sp["k"]:.1f} ms）的间隔占 {sp["frac"]:.1%}，'
              f'却占了生成时间的 {sp["time_frac"]:.1%}；p99/p50 = {sp["p99_over_p50"]:.1f}，最长一次 {sp["max_ms"]:.0f} ms')
    g = r.gp(slos)
    if g:
        by = '，'.join(f'{k} {v:.1%}' for k, v in g['slo_attainment_by_metric'].items())
        print(f'# goodput {g["request_goodput"]:.2f} req/s，全部达标 {g["slo_attainment"]:.1%}（{by}）')


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(Run(json.loads(line), p if len(paths) > 1 else ''))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--goodput', nargs='+', metavar='KEY:VALUE', default=['ttft:2000', 'tpot:50'])
    ap.add_argument('--target', type=float, default=0.99, help='达标率门槛，默认 99%%')
    ap.add_argument('--detail', type=str, help='打印某一行的完整分位数：写请求率或 c=并发度，all 表示全部')
    ap.add_argument('--steady', type=float, help='稳态窗口的并发阈值比例，和 steady_state_serving 同义')
    a = ap.parse_args()
    slos = parse_slos(a.goodput)
    rows = load(a.files)
    if not rows:
        sys.exit('文件里没有结果')
    print_table(rows, slos, a.target)
    if a.detail:
        for r in rows:
            key = r.label.split()[-1]
            if a.detail == 'all' or a.detail in (key, r.label, f'{r.rate}') or \
                    (a.detail.startswith('c=') and str(r.conc) == a.detail[2:]):
                print()
                print_detail(r, slos)
    if a.steady:
        print()
        for r in rows:
            w = steady_window(r.res, a.steady) if r.has_det else None
            if not w:
                print(f'# {r.label}：结果里没有 start_times，算不了稳态窗口（真实 serving 输出不带逐请求发出时刻）')
                continue
            print(f'# {r.label}：稳态窗口 {w["start"]:.1f}–{w["end"]:.1f} s（{w["dur"]:.1f} s，并发 ≥ {w["threshold"]}/{w["peak"]}），'
                  f'窗口内输出 {w["output_throughput"]:.0f} tok/s，对比全程 {r.m["output_throughput"]:.0f}；'
                  f'窗口内发出的 {w["n_start"]} 条 TTFT p99 {w["p99_ttft_ms"]:.0f} ms')


if __name__ == '__main__':
    main()
