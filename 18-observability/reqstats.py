"""读开了 --enable-request-time-stats-logging 的服务日志，把每条请求拆成排队 + 前向，找出最慢的那几条，
再对回同一时刻的 Prefill / Decode batch 日志行，看它进来时服务端处于什么状态。

ReqTimeStats 行格式（主干 8fabf5e，managers/schedule_batch.py:2206 与 observability/req_time_stats.py:1099）：
    ReqTimeStats(rid=…, input_len=…, cached_input_len=…, output_len=…, attempts=…, type=unified): \
        queue_duration=…ms, forward_duration=…ms, entry_time=<unix 秒>
  type=prefill / decode（PD 分离）时正文多出 bootstrap / transfer 等字段，这里一并解析成字典。
  entry_time 是墙钟（unix 秒，3 位小数）；日志行开头的 asctime 是本地时间、精确到秒。
  两者的时区差按「日志时刻 ≈ entry_time + queue + forward」自动估出来，取整到 15 分钟。

用法：
    python3 reqstats.py server_reqstats.log
    python3 reqstats.py server.log --top 5 --bin 10
    python3 reqstats.py server.log --json
"""
import argparse
import calendar
import json
import re
import sys
import time

TS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\] ')
REQ = re.compile(r'ReqTimeStats\((.*?)\): (.*)$')
KV = re.compile(r'(\w+)=([^,]+?)(?:ms| GB/s| MB)?(?:,|$)')
BATCH = re.compile(r'(Prefill|Decode) batch(?: \[\d+\])?, (.*)')
FIELD = re.compile(r'(#?[a-zA-Z][\w\- ()/]*?): ([^,]+)')


def num(s):
    s = s.strip()
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return float(s) if ('.' in s or 'e' in s) else int(s)
    except ValueError:
        return s


def parse_ts(s):
    return calendar.timegm(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


def parse(lines):
    reqs, batches = [], []
    for line in lines:
        m = TS.match(line)
        if not m:
            continue
        ts = parse_ts(m.group(1))
        body = line[m.end():].rstrip('\n')
        r = REQ.search(body)
        if r:
            head = dict((k, num(v)) for k, v in (p.split('=', 1) for p in r.group(1).split(', ') if '=' in p))
            dur = {}
            for k, v in KV.findall(r.group(2)):
                dur[k] = num(v)
            reqs.append({'log_ts': ts, **head, **dur})
            continue
        b = BATCH.search(body)
        if b:
            f = {k.strip(): num(v) for k, v in FIELD.findall(b.group(2))}
            batches.append({'ts': ts, 'kind': b.group(1), **f})
    return reqs, batches


def tz_offset(reqs):
    """日志时刻（本地，按 UTC 解析）减去 unix 墙钟的差，取整到 15 分钟。"""
    ds = sorted(r['log_ts'] - (r['entry_time'] + (r.get('queue_duration', 0) + r.get('forward_duration', 0)) / 1e3)
                for r in reqs if 'entry_time' in r)
    if not ds:
        return 0
    med = ds[len(ds) // 2]
    return round(med / 900) * 900


def pct(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def state_at(batches, t):
    """t（日志时钟）之前最近的一行 Prefill 和 Decode。"""
    pre = dec = None
    for b in batches:
        if b['ts'] > t:
            break
        if b['kind'] == 'Prefill':
            pre = b
        else:
            dec = b
    return pre, dec


def analyze(reqs, batches, top=5, bin_s=5):
    R = {'n': len(reqs), 'types': {}}
    for r in reqs:
        R['types'][r.get('type', '?')] = R['types'].get(r.get('type', '?'), 0) + 1
    if not reqs:
        return R
    off = tz_offset(reqs)
    R['tz_offset_s'] = off
    for r in reqs:
        r['q'] = r.get('queue_duration', 0) / 1e3
        r['f'] = r.get('forward_duration', 0) / 1e3
        r['total'] = r['q'] + r['f']
        r['t_in'] = r['entry_time'] + off if 'entry_time' in r else r['log_ts'] - r['total']
    qs, fs, ts_ = [r['q'] for r in reqs], [r['f'] for r in reqs], [r['total'] for r in reqs]
    R['dist'] = {k: {f'p{q}': pct(v, q) for q in (50, 90, 99)} | {'max': max(v)}
                 for k, v in (('queue', qs), ('forward', fs), ('total', ts_))}
    slow = [r for r in reqs if r['total'] >= pct(ts_, 99)]
    R['slow_queue_share'] = sum(r['q'] for r in slow) / sum(r['total'] for r in slow) if slow else 0.0
    # attempts 是 prefill_attempt_count：只有 PD 分离 prefill 侧的乐观预填会加（disaggregation/prefill.py:531、:1843），
    # unified 模式恒为 0；retract 不改它，retract 的代价体现在 queue_duration 里（多次排队累加）
    R['retried'] = sum(1 for r in reqs if r.get('type') == 'prefill' and (r.get('attempts') or 0) > 1)
    batches = sorted(batches, key=lambda b: b['ts'])
    R['top'] = []
    for r in sorted(reqs, key=lambda r: -r['total'])[:top]:
        pre, dec = state_at(batches, r['t_in'])
        R['top'].append({k: r.get(k) for k in ('rid', 'input_len', 'output_len', 'cached_input_len', 'attempts',
                                               'q', 'f', 'total', 't_in')}
                        | {'prefill_line': pre, 'decode_line': dec})
    # 按进入时刻分桶，看排队随时间怎么涨
    t0 = min(r['t_in'] for r in reqs)
    bins = {}
    for r in reqs:
        k = int((r['t_in'] - t0) // bin_s)
        bins.setdefault(k, []).append(r)
    R['bins'] = []
    for k in sorted(bins):
        rs = bins[k]
        _, dec = state_at(batches, t0 + (k + 1) * bin_s)
        R['bins'].append({'t': k * bin_s, 'n': len(rs), 'q50': pct([r['q'] for r in rs], 50),
                          'q_max': max(r['q'] for r in rs), 'f50': pct([r['f'] for r in rs], 50),
                          'running': dec.get('#running-req') if dec else None,
                          'queue': dec.get('#queue-req') if dec else None,
                          'usage': dec.get('token usage') if dec else None,
                          'graph': dec.get('cuda graph') if dec else None})
    R['findings'] = diagnose(R)
    return R


def diagnose(R):
    F = []
    d = R['dist']
    if R['slow_queue_share'] >= 0.3:
        F.append(('[调整]', 'SLOW_BY_QUEUE',
                  f'最慢 1% 的请求里，排队占总时长的 {R["slow_queue_share"]:.0%}：这些请求在等调度，先查进来那一刻的 '
                  f'#queue-req 和 token usage'))
    else:
        F.append(('[提示]', 'SLOW_BY_FORWARD',
                  f'最慢 1% 的请求里排队只占 {R["slow_queue_share"]:.0%}，慢在前向：输出长，或 decode batch 大导致每步慢，'
                  f'对照下面的分桶看 cuda graph 是否掉成 False'))
    big = [b for b in R['bins'] if b['graph'] is False]
    if big:
        F.append(('[调整]', 'EAGER_WINDOW',
                  f'{len(big)} 个时间桶里最近一行 Decode batch 是 cuda graph: False，从第 {big[0]["t"]} s 起；'
                  f'这段时间进来的请求前向中位数 {pct([b["f50"] for b in big], 50):.2f} s，'
                  f'之前是 {pct([b["f50"] for b in R["bins"] if b["graph"] is True] or [float("nan")], 50):.2f} s。'
                  f'前向变慢也可能只是 batch 变大、prefill 插得多，eager 是不是原因要抓 trace 看'))
    if R['retried']:
        F.append(('[提示]', 'PREFILL_RETRY', f'{R["retried"]} 条 prefill 侧请求 attempts > 1：乐观预填让出过，'
                                             f'对照 sglang:num_prefill_retries_total 看重试率'))
    if d['queue']['p99'] < 0.05:
        F.append(('[正常]', 'QUEUE_OK', f'排队 p99 {d["queue"]["p99"] * 1e3:.0f} ms'))
    return F


def report(R):
    print('== 请求 ==')
    print(f'  {R["n"]} 条 ReqTimeStats，类型 ' + '，'.join(f'{k} {v}' for k, v in R['types'].items()))
    if not R['n']:
        print('  [错误] 没找到 ReqTimeStats 行：启动时要加 --enable-request-time-stats-logging')
        return
    print(f'  日志时钟与 entry_time 的差 {R["tz_offset_s"] / 3600:+.2f} h（按时区自动对齐）')
    print('== 分布（秒）==')
    print(f'  {"":<10}{"p50":>8}{"p90":>8}{"p99":>8}{"max":>8}')
    for k, cn in (('queue', '排队'), ('forward', '前向'), ('total', '合计')):
        v = R['dist'][k]
        print(f'  {cn:<8}{v["p50"]:>8.2f}{v["p90"]:>8.2f}{v["p99"]:>8.2f}{v["max"]:>8.2f}')
    print(f'  最慢 1% 的请求里，排队占 {R["slow_queue_share"]:.0%}')
    print('== 最慢的几条，和它进来时的服务端状态 ==')
    for r in R['top']:
        print(f'  rid={r["rid"][:12]}… 输入 {r["input_len"]} 输出 {r["output_len"]}：排队 {r["q"]:.2f} s + 前向 {r["f"]:.2f} s'
              f' = {r["total"]:.2f} s，进入于 {time.strftime("%H:%M:%S", time.gmtime(r["t_in"]))}')
        pre, dec = r['prefill_line'], r['decode_line']
        if dec:
            print(f'    最近的 Decode 行：#running-req {dec.get("#running-req")}，#queue-req {dec.get("#queue-req")}，'
                  f'token usage {dec.get("token usage")}，cuda graph {dec.get("cuda graph")}')
        if pre:
            print(f'    最近的 Prefill 行：#queue-req {pre.get("#queue-req")}，#pending-token {pre.get("#pending-token")}')
    print('== 按进入时刻分桶 ==')
    print(f'  {"t s":>5}{"请求":>6}{"排队 p50":>10}{"排队 max":>10}{"前向 p50":>10}{"running":>9}{"queue":>7}{"usage":>7}  graph')
    for b in R['bins']:
        print(f'  {b["t"]:>5}{b["n"]:>6}{b["q50"]:>10.2f}{b["q_max"]:>10.2f}{b["f50"]:>10.2f}'
              f'{b["running"] if b["running"] is not None else "-":>9}{b["queue"] if b["queue"] is not None else "-":>7}'
              f'{b["usage"] if b["usage"] is not None else "-":>7}  {b["graph"]}')
    print('== 诊断 ==')
    for tag, code, msg in R['findings']:
        print(f'  {tag} {code}：{msg}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--bin', type=float, default=10.0, help='按进入时刻分桶的宽度，秒')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    with open(a.log, errors='replace') as f:
        reqs, batches = parse(f)
    R = analyze(reqs, batches, a.top, a.bin)
    if a.json:
        json.dump(R, sys.stdout, ensure_ascii=False, indent=1, default=str)
        print()
    else:
        report(R)
    return R


if __name__ == '__main__':
    main()
