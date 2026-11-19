"""定位到原因以后先算账：同一份到达序列（夹具负载的后 45 s，每秒 14 条），换几种处理办法，看 TTFT、ITL、前向怎么变。

夹具里的慢请求有两个嫌疑：decode batch 超过图上限掉回 eager，和到达率上来以后容量不够。
trace 已经说明 8B 单卡 bs 206 时 eager 几乎不花钱（trace_summary.py trace_8b_*.json.gz），这里只比和容量有关的办法：
  base       原样：1 卡，chunk 8192，不开 mixed
  mixed      --enable-mixed-chunk：prefill 那一步顺带给在跑请求出 token，压 ITL 尖峰
  chunk2k    --chunked-prefill-size 2048 + mixed：每块更短，decode 每次被挡得更短
  replica2   加一个副本，请求按轮询分到两台（router round_robin），每台每秒 7 条
  limit10    入口限流到每秒 10 条，超出的请求在网关直接拒掉（429），只统计放进来的，另报拒绝比例

单步耗时用 03-bench/sim_serve.py 的 Roofline 模型，和 make_fixtures.py 造夹具是同一套，结果用来比方向和量级。

用法：
    python3 whatif.py
    python3 whatif.py --json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
import sim_serve  # noqa: E402

import make_fixtures as MF  # noqa: E402

WIN = MF.SNAP                # 只看 45-90 s 进来的请求


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(arr, chunk=None, mixed=False):
    srv = sim_serve.Server('qwen3-8b', 'h100-sxm', chunk=chunk, mixed=mixed)
    steps = []
    reqs = sim_serve.simulate(srv, len(arr), 2048, 256, 0.5, arrivals=arr, steps=steps)
    return reqs, steps


def stats(reqs, steps, n_gpu=1):
    W = [r for r in reqs if WIN[0] <= r.arr < WIN[1]]
    itl = [x for r in W for x in r.itl]
    dec = [s for s in steps if s[0] == 'decode' and WIN[0] <= s[1] < WIN[1]]
    pre = [s for s in steps if s[0] == 'prefill' and WIN[0] <= s[1] < WIN[1]]
    pt = sum(s[2] - s[1] for s in pre)
    dt = sum(s[2] - s[1] for s in dec)
    return {
        'n': len(W), 'gpus': n_gpu,
        'ttft_p99': pct([r.ttft for r in W], 0.99),
        'queue_p99': pct([r.fwd - r.send for r in W], 0.99),
        'itl_p99': pct(itl, 0.99),
        'itl_max': max(itl) if itl else float('nan'),
        'fwd_p50': pct([r.end - r.fwd for r in W], 0.50),
        'e2e_p99': pct([r.end - r.send for r in W], 0.99),
        'prefill_share': pt / (pt + dt) if pt + dt else float('nan'),
        'bs_max': max((s[4] for s in dec), default=0),
        'over_graph': sum(1 for s in dec if s[4] > MF.GRAPH_MAX_BS) / len(dec) if dec else 0.0,
    }


def merge(parts):
    """两台各自的结果合成一份：分位数在合并后的请求上重算。"""
    reqs = [r for rs, _ in parts for r in rs]
    steps = [s for _, ss in parts for s in ss]
    R = stats(reqs, steps, len(parts))
    R['bs_max'] = max(stats(rs, ss)['bs_max'] for rs, ss in parts)
    return R


def scenarios():
    arr = MF.arrivals()
    out = {}
    out['base'] = stats(*run(arr))
    out['mixed'] = stats(*run(arr, mixed=True))
    out['chunk2k'] = stats(*run(arr, chunk=2048, mixed=True))
    out['replica2'] = merge([run(arr[0::2]), run(arr[1::2])])
    # 限流：令牌桶每秒 10 个、桶深 10，后半段没拿到令牌的请求直接拒掉
    lim, tokens, last = [], 10.0, WIN[0]
    for t in arr:
        if t < WIN[0]:
            lim.append(t)
            continue
        tokens = min(10.0, tokens + (t - last) * 10)
        last = t
        if tokens >= 1:
            tokens -= 1
            lim.append(t)
    R = stats(*run(lim))
    n_win = sum(1 for t in arr if t >= WIN[0])
    R['rejected'] = 1 - R['n'] / n_win
    out['limit10'] = R
    return out


LABEL = {'base': '原样', 'mixed': '开 mixed', 'chunk2k': 'chunk 2048 + mixed', 'replica2': '两副本轮询',
         'limit10': '限流 10 条/s'}


def pad(s, n):
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    return s + ' ' * max(0, n - w)


def report(out):
    print('同一份到达序列，只看 45-90 s 进来的请求（秒；ITL 毫秒）')
    print('  ' + pad('办法', 20) + '卡  TTFT p99  排队 p99   ITL p99   ITL max  前向 p50   E2E p99  prefill 占比  bs max  超图上限')
    for k, R in out.items():
        print(f'  {pad(LABEL[k], 20)}{R["gpus"]:>2}{R["ttft_p99"]:>10.2f}{R["queue_p99"]:>10.2f}{R["itl_p99"] * 1e3:>10.0f}'
              f'{R["itl_max"] * 1e3:>10.0f}{R["fwd_p50"]:>10.2f}{R["e2e_p99"]:>10.2f}{R["prefill_share"]:>14.0%}'
              f'{R["bs_max"]:>8}{R["over_graph"]:>10.0%}')
    b = out['base']
    r2 = out['replica2']
    print(f'  两副本：TTFT p99 {b["ttft_p99"]:.2f} → {r2["ttft_p99"]:.2f} s，前向中位数 {b["fwd_p50"]:.2f} → '
          f'{r2["fwd_p50"]:.2f} s，decode batch 峰值 {b["bs_max"]} → {r2["bs_max"]}，图上限不用动')
    m = out['mixed']
    print(f'  mixed：ITL max {b["itl_max"] * 1e3:.0f} → {m["itl_max"] * 1e3:.0f} ms；decode 搭在 prefill 上不再停，'
          f'请求走得快，running 峰值 {b["bs_max"]} → {m["bs_max"]}，TTFT p99 也跟着降到 {m["ttft_p99"]:.2f} s')
    L = out['limit10']
    print(f'  限流：放进来的请求 TTFT p99 {L["ttft_p99"]:.2f} s，代价是后半段 {L["rejected"]:.0%} 的请求被拒')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    out = scenarios()
    if a.json:
        json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
        print()
    else:
        report(out)
    return out


if __name__ == '__main__':
    main()
