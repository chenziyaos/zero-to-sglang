"""读 torch profiler 导出的 Chrome trace（*.trace.json.gz），回答一个问题：这一段时间里 GPU 在忙什么、在等谁。

sglang 的 trace 里有三类东西可以对上号（主干 8fabf5e）：
  - CPU 线程上的 scheduler.* 跨度：调度循环的各阶段（scheduler_stage_metrics.py:128，profile_range）
  - step[DECODE bs=… …] 跨度：每次前向一个，CPU 一份（user_annotation）、GPU 流上一份（gpu_user_annotation），
    开 detailed_annotations 后带 g_sq / g_sk 等聚合量（profile_utils.py:462）
  - kernel 事件：cat=kernel，靠 args.correlation 和 CPU 侧的 cudaLaunchKernel / cudaGraphLaunch 配对

输出：
  1. 每步的 GPU 忙碌时间、墙钟步长、GPU 空闲比例
  2. kernel 按名字聚合的耗时排行
  3. CPU 侧各 scheduler.* 跨度的占比
  4. 判定：GPU 空闲比例高且 CPU 一直在发 kernel，是发射受限（launch-bound）；GPU 几乎不空，是 GPU 受限
  5. 给了 detailed_annotations 时，按 step 名字里的 g_sk 算每步 KV 读量，和实测步长比一个 Roofline 下限

用法：
    python3 trace_summary.py trace_eager.json.gz
    python3 trace_summary.py trace_eager.json.gz trace_graph.json.gz      # 两份对比
    python3 trace_summary.py /tmp/profiles/1730000000.0-TP-0.trace.json.gz --top 15
"""
import argparse
import gzip
import json
import re
import sys

STEP = re.compile(r'^step\[(DECODE|EXTEND|MIXED|TARGET_VERIFY|VERIFY|DRAFT_EXTEND|DRAFT|IDLE)\b([^\]]*)\]$')
KV = re.compile(r'(\w+)=(\d+)')
LAUNCH = ('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel', 'cuLaunchKernelEx', 'cudaGraphLaunch',
          'cudaMemcpyAsync', 'cudaLaunchCooperativeKernel')


def load(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        obj = json.load(f)
    return obj['traceEvents'] if isinstance(obj, dict) else obj


def parse_step(name):
    m = STEP.match(name)
    if not m:
        return None
    d = {'mode': m.group(1)}
    for k, v in KV.findall(m.group(2)):
        d[k] = int(v)
    return d


def union_len(iv):
    """区间并集的总长。"""
    tot, end = 0.0, None
    start = None
    for a, b in sorted(iv):
        if end is None or a > end:
            if end is not None:
                tot += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        tot += end - start
    return tot


def summarize(events):
    X = [e for e in events if e.get('ph') == 'X']
    kernels = [e for e in X if e.get('cat') == 'kernel']
    gpu_steps = [e for e in X if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('step[')]
    cpu_steps = [e for e in X if e.get('cat') == 'user_annotation' and e['name'].startswith('step[')]
    sched = [e for e in X if e.get('cat') == 'user_annotation' and e['name'].startswith('scheduler.')]
    launches = [e for e in X if e.get('cat') == 'cuda_runtime' and e['name'] in LAUNCH]
    syncs = [e for e in X if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e['name']]
    graph_launches = sum(1 for e in launches if e['name'] == 'cudaGraphLaunch')

    R = {'n_kernels': len(kernels), 'n_launch': len(launches), 'graph_launches': graph_launches}
    if not kernels:
        R['error'] = '没有 kernel 事件：activities 里没开 GPU，或 trace 还没写完'
        return R
    t0 = min(e['ts'] for e in X)
    t1 = max(e['ts'] + e['dur'] for e in X)
    R['span_us'] = t1 - t0

    # 每步：用 GPU 侧 step 跨度对 kernel 归属，墙钟取相邻两步开始时刻的间隔
    steps = []
    cs = sorted(cpu_steps, key=lambda e: e['ts'])
    gs = sorted(gpu_steps, key=lambda e: e['ts'])
    for i, g in enumerate(gs):
        a, b = g['ts'], g['ts'] + g['dur']
        ks = [k for k in kernels if a - 1e-6 <= k['ts'] and k['ts'] + k['dur'] <= b + 1e-6]
        busy = union_len([(k['ts'], k['ts'] + k['dur']) for k in ks])
        info = parse_step(g['name']) or {}
        wall = None
        if i + 1 < len(gs):
            # 墙钟取 GPU 侧相邻两步的开始间隔：GPU 受限时 CPU 会领先一两步，按 CPU 切会让空闲比例算成负数
            wall = max(gs[i + 1]['ts'] - a, cs[i + 1]['ts'] - cs[i]['ts'] if i + 1 < len(cs) else 0.0)
        steps.append({'name': g['name'], 'info': info, 'gpu_span': b - a, 'busy': busy, 'wall': wall,
                      'n_kernels': len(ks)})
    R['steps'] = steps

    # 稳态：去掉第一步（前面没有上一步可以重叠）和最后一步（没有下一步来定墙钟）
    body = [s for s in steps[1:] if s['wall']]
    if body:
        wall = sum(s['wall'] for s in body)
        busy = sum(s['busy'] for s in body)
        R['wall_per_step'] = wall / len(body)
        R['busy_per_step'] = busy / len(body)
        R['gpu_idle'] = 1 - busy / wall if wall else 0.0
        R['kernels_per_step'] = sum(s['n_kernels'] for s in body) / len(body)
    else:
        busy = union_len([(k['ts'], k['ts'] + k['dur']) for k in kernels])
        R['wall_per_step'] = R['busy_per_step'] = None
        R['gpu_idle'] = 1 - busy / R['span_us']

    # kernel 排行
    agg = {}
    for k in kernels:
        a = agg.setdefault(k['name'], [0, 0.0])
        a[0] += 1
        a[1] += k['dur']
    ktot = sum(v[1] for v in agg.values())
    R['top'] = sorted(((n, c, d, d / ktot) for n, (c, d) in agg.items()), key=lambda x: -x[2])
    R['kernel_total_us'] = ktot
    R['small_kernels'] = sum(1 for k in kernels if k['dur'] < 3.0) / len(kernels)

    # CPU 侧：scheduler.* 和发射、同步
    cpu_pids = {e['pid'] for e in sched} or {e['pid'] for e in launches}
    cpu_span = R['span_us']
    sagg = {}
    for e in sched:
        sagg[e['name']] = sagg.get(e['name'], 0.0) + e['dur']
    R['sched'] = {k: v / cpu_span for k, v in sagg.items()}
    R['launch_busy'] = union_len([(e['ts'], e['ts'] + e['dur']) for e in launches]) / cpu_span
    R['sync_wait'] = sum(e['dur'] for e in syncs) / cpu_span
    R['cpu_pids'] = sorted(cpu_pids)

    # 发射与 kernel 之间的排队：kernel 开始 - 对应 launch 结束
    corr = {e.get('args', {}).get('correlation'): e for e in launches}
    lags = []
    for k in kernels:
        c = k.get('args', {}).get('correlation')
        if c in corr and corr[c]['name'] != 'cudaGraphLaunch':
            lags.append(k['ts'] - (corr[c]['ts'] + corr[c]['dur']))
    lags.sort()
    R['launch_lag_p50'] = lags[len(lags) // 2] if lags else None
    R['launch_lag_small'] = (sum(1 for x in lags if x < 2.0) / len(lags)) if lags else None
    R['verdict'] = verdict(R)
    return R


def verdict(R):
    idle = R['gpu_idle']
    if R['graph_launches'] and not (R['launch_lag_p50'] is not None):
        if idle < 0.15:
            return ('GPU_BOUND', f'CUDA graph 回放，GPU 空闲 {idle:.0%}：GPU 是瓶颈，往下看 kernel 排行')
        return ('CPU_GAP', f'CUDA graph 回放，GPU 仍空闲 {idle:.0%}：空隙在两步之间（调度、结果处理、同步），'
                           f'看 scheduler.* 跨度')
    if idle >= 0.3 and R['launch_lag_small'] is not None and R['launch_lag_small'] >= 0.5:
        return ('LAUNCH_BOUND', f'GPU 空闲 {idle:.0%}，{R["launch_lag_small"]:.0%} 的 kernel 在 launch 返回后 2 µs 内就开跑：'
                                f'GPU 一直在等 CPU 发下一个 kernel。小 kernel 占 {R["small_kernels"]:.0%}。'
                                f'该走 CUDA graph（检查 bs 是否超过 --cuda-graph-max-bs-decode）或做算子融合')
    if idle >= 0.15:
        return ('CPU_GAP', f'GPU 空闲 {idle:.0%}，但 kernel 不是一发就跑：空隙在步与步之间，看 scheduler.* 跨度')
    return ('GPU_BOUND', f'GPU 空闲 {idle:.0%}：GPU 是瓶颈，往下看 kernel 排行，对照 Roofline 判断是带宽还是算力')


def report(path, R, top=10):
    print(f'== {path} ==')
    if 'error' in R:
        print(f'  [错误] {R["error"]}')
        return
    print(f'  跨度 {R["span_us"] / 1e3:.2f} ms，{len(R["steps"])} 个 step，{R["n_kernels"]} 个 kernel，'
          f'{R["n_launch"]} 次发射（其中 cudaGraphLaunch {R["graph_launches"]}）')
    if R['wall_per_step']:
        print(f'  稳态每步：墙钟 {R["wall_per_step"] / 1e3:.2f} ms，GPU 忙 {R["busy_per_step"] / 1e3:.2f} ms，'
              f'GPU 空闲 {R["gpu_idle"]:.0%}，每步 {R["kernels_per_step"]:.0f} 个 kernel')
    s0 = R['steps'][0] if R['steps'] else None
    if s0 and s0['info']:
        info = s0['info']
        extra = ', '.join(f'{k}={v}' for k, v in info.items() if k != 'mode')
        print(f'  step 标注：{info["mode"]} {extra}')
    print(f'  kernel 排行（共 {R["kernel_total_us"] / 1e3:.2f} ms GPU 时间，<3 µs 的小 kernel 占个数的 {R["small_kernels"]:.0%}）')
    for n, c, d, sh in R['top'][:top]:
        print(f'    {n:<28}{c:>6} 次 {d / 1e3:>9.2f} ms  {sh:>6.1%}  均 {d / c:>7.1f} µs')
    print('  CPU 侧（占跨度）：' + '，'.join(f'{k} {v:.0%}' for k, v in sorted(R['sched'].items(), key=lambda x: -x[1]))
          + f'；发射调用 {R["launch_busy"]:.0%}，同步等待 {R["sync_wait"]:.0%}')
    if R['launch_lag_p50'] is not None:
        print(f'  launch 返回到 kernel 开跑：p50 {R["launch_lag_p50"]:.1f} µs，<2 µs 的占 {R["launch_lag_small"]:.0%}')
    code, msg = R['verdict']
    tag = '[调整]' if code == 'LAUNCH_BOUND' else '[提示]'
    print(f'  {tag} {code}：{msg}')


def compare(a, b, na, nb):
    if not (a.get('wall_per_step') and b.get('wall_per_step')):
        return
    print('== 对比 ==')
    print(f'  {na}：每步 {a["wall_per_step"] / 1e3:.2f} ms，GPU 空闲 {a["gpu_idle"]:.0%}')
    print(f'  {nb}：每步 {b["wall_per_step"] / 1e3:.2f} ms，GPU 空闲 {b["gpu_idle"]:.0%}')
    print(f'  步长比 {a["wall_per_step"] / b["wall_per_step"]:.2f}x，GPU 忙碌时间比 '
          f'{a["busy_per_step"] / b["busy_per_step"]:.2f}x（kernel 本身没变，差在空隙）')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    Rs = [summarize(load(p)) for p in a.traces]
    if a.json:
        out = []
        for p, R in zip(a.traces, Rs):
            R = dict(R)
            R.pop('steps', None)
            R['top'] = R.get('top', [])[:a.top]
            out.append({'trace': p, **R})
        json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
        print()
        return Rs
    for p, R in zip(a.traces, Rs):
        report(p, R, a.top)
    if len(Rs) == 2:
        compare(Rs[0], Rs[1], a.traces[0], a.traces[1])
    return Rs


if __name__ == '__main__':
    main()
