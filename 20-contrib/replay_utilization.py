"""把 _calculate_utilization 从源码里抠出来，在假的 scheduler 上对比修前修后。

不需要 torch，也不需要装 sglang：用 ast 从文件里取出这一个方法，换上最小的桩对象执行。
修前取 git show <rev>:<path>，修后取工作区里的文件（先 git apply upstream/utilization_fix.patch）。

    python3 replay_utilization.py ../../sglang            # 修前 8fabf5e，修后 = 工作区
    python3 replay_utilization.py ../../sglang --rev HEAD
"""
import argparse
import ast
import enum
import subprocess
import sys
import textwrap
import unicodedata
from types import SimpleNamespace

PATH = 'python/sglang/srt/managers/scheduler_components/metrics_reporter.py'


class DisaggregationMode(enum.Enum):
    NULL = 'null'
    PREFILL = 'prefill'
    DECODE = 'decode'


def extract(src, name='_calculate_utilization'):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {'DisaggregationMode': DisaggregationMode}
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
            return ns[name]
    raise SystemExit(f'{name} not found')


def run(fn, mode=DisaggregationMode.NULL, max_running=64, slo=None, running=0, token_usage=0.0):
    sch = SimpleNamespace(disaggregation_mode=mode, max_running_requests=max_running)
    if slo is not None:
        sch.max_running_requests_under_SLO = slo
    stats = SimpleNamespace(utilization=0.0, token_usage=token_usage,
                            num_running_reqs=SimpleNamespace(total=running))
    fn(SimpleNamespace(scheduler=sch, stats=stats))
    return round(stats.utilization, 4)


CASES = [
    ('48 / 64 在跑，KV 6%', dict(running=48, token_usage=0.06)),
    ('8 在跑，KV 72%', dict(running=8, token_usage=0.72)),
    ('空闲', dict()),
    ('设了 SLO 上限 16，8 在跑', dict(slo=16, running=8)),
    ('SLO 上限为 0，16 在跑', dict(slo=0, running=16)),
    ('decode 实例，48 在跑', dict(mode=DisaggregationMode.DECODE, running=48)),
    ('prefill 实例', dict(mode=DisaggregationMode.PREFILL, running=48, token_usage=0.5)),
]


def pad(s, w):
    n = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    return s + ' ' * max(0, w - n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('--rev', default='8fabf5e')
    a = ap.parse_args(argv)
    old_src = subprocess.run(['git', '-C', a.repo, 'show', f'{a.rev}:{PATH}'],
                             capture_output=True, text=True, check=True).stdout
    with open(f'{a.repo}/{PATH}') as fh:
        new_src = fh.read()
    if old_src == new_src:
        print('工作区和 --rev 一样，先 git apply upstream/utilization_fix.patch', file=sys.stderr)
    old, new = extract(old_src), extract(new_src)
    print(pad('场景', 28) + pad(a.rev, 10) + '工作区')
    for name, kw in CASES:
        print(pad(name, 28) + pad(str(run(old, **kw)), 10) + str(run(new, **kw)))


if __name__ == '__main__':
    main()
