"""提 PR 之前的自查：标题、测试路径、要谁批、改动大小。

规则都抄自 sgl-project/sglang 主干 8fabf5e：
  标题      .github/workflows/pr-title.yml:29-42（用例 .github/scripts/pr-title.test.cjs）
  谁来批    .github/CODEOWNERS（后出现的匹配覆盖先出现的）、.github/MAINTAINER.md 的 Merge Oncall 表
  测试位置  test/registered/unit/README.md:9-13 的镜像规则
  大小      Google eng-practices「Small CLs」：100 行左右合适，1000 行通常太大

用法：
  python3 check_pr.py title '[Fix] observability: Fall back to max_running_requests'
  python3 check_pr.py plan --title '...' --numstat data/numstat_utilization_fix.txt [--repo ../../sglang]
  python3 check_pr.py case          # 回放 issue #22713 的六个 PR
  python3 check_pr.py stats         # 300 个最近关闭的 PR 的合并统计
只用标准库。
"""
import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# ---------------------------------------------------------------- 标题
TYPES = ['Feat', 'Fix', 'Refactor', 'Perf', 'Doc', 'Test', 'CI', 'Chore']
BRANDS = ['Intel', 'Apple', 'Huawei', 'Ascend']
# JS 里没有 m 标志的 $ 只匹配串尾；Python 的 $ 还会放过末尾一个换行，所以这里用 \Z
TITLE_RE = re.compile(
    r'^\[(' + '|'.join(TYPES) + r')\] (?:\[([A-Za-z0-9][A-Za-z0-9._+-]*)\] )?'
    r'[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*: \S(?:[^\r\n]*\S)?\Z')
REVERT_RE = re.compile(r'^Revert "\S(?:[^\r\n]*\S)?"\Z')


def title_ok(title):
    m = TITLE_RE.match(title)
    hw = m.group(2) if m else None
    brand = next((b for b in BRANDS if hw and b.lower() == hw.lower()), None)
    hw_ok = (not hw) or hw == (brand or hw.upper())
    is_revert = bool(REVERT_RE.match(title))
    if ((not (m and hw_ok)) and not is_revert) or title != title.strip():
        return False
    return True


def title_why(title):
    """校验失败时给一句人话；只做诊断，判定以 title_ok 为准。"""
    if title_ok(title):
        return 'ok'
    if title != title.strip():
        return '首尾有空白或换行'
    if '\n' in title or '\r' in title:
        return '标题里有换行'
    if title.lower().startswith('revert'):
        return 'Revert 要用 GitHub 默认格式：Revert "原标题"'
    m = re.match(r'^\[([^\]]*)\]', title)
    if not m:
        return '缺 [Type]，开头要是 [Feat] [Fix] [Refactor] [Perf] [Doc] [Test] [CI] [Chore] 之一'
    t = m.group(1)
    if t not in TYPES:
        hint = next((x for x in TYPES if x.lower() == t.lower()), None)
        return f'类型 {t} 不在列表里' + (f'，大小写要写成 {hint}' if hint else '')
    rest = title[m.end():]
    if not rest.startswith(' ') or rest.startswith('  '):
        return '[Type] 后面要正好一个空格'
    rest = rest[1:]
    h = re.match(r'^\[([^\]]*)\]', rest)
    if h:
        hw = h.group(1)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+-]*', hw):
            return f'硬件标签 [{hw}] 不能为空、不能带空格'
        brand = next((b for b in BRANDS if b.lower() == hw.lower()), None)
        want = brand or hw.upper()
        if hw != want:
            return f'硬件标签写成 [{want}]（只有 Intel、Apple、Huawei、Ascend 用首字母大写，其余全大写）'
        rest = rest[h.end():]
        if rest.startswith('['):
            return '硬件标签只能有一个，且标签之间要有空格'
        if not rest.startswith(' ') or rest.startswith('  '):
            return '硬件标签后面要正好一个空格'
        rest = rest[1:]
    if ':' not in rest:
        return '缺 scope，要写成「scope: 描述」，例如 attention: Fix ...'
    scope, _, desc = rest.partition(':')
    if not scope:
        return 'scope 不能为空'
    if not re.fullmatch(r'[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*', scope):
        return f'scope「{scope}」只能是字母数字加 . _ / -，不能有空格'
    if not desc.startswith(' ') or desc.startswith('  ') or desc.startswith(' \t'):
        return '冒号后面要正好一个空格'
    if not desc.strip():
        return '描述不能为空'
    return '格式不符'


# ---------------------------------------------------------------- 测试镜像路径
SRT = 'python/sglang/srt/'


def mirror_test(path):
    """srt/X/Y.py -> test/registered/unit/X/test_Y.py；不在 srt 下返回 None。"""
    if not path.startswith(SRT) or not path.endswith('.py'):
        return None
    rel = path[len(SRT):]
    d, f = os.path.split(rel)
    if f == '__init__.py':
        return None
    return os.path.join('test/registered/unit', d, 'test_' + f)


def is_test(path):
    return path.startswith('test/') or os.path.basename(path).startswith('test_')


# ---------------------------------------------------------------- CODEOWNERS
def _pat_regex(pat):
    """GitHub CODEOWNERS 用 gitignore 语义：/ 开头锚定根，否则任意层；
    不带通配的路径同时匹配它下面的所有文件；* 不跨目录，** 跨目录。"""
    anchored = pat.startswith('/')
    p = pat.strip('/')
    out, i = '', 0
    while i < len(p):
        if p.startswith('**/', i):
            out += '(?:.*/)?'
            i += 3
        elif p.startswith('**', i):
            out += '.*'
            i += 2
        elif p[i] == '*':
            out += '[^/]*'
            i += 1
        elif p[i] == '?':
            out += '[^/]'
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    head = '^' if anchored else '^(?:.*/)?'
    return re.compile(head + out + '(?:/.*)?$')


def load_codeowners(path=None):
    path = path or os.path.join(DATA, 'CODEOWNERS.8fabf5e')
    rules = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            rules.append((n, parts[0], _pat_regex(parts[0]), parts[1:]))
    return rules


def owners_of(path, rules):
    """返回 (行号, 模式, owners)；后出现的匹配覆盖先出现的，没匹配返回 None。"""
    hit = None
    for n, pat, rx, owners in rules:
        if rx.match(path):
            hit = (n, pat, owners)
    return hit


# MAINTAINER.md 的 Merge Oncall 表（按 related files 前缀，取最长匹配）
ONCALL = [
    ('Scheduler', ['merrymercy', 'hnyls2002', 'cctry'],
     ['python/sglang/srt/managers', 'python/sglang/srt/model_executor']),
    ('Diffusion', ['mickqian', 'BBuf'], ['python/sglang/multimodal_gen']),
    ('PD disaggregation', ['ByronHsu', 'cctry', 'ShangmingCai'], ['python/sglang/srt/disaggregation']),
    ('KV Cache', ['ispobock', 'xiezhq-hermann'], ['python/sglang/srt/mem_cache']),
    ('Parallelism', ['ch-wan', 'fzyzcjy'],
     ['python/sglang/srt/eplb', 'python/sglang/srt/distributed', 'python/sglang/srt/layers/dp_attention.py']),
    ('Kernel', ['BBuf'], ['python/sglang/kernels', 'sgl-kernel']),
    ('Speculative decoding', ['hnyls2002', 'Qiaolin-Yu'], ['python/sglang/srt/speculative']),
    ('NV and model-specific', ['Fridge003', 'ishandhanani', 'Qiaolin-Yu'],
     ['python/sglang/srt/models', 'python/sglang/srt/layers/attention']),
    ('NPU', ['iforgetmyname'], ['python/sglang/srt/hardware_backend/npu']),
    ('CI, Release, Package', ['Kangyan-Zhou', 'Fridge003'], ['.github/workflows']),
    ('Router, API', ['slin1237'],
     ['sgl-model-gateway', 'python/sglang/srt/grpc', 'python/sglang/srt/entrypoints']),
]


def oncall_of(path):
    best = None
    for area, people, prefixes in ONCALL:
        for pre in prefixes:
            if path == pre or path.startswith(pre.rstrip('/') + '/'):
                if best is None or len(pre) > best[2]:
                    best = (area, people, len(pre))
    return (best[0], best[1]) if best else None


# ---------------------------------------------------------------- plan：一份改动的体检
def read_numstat(path):
    """读 git diff --numstat 的输出；二进制文件的 - 记作 0。"""
    if path == '-':
        lines = sys.stdin.read().splitlines()
    else:
        with open(path) as fh:
            lines = fh.read().splitlines()
    rows = []
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        a, d, f = line.split('\t')
        rows.append((int(a) if a != '-' else 0, int(d) if d != '-' else 0, f))
    return rows


def top_dir(path):
    """按「改动属于哪块」粗分：srt 下取前两级，其余取第一级。"""
    if path.startswith(SRT):
        return SRT + path[len(SRT):].split('/')[0]
    if path.startswith('test/registered/unit/'):
        return None  # 测试跟着源码走，不单独算一块
    return path.split('/')[0]


def plan(title, rows, repo=None, rules=None, quiet=False):
    rules = rules or load_codeowners()
    out = []
    add = sum(a for a, _, _ in rows)
    dele = sum(d for _, d, _ in rows)
    src = [f for _, _, f in rows if not is_test(f)]
    tests = [f for _, _, f in rows if is_test(f)]
    issues = []

    ok = title_ok(title) if title else None
    out.append(f'标题  {title!r}')
    out.append(f'      {"通过" if ok else "不通过：" + title_why(title)}' if title else '      （未给标题）')
    if title and not ok:
        issues.append('标题不过 pr-title 检查')

    out.append(f'规模  {len(rows)} 个文件，+{add} / -{dele}，源码 {len(src)} 个、测试 {len(tests)} 个')
    total = add + dele
    if total > 1000:
        issues.append(f'改动 {total} 行，超过 1000 行，拆')
    elif total > 400:
        issues.append(f'改动 {total} 行，偏大，看能否拆成几个可以单独合的 PR')

    areas = sorted({top_dir(f) for f in src if top_dir(f)})
    out.append(f'范围  {len(areas)} 块：' + '、'.join(a.replace(SRT, 'srt/') for a in areas))
    if len(areas) >= 3:
        issues.append(f'一个 PR 动了 {len(areas)} 块代码，审的人要在几块之间切换，先确认它们是一件事')

    groups = {}
    out.append('谁来批')
    for f in src + tests:
        hit = owners_of(f, rules)
        oc = oncall_of(f)
        if hit:
            key = hit[1]
            groups.setdefault(key, hit[2])
            who = f'CODEOWNERS:{hit[0]} {hit[1]} -> ' + ' '.join('@' + o.lstrip('@') for o in hit[2][:4]) + (' …' if len(hit[2]) > 4 else '')
        else:
            who = '无 CODEOWNERS 保护'
        out.append(f'      {f}')
        out.append(f'        {who}')
        if oc:
            out.append(f'        Merge Oncall（{oc[0]}）：' + ' '.join('@' + p for p in oc[1]))
    out.append(f'      需要 {len(groups)} 组 CODEOWNERS 各至少一人批准')

    out.append('测试')
    need = []
    for f in src:
        t = mirror_test(f)
        if not t:
            continue
        has = t in tests
        exists = None
        if repo:
            exists = os.path.exists(os.path.join(repo, t))
        tag = '本 PR 里有' if has else ('仓库里已有' if exists else ('仓库里没有' if exists is False else '本 PR 里没有'))
        out.append(f'      {f.replace(SRT, "srt/")} -> {t}  [{tag}]')
        if not has:
            need.append(f)
    if src and not tests:
        issues.append('没有带测试；改了 srt 的逻辑就要有一条能在 CPU 上跑的单测，覆盖率门槛是改动行的 60%')

    out.append('结论  ' + ('可以提' if not issues else f'{len(issues)} 处要先处理'))
    for i in issues:
        out.append('      - ' + i)
    if not quiet:
        print('\n'.join(out))
    return issues, out


# ---------------------------------------------------------------- case：issue #22713 的六个 PR
def case():
    with open(os.path.join(DATA, 'case_22713.json')) as fh:
        c = json.load(fh)
    rules = load_codeowners()
    print(f'issue #{c["issue"]["number"]}：{c["issue"]["title"]}')
    print(f'  {c["issue"]["opened"]} 提出，{c["issue"]["closed"]} 因无活动自动关闭；8fabf5e 上 bug 仍在')
    print(f'  标题检查 {c["title_check_since"]} 才上线，「今天的标题」一列按现行规则回看，不是当年被拒的原因\n')
    print(f'  {"PR":<7}{"作者身份":<13}{"开":<7}{"关":<7}{"结果":<10}{"文件":>4}{"行":>6}  {"块":>2}{"owner组":>8}  {"今天的标题":<6}{"测试":<6}')
    for p in c['prs']:
        rows = [(a, d, f) for a, d, f in p['files']]
        issues, _ = plan(p['title'], rows, rules=rules, quiet=True)
        src = [f for _, _, f in rows if not is_test(f)]
        areas = {top_dir(f) for f in src if top_dir(f)}
        groups = {owners_of(f, rules)[1] for _, _, f in rows if owners_of(f, rules)}
        lines = sum(a + d for a, d, _ in rows)
        has_test = any(is_test(f) for _, _, f in rows)
        print(f'  #{p["number"]:<6}{p["assoc"]:<13}{p["opened"]:<7}{p["closed"]:<7}{p["outcome"]:<10}{len(rows):>4}{lines:>6}  '
              f'{len(areas):>2}{len(groups):>8}  {"过" if title_ok(p["title"]) else "不过":<6}{"有" if has_test else "无":<6}')
    print()
    for p in c['prs']:
        print(f'  #{p["number"]} {p["title"]}')
        print(f'      {p["note"]}')
    print('\n按同样的规则体检一份聚焦的修法：')
    fix = c['proposal']
    plan(fix['title'], [tuple(x) for x in fix['files']], rules=rules)


# ---------------------------------------------------------------- stats：合并统计
def _ts(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')


def _pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(path=None):
    path = path or os.path.join(DATA, 'prs_closed_20261025_1030.json')
    with open(path) as fh:
        prs = json.load(fh)
    merged = [p for p in prs if p['m']]
    closed = [p for p in prs if not p['m']]
    print(f'快照：{len(prs)} 个 PR，关闭时间 {min(p["x"] for p in prs)[:10]} 到 {max(p["x"] for p in prs)[:10]}')
    print(f'  合并 {len(merged)}，关闭未合并 {len(closed)}')
    hm = [(_ts(p['m']) - _ts(p['c'])).total_seconds() / 3600 for p in merged]
    hc = [(_ts(p['x']) - _ts(p['c'])).total_seconds() / 3600 for p in closed]
    print(f'  从开到合并（小时）：p25 {_pct(hm, .25):.1f}  中位 {statistics.median(hm):.1f}  p75 {_pct(hm, .75):.1f}  p90 {_pct(hm, .9):.1f}')
    print(f'  从开到关闭未合并（小时）：中位 {statistics.median(hc):.1f}')
    tm = sum(title_ok(p['t']) for p in merged)
    tc = sum(title_ok(p['t']) for p in closed)
    print(f'  标题过 pr-title 检查：合并的 {tm}/{len(merged)}，没合并的 {tc}/{len(closed)}')
    print('  作者身份        合并  未合并  合并率')
    for a in ['COLLABORATOR', 'MEMBER', 'CONTRIBUTOR', 'NONE']:
        m = sum(1 for p in merged if p['a'] == a)
        c = sum(1 for p in closed if p['a'] == a)
        if m + c:
            print(f'  {a:<14}{m:>6}{c:>8}{m / (m + c):>8.0%}')
    types = {}
    for p in merged:
        mm = TITLE_RE.match(p['t'])
        if mm:
            types[mm.group(1)] = types.get(mm.group(1), 0) + 1
    print('  合并 PR 的类型：' + '  '.join(f'{k} {v}' for k, v in sorted(types.items(), key=lambda x: -x[1])))
    bad = [p['t'] for p in closed if not title_ok(p['t'])][:5]
    print('  没合并、标题也不过的例子：')
    for t in bad:
        print(f'    {t!r:<70} {title_why(t)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('title')
    t.add_argument('titles', nargs='+')
    p = sub.add_parser('plan')
    p.add_argument('--title', default='')
    p.add_argument('--numstat', required=True, help='git diff --numstat origin/main...HEAD 的输出，- 表示 stdin')
    p.add_argument('--repo', help='本地 sglang 仓库根，用来查镜像测试是否已存在')
    p.add_argument('--codeowners', help='换成仓库里最新的 .github/CODEOWNERS')
    sub.add_parser('case')
    s = sub.add_parser('stats')
    s.add_argument('--json')
    o = sub.add_parser('owners')
    o.add_argument('paths', nargs='+')
    o.add_argument('--codeowners')
    a = ap.parse_args(argv)
    if a.cmd == 'title':
        bad = 0
        for x in a.titles:
            ok = title_ok(x)
            bad += not ok
            print(f'{"ok  " if ok else "FAIL"} {x!r}' + ('' if ok else f'  <- {title_why(x)}'))
        return 1 if bad else 0
    if a.cmd == 'plan':
        rules = load_codeowners(a.codeowners)
        issues, _ = plan(a.title, read_numstat(a.numstat), a.repo, rules)
        return 1 if issues else 0
    if a.cmd == 'owners':
        rules = load_codeowners(a.codeowners)
        for f in a.paths:
            hit = owners_of(f, rules)
            oc = oncall_of(f)
            print(f)
            print('  CODEOWNERS ' + (f'第 {hit[0]} 行 {hit[1]}: ' + ' '.join(hit[2]) if hit else '无'))
            print('  Merge Oncall ' + (f'{oc[0]}: ' + ' '.join('@' + x for x in oc[1]) if oc else '无'))
        return 0
    if a.cmd == 'case':
        case()
        return 0
    if a.cmd == 'stats':
        stats(a.json)
        return 0


if __name__ == '__main__':
    sys.exit(main())
