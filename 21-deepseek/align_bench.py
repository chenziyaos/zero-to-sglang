"""SGL-21：对比两份压测配置，列出会让结论偏向某一方的差别。

四项测法 + 两项配置，取自 LMSYS「More Than DeepInfer」博客（2026-10-24）的复测说明。
配置文件是 JSON，字段见 data/lmsys_sglang.json 与 data/deepinfer_readme.json。

  python3 align_bench.py data/deepinfer_readme.json data/lmsys_sglang.json
"""
import json
import sys

ITEMS = [
    ('prefill_window', 'Prefill 计时窗口',
     '只统计部分 GPU 或开头几批时，不满载区间会混进来'),
    ('prefill_unique_prompts', 'Prefill 输入是否每请求不同',
     '重复同一个 prompt 会让缓存或预取路径受益'),
    ('engram_placement', 'Engram 表放置',
     '主机内存换 KV 容量，显存换查表速度，两边要一致'),
    ('draft_redundant_experts', '草稿模型冗余专家数',
     '冗余专家改变 EP 负载均衡，影响每步时间'),
    ('draft_tokens', '投机草稿 token 数',
     '决定每步接受长度，吞吐要按同一接受长度折算'),
    ('verify', '投机验证方式',
     '静态验证与按置信度调度的验证，每步开销不同'),
]


def diff(a, b):
    rows = []
    for key, name, why in ITEMS:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            rows.append((name, va, vb, '缺失，复测报告里要补上'))
        elif va != vb:
            rows.append((name, va, vb, why))
    return rows


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = (json.load(open(f)) for f in argv)
    rows = diff(a, b)
    la, lb = a.get('name', argv[0]), b.get('name', argv[1])
    if not rows:
        print('六项一致，差距可以归到引擎本身')
        return 0
    for name, va, vb, why in rows:
        print(f'{name}\n  {la}: {va}\n  {lb}: {vb}\n  影响：{why}')
    print(f'\n{len(rows)} 项未对齐，先对齐再比倍数')
    return 1


if __name__ == '__main__':
    sys.exit(main())
