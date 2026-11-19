"""20-contrib 的用例：标题规则用 sglang 自己的 pr-title.test.cjs 里的全部例子。"""
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    with open(os.path.join(HERE, 'data', name)) as fh:
        return json.load(fh)


sys.path.insert(0, HERE)
import check_pr as cp  # noqa: E402

# 照抄 .github/scripts/pr-title.test.cjs @ 8fabf5e
VALID = [
    '[Fix] Attention: Fix output',
    '[Feat] Qwen3.5: add model support',
    '[Fix] Flash-Attn: FIX OUTPUT',
    '[Fix] [Apple] MLX: fix output',
    '[Fix] [HYGON] cpu: Fix output',
    '[Fix] [QUALCOMM] cpu: Fix output',
    '[Fix] [Huawei] model/Qwen3: Fix output',
    '[Fix] [Ascend] attention: Fix output',
    '[Perf] [TPU] Attention: optimize decode',
    'Revert "[Fix] attention: Fix output"',
    'Revert "A legacy title without tags or scope"',
    'Revert "Revert "[Fix] attention: Fix output""',
    'Revert "x"',
    *[f'[{t}] attention: Update implementation' for t in cp.TYPES],
    '[Feat] qwen3: Add model support',
    '[Perf] [AMD] flash-attn: Optimize decode performance',
    '[Fix] [Intel] sgl-kernel: Fix output',
    '[Fix] [NVIDIA] model/qwen3: Fix output',
    '[Test] qwen3.5: Add coverage',
    '[CI] pr_title: Check format',
    '[Perf] [MI300X] moe: Optimize dispatch',
    '[Doc] contribution: 文档更新',
    '[Chore] ci: x',
    '[Fix] ci: Handle `quotes`, "text", and $(commands) as data',
]

INVALID = [
    '[Fix] [Hygon] cpu: Fix output',
    '[Fix] [Qualcomm] cpu: Fix output',
    *[f'[Fix] [{h}] attention: Fix output' for h in
      ['amd', 'Amd', 'nvidia', 'Nvidia', 'intel', 'INTEL', 'apple', 'APPLE', 'huawei',
       'HUAWEI', 'ascend', 'ASCEND', 'tpu', 'Tpu', 'mi300x', 'Mi300X']],
    '[Revert] attention: Revert output fix',
    'Revert ""',
    'Revert "   "',
    'Revert " title"',
    'Revert "title "',
    'Revert "title',
    'Revert title',
    'revert "title"',
    'Revert  "title"',
    'Revert "title" extra text',
    'Revert "title"\n',
    'Revert "multi\nline"',
    'Fix KV cache indexing',
    '[Fix] Fix KV cache indexing',
    '[Feature] model: Add support',
    '[fix] attention: Fix output',
    '[Fix][AMD] attention: Fix output',
    '[Fix] [AMD][NVIDIA] attention: Fix output',
    '[Fix] [AMD] [NVIDIA] attention: Fix output',
    '[Fix] [] attention: Fix output',
    '[Fix] [ AMD ] attention: Fix output',
    '[Fix] [AMD GPU] attention: Fix output',
    '[Fix] : Fix output',
    '[Fix] flash attn: Fix output',
    '[Fix] attention : Fix output',
    '[Fix] attention:Fix output',
    '[Fix] attention:  Fix output',
    '[Fix] attention: ',
    '[Fix] attention:    ',
    '[Fix]  attention: Fix output',
    '[Fix]\tattention: Fix output',
    '[Fix] attention:\tFix output',
    ' [Fix] attention: Fix output',
    '[Fix] attention: Fix output ',
    '[Fix] attention: Fix output\n',
    '[Fix] attention: Fix\noutput',
]


class TestTitle(unittest.TestCase):
    def test_official_valid(self):
        for t in VALID:
            with self.subTest(t=t):
                self.assertTrue(cp.title_ok(t))

    def test_official_invalid(self):
        for t in INVALID:
            with self.subTest(t=t):
                self.assertFalse(cp.title_ok(t))

    def test_counts_match_upstream(self):
        # 13 条单列 + 8 种类型展开 + 10 条单列；2 + 16 种硬件写法 + 36 条
        self.assertEqual(len(VALID), 31)
        self.assertEqual(len(INVALID), 54)

    def test_trailing_newline_needs_backslash_z(self):
        # Python 的 $ 会放过末尾一个换行，JS 不会；这条用例防止有人把 \Z 改回 $
        self.assertFalse(cp.title_ok('[Fix] attention: Fix output\n'))

    def test_why_messages(self):
        self.assertIn('Bugfix', cp.title_why('[Bugfix] Fall back to max_running_requests'))
        self.assertIn('Fix', cp.title_why('[fix] attention: x'))
        self.assertIn('scope', cp.title_why('[Fix] Fix KV cache indexing'))
        self.assertIn('AMD', cp.title_why('[Fix] [amd] attention: x'))
        self.assertIn('Huawei', cp.title_why('[Fix] [HUAWEI] attention: x'))
        self.assertIn('空格', cp.title_why('[Fix] flash attn: Fix output'))
        self.assertIn('冒号', cp.title_why('[Fix] attention:Fix output'))
        self.assertIn('Revert', cp.title_why('revert "title"'))
        self.assertIn('首尾', cp.title_why(' [Fix] attention: Fix output'))

    @unittest.skipUnless(subprocess.run(['which', 'node'], capture_output=True).returncode == 0, 'no node')
    def test_agrees_with_upstream_js(self):
        js = r'''
const types = ['Feat','Fix','Refactor','Perf','Doc','Test','CI','Chore'];
const pattern = new RegExp(String.raw`^\[(${types.join('|')})\] (?:\[([A-Za-z0-9][A-Za-z0-9._+-]*)\] )?` +
  String.raw`[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*: \S(?:[^\r\n]*\S)?$`);
const brands = ['Intel','Apple','Huawei','Ascend'];
const titles = JSON.parse(require('fs').readFileSync(0, 'utf8'));
console.log(JSON.stringify(titles.map(t => {
  const m = pattern.exec(t); const hw = m?.[2];
  const b = brands.find(n => n.toLowerCase() === hw?.toLowerCase());
  const okhw = !hw || hw === (b || hw.toUpperCase());
  const rev = /^Revert "\S(?:[^\r\n]*\S)?"$/.test(t);
  return !((!(m && okhw) && !rev) || t !== t.trim());
})));'''
        titles = VALID + INVALID + [p['t'] for p in load_json('prs_closed_20261025_1030.json')]
        r = subprocess.run(['node', '-e', js], input=json.dumps(titles), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(r.stdout), [cp.title_ok(t) for t in titles])


class TestMirror(unittest.TestCase):
    def test_readme_examples(self):
        self.assertEqual(cp.mirror_test('python/sglang/srt/mem_cache/radix_cache.py'),
                         'test/registered/unit/mem_cache/test_radix_cache.py')
        self.assertEqual(cp.mirror_test('python/sglang/srt/sampling/sampling_params.py'),
                         'test/registered/unit/sampling/test_sampling_params.py')

    def test_nested(self):
        self.assertEqual(cp.mirror_test('python/sglang/srt/managers/scheduler_components/metrics_reporter.py'),
                         'test/registered/unit/managers/scheduler_components/test_metrics_reporter.py')

    def test_outside_srt(self):
        self.assertIsNone(cp.mirror_test('sgl-model-gateway/src/main.rs'))
        self.assertIsNone(cp.mirror_test('python/sglang/srt/layers/__init__.py'))


class TestCodeowners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = cp.load_codeowners()

    def owner(self, path):
        hit = cp.owners_of(path, self.rules)
        return hit and hit[1]

    def test_last_match_wins_over_parent(self):
        self.assertEqual(self.owner('python/sglang/srt/managers/scheduler_pp_mixin.py'),
                         '/python/sglang/srt/managers/scheduler_pp_mixin.py')
        self.assertEqual(self.owner('python/sglang/srt/managers/scheduler.py'),
                         '/python/sglang/srt/managers')

    def test_duplicate_pattern_takes_later_line(self):
        hit = cp.owners_of('docs/docs/index.mdx', self.rules)
        self.assertEqual(hit[1], '/docs')
        self.assertIn('@JustinTong0323', hit[2])

    def test_single_star_does_not_cross_dirs(self):
        self.assertEqual(self.owner('python/sglang/srt/mem_cache/embedding_pool.py'),
                         '/python/sglang/srt/mem_cache/embedding_*.py')
        self.assertEqual(self.owner('python/sglang/srt/mem_cache/sub/embedding_pool.py'),
                         '/python/sglang/srt/mem_cache')

    def test_double_star(self):
        self.assertEqual(self.owner('python/sglang/kernels/jit/csrc/paged_experts.cu'),
                         '/python/sglang/kernels/**/paged_experts.*')

    def test_prefix_is_not_substring(self):
        self.assertEqual(self.owner('python/sglang/srt/managers_extra/x.py'), None)

    def test_unit_tests_unprotected(self):
        self.assertIsNone(self.owner('test/registered/unit/managers/test_x.py'))

    def test_oncall_longest_prefix(self):
        self.assertEqual(cp.oncall_of('python/sglang/srt/layers/attention/flashinfer_backend.py')[0],
                         'NV and model-specific')
        self.assertEqual(cp.oncall_of('python/sglang/srt/layers/dp_attention.py')[0], 'Parallelism')
        self.assertIsNone(cp.oncall_of('python/sglang/srt/layers/linear.py'))


class TestPlan(unittest.TestCase):
    def run_plan(self, title, rows):
        with redirect_stdout(io.StringIO()):
            issues, out = cp.plan(title, rows)
        return issues, out

    def test_focused_fix_passes(self):
        rows = cp.read_numstat(os.path.join(HERE, 'data/numstat_utilization_fix.txt'))
        issues, _ = self.run_plan('[Fix] observability: Fall back to max_running_requests for sglang:utilization', rows)
        self.assertEqual(issues, [])

    def test_no_test_flagged(self):
        issues, _ = self.run_plan('[Fix] observability: x',
                                  [(11, 3, 'python/sglang/srt/managers/scheduler_components/metrics_reporter.py')])
        self.assertTrue(any('测试' in i for i in issues))

    def test_mixed_areas_flagged(self):
        case = load_json('case_22713.json')
        p = next(x for x in case['prs'] if x['number'] == 23212)
        issues, _ = self.run_plan('[Fix] observability: x', [tuple(f) for f in p['files']])
        self.assertTrue(any('3 块' in i for i in issues))

    def test_size_thresholds(self):
        big = [(900, 200, 'python/sglang/srt/layers/linear.py'), (10, 0, 'test/registered/unit/layers/test_linear.py')]
        issues, _ = self.run_plan('[Refactor] layers: x', big)
        self.assertTrue(any('1000' in i for i in issues))


class TestStats(unittest.TestCase):
    def test_snapshot_numbers(self):
        f = io.StringIO()
        with redirect_stdout(f):
            cp.stats()
        s = f.getvalue()
        self.assertIn('合并 242，关闭未合并 58', s)
        self.assertIn('合并的 242/242', s)
        self.assertIn('中位 10.2', s)


class TestReplay(unittest.TestCase):
    """用补丁里的前后两段源码回放，不依赖 sglang 克隆。"""

    @classmethod
    def setUpClass(cls):
        import replay_utilization as ru
        cls.ru = ru
        with open(os.path.join(HERE, 'upstream/utilization_fix.patch')) as fh:
            hunk = fh.read().split('@@', 2)[2].split('diff --git')[0]
        body = hunk.split('\n', 1)[1]
        old = [ln[1:] for ln in body.splitlines() if ln[:1] in (' ', '-')]
        new = [ln[1:] for ln in body.splitlines() if ln[:1] in (' ', '+')]
        head = ['class R:', '    def _calculate_utilization(self):']
        cls.old = staticmethod(ru.extract('\n'.join(head + old)))
        cls.new = staticmethod(ru.extract('\n'.join(head + new)))

    def test_before_fix_always_zero_without_slo(self):
        for kw in [dict(running=48, token_usage=0.06), dict(running=8, token_usage=0.72)]:
            self.assertEqual(self.ru.run(self.old, **kw), 0.0)

    def test_after_fix_tracks_load(self):
        ru = self.ru
        self.assertEqual(ru.run(self.new, running=48, token_usage=0.06), 0.75)
        self.assertEqual(ru.run(self.new, running=8, token_usage=0.72), 0.8)
        self.assertEqual(ru.run(self.new, slo=16, running=8), 0.5)
        self.assertEqual(ru.run(self.new, mode=ru.DisaggregationMode.PREFILL, running=48), -1)


if __name__ == '__main__':
    unittest.main()
