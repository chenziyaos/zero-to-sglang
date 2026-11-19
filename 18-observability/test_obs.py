"""SGL-18 实验的边界用例：python3 -m unittest test_obs -v"""
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reqstats as RQ  # noqa: E402
import scrape_metrics as SM  # noqa: E402
import trace_summary as TS  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def fx(name):
    return os.path.join(HERE, name)


def read(path, mode='r'):
    with open(path, mode) as f:
        return f.read()


INF = math.inf


# ---------------------------------------------------------------- /metrics 解析
class TestParse(unittest.TestCase):
    def test_labels_and_special_values(self):
        text = '\n'.join([
            '# HELP sglang:x demo',
            '# TYPE sglang:x gauge',
            'sglang:x{model_name="m",pid="1"} 1.5',
            'sglang:x{model_name="m",pid="2"} NaN',
            'sglang:y_bucket{le="+Inf",model_name="m"} 7.0',
            'sglang:z 3 1700000000000',          # 带时间戳的样本
            'garbage line without value',
        ])
        s, types = SM.parse(text)
        self.assertEqual(types['sglang:x'], 'gauge')
        self.assertEqual(s[('sglang:x', frozenset({('model_name', 'm'), ('pid', '1')}))], 1.5)
        self.assertTrue(math.isnan(s[('sglang:x', frozenset({('model_name', 'm'), ('pid', '2')}))]))
        self.assertEqual(s[('sglang:y_bucket', frozenset({('le', '+Inf'), ('model_name', 'm')}))], 7.0)
        self.assertEqual(s[('sglang:z', frozenset())], 3.0)
        self.assertEqual(len(s), 4)

    def test_escaped_quote_in_label(self):
        s, _ = SM.parse('sglang:a{k="say \\"hi\\"",j="x"} 2')
        self.assertIn(('j', 'x'), next(iter(s))[1])

    def test_multi_rank_sum(self):
        text = ('sglang:num_requests_total{tp_rank="0"} 10\n'
                'sglang:num_requests_total{tp_rank="1"} 12\n'
                'sglang:num_requests_total{tp_rank="2"} 0\n')
        s, _ = SM.parse(text)
        self.assertEqual(SM.total(s, 'sglang:num_requests_total'), 22)
        self.assertEqual(SM.total(s, 'sglang:num_requests_total', {'tp_rank': '1'}), 12)
        self.assertIsNone(SM.total(s, 'sglang:nope'))

    def test_by_label_merges_other_labels(self):
        text = ('sglang:s{category="idle",tp_rank="0"} 1\n'
                'sglang:s{category="idle",tp_rank="1"} 2\n'
                'sglang:s{category="run_batch",tp_rank="0"} 5\n')
        s, _ = SM.parse(text)
        self.assertEqual(SM.by_label(s, 'sglang:s', 'category'), {'idle': 3, 'run_batch': 5})

    def test_buckets_sum_across_ranks_sorted(self):
        text = ('h_bucket{le="0.5",r="0"} 1\nh_bucket{le="+Inf",r="0"} 2\n'
                'h_bucket{le="0.5",r="1"} 3\nh_bucket{le="+Inf",r="1"} 3\nh_bucket{le="0.1",r="1"} 0\n')
        s, _ = SM.parse(text)
        self.assertEqual(SM.buckets(s, 'h'), [(0.1, 0.0), (0.5, 4.0), (INF, 5.0)])


# ---------------------------------------------------------------- 分位数
class TestQuantile(unittest.TestCase):
    B = [(0.1, 10.0), (0.5, 30.0), (1.0, 40.0), (INF, 40.0)]

    def test_interpolate_in_bucket(self):
        v, lo, hi = SM.quantile(0.5, self.B)          # rank 20，落在 (0.1, 0.5]，桶内第 10/20
        self.assertAlmostEqual(v, 0.3)
        self.assertEqual((lo, hi), (0.1, 0.5))

    def test_first_bucket_lower_bound_zero(self):
        v, lo, hi = SM.quantile(0.25, self.B)         # rank 10，正好是第一个桶的上沿
        self.assertAlmostEqual(v, 0.1)
        self.assertEqual(lo, 0.0)
        v, _, _ = SM.quantile(0.05, self.B)
        self.assertAlmostEqual(v, 0.02)

    def test_inf_bucket_returns_last_finite(self):
        b = [(0.1, 1.0), (1.0, 2.0), (INF, 10.0)]
        v, lo, hi = SM.quantile(0.99, b)
        self.assertEqual(v, 1.0)
        self.assertEqual(hi, INF)

    def test_empty_and_malformed(self):
        for b in ([], [(0.1, 0.0), (INF, 0.0)], [(0.1, 1.0), (1.0, 2.0)]):   # 空、零样本、缺 +Inf
            self.assertTrue(math.isnan(SM.quantile(0.5, b)[0]))

    def test_flat_bucket_returns_upper_bound(self):
        b = [(0.1, 5.0), (0.2, 5.0), (INF, 5.0)]
        self.assertAlmostEqual(SM.quantile(1.0, b)[0], 0.1)

    def test_delta_window(self):
        b0 = [(0.1, 10.0), (1.0, 10.0), (INF, 10.0)]
        b1 = [(0.1, 10.0), (1.0, 30.0), (INF, 30.0)]
        d = SM.delta_buckets(b1, b0)
        self.assertEqual(d, [(0.1, 0.0), (1.0, 20.0), (INF, 20.0)])
        v, lo, hi = SM.quantile(0.5, d)               # 窗口里的样本全在 (0.1, 1.0]
        self.assertEqual((lo, hi), (0.1, 1.0))
        self.assertAlmostEqual(v, 0.55)


# ---------------------------------------------------------------- 指标诊断（夹具）
class TestScrape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t0 = read(fx('metrics_t0.txt'))
        cls.t1 = read(fx('metrics_t1.txt'))
        cls.cum = SM.analyze(cls.t0)
        cls.win = SM.analyze(cls.t1, cls.t0)

    def codes(self, R):
        return {c for _, c, _ in R['findings']}

    def test_window_from_stage_seconds(self):
        self.assertEqual(self.win['window_mode'], 'delta')
        self.assertAlmostEqual(self.win['window_s'], 45.0, delta=1.0)
        self.assertAlmostEqual(sum(self.win['stage_share'].values()), 1.0, places=6)

    def test_manual_window_overrides(self):
        R = SM.analyze(self.t1, self.t0, window=90.0)
        self.assertEqual(R['window_s'], 90.0)
        self.assertEqual(R['window_source'], '参数给定')

    def test_phase_a_healthy(self):
        c = self.codes(self.cum)
        self.assertIn('GRAPH_OK', c)
        self.assertIn('QUEUE_OK', c)
        self.assertNotIn('EAGER_DECODE', c)

    def test_phase_b_findings(self):
        c = self.codes(self.win)
        for code in ('QUEUE_DOMINANT', 'ITL_SPIKES', 'EAGER_DECODE', 'NO_DEVICE_TIMER', 'UTILIZATION_ZERO'):
            self.assertIn(code, c)
        self.assertGreater(self.win['decode_eager_share'], 0.4)
        self.assertGreater(self.win['hist']['ttft']['p99'], self.cum['hist']['ttft']['p99'] * 5)

    def test_delta_counts_consistent(self):
        n0 = SM.total(SM.parse(self.t0)[0], 'sglang:num_requests_total')
        n1 = SM.total(SM.parse(self.t1)[0], 'sglang:num_requests_total')
        self.assertEqual(self.win['rate']['requests'], n1 - n0)
        self.assertEqual(self.win['hist']['e2e']['count'], n1 - n0)

    def test_device_timer_present_clears_hint(self):
        extra = '\nsglang:forward_execution_seconds_total{model_name="m"} 12.0\n'
        R = SM.analyze(self.t1 + extra, self.t0)
        self.assertNotIn('NO_DEVICE_TIMER', self.codes(R))

    def test_json_clean(self):
        out = io.StringIO()
        with redirect_stdout(out):
            SM.main([fx('metrics_t0.txt'), fx('metrics_t1.txt'), '--json'])
        obj = json.loads(out.getvalue())                 # NaN / Inf 已转成 null
        self.assertEqual(obj['window_mode'], 'delta')

    def test_empty_exposition(self):
        R = quiet(SM.analyze, '')
        self.assertEqual(R['hist'], {})
        self.assertIsNone(R['decode_eager_share'])
        quiet(SM.report, R)


# ---------------------------------------------------------------- trace
def X(name, cat, ts, dur, pid=1, tid=1, args=None):
    e = {'ph': 'X', 'name': name, 'cat': cat, 'ts': ts, 'dur': dur, 'pid': pid, 'tid': tid}
    if args:
        e['args'] = args
    return e


def synth_trace(gap_us, n_steps=4, kernels=20, kdur=2.0, graph=False):
    """每步 kernels 个 kdur µs 的 kernel，步与步之间（或 kernel 之间）留 gap_us 的空。"""
    ev, t, corr = [], 0.0, 0
    for s in range(n_steps):
        name = f'step[DECODE bs=8 g_sq=8 g_sk={800 + s}]'
        start = t
        ev.append(X(name, 'user_annotation', start, 1.0))
        k_t = start + 1.0
        if graph:
            corr += 1
            ev.append(X('cudaGraphLaunch', 'cuda_runtime', start, 1.0, args={'correlation': corr}))
        for i in range(kernels):
            if not graph:
                corr += 1
                ev.append(X('cudaLaunchKernel', 'cuda_runtime', k_t - 1.0, 1.0, args={'correlation': corr}))
            ev.append(X(f'k{i % 3}', 'kernel', k_t, kdur, pid=2, tid=7, args={'correlation': corr}))
            k_t += kdur + (0.0 if graph else gap_us)
        ev.append(X(name, 'gpu_user_annotation', start + 1.0, k_t - start - 1.0, pid=2, tid=7))
        t = k_t + (gap_us if graph else 0.0)
    return ev


class TestTrace(unittest.TestCase):
    def test_parse_step(self):
        d = TS.parse_step('step[MIXED bs=66 c=2 g=64 c_sq=2048 g_sk=65600]')
        self.assertEqual(d['mode'], 'MIXED')
        self.assertEqual(d['g_sk'], 65600)
        self.assertEqual(TS.parse_step('step[TARGET_VERIFY bs=2 g_sq=6]')['g_sq'], 6)
        self.assertIsNone(TS.parse_step('scheduler.run_batch'))
        self.assertIsNone(TS.parse_step('step[UNKNOWN bs=1]'))

    def test_union_len(self):
        self.assertEqual(TS.union_len([]), 0.0)
        self.assertEqual(TS.union_len([(0, 2), (1, 3), (5, 6)]), 4.0)
        self.assertEqual(TS.union_len([(5, 6), (0, 10)]), 10.0)    # 乱序、包含
        self.assertEqual(TS.union_len([(0, 1), (1, 2)]), 2.0)       # 首尾相接

    def test_launch_bound(self):
        R = TS.summarize(synth_trace(gap_us=4.0))
        self.assertEqual(R['verdict'][0], 'LAUNCH_BOUND')
        self.assertGreater(R['gpu_idle'], 0.5)
        self.assertEqual(R['launch_lag_small'], 1.0)

    def test_gpu_bound_with_graph(self):
        R = TS.summarize(synth_trace(gap_us=0.5, graph=True, kdur=10.0))
        self.assertEqual(R['verdict'][0], 'GPU_BOUND')
        self.assertEqual(R['graph_launches'], 4)
        self.assertIsNone(R['launch_lag_p50'])

    def test_graph_with_big_gap_is_cpu_gap(self):
        R = TS.summarize(synth_trace(gap_us=200.0, graph=True))
        self.assertEqual(R['verdict'][0], 'CPU_GAP')

    def test_no_kernels(self):
        R = TS.summarize([X('step[DECODE bs=1]', 'user_annotation', 0, 5)])
        self.assertIn('error', R)
        quiet(TS.report, 'x', R)

    def test_list_form_and_gzip(self):
        ev = synth_trace(gap_us=1.0)
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, 'a.json'), os.path.join(d, 'b.json.gz')
            with open(p1, 'w') as f:
                json.dump(ev, f)                                   # 裸数组
            with gzip.open(p2, 'wt') as f:
                json.dump({'traceEvents': ev}, f)                  # 对象 + gzip
            self.assertEqual(len(TS.load(p1)), len(TS.load(p2)))

    def test_fixture_eager_vs_graph(self):
        Re = TS.summarize(TS.load(fx('trace_eager.json.gz')))
        Rg = TS.summarize(TS.load(fx('trace_graph.json.gz')))
        self.assertEqual(Re['verdict'][0], 'LAUNCH_BOUND')
        self.assertEqual(Rg['verdict'][0], 'GPU_BOUND')
        self.assertGreater(Re['wall_per_step'] / Rg['wall_per_step'], 2.0)
        self.assertAlmostEqual(Re['busy_per_step'] / Rg['busy_per_step'], 1.0, delta=0.05)

    def test_fixture_8b_eager_is_cheap(self):
        # 8B 单卡 bs 206：GPU 前向 18 ms 远长于 510 个 kernel 的发射 3 ms，eager 不吃亏
        Re = TS.summarize(TS.load(fx('trace_8b_eager.json.gz')))
        Rg = TS.summarize(TS.load(fx('trace_8b_graph.json.gz')))
        self.assertEqual(Re['verdict'][0], 'GPU_BOUND')
        self.assertGreaterEqual(Re['gpu_idle'], 0.0)
        self.assertAlmostEqual(Re['wall_per_step'] / Rg['wall_per_step'], 1.0, delta=0.05)


# ---------------------------------------------------------------- ReqTimeStats
TS_LINE = '[2026-10-30 10:00:{s:02d} TP0] '


class TestReqStats(unittest.TestCase):
    def test_unified_line(self):
        line = (TS_LINE.format(s=5) + 'ReqTimeStats(rid=abc, input_len=100, cached_input_len=40, output_len=7, '
                'attempts=0, type=unified): queue_duration=12.50ms, forward_duration=480.00ms, entry_time=1793325604.000')
        reqs, _ = RQ.parse([line])
        r = reqs[0]
        self.assertEqual((r['rid'], r['input_len'], r['cached_input_len'], r['type']), ('abc', 100, 40, 'unified'))
        self.assertEqual(r['queue_duration'], 12.5)
        self.assertEqual(r['forward_duration'], 480.0)
        self.assertAlmostEqual(r['entry_time'], 1793325604.0)

    def test_pd_prefill_line(self):
        line = (TS_LINE.format(s=1) + 'ReqTimeStats(rid=p1, bootstrap_room=77, input_len=4096, cached_input_len=0, '
                'output_len=1, attempts=2, type=prefill): bootstrap_duration=3.10ms, queue_duration=5.00ms, '
                'forward_duration=210.00ms, entry_time=1793325600.500, transfer_speed=41.25 GB/s, transfer_total=512.00 MB')
        r = RQ.parse([line])[0][0]
        self.assertEqual(r['bootstrap_room'], 77)
        self.assertEqual(r['transfer_speed'], 41.25)
        self.assertEqual(r['transfer_total'], 512.0)
        self.assertEqual(r['bootstrap_duration'], 3.1)
        R = quiet(RQ.analyze, [r], [])
        self.assertEqual(R['retried'], 1)
        self.assertIn('PREFILL_RETRY', {c for _, c, _ in R['findings']})

    def test_pd_decode_line(self):
        line = (TS_LINE.format(s=2) + 'ReqTimeStats(rid=d1, bootstrap_room=77, input_len=4096, cached_input_len=0, '
                'output_len=300, attempts=0, type=decode): bootstrap_duration=1.00ms, alloc_wait_duration=0.50ms, '
                'transfer_duration=12.00ms, queue_duration=0.20ms, forward_duration=6000.00ms, entry_time=1793325600.000')
        r = RQ.parse([line])[0][0]
        self.assertEqual(r['type'], 'decode')
        self.assertEqual(r['alloc_wait_duration'], 0.5)
        self.assertEqual(r['transfer_duration'], 12.0)

    def test_unified_attempts_zero_not_counted(self):
        line = (TS_LINE.format(s=5) + 'ReqTimeStats(rid=a, input_len=1, cached_input_len=0, output_len=1, '
                'attempts=3, type=unified): queue_duration=1.00ms, forward_duration=1.00ms, entry_time=1793325604.998')
        R = quiet(RQ.analyze, *RQ.parse([line]))
        self.assertEqual(R['retried'], 0)

    def test_tz_offset(self):
        # 日志时钟比 entry_time 快 8 小时（UTC+8 的 asctime），且日志只精确到秒
        base = RQ.parse_ts('2026-10-30 10:00:00')
        reqs = [{'log_ts': base + 8 * 3600 + i, 'entry_time': base + i - 0.7, 'queue_duration': 100.0,
                 'forward_duration': 300.0} for i in range(9)]
        self.assertEqual(RQ.tz_offset(reqs), 8 * 3600)
        reqs = [{'log_ts': base - 5.5 * 3600, 'entry_time': base - 1.0}]          # UTC-5:30
        self.assertEqual(RQ.tz_offset(reqs), -5.5 * 3600)
        self.assertEqual(RQ.tz_offset([{'log_ts': base}]), 0)

    def test_state_at_picks_latest_before(self):
        b = [{'ts': 10, 'kind': 'Decode', 'n': 1}, {'ts': 12, 'kind': 'Prefill', 'n': 2},
             {'ts': 15, 'kind': 'Decode', 'n': 3}, {'ts': 30, 'kind': 'Decode', 'n': 4}]
        pre, dec = RQ.state_at(b, 20)
        self.assertEqual((pre['n'], dec['n']), (2, 3))
        self.assertEqual(RQ.state_at(b, 5), (None, None))

    def test_batch_line_fields(self):
        line = (TS_LINE.format(s=3) + 'Decode batch, #running-req: 206, #token: 340000, token usage: 0.94, '
                'cuda graph: False, gen throughput (token/s): 2100.50, #queue-req: 15')
        _, b = RQ.parse([line])
        self.assertEqual(b[0]['#running-req'], 206)
        self.assertEqual(b[0]['token usage'], 0.94)
        self.assertIs(b[0]['cuda graph'], False)
        self.assertEqual(b[0]['gen throughput (token/s)'], 2100.5)

    def test_missing_reqtimestats(self):
        out = io.StringIO()
        with redirect_stdout(out):
            R = RQ.analyze([], [])
            RQ.report(R)
        self.assertEqual(R['n'], 0)
        self.assertIn('--enable-request-time-stats-logging', out.getvalue())

    def test_fixture_story(self):
        with open(fx('server_reqstats.log')) as f:
            R = RQ.analyze(*RQ.parse(f), top=3, bin_s=10)
        self.assertEqual(R['tz_offset_s'], 0)
        codes = {c for _, c, _ in R['findings']}
        self.assertIn('SLOW_BY_FORWARD', codes)
        self.assertIn('EAGER_WINDOW', codes)
        top = R['top'][0]
        self.assertAlmostEqual(top['q'] + top['f'], top['total'])
        self.assertIs(top['decode_line']['cuda graph'], False)


# ---------------------------------------------------------------- 算账：容量办法
class TestWhatIf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import whatif as W
        cls.W = W
        cls.out = W.scenarios()

    def test_capacity_fixes_queue(self):
        b, r2, lim = self.out['base'], self.out['replica2'], self.out['limit10']
        self.assertGreater(b['ttft_p99'], 1.0)
        self.assertLess(r2['ttft_p99'], b['ttft_p99'] / 5)
        self.assertLess(r2['bs_max'], self.W.MF.GRAPH_MAX_BS)       # 加副本以后 batch 回到图上限以内
        self.assertGreater(lim['rejected'], 0.1)

    def test_mixed_and_chunk_cut_itl_tail(self):
        b, m, c = self.out['base'], self.out['mixed'], self.out['chunk2k']
        self.assertLess(m['itl_max'], b['itl_max'] / 3)
        self.assertLess(c['itl_p99'], m['itl_p99'])

    def test_report_runs(self):
        quiet(self.W.report, self.out)


# ---------------------------------------------------------------- 夹具可复现
class TestFixtures(unittest.TestCase):
    def test_deterministic(self):
        import make_fixtures as MF
        names = ['metrics_t0.txt', 'metrics_t1.txt', 'server_reqstats.log', 'trace_eager.json.gz',
                 'trace_graph.json.gz', 'trace_8b_eager.json.gz', 'trace_8b_graph.json.gz']
        before = {n: hashlib.sha256(read(fx(n), 'rb')).hexdigest() for n in names}
        with tempfile.TemporaryDirectory() as d:
            old = MF.HERE
            MF.HERE = d
            try:
                quiet(MF.main)
            finally:
                MF.HERE = old
            after = {n: hashlib.sha256(read(os.path.join(d, n), 'rb')).hexdigest() for n in names}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
