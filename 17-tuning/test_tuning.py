"""SGL-17 实验的边界用例：python3 -m unittest test_tuning -v"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import parse_log as P  # noqa: E402
import tune_sim as T  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ---------------------------------------------------------------- 合成日志
def ts(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f'[2026-10-28 {10 + h:02d}:{m:02d}:{s:02d} TP0] '


def args_line(**kw):
    d = dict(mem_fraction_static=0.83, max_running_requests=None, chunked_prefill_size=8192,
             schedule_policy='fcfs', schedule_conservativeness=1.0, cuda_graph_max_bs_decode=256)
    d.update(kw)
    return f'[2026-10-28 09:59:00] server_args={d!r}'


def startup(total=365312, chunk=8192, running=4096, avail=7.0):
    return (ts(0) + f'max_total_num_tokens={total}, chunked_prefill_size={chunk}, max_prefill_tokens=16384, '
            f'max_running_requests={running}, context_len=40960, available_gpu_mem={avail:.2f} GB')


def decode(t, run, use, q, graph=True, tput=10000.0, extra=''):
    return (ts(t) + f'Decode batch, #running-req: {run}, #token: {int(use * 365312)}, token usage: {use:.2f}, '
            f'{extra}cuda graph: {graph}, gen throughput (token/s): {tput:.2f}, #queue-req: {q}')


def retract(t, n=3, old=0.1, new=0.35):
    return (ts(t) + f'KV cache pool is full. Retract requests. #retracted_reqs: {n}, #new_tokens_gained: {n * 800}, '
            f'#new_token_ratio: {old:.4f} -> {new:.4f}')


def steady(n=20, run=200, use=0.92, q=300, graph=True, step=5, **kw):
    return [decode(10 + i * step, run, use, q, graph, **kw) for i in range(n)]


def codes(lines, **kw):
    return [f['code'] for f in P.diagnose(P.parse(lines), **kw)]


def find(lines, code):
    return next(f for f in P.diagnose(P.parse(lines)) if f['code'] == code)


# ---------------------------------------------------------------- 正则与字段
class TestRegex(unittest.TestCase):
    def test_ts_real(self):
        t, body = P.parse_ts('[2026-10-28 15:30:01 TP0] Decode batch, x: 1')
        self.assertIsNotNone(t)
        self.assertTrue(body.startswith('Decode batch'))

    def test_ts_real_with_dp_tag(self):
        t, body = P.parse_ts('[2026-10-28 15:30:01 DP1 TP3] hello')
        self.assertEqual(body, 'hello')

    def test_ts_sim(self):
        t, body = P.parse_ts('[t=  12.500s] Prefill batch, a: 1')
        self.assertAlmostEqual(t, 12.5)
        self.assertTrue(body.startswith('Prefill'))

    def test_ts_none(self):
        t, body = P.parse_ts('INFO:     Uvicorn running')
        self.assertIsNone(t)

    def test_startup_gpu(self):
        m = P.STARTUP.search(startup(avail=12.17))
        self.assertEqual(m.group(1), '365312')
        self.assertEqual(m.group(6), '12.17')

    def test_startup_cpu(self):
        line = startup().replace('available_gpu_mem', 'available_cpu_mem')
        self.assertIsNotNone(P.STARTUP.search(line))

    def test_startup_chunk_disabled(self):
        self.assertIsNotNone(P.STARTUP.search(startup(chunk=-1)))

    def test_batch_with_iter(self):
        m = P.BATCH.search('Decode batch [1234], #running-req: 3, token usage: 0.10, cuda graph: True')
        self.assertEqual(m.group(1), 'Decode')

    def test_retract_with_mamba(self):
        line = ('KV cache pool is full. Retract requests. #retracted_reqs: 2, #new_tokens_gained: 1500, '
                '#mamba_num_gained: 2, #new_token_ratio: 0.1000 -> 0.3500')
        m = P.RETRACT.search(line)
        self.assertEqual(m.group(4), '0.3500')

    def test_testing_retraction_ignored(self):
        line = 'Testing retraction. #retracted_reqs: 1, #new_tokens_gained: 10, #new_token_ratio: 0.1 -> 0.2'
        self.assertIsNone(P.RETRACT.search(line))

    def test_decode_capture(self):
        line = ('Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16, 24, ..., 256], elapsed=7.86 s, '
                'mem usage=0.63 GB, avail mem=12.17 GB.')
        R = P.parse([line])
        self.assertEqual(R['graph_max_bs'], 256)

    def test_prefill_capture_not_decode(self):
        line = 'Capture target prefill CUDA graph end. elapsed=12.04 s, mem usage=0.41 GB, avail mem=13.1 GB'
        self.assertIsNone(P.parse([line])['graph_max_bs'])

    def test_oom(self):
        R = P.parse(['Prefill out of memory. Try to lower your batch size.', 'Decode out of memory. Try to lower your batch size.'])
        self.assertEqual(R['oom'], ['Prefill', 'Decode'])

    def test_reduced(self):
        R = P.parse(['max_running_requests was reduced from the requested 4096 to 3000 due to KV'])
        self.assertEqual(R['reduced'], (4096, 3000))

    def test_capping(self):
        R = P.parse(['Capping max_new_tokens of request abc to SGLANG_MAX_NEW_TOKENS_LIMIT=8192 (requested: 32768)'])
        self.assertEqual(R['capping'], [('abc', 8192, '32768')])

    def test_kv_budget(self):
        R = P.parse(['Request prompt exceeds the KV memory budget: input_len=50000, budget=40000'])
        self.assertEqual(R['kv_budget'], [50000])

    def test_abort(self):
        R = P.parse(['retract_decode: aborted last request r9 due to OOM',
                     'retract_decode: aborted request r3: no KV backup'])
        self.assertEqual(R['abort_last'], ['r9'])
        self.assertEqual(R['abort_req'][0][0], 'r3')

    def test_comment_and_blank_skipped(self):
        R = P.parse(['# Decode batch, #running-req: 5', '', '   '])
        self.assertEqual(R['decode'], [])


class TestFields(unittest.TestCase):
    def test_types(self):
        f = P.parse_fields('#running-req: 12, token usage: 0.45, cuda graph: True, gen throughput (token/s): 1.50')
        self.assertEqual(f['#running-req'], 12)
        self.assertAlmostEqual(f['token usage'], 0.45)
        self.assertIs(f['cuda graph'], True)
        self.assertAlmostEqual(f['gen throughput (token/s)'], 1.5)

    def test_false_bool(self):
        self.assertIs(P.parse_fields('cuda graph: False')['cuda graph'], False)

    def test_string_value(self):
        self.assertEqual(P.parse_fields('mode: abc')['mode'], 'abc')

    def test_part_without_colon_skipped(self):
        self.assertEqual(P.parse_fields('junk, a: 1'), {'a': 1})

    def test_usage_plain(self):
        self.assertEqual(P.usage_of({'token usage': 0.5}), (0.5, None, None))

    def test_usage_swa(self):
        f = P.parse_fields('#full token: 10, full token usage: 0.40, #swa token: 9, swa token usage: 0.95')
        self.assertEqual(P.usage_of(f), (0.40, 0.95, None))

    def test_usage_mamba(self):
        f = P.parse_fields('token usage: 0.30, mamba num: 5, mamba usage: 0.80')
        self.assertEqual(P.usage_of(f)[2], 0.80)

    def test_graph_field_variants(self):
        self.assertIs(P.graph_of({'cuda graph': False}), False)
        self.assertIs(P.graph_of({'npu graph': True}), True)
        self.assertIsNone(P.graph_of({'token usage': 0.1}))


class TestServerArgs(unittest.TestCase):
    def test_dict_repr(self):
        a = P.parse_server_args(args_line(mem_fraction_static=0.88, schedule_policy='lpm'))
        self.assertEqual(a['mem_fraction_static'], 0.88)
        self.assertEqual(a['schedule_policy'], 'lpm')
        self.assertIsNone(a['max_running_requests'])

    def test_dict_drops_unknown_keys(self):
        a = P.parse_server_args("server_args={'api_key': 'x', 'page_size': 16}")
        self.assertEqual(a, {'page_size': 16})

    def test_dict_fallback_when_not_literal(self):
        a = P.parse_server_args("server_args={'mem_fraction_static': 0.9, 'x': <object at 0x1>}")
        self.assertEqual(a['mem_fraction_static'], 0.9)

    def test_old_servergs_repr(self):
        line = ("server_args=ServerArgs(model_path='Qwen/Qwen3-8B', mem_fraction_static=0.85, "
                "schedule_policy='fcfs', cuda_graph_max_bs_decode=None, tp_size=2)")
        a = P.parse_server_args(line)
        self.assertEqual(a['model_path'], 'Qwen/Qwen3-8B')
        self.assertEqual(a['mem_fraction_static'], 0.85)
        self.assertIsNone(a['cuda_graph_max_bs_decode'])
        self.assertEqual(a['tp_size'], 2)

    def test_none_when_absent(self):
        self.assertIsNone(P.parse_server_args('hello'))


# ---------------------------------------------------------------- 统计
class TestStats(unittest.TestCase):
    def test_pct(self):
        self.assertEqual(P.pct([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(P.pct([0, 10], 90), 9.0)
        self.assertIsNone(P.pct([], 50))

    def test_idle_rows_excluded(self):
        lines = [decode(0, 0, 0.0, 0)] + steady(4)
        self.assertEqual(P.stats(P.parse(lines))['n'], 4)

    def test_span_and_retract_rate(self):
        lines = steady(13, step=5) + [retract(30)]          # 60 s 内 1 次
        S = P.stats(P.parse(lines))
        self.assertEqual(S['span_s'], 60)
        self.assertAlmostEqual(S['retract_per_min'], 1.0)

    def test_retract_without_timestamps(self):
        R = P.parse(['Decode batch, #running-req: 5, token usage: 0.5, cuda graph: True, #queue-req: 0',
                     'KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 5, '
                     '#new_token_ratio: 0.1000 -> 0.2000'])
        self.assertEqual(P.stats(R)['retract_per_min'], float('inf'))

    def test_graph_split(self):
        lines = [decode(i, 200, 0.9, 10, True) for i in range(6)] + \
                [decode(10 + i, 300 + i, 0.9, 10, False) for i in range(4)]
        S = P.stats(P.parse(lines))
        self.assertAlmostEqual(S['graph_false'], 0.4)
        self.assertEqual(S['graph_true_max'], 200)
        self.assertEqual(S['graph_false_min'], 300)

    def test_loose_share(self):
        lines = steady(5, use=0.5, q=10) + steady(5, use=0.95, q=10)
        self.assertAlmostEqual(P.stats(P.parse(lines))['loose'], 0.5)

    def test_queue_zero_not_loose(self):
        self.assertEqual(P.stats(P.parse(steady(5, use=0.3, q=0)))['loose'], 0)

    def test_at_cap(self):
        lines = [startup(running=200)] + steady(4, run=200)
        self.assertEqual(P.stats(P.parse(lines))['at_cap'], 1.0)


# ---------------------------------------------------------------- 诊断
class TestDiagnose(unittest.TestCase):
    def test_floor2(self):
        self.assertEqual(P.floor2(0.8999999), 0.89)
        self.assertEqual(P.floor2(0.89), 0.89)        # 浮点误差不应把 0.89 压成 0.88

    def test_gpu_total(self):
        R = P.parse([ts(0) + 'Load weight begin. avail mem=78.93 GB'])
        self.assertAlmostEqual(P.gpu_total_gib(R), 79.53)
        self.assertEqual(P.gpu_total_gib(R, 80), 80)
        self.assertIsNone(P.gpu_total_gib(P.parse([])))

    def test_no_decode(self):
        self.assertIn('NO_DECODE', codes([startup()]))

    def test_oom_prefill_steps_chunk(self):
        f = find([startup(chunk=8192), 'Prefill out of memory.'], 'OOM_PREFILL')
        self.assertEqual(f['flags']['--chunked-prefill-size'], 4096)
        f = find([startup(chunk=4096), 'Prefill out of memory.'], 'OOM_PREFILL')
        self.assertEqual(f['flags']['--chunked-prefill-size'], 2048)

    def test_oom_decode(self):
        self.assertIn('OOM_DECODE', codes(['Decode out of memory.']))

    def test_retract_abort(self):
        self.assertIn('RETRACT_ABORT', codes(['retract_decode: aborted last request r1 due to OOM']))

    def test_kv_budget(self):
        f = find([startup(total=30000), 'Request prompt exceeds the KV memory budget: input_len=40000'], 'KV_BUDGET')
        self.assertIn('30000', f['msg'])

    def test_avail_high_target(self):
        lines = [args_line(mem_fraction_static=0.83), ts(0) + 'Load weight begin. avail mem=78.93 GB',
                 startup(avail=12.26)] + steady()
        f = find(lines, 'AVAIL_HIGH')
        self.assertEqual(f['flags']['--mem-fraction-static'], 0.89)

    def test_avail_high_without_frac(self):
        f = find([startup(avail=15)] + steady(), 'AVAIL_HIGH')
        self.assertEqual(f['flags'], {})

    def test_avail_bands(self):
        self.assertIn('AVAIL_SLACK', codes([startup(avail=9.0)] + steady()))
        self.assertIn('AVAIL_OK', codes([startup(avail=6.5)] + steady()))
        f = find([args_line(mem_fraction_static=0.92), startup(avail=4.0)] + steady(), 'AVAIL_LOW')
        self.assertEqual(f['flags']['--mem-fraction-static'], 0.9)

    def test_running_reduced(self):
        self.assertIn('RUNNING_REDUCED', codes(['max_running_requests was reduced from the requested 9 to 4']))

    def test_queue_idle_suppresses_admit(self):
        c = codes(steady(use=0.4, q=0))
        self.assertIn('QUEUE_IDLE', c)
        self.assertNotIn('ADMIT_TIGHT', c)
        self.assertNotIn('USAGE_OK', c)

    def test_queue_deep(self):
        f = find(steady(q=3000), 'QUEUE_DEEP')
        self.assertEqual(f['flags']['--max-queued-requests'], 2000)

    def test_running_cap(self):
        lines = [startup(running=128)] + steady(run=128, use=0.5, q=500)
        f = find(lines, 'RUNNING_CAP')
        self.assertEqual(f['flags']['--max-running-requests'], 256)

    def test_retract_freq(self):
        lines = steady(13) + [retract(20 + i) for i in range(4)]
        f = find(lines, 'RETRACT_FREQ')
        self.assertEqual(f['flags']['--schedule-conservativeness'], 1.3)
        self.assertIn('0.35', f['msg'])

    def test_retract_freq_beyond_13(self):
        lines = [args_line(schedule_conservativeness=1.3)] + steady(13) + [retract(20 + i) for i in range(4)]
        self.assertEqual(find(lines, 'RETRACT_FREQ')['flags']['--schedule-conservativeness'], 1.6)

    def test_admit_tight(self):
        f = find(steady(use=0.6, q=300), 'ADMIT_TIGHT')
        self.assertEqual(f['flags']['--schedule-conservativeness'], 0.3)

    def test_admit_tight_already_low(self):
        f = find([args_line(schedule_conservativeness=0.3)] + steady(use=0.6, q=300), 'ADMIT_TIGHT')
        self.assertEqual(f['flags'], {})

    def test_retract_ok_and_usage_ok(self):
        lines = steady(25) + [retract(30)]                  # 120 s 内 1 次
        c = codes(lines)
        self.assertIn('RETRACT_OK', c)
        self.assertIn('USAGE_OK', c)

    def test_retract_freq_blocks_usage_ok(self):
        lines = steady(13) + [retract(20 + i) for i in range(4)]
        self.assertNotIn('USAGE_OK', codes(lines))

    def test_swa_bind(self):
        extra = 'swa token usage: 0.97, '
        lines = [decode(10 + i, 100, 0.40, 50, extra=extra) for i in range(5)]
        self.assertIn('SWA_BIND', codes(lines))

    def test_full_bind(self):
        extra = 'swa token usage: 0.30, '
        lines = [decode(10 + i, 100, 0.95, 50, extra=extra) for i in range(5)]
        self.assertIn('FULL_BIND', codes(lines))

    def test_graph_fallback_rounds_to_32(self):
        cap = 'Capture target decode CUDA graph end. bs=[1, 2, ..., 256], elapsed=1 s, mem usage=0.6 GB, avail mem=9 GB.'
        lines = [cap] + steady(5, run=200) + [decode(100 + i, 260 + i * 10, 0.92, 100, False) for i in range(5)]
        f = find(lines, 'GRAPH_FALLBACK')
        self.assertEqual(f['flags']['--cuda-graph-max-bs-decode'], 320)
        self.assertIn('128 MB', f['msg'])

    def test_graph_fallback_without_capture_line(self):
        lines = steady(5, run=200) + [decode(100 + i, 300, 0.92, 100, False) for i in range(5)]
        self.assertIn('GRAPH_FALLBACK', codes(lines))

    def test_graph_other(self):
        lines = steady(5, run=200) + [decode(100 + i, 100, 0.92, 100, False) for i in range(5)]
        self.assertIn('GRAPH_OTHER', codes(lines))

    def test_graph_below_share_silent(self):
        lines = steady(9, run=200) + [decode(100, 300, 0.92, 100, False)]
        c = codes(lines)
        self.assertNotIn('GRAPH_FALLBACK', c)
        self.assertNotIn('GRAPH_OTHER', c)

    def test_lpm_fallback(self):
        self.assertIn('LPM_FALLBACK', codes([args_line(schedule_policy='lpm')] + steady(q=500)))
        self.assertNotIn('LPM_FALLBACK', codes([args_line(schedule_policy='lpm')] + steady(q=50)))
        self.assertNotIn('LPM_FALLBACK', codes([args_line(schedule_policy='fcfs')] + steady(q=500)))

    def test_capping(self):
        line = 'Capping max_new_tokens of request a to SGLANG_MAX_NEW_TOKENS_LIMIT=4096 (requested: 9999)'
        self.assertIn('CAPPING', codes([line] + steady()))

    def test_spec_low(self):
        lines = [decode(10 + i, 50, 0.9, 200, extra='accept len: 1.30, ') for i in range(5)]
        self.assertIn('SPEC_LOW', codes(lines))
        lines = [decode(10 + i, 50, 0.9, 200, extra='accept len: 2.80, ') for i in range(5)]
        self.assertNotIn('SPEC_LOW', codes(lines))

    def test_merged_flags_first_wins(self):
        fs = [P.finding('A', 'warn', '', {'--x': 1}), P.finding('B', 'warn', '', {'--x': 2, '--y': 3})]
        self.assertEqual(P.merged_flags(fs), {'--x': 1, '--y': 3})


class TestSamples(unittest.TestCase):
    def load(self, name):
        with open(os.path.join(HERE, name)) as fh:
            return P.parse(fh)

    def test_before(self):
        R = self.load('sample_before.log')
        c = {f['code']: f for f in P.diagnose(R)}
        for k in ('AVAIL_HIGH', 'ADMIT_TIGHT', 'GRAPH_FALLBACK', 'LPM_FALLBACK'):
            self.assertIn(k, c)
        flags = P.merged_flags(P.diagnose(R))
        self.assertEqual(flags['--mem-fraction-static'], 0.89)
        self.assertEqual(flags['--schedule-conservativeness'], 0.3)

    def test_after(self):
        fs = P.diagnose(self.load('sample_after.log'))
        self.assertEqual({f['level'] for f in fs}, {'ok'})
        self.assertEqual(P.merged_flags(fs), {})

    def test_report_runs(self):
        R = self.load('sample_before.log')
        out = quiet(P.report, R, P.stats(R), P.diagnose(R))
        self.assertIsNone(out)


# ---------------------------------------------------------------- tune_sim
class TestTuneSim(unittest.TestCase):
    def test_band(self):
        self.assertIn('偏紧', T.band(4.9))
        self.assertEqual(T.band(5.0), '合适')
        self.assertEqual(T.band(7.99), '合适')
        self.assertIn('略松', T.band(9))
        self.assertIn('偏松', T.band(30))

    def test_max_num_reqs_default_clamped(self):
        self.assertEqual(T.resolve_max_num_reqs(366669, 40960), (4096, False))
        self.assertEqual(T.resolve_max_num_reqs(100000, 131072), (2048, False))

    def test_max_num_reqs_half_tokens(self):
        self.assertEqual(T.resolve_max_num_reqs(3000, 40960), (1500, False))

    def test_max_num_reqs_user_reduced(self):
        self.assertEqual(T.resolve_max_num_reqs(6000, 40960, 4096), (3000, True))
        self.assertEqual(T.resolve_max_num_reqs(366669, 40960, 8192), (8192, False))

    def test_req_to_token(self):
        self.assertAlmostEqual(T.req_to_token_gib(4096, 131072), 2.0, places=2)

    def test_frac_monotonic(self):
        rows = quiet(T.scenario_frac)
        toks = [L['tokens'] for _, L, _ in rows[1:]]
        av = [a for _, _, a in rows[1:]]
        self.assertEqual(toks, sorted(toks))
        self.assertEqual(av, sorted(av, reverse=True))

    def test_frac_target_band(self):
        rows = quiet(T.scenario_frac)
        ok = [f for f, _, a in rows if 5 <= a < 8]
        self.assertEqual(ok[0], 0.89)

    def test_admit_direction(self):
        rows = {(mn, c): m for mn, c, _, m in T.admit_rows()}
        self.assertGreater(rows[(4096, 0.3)]['usage'], rows[(4096, 1.0)]['usage'])
        self.assertGreater(rows[(512, 1.0)]['usage'], rows[(4096, 0.3)]['usage'])
        self.assertGreater(rows[(512, 1.0)]['out_tps'], rows[(4096, 1.0)]['out_tps'])

    def test_retract_direction(self):
        r = [m['retract_events'] for _, _, m in T.retract_rows()]
        self.assertGreater(r[0], r[1])
        self.assertGreaterEqual(r[1], r[2])

    def test_graph_rows_launch_bound(self):
        rows = {(n, op, bs): (e, g) for n, op, bs, _, _, e, g in T.graph_rows()}
        e, g = rows[('llama-70b-tp8', 15.0, 384)]
        self.assertGreater(e / g, 1.3)
        e, g = rows[('qwen3-8b', 6.0, 384)]
        self.assertLess(abs(e / g - 1), 0.02)

    def test_dump_parses(self):
        quiet(T.scenario_admit, True)
        path = os.path.join(HERE, 'sim_admit.log')
        with open(path) as fh:
            R = P.parse(fh)
        self.assertEqual(R['args']['schedule_conservativeness'], 1.0)
        self.assertIn('ADMIT_TIGHT', [f['code'] for f in P.diagnose(R)])


if __name__ == '__main__':
    unittest.main()
