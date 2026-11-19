"""plan.py 的单元测试：python3 -m unittest test_plan -v"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan  # noqa: E402


class KVBytes(unittest.TestCase):
    def test_bf16_256kib(self):
        self.assertEqual(plan.kv_bytes_per_token('bf16'), 256 * 1024)

    def test_fp8_halves(self):
        self.assertEqual(plan.kv_bytes_per_token('fp8'), 128 * 1024)

    def test_tp_splits_heads(self):
        self.assertEqual(plan.kv_bytes_per_token('bf16', 2), 128 * 1024)
        self.assertEqual(plan.kv_bytes_per_token('bf16', 8), 32 * 1024)

    def test_tp16_replicates(self):
        # 8 个 KV 头，TP16 时每卡仍要放一个完整头
        self.assertEqual(plan.kv_bytes_per_token('bf16', 16), plan.kv_bytes_per_token('bf16', 8))


class SglangDefaults(unittest.TestCase):
    def test_capacity_buckets(self):
        self.assertEqual(plan.capacity_defaults(24564), (4096, 48))
        self.assertEqual(plan.capacity_defaults(24564, tp=4), (4096, 160))
        self.assertEqual(plan.capacity_defaults(46068), (4096, 32))
        self.assertEqual(plan.capacity_defaults(97871), (8192, 256))
        self.assertEqual(plan.capacity_defaults(183359), (16384, 512))

    def test_prefill_graph_count(self):
        self.assertEqual(len(plan.prefill_graph_bs(2048)), 42)
        self.assertEqual(len(plan.prefill_graph_bs(4096)), 50)
        self.assertEqual(len(plan.prefill_graph_bs(8192)), 58)

    def test_decode_graph_ends_with_max(self):
        self.assertEqual(plan.decode_graph_bs(48)[-1], 48)
        self.assertEqual(plan.decode_graph_bs(256)[:5], [1, 2, 4, 8, 12])

    def test_h20_hits_10g_floor(self):
        mf, reserved, *_ = plan.default_mem_fraction(97871)
        # 512 + 8192×1.5 + 128 + 512 + 58×8 = 13904 MiB，高于 10 GiB 下限
        self.assertAlmostEqual(reserved, 13904, places=3)
        self.assertEqual(mf, 0.858)

    def test_small_chunk_raises_fraction(self):
        default = plan.default_mem_fraction(24564)[0]
        tuned = plan.default_mem_fraction(24564, chunk=2048)[0]
        self.assertLess(default, 0.71)
        self.assertGreater(tuned, 0.82)


class Capacity(unittest.TestCase):
    def test_h20_fp8_about_200k(self):
        t, _ = plan.kv_tokens(plan.GPUS['h20-96'], 'fp8')
        self.assertTrue(190_000 < t < 210_000, t)

    def test_4090_awq_default_negative(self):
        kv, _ = plan.kv_budget_mib(plan.GPUS['rtx4090'], 'awq')
        self.assertLess(kv, 0)

    def test_4090_awq_tuned_tiny(self):
        t, _ = plan.kv_tokens(plan.GPUS['rtx4090'], 'awq', chunk=2048)
        self.assertTrue(0 < t < 8192, t)

    def test_choose_h20_single_card(self):
        p, tp, *_ = plan.choose(plan.GPUS['h20-96'], 65536)
        self.assertEqual((p, tp), ('fp8', 1))

    def test_choose_ampere_prefers_awq(self):
        p, tp, *_ = plan.choose(plan.GPUS['a100-80'], 65536)
        self.assertEqual((p, tp), ('awq', 1))

    def test_choose_4090_needs_two_cards(self):
        p, tp, *_ = plan.choose(plan.GPUS['rtx4090'], 65536)
        self.assertEqual(tp, 2)

    def test_draft_costs_kv(self):
        g = plan.GPUS['h20-96']
        self.assertGreater(plan.kv_tokens(g, 'fp8')[0], plan.kv_tokens(g, 'fp8', spec=True)[0])


class FP8Split(unittest.TestCase):
    def test_tp_ok(self):
        for tp in (1, 2, 4, 8):
            self.assertTrue(plan.fp8_tp_ok(tp), tp)

    def test_bad_intermediate(self):
        # 25600 / 8 = 3200 = 25×128 能整除；换成 5376 = 42×128，TP4 下每片 1344 不是 128 的倍数
        self.assertFalse(plan.fp8_tp_ok(4, intermediate=128 * 42))


class Command(unittest.TestCase):
    def test_yarn_only_beyond_native(self):
        self.assertNotIn('yarn', plan.command('fp8', 1, 32768))
        cmd = plan.command('fp8', 1, 131072)
        self.assertIn('"factor":4.0', cmd)

    def test_parsers_and_spec(self):
        cmd = plan.command('fp8', 2, spec=True)
        self.assertIn('--reasoning-parser qwen3 --tool-call-parser qwen25', cmd)
        self.assertIn('--tp 2', cmd)
        self.assertIn('EAGLE3', cmd)

    def test_report_runs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan.main(['--gpu', 'rtx4090', '--n', '1', '--precision', 'awq'])
        self.assertIn('--context-length 4096', buf.getvalue())


class Qwen38(unittest.TestCase):
    def test_state_bytes(self):
        self.assertAlmostEqual(plan.q38_state_bytes() / 1e6, 153.9, places=1)
        self.assertAlmostEqual(plan.q38_state_bytes('bfloat16') / 1e6, 78.4, places=1)

    def test_kv_bytes(self):
        self.assertEqual(plan.q38_kv_bytes('fp8'), 32 * 1024)

    def test_ratio_matches_cookbook(self):
        self.assertAlmostEqual(plan.q38_ratio(9216, spec='eagle'), 4.59, places=2)
        self.assertAlmostEqual(plan.q38_ratio(9216), 2.55, places=2)

    def test_capacity_splits_budget(self):
        req, tok = plan.q38_capacity(50, 1.0)
        self.assertEqual(tok, int(25 * plan.GIB / plan.q38_kv_bytes()))
        self.assertEqual(req, int(25 * plan.GIB / plan.q38_state_bytes() / 5))


if __name__ == '__main__':
    unittest.main()
