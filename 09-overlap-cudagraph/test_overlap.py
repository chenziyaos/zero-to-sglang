"""sim_overlap / cuda_graph_sim / cost 的边界情况。python3 -m unittest test_overlap -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost import PRESETS, gpu_ms, launch_ms, n_kernels, with_  # noqa: E402
from cuda_graph_sim import decode_bs, mem_fraction, pad_to, prefill_tokens, tier, to_capture  # noqa: E402
from sim_overlap import forward_on_gpu, simulate, toy_next  # noqa: E402


class TestCost(unittest.TestCase):
    def test_kernel_count(self):
        c = PRESETS['qwen3-8b']
        self.assertEqual(n_kernels(c), 36 * 14 + 6)
        self.assertEqual(n_kernels(with_(c, tp=2)), 36 * 16 + 6)   # TP 每层多两个 all-reduce

    def test_decode_is_bandwidth_bound(self):
        c = PRESETS['qwen3-8b']
        self.assertLess(gpu_ms(c, 12) - gpu_ms(c, 9), 0.05 * gpu_ms(c, 9))

    def test_graph_launch_constant(self):
        c = PRESETS['llama-70b-tp8']
        self.assertEqual(launch_ms(c, True), launch_ms(PRESETS['qwen3-0.6b'], True))
        self.assertGreater(launch_ms(c, False), 50 * launch_ms(c, True))


class TestOverlap(unittest.TestCase):
    def test_overlap_never_slower(self):
        for name, c in PRESETS.items():
            for bs in (1, 32, 256):
                for graph in (False, True):
                    a = simulate(c, bs, 16, False, graph)['step']
                    b = simulate(c, bs, 16, True, graph)['step']
                    self.assertLessEqual(b, a + 1e-9, (name, bs, graph))

    def test_overlap_with_graph_fills_gpu(self):
        r = simulate(PRESETS['qwen3-0.6b'], 1, 16, True, True)
        self.assertAlmostEqual(r['step'], gpu_ms(PRESETS['qwen3-0.6b'], 1), places=6)
        self.assertGreater(r['gpu_busy'], 0.999)

    def test_eager_launch_bound(self):
        """eager 且启动比 GPU 慢时，overlap 加速很小，步长被启动卡住。"""
        c = PRESETS['qwen3-0.6b']
        a = simulate(c, 1, 16, False, False)['step']
        b = simulate(c, 1, 16, True, False)['step']
        self.assertLess(a / b, 1.05)
        self.assertGreaterEqual(b, launch_ms(c, False))

    def test_eager_kernels_wait_for_issue(self):
        c = PRESETS['qwen3-0.6b']
        cpu_end, gs, ge, g = forward_on_gpu(c, 1, 0.0, 0.0, False)
        self.assertAlmostEqual(cpu_end, launch_ms(c, False))
        self.assertGreater(ge - gs, g)            # GPU 上被启动拉长

    def test_toy_model_deterministic(self):
        self.assertEqual(toy_next(11, 1), toy_next(11, 1))
        self.assertNotEqual(toy_next(11, 1), toy_next(11, 2))


class TestBuckets(unittest.TestCase):
    def test_normal_list(self):
        bs = decode_bs(256)
        self.assertEqual(bs[:7], [1, 2, 4, 8, 12, 16, 24])
        self.assertEqual(len(bs), 36)
        self.assertEqual(bs[-1], 256)

    def test_max_bs_appended(self):
        self.assertEqual(decode_bs(100)[-1], 100)
        self.assertEqual(decode_bs(300)[-3:], [272, 288, 300])

    def test_spec_denser(self):
        self.assertEqual(decode_bs(256, 'spec')[:10], [1, 2, 3, 4, 5, 6, 7, 8, 10, 12])

    def test_pad(self):
        bs = decode_bs(256)
        self.assertEqual(pad_to(5, bs), 8)
        self.assertEqual(pad_to(8, bs), 8)
        self.assertEqual(pad_to(37, bs), 40)
        with self.assertRaises(AssertionError):
            pad_to(257, bs)

    def test_alignment_filter(self):
        cap, mul = to_capture(decode_bs(256), 4096, attn_tp=4, gathered=True)
        self.assertEqual(mul, 4)
        self.assertTrue(all(b % 4 == 0 for b in cap))
        cap, _ = to_capture(decode_bs(256), 50)
        self.assertEqual(cap[-1], 50)
        cap, _ = to_capture(decode_bs(256), 50, attn_tp=4, gathered=True)
        self.assertEqual(cap[-1], 52)

    def test_prefill_tokens(self):
        t = prefill_tokens(8192)
        self.assertEqual(t[:3], [4, 8, 12])
        self.assertEqual(t[-1], 8192)
        self.assertNotIn(1024 + 64, t)


class TestDefaults(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(tier(24564, 1), (4096, 48))
        self.assertEqual(tier(24564, 4), (4096, 160))
        self.assertEqual(tier(40960, 1), (4096, 32))
        self.assertEqual(tier(81559, 1), (8192, 256))
        self.assertEqual(tier(81559, 8), (8192, 512))
        self.assertEqual(tier(183359, 1), (16384, 512))

    def test_activation_dominates(self):
        f, res, parts, floor, _, _ = mem_fraction(81559, 1)
        self.assertFalse(floor)                       # 默认 chunk 8192 时预留 13.9 GB，超过 10 GB 保底
        self.assertEqual(parts['act'], 8192 * 1.5)
        self.assertEqual(f, 0.83)

    def test_floor_hides_max_bs(self):
        a = mem_fraction(81559, 1, chunk=2048)
        b = mem_fraction(81559, 1, chunk=2048, max_running=16)
        self.assertTrue(a[3])
        self.assertEqual(a[0], b[0])

    def test_small_gpu_max_running(self):
        self.assertGreater(mem_fraction(24564, 1, max_running=8)[0], mem_fraction(24564, 1)[0])


if __name__ == '__main__':
    unittest.main()
