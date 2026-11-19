"""SGL-12 实验的边界用例：python3 -m unittest test_quant -v"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import mem_sim as M  # noqa: E402
import quant_sim as Q  # noqa: E402
import resolve_sim as R  # noqa: E402

P = R.PLATFORMS


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TestFormats(unittest.TestCase):
    def test_grid_sizes_and_max(self):
        self.assertEqual(Q.E4M3[-1], 448)
        self.assertEqual(Q.E5M2[-1], 57344)
        self.assertEqual(Q.E2M1, [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        self.assertEqual(len(Q.E4M3) - 1, 126)

    def test_round_saturates_and_ties_to_even(self):
        self.assertEqual(Q.round_to_grid(1000.0, Q.E4M3), 448)
        self.assertEqual(Q.round_to_grid(-1000.0, Q.E4M3), -448)
        self.assertEqual(Q.round_to_grid(2.5, Q.E2M1), 2)      # 2 和 3 之间取偶数尾数
        self.assertEqual(Q.round_to_grid(5.0, Q.E2M1), 4)

    def test_ue8m0_rounds_up_to_power_of_two(self):
        self.assertEqual(Q.ceil_to_ue8m0(0.3), 0.5)
        self.assertEqual(Q.ceil_to_ue8m0(1.0), 1.0)


class TestGranularity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Q.make_weight(64, 256)

    def test_int8_per_tensor_hurt_by_outliers(self):
        self.assertGreater(Q.rel_err(self.w, Q.int8_per_tensor(self.w)),
                           2 * Q.rel_err(self.w, Q.int8_per_row(self.w)))

    def test_fp8_insensitive_to_scale_granularity(self):
        a = Q.rel_err(self.w, Q.fp8_per_tensor(self.w))
        b = Q.rel_err(self.w, Q.fp8_per_row(self.w))
        self.assertLess(abs(a - b) / b, 0.2)

    def test_4bit_ordering(self):
        sym = Q.rel_err(self.w, Q.int_group(self.w, 4, 128, True))
        zp = Q.rel_err(self.w, Q.int_group(self.w, 4, 128, False))
        nv = Q.rel_err(self.w, Q.nvfp4(self.w))
        self.assertLess(zp, sym)
        self.assertLess(nv, Q.rel_err(self.w, Q.mxfp4(self.w)))


class TestAlgorithms(unittest.TestCase):
    def test_smoothquant_beats_per_tensor_int8(self):
        x, _ = Q.make_act(16, 256)
        w = Q.make_weight(64, 256, seed=2)
        base = Q._w8a8(x, w, Q.int8_per_tensor, Q.int8_per_row)
        xs, ws = Q.smooth(x, w, 0.5)
        sm = Q.rel_err(Q.matmul_t(x, w), Q.matmul_t(Q.int8_per_tensor(xs), Q.int8_per_row(ws)))
        self.assertLess(sm, base / 2)

    def test_gptq_beats_rtn(self):
        x, _ = Q.make_act(128, 64, seed=5, n_outlier=3, mag=10)
        w = Q.make_weight(16, 64, seed=6, outlier_frac=0.0)
        h = Q._gram(x)
        rtn = Q.out_err_h(w, Q.int_group(w, 4, -1, True), h)
        g = Q.out_err_h(w, Q.gptq(w, h, 4), h)
        self.assertLess(g, rtn)

    def test_kv_scale1_saturates_on_big_outlier(self):
        res = quiet(Q.scenario_kv, seq=64, d=64)
        e4_s1, e4_cal, e5_s1, _ = res[600]
        self.assertGreater(e4_s1, 0.5)
        self.assertLess(e4_cal, e4_s1)
        self.assertLess(e5_s1, e4_s1)


class TestResolve(unittest.TestCase):
    def run_(self, plat, cli, ck, dtype='float16'):
        return R.run_case(plat, cli, ck, dtype)[0]

    def test_awq_auto_upgrades_to_marlin(self):
        self.assertEqual(self.run_('h100', None, 'awq'), 'awq_marlin')

    def test_awq_explicit_bf16_rejected(self):
        self.assertIn('bfloat16 is not supported', self.run_('h100', 'awq', 'awq', 'bfloat16'))

    def test_awq_on_t4_stays_awq(self):
        self.assertEqual(self.run_('t4', None, 'awq'), 'awq')

    def test_plain_gptq_unknown_on_cuda(self):
        self.assertIn('Unknown quantization method: gptq', self.run_('a100', 'gptq', 'gptq'))
        self.assertIn('Unknown', self.run_('a100', None, 'gptq-asym'))

    def test_gptq_on_rocm(self):
        self.assertIn('not supported in ROCm', self.run_('mi300x', None, 'gptq'))

    def test_mismatch(self):
        self.assertIn('does not match', self.run_('h100', 'awq', 'fp8-block', 'bfloat16'))

    def test_compatible_override(self):
        self.assertEqual(self.run_('h100', 'w8a8_fp8', 'compressed-tensors', 'bfloat16'), 'w8a8_fp8')

    def test_capability(self):
        self.assertIn('Minimum capability: 89', self.run_('a100', 'w8a8_fp8', None, 'bfloat16'))
        self.assertEqual(self.run_('a100', None, 'fp8-block', 'bfloat16'), 'fp8')

    def test_modelopt_fp4(self):
        self.assertEqual(self.run_('b200', None, 'modelopt-fp4', 'bfloat16'), 'modelopt_fp4')


class TestMarlinShape(unittest.TestCase):
    def test_marlin_types(self):
        self.assertEqual(R.marlin_types(True, 75), [])
        self.assertEqual(R.marlin_types(True, 80), ['uint4'])
        self.assertIn('uint4b8', R.marlin_types(False, 90))

    def test_qwen72b_unpadded_only_tp1(self):
        rows = R.shape_report('qwen2.5-72b-bf16')
        self.assertTrue(rows[0][1])
        self.assertFalse(any(r[1] for r in rows[1:]))

    def test_qwen72b_padded_all_tp(self):
        self.assertTrue(all(r[1] and r[3] for r in R.shape_report('qwen2.5-72b-awq')))


class TestFp8Path(unittest.TestCase):
    def test_ampere_uses_marlin(self):
        self.assertIn('Marlin', R.fp8_linear_path(P['a100'], block=True))

    def test_hopper_block_deepgemm(self):
        self.assertIn('DeepGEMM', R.fp8_linear_path(P['h100'], block=True))
        self.assertIn('Triton', R.fp8_linear_path(P['h100-nodg'], block=True))

    def test_l40s_old_cuda_per_tensor(self):
        self.assertIn('逐张量', R.fp8_linear_path(P['l40s-cu121'], block=False))


class TestKV(unittest.TestCase):
    def test_kv_dtype(self):
        self.assertEqual(R.kv_cache_dtype('auto', P['h100'], ckpt_kv_algo='FP8'), 'float8_e4m3fn')
        self.assertEqual(R.kv_cache_dtype('fp8_e5m2', P['mi300x']), 'float8_e4m3fnuz')
        with self.assertRaises(R.QuantError):
            R.kv_cache_dtype('int8', P['h100'])
        with self.assertRaises(R.QuantError):
            R.kv_cache_dtype('nvfp4', P['h100'], torch_fp4=False)

    def test_kv_scales(self):
        self.assertEqual(R.kv_scales(), (1.0, 1.0))
        self.assertEqual(R.kv_scales(0.12, -1), (0.12, 0.12))
        self.assertEqual(R.kv_scales(0.1, 0.05, fnuz=True), (0.2, 0.1))


class TestMem(unittest.TestCase):
    def test_checkpoint_sizes_within_1pct(self):
        for key, fmt, real, _ in M.HF_SIZES:
            self.assertAlmostEqual(M.checkpoint_gb(key, fmt) / real, 1, delta=0.01)

    def test_kv_fp8_doubles_tokens(self):
        a = M.kv_tokens('qwen3-8b', 'h100-sxm', 1, 'bf16', 'bf16')[0]
        b = M.kv_tokens('qwen3-8b', 'h100-sxm', 1, 'bf16', 'fp8')[0]
        self.assertAlmostEqual(b / a, 2, delta=0.001)

    def test_70b_single_h100_needs_int4(self):
        self.assertEqual(M.kv_tokens('llama3.1-70b', 'h100-sxm', 1, 'fp8-block', 'bf16')[0], 0)
        self.assertGreater(M.kv_tokens('llama3.1-70b', 'h100-sxm', 1, 'awq-int4-g128', 'bf16')[0], 50000)

    def test_w4a16_speedup_decays_with_batch(self):
        a10 = M.CARDS['a10']
        n, k = 12288, 4096
        sp = [M.linear_time_us(a10, n, k, b, 'bf16') / M.linear_time_us(a10, n, k, b, 'w4a16')
              for b in (1, 64, 256)]
        self.assertGreater(sp[0], 3.8)
        self.assertLess(sp[2], 1.01)
        self.assertTrue(sp[0] > sp[1] > sp[2])


if __name__ == '__main__':
    unittest.main()
