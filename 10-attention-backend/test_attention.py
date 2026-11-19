"""SGL-10 的边界用例：后端选择、成本模型、索引和 LSE 合并。python3 -m unittest test_attention -v"""
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import attn_cost as ac
import chooser as ch
import paged_attn as pa
from common.specs import MODELS  # noqa: E402


class TestChooser(unittest.TestCase):
    def r(self, p, m, **kw):
        return ch.resolve(ch.PLATFORMS[p], ch.MODELS[m], ch.Args(**kw))

    def test_hopper_default_fa3(self):
        for m in ('qwen3-8b', 'gpt-oss-120b', 'deepseek-v3'):
            self.assertEqual(self.r('h100', m).prefill, 'fa3')

    def test_old_cuda_drops_to_flashinfer(self):
        self.assertEqual(self.r('h100-cu121', 'qwen3-8b').prefill, 'flashinfer')
        self.assertEqual(self.r('h100-cu121', 'gpt-oss-120b').prefill, 'triton')   # attention sink

    def test_b200_trtllm_page64(self):
        r = self.r('b200', 'qwen3-8b')
        self.assertEqual((r.prefill, r.args.page_size), ('trtllm_mha', 64))
        r = self.r('b200', 'deepseek-v3')
        self.assertEqual((r.prefill, r.args.page_size), ('trtllm_mla', 64))

    def test_dsa_override(self):
        self.assertEqual(self.r('h100', 'deepseek-v3.2').prefill, 'dsa')

    def test_spec_topk_unset_differs_by_platform(self):
        self.assertEqual(self.r('h100', 'qwen3-8b', speculative_algorithm='EAGLE').prefill, 'fa3')
        self.assertEqual(self.r('b200', 'qwen3-8b', speculative_algorithm='EAGLE').prefill, 'flashinfer')

    def test_spec_topk1_with_page16_leaves_fa3(self):
        r = self.r('h100', 'qwen3-8b', speculative_algorithm='EAGLE', speculative_eagle_topk=1, page_size=16)
        self.assertEqual(r.prefill, 'flashinfer')

    def test_fa3_e5m2_fallback(self):
        r = self.r('h100', 'qwen3-8b', attention_backend='fa3', kv_cache_dtype='fp8_e5m2')
        self.assertEqual(r.prefill, 'triton')

    def test_errors(self):
        self.assertIsNotNone(self.r('a100', 'deepseek-v3', attention_backend='fa3').error)
        self.assertIsNotNone(self.r('h100', 'deepseek-v3', attention_backend='trtllm_mla').error)
        self.assertIsNotNone(self.r('b200', 'deepseek-v3', enable_deterministic_inference=True).error)

    def test_fa4_forces_page128(self):
        self.assertEqual(self.r('b200', 'qwen3-8b', attention_backend='fa4').args.page_size, 128)

    def test_split_same_merges(self):
        r = self.r('h100', 'qwen3-8b', prefill_attention_backend='fa3', decode_attention_backend='fa3')
        self.assertEqual(r.args.attention_backend, 'fa3')

    def test_cutedsl_fills_prefill(self):
        r = self.r('b200', 'deepseek-v3', decode_attention_backend='cutedsl_mla')
        self.assertEqual((r.prefill, r.decode), ('trtllm_mla', 'cutedsl_mla'))

    def test_matrix(self):
        self.assertNotIn('triton', ch.candidates(False, fp8=True, page=True))
        self.assertEqual(ch.candidates(True, topk_n=True), ['fa3', 'triton'])


class TestCost(unittest.TestCase):
    def test_gqa_intensity_equals_group(self):
        for k in ('qwen3-8b', 'llama3.1-70b', 'qwen3-235b-a22b'):
            m = MODELS[k]
            self.assertAlmostEqual(ac.decode_attn(m, 4096)[2], m.heads / m.kv_heads)
            self.assertAlmostEqual(ac.decode_attn(m, 4096, kv_bytes=1)[2], 2 * m.heads / m.kv_heads)

    def test_intensity_independent_of_ctx(self):
        m = MODELS['qwen3-8b']
        self.assertAlmostEqual(ac.decode_attn(m, 512)[2], ac.decode_attn(m, 131072)[2])

    def test_mla_near_ridge_and_tp_drops(self):
        m = MODELS['deepseek-v3']
        self.assertTrue(200 < ac.decode_attn(m, 4096)[2] < 300)
        self.assertTrue(ac.decode_attn(m, 4096, tp=8)[2] < 40)

    def test_tc_fill(self):
        self.assertAlmostEqual(ac.tc_tile_fill(4), 0.25)
        self.assertAlmostEqual(ac.tc_tile_fill(16), 1.0)
        self.assertAlmostEqual(ac.tc_tile_fill(2, 8), 1.0)

    def test_split_helps_small_batch_only(self):
        m = MODELS['qwen3-8b']
        self.assertGreater(ac.best_splits(m, 'h100-sxm', 1, 32768), 1)
        self.assertEqual(ac.best_splits(m, 'h100-sxm', 64, 4096), 1)


class TestPaged(unittest.TestCase):
    def test_csr(self):
        r2t, idx, lens = pa.demo_pool(1)
        indptr, ind = pa.build_csr(r2t, idx, lens)
        self.assertEqual(indptr, [0, 5, 8, 15])
        self.assertEqual(len(ind), sum(lens))

    def test_page_table_identity_at_page1(self):
        r2t, idx, lens = pa.demo_pool(1)
        self.assertEqual(pa.page_table(r2t, idx, lens, 1), [r2t[i][:n] for i, n in zip(idx, lens)])

    def test_page_table_page4(self):
        r2t, idx, lens = pa.demo_pool(4)
        self.assertEqual(pa.page_table(r2t, idx, lens, 4), [[4, 1], [2], [10, 7]])
        self.assertFalse(pa.check_page_aligned([16, 17, 19, 18], 4))

    def test_merge_equals_full(self):
        rng = random.Random(3)
        d = 8
        mk = lambda n: pa.rand_vecs(rng, n, d)   # noqa: E731
        pk, pv, nk, nv, qs = mk(20), mk(20), mk(5), mk(5), mk(5)
        s = 1 / math.sqrt(d)
        self.assertLess(pa.max_abs_diff(pa.extend_two_pass(qs, pk, pv, nk, nv, s),
                                        pa.extend_one_pass(qs, pk, pv, nk, nv, s)), 1e-12)

    def test_merge_empty_prefix(self):
        o, l = pa.merge_state([0.0], -math.inf, [2.0], 1.5)
        self.assertEqual((o, l), ([2.0], 1.5))

    def test_split_any_count(self):
        rng = random.Random(5)
        ks, vs, q = pa.rand_vecs(rng, 333, 8), pa.rand_vecs(rng, 333, 8), pa.rand_vecs(rng, 1, 8)[0]
        ref, ref_l = pa.attend(q, ks, vs, 0.3)
        for s in (1, 3, 7, 64, 333):
            o, l, _ = pa.split_k(q, ks, vs, 0.3, s)
            self.assertLess(max(abs(a - b) for a, b in zip(o, ref)), 1e-12)
            self.assertAlmostEqual(l, ref_l)

    def test_prefix_truncation(self):
        self.assertEqual(pa.prefix_hit(515, 64), 512)
        self.assertEqual(pa.prefix_hit(63, 64), 0)
        self.assertEqual(pa.prefix_hit(515, 1), 515)

    def test_contract(self):
        self.assertEqual(pa.check_in_graph(pa.GOOD_IN_GRAPH), [])
        self.assertEqual(len(pa.check_in_graph(pa.BAD_IN_GRAPH)), 5)

    def test_hybrid(self):
        self.assertEqual(pa.select_backend('idle'), 'decode')
        self.assertEqual(pa.select_backend('target_verify'), 'prefill')
        self.assertEqual(pa.select_backend('target_verify', 'decode'), 'decode')
        self.assertEqual(pa.select_backend('mixed'), 'prefill')


if __name__ == '__main__':
    unittest.main()
