"""SGL-11 实验的边界用例：python3 -m unittest test_parallel -v"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import eplb_sim as E  # noqa: E402
import sim_parallel as S  # noqa: E402
import tbo_sim as T  # noqa: E402
from common.specs import MODELS  # noqa: E402

DSV3 = MODELS['deepseek-v3']


class TestResolve(unittest.TestCase):
    def test_attn_tp_and_moe_tp(self):
        p = S.resolve(DSV3, S.Par(tp=16, attn_dp=2, attn_cp=2, ep=4))
        self.assertEqual(p.attn_tp, 4)
        self.assertEqual(p.moe_tp, 4)

    def test_dp_and_attn_dp_conflict(self):
        with self.assertRaises(S.ConfigError):
            S.resolve(DSV3, S.Par(tp=8, dp=2, attn_dp=2))

    def test_a2a_forces_ep_eq_tp(self):
        p = S.resolve(DSV3, S.Par(tp=16, attn_dp=16, a2a='deepep', ep=4))
        self.assertEqual(p.ep, 16)
        self.assertEqual(p.moe_tp, 1)

    def test_moe_dense_tp_only_1_or_tp(self):
        with self.assertRaises(S.ConfigError):
            S.resolve(DSV3, S.Par(tp=8, moe_dense_tp=4))
        S.resolve(DSV3, S.Par(tp=8, moe_dense_tp=8))
        S.resolve(DSV3, S.Par(tp=8, moe_dense_tp=1))

    def test_redundant_makes_ep24_legal(self):
        with self.assertRaises(S.ConfigError):
            S.resolve(DSV3, S.Par(tp=24, attn_dp=24, a2a='deepep'))
        S.resolve(DSV3, S.Par(tp=24, attn_dp=24, a2a='deepep', redundant=32))

    def test_heads_divisible(self):
        with self.assertRaises(S.ConfigError):
            S.resolve(DSV3, S.Par(tp=6))


class TestMemory(unittest.TestCase):
    def test_dsv3_fp8_tp8_weights(self):
        c = S.capacity(DSV3, S.resolve(DSV3, S.Par(tp=8)), 'h200', 'fp8')
        self.assertAlmostEqual(c['weight_gb'], 84.9, delta=1.0)

    def test_mla_kv_duplicated_under_tp(self):
        p = S.resolve(DSV3, S.Par(tp=8))
        self.assertEqual(S.kv_dup(DSV3, p), 8)
        q = S.resolve(DSV3, S.Par(tp=8, attn_dp=8))
        self.assertEqual(S.kv_dup(DSV3, q), 1)
        self.assertEqual(S.kv_per_gpu_per_token(DSV3, p), 61 * 576 * 2)

    def test_dp_attention_multiplies_total_tokens(self):
        a = S.capacity(DSV3, S.resolve(DSV3, S.Par(tp=8)), 'h200', 'fp8')
        b = S.capacity(DSV3, S.resolve(DSV3, S.Par(tp=8, attn_dp=8, dp_lm_head=True)), 'h200', 'fp8')
        self.assertGreater(b['tok_total'], 4 * a['tok_total'])
        self.assertLess(b['tok_gpu'], a['tok_gpu'])

    def test_gqa_kv_heads_floor_one(self):
        m = MODELS['llama3.1-70b']          # 8 个 KV 头
        p = S.resolve(m, S.Par(tp=8))
        self.assertEqual(S.kv_dup(m, p), 1)
        self.assertEqual(S.kv_per_gpu_per_token(m, S.Par(tp=16)), S.kv_per_gpu_per_token(m, p))

    def test_tied_embedding_no_lm_head(self):
        self.assertEqual(S.components(MODELS['qwen3-0.6b'])['lm_head'], 0)
        self.assertGreater(S.components(DSV3)['lm_head'], 0)


class TestComm(unittest.TestCase):
    def test_padding_rules(self):
        self.assertEqual(S.padding_mode([64] * 8), 'SUM_LEN')
        self.assertEqual(S.padding_mode([512] * 8), 'MAX_LEN')
        self.assertEqual(S.padding_mode([256] * 7 + [0]), 'MAX_LEN')
        self.assertEqual(S.padding_mode([4096, 0, 0, 0], extend=True), 'SUM_LEN')
        self.assertEqual(S.padding_mode([64, 1], a2a='pplx'), 'MAX_LEN')
        self.assertEqual(S.padding_mode([64] * 8, spec=True), 'MAX_LEN')
        self.assertEqual(S.padding_mode([512, 8, 8, 8, 8, 8, 8, 8]), 'SUM_LEN')

    def test_dp_gather_halves_tp_allreduce(self):
        tp = 2 * S.ring_allreduce(8 * 128 * 7168 * 2, 8)
        b = S.dp_gather_bytes([128] * 8, 7168, 'MAX_LEN')
        self.assertAlmostEqual(b['gather'] + b['scatter'], tp / 2)

    def test_deepep_readme_reconcile(self):
        d, c = S.deepep_bytes(128, 7168, 8)
        for ep, (td, tc) in S.LL_MEASURED.items():
            bd, bc = S.LL_BW[ep]
            self.assertLess(abs(d / (bd * 1e3) - td) / td, 0.05)
            self.assertLess(abs(c / (bc * 1e3) - tc) / tc, 0.05)

    def test_ll_buffer_independent_of_ranks(self):
        self.assertAlmostEqual(S.ll_buffer_bytes(128, 7168, 256) / 1e9, 1.91, places=2)
        self.assertEqual(S.ll_buffer_bytes(128, 7168, 256) % 128, 0)


class TestEPLB(unittest.TestCase):
    def test_readme_example(self):
        w = [[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
             [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]]
        expect = [[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
                  [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]]
        for row, exp in zip(w, expect):
            self.assertEqual(E.rebalance(row, 16, 4, 2, 8)[0], exp)

    def test_packing_exact_capacity(self):
        idx, rank = E.balanced_packing([9, 8, 7, 6, 5, 4], 3)
        for pk in range(3):
            self.assertEqual(idx.count(pk), 2)
        self.assertEqual(sorted(rank), [0, 0, 0, 1, 1, 1])

    def test_replicate_hottest(self):
        phy2log, _, cnt = E.replicate_experts([100, 10, 10, 10], 6)
        self.assertEqual(cnt[0], 3)
        self.assertEqual(phy2log[4:], [0, 0])

    def test_auto_algorithm(self):
        w = [1.0] * 256
        self.assertEqual(E.rebalance(w, 288, 8, 4, 32)[2], 'hierarchical')
        self.assertEqual(E.rebalance(w, 288, 8, 9, 72)[2], 'global')

    def test_hierarchical_keeps_group_on_node(self):
        p = E.make_load(256, 8, 1, seed=3)[0]
        m = E.rebalance(p, 288, 8, 4, 32, 'hierarchical')[0]
        node_of_group = {}
        for k, e in enumerate(m):
            node_of_group.setdefault(e // 32, set()).add(k // 9 // 8)
        self.assertTrue(all(len(s) == 1 for s in node_of_group.values()))

    def test_eplb_beats_trivial(self):
        p = E.make_load(256, 8, 1, seed=1)[0]
        m = E.rebalance(p, 288, 8, 4, 32)[0]
        self.assertGreater(E.balancedness(E.gpu_load(p, m, 32)),
                           E.balancedness(E.gpu_load(p, E.trivial_map(256, 288), 32)) + 0.3)


class TestTBO(unittest.TestCase):
    def test_decode_half(self):
        self.assertEqual(T.split_decode(7), (3, 3))
        self.assertEqual(T.split_decode(5, 4), (2, 8))

    def test_extend_balanced(self):
        self.assertEqual(T.split_extend([1000] * 4), ('balanced', 2, 2000))
        self.assertEqual(T.split_extend([3000, 2000, 1000]), ('balanced', 1, 3000))

    def test_extend_two_chunk(self):
        self.assertEqual(T.split_extend([8192]), ('two-chunk', 0, 4096))
        self.assertEqual(T.split_extend([4000, 100, 100, 100])[0], 'two-chunk')

    def test_two_chunk_fallback_tiny(self):
        self.assertEqual(T.split_extend([1]), ('balanced', 0, 0))

    def test_threshold_zero_disables(self):
        self.assertEqual(T.split_extend([8192], threshold=0.0)[0], 'balanced')

    def test_balanced_stops_at_first_increase(self):
        # 差值 9→7→... 一旦变大就停，不会找全局最优
        self.assertEqual(T.split_balanced([1, 1, 10, 1, 1, 1, 1, 1, 1, 1]), 3)

    def test_decode_small_batch_loses(self):
        self.assertLess(T.decode_layer(8, 2000, 72)['off'], T.decode_layer(8, 2000, 72)['on'])
        r = T.decode_layer(256, 2000, 72)
        self.assertGreater(r['off'], r['on'] * 1.2)


if __name__ == '__main__':
    unittest.main()
