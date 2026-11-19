"""SGL-13 实验的边界用例：python3 -m unittest test_spec -v"""
import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import adaptive_sim as A  # noqa: E402
import resolve_sim as R  # noqa: E402
import spec_sim as S  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TestTheory(unittest.TestCase):
    def test_expected_tokens_limits(self):
        self.assertAlmostEqual(S.expected_tokens(0.8, 0), 1.0)
        self.assertAlmostEqual(S.expected_tokens(0.8, 3), 1 + 0.8 + 0.64 + 0.512)
        self.assertAlmostEqual(S.expected_tokens(1.0, 4), 5.0)

    def test_alpha_roundtrip(self):
        for a in (0.4, 0.7, 0.9):
            L = S.expected_tokens(a, 5)
            self.assertAlmostEqual(S.alpha_from_accept_len(L, 5), a, places=4)

    def test_best_k_shrinks_with_cost(self):
        ks = [S.best_k(0.8, c) for c in (0.02, 0.05, 0.1, 0.2, 0.4)]
        self.assertEqual(ks, sorted(ks, reverse=True))
        self.assertGreater(ks[0], ks[-1])

    def test_accept_rate_below_alpha(self):
        r = quiet(S.scenario_theory)
        for L, k in [(2.4, 3), (3.6, 7)]:
            a = S.alpha_from_accept_len(L, k)
            self.assertLess(S.sglang_accept_rate(L, k + 1), a)
        self.assertIn('best_k', r)


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = quiet(S.scenario_sampling, n_tokens=60000, seed=1)

    def test_lossless_rules_keep_distribution(self):
        for name in ['链式 rejection sampling, k=3', 'block verification, k=3',
                     'target-only 树, 深 3 宽 1', 'target-only 树, 深 3 宽 3']:
            self.assertLess(self.r[name]['tv_uni'], 0.012, name)
            self.assertLess(self.r[name]['tv_bi'], 0.02, name)

    def test_thresholds_break_distribution(self):
        for name in ['target-only 树, 宽 3, threshold_acc=0.5', 'target-only 树, 宽 3, threshold_single=0.3']:
            self.assertGreater(self.r[name]['tv_uni'], 0.1, name)

    def test_block_verification_accepts_more(self):
        self.assertGreater(self.r['block verification, k=3']['accept_len'],
                           self.r['链式 rejection sampling, k=3']['accept_len'])

    def test_chain_matches_formula(self):
        a = self.r['alpha']
        self.assertAlmostEqual(self.r['链式 rejection sampling, k=3']['accept_len'],
                               S.expected_tokens(a, 3), delta=0.05)

    def test_sample_skips_zero_mass(self):
        self.assertEqual(S.sample([0, 0, 1.0], 0.99), 2)
        self.assertEqual(S.sample([0.5, 0, 0.5], 0.5), 2)


class TestTree(unittest.TestCase):
    def test_root_children_and_mask(self):
        # 第一层 4 个孩子，第二层从 parent 1 展开一个（全局下标 4 -> tb=1 -> 根的孩子 0）
        parent_list = [-1, 0, 1, 2, 3]
        selected = [0, 1, 2, 3, 4]
        t = S.build_tree(parent_list, selected, topk=4, seq_len=10)
        self.assertEqual(t['parent'][1:5], [0, 0, 0, 0])
        self.assertEqual(t['parent'][5], 1)
        self.assertEqual(t['positions'], [10, 11, 11, 11, 11, 12])
        self.assertEqual(t['mask'][5], [1, 1, 0, 0, 0, 1])
        # next_token 指向最新加入（下标最小）的孩子，兄弟链往后串
        self.assertEqual(t['next_token'][0], 1)
        self.assertEqual([t['next_sibling'][i] for i in (1, 2, 3)], [2, 3, 4])

    def test_chain_tree(self):
        sl, tl, pl, f = S.draft_tree(S.ToyLM(seed=3), [1, 2], steps=4, topk=1)
        parent_list, selected, dt = S.organize_draft_results(sl, tl, pl, 5)
        t = S.build_tree(parent_list, selected, topk=1)
        self.assertEqual(t['depth'], [0, 1, 2, 3, 4])
        self.assertEqual(f, 3)

    def test_candidate_count(self):
        sl, tl, pl, _ = S.draft_tree(S.ToyLM(seed=0), [5, 6], steps=3, topk=4)
        self.assertEqual(len(sl), 4 + 2 * 16)
        self.assertEqual(len(pl), 3)

    def test_greedy_spec_is_lossless(self):
        r = quiet(S.scenario_tree, n_req=10, n_new=32)
        for cfg, v in r.items():
            self.assertTrue(v['same'], cfg)

    def test_topk1_forces_ndt(self):
        t = S.ToyLM(seed=0)
        _, st = S.spec_generate(t, S.ToyLM(noise=1.0, base=t), [1, 2, 3], 10, steps=3, topk=1, ndt=16)
        self.assertEqual(st['ndt'], 4)

    def test_perfect_draft_accepts_all(self):
        t = S.ToyLM(seed=0)
        _, st = S.spec_generate(t, S.ToyLM(noise=0.0, base=t), [1, 2], 40, steps=3, topk=1, ndt=4)
        self.assertEqual(st['correct'], 3 * st['rounds'])

    def test_toy_lm_deterministic_across_hash_seed(self):
        self.assertEqual(S._mix((1, 2), 0, 1), S._mix((1, 2), 0, 1))
        self.assertNotEqual(S._mix((1, 2), 0, 0), S._mix((2, 1), 0, 0))


class TestBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = quiet(S.scenario_batch)

    def test_wide_tree_loses_at_large_batch(self):
        row = self.r[(1024, (3, 4, 16))]
        self.assertGreater(row[1], 2.5)
        self.assertLess(row[256], 1.0)

    def test_chain_robust(self):
        self.assertGreater(min(self.r[(1024, (3, 1, 4))].values()), 2.0)

    def test_long_context_capped_by_pool(self):
        row = self.r[(8192, (3, 4, 16))]
        self.assertNotIn(64, row)
        self.assertGreater(row[32], self.r[(1024, (3, 4, 16))][32])


class TestNgram(unittest.TestCase):
    def test_copy_vs_fresh(self):
        r = quiet(S.scenario_ngram)
        self.assertGreater(r['copy'], 2.0)
        self.assertLess(r['fresh'], 1.2)

    def test_draft_finds_suffix(self):
        hist = [1, 2, 3, 4, 5, 9, 1, 2, 3]
        branches = S.ngram_draft(hist, max_depth=4, min_match=1, breadth=2, budget=6)
        self.assertTrue(any(list(b[:2]) == [4, 5] for b in branches))


class TestResolve(unittest.TestCase):
    def test_auto_params(self):
        self.assertEqual(R.resolve(algorithm='eagle3', draft_model_path='d')[0]['num_steps'], 5)
        a, _ = R.resolve(algorithm='EAGLE', arch='Qwen3ForCausalLM', draft_model_path='d')
        self.assertEqual((a['num_steps'], a['eagle_topk'], a['num_draft_tokens']), (3, 1, 4))
        a, _ = R.resolve(algorithm='STANDALONE', draft_model_path='d')
        self.assertEqual(a['num_draft_tokens'], 4)

    def test_nextn_alias_and_mtp_path(self):
        a, _ = R.resolve(algorithm='nextn', arch='DeepseekV3ForCausalLM', model_path='/m/dsv3')
        self.assertEqual(a['algorithm'], 'EAGLE')
        self.assertEqual(a['draft_model_path'], '/m/dsv3')

    def test_topk1_rewrites_ndt(self):
        a, log = R.resolve(algorithm='EAGLE', draft_model_path='d', num_steps=3, eagle_topk=1,
                           num_draft_tokens=16)
        self.assertEqual(a['num_draft_tokens'], 4)
        self.assertTrue(any('adjusted' in x for x in log))

    def test_partial_params_assert(self):
        with self.assertRaises(AssertionError):
            R.resolve(algorithm='EAGLE', draft_model_path='d', eagle_topk=4)

    def test_rejection_sampling_constraints(self):
        with self.assertRaises(ValueError):
            R.resolve(algorithm='EAGLE', draft_model_path='d', num_steps=3, eagle_topk=4,
                      num_draft_tokens=8, use_rejection_sampling=True)
        with self.assertRaises(NotImplementedError):
            R.resolve(algorithm='STANDALONE', draft_model_path='d', num_steps=3, eagle_topk=1,
                      num_draft_tokens=4, use_rejection_sampling=True)

    def test_block_verification_enables_rejection(self):
        a, _ = R.resolve(algorithm='EAGLE3', draft_model_path='d', num_steps=3, eagle_topk=1,
                         num_draft_tokens=4, use_block_verification=True)
        self.assertTrue(a['use_rejection_sampling'])
        with self.assertRaises(ValueError):
            R.resolve(algorithm='NGRAM', use_block_verification=True)

    def test_page_tree_backend(self):
        with self.assertRaises(ValueError):
            R.resolve(algorithm='EAGLE', draft_model_path='d', num_steps=3, eagle_topk=4,
                      num_draft_tokens=8, page_size=64, attention_backend='flashmla')
        R.resolve(algorithm='EAGLE', draft_model_path='d', num_steps=3, eagle_topk=4,
                  num_draft_tokens=8, page_size=64, attention_backend='fa3')

    def test_adaptive(self):
        a, _ = R.resolve(algorithm='EAGLE', draft_model_path='d', adaptive=True)
        self.assertEqual((a['num_steps'], a['eagle_topk'], a['num_draft_tokens']), (3, 1, 4))
        with self.assertRaises(ValueError):
            R.resolve(algorithm='EAGLE', draft_model_path='d', adaptive=True, num_steps=4)
        a, log = R.resolve(algorithm='EAGLE', draft_model_path='d', adaptive=True, num_steps=5,
                           eagle_topk=4, num_draft_tokens=8)
        self.assertFalse(a['adaptive'])

    def test_ngram_defaults(self):
        a, _ = R.resolve(algorithm='ngram')
        self.assertEqual((a['eagle_topk'], a['num_draft_tokens'], a['num_steps']), (10, 12, 1))
        with self.assertRaises(ValueError):
            R.resolve(algorithm='NGRAM', dp_attention=True)

    def test_max_running_respects_user(self):
        a, _ = R.resolve(algorithm='EAGLE', draft_model_path='d', max_running_requests=128)
        self.assertEqual(a['max_running_requests'], 128)

    def test_alloc_len(self):
        self.assertEqual(R.alloc_len_per_decode(3, 1, 4), 4)
        self.assertEqual(R.alloc_len_per_decode(5, 4, 8), 20)
        self.assertEqual(R.alloc_len_per_decode(5, 4, 8, page_size=64), 512)
        self.assertEqual(R.alloc_len_per_decode(5, 4, 8, page_size=64, has_draft_kv=False), 20)

    def test_memory_cell_scale(self):
        r = quiet(R.scenario_memory)
        self.assertAlmostEqual(r['tokens_spec'] / r['tokens'], 32 / 33, places=3)


class TestAdaptive(unittest.TestCase):
    def test_candidates_union(self):
        self.assertEqual(A.resolve_candidate_steps(), [0, 1, 3, 5, 7])

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            A.load_config({'x': 1})
        with self.assertRaises(ValueError):
            A.load_config({'1': {'candidate_steps': [-1]}})

    def test_warmup_then_interval(self):
        s = A.AdaptiveStepSlot(3, {'candidate_steps': [1, 3, 5, 7]})
        changed = [s.update([5]) for _ in range(20)]
        self.assertEqual(changed[:14], [False] * 14)      # 10 轮 warmup + 前 4 轮不在评估点
        self.assertEqual(changed.index(True), 14)

    def test_drop_to_zero_and_probe(self):
        s = A.AdaptiveStepSlot(1, {'candidate_steps': [0, 1, 3], 'down_hysteresis': 0.0})
        for _ in range(15):
            s.update([0])
        self.assertEqual(s.current_steps, 0)
        for _ in range(5):
            s.update([0])
        self.assertEqual(s.current_steps, 1)               # steps=0 之后下一个评估点就去试探

    def test_hysteresis(self):
        s = A.AdaptiveStepSlot(3, {'candidate_steps': [1, 3, 5]})
        s.ema_accept_len = 2.3                              # > 0.25 不降（1-0.5-0.25），<= 2.5 不升
        self.assertFalse(s._recompute())
        s.ema_accept_len = 0.2
        self.assertTrue(s._recompute())
        self.assertEqual(s.current_steps, 1)

    def test_route(self):
        r = quiet(A.scenario_route)
        self.assertEqual(r[9][1], 8)
        self.assertEqual(r[33][1:], (32, 1))
        self.assertEqual(r[100][2], 0)

    def test_adaptive_beats_fixed(self):
        r = quiet(A.scenario_trace)
        self.assertGreater(r['adaptive'], max(r[k] for k in (1, 3, 5, 7)))


if __name__ == '__main__':
    random.seed(0)
    unittest.main()
