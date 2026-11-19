"""SGL-16 实验的边界用例：python3 -m unittest test_struct_lora -v"""
import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import lora_sim as L  # noqa: E402
import resolve_sim as R  # noqa: E402
import struct_sim as S  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ================================================================ 结构化输出
class TestMask(unittest.TestCase):
    def test_bitmask_qwen3(self):
        self.assertEqual(S.mask_bytes(151936, 1), 18992)

    def test_bool_is_8x(self):
        for v in S.VOCAB.values():
            self.assertLessEqual(S.mask_bytes(v, 1, 'bool') / S.mask_bytes(v, 1), 8.0)
            self.assertGreater(S.mask_bytes(v, 1, 'bool') / S.mask_bytes(v, 1), 7.99)

    def test_bitmask_rounds_up(self):
        self.assertEqual(S.mask_bytes(33, 1), 8)

    def test_fill_time_linear(self):
        self.assertAlmostEqual(S.fill_time(256, 40e-6), 256 * 40e-6)
        self.assertAlmostEqual(S.fill_time(256, 40e-6, batched_threads=8), 32 * 40e-6)

    def test_cfg_exceeds_decode_at_256(self):
        _, rows = quiet(S.scenario_mask)
        t, fills = rows[256]
        self.assertGreater(fills[1], t)
        self.assertLess(fills[0], t)


class TestTokenFilter(unittest.TestCase):
    def test_allow_mode(self):
        row = [0] * 2
        S.set_token_filter(row, [1, 33], is_allowed=True)
        self.assertEqual([S.bit(row, t) for t in (0, 1, 32, 33)], [0, 1, 0, 1])

    def test_block_mode(self):
        row = [0] * 2
        S.set_token_filter(row, [5], is_allowed=False)
        self.assertEqual(S.bit(row, 5), 0)
        self.assertEqual(S.bit(row, 6), 1)
        self.assertEqual(row[1], S.M32)

    def test_no_reset_narrows(self):
        row = [0] * 1
        S.set_token_filter(row, [1, 2, 3], is_allowed=True)
        S.set_token_filter(row, [2], is_allowed=False, reset=False)
        self.assertEqual([S.bit(row, t) for t in (1, 2, 3)], [1, 0, 1])


class TestToyGrammar(unittest.TestCase):
    def setUp(self):
        self.V = S.TOY_VOCAB
        self.g = S.ToyGrammar(S.ToyRegex(S.TOY_RE), self.V, self.V.index('<eos>'))

    def names(self):
        return [self.V[t] for t in self.g.allowed()]

    def test_start_allows_brace_only(self):
        self.assertEqual(set(self.names()), {'{', '{"'})

    def test_cross_boundary_token(self):
        for s in ('{"', 'n'):
            self.g.accept_token(self.V.index(s))
        self.assertIn('":1', self.names())
        self.assertIn('":', self.names())

    def test_eos_only_when_final(self):
        for s in ('{"', 'n', '":1'):
            self.g.accept_token(self.V.index(s))
        self.assertNotIn('<eos>', self.names())
        self.g.accept_token(self.V.index('}'))
        self.assertEqual(self.names(), ['<eos>'])

    def test_illegal_token_asserts(self):
        with self.assertRaises(AssertionError):
            self.g.accept_token(self.V.index('a'))

    def test_rollback_restores(self):
        for s in ('{"', 'n', '":1', '12'):
            self.g.accept_token(self.V.index(s))
        self.g.rollback(2)
        self.assertEqual(self.g.state, (3, 0))

    def test_rollback_limit(self):
        with self.assertRaises(AssertionError):
            self.g.rollback(S.MAX_ROLLBACK_TOKENS + 1)

    def test_mask_matches_allowed(self):
        mask = S.new_mask(1, len(self.V))
        self.g.fill_vocab_mask(mask, 0)
        self.assertEqual([t for t in range(len(self.V)) if S.bit(mask[0], t)], self.g.allowed())


class TestTraverseTree(unittest.TestCase):
    def test_scenario_tree(self):
        _, vis, g2 = quiet(S.scenario_fsm)
        self.assertEqual(vis, [0, 1, 2, 3, 6, 4])
        self.assertEqual(g2.state, (0, 0))
        self.assertEqual(len(g2.hist), 1)

    def test_out_of_vocab_draft_rejected(self):
        V = S.TOY_VOCAB
        g = S.ToyGrammar(S.ToyRegex(S.TOY_RE), V, V.index('<eos>'))
        draft = [-1, len(V) + 3]
        mask = S.new_mask(2, len(V) + 32)
        vis = S.traverse_tree([1, -1], [-1, -1], draft, g, mask, vocab=len(V))
        self.assertEqual(vis, [0])


class TestQueue(unittest.TestCase):
    def test_no_dedup_compiles_each(self):
        w, n = S.simulate_compile([(0.0, 'A')] * 10, 4, lambda k: 1.0)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(w[-1], 3.0)

    def test_dedup_compiles_once(self):
        w, n = S.simulate_compile([(0.0, 'A')] * 10, 4, lambda k: 1.0, dedup=True)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(w[-1], 1.0)

    def test_cache_hit_after_done(self):
        _, n = S.simulate_compile([(0.0, 'A'), (2.0, 'A')], 1, lambda k: 1.0)
        self.assertEqual(n, 1)

    def test_key_not_sorted(self):
        self.assertNotEqual(S.schema_key({'a': 1, 'b': 2}), S.schema_key({'b': 2, 'a': 1}))

    def test_tp_sync(self):
        ready, failed = S.tp_sync([({1, 2}, set()), ({2}, {9})])
        self.assertEqual((ready, failed), ({2}, {9}))

    def test_timeout_idle_50s(self):
        self.assertAlmostEqual(S.timeout_seconds(0), 50.0)

    def test_timeout_nonoverlap_longer(self):
        self.assertAlmostEqual(S.timeout_seconds(0.02, False) - S.timeout_seconds(0.02), 50.0)


class TestOverlap(unittest.TestCase):
    def test_overlap_hides_small_fill(self):
        self.assertEqual(S.step_time(0.01, 0.005), S.step_time(0.01, 0))

    def test_sync_adds_fill(self):
        self.assertAlmostEqual(S.step_time(0.01, 0.005, mode='sync') - S.step_time(0.01, 0, mode='sync'), 0.005)

    def test_overlap_exposes_large_fill(self):
        self.assertGreater(S.step_time(0.01, 0.02), S.step_time(0.01, 0))


class TestThink(unittest.TestCase):
    def test_kmp_partial_fallback(self):
        m = S.TokenSequenceMatcher([1, 1, 2])
        k = 0
        for t in (1, 1, 1):
            k = m.advance(k, t)
        self.assertEqual(k, 2)
        self.assertEqual(m.advance(k, 2), 3)

    def test_empty_pattern(self):
        with self.assertRaises(ValueError):
            S.TokenSequenceMatcher([])

    def test_budget_forces_end(self):
        rows, r = quiet(S.run_think, ['想'] * 20, ['{"', 'n', '":1', '3', '}', '<eos>'], 6)
        thinking = [x for x in rows if x[0] == '思考']
        self.assertEqual(len(thinking), 9)
        self.assertEqual([x[2] for x in thinking[-3:]], ['</', 'think', '>'])
        self.assertTrue(r.grammar.is_terminated())

    def test_excluded_rewritten(self):
        rows, _ = quiet(S.scenario_think)[:2]
        self.assertEqual(rows[2][1:3], ('<tool_call>', '想'))

    def test_no_budget_no_filter(self):
        T = S.THINK_VOCAB.index
        r = S.Reasoner([T('</'), T('think'), T('>')], max_think=-1)
        mask = S.new_mask(1, len(S.THINK_VOCAB))
        r.fill_vocab_mask(mask, 0)
        self.assertTrue(all(S.bit(mask[0], t) for t in range(len(S.THINK_VOCAB))))

    def test_rollback_across_end(self):
        *_, snaps, r2 = quiet(S.scenario_think)
        self.assertEqual((r2.tokens_in_think, r2.matched, r2.tokens_after_end), snaps[1])

    def test_answer_grammar_not_fed_while_thinking(self):
        T = S.THINK_VOCAB.index
        inner = S.ToyGrammar(S.ToyRegex(S.TOY_RE), S.THINK_VOCAB, T('<eos>'))
        r = S.Reasoner([T('</'), T('think'), T('>')], inner=inner)
        r.accept_token(T('想'))
        self.assertEqual(len(inner.hist), 1)


# ================================================================ LoRA
class TestLoRAMemory(unittest.TestCase):
    def test_llama8b_r64_all(self):
        self.assertEqual(L.pool_bytes('llama3.1-8b', 64, 8), int(2.5 * 2 ** 30))

    def test_linear_in_rank_and_slots(self):
        a = L.pool_bytes('llama3.1-8b', 16, 8)
        self.assertEqual(L.pool_bytes('llama3.1-8b', 64, 8), 4 * a)
        self.assertEqual(L.pool_bytes('llama3.1-8b', 16, 32), 4 * a)

    def test_attn_smaller_than_all(self):
        self.assertLess(L.pool_bytes('llama3.1-8b', 64, 8, ('qkv_proj', 'o_proj')) * 3,
                        L.pool_bytes('llama3.1-8b', 64, 8))

    def test_adapter_is_one_slot(self):
        self.assertEqual(L.adapter_bytes('llama3.1-8b', 64) * 8, L.pool_bytes('llama3.1-8b', 64, 8))

    def test_tp_row_parallel_splits_a(self):
        a, b = L.lora_shapes('llama3.1-70b', 'down_proj', 64, 8, tp=4)
        a1, b1 = L.lora_shapes('llama3.1-70b', 'down_proj', 64, 8, tp=1)
        self.assertEqual(a[2] * 4, a1[2])
        self.assertEqual(b, b1)

    def test_qkv_stack(self):
        a, _ = L.lora_shapes('llama3.1-8b', 'qkv_proj', 16, 8)
        self.assertEqual(a[1], 48)


class TestLoRAAdmit(unittest.TestCase):
    def test_validate_over_mlpb(self):
        self.assertFalse(L.validate_lora_batch({None, 'A', 'B', 'C', 'D'}, 4, set()))

    def test_pinned_reserves_slot(self):
        self.assertTrue(L.validate_lora_batch({'B', 'C', 'D'}, 4, {'A'}))
        self.assertFalse(L.validate_lora_batch({None, 'B', 'C', 'D'}, 4, {'A'}))

    def test_schedule_skips_not_blocks(self):
        rows, run = L.schedule_pass([('r1', 'B'), ('r2', 'C'), ('r3', 'D'), ('r4', 'A')], {None, 'A'}, 4)
        self.assertEqual([x[2] for x in rows], ['进（新占一个槽）', '进（新占一个槽）', '跳过', '进（已在跑）'])
        self.assertEqual(run, {None, 'A', 'B', 'C'})

    def test_pool_empty_first(self):
        p = L.Pool(4)
        p.prepare({None, 'A'})
        p.prepare({'B'})
        self.assertEqual(p.loads, 2)
        self.assertIn('A', p.slots)

    def test_pool_lru_evicts_oldest(self):
        p = L.Pool(2)
        p.prepare({'A'})
        p.prepare({'B'})
        p.prepare({'A'})
        p.prepare({'C'})
        self.assertEqual(sorted(p.slots), ['A', 'C'])

    def test_pool_fifo_evicts_first_in(self):
        p = L.Pool(2, 'fifo')
        for b in ({'A'}, {'B'}, {'A'}, {'C'}):
            p.prepare(b)
        self.assertEqual(sorted(p.slots), ['B', 'C'])

    def test_pool_prefers_evicting_adapter_over_base(self):
        p = L.Pool(2)
        p.prepare({None})
        p.prepare({'A'})
        p.prepare({'B'})
        self.assertIn(None, p.slots)

    def test_pool_all_pinned_raises(self):
        p = L.Pool(2, pinned={'A', 'B'})
        p.prepare({'A'})
        p.prepare({'B'})
        with self.assertRaises(ValueError):
            p.prepare({'C'})

    def test_lru_beats_fifo_with_spare_slots(self):
        _, res = quiet(L.scenario_admit)
        self.assertLess(res[(1.2, 16)][0], res[(1.2, 16)][1])


class TestLoRASegments(unittest.TestCase):
    def test_docstring_example(self):
        perm, w = L.permutation([0, 1, 0], [2, 1, 3])
        self.assertEqual(perm, [0, 1, 3, 4, 5, 2])
        self.assertEqual(L.segments(w, 3), ([0, 0, 1], [0, 3, 5, 6]))

    def test_chunk_size_default_16(self):
        self.assertEqual(L.chunk_size_for(10000), 16)

    def test_chunk_size_thresholds(self):
        self.assertEqual([L.chunk_size_for(n, 128) for n in (10, 64, 255, 256)], [16, 32, 32, 128])

    def test_segments_cover_all_tokens(self):
        rnd = random.Random(3)
        ws = [rnd.randrange(5) for _ in range(30)]
        lens = [rnd.randint(1, 90) for _ in range(30)]
        _, w = L.permutation(ws, lens)
        wi, ip = L.segments(w, 32)
        self.assertEqual(ip[-1], sum(lens))
        self.assertTrue(all(0 < ip[i + 1] - ip[i] <= 32 for i in range(len(wi))))

    def test_triton_long_request_waste(self):
        la, ac = L.triton_blocks([8000, 10], [0, 1])
        self.assertEqual(la, 1000)
        self.assertEqual(ac, 501)


class TestLoRADrain(unittest.TestCase):
    def test_no_drainer_starves(self):
        d, _ = L.run_drain(0.0, horizon=20)
        self.assertIsNone(d.start)

    def test_drainer_admits(self):
        d, _ = L.run_drain(1.0, horizon=20)
        self.assertIsNotNone(d.start)
        self.assertGreater(d.start - d.arr, 1.0)

    def test_drain_victim_least_remaining(self):
        dr = L.Drainer(2, 0.5)
        a, b = L.Req(0, 'A', 0, 100), L.Req(1, 'B', 0, 300)
        w = L.Req(2, 'C', 0, 50)
        dr.update([w], [a, b], now=1.0)
        self.assertEqual(dr.st('A')['drain'], 'C')
        self.assertIsNone(dr.st('B')['drain'])

    def test_drain_tolerance(self):
        dr = L.Drainer(2, 0.5)
        a, b = L.Req(0, 'A', 0, 100), L.Req(1, 'B', 0, 300)
        dr.update([L.Req(2, 'C', 0, 50)], [a, b], now=1.0)
        self.assertTrue(dr.can_schedule(L.Req(3, 'A', 1, 120)))
        self.assertFalse(dr.can_schedule(L.Req(4, 'A', 1, 121)))


# ================================================================ 启动参数
class TestResolve(unittest.TestCase):
    def test_unknown_kw(self):
        with self.assertRaises(TypeError):
            R.resolve(foo=1)

    def test_grammar_default_xgrammar(self):
        a, _, _ = R.resolve()
        self.assertEqual(a['grammar_backend'], 'xgrammar')

    def test_invalid_backend(self):
        with self.assertRaises(R.StructArgError):
            R.resolve(grammar_backend='lark')

    def test_strict_none(self):
        with self.assertRaisesRegex(R.StructArgError, 'token filtering'):
            R.resolve(grammar_backend='none', enable_strict_thinking=True)

    def test_paths_enable_lora(self):
        a, _, w = R.resolve(lora_paths=['x=/p'])
        self.assertTrue(a['enable_lora'])
        self.assertEqual(a['lora_paths'], [('x', '/p', False)])
        self.assertIn('--enable-lora is set to True', w[0])

    def test_paths_parse_forms(self):
        out = R.parse_lora_paths(['a', 'b=/p=q', dict(lora_name='c', lora_path='/c', pinned=True)])
        self.assertEqual(out, [('a', 'a', False), ('b', '/p=q', False), ('c', '/c', True)])

    def test_paths_dict_missing_key(self):
        with self.assertRaisesRegex(R.StructArgError, "'lora_name' and 'lora_path'"):
            R.parse_lora_paths([dict(lora_name='c')])

    def test_paths_bad_type(self):
        with self.assertRaises(R.StructArgError):
            R.parse_lora_paths([3])

    def test_overlap_limit_boundary(self):
        R.resolve(lora_paths=['a'], enable_lora_overlap_loading=True, max_loaded_loras=16)
        with self.assertRaises(R.StructArgError):
            R.resolve(lora_paths=['a'], enable_lora_overlap_loading=True, max_loaded_loras=17)

    def test_chunk_power_of_two(self):
        for c in (16, 32, 64, 128):
            R.resolve(lora_paths=['a'], max_lora_chunk_size=c)
        for c in (8, 48, 256):
            with self.assertRaises(R.StructArgError):
                R.resolve(lora_paths=['a'], max_lora_chunk_size=c)

    def test_pinned_limit(self):
        mk = lambda n: [dict(lora_name=f'a{i}', lora_path='/p', pinned=True) for i in range(n)]  # noqa: E731
        R.resolve(lora_paths=mk(7))
        with self.assertRaises(R.StructArgError):
            R.resolve(lora_paths=mk(8))

    def test_spec_whitelist(self):
        for alg in ('NGRAM', 'EAGLE', 'EAGLE3', 'DFLASH', 'DSPARK'):
            R.resolve(lora_paths=['a'], speculative_algorithm=alg)
        with self.assertRaises(R.StructArgError):
            R.resolve(lora_paths=['a'], speculative_algorithm='STANDALONE')

    def test_dspark_ragged(self):
        with self.assertRaisesRegex(R.StructArgError, 'RAGGED'):
            R.resolve(dict(SGLANG_RAGGED_VERIFY_MODE='dynamic'), lora_paths=['a'], speculative_algorithm='DSPARK')

    def test_v2_skipped_for_lora_moe(self):
        R.resolve(lora_paths=['a'], lora_backend='triton_v2', enable_two_batch_overlap=True,
                  moe_runner_backend='lora_triton')
        with self.assertRaises(R.StructArgError):
            R.resolve(lora_paths=['a'], lora_backend='triton_v2', enable_two_batch_overlap=True)

    def test_uno_strict(self):
        with self.assertRaisesRegex(R.StructArgError, 'strict'):
            R.resolve(speculative_algorithm='UNO', uno_lora_path='/u', speculative_num_draft_tokens=4,
                      enable_strict_thinking=True)

    def test_cases_all_run(self):
        out = quiet(R.scenario_resolve)
        self.assertEqual(len(out), len(R.CASES))
        self.assertEqual(sum(isinstance(v, R.StructArgError) for v in out.values()), 12)

    def test_request_max_thinking_needs_strict(self):
        a, _, _ = R.resolve()
        self.assertTrue(R.request_fate(a, max_thinking_tokens=10).startswith('400'))

    def test_request_json_with_none(self):
        a, _, _ = R.resolve(grammar_backend='none')
        self.assertTrue(R.request_fate(a, json_schema=True).startswith('中止'))


if __name__ == '__main__':
    unittest.main()
