"""SGL-15 实验的边界用例：python3 -m unittest test_pd -v"""
import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import pd_sim as P  # noqa: E402
import resolve_sim as R  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TestContiguous(unittest.TestCase):
    def test_both_contiguous(self):
        self.assertEqual(P.group_contiguous([5, 6, 7, 8], [100, 101, 102, 103]), [(5, 100, 4)])

    def test_dst_breaks(self):
        self.assertEqual(P.group_contiguous([5, 6, 7, 8], [100, 101, 200, 201]), [(5, 100, 2), (7, 200, 2)])

    def test_src_breaks(self):
        self.assertEqual(P.group_contiguous([1, 2, 9], [3, 4, 5]), [(1, 3, 2), (9, 5, 1)])

    def test_empty(self):
        self.assertEqual(P.group_contiguous([], []), [])

    def test_unequal_raises(self):
        with self.assertRaises(ValueError):
            P.group_contiguous([1, 2], [1])

    def test_lengths_sum(self):
        rng = random.Random(1)
        src = sorted(rng.sample(range(1000), 200))
        dst = sorted(rng.sample(range(1000), 200))
        self.assertEqual(sum(n for _, _, n in P.group_contiguous(src, dst)), 200)


class TestRankMapping(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(P.rank_mapping(3, 4, 4)[:4], ([3], 3, 1, 1))
        self.assertIsNone(P.rank_mapping(3, 4, 4)[4])

    def test_decode_larger(self):
        ranks, t, dst_info, resp, warn = P.rank_mapping(5, 8, 2)
        self.assertEqual((ranks, t, dst_info, resp), ([1], 1, 4, 1))
        self.assertIn('NOT guaranteed', warn)

    def test_decode_smaller_mha(self):
        ranks, t, _, resp, _ = P.rank_mapping(1, 2, 8)
        self.assertEqual(ranks, [4, 5, 6, 7])
        self.assertEqual((t, resp), (4, 4))

    def test_decode_smaller_mla_one_response(self):
        ranks, _, _, resp, warn = P.rank_mapping(0, 1, 8, mla=True)
        self.assertEqual(len(ranks), 8)
        self.assertEqual(resp, 1)
        self.assertIsNone(warn)

    def test_mla_sharding_needs_equal(self):
        with self.assertRaises(AssertionError):
            P.rank_mapping(0, 4, 8, mla=True, mla_sharding=True)


class TestHeadsAndStaging(unittest.TestCase):
    def test_head_slice_p_bigger(self):
        # 8 头，P TP4 -> D TP2：每个 P rank 2 头，落到 D rank 本地的 0 或 2
        self.assertEqual(P.head_slice(4, 2, 0, 0, 8), (0, 2, 0, 2))
        self.assertEqual(P.head_slice(4, 2, 1, 0, 8), (0, 2, 2, 2))

    def test_head_slice_d_bigger(self):
        self.assertEqual(P.head_slice(2, 4, 0, 1, 8), (2, 2, 0, 2))

    def test_staging_writers(self):
        n, wb, tot = P.staging_layout(4, 2, 0, 8, 1000, 256, 32)
        self.assertEqual(n, 2)
        self.assertEqual(wb[0], wb[1])
        self.assertEqual(tot, sum(wb))

    def test_staging_equal_tp_one_writer(self):
        self.assertEqual(P.staging_layout(2, 2, 0, 8, 10, 256, 2)[0], 1)

    def test_grid(self):
        self.assertEqual(P.grid_tokens(8192, 64), 8192)
        self.assertEqual(P.grid_tokens(1000, 64), 960)
        self.assertEqual(P.grid_segments(0, 20000, 0, 8192), [(0, 8192), (8192, 16384), (16384, 20000)])
        self.assertEqual(P.grid_segments(5, 5, 0, 8192), [(5, 5)])


class TestSendBoundaries(unittest.TestCase):
    def test_middle_chunk_drops_partial_page(self):
        self.assertEqual(P.send_end(0, 10192, 20000, 64, last=False), 10176)

    def test_last_chunk_sends_all(self):
        self.assertEqual(P.send_end(18368, 20000, 20000, 64, last=True), 20000)

    def test_skip_when_behind_start(self):
        self.assertIsNone(P.send_end(100, 120, 20000, 64, last=False))

    def test_staging_aligns_to_grid(self):
        self.assertEqual(P.send_end(0, 10192, 20000, 64, last=False, staging=True, chunk=8192), 8192)

    def test_early_send_needs_page_alignment(self):
        self.assertIsNone(P.early_send_end(2000, 0, 0, 64))
        self.assertEqual(P.early_send_end(1984, 0, 0, 64), 1984)

    def test_early_send_excludes_host_hit(self):
        self.assertIsNone(P.early_send_end(2048, 2048, 0, 64))
        self.assertEqual(P.early_send_end(4096, 2048, 0, 64), 2048)

    def test_early_send_waits_bootstrap(self):
        self.assertIsNone(P.early_send_end(1984, 0, 0, 64, pending_bootstrap=True))

    def test_kvpoll_min(self):
        self.assertEqual(P.kvpoll_min([4, 4, 3, 4]), 3)
        self.assertEqual(P.POLL[P.kvpoll_min([4, 0, 4])], 'Failed')


class TestAllocator(unittest.TestCase):
    def test_fresh_contiguous(self):
        a = P.TokenAllocator(100)
        self.assertEqual(a.alloc(5), [1, 2, 3, 4, 5])

    def test_need_sort_merges_when_short(self):
        a = P.TokenAllocator(10, need_sort=True)
        x = a.alloc(6)
        a.free(x[::-1])
        self.assertEqual(a.alloc(8), list(range(1, 9)))

    def test_no_sort_keeps_order(self):
        a = P.TokenAllocator(10, need_sort=False)
        x = a.alloc(6)
        a.free(x[::-1])
        self.assertEqual(a.alloc(6), [7, 8, 9, 10, 6, 5])

    def test_sort_fewer_runs(self):
        s, _ = P.runs_after_churn(True, trials=10)
        u, _ = P.runs_after_churn(False, trials=10)
        self.assertLess(s, u)

    def test_descriptors(self):
        self.assertEqual(P.descriptors('llama3.1-8b', 1, runs=3), 3 * 64)
        self.assertEqual(P.descriptors('deepseek-v3', 8, runs=1), 61)
        self.assertEqual(P.descriptors('llama3.1-8b', 2, tokens=10, mode='slice'), 640)


class TestTransfer(unittest.TestCase):
    def test_pipeline_tail_is_last_chunk(self):
        srv = P.Server('llama3.1-70b', 'h100-sxm', 4)
        cell = P.ledger('llama3.1-70b', 'h100-sxm', 4)['cell']
        t, link, tail = P.chunk_pipeline(srv, 32768, cell, 40.0)
        self.assertAlmostEqual(tail, srv.chunk * cell / 40e9, places=6)
        self.assertGreater(link, t)

    def test_slow_link_piles_up(self):
        srv = P.Server('llama3.1-8b', 'h100-sxm', 1)
        cell = P.ledger('llama3.1-8b', 'h100-sxm', 1)['cell']
        _, _, tail = P.chunk_pipeline(srv, 32768, cell, 1.2)
        self.assertGreater(tail, srv.chunk * cell / 1.2e9 * 1.5)

    def test_transfer_scenario(self):
        r = quiet(P.scenario_transfer, 32768)
        self.assertGreater(r['desc']['slice'], 100 * r['desc']['fresh'])
        t = r['tails']
        self.assertAlmostEqual(t[32768][1], t[131072][1], places=3)


class TestCapacity(unittest.TestCase):
    def test_ratio_scenario(self):
        r = quiet(P.scenario_ratio)
        self.assertGreater(r['RAG 摘要']['ratio'], r['聊天']['ratio'])
        self.assertLess(r['代码补全长输出']['ratio'], 1.0)
        self.assertGreater(r['聊天']['pd'], r['聊天']['colo'])

    def test_decode_capacity_memory_bound(self):
        srv = P.Server('llama3.1-8b', 'h100-sxm', 1)
        b, _, _ = P.decode_capacity(srv, 65536, 512, 1.0, srv.cap)
        self.assertLessEqual(b * (65536 + 512 + P.RESERVED), srv.cap)


class TestColo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.long_rdma = P.compare('long', link='RDMA 400G/卡')
        cls.long_tcp = P.compare('long', link='TCP 100GbE÷8')
        cls.short = P.compare('short', link='RDMA 400G/卡')

    def test_pd_kills_itl_spikes(self):
        co, pd = self.long_rdma
        self.assertLess(pd['itl_max'], co['itl_max'] / 10)
        self.assertLess(pd['itl_p99'], co['itl_p99'])

    def test_short_prompts_pd_loses(self):
        co, pd = self.short
        self.assertGreater(co['tps'], pd['tps'])
        self.assertGreater(pd['itl_p50'], co['itl_p50'])

    def test_tcp_pd_loses_ttft(self):
        co, pd = self.long_tcp
        self.assertGreater(pd['ttft_p50'], 10 * co['ttft_p50'])

    def test_d_pool_too_small_raises(self):
        reqs = [P.PDReq(0, 'long', 500_000, 10, 0.0)]
        with self.assertRaises(ValueError):
            P.run_pd(reqs, 'llama3.1-8b', 'h100-sxm', 1, 40.0)


class TestAdmission(unittest.TestCase):
    def test_short_output_reserves_512(self):
        a = P.DecodeAdmission(10_000)
        self.assertEqual(a.required(100, 100, 0, 10, 0), 612)

    def test_long_output_uses_max_new(self):
        a = P.DecodeAdmission(10_000)
        self.assertEqual(a.required(100, 100, 0, 3000, 0), 3100)

    def test_clip(self):
        a = P.DecodeAdmission(10 ** 6)
        self.assertEqual(a.required(100, 100, 0, 10 ** 6, 0), 100 + P.CLIP)

    def test_hol(self):
        a = P.DecodeAdmission(90_000)
        n, why, _ = a.pop_preallocated([(100_000, 256)] + [(2000, 256)] * 10)
        self.assertEqual(n, 0)
        self.assertIn('100000', why)
        b = P.DecodeAdmission(90_000)
        self.assertEqual(b.pop_preallocated([(2000, 256)] * 10 + [(100_000, 256)])[0], 10)

    def test_waiting_queue_shrinks_budget(self):
        pool = 415_401
        running = [(1500, 500, 2048)] * 100
        n0, _ = P.max_admit(pool, 4096, 512, running, waiting=0)
        n1, _ = P.max_admit(pool, 4096, 512, running, waiting=300)
        self.assertGreater(n0, n1)

    def test_retracted_first(self):
        a = P.DecodeAdmission(100_000)
        a.retracted = [30_000] * 3
        self.assertEqual(a.pop_preallocated([(4096, 512)] * 20)[0], 2)

    def test_req_slots(self):
        a = P.DecodeAdmission(10 ** 6, req_slots=3)
        n, why, _ = a.pop_preallocated([(10, 10)] * 10)
        self.assertEqual(n, 3)
        self.assertIn('req_to_token_pool', why)


class TestTimeline(unittest.TestCase):
    def test_timeline(self):
        r = quiet(P.scenario_timeline)
        self.assertEqual([s[1] for s in r['sends']], [10176, 18368, 20000])
        self.assertEqual(r['staging_end'], 8192)
        e0, l0 = r['policy'][0.0]
        e10, l10 = r['policy'][10.0]
        self.assertAlmostEqual(e10 - e0, 10.0)
        self.assertEqual(l0, l10)


class TestResolve(unittest.TestCase):
    def test_decode_disables_radix(self):
        a, _, w = R.resolve(disaggregation_mode='decode')
        self.assertTrue(a['disable_radix_cache'])
        self.assertTrue(any('disabled' in x for x in w))

    def test_prefill_untouched(self):
        a, env, w = R.resolve(disaggregation_mode='prefill')
        self.assertFalse(a['disable_radix_cache'])
        self.assertEqual((env, w), ({}, []))

    def test_extra_slots(self):
        self.assertEqual(R.resolve(disaggregation_mode='decode', max_running_requests=16)[0]
                         ['disaggregation_decode_extra_slots'], 32)
        self.assertEqual(R.resolve(disaggregation_mode='decode', max_running_requests=33)[0]
                         ['disaggregation_decode_extra_slots'], 0)
        self.assertEqual(R.resolve(disaggregation_mode='decode', max_running_requests=256, dp_size=8,
                                   enable_dp_attention=True)[0]['disaggregation_decode_extra_slots'], 64)

    def test_explicit_extra_slots_kept(self):
        a, _, _ = R.resolve(disaggregation_mode='decode', max_running_requests=8, disaggregation_decode_extra_slots=0)
        self.assertEqual(a['disaggregation_decode_extra_slots'], 0)

    def test_mooncake_tcp(self):
        a, env, _ = R.resolve(disaggregation_mode='prefill', disaggregation_transfer_backend='mooncake_tcp',
                              disaggregation_ib_device='mlx5_0')
        self.assertEqual(a['disaggregation_transfer_backend'], 'mooncake')
        self.assertIsNone(a['disaggregation_ib_device'])
        self.assertEqual(env, {'MC_FORCE_TCP': '1'})

    def test_prefill_complete(self):
        a, _, _ = R.resolve(disaggregation_mode='prefill', disaggregation_decode_allocation_policy='prefill_complete')
        self.assertEqual(a['optimistic_prefill_attempts'], 1)
        with self.assertRaises(R.PDArgError):
            R.resolve(disaggregation_mode='prefill', disaggregation_decode_allocation_policy='prefill_complete',
                      tp_size=4)
        a, _, _ = R.resolve(disaggregation_mode='prefill', disaggregation_decode_allocation_policy='prefill_complete',
                            tp_size=8, dp_size=8, enable_dp_attention=True)
        self.assertEqual(a['optimistic_prefill_attempts'], 1)

    def test_prefill_complete_lists_all(self):
        with self.assertRaises(R.PDArgError) as cm:
            R.resolve({'SGLANG_DISAGG_STAGING_BUFFER': True}, disaggregation_mode='decode',
                      disaggregation_decode_allocation_policy='prefill_complete',
                      disaggregation_decode_enable_radix_cache=True, disaggregation_transfer_backend='nixl')
        msg = str(cm.exception)
        for part in ('Mooncake', 'decode radix', 'staging'):
            self.assertIn(part, msg)

    def test_staging_backend(self):
        with self.assertRaises(R.PDArgError):
            R.resolve({'SGLANG_DISAGG_STAGING_BUFFER': True}, disaggregation_mode='decode',
                      disaggregation_transfer_backend='mori')
        R.resolve({'SGLANG_DISAGG_STAGING_BUFFER': True}, disaggregation_mode='decode',
                  disaggregation_transfer_backend='nixl')

    def test_fake_prefill(self):
        with self.assertRaises(R.PDArgError):
            R.resolve(disaggregation_mode='prefill', disaggregation_transfer_backend='fake')

    def test_role_switch(self):
        with self.assertRaises(R.PDArgError) as cm:
            R.resolve(disaggregation_mode='decode', enable_pd_role_switch=True, speculative_algorithm='EAGLE')
        self.assertIn('speculative', str(cm.exception))

    def test_host_receive(self):
        a, _, _ = R.resolve(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=0.5)
        self.assertEqual(a['disaggregation_decode_retraction_backup'], 'host_pool')
        with self.assertRaises(R.PDArgError):
            R.resolve(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=1.5)
        with self.assertRaises(R.PDArgError):
            R.resolve(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=0.5,
                      disaggregation_decode_retraction_backup='cpu_tensor')

    def test_unknown(self):
        with self.assertRaises(TypeError):
            R.resolve(foo=1)

    def test_threads(self):
        self.assertEqual(R.thread_pool(128), (8, 12))
        self.assertEqual(R.thread_pool(32), (4, 4))
        self.assertEqual(R.thread_pool(1024)[0], 12)
        with self.assertRaises(R.PDArgError):
            R.thread_pool(128, env_override=2)

    def test_scenarios_run(self):
        out = quiet(R.scenario_resolve)
        self.assertIsInstance(out['P fake 后端'], R.PDArgError)
        self.assertIsInstance(out['D 默认'], tuple)


if __name__ == '__main__':
    unittest.main()
