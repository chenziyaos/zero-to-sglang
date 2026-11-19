"""SGL-14 实验的边界用例：python3 -m unittest test_hicache -v"""
import hashlib
import io
import os
import struct
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import hicache_sim as H  # noqa: E402
import resolve_sim as R  # noqa: E402
from common.specs import GPUS, MODELS  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TestCost(unittest.TestCase):
    def test_load_linear(self):
        self.assertAlmostEqual(H.load_ms(2000, 1000, 1.0), 2 * H.load_ms(1000, 1000, 1.0))
        self.assertAlmostEqual(H.load_ms(1_000_000, 1000, 1.0), 1000.0)

    def test_prefill_attention_grows_with_ctx(self):
        m, c = MODELS['llama3.1-8b'], GPUS['h100-sxm']
        self.assertGreater(H.prefill_ms(m, c, 1, 2048, 32768), H.prefill_ms(m, c, 1, 2048, 0))

    def test_prefill_tp_splits(self):
        m, c = MODELS['llama3.1-70b'], GPUS['h100-sxm']
        self.assertAlmostEqual(H.prefill_ms(m, c, 1, 8192), 4 * H.prefill_ms(m, c, 4, 8192), places=6)

    def test_pipeline_bounds(self):
        for load, comp in ((100, 10), (10, 100), (50, 50)):
            p = H.pipeline_ms(load, comp, 32)
            self.assertGreaterEqual(p, max(load, comp) - 1e-9)
            self.assertLessEqual(p, load + comp + 1e-9)

    def test_pipeline_load_bound_tail(self):
        # 搬运远慢于计算：总时间 = 全部搬完 + 最后一层的计算
        self.assertAlmostEqual(H.pipeline_ms(320, 32, 32), 320 + 1)

    def test_l2_always_wins(self):
        r = quiet(H.scenario_cost)
        for model, v in r.items():
            self.assertEqual(v['breakeven']['L2 PCIe Gen5 x16'], 0, model)
            self.assertLess(v['ttft']['l2_overlap'], v['ttft']['full'] / 5, model)

    def test_nvme_breakeven_needs_long_prefix(self):
        r = quiet(H.scenario_cost)
        be = r['deepseek-v3']['breakeven']['L3 本地NVMe÷8']
        self.assertTrue(be is None or be > 32768)
        # 激活参数少、FP8 的 MoE：短前缀时重算比 3FS 还便宜
        self.assertGreater(r['deepseek-v3']['breakeven']['L3 3FS 40GiB/s÷8'], 0)
        self.assertEqual(r['llama3.1-8b']['breakeven']['L3 3FS 40GiB/s÷8'], 0)

    def test_overlap_not_worse_than_serial(self):
        r = quiet(H.scenario_cost)
        for v in r.values():
            self.assertLessEqual(v['ttft']['l2_overlap'], v['ttft']['l2_serial'] + 1e-9)
            self.assertLessEqual(v['ttft']['l3_overlap'], v['ttft']['l3_serial'] + 1e-9)


class TestTierSim(unittest.TestCase):
    def keys(self, sid, n):
        return [(sid, p) for p in range(n)]

    def test_gpu_only_drops_on_evict(self):
        s = H.TierSim(4, 0)
        s.request(self.keys(0, 4))
        s.request(self.keys(1, 4))
        self.assertEqual(s.st['dropped'], 4)
        self.assertEqual(s.request(self.keys(0, 4)), (0, 0, 0, 4))

    def test_write_through_demotes_to_host(self):
        s = H.TierSim(4, 8)
        s.request(self.keys(0, 4))
        self.assertEqual(s.st['d2h'], 4)               # 第一次命中就备份
        s.request(self.keys(1, 4))
        self.assertEqual(s.st['dropped'], 0)
        self.assertEqual(s.request(self.keys(0, 4)), (0, 4, 0, 0))

    def test_selective_needs_two_hits(self):
        s = H.TierSim(8, 16, policy='write_through_selective')
        s.request(self.keys(0, 2))
        self.assertEqual(s.st['d2h'], 0)
        s.request(self.keys(0, 2))
        self.assertEqual(s.st['d2h'], 2)

    def test_write_back_only_on_eviction(self):
        s = H.TierSim(4, 8, policy='write_back')
        s.request(self.keys(0, 4))
        s.request(self.keys(0, 4))
        self.assertEqual(s.st['d2h'], 0)
        s.request(self.keys(1, 4))
        self.assertEqual(s.st['d2h'], 4)
        self.assertEqual(s.request(self.keys(0, 4)), (0, 4, 0, 0))

    def test_host_equal_device_useless_for_write_through(self):
        # 新请求的备份把刚降级的主机副本挤掉，主机池只是显存的镜像
        s = H.TierSim(4, 4)
        s.request(self.keys(0, 4))
        s.request(self.keys(1, 4))
        self.assertEqual(s.st['d2h'], 8)
        self.assertEqual(len(s.host_only), 0)
        self.assertEqual(s.request(self.keys(0, 4))[1], 0)

    def test_write_back_effective_capacity(self):
        wt, wb = H.TierSim(4, 4), H.TierSim(4, 4, policy='write_back')
        for s in (wt, wb):
            s.request(self.keys(0, 4))
            s.request(self.keys(1, 4))
        self.assertEqual(wb.request(self.keys(0, 4)), (0, 4, 0, 0))
        self.assertEqual(wt.request(self.keys(0, 4))[1], 0)

    def test_prefix_order_device_then_host(self):
        s = H.TierSim(3, 8)
        s.request(self.keys(0, 3))
        s.request(self.keys(1, 2))       # 挤掉会话 0 最深的两页
        d, h, l3, miss = s.request(self.keys(0, 3))
        self.assertEqual((d, h), (1, 2))

    def test_l3_prefetch_threshold(self):
        s = H.TierSim(4, 64, storage=True, page=64, prefetch_threshold=256)
        s.l3.update(self.keys(9, 3))     # 3 页 = 192 token，不够 256
        self.assertEqual(s.request(self.keys(9, 3)), (0, 0, 0, 3))
        s2 = H.TierSim(8, 64, storage=True, page=64, prefetch_threshold=256)
        s2.l3.update(self.keys(9, 5))
        self.assertEqual(s2.request(self.keys(9, 5)), (0, 0, 5, 0))

    def test_l3_prefetch_limited_by_half_host(self):
        s = H.TierSim(32, 20, storage=True, page=64, prefetch_threshold=64)
        s.l3.update(self.keys(9, 16))
        d, h, l3, miss = s.request(self.keys(9, 16))
        self.assertEqual(l3, 10)
        self.assertEqual(s.st['l3_host_capacity'], 6)

    def test_multiturn_ordering(self):
        r = quiet(H.scenario_multiturn)
        hit = {k: 1 - v['st']['miss'] / v['n'] for k, v in r.items()}
        self.assertAlmostEqual(hit['只有 GPU'], hit['GPU+主机 ratio1'], places=3)   # 多补的一页
        self.assertGreater(hit['GPU+主机 wt'], hit['只有 GPU'])
        self.assertGreaterEqual(hit['GPU+主机 write_back'], hit['GPU+主机 wt'])
        self.assertGreater(hit['GPU+主机+L3 wt'], hit['GPU+主机 write_back'])
        self.assertLess(r['GPU+主机+L3 wt']['prefill_s'], r['只有 GPU']['prefill_s'])


class TestPrefetch(unittest.TestCase):
    def test_timeout_formula(self):
        self.assertAlmostEqual(H.prefetch_timeout_s(512), 1.0 + 512 * 64 / 1024 * 0.25)
        self.assertAlmostEqual(H.prefetch_timeout_s(10, page=1, base=2, per_ki=0.1), 2 + 10 / 1024 * 0.1)

    def test_too_short(self):
        p = H.prefetch_plan(255, 70000, 1.0, 'wait_complete', 0.0)
        self.assertEqual((p['fetched'], p['reason']), (0, 'declined_too_short'))
        p = H.prefetch_plan(300, 70000, 1.0, 'wait_complete', 0.0)   # 页对齐成 256
        self.assertEqual(p['fetched'], 256)

    def test_policies_order(self):
        for pol in ('best_effort', 'timeout', 'wait_complete'):
            p = H.prefetch_plan(32768, 70272, 100.0, pol, 0.5)
            self.assertEqual(p['fetched'], 32768)                    # 排队已经盖住了
        be = H.prefetch_plan(32768, 70272, 0.1, 'best_effort', 0.2)
        to = H.prefetch_plan(32768, 70272, 0.1, 'timeout', 0.2)
        wc = H.prefetch_plan(32768, 70272, 0.1, 'wait_complete', 0.2)
        self.assertLessEqual(be['admit_s'], to['admit_s'])
        self.assertLessEqual(to['admit_s'], wc['admit_s'])
        self.assertEqual(to['reason'], 'deadline')
        self.assertAlmostEqual(to['admit_s'], 0.2 + to['timeout_s'])

    def test_best_effort_whole_batches(self):
        p = H.prefetch_plan(65536, 70272, 1.0, 'best_effort', 0.7)
        self.assertEqual(p['fetched'] % (H.STORAGE_BATCH_PAGES * H.PAGE), 0)

    def test_suggest_per_ki_small(self):
        r = quiet(H.scenario_prefetch)
        self.assertLess(r['suggest_per_ki'], 0.25)


class TestHash(unittest.TestCase):
    def test_matches_manual_sha(self):
        toks = list(range(10))
        h = H.page_hashes(toks, page=4)
        d0 = hashlib.sha256(struct.pack('<4I', 0, 1, 2, 3)).digest()
        self.assertEqual(h[0], d0.hex())
        d1 = hashlib.sha256(d0 + struct.pack('<4I', 4, 5, 6, 7)).hexdigest()
        self.assertEqual(h[1], d1)
        self.assertEqual(len(h), 3)                                   # 最后 2 个 token 也成一页

    def test_chain_divergence(self):
        a = H.page_hashes([1] * 8 + [2] * 4, page=4)
        b = H.page_hashes([1] * 8 + [3] * 4, page=4)
        self.assertEqual(a[:2], b[:2])
        self.assertNotEqual(a[2], b[2])

    def test_namespace(self):
        self.assertIsNone(H.namespace_seed())
        self.assertNotEqual(H.namespace_seed('', None), H.namespace_seed(None, ''))
        self.assertNotEqual(H.namespace_seed('ab', 'c'), H.namespace_seed('a', 'bc'))
        t = list(range(8))
        self.assertNotEqual(H.page_hashes(t, 4)[0], H.page_hashes(t, 4, cache_salt='x')[0])

    def test_file_key(self):
        self.assertEqual(H.file_key('ab', 'org/m', 1, 2), 'ab_org-m_1_2.bin')
        self.assertEqual(H.file_key('ab', 'org/m', 1, 2, is_mla=True), 'ab_org-m.bin')
        self.assertEqual(H.file_key('ab', 'm', 0, 4, pp=(1, 2)), 'ab_m_0_4_2_1.bin')


class TestResolve(unittest.TestCase):
    def test_defaults(self):
        a, w = R.resolve(enable_hierarchical_cache=True)
        self.assertEqual((a['hicache_ratio'], a['hicache_mem_layout'], a['hicache_io_backend']),
                         (2.0, 'page_first', 'kernel'))
        self.assertEqual(w, [])

    def test_buffer_only(self):
        with self.assertRaises(R.HiCacheArgError):
            R.resolve(enable_hierarchical_cache=True, hicache_host_memory_mode='buffer_only')
        a, w = R.resolve(enable_hierarchical_cache=True, hicache_host_memory_mode='buffer_only',
                         hicache_storage_backend='file', hicache_write_policy='write_back')
        self.assertEqual((a['hicache_ratio'], a['hicache_write_policy']), (1.2, 'write_through'))

    def test_layout_io(self):
        a, w = R.resolve(enable_hierarchical_cache=True, hicache_mem_layout='page_first_direct')
        self.assertEqual(a['hicache_io_backend'], 'direct')
        a, w = R.resolve(enable_hierarchical_cache=True, hicache_io_backend='direct')
        self.assertEqual(a['hicache_mem_layout'], 'page_first_direct')
        self.assertIn('page first direct layout', w[0])

    def test_mooncake_layer_first(self):
        a, _ = R.resolve(enable_hierarchical_cache=True, hicache_storage_backend='mooncake',
                         hicache_mem_layout='layer_first')
        self.assertEqual(a['hicache_mem_layout'], 'page_first')
        a, _ = R.resolve(enable_hierarchical_cache=True, hicache_storage_backend='hf3fs',
                         hicache_mem_layout='layer_first')
        self.assertEqual(a['hicache_mem_layout'], 'layer_first')
        a, w = R.resolve(enable_hierarchical_cache=True, hicache_storage_backend='mooncake',
                         hicache_mem_layout='layer_first', dcp_size=2, use_mla=True)
        self.assertEqual((a['hicache_mem_layout'], w), ('layer_first', []))

    def test_explicit_disables_auto(self):
        a, _ = R.resolve(enable_hierarchical_cache=True, hicache_size=30)
        self.assertIsNone(a['hicache_host_memory_fraction'])
        a, _ = R.resolve(enable_hierarchical_cache=True, hicache_ratio=3.0)
        self.assertEqual((a['hicache_ratio'], a['hicache_host_memory_fraction']), (3.0, None))

    def test_decode(self):
        a, _ = R.resolve(disaggregation_mode='decode')
        self.assertIsNone(a['hicache_ratio'])
        with self.assertRaises(R.HiCacheArgError):
            R.resolve(disaggregation_mode='decode', enable_hierarchical_cache=True,
                      hicache_host_memory_mode='buffer_only', hicache_storage_backend='file')
        a, _ = R.resolve(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=0.5)
        self.assertEqual(a['hicache_mem_layout'], 'layer_first')
        with self.assertRaises(R.HiCacheArgError):
            R.resolve(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=0.5,
                      hicache_storage_backend='mooncake')

    def test_decode_ratio(self):
        self.assertEqual(R.decode_ratio('host_pool', True, 0, 32768, 400000), 2.0)
        self.assertAlmostEqual(R.decode_ratio('host_pool', False, 0, 32768, 400000), 0.2)
        r = R.decode_ratio('host_pool', False, 0, 131072, 400000)
        self.assertAlmostEqual(r, 131072 / 400000)
        self.assertAlmostEqual(R.decode_ratio('host_pool', False, 0, 131072, 400000, receive_threshold=0.5),
                               2 * 131072 / 400000)

    def test_dcp(self):
        with self.assertRaises(NotImplementedError):
            R.resolve(enable_hierarchical_cache=True, dcp_size=2)
        R.resolve(enable_hierarchical_cache=True, dcp_size=2, use_mla=True, speculative_algorithm='DSPARK')

    def test_inactive_skips(self):
        a, w = R.resolve(hicache_mem_layout='page_first_direct')
        self.assertEqual((a['hicache_io_backend'], w), ('kernel', []))

    def test_fraction_range(self):
        for f in (0, 1.01, -0.1):
            with self.assertRaises(R.HiCacheArgError):
                R.resolve(enable_hierarchical_cache=True, hicache_host_memory_fraction=f)


class TestSizing(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(R.host_budget_bytes(90 * R.GiB, 8), 10 * R.GiB)

    def test_auto_ratio_caps(self):
        dev = 40 * R.GiB
        r, b = R.auto_ratio(2.0, dev, 2048 * R.GiB, 8)
        self.assertEqual(r, 2.0)
        r, b = R.auto_ratio(2.0, dev, 256 * R.GiB, 8)
        self.assertAlmostEqual(r, int((246 * R.GiB) // 8 * 0.8) * 0.95 / dev)
        with self.assertRaises(R.HiCacheArgError):
            R.auto_ratio(2.0, dev, 8 * R.GiB, 8)
        self.assertEqual(R.auto_ratio(3.0, dev, 1, 8, fraction=None), (3.0, None))

    def test_host_pool_padding(self):
        size, warn = R.host_pool_tokens(1000, 100, 64, ratio=2.0)
        self.assertEqual(size, (2000 // 64 + 1) * 64)
        self.assertIsNone(warn)
        size, warn = R.host_pool_tokens(1000, 100, 64, ratio=0.5)
        self.assertIsNotNone(warn)

    def test_hicache_size_units(self):
        size, _ = R.host_pool_tokens(10, 1000, 1, hicache_size_gb=1)
        self.assertEqual(size, 1_000_001)          # 1e9 字节 / 1000 B，再补一页

    def test_check_alloc(self):
        with self.assertRaises(R.HiCacheArgError):
            R.check_alloc(1000, 1000, 999_999)
        self.assertEqual(R.check_alloc(1000, 1000, 1_000_000), 1_000_000)

    def test_scenarios_run(self):
        out = quiet(R.scenario_resolve)
        self.assertIsInstance(out['DCP + MHA 模型'], NotImplementedError)
        res = quiet(R.scenario_size, host_gib=512)
        ratio = res['默认（ratio 2，fraction 0.8）'][0]
        self.assertLess(ratio, 1.0)


if __name__ == '__main__':
    unittest.main()
