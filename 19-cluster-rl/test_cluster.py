"""SGL-19 实验的边界用例：python3 -m unittest test_cluster -v"""
import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rollout_sim as RS  # noqa: E402
import sim_router as SR  # noqa: E402


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def workers(n=4):
    srv = SR.sim_serve.Server('qwen3-8b', 'h100-sxm')
    return [SR.Worker(i, srv) for i in range(n)]


SMALL = None


def small():
    """测试用的小负载：每秒 3 个会话、60 s，几百条请求，跑一次不到 1 秒。"""
    global SMALL
    if SMALL is None:
        SMALL = SR.make_workload(rate=3.0, duration=60.0)
    return SMALL


# ---------------------------------------------------------------- 哈希环
class TestRing(unittest.TestCase):
    URLS = [f'http://worker{i}:30000' for i in range(9)]
    KEYS = [f'session-{i}' for i in range(5000)]

    def moved(self, a, b):
        r1, r2 = SR.HashRing(self.URLS[:a]), SR.HashRing(self.URLS[:b])
        return sum(1 for k in self.KEYS if r1.find(k) != r2.find(k)) / len(self.KEYS)

    def test_add_moves_about_one_over_n(self):
        self.assertAlmostEqual(self.moved(4, 5), 0.2, delta=0.05)
        self.assertAlmostEqual(self.moved(8, 9), 1 / 9, delta=0.04)

    def test_mod_moves_most_keys(self):
        mod = sum(1 for k in self.KEYS if SR.h64(k.encode()) % 4 != SR.h64(k.encode()) % 5) / len(self.KEYS)
        self.assertGreater(mod, 0.7)

    def test_removed_keys_only_move_from_removed_worker(self):
        r4, r3 = SR.HashRing(self.URLS[:4]), SR.HashRing(self.URLS[:3])
        for k in self.KEYS[:1000]:
            u = r4.find(k)
            if u != self.URLS[3]:
                self.assertEqual(u, r3.find(k))

    def test_find_skips_unhealthy(self):
        r = SR.HashRing(self.URLS[:4])
        k = 'session-42'
        first = r.find(k)
        second = r.find(k, lambda u: u != first)
        self.assertNotEqual(first, second)
        self.assertIsNotNone(second)
        self.assertIsNone(r.find(k, lambda u: False))

    def test_vnodes_balance(self):
        def ratio(vn):
            r = SR.HashRing(self.URLS[:4], vn)
            c = {}
            for k in self.KEYS:
                u = r.find(k)
                c[u] = c.get(u, 0) + 1
            return max(c.values()) / (len(self.KEYS) / 4)
        self.assertLess(ratio(150), 1.15)
        self.assertGreater(ratio(1), ratio(150))

    def test_empty_ring(self):
        self.assertIsNone(SR.HashRing([]).find('x'))


# ---------------------------------------------------------------- 副本上的前缀缓存
class TestPrefixCache(unittest.TestCase):
    def test_match_stops_at_first_miss_and_limit(self):
        c = SR.PrefixCache()
        c.insert([1, 2, 3], 0.0)
        self.assertEqual(c.match([1, 2, 3, 4], 10), 3)
        self.assertEqual(c.match([1, 9, 3], 10), 1)
        self.assertEqual(c.match([1, 2, 3], 2), 2)
        self.assertEqual(c.tokens, 3 * SR.BLK)

    def test_locked_blocks_survive_eviction(self):
        c = SR.PrefixCache()
        c.insert([1, 2, 3], 0.0)
        c.touch_lock([1, 2, 3], 3, 1.0, +1)
        self.assertEqual(c.evict(10 ** 6), 0)
        c.touch_lock([1, 2, 3], 3, 2.0, -1)
        self.assertEqual(c.evict(10 ** 6), 3 * SR.BLK)

    def test_evicts_leaf_before_parent_and_lru(self):
        c = SR.PrefixCache()
        c.insert([1, 2], 0.0)
        c.insert([1, 3], 5.0)
        c.evict(SR.BLK)
        self.assertNotIn(2, c.b)                # 更老的叶子先走
        self.assertIn(1, c.b)                   # 还有孩子的父节点不动
        c.evict(2 * SR.BLK)
        self.assertEqual(c.b, {})


# ---------------------------------------------------------------- 网关的近似树
class TestApproxTree(unittest.TestCase):
    def test_last_tenant_is_the_creator(self):
        t = SR.ApproxTree()
        t.insert([1, 2, 3], 'A', 0.0)
        t.insert([1, 2, 3, 4], 'B', 1.0)
        self.assertEqual(t.match([1, 2, 3]), (3, 'A'))
        self.assertEqual(t.match([1, 2, 3, 4, 5]), (4, 'B'))

    def test_remove_tenant_falls_back(self):
        t = SR.ApproxTree()
        t.insert([1, 2], 'A', 0.0)
        t.insert([1, 2], 'B', 1.0)
        t.remove_tenant('A')
        self.assertEqual(t.match([1, 2]), (2, 'B'))
        t.remove_tenant('B')
        self.assertEqual(t.match([1, 2])[0], 0)

    def test_size_counts_per_tenant(self):
        t = SR.ApproxTree()
        t.insert([1, 2, 3], 'A', 0.0)
        t.insert([1, 2, 9], 'B', 0.0)
        self.assertEqual(t.size['A'], 3 * SR.BLK)
        self.assertEqual(t.size['B'], 3 * SR.BLK)

    def test_evict_oldest_leaves_of_oversized_tenant(self):
        t = SR.ApproxTree()
        t.insert([1, 2, 3], 'A', 0.0)
        t.insert([1, 5, 6], 'A', 9.0)
        t.insert([7], 'B', 0.0)
        t.evict(4 * SR.BLK)
        self.assertLessEqual(t.size['A'], 4 * SR.BLK)
        self.assertEqual(t.match([1, 5, 6]), (3, 'A'))     # 新的留下
        self.assertLess(t.match([1, 2, 3])[0], 3)            # 老的被剪
        self.assertEqual(t.size['B'], SR.BLK)                # 没超限的租户不动


# ---------------------------------------------------------------- 策略
class FakeReq:
    def __init__(self, blocks, inp=None):
        self.blocks = blocks
        self.inp = inp or len(blocks) * SR.BLK
        self.routed_match = 0.0


class TestPolicies(unittest.TestCase):
    def test_round_robin_skips_unhealthy(self):
        w = workers(3)
        w[1].healthy = False
        p = SR.RoundRobin(w, random.Random(0))
        self.assertEqual([p.select(None, None) for _ in range(4)], [0, 2, 0, 2])

    def test_power_of_two_prefers_polled_tokens(self):
        w = workers(2)
        p = SR.PowerOfTwo(w, random.Random(0))
        w[0].load, w[1].load = 0, 50
        p.cached = {0: 9000, 1: 100}
        self.assertEqual(p.select(None, None), 1)

    def test_power_of_two_falls_back_to_inflight_and_first_on_tie(self):
        w = workers(2)
        p = SR.PowerOfTwo(w, random.Random(0))
        p.cached = {0: 100, 1: 100}            # token 相等，比在途数
        w[0].load, w[1].load = 7, 3
        self.assertEqual(p.select(None, None), 1)
        w[0].load = w[1].load = 3              # 再相等取先抽到的那台
        rng = random.Random(5)
        i1 = rng.randrange(2)
        p.rng = random.Random(5)
        self.assertEqual(p.select(None, None), i1)

    def test_prefix_hash_load_ok(self):
        p = SR.PrefixHash(workers(4), random.Random(0))
        self.assertTrue(p.load_ok(100, 0, 4))
        self.assertTrue(p.load_ok(6, 20, 4))     # (20 + 1) / 4 x 1.25 = 6.56
        self.assertFalse(p.load_ok(7, 20, 4))

    def test_cache_aware_follows_match(self):
        w = workers(3)
        p = SR.CacheAware(w, random.Random(0))
        p.tree.insert([1, 2, 3, 4], w[2].url, 0.0)
        self.assertEqual(p.select(FakeReq([1, 2, 3, 4]), None), 2)
        self.assertEqual(p.branch['hit'], 1)

    def test_cache_aware_low_match_goes_min_load(self):
        w = workers(3)
        w[0].load, w[1].load, w[2].load = 5, 1, 5
        p = SR.CacheAware(w, random.Random(0))
        p.tree.insert([1], w[2].url, 0.0)
        self.assertEqual(p.select(FakeReq([1, 8, 9, 10]), None), 1)    # 匹配率 25% < 0.3
        self.assertEqual(p.branch['low_match'], 1)

    def test_cache_aware_imbalance_overrides_match(self):
        w = workers(3)
        w[0].load, w[1].load, w[2].load = 100, 10, 100
        p = SR.CacheAware(w, random.Random(0))
        p.tree.insert([1, 2], w[2].url, 0.0)
        self.assertEqual(p.select(FakeReq([1, 2]), None), 1)
        self.assertEqual(p.branch['imbalanced'], 1)

    def test_cache_aware_ignores_dead_worker_load(self):
        w = workers(4)
        for x in w:
            x.load = 100
        w[0].healthy, w[0].load = False, 0     # 摘掉的台在途数归 0，不该触发失衡
        p = SR.CacheAware(w, random.Random(0))
        p.tree.insert([1, 2], w[3].url, 0.0)
        self.assertEqual(p.select(FakeReq([1, 2]), None), 3)
        self.assertEqual(p.branch['imbalanced'], 0)

    def test_cache_aware_stale_tenant_removed(self):
        w = workers(3)
        p = SR.CacheAware(w, random.Random(0))
        p.tree.insert([1, 2], w[1].url, 0.0)
        w[1].healthy = False
        self.assertEqual(p.select(FakeReq([1, 2]), None), 0)            # 退回第一台健康的
        self.assertNotIn(w[1].url, p.tree.size)


# ---------------------------------------------------------------- 端到端
class TestRun(unittest.TestCase):
    def test_workload_deterministic(self):
        a = SR.make_workload(rate=3.0, duration=20.0)
        b = SR.make_workload(rate=3.0, duration=20.0)
        self.assertEqual([[r.inp for r in s.reqs] for s in a], [[r.inp for r in s.reqs] for s in b])
        self.assertEqual(a[0].reqs[-1].blocks[:5], b[0].reqs[-1].blocks[:5])

    def test_later_turns_extend_earlier_prefix(self):
        s = next(x for x in small() if len(x.reqs) >= 2)
        r0, r1 = s.reqs[0], s.reqs[1]
        self.assertEqual(r1.blocks[:len(r0.blocks)], r0.blocks)

    def test_every_request_finishes(self):
        for p in SR.POLICIES:
            R = SR.run(p, sessions=small())
            self.assertEqual(R['n'], sum(len(s.reqs) for s in small()))
            self.assertTrue(0 <= R['hit'] < 1)

    def test_affinity_beats_round_robin(self):
        rr = SR.run('round_robin', sessions=small())
        ch = SR.run('consistent_hashing', sessions=small())
        ca = SR.run('cache_aware', sessions=small())
        self.assertGreater(ch['hit'], rr['hit'] + 0.1)
        self.assertGreater(ca['hit'], rr['hit'] + 0.1)
        self.assertEqual(ch['sessions_split'], 0.0)

    def test_remove_retries_and_never_routes_to_dead(self):
        sess = small()
        R = SR.run('consistent_hashing', sessions=sess, remove_at=30.0, remove_idx=1)
        self.assertGreater(R['retried'], 0)
        for s in sess:
            for r in s.reqs:
                if r.arr >= 30.0:
                    self.assertNotEqual(r.worker, 1)

    def test_ring_report_json(self):
        res = SR.ring_report(as_json=True)
        self.assertLess(res['8->9']['ring'], 0.2)
        self.assertGreater(res['8->9']['mod'], 0.8)


# ---------------------------------------------------------------- rollout
class TestRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sync = RS.run(0.0, steps=4, P=16)
        cls.april = RS.run(0.5, steps=4, P=16)
        cls.drop = RS.run(0.5, steps=4, P=16, recycle=False)

    def test_sync_trains_everything_on_policy(self):
        self.assertEqual(self.sync['samples_trained'], 4 * 16 * 8)
        self.assertEqual(self.sync['stale_token_share'], 0.0)
        self.assertEqual(self.sync['max_lag'], 0)
        self.assertEqual(self.sync['dropped_tokens'], 0)

    def test_april_faster_but_off_policy(self):
        self.assertEqual(self.april['samples_trained'], 4 * 16 * 8)
        self.assertGreater(self.april['trained_tok_per_s'], self.sync['trained_tok_per_s'])
        self.assertGreater(self.april['stale_token_share'], 0.0)
        self.assertGreaterEqual(self.april['max_lag'], 1)
        self.assertEqual(self.april['dropped_tokens'], 0)

    def test_drop_biases_toward_short(self):
        self.assertGreater(self.drop['dropped_tokens'], 0)
        self.assertEqual(self.drop['stale_token_share'], 0.0)
        self.assertLess(self.drop['out_mean'], self.sync['out_mean'])

    def test_sync_tail(self):
        r = self.sync['steps'][0]
        self.assertLess(r['t90'], r['time'])
        self.assertLessEqual(r['out_max'], RS.MAX_NEW)

    def test_retract_when_kv_small(self):
        srv = RS.sim_serve.Server('qwen3-8b', 'h100-sxm')
        srv.cap = 40000
        R = RS.run(0.0, steps=1, P=8, srv=srv)
        self.assertGreater(R['retracts'], 0)
        self.assertEqual(R['samples_trained'], 64)

    def test_lengths_deterministic_and_capped(self):
        a = [s.out for _ in range(20) for s in RS.LengthGen(seed=1).group(0).samples]
        g1, g2 = RS.LengthGen(seed=1), RS.LengthGen(seed=1)
        self.assertEqual([s.out for s in g1.group(0).samples], [s.out for s in g2.group(0).samples])
        self.assertTrue(all(16 <= x <= RS.MAX_NEW for x in a))

    def test_recycled_sample_mixes_versions(self):
        eng = RS.Engine(RS.sim_serve.Server('qwen3-8b', 'h100-sxm'), 0)
        s = RS.Sample(0, 0, 300, 500)
        eng.submit(s)
        for _ in range(50):
            eng.step()
        back = eng.abort_all()
        self.assertEqual(back, [s])
        self.assertGreater(s.gen, 0)
        eng2 = RS.Engine(eng.srv, 1)
        eng2.submit(s)
        while not s.done:
            eng2.step()
        self.assertEqual([v for v, _ in s.segs], [0, 1])
        self.assertEqual(eng2.recompute, s.segs[1][1])


class TestSync(unittest.TestCase):
    def test_calibration(self):
        S = RS.sync_table()
        glm = S['calibration'][0]
        self.assertAlmostEqual(glm['eff_gbs'], 7.6, delta=0.1)

    def test_bucketing_beats_call_per_tensor(self):
        for model, tp in [('qwen3-235b-a22b', 8), ('deepseek-v3', 16)]:
            r = RS.sync_cost(model, tp, 1, 9.0)
            self.assertLess(r['bucketed'], r['per_tensor_call'] / 10)
            self.assertGreater(r['tensors'], 10000)

    def test_bandwidth_dominates_small_model(self):
        r = RS.sync_cost('qwen3-8b', 1, 1, 9.0)
        xfer = r['bytes_per_gpu'] / 9e9
        self.assertGreater(xfer / r['bucketed'], 0.6)

    def test_reports_run(self):
        quiet(RS.report_sync, RS.sync_table())
        quiet(SR.ring_report)


if __name__ == '__main__':
    unittest.main()
