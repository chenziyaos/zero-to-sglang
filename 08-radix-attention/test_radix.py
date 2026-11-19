"""radix_core / sim_radix 的边界情况。python3 -m unittest test_radix -v"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radix_core import PagedAllocator, RadixCache, RadixKey  # noqa: E402
from sim_radix import WORKLOADS, Engine, Req, run  # noqa: E402


def tree(n=1024, ps=1, policy='lru'):
    a = PagedAllocator(n, ps)
    return a, RadixCache(a, ps, policy)


class TestKey(unittest.TestCase):
    def test_page_floor(self):
        k = RadixKey(range(10))
        self.assertEqual(len(k.page_aligned(4)), 8)
        self.assertEqual(k.match(RadixKey([0, 1, 2, 3, 4, 5, 9]), 4), 4)   # 6 个相同，向下取整到 4

    def test_namespace(self):
        self.assertNotEqual(RadixKey([1, 2]).child_key(1), RadixKey([1, 2], 'lora').child_key(1))
        self.assertNotEqual(RadixKey([1, 2], 'lora').child_key(1), RadixKey([1, 2], 'lora', 's').child_key(1))


class TestTree(unittest.TestCase):
    def test_short_key_is_noop(self):
        a, t = tree(ps=4)
        pre, node = t.insert(RadixKey([1, 2, 3]), [4, 5, 6])
        self.assertEqual((pre, t.total_size()), (0, 0))      # 不满一页不进树
        self.assertIs(node, t.root)

    def test_split_keeps_value(self):
        a, t = tree()
        v = a.alloc_pages(5)
        t.insert(RadixKey([1, 2, 3, 4, 5]), v)
        got, node = t.match_prefix(RadixKey([1, 2, 9]))
        self.assertEqual(got, v[:2])
        self.assertEqual(t.num_nodes(), 2)
        self.assertEqual(t.total_size(), 5)

    def test_hit_count_not_double(self):
        a, t = tree()
        k = RadixKey(range(8))
        v = a.alloc_pages(8)
        t.insert(k[:4], v[:4], inserted_len=0)
        t.insert(k, v, inserted_len=4)                       # chunked prefill 第二块
        first = t.root.children[0]
        self.assertEqual(first.hit_count, 1)

    def test_lock_books(self):
        a, t = tree()
        _, n = t.insert(RadixKey(range(6)), a.alloc_pages(6))
        t.inc_lock_ref(n)
        t.inc_lock_ref(n)
        self.assertEqual((t.evictable_size(), t.protected_size()), (0, 6))
        t.dec_lock_ref(n)
        self.assertEqual(t.protected_size(), 6)
        t.dec_lock_ref(n)
        self.assertEqual((t.evictable_size(), t.protected_size()), (6, 0))

    def test_evict_skips_locked_and_cascades(self):
        a, t = tree()
        _, n1 = t.insert(RadixKey([1, 2, 3]), a.alloc_pages(3))
        _, n2 = t.insert(RadixKey([1, 2, 3, 4, 5]), a.alloc_pages(5))
        _, n3 = t.insert(RadixKey([7, 8]), a.alloc_pages(2))
        t.inc_lock_ref(n3)
        self.assertEqual(t.evict(100), 5)                    # [4,5] 先删，[1,2,3] 变成叶子接着删；[7,8] 锁着
        self.assertEqual(t.total_size(), 2)

    def test_priority_tiebreak(self):
        a, t = tree(policy='priority')
        t.insert(RadixKey([1, 2]), a.alloc_pages(2), priority=5)
        t.insert(RadixKey([3, 4]), a.alloc_pages(2), priority=0)
        t.evict(1)
        self.assertIn(1, t.root.children)
        self.assertNotIn(3, t.root.children)


class TestEngine(unittest.TestCase):
    def test_idle_invariant_after_workload(self):
        for ps in (1, 16):
            for name in ('fewshot', 'chat'):
                reqs = list(WORKLOADS[name](random.Random(0)))[:200]
                e = run(reqs, 20_000, ps=ps)
                a, ev, p, u, tot = e.invariant()
                self.assertEqual((p, u), (0, 0))
                self.assertEqual(a + ev, tot)

    def test_max_prefix_leaves_one(self):
        e = Engine(256)
        r = Req(range(10), out_tokens=[])
        e.admit(r)
        e.release(r)
        r2 = Req(range(10))
        e.admit(r2)
        self.assertEqual(r2.prefix_len, 9)

    def test_forget_free_breaks_books(self):
        e = Engine(128, 4)
        r = Req(range(9), out_tokens=[1, 2, 3])          # 9 + 2 = 11 个有 KV，尾页不满不进树
        e.admit(r)
        e.checkpoint(r)
        e.decode(r, 3)
        e.release(r, forget_free=True)
        a, ev, p, u, tot = e.invariant()
        self.assertLess(a + ev + p + u, tot)


if __name__ == '__main__':
    unittest.main()
