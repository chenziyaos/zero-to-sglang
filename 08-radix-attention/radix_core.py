"""sim_radix.py 用到的核心数据结构：RadixKey、TreeNode、RadixCache、分页分配器、驱逐策略。

一行行对着 sglang 8fabf5e 的 python/sglang/srt/mem_cache/radix_cache.py 和 evict_policy.py 写，
差别只有三处：
  - 时间用逻辑时钟（每次访问 +1），不用 time.monotonic()，结果可复现；
  - evictable_leaves 不做增量维护，驱逐时现场遍历一遍（源码在 _update_leaf_status 里增量维护，结果一样）；
  - T-LRU 的两个长度字段也放进经典树里算（源码只在 UnifiedRadixCache 里维护，经典树上选 tlru 会被拒绝）。
"""
import heapq
import itertools


def ceil_align(x, ps):
    return -(-x // ps) * ps


class Clock:
    """代替 time.monotonic()：每调一次 +1。"""

    def __init__(self):
        self.t = 0

    def __call__(self):
        self.t += 1
        return self.t


# ----------------------------------------------------------------------------- RadixKey
class RadixKey:
    """radix_cache.py:59。token 序列 + 命名空间（extra_key，比如 lora_id；cache_salt）。

    命名空间只体现在 child_key 里：根节点下面先按 (extra_key, 首页 token) 分叉，
    所以不同 LoRA、不同 salt 的请求哪怕 token 完全一样，也走不到同一棵子树。
    """
    __slots__ = ('tokens', 'extra_key', 'salt')

    def __init__(self, tokens, extra_key=None, salt=None):
        self.tokens = tuple(tokens)
        self.extra_key = extra_key
        self.salt = salt

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, s):
        return RadixKey(self.tokens[s], self.extra_key, self.salt)

    def page_aligned(self, ps):
        # :147，向下取整到整页；不满一页的尾巴不参与匹配、也不进树
        return self if ps == 1 else self[: len(self) // ps * ps]

    def match(self, other, ps):
        # :178 / :182，源码用倍增 + 二分找第一个不同的位置，这里直接逐个比，结果一样
        n = 0
        for a, b in zip(self.tokens, other.tokens):
            if a != b:
                break
            n += 1
        return n if ps == 1 else n // ps * ps

    def child_key(self, ps):
        # :226 child_key_at：page 1 用单个 token，page > 1 用整页 token 组成的 tuple
        plain = self.tokens[0] if ps == 1 else self.tokens[:ps]
        if self.salt is not None:
            return ((self.extra_key, self.salt), plain)
        return plain if self.extra_key is None else (self.extra_key, plain)


# ----------------------------------------------------------------------------- TreeNode
_ids = itertools.count()


class TreeNode:
    """radix_cache.py:250。value 是这段 key 对应的 KV 槽位号（req_to_token 那一行的切片拷贝）。"""
    __slots__ = ('id', 'children', 'parent', 'key', 'value', 'lock_ref', 'last_access_time',
                 'creation_time', 'hit_count', 'priority', 'tlru_cached', 'tlru_hist')

    def __init__(self, now, priority=0):
        self.id = next(_ids)
        self.children = {}
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = now
        self.creation_time = now
        self.hit_count = 0
        self.priority = priority
        self.tlru_cached = 0   # 根到本节点的 token 数（论文里的 cached prefix）
        self.tlru_hist = 0     # 这条分支到过的最大深度（论文里的 L），尾巴被裁掉也不缩

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


# ----------------------------------------------------------------------------- 驱逐策略
def make_policy(name, tlru_threshold=0, tlru_next=0, slru_threshold=2):
    """evict_policy.py。返回值越小越先被驱逐，堆是小根堆。"""
    if name == 'lru':        # :16
        return lambda n: n.last_access_time
    if name == 'lfu':        # :21
        return lambda n: (n.hit_count, n.last_access_time)
    if name == 'fifo':       # :26
        return lambda n: n.creation_time
    if name == 'mru':        # :31
        return lambda n: -n.last_access_time
    if name == 'priority':   # :41
        return lambda n: (n.priority, n.last_access_time)
    if name == 'slru':       # :84，命中 >= 2 次的进保护段
        return lambda n: (1 if n.hit_count >= slru_threshold else 0, n.last_access_time)
    if name == 'tlru':       # :49
        def tlru(n):
            budget = max(n.tlru_hist + tlru_next - tlru_threshold, 0)
            safe = n.tlru_cached - len(n.key) >= budget
            return (-1 if safe else 0, n.last_access_time)
        return tlru
    raise ValueError(name)


# ----------------------------------------------------------------------------- 分配器
class PagedAllocator:
    """简化版的 token_to_kv 分配器：以页为单位分配，槽位号 = 页号 * ps + 页内偏移，页 0 留作 padding。
    完整的三段式分配见 ../07-memory-pool/sim_mempool.py，这里只关心「多少槽位在空闲表里」。"""

    def __init__(self, num_tokens, ps):
        self.ps = ps
        self.num_pages = num_tokens // ps
        self.size = self.num_pages * ps
        self.free_pages = []      # 还回来的页，尾部弹出
        self.fresh = 1            # 从没分出去过的页从这里往后数，等价于 arange(1, num_pages + 1)，省内存

    def available_size(self):
        return (len(self.free_pages) + self.num_pages + 1 - self.fresh) * self.ps

    def alloc_pages(self, n_pages):
        if n_pages * self.ps > self.available_size():
            return None
        out = []
        for _ in range(n_pages):
            if self.free_pages:
                p = self.free_pages.pop()
            else:
                p = self.fresh
                self.fresh += 1
            out.extend(range(p * self.ps, p * self.ps + self.ps))
        return out

    def free(self, slots):
        # paged.py free：槽位号 // ps 去重后整页归还
        pages = sorted({s // self.ps for s in slots})
        self.free_pages.extend(reversed(pages))


# ----------------------------------------------------------------------------- RadixCache
class RadixCache:
    """radix_cache.py:281。只保留 KV 槽位相关的逻辑，HiCache、事件上报、bigram（EAGLE）都去掉。"""

    def __init__(self, allocator, ps=1, policy='lru', **policy_kw):
        self.alloc = allocator
        self.ps = ps
        self.now = Clock()
        self.policy_name = policy
        self.get_priority = make_policy(policy, **policy_kw)
        self.evict_calls = 0
        self.evicted_tokens = 0
        self.reset()

    def reset(self):
        # :343，根节点 lock_ref = 1，永远不会被驱逐；priority 取最小，任何真实优先级都比它高
        self.root = TreeNode(self.now(), priority=-(1 << 62))
        self.root.key = RadixKey(())
        self.root.value = []
        self.root.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    # ---- 匹配 ------------------------------------------------------------------
    def match_prefix(self, key):
        """:364。返回 (槽位列表, 最后一个节点)。匹配停在某个节点中间时，把它分裂成两段。"""
        key = key.page_aligned(self.ps)
        if len(key) == 0:
            return [], self.root
        t = self.now()
        node = self.root
        node.last_access_time = t
        value = []
        while len(key) and key.child_key(self.ps) in node.children:
            child = node.children[key.child_key(self.ps)]
            child.last_access_time = t
            n = child.key.match(key, self.ps)
            if n < len(child.key):
                new = self._split_node(child, n)
                value.extend(new.value)
                node = new
                break
            value.extend(child.value)
            node = child
            key = key[n:]
        return value, node

    def _split_node(self, child, n):
        """:677。child 拆成 new(前 n 个) -> child(剩下的)。new 继承 priority、hit_count、lock_ref，
        这样分裂前后加锁计数不变，evictable / protected 账也不用动。"""
        new = TreeNode(self.now(), priority=child.priority)
        new.hit_count = child.hit_count
        new.lock_ref = child.lock_ref
        new.creation_time = child.creation_time
        new.parent = child.parent
        new.key = child.key[:n]
        new.value = child.value[:n]
        new.children = {child.key[n:].child_key(self.ps): child}
        new.tlru_cached = new.parent.tlru_cached + n
        new.tlru_hist = child.tlru_hist
        new.parent.children[new.key.child_key(self.ps)] = new
        child.parent = new
        child.key = child.key[n:]
        child.value = child.value[n:]
        return new

    # ---- 插入 ------------------------------------------------------------------
    def insert(self, key, value, inserted_len=0, priority=0):
        """:428 -> _insert_helper :705。返回 (已经在树里的前缀长度, 最后一个节点)。

        - 路径上的 priority 取 max，高优先级会一路传到根附近的共享前缀；
        - hit_count 只在「这一段这条请求以前没插过」时 +1（inserted_len 之后的部分），
          同一条请求 chunked prefill 分几次插入，不会把自己算成多次命中；
        - 新叶子的 len(key) 计入 evictable_size_：刚插进来、没人锁，随时可以被驱逐。
        """
        key = key.page_aligned(self.ps)
        value = value[: len(key)]
        t = self.now()
        node = self.root
        node.last_access_time = t
        node.priority = max(node.priority, priority)
        total = 0
        while len(key) and key.child_key(self.ps) in node.children:
            node = node.children[key.child_key(self.ps)]
            node.last_access_time = t
            n = node.key.match(key, self.ps)
            total += n
            key, value = key[n:], value[n:]
            if n < len(node.key):
                node = self._split_node(node, n)
            node.priority = max(node.priority, priority)
            if total > inserted_len:
                node.hit_count += 1
        if len(key):
            new = TreeNode(t, priority=priority)
            new.parent = node
            new.key = key
            new.value = list(value)
            if total + len(key) > inserted_len:
                new.hit_count += 1
            new.tlru_cached = node.tlru_cached + len(key)
            new.tlru_hist = new.tlru_cached
            cur = node
            while cur is not None and cur.tlru_hist < new.tlru_cached:   # unified_tree_core.py:191
                cur.tlru_hist = new.tlru_cached
                cur = cur.parent
            node.children[key.child_key(self.ps)] = new
            self.evictable_size_ += len(key)
            node = new
        return total, node

    # ---- 加锁 ------------------------------------------------------------------
    def inc_lock_ref(self, node):
        """:582。从 node 走到根，每个节点 lock_ref + 1；0 -> 1 的那一刻把它的长度从 evictable 挪到 protected。"""
        delta = 0
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node):
        """:597。反过来，1 -> 0 的时候挪回 evictable。"""
        delta = 0
        while node is not self.root:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1          # 源码这里没有 >= 0 的断言，多解一次锁会变成 -1
            node = node.parent
        return delta

    # ---- 驱逐 ------------------------------------------------------------------
    def evictable_leaves(self):
        """_update_leaf_status :790 的判定：有 KV、没人锁、没有子节点。"""
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children.values())
            elif n is not self.root and n.lock_ref == 0:
                out.append(n)
        return out

    def evict(self, num_tokens):
        """:552。所有可驱逐叶子按策略建小根堆，弹一个删一个；父节点因此变成无锁叶子时，压回堆里继续参与。
        返回实际驱逐的 token 数（可能比要求的多，因为一次删整个节点）。"""
        self.evict_calls += 1
        heap = [(self.get_priority(n), n.id, n) for n in self.evictable_leaves()]
        heapq.heapify(heap)
        done = 0
        while done < num_tokens and heap:
            _, _, x = heapq.heappop(heap)
            self.alloc.free(x.value)
            done += len(x.value)
            self._delete_leaf(x)
            p = x.parent
            if p is not self.root and not p.children and p.lock_ref == 0:
                heapq.heappush(heap, (self.get_priority(p), p.id, p))
        self.evicted_tokens += done
        return done

    def _delete_leaf(self, node):
        # :780
        v = node.parent.children.pop(node.key.child_key(self.ps))
        assert v is node
        self.evictable_size_ -= len(node.key)

    # ---- 统计 ------------------------------------------------------------------
    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def total_size(self):
        s, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            s += len(n.value)
            stack.extend(n.children.values())
        return s

    def num_nodes(self):
        c, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            c += 1
            stack.extend(n.children.values())
        return c - 1

    def dump(self, show_value=False, indent='  '):
        lines = []

        def rec(n, d):
            for ch in sorted(n.children.values(), key=lambda c: c.key.tokens):
                tok = list(ch.key.tokens[:12])
                more = '…' if len(ch.key) > 12 else ''
                extra = f' val={ch.value}' if show_value else ''
                ns = f' ns={ch.key.extra_key}' if ch.key.extra_key is not None else ''
                lines.append(f'{indent}{"  " * d}{tok}{more}{ns} len={len(ch.key)} lock={ch.lock_ref} '
                             f'hit={ch.hit_count} prio={ch.priority}{extra}')
                rec(ch, d + 1)
        rec(self.root, 0)
        return '\n'.join(lines)
