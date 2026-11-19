"""多副本前面的路由：同一份多轮会话负载，换五种路由策略，看前缀命中率、负载均衡和 TTFT 怎么变。

负载（默认）：4 个 Qwen3-8B / 1xH100 副本；8 个应用，各自一段 1000-3000 token 的 system prompt，
应用热度按 Zipf 分布；会话按泊松过程到达（对比和扫参每秒 8 个，摘台实验每秒 4.8 个，
摘掉一台后剩下三台还能接住），每个会话 2-10 轮，下一轮在上一轮答完后隔一段思考时间才发，
每轮的 prompt = system prompt + 之前所有轮的问答 + 这一轮的问题。

副本模型（每台各自独立）：
  - 容量、chunk、单步耗时取自 ../03-bench/sim_serve.py 的 Server（Roofline 下限 / 利用率 + 固定开销）
  - 调度同 sim_serve：prefill 优先、一次只切一条请求、准入时按 输入 + 输出 预留 KV
  - 前缀缓存：按 16 token 一块的哈希链近似 RadixCache（SGL-08），只缓存整块；
    prefill 完把 prompt 插进树并加锁，请求结束时把 prompt + 输出一起插入、解锁；
    放不下时从没加锁的叶子按 LRU 驱逐（evict_to_free_tokens）

路由策略（对齐 sgl-model-gateway 主干 8fabf5e 的 src/policies/*.rs）：
  round_robin         计数器取模（round_robin.rs:31）
  power_of_two        随机挑两台，比轮询到的 token 负载（每 5 s 拉一次 /v1/loads 的 num_total_tokens），
                      相等或缺失时比在途请求数，再相等取第一台（power_of_two.rs:107-131）
  consistent_hashing  请求头 X-SMG-Routing-Key 填会话 id，150 个虚节点的哈希环（worker_registry.rs:31）。
                      真网关按 X-SMG-Target-Worker、X-SMG-Routing-Key、authorization / x-forwarded-for / cookie
                      的顺序找 key，都没有就随机（consistent_hashing.rs:105-160）；Python 启动器
                      （sglang_router.launch_router）的 --policy 能选它，Rust 二进制 main.rs 的列表里没有
  prefix_hash         前 256 个 token 的哈希上环，目标台负载超过 平均 x 1.25 就换负载最低的台（prefix_hash.rs:117-197）。
                      真网关只有 gRPC 路径会把 token 交给策略，HTTP 路径传 tokens=None，策略返回 NoTokens
                      （router.rs:134-180、prefix_hash.rs:211）；这里假设走的是 gRPC
  cache_aware         近似基数树按租户记录每台见过的请求文本；负载失衡（差值 > abs 且 最大 > 最小 x rel）时
                      走最小在途数，否则 匹配率 > cache_threshold 去匹配最长的那台，不然去在途数最少的台
                      （cache_aware.rs:387-511）。默认用 CLI 的 0.3 / 64 / 1.5，--lib-defaults 换成库默认的 0.5 / 32 / 1.1

和真网关的差别：网关的树按字符存原文，这里按 token 块存，匹配率在每 token 字符数均匀时相同；
blake3 / xxh3 不在标准库里，换成 blake2b，环上的分布性质一样；负载 load() 是网关自己数的在途请求数，这里同样。

用法：
    python3 sim_router.py                     # 五种策略对比
    python3 sim_router.py --sweep             # cache_aware 的 balance_abs_threshold 与 cache_threshold
    python3 sim_router.py --remove            # 跑到一半摘掉一台
    python3 sim_router.py --ring              # 哈希环：扩缩容时搬走多少 key、虚节点数和均衡
    python3 sim_router.py --policy cache_aware --json
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import random
import sys
from bisect import bisect_left

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
import sim_serve  # noqa: E402

BLK = 16                     # 前缀缓存的块大小（token）
VNODES = 150                 # worker_registry.rs:31 VIRTUAL_NODES_PER_WORKER
POLL = 5.0                   # main.rs:775 load_check_interval_secs
CHARS_PER_TOKEN = 4          # max_tree_size 按字符计，换算成 token 用


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little')


# ---------------------------------------------------------------- 负载
class Req:
    __slots__ = ('rid', 'sess', 'turn', 'app', 'inp', 'out', 'blocks', 'arr', 'worker', 'matched',
                 'done_in', 'fwd', 'ttft', 'end', 'start', 'private', 'locked', 'routed_match', 'retries')

    def __init__(self, rid, sess, turn, app, inp, out, blocks):
        self.rid, self.sess, self.turn, self.app = rid, sess, turn, app
        self.inp, self.out, self.blocks = inp, out, blocks
        self.arr = self.fwd = self.ttft = self.end = None
        self.worker = None
        self.matched = 0
        self.done_in = 0
        self.start = 0
        self.private = 0
        self.locked = 0
        self.routed_match = 0.0
        self.retries = 0


class Session:
    __slots__ = ('sid', 'app', 'start', 'reqs', 'think', 'tokens')


def make_workload(seed=7, apps=8, zipf=1.0, rate=8.0, duration=180.0, turns=(2, 10),
                  user=(50, 400), out=(100, 500), think=6.0, sys_len=(1000, 3000)):
    """返回会话列表。token 用互不相同的整数代替，块哈希沿链计算：前缀相同 <=> 块哈希相同。"""
    rng = random.Random(seed)
    sys_tok = []
    for a in range(apps):
        n = rng.randint(*sys_len)
        sys_tok.append([a * 10_000_000 + i for i in range(n)])
    w = [1 / (k + 1) ** zipf for k in range(apps)]
    nxt = 10 ** 9
    sessions, t, sid, rid = [], 0.0, 0, 0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            break
        s = Session()
        s.sid, s.app, s.start = sid, rng.choices(range(apps), w)[0], t
        stream = list(sys_tok[s.app])
        nt = rng.randint(*turns)
        lens = []
        for _ in range(nt):
            u, o = rng.randint(*user), rng.randint(*out)
            lens.append((len(stream) + u, o))
            stream.extend(range(nxt, nxt + u + o))
            nxt += u + o
        chain, prev = [], 0
        for i in range(0, len(stream) - BLK + 1, BLK):
            prev = h64(prev.to_bytes(8, 'little') + bytes(str(stream[i:i + BLK]), 'ascii'))
            chain.append(prev)
        s.tokens = stream[:256]
        s.reqs = []
        for k, (inp, o) in enumerate(lens):
            s.reqs.append(Req(rid, sid, k, s.app, inp, o, chain[:inp // BLK]))
            rid += 1
        s.think = [rng.expovariate(1 / think) for _ in range(nt)]
        sessions.append(s)
        sid += 1
    return sessions


# ---------------------------------------------------------------- 副本：前缀缓存
class Blk:
    __slots__ = ('parent', 'nchild', 'last', 'lock')

    def __init__(self, parent, now):
        self.parent, self.nchild, self.last, self.lock = parent, 0, now, 0


class PrefixCache:
    def __init__(self):
        self.b = {}
        self.heap = []

    @property
    def tokens(self):
        return len(self.b) * BLK

    def match(self, chain, limit):
        n = 0
        for h in chain[:limit]:
            if h not in self.b:
                break
            n += 1
        return n

    def touch_lock(self, chain, n, now, d):
        for h in chain[:n]:
            x = self.b[h]
            x.last = now
            x.lock += d
            if x.lock == 0 and x.nchild == 0:
                heapq.heappush(self.heap, (now, h))

    def insert(self, chain, now):
        prev = 0
        for h in chain:
            if h not in self.b:
                self.b[h] = Blk(prev, now)
                if prev:
                    self.b[prev].nchild += 1
                heapq.heappush(self.heap, (now, h))
            prev = h

    def evict(self, need):
        freed = 0
        while freed < need and self.heap:
            t, h = heapq.heappop(self.heap)
            x = self.b.get(h)
            if x is None or x.lock or x.nchild or x.last != t:
                continue
            del self.b[h]
            freed += BLK
            if x.parent:
                p = self.b[x.parent]
                p.nchild -= 1
                if p.nchild == 0 and p.lock == 0:
                    heapq.heappush(self.heap, (p.last, x.parent))
        return freed


class Worker:
    def __init__(self, idx, srv):
        self.idx, self.srv = idx, srv
        self.url = f'http://worker{idx}:30000'
        self.cache = PrefixCache()
        self.waiting = []
        self.running = {}
        self.fin = []
        self.dcount = 0
        self.sum_inp = self.sum_start = 0
        self.private = 0
        self.busy_until = None
        self.busy = 0.0
        self.load = 0            # 网关侧的在途请求数（Worker::load）
        self.healthy = True
        self.pending = None      # 正在执行的步

    def used(self):
        return self.cache.tokens + self.private

    def polled_tokens(self):
        """/v1/loads 的 num_total_tokens 近似：在跑请求已占的 KV + 等待队列的序列长度。"""
        n = len(self.running)
        run = self.sum_inp + n * (1 + self.dcount) - self.sum_start
        return run + sum(r.inp for r in self.waiting)

    def t_decode(self):
        n = len(self.running)
        avg = (self.sum_inp + n * (1 + self.dcount) - self.sum_start) / n
        d = self.srv.s.decode(n, avg)
        return self.srv._t(d['flops'], d['bytes'])

    def plan(self, now):
        """决定下一步做什么，返回耗时；没活返回 None。"""
        srv = self.srv
        parts, done = [], []
        budget = srv.chunk
        for r in self.waiting:
            if budget <= 0:
                break
            if r.fwd is None:
                m = self.cache.match(r.blocks, (r.inp - 1) // BLK)
                need = (r.inp - m * BLK) + r.out
                free = srv.cap - self.used()
                if need > free:
                    self.cache.evict(need - free)
                    m = self.cache.match(r.blocks, (r.inp - 1) // BLK)
                    need = (r.inp - m * BLK) + r.out
                    if need > srv.cap - self.used():
                        break
                r.fwd, r.matched, r.done_in = now, m * BLK, m * BLK
                self.cache.touch_lock(r.blocks, m, now, +1)
                r.locked = m
                r.private = need
                self.private += need
            take = min(budget, r.inp - r.done_in)
            parts.append((take, r.done_in))
            r.done_in += take
            budget -= take
            if r.done_in == r.inp:
                done.append(r)
            else:
                break
        if parts:
            self.pending = ('prefill', done)
            return srv.t_prefill(parts)
        if self.running:
            self.pending = ('decode', None)
            return self.t_decode()
        self.pending = None
        return None

    def apply(self, now):
        """步结束：返回这一步里结束的请求。"""
        kind, done = self.pending
        finished = []
        if kind == 'prefill':
            for r in done:
                self.waiting.remove(r)
                n_before = len(self.cache.b)
                self.cache.insert(r.blocks, now)
                full = len(r.blocks)
                self.cache.touch_lock(r.blocks[r.locked:full], full - r.locked, now, +1)
                r.locked = full
                moved = (len(self.cache.b) - n_before) * BLK
                self.private -= min(moved, r.private)
                r.private -= min(moved, r.private)
                r.ttft = now - r.arr
                r.start = self.dcount
                if r.out == 1:
                    finished.append(r)
                    continue
                self.running[r.rid] = r
                self.sum_inp += r.inp
                self.sum_start += r.start
                heapq.heappush(self.fin, (r.start + r.out - 1, r.rid))
        else:
            self.dcount += 1
            while self.fin and self.fin[0][0] <= self.dcount:
                _, rid = heapq.heappop(self.fin)
                r = self.running.pop(rid)
                self.sum_inp -= r.inp
                self.sum_start -= r.start
                finished.append(r)
        for r in finished:
            r.end = now
            self.cache.touch_lock(r.blocks, r.locked, now, -1)
            self.private -= r.private
            r.private = 0
            self.load -= 1
        return finished

    def drain(self):
        """摘掉这台时，网关把它手上的请求重试到别处（--retry-max-retries）。"""
        reqs = list(self.waiting) + list(self.running.values())
        for r in reqs:
            if r.locked:
                self.cache.touch_lock(r.blocks, r.locked, 0, -1)
            r.fwd = None
            r.private = r.locked = r.done_in = r.matched = 0
            r.retries += 1
        self.waiting, self.running, self.fin = [], {}, []
        self.sum_inp = self.sum_start = self.private = 0
        self.load = 0
        self.pending = None
        return reqs


# ---------------------------------------------------------------- 哈希环
class HashRing:
    """worker_registry.rs:44-120：每台 150 个虚节点，位置 = hash(url + '#' + vnode)，顺时针找第一台健康的。"""

    def __init__(self, urls, vnodes=VNODES):
        e = []
        for u in urls:
            for v in range(vnodes):
                e.append((h64(u.encode() + b'#' + v.to_bytes(8, 'little')), u))
        e.sort()
        self.pos = [p for p, _ in e]
        self.url = [u for _, u in e]

    def find(self, key, ok=lambda u: True):
        if not self.pos:
            return None
        i = bisect_left(self.pos, h64(key.encode()))
        n = len(self.pos)
        seen = set()
        for k in range(n):
            u = self.url[(i + k) % n]
            if u in seen:
                continue
            if ok(u):
                return u
            seen.add(u)
        return None


# ---------------------------------------------------------------- 网关的近似树（cache_aware）
class TNode:
    __slots__ = ('parent', 'h', 'children', 'tenants', 'last_tenant')

    def __init__(self, parent, h, tenant):
        self.parent, self.h = parent, h
        self.children = {}
        self.tenants = {}
        self.last_tenant = tenant


class ApproxTree:
    """tree.rs：每个节点记录哪些租户（worker url）插入过这段前缀；
    匹配走到最深的节点，取该节点的 last_tenant（建节点的那台；分裂时继承），失效时退回任意一个租户（:571-598）。"""

    def __init__(self):
        self.root = TNode(None, 0, None)
        self.size = {}

    def insert(self, chain, tenant, now):
        self.root.tenants.setdefault(tenant, 0)
        self.size.setdefault(tenant, 0)
        node = self.root
        for h in chain:
            c = node.children.get(h)
            if c is None:
                c = TNode(node, h, tenant)
                node.children[h] = c
            if tenant not in c.tenants:
                self.size[tenant] += BLK
                c.tenants[tenant] = 0
            node = c
        node.tenants[tenant] = now

    def match(self, chain):
        node, n = self.root, 0
        for h in chain:
            c = node.children.get(h)
            if c is None or not c.tenants:
                break
            node, n = c, n + 1
        t = node.last_tenant if node.last_tenant in node.tenants else (min(node.tenants) if node.tenants else None)
        return n, t

    def remove_tenant(self, tenant):
        st = [self.root]
        while st:
            x = st.pop()
            x.tenants.pop(tenant, None)
            st.extend(x.children.values())
        self.size.pop(tenant, None)

    def evict(self, max_tokens):
        """evict_tenant_by_size（tree.rs:718）：每个租户超过上限时，按时间戳从老到新删它的叶子。"""
        over = {t for t, s in self.size.items() if s > max_tokens}
        if not over:
            return
        heap, st = [], [self.root]
        while st:
            x = st.pop()
            st.extend(x.children.values())
            if x is self.root:
                continue
            for t in x.tenants:
                if t in over and not any(t in c.tenants for c in x.children.values()):
                    heapq.heappush(heap, (x.tenants[t], id(x), t, x))
        while heap:
            ts, _, t, x = heapq.heappop(heap)
            if self.size[t] <= max_tokens or t not in x.tenants:
                continue
            del x.tenants[t]
            self.size[t] -= BLK
            p = x.parent
            if not x.tenants and not x.children:
                del p.children[x.h]
            if p is not self.root and t in p.tenants and not any(t in c.tenants for c in p.children.values()):
                heapq.heappush(heap, (p.tenants[t], id(p), t, p))


# ---------------------------------------------------------------- 策略
class Policy:
    name = ''

    def __init__(self, workers, rng):
        self.w, self.rng = workers, rng

    def healthy(self):
        return [i for i, w in enumerate(self.w) if w.healthy]

    def on_remove(self, idx):
        pass

    def tick(self, now):
        pass


class RoundRobin(Policy):
    name = 'round_robin'

    def __init__(self, *a):
        super().__init__(*a)
        self.c = 0

    def select(self, req, sess):
        h = self.healthy()
        i = h[self.c % len(h)]
        self.c += 1
        return i


class PowerOfTwo(Policy):
    name = 'power_of_two'

    def __init__(self, *a):
        super().__init__(*a)
        self.cached = {}

    def tick(self, now):
        self.cached = {i: w.polled_tokens() for i, w in enumerate(self.w) if w.healthy}

    def select(self, req, sess):
        h = self.healthy()
        if len(h) == 1:
            return h[0]
        i1 = self.rng.randrange(len(h))
        i2 = (i1 + 1 + self.rng.randrange(len(h) - 1)) % len(h)
        a, b = h[i1], h[i2]
        t1, t2 = self.cached.get(a), self.cached.get(b)
        if t1 is not None and t2 is not None and t1 != t2:
            l1, l2 = t1, t2
        else:
            l1, l2 = self.w[a].load, self.w[b].load
        return a if l1 <= l2 else b


class ConsistentHashing(Policy):
    name = 'consistent_hashing'

    def __init__(self, *a):
        super().__init__(*a)
        self.ring = HashRing([w.url for w in self.w])
        self.by_url = {w.url: i for i, w in enumerate(self.w)}

    def select(self, req, sess):
        u = self.ring.find(f'session-{req.sess}', lambda u: self.w[self.by_url[u]].healthy)
        return self.by_url[u]


class PrefixHash(Policy):
    name = 'prefix_hash'

    def __init__(self, workers, rng, prefix_token_count=256, load_factor=1.25):
        super().__init__(workers, rng)
        self.n, self.f = prefix_token_count, load_factor
        self.ring = HashRing([w.url for w in self.w])
        self.by_url = {w.url: i for i, w in enumerate(self.w)}
        self.branch = {'RingHit': 0, 'LoadBalanceWalk': 0}

    def load_ok(self, load, total, n):
        if total == 0:
            return True
        return load <= (total + 1) / n * self.f

    def select(self, req, sess):
        h = self.healthy()
        total = sum(self.w[i].load for i in h)
        key = format(h64(bytes(str(sess.tokens[:self.n]), 'ascii')), '016x')
        u = self.ring.find(key, lambda u: self.w[self.by_url[u]].healthy)
        i = self.by_url[u]
        if self.load_ok(self.w[i].load, total, len(h)):
            self.branch['RingHit'] += 1
            return i
        self.branch['LoadBalanceWalk'] += 1
        ok = [j for j in h if self.load_ok(self.w[j].load, total, len(h))]
        return min(ok, key=lambda j: self.w[j].load) if ok else i


class CacheAware(Policy):
    name = 'cache_aware'

    def __init__(self, workers, rng, cache_threshold=0.3, balance_abs_threshold=64, balance_rel_threshold=1.5,
                 eviction_interval=120, max_tree_size=2 ** 26):
        super().__init__(workers, rng)
        self.th, self.abs, self.rel = cache_threshold, balance_abs_threshold, balance_rel_threshold
        self.evi, self.max_tok = eviction_interval, max_tree_size // CHARS_PER_TOKEN
        self.tree = ApproxTree()
        for w in self.w:
            self.tree.insert([], w.url, 0)
        self.by_url = {w.url: i for i, w in enumerate(self.w)}
        self.next_evict = eviction_interval
        self.branch = {'imbalanced': 0, 'hit': 0, 'low_match': 0}

    def tick(self, now):
        if self.evi and now >= self.next_evict:
            self.tree.evict(self.max_tok)
            self.next_evict += self.evi

    def on_remove(self, idx):
        self.tree.remove_tenant(self.w[idx].url)

    def min_load(self, h):
        m = min(self.w[i].load for i in h)
        return self.rng.choice([i for i in h if self.w[i].load == m])

    def select(self, req, sess, now=0.0):
        h = self.healthy()
        # router.rs:151 先滤掉不可用的台再交给策略，所以 min / max 只在健康的台里取
        loads = [self.w[i].load for i in h]
        mn, mx = min(loads), max(loads)
        if mx - mn > self.abs and mx > mn * self.rel:
            i = self.min_load(h)
            self.branch['imbalanced'] += 1
        else:
            n, t = self.tree.match(req.blocks)
            rate = n * BLK / req.inp
            req.routed_match = rate
            i = None
            if rate > self.th and t is not None:
                j = self.by_url[t]
                if self.w[j].healthy:
                    i = j
                    self.branch['hit'] += 1
            if i is None:
                if rate > self.th and t is not None:
                    # 匹配到的那台已经不在：删掉它的租户，退回第一台健康的（cache_aware.rs:489-511）
                    self.tree.remove_tenant(t)
                    i = h[0]
                else:
                    i = self.min_load(h)
                    self.branch['low_match'] += 1
        self.tree.insert(req.blocks, self.w[i].url, now)
        return i


POLICIES = {'round_robin': RoundRobin, 'power_of_two': PowerOfTwo, 'consistent_hashing': ConsistentHashing,
            'prefix_hash': PrefixHash, 'cache_aware': CacheAware}


# ---------------------------------------------------------------- 事件循环
def run(policy='cache_aware', n_workers=4, sessions=None, seed=1, remove_at=None, remove_idx=0, **kw):
    sessions = sessions if sessions is not None else make_workload()
    for s in sessions:
        for r in s.reqs:
            r.__init__(r.rid, r.sess, r.turn, r.app, r.inp, r.out, r.blocks)
    srv = sim_serve.Server('qwen3-8b', 'h100-sxm')
    workers = [Worker(i, srv) for i in range(n_workers)]
    rng = random.Random(seed)
    pol = POLICIES[policy](workers, rng, **kw)
    ev, seq = [], 0

    def push(t, kind, x):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, x))
        seq += 1

    for s in sessions:
        push(s.start, 'arrive', (s, 0))
    push(0.0, 'poll', None)
    if remove_at is not None:
        push(remove_at, 'remove', remove_idx)
    by_sess = {s.sid: s for s in sessions}
    remaining = sum(len(s.reqs) for s in sessions)
    assign = {}

    def kick(w, now):
        if w.pending is None and w.healthy:
            dt = w.plan(now)
            if dt is not None:
                w.busy += dt
                push(now + dt, 'step', w.idx)

    def dispatch(r, now):
        s = by_sess[r.sess]
        if isinstance(pol, CacheAware):
            i = pol.select(r, s, now)
        else:
            i = pol.select(r, s)
        r.worker = i
        assign.setdefault(r.sess, []).append(i)
        w = workers[i]
        w.load += 1
        w.waiting.append(r)
        kick(w, now)

    while remaining:
        now, _, kind, x = heapq.heappop(ev)
        if kind == 'arrive':
            s, k = x
            r = s.reqs[k]
            r.arr = now
            dispatch(r, now)
        elif kind == 'poll':
            pol.tick(now)
            push(now + POLL, 'poll', None)
        elif kind == 'remove':
            w = workers[x]
            w.healthy = False
            pol.on_remove(x)
            for r in w.drain():
                dispatch(r, now)
        else:
            w = workers[x]
            if not w.healthy or w.pending is None:
                continue
            for r in w.apply(now):
                remaining -= 1
                s = by_sess[r.sess]
                if r.turn + 1 < len(s.reqs):
                    push(now + s.think[r.turn], 'arrive', (s, r.turn + 1))
            w.pending = None
            kick(w, now)
    return summarize(pol, workers, sessions, assign, remove_at)


def summarize(pol, workers, sessions, assign, remove_at=None):
    reqs = [r for s in sessions for r in s.reqs]
    end = max(r.end for r in reqs)
    inp = sum(r.inp for r in reqs)
    hit = sum(r.matched for r in reqs)
    per = [sum(1 for r in reqs if r.worker == w.idx) for w in workers]
    busy = [w.busy / end for w in workers]
    hb = [b for b, w in zip(busy, workers) if w.healthy or remove_at is None]
    moved = sum(1 for a in assign.values() if len(set(a)) > 1)
    R = {
        'policy': pol.name,
        'n': len(reqs),
        'hit': hit / inp,
        'prefill_tok': inp - hit,
        'ttft_p50': pct([r.ttft for r in reqs], 0.5),
        'ttft_p99': pct([r.ttft for r in reqs], 0.99),
        'e2e_p99': pct([r.end - r.arr for r in reqs], 0.99),
        'busy': busy,
        'busy_max_mean': max(hb) / (sum(hb) / len(hb)),
        'reqs_per_worker': per,
        'sessions_split': moved / len(assign),
        'makespan': end,
    }
    if isinstance(pol, CacheAware):
        R['branch'] = dict(pol.branch)
        rm = [r.routed_match for r in reqs if r.routed_match]
        R['router_match'] = sum(r.routed_match * r.inp for r in reqs) / inp
        R['router_match_n'] = len(rm)
    if isinstance(pol, PrefixHash):
        R['branch'] = dict(pol.branch)
    if remove_at is not None:
        after = [r for r in reqs if r.arr >= remove_at]
        before = [r for r in reqs if r.arr < remove_at]
        R['hit_before'] = sum(r.matched for r in before) / sum(r.inp for r in before)
        R['hit_after'] = sum(r.matched for r in after) / sum(r.inp for r in after)
        R['ttft_p99_after'] = pct([r.ttft for r in after], 0.99)
        R['retried'] = sum(1 for r in reqs if r.retries)
    return R


# ---------------------------------------------------------------- 报告
def pad(s, n):
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    return s + ' ' * max(0, n - w)


def head():
    print('  ' + pad('策略', 26) + '命中率  prefill 千 token  TTFT p50  TTFT p99  E2E p99  忙碌 max/mean  会话跨台')


def row(label, R):
    print(f'  {pad(label, 26)}{R["hit"]:>6.0%}{R["prefill_tok"] / 1e3:>18,.0f}{R["ttft_p50"]:>10.2f}'
          f'{R["ttft_p99"]:>10.2f}{R["e2e_p99"]:>9.1f}{R["busy_max_mean"]:>15.2f}{R["sessions_split"]:>10.0%}')


def workload_line(sessions):
    reqs = [r for s in sessions for r in s.reqs]
    apps = {}
    for s in sessions:
        apps[s.app] = apps.get(s.app, 0) + 1
    top = max(apps.values()) / len(sessions)
    print(f'负载：{len(sessions)} 个会话、{len(reqs)} 条请求，prompt 中位数 {pct([r.inp for r in reqs], 0.5):,.0f} token、'
          f'最长 {max(r.inp for r in reqs):,}，最热的应用占 {top:.0%} 的会话；4 台 Qwen3-8B / H100，每台 KV 池 '
          f'{sim_serve.Server("qwen3-8b", "h100-sxm").cap:,} token')


def compare(sessions):
    out = {}
    for p in ['round_robin', 'power_of_two', 'consistent_hashing', 'prefix_hash', 'cache_aware']:
        out[p] = run(p, sessions=sessions)
    out['cache_aware_lib'] = run('cache_aware', sessions=sessions, cache_threshold=0.5,
                                 balance_abs_threshold=32, balance_rel_threshold=1.1)
    return out


LABEL = {'round_robin': 'round_robin', 'power_of_two': 'power_of_two', 'consistent_hashing': 'consistent_hashing',
         'prefix_hash': 'prefix_hash', 'cache_aware': 'cache_aware（CLI 默认）',
         'cache_aware_lib': 'cache_aware（库默认）'}


def report_compare(out):
    print('五种策略，同一份负载（秒）')
    head()
    for k, R in out.items():
        row(LABEL[k], R)
    ca = out['cache_aware']
    b = ca['branch']
    tot = sum(b.values())
    print(f'  cache_aware 分支：命中 {b["hit"] / tot:.0%}、低匹配走最少在途 {b["low_match"] / tot:.0%}、'
          f'失衡走最少在途 {b["imbalanced"] / tot:.0%}；网关树估的匹配率 {ca["router_match"]:.0%}，副本上实际命中 {ca["hit"]:.0%}')
    ph = out['prefix_hash']['branch']
    print(f'  prefix_hash 分支：环上目标台 {ph["RingHit"]}、超载改投 {ph["LoadBalanceWalk"]}；'
          f'每台请求数 {out["prefix_hash"]["reqs_per_worker"]}')
    print(f'  每台忙碌比例：round_robin {fmt_busy(out["round_robin"])}，cache_aware {fmt_busy(ca)}，'
          f'prefix_hash {fmt_busy(out["prefix_hash"])}')


def fmt_busy(R):
    return ' / '.join(f'{b:.0%}' for b in R['busy'])


def sweep(sessions):
    out = {}
    for a in [4, 16, 64, 256]:
        out[f'abs={a}'] = run('cache_aware', sessions=sessions, balance_abs_threshold=a)
    for th in [0.1, 0.5, 0.9]:
        out[f'th={th}'] = run('cache_aware', sessions=sessions, cache_threshold=th)
    out['tree=200k'] = run('cache_aware', sessions=sessions, max_tree_size=200_000 * CHARS_PER_TOKEN,
                           eviction_interval=10)
    return out


def report_sweep(out):
    print('cache_aware 调参（其余参数取 CLI 默认 0.3 / 64 / 1.5）')
    head()
    for k, R in out.items():
        b = R['branch']
        tot = sum(b.values())
        row(k, R)
        print(f'  {"":26}  分支：命中 {b["hit"] / tot:.0%} / 低匹配 {b["low_match"] / tot:.0%} / 失衡 {b["imbalanced"] / tot:.0%}；'
              f'树估匹配率 {R["router_match"]:.0%}')


def remove(sessions):
    out = {}
    for p in ['round_robin', 'consistent_hashing', 'cache_aware']:
        out[p] = run(p, sessions=sessions, remove_at=90.0, remove_idx=1)
    return out


def report_remove(out):
    print('第 90 s 摘掉 worker1，手上的请求由网关重试到其余三台')
    print('  ' + pad('策略', 22) + '摘前命中  摘后命中  摘后 TTFT p99  重试请求  会话跨台')
    for k, R in out.items():
        print(f'  {pad(k, 22)}{R["hit_before"]:>8.0%}{R["hit_after"]:>10.0%}{R["ttft_p99_after"]:>15.2f}'
              f'{R["retried"]:>10}{R["sessions_split"]:>10.0%}')


def ring_report(as_json=False):
    keys = [f'session-{i}' for i in range(20000)]
    urls = [f'http://worker{i}:30000' for i in range(9)]
    res = {}

    def assign(ring, n):
        return [ring.find(k) for k in keys] if ring else [urls[h64(k.encode()) % n] for k in keys]

    for a, b in [(4, 5), (4, 3), (8, 9)]:
        r1, r2 = HashRing(urls[:a]), HashRing(urls[:b])
        x1, x2 = assign(r1, a), assign(r2, b)
        m1, m2 = assign(None, a), assign(None, b)
        res[f'{a}->{b}'] = {
            'ring': sum(1 for u, v in zip(x1, x2) if u != v) / len(keys),
            'mod': sum(1 for u, v in zip(m1, m2) if u != v) / len(keys),
            'ideal': abs(b - a) / max(a, b),
        }
    bal = {}
    for vn in [1, 10, 150, 1000]:
        r = HashRing(urls[:4], vn)
        c = {}
        for k in keys:
            u = r.find(k)
            c[u] = c.get(u, 0) + 1
        v = list(c.values()) + [0] * (4 - len(c))
        bal[vn] = max(v) / (len(keys) / 4)
    res['balance'] = bal
    if as_json:
        return res
    print('哈希环：2 万个会话 key，扩缩容时换了落点的比例')
    print('  变化    环（150 虚节点）  取模 hash % N   理想')
    for k in ['4->5', '4->3', '8->9']:
        x = res[k]
        print(f'  {k:<8}{x["ring"]:>14.1%}{x["mod"]:>15.1%}{x["ideal"]:>8.1%}')
    print('  4 台时最忙一台的 key 数 / 平均：' + '，'.join(f'{vn} 虚节点 {b:.2f}' for vn, b in bal.items()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', choices=list(POLICIES))
    ap.add_argument('--lib-defaults', action='store_true', help='cache_aware 用库默认 0.5 / 32 / 1.1')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--remove', action='store_true')
    ap.add_argument('--ring', action='store_true')
    ap.add_argument('--rate', type=float, default=None,
                    help='会话到达率，个/秒；默认对比和调参 8（接近 4 台的容量），摘台 4.8（摘掉一台后仍放得下）')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.ring:
        res = ring_report(a.json)
        if a.json:
            print(json.dumps(res, indent=1))
        return res
    rate = a.rate or (4.8 if a.remove else 8.0)
    sessions = make_workload(rate=rate)
    if a.policy:
        kw = dict(cache_threshold=0.5, balance_abs_threshold=32, balance_rel_threshold=1.1) \
            if a.lib_defaults and a.policy == 'cache_aware' else {}
        out = {a.policy: run(a.policy, sessions=sessions, **kw)}
        rep = None
    elif a.sweep:
        out, rep = sweep(sessions), report_sweep
    elif a.remove:
        out, rep = remove(sessions), report_remove
    else:
        out, rep = compare(sessions), report_compare
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return out
    workload_line(sessions)
    if rep:
        rep(out)
    else:
        head()
        for k, R in out.items():
            row(k, R)
    return out


if __name__ == '__main__':
    main()
