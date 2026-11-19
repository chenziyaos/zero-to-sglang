"""SGLang RadixAttention 前缀缓存的 CPU 复刻：一棵基数树加一个分页分配器，把一条请求从匹配到释放的
每一步、每个节点的 lock_ref、evictable / protected 两本账都打印出来。

目的：不装 GPU，看清下面几件事，每一步都能对上源码行号。
  1. match_prefix 怎么沿树往下走、为什么匹配停在节点中间时要分裂节点；
  2. insert 怎么复用已有前缀、为什么同一条请求插两次不会把 hit_count 算两遍；
  3. 请求在跑的时候为什么要锁住路径（inc_lock_ref），锁住的 token 从 evictable 挪到 protected；
  4. checkpoint（prefill 结束、chunked prefill 每一块之后）怎么把新算出的 KV 交给树、把锁往深处挪、
     把和树里重复的槽位还给分配器；
  5. 驱逐只删叶子、按策略排序、父节点变成叶子以后接着参与；lru / lfu / slru / priority / tlru 各自在什么负载下占优；
  6. 两本账加上请求私有的槽位必须等于池子总量，空闲时这条不变式怎么帮你抓泄漏。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - RadixKey：命名空间（extra_key = lora_id 等、cache_salt）进 child_key，page > 1 时 child_key 是整页 token 的 tuple；
              匹配结果和 key 都向下取整到整页
  - match_prefix / _split_node / insert / _insert_helper：逐行同构，包括 priority 沿路径取 max、
              inserted_len 之后才计 hit_count、新叶子计入 evictable
  - _insert_cache / checkpoint：插入后释放 [cache_protected_len, 树里已有前缀) 这段重复槽位，
              重新匹配后改写 req_to_token 行，cache_protected_len 前移，锁从旧节点换到新节点；
              split_prompt 时再插一次 prompt 部分、优先级 + 1，驱逐时先丢输出
  - release_kv_cache：checkpoint 之后只还 [cache_protected_len, 已分配)；最后一个生成的 token 没有 KV
  - evict：可驱逐叶子建小根堆，弹出即 free，父节点无子、无锁时压回堆
  - 匹配上限 max_prefix_len = input_len - 1（_compute_max_prefix_len），至少留一个 token 给 prefill 算 logits
  - 不变式：available + evictable + protected + uncached == total（invariant_checker.py:79），
           uncached = Σ ceil_align(kv_allocated_len, ps) - cache_protected_len
不对齐的部分：
  - 时间用逻辑时钟；evictable_leaves 现场遍历，不做增量维护
  - 线上默认是 UnifiedRadixCache + Rust 树核，SWA / Mamba 组件、HiCache 主机层、会话缓存都不模拟
  - T-LRU 在源码里只有 UnifiedRadixCache 支持，这里放进经典树里只为对比策略
  - 调度器只模拟「一批准入 → prefill → checkpoint → decode → 释放」，不模拟 chunked prefill 的预算、retract

用法：
    python3 sim_radix.py                          # walk：源码自带的小例子 + 两条请求的完整生命周期
    python3 sim_radix.py --scenario walk --page-size 4
    python3 sim_radix.py --scenario hitrate       # 四种负载 × 缓存大小的命中率
    python3 sim_radix.py --scenario policy        # lru / lfu / slru / priority / tlru 在三种负载下
    python3 sim_radix.py --scenario order         # 同一批请求 FCFS 和按前缀聚类的命中率
    python3 sim_radix.py --scenario nsample       # n > 1 时为什么要先发一条 max_new_tokens=0 的预热请求
    python3 sim_radix.py --scenario pagesize      # page size 对命中、节点数的影响
    python3 sim_radix.py --scenario leak          # 忘记解锁、忘记释放、重复释放、重复解锁
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
from radix_core import PagedAllocator, RadixCache, RadixKey, ceil_align  # noqa: E402


# ----------------------------------------------------------------------------- 请求
class Req:
    """只保留和 KV 生命周期有关的字段（schedule_batch.py ReqKvInfo :958 附近）。"""
    _n = 0

    def __init__(self, tokens, max_new=0, extra_key=None, priority=0, rid=None, out_tokens=None):
        Req._n += 1
        self.rid = rid if rid is not None else f'r{Req._n}'
        self.input = list(tokens)
        self.max_new = max_new
        self.out_tokens = out_tokens          # 生成内容；None 时用 rid 派生的随机 token，保证各条请求不同
        self.output = []
        self.extra_key = extra_key
        self.priority = priority
        self.row = []                         # req_to_token 这一行：已分配的槽位（整页）
        self.prefix_len = 0
        self.last_node = None
        self.lock = None
        self.cache_protected_len = 0          # [0, 这里) 归树所有，请求不负责释放
        self.cache_inserted_len = 0           # [0, 这里) 本请求已经插过，再插不重复计 hit_count
        self.cached_tokens = 0                # 准入时命中的前缀长度（metrics 里的 #cached-token）

    @property
    def seq(self):
        return self.input + self.output

    def committed(self):
        # 已经算出 KV 的 token 数：最后一个生成的 token 还没进过前向，没有 KV
        return len(self.input) + max(len(self.output) - 1, 0)

    def max_prefix_len(self):
        # _compute_max_prefix_len :1773，至少留一个 token 进 prefill，才有 logits 采样第一个输出
        return len(self.input) - 1


# ----------------------------------------------------------------------------- 引擎
class Engine:
    """把 match_kv_cache / alloc / checkpoint_kv_cache / release_kv_cache 串起来的最小调度器。"""

    def __init__(self, total_tokens, ps=1, policy='lru', **kw):
        self.alloc = PagedAllocator(total_tokens, ps)
        self.tree = RadixCache(self.alloc, ps, policy, **kw)
        self.ps = ps
        self.live = []
        self.input_tokens = 0
        self.cached_tokens = 0
        self.prefill_tokens = 0
        self.rejected = 0
        self.log = None

    def say(self, *a):
        if self.log:
            self.log(*a)

    # -- 准入：match_kv_cache (common.py:221) + 加锁 + 分配 -------------------------------
    def admit(self, req):
        key = RadixKey(req.input[: req.max_prefix_len()], req.extra_key)
        value, last = self.tree.match_prefix(key)
        req.prefix_len = len(value)
        req.last_node = last
        req.row = list(value)
        req.cache_protected_len = len(value)
        self.tree.inc_lock_ref(last)                  # schedule_policy.py:1700 req.lock = tree_cache.lock(last_node)
        req.lock = last
        need = ceil_align(len(req.input), self.ps) - len(value)
        if not self._reserve(need):
            self.tree.dec_lock_ref(last)
            req.lock = None
            self.rejected += 1
            return False
        req.row += self.alloc.alloc_pages(need // self.ps)
        req.cached_tokens = len(value)
        self.input_tokens += len(req.input)
        self.cached_tokens += len(value)
        self.prefill_tokens += len(req.input) - len(value)
        self.live.append(req)
        return True

    def _reserve(self, need):
        # allocator/base.py:155 evict_to_free_tokens：差多少驱逐多少
        short = need - self.alloc.available_size()
        if short > 0:
            self.tree.evict(short)
        return self.alloc.available_size() >= need

    # -- prefill 结束：checkpoint_kv_cache (common.py) -> RadixCache.checkpoint :510 ----------
    def checkpoint(self, req, up_to=None, split_prompt=True):
        tree, ps = self.tree, self.ps
        up_to = req.committed() if up_to is None else up_to
        key = RadixKey(req.seq[:up_to], req.extra_key).page_aligned(ps)
        klen = len(key)
        # _insert_cache :450
        prefix, _ = tree.insert(key, req.row[:klen], inserted_len=req.cache_inserted_len, priority=req.priority)
        req.cache_inserted_len = max(req.cache_inserted_len, klen)
        if split_prompt:
            pkey = RadixKey(req.input, req.extra_key).page_aligned(ps)
            if 0 < len(pkey) < klen:
                tree.insert(pkey, req.row[: len(pkey)], inserted_len=len(pkey), priority=req.priority + 1)
        dup = req.row[req.cache_protected_len:prefix]
        if dup:
            self.say(f'    树里已有 [{req.cache_protected_len}, {prefix}) 这段（别的请求先插了），'
                     f'本请求那份重复槽位还给分配器：{dup[:8]}{"…" if len(dup) > 8 else ""}')
            self.alloc.free(dup)
        value, new_last = tree.match_prefix(key)
        assert len(value) == klen
        req.row[req.cache_protected_len:klen] = value[req.cache_protected_len:]
        req.cache_protected_len = klen
        tree.dec_lock_ref(req.lock)
        tree.inc_lock_ref(new_last)
        req.lock = req.last_node = new_last

    # -- decode：只在跨页时取新页 -------------------------------------------------------
    def decode(self, req, n=1, rng=None):
        for _ in range(n):
            if req.out_tokens is not None:
                t = req.out_tokens[len(req.output)]
            else:
                t = (rng or random).randrange(100000, 10 ** 9)
            # 追加一个输出 token，前一个 token 的 KV 在这一步写入：需要 len(input) + len(output) 个槽位
            if len(req.input) + len(req.output) > len(req.row):
                if not self._reserve(self.ps):
                    return False
                req.row += self.alloc.alloc_pages(1)
            req.output.append(t)
        return True

    # -- 结束：release_kv_cache (common.py:418) ----------------------------------------
    def release(self, req, checkpoint=True, forget_unlock=False, forget_free=False):
        owned = req.committed()
        if checkpoint:
            self.checkpoint(req, up_to=owned)
        if not forget_free:
            self.alloc.free(req.row[req.cache_protected_len:])   # [cache_protected_len, 已分配) 整页还回去
        if not forget_unlock:
            self.tree.dec_lock_ref(req.lock)
        req.lock = None
        req.row = []
        self.live.remove(req)

    # -- 不变式：invariant_checker.py:79 ----------------------------------------------
    def uncached(self):
        return sum(ceil_align(len(r.row), self.ps) - r.cache_protected_len for r in self.live)

    def invariant(self):
        a = self.alloc.available_size()
        e = self.tree.evictable_size()
        p = self.tree.protected_size()
        u = self.uncached()
        return a, e, p, u, self.alloc.size

    def books(self):
        a, e, p, u, t = self.invariant()
        ok = '==' if a + e + p + u == t else '!='
        return f'available {a} + evictable {e} + protected {p} + uncached {u} = {a + e + p + u} {ok} total {t}'

    def hit_rate(self):
        return self.cached_tokens / max(self.input_tokens, 1)


# ============================================================================= 场景 1：walk
def scenario_walk(ps):
    print(f'================ page_size = {ps} ================')
    if ps == 1:
        print('[0] radix_cache.py 末尾 __main__ 的例子：五次插入，再匹配 [1,2,3,13,14]')
        alloc = PagedAllocator(64, 1)
        t = RadixCache(alloc, 1)
        for s in ([1, 2, 3], [1, 2, 3], [1, 2, 4, 5], [1, 2, 4, 5, 6, 7], [8, 9, 10, 11, 12]):
            key = RadixKey(s)
            pre, _ = t.insert(key, alloc.alloc_pages(len(s)))
            print(f'  insert {s!s:<20} 树里已有前缀 {pre}，新增 {len(s) - pre} 个 token')
        print(t.dump())
        v, node = t.match_prefix(RadixKey([1, 2, 3, 13, 14]))
        print(f'  match [1,2,3,13,14] -> 命中 {len(v)} 个，停在 key={list(node.key.tokens)} 的节点')
        print('  第二次 insert [1,2,3] 树里已有 3 个，新增 0；[1,2,4,5] 让 [1,2,3] 在第 2 个 token 处分裂成 [1,2] -> [3]\n')

    eng = Engine(256, ps)
    eng.log = print
    sys_prompt = list(range(1000, 1012))      # 12 token 的系统提示词
    a = Req(sys_prompt + [1, 2, 3, 4, 5], rid='A', out_tokens=[901, 902, 903, 904])
    b = Req(sys_prompt + [7, 8, 9], rid='B', out_tokens=[801, 802])
    print('[1] A 准入：树是空的，命中 0')
    eng.admit(a)
    print(f'  A row = {a.row}  cache_protected_len = {a.cache_protected_len}')
    print(f'  {eng.books()}')
    print('\n[2] A 的 prefill 跑完，checkpoint：输入 17 个 token 进树，锁从根挪到新叶子')
    eng.checkpoint(a)
    print(eng.tree.dump())
    print(f'  A cache_protected_len = {a.cache_protected_len}（向下取整到页），'
          f'cache_inserted_len = {a.cache_inserted_len}')
    print(f'  {eng.books()}')
    print('\n[3] B 准入：和 A 共享 12 token 系统提示词，匹配停在 A 的节点中间 -> 分裂')
    eng.admit(b)
    print(eng.tree.dump())
    print(f'  B 命中 {b.prefix_len}，B row 前 {b.prefix_len} 个槽位就是树里的：{b.row[:b.prefix_len]}')
    print(f'  {eng.books()}')
    print('  分裂出来的 [1000..] 节点继承了 A 叶子的 lock_ref=1，B 再锁一次变成 2；protected 不重复计')
    print('\n[4] 两条都 decode，A 生成 4 个、B 生成 2 个后结束；B 先走')
    eng.decode(a, 4)
    eng.decode(b, 2)
    eng.checkpoint(b)
    eng.release(b)
    print(f'  B 释放后：{eng.books()}')
    eng.release(a)
    print(f'  A 释放后：{eng.books()}')
    print(eng.tree.dump())
    print('  A 的输出（4 个里前 3 个有 KV）跟在 prompt 后面，split_prompt 把 prompt 边界单独切成一个节点，'
          '优先级 +1：priority 策略下先丢输出、留 prompt')
    print('\n[5] 完全一样的请求再来一次：max_prefix_len = input_len - 1')
    c = Req(sys_prompt + [1, 2, 3, 4, 5], rid='C')
    eng.admit(c)
    print(f'  输入 {len(c.input)} 个，树里有完整的 {len(c.input)} 个，但只匹配前 {len(c.input) - 1} 个，'
          f'再向下取整到页：命中 {c.prefix_len}，prefill 重算 {len(c.input) - c.prefix_len} 个')
    eng.release(c, checkpoint=False)
    print('\n[6] 同样的 token，换一个 LoRA（extra_key）：命名空间不同，不共享')
    d = Req(sys_prompt + [1, 2, 3, 4, 5], rid='D', extra_key='lora-7')
    eng.admit(d)
    print(f'  D 命中 {d.prefix_len}；child_key 带上 extra_key 以后，根下面是另一个分支')
    eng.checkpoint(d)
    eng.release(d)
    print('\n[7] 池子要腾 10 个 token：驱逐只动没锁的叶子，按 LRU')
    before = eng.tree.total_size()
    got = eng.tree.evict(10)
    print(f'  要 10，实际删了 {got}（一次删整个节点），树里 {before} -> {eng.tree.total_size()}')
    print(eng.tree.dump())
    print(f'  {eng.books()}')


# ============================================================================= 负载
class Tok:
    """发不重复的 token 号：每次 take(n) 给 n 个从没出现过的 token。"""

    def __init__(self, start=10 ** 6):
        self.n = start

    def take(self, k):
        out = list(range(self.n, self.n + k))
        self.n += k
        return out


def wl_fewshot(rng, n_req=1200, n_tpl=8, tpl_len=1200, q=(20, 60), out=(10, 40)):
    """少样本 / 固定长系统提示词：n_tpl 个模板按 Zipf 分布被选中，后面跟一个短问题。"""
    tk = Tok()
    tpls = [tk.take(tpl_len) for _ in range(n_tpl)]
    w = [1 / (i + 1) for i in range(n_tpl)]
    for _ in range(n_req):
        t = rng.choices(range(n_tpl), w)[0]
        yield dict(input=tpls[t] + tk.take(rng.randint(*q)), out=rng.randint(*out), group=t)


def wl_chat(rng, n_users=60, turns=8, sys_len=300, msg=(30, 120), out=(80, 300)):
    """多轮对话：每轮输入 = 系统提示词 + 历史全部轮次 + 新消息；用户之间交错到达。"""
    tk = Tok()
    sysp = tk.take(sys_len)
    hist = {u: list(sysp) for u in range(n_users)}
    left = {u: turns for u in range(n_users)}
    while left:
        u = rng.choice(list(left))
        inp = hist[u] + tk.take(rng.randint(*msg))
        o = tk.take(rng.randint(*out))
        yield dict(input=inp, out=len(o), out_tokens=o, group=u, user=u)
        hist[u] = inp + o
        left[u] -= 1
        if not left[u]:
            del left[u]


def wl_selfcons(rng, n_q=150, n=8, q_len=(300, 900), out=(150, 400)):
    """self-consistency / 并行采样：同一个问题采 n 条，每条输出不同。"""
    tk = Tok()
    for g in range(n_q):
        q = tk.take(rng.randint(*q_len))
        for _ in range(n):
            yield dict(input=list(q), out=rng.randint(*out), group=g)


def wl_unique(rng, n_req=600, ln=(200, 1500), out=(50, 300)):
    """没有任何共享：RAG 每次检索出不同文档、批量离线摘要等。"""
    tk = Tok()
    for _ in range(n_req):
        yield dict(input=tk.take(rng.randint(*ln)), out=rng.randint(*out), group=-1)


WORKLOADS = {'fewshot': wl_fewshot, 'chat': wl_chat, 'selfcons': wl_selfcons, 'unique': wl_unique}


def run(reqs, total, ps=1, policy='lru', batch=16, order='fcfs', warmup=False, max_waiting=10 ** 9, **kw):
    """把一串请求按「一批准入 -> prefill -> checkpoint -> decode 到结束 -> 释放」跑完。
    max_waiting   ：等待队列里最多有多少条（模拟到达速率），默认全部请求一开始就在队里；
    order='lpm'   ：每批开始前把等待队列按树里的命中长度排序（schedule_policy.py 的 lpm），队列超过 128 条退回 fcfs；
    order='dedup' ：批内去重，和等待队列里排在前面的请求共享 >= 32 token 的，推到下一批
                    （_compute_prefix_matches :356 用 waiting_queue_radix_tree 做的事）；
    warmup=True   ：同一组（同一个问题的 n 次采样）第一次出现时，先跑一条 max_new_tokens=0 的请求把前缀放进树
                    （tokenizer_manager.py:2130 附近，n > 1 时的做法）。
    命中率只统计真实请求，预热请求不算。"""
    eng = Engine(total, ps, policy, **kw)
    rng = random.Random(7)
    stream = list(reqs)
    queue, stream = stream[:max_waiting], stream[max_waiting:]
    per_req = []
    warmed = set()
    while queue:
        if order == 'lpm' and len(queue) <= 128:
            # schedule_policy.py:314 _determine_active_policy：等待队列超过 128 条时 lpm 退回 fcfs
            for r in queue:
                r['_m'] = len(eng.tree.match_prefix(RadixKey(r['input'][:-1], r.get('extra_key')))[0])
            queue = sorted(queue, key=lambda r: -r['_m'])
        if order == 'dedup':
            cur, rest, seen = [], [], RadixCache(PagedAllocator(10 ** 9, 1), 1)
            for spec in queue:
                if len(cur) >= batch:
                    rest.append(spec)
                    continue
                k = RadixKey(spec['input'][:-1], spec.get('extra_key'))
                if len(seen.match_prefix(k)[0]) >= 32:
                    rest.append(spec)
                else:
                    seen.insert(k, list(range(len(k))))
                    cur.append(spec)
            queue = rest
        else:
            cur, queue = queue[:batch], queue[batch:]
        running, back = [], []
        for spec in cur:
            if back:                        # 前面有一条没放下，后面的也等下一批，保持 FCFS 顺序
                back.append(spec)
                continue
            if warmup and spec['group'] >= 0 and spec['group'] not in warmed:
                warmed.add(spec['group'])
                w = Req(spec['input'], 0, spec.get('extra_key'))
                if eng.admit(w):
                    eng.release(w)          # max_new_tokens=0：prefill 完直接结束，release 时 checkpoint 进树
            r = Req(spec['input'], spec['out'], spec.get('extra_key'), spec.get('priority', 0),
                    out_tokens=spec.get('out_tokens'))
            r.spec = spec
            if eng.admit(r):
                running.append(r)
                per_req.append((spec, len(r.input), r.prefix_len))
            elif running:
                back.append(spec)           # 池子不够：留到下一批（真实调度器里就是继续排队）
            else:
                eng.dropped = getattr(eng, 'dropped', 0) + 1   # 一条请求比整个池子还大
        queue = back + queue
        n_new = max(max_waiting - len(queue), 0)
        queue, stream = queue + stream[:n_new], stream[n_new:]
        for r in running:
            eng.checkpoint(r)
        for r in running:
            eng.decode(r, r.max_new, rng)
        for r in running:
            eng.release(r)
    eng.per_req = per_req
    eng.hit = sum(x[2] for x in per_req) / max(sum(x[1] for x in per_req), 1)
    return eng


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


# ============================================================================= 场景 2：hitrate
def scenario_hitrate():
    print('四种负载，池子从 8K 到 256K token，策略 lru，page 1，每批 16 条')
    print('命中率 = Σ 准入时命中的前缀 / Σ 输入 token（metrics_reporter.py 的 cache hit rate 同口径）')
    print('selfcons 按 sglang 对 n > 1 的默认做法，先发一条 max_new_tokens=0 的预热请求\n')
    sizes = [8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
    print(f'{"负载":<10}{"输入 token":>12}' + ''.join(f'{s // 1000:>8}K' for s in sizes) + f'{"无限大":>10}')
    for name, fn in WORKLOADS.items():
        reqs = list(fn(random.Random(1)))
        row = []
        for s in sizes + [10 ** 7]:
            row.append(run(reqs, s, warmup=(name == 'selfcons')).hit)
        tot = sum(len(r['input']) for r in reqs)
        print(f'{name:<10}{tot:>12,}' + ''.join(f'{h:>9.1%}' for h in row[:-1]) + f'{row[-1]:>10.1%}')


# ============================================================================= 场景 3：policy
def wl_scan(rng, n=1500, n_tpl=6, tpl_len=900, burst_every=150, burst=60, scan_len=(1500, 3000)):
    """热模板 + 周期性的扫描洪峰：平时 6 个模板反复命中，每隔 150 条突然来 60 条互不相同的长文档。"""
    tk = Tok()
    tpls = [tk.take(tpl_len) for _ in range(n_tpl)]
    for i in range(n):
        if i % burst_every < burst and i >= burst_every:
            yield dict(input=tk.take(rng.randint(*scan_len)), out=rng.randint(10, 60), group=-1)
        else:
            t = rng.randrange(n_tpl)
            yield dict(input=tpls[t] + tk.take(rng.randint(20, 60)), out=rng.randint(10, 60),
                       group=t, priority=1)


def wl_shift(rng, n_old=800, n_new=1600, tpl_len=900):
    """热点迁移：前 800 条集中在 2 个旧模板上（每个模板被命中几百次），之后换成 4 个新模板，旧模板再也不来。"""
    tk = Tok()
    old = [tk.take(tpl_len) for _ in range(2)]
    new = [tk.take(tpl_len) for _ in range(4)]
    for _ in range(n_old):
        t = rng.randrange(2)
        yield dict(input=old[t] + tk.take(rng.randint(20, 60)), out=rng.randint(10, 60), group=t)
    for _ in range(n_new):
        t = rng.randrange(4)
        yield dict(input=new[t] + tk.take(rng.randint(20, 60)), out=rng.randint(10, 60), group=10 + t)


def stale_tokens(tree, lo, hi):
    """树里首 token 落在 [lo, hi) 的节点（以及它们的子树）一共多少 token。"""
    tot, stack = 0, [(c, False) for c in tree.root.children.values()]
    while stack:
        n, inside = stack.pop()
        inside = inside or lo <= n.key.tokens[0] < hi
        if inside:
            tot += len(n.key)
        stack.extend((c, inside) for c in n.children.values())
    return tot


def scenario_policy():
    print('五种策略 × 三种负载，page 1。P90 是每条请求真正要 prefill 的 token 数（TTFT 的代理）；')
    print('超 SLO = prefill token 数超过 1024 的请求占比，假设 1024 token 以内能满足 TTFT SLO。')
    print('tlru 的 threshold 就取这个 1024，next_prompt_estimate 取 75（新消息长度的均值）。\n')
    cases = [
        ('scan', wl_scan, 48_000, 16, '热模板 + 扫描洪峰，池子 48K，每批 16 条'),
        ('shift', wl_shift, 4_600, 2, '热点迁移，池子 4.6K（只放得下 4 个模板多一点），每批 2 条'),
        ('chat', lambda r: wl_chat(r, n_users=80, turns=10), 90_000, 16, '多轮对话 80 用户 × 10 轮，池子 90K，每批 16 条'),
    ]
    pols = [('lru', {}), ('lfu', {}), ('slru', {}), ('priority', {}),
            ('tlru', dict(tlru_threshold=1024, tlru_next=75))]
    for name, fn, size, bs, desc in cases:
        reqs = list(fn(random.Random(3)))
        print(f'[{name}] {desc}，{len(reqs)} 条')
        print(f'  {"策略":<10}{"命中率":>8}{"P90":>8}{"P99":>8}{"超 SLO":>9}{"驱逐 token":>12}'
              + (f'{"迁移后命中":>9}{"旧模板残留":>8}' if name == 'shift' else ''))
        for pol, kw in pols:
            e = run(reqs, size, policy=pol, batch=bs, **kw)
            pre = [x[1] - x[2] for x in e.per_req]
            over = sum(p > 1024 for p in pre) / len(pre)
            extra = ''
            if name == 'shift':
                late = e.per_req[800:]
                extra = (f'{sum(x[2] for x in late) / sum(x[1] for x in late):>11.1%}'
                         f'{stale_tokens(e.tree, 10 ** 6, 10 ** 6 + 1800):>10,}')
            print(f'  {pol:<10}{e.hit:>9.1%}{pct(pre, .9):>8}{pct(pre, .99):>8}{over:>9.1%}'
                  f'{e.tree.evicted_tokens:>12,}{extra}')
        print()
    print('shift 最后两列：迁移之后 1600 条的命中率；跑完时树里还留着多少旧模板的 token（之后再也没人用）')


# ============================================================================= 场景 4：order
def scenario_order():
    print('8 个模板（每个 1200 token，共 9600）按到达顺序轮流来，800 条，lru，每批 16 条。')
    print('池子 8K 放不下全部模板，20K 放得下。等待队列长度用 max_waiting 控制：128 条以内 lpm 才生效\n')
    rng = random.Random(5)
    tk = Tok()
    tpls = [tk.take(1200) for _ in range(8)]
    reqs = [dict(input=tpls[i % 8] + tk.take(rng.randint(20, 60)), out=rng.randint(30, 200), group=i % 8)
            for i in range(800)]
    print(f'  {"池子":>6}{"顺序":>7}{"队列":>6}{"命中率":>9}{"prefill token":>15}{"驱逐 token":>12}')
    for size in (8_000, 20_000):
        for order, mw in (('fcfs', 128), ('lpm', 800), ('lpm', 128), ('dedup', 128)):
            e = run(reqs, size, order=order, max_waiting=mw)
            print(f'  {size // 1000:>5}K{order:>7}{mw:>6}{e.hit:>9.1%}{e.prefill_tokens:>15,}{e.tree.evicted_tokens:>12,}')
    print('\n  8K + fcfs：8 个模板循环访问、池子只放得下 6 个，LRU 每次都恰好把下一个要用的模板赶走，命中归零。')
    print('  这是 LRU 在循环扫描下的经典病态。lpm 把命中长的请求排前面，同一个模板的请求聚在一起跑，命中回到 95%。')
    print('  但队列一旦超过 128 条，lpm 退回 fcfs（排序本身要对每条请求做一次 match_prefix，队列长了扛不住），')
    print('  「lpm 800」前 672 条和 fcfs 完全一样，只有最后队列排空到 128 条以内才开始排序，14.8% 全来自尾巴。')
    print('  高负载时恰恰是队列最长的时候，lpm 的收益在最需要它的时候消失。')
    print('  20K 池子够大，顺序几乎不影响命中；dedup 把同批共享前缀的请求推到下一批，省掉批内重复 prefill。')


# ============================================================================= 场景 5：nsample
def scenario_nsample():
    print('self-consistency：150 个问题，每个采 8 条（n = 8），问题 300-900 token。池子 64K，每批 16 条。\n')
    reqs = list(wl_selfcons(random.Random(1)))
    print(f'  {"做法":<26}{"命中率":>8}{"prefill token":>15}')
    for warm, desc in ((False, '8 条一起进同一批'), (True, '先发一条 max_new_tokens=0')):
        e = run(reqs, 64_000, warmup=warm)
        print(f'  {desc:<26}{e.hit:>9.1%}{e.prefill_tokens:>15,}')
    print('\n  不预热时，同一批的 8 条在准入时都看不到彼此（前缀还没进树），8 份 prompt 各算一遍；')
    print('  预热把 prompt 先放进树，8 条都能命中 input_len - 1 个 token。tokenizer_manager 对 n > 1 默认就这么做。')


# ============================================================================= 场景 6：pagesize
def scenario_pagesize():
    print('page size 对前缀缓存的影响：fewshot 和 chat 两种负载，池子 64K，lru\n')
    print(f'  {"负载":<9}{"ps":>4}{"命中率":>9}{"节点数":>8}{"平均节点长度":>12}{"树里 token":>11}')
    for name in ('fewshot', 'chat'):
        reqs = list(WORKLOADS[name](random.Random(1)))
        for ps in (1, 16, 64, 128):
            e = run(reqs, 64_000, ps=ps)
            n = e.tree.num_nodes()
            print(f'  {name:<9}{ps:>4}{e.hit:>9.1%}{n:>8}{e.tree.total_size() / max(n, 1):>12.0f}{e.tree.total_size():>11,}')
    print('\n  ps 变大以后匹配按整页截断，每条请求平均丢半页命中；节点数变少，匹配时少走几步。')
    print('  ps 怎么选还要看 attention 后端，FlashInfer / FA3 的约束见 SGL-07。')


# ============================================================================= 场景 7：leak
def _fresh_engine():
    eng = Engine(128, 4)
    sysp = list(range(1000, 1032))                      # 32 token 公共前缀
    a = Req(sysp + [1, 2, 3, 4], rid='A', out_tokens=[901, 902, 903])
    eng.admit(a)
    eng.checkpoint(a)
    eng.decode(a, 3)
    return eng, sysp, a


def scenario_leak():
    print('四种常见 bug，每种都从同一个起点出发：池子 128，page 4，A（36 token 输入，生成 3 个）跑完 prefill 和 decode，')
    print('然后在 release 这一步出错。每种之后看一次空闲时的账（scheduler.py on_idle 里跑的就是这条不变式）\n')

    print('[0] 正确释放')
    eng, sysp, a = _fresh_engine()
    eng.release(a)
    print(f'  {eng.books()}')
    print(f'  空闲时 protected 必须是 0，uncached 必须是 0；token usage = 1 - (available + evictable) / total = '
          f'{1 - (eng.alloc.available_size() + eng.tree.evictable_size()) / eng.alloc.size:.0%}\n')

    print('[1] 忘记解锁（release 漏了 dec_lock_ref）')
    eng, sysp, a = _fresh_engine()
    eng.release(a, forget_unlock=True)
    a_, e_, p_, u_, t_ = eng.invariant()
    print(f'  {eng.books()}')
    print(f'  不变式照样成立：锁住的 {p_} 个 token 记在 protected 里，账是平的，_check_all_pools 抓不到。')
    print(f'  症状是空闲时 token usage = {1 - (a_ + e_) / t_:.0%} 不归零（pool_stats_observer._get_token_info：')
    print('  num_used = total - (available + evictable)），而且这些 token 永远驱逐不掉：')
    got = eng.tree.evict(10 ** 9)
    print(f'  evict(全部) 实际删掉 {got}，树里还剩 {eng.tree.total_size()}，全是被锁住的那条路径')
    print('  每漏一次，可用池子就永久少一截，几小时后表现为 #queue-req 越排越长、token usage 降不下来。')
    print('  定位：空闲时 tree_cache.protected_size() 必须是 0（没有会话缓存时），在测试里断言它最直接；')
    print('  线上看 token usage 在空闲时是否回落到 0\n')

    print('[2] 忘记释放（release 漏了 free [cache_protected_len, 已分配)）')
    eng, sysp, a = _fresh_engine()
    lost = len(a.row) - a.cache_protected_len
    eng.release(a, forget_free=True)
    print(f'  {eng.books()}')
    print(f'  少了 {lost} 个：A 有 KV 的 38 个 token 里，整页的 36 个进了树；第 37、38 个 token 占的那一页不满，')
    print('  不进树，本该由 release 还给分配器，现在它既不在树里、也不在空闲表里、也不属于任何活着的请求。')
    print('  sglang 空闲时报：pool memory leak detected! [full] total=..., available=..., evictable=..., '
          'protected=..., session_held=..., uncached=0')
    print('  （scheduler.py:5370 -> invariant_checker._report_leak；SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE '
          '默认开，直接抛异常）\n')

    print('[3] 重复释放（同一段槽位 free 两次，比如 abort 和正常结束两条路径都走了 release）')
    eng, sysp, a = _fresh_engine()
    tail = a.row[a.cache_protected_len:]
    eng.release(a)
    eng.alloc.free(tail)
    print(f'  {eng.books()}')
    print('  账多出来了：同一页在空闲表里出现两次，available 多算。')
    x = eng.alloc.alloc_pages(1)
    y = eng.alloc.alloc_pages(1)
    print(f'  接下来两次分配拿到 {x} 和 {y}：同一页分给了两条请求，KV 互相覆盖，输出乱码但不报错')
    print('  sglang 打开 SGLANG_CHECK_KV_PAGE_INVARIANTS=1 时报：KV double free: sub-pool 0 has N duplicate pages.')
    print('  （invariant_checker._check_kv_page_invariants；默认关，测试和排障时打开）\n')

    print('[4] 重复解锁（dec_lock_ref 调了两次）')
    eng, sysp, a = _fresh_engine()
    node = a.lock
    eng.release(a)
    eng.tree.dec_lock_ref(node)
    print(f'  {eng.books()}')
    path, n = [], node
    while n is not eng.tree.root:
        path.append(n.lock_ref)
        n = n.parent
    print(f'  账还是平的，但路径上 lock_ref = {path}（radix_cache.py 的 dec_lock_ref 没有 >= 0 断言）')
    b = Req(sysp + [1, 2, 3, 4, 7], rid='B')
    eng.admit(b)
    print(f'  B 命中 {b.prefix_len} 并加锁，lock_ref 从 -1 变 0，没有经过 0 -> 1，所以没挪进 protected：')
    print(f'  {eng.books()}')
    leaves = [lf for lf in eng.tree.evictable_leaves()]
    got = eng.tree.evict(10 ** 9)
    print(f'  B 还在跑，evict(全部) 删掉 {got} 个 token（可驱逐叶子 {len(leaves)} 个），'
          f'B 正在用的前缀槽位 {b.row[:4]}… 已经回到空闲表')
    print('  这是最难查的一种：没有任何账不平，下一条请求拿到这些槽位写入新 KV，B 读到的是别人的 KV。')
    print('  Unified 树的 sanity_check 会报 lock_ref=-1（unified_tree_core.py:3088），经典树没有这道检查；')
    print('  而且它只在 SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK=1（CI 默认开）且模型是 SWA / Mamba 混合时才被调用（invariant_checker.py:484）')


# ============================================================================= main
SCENARIOS = {
    'walk': None, 'hitrate': scenario_hitrate, 'policy': scenario_policy, 'order': scenario_order,
    'nsample': scenario_nsample, 'pagesize': scenario_pagesize, 'leak': scenario_leak,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='walk')
    ap.add_argument('--page-size', type=int, default=None, help='只对 walk 生效，不给时依次跑 1 和 4')
    args = ap.parse_args()
    if args.scenario == 'walk':
        for ps in ([args.page_size] if args.page_size else [1, 4]):
            scenario_walk(ps)
            print()
    else:
        SCENARIOS[args.scenario]()


if __name__ == '__main__':
    main()
