"""SGLang 调度器的 CPU 模拟：waiting / running 两条队列、PrefillAdder 预算、排序策略、chunked prefill 续跑、
new_token_ratio 衰减和 KV 不够时的 retract，日志格式和线上一致。

目的：不用 GPU 也能复现调度器的三类行为，并且数字能和源码逐项对上。
  1. 同一批共享前缀请求，fcfs / lpm / dfs-weight 的缓存命中差多少，为什么队列超过 128 条后 lpm 退化成 fcfs；
  2. schedule_conservativeness 怎么通过 new_token_ratio 改变准入松紧，retract 次数和 token usage 怎么此消彼长；
  3. 打开优先级调度后，高优先级请求怎么抢占在跑请求，被抢占的请求回到队尾重算。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - 主循环：event_loop_normal，没有 overlap。每轮先试 prefill，凑不出来才 decode（get_next_batch_to_run）
  - 准入：PrefillAdder + PrefillBudget。remaining_total = 空闲 + 可驱逐 - total_offset，
          total_offset 先扣在跑请求 min(剩余输出, 4096) x new_token_ratio，每收一条再加 输入 + 1 页 + 输出预留；
          rem_chunk_tokens / rem_input_tokens 两个计算预算；超过 chunk 的请求被截断成 chunked_req，下一轮先续跑
  - 排序：fcfs、lof、lpm、dfs-weight。lpm 带批内前缀去重（阈值 32），队列 > 128 时退回 fcfs
  - 前缀缓存：token 级 radix 树，引用计数加锁，LRU 驱逐叶子；prefill 每步后把算好的部分插进树
  - decode：每条请求每步要 1 个 token（page_size=1）。不够就 retract：按 (已输出, -输入长度) 逆序保留，
            输出最少的先退、同样少时输入最长的先退；退掉的请求释放 KV、重新排到 waiting 队尾，
            new_token_ratio 改成 min(1, (已输出总和 + 20 x 剩余条数) / (max_new 总和 + 1))
  - new_token_ratio：初值 0.7 x conservativeness，每个无 retract 的 decode step 线性降，600 步降到初值的 0.14，空闲时复位
不对齐的部分：
  - 时间：单步耗时取 ../03-bench/sim_serve.py 的 Roofline 下限 / 利用率模型，只用来给日志算吞吐和 TTFT，没有性能意义
  - 没有 overlap、mixed chunk、page_size > 1、HiCache、投机解码、多 DP
  - req_to_token 行数按 max_running_requests 算，不单独建池

用法：
    python3 sim_scheduler.py --scenario policy                 # fcfs / lpm / dfs-weight 对比
    python3 sim_scheduler.py --scenario policy --num 400       # 队列超过 128，lpm 退化
    python3 sim_scheduler.py --scenario policy --interleaved   # 各组轮流到达，同批无重复前缀
    python3 sim_scheduler.py --scenario ratio                  # conservativeness 0.3 / 1.0 / 1.3
    python3 sim_scheduler.py --scenario priority               # 优先级抢占
    python3 sim_scheduler.py --workload random --num 300 --max-total-tokens 60000 --log --log-head 30
"""
import argparse
import heapq
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
from sim_ledger import ledger  # noqa: E402
from sim_serve import Server  # noqa: E402

# environ.py 与 schedule_policy.py 的默认值
INIT_NEW_TOKEN_RATIO = 0.7
MIN_NEW_TOKEN_RATIO_FACTOR = 0.14
NEW_TOKEN_RATIO_DECAY_STEPS = 600
RETRACT_DECODE_STEPS = 20
CLIP_MAX_NEW_TOKENS = 4096
IN_BATCH_CHECK = 32
IN_BATCH_DEPRIORITIZE = 32
WAITING_QUEUE_PREFIX_MATCH_MAX = 128
DECODE_LOG_INTERVAL = 40
PAGE = 1

CONTINUE, NO_TOKEN, OTHER = 'CONTINUE', 'NO_TOKEN', 'OTHER'


# ---------------------------------------------------------------- radix 树
class Node:
    __slots__ = ('key', 'children', 'parent', 'lock', 't')

    def __init__(self, key=(), parent=None):
        self.key, self.children, self.parent, self.lock, self.t = key, {}, parent, 0, 0


class Radix:
    """token 级前缀树。evictable / protected 两个计数和 RadixCache 同口径：lock 为 0 的节点可驱逐。"""

    def __init__(self, pool=None):
        self.root = Node()
        self.root.lock = 1
        self.pool = pool
        self.evictable = 0
        self.protected = 0
        self.clock = 0

    @property
    def size(self):
        return self.evictable + self.protected

    def _split(self, c, n):
        p = c.parent
        new = Node(c.key[:n], p)
        new.lock, new.t = c.lock, c.t
        p.children[new.key[0]] = new
        c.key = c.key[n:]
        c.parent = new
        new.children[c.key[0]] = c
        return new

    def _walk(self, toks, split=True):
        node, i = self.root, 0
        self.clock += 1
        while i < len(toks):
            c = node.children.get(toks[i])
            if c is None:
                break
            k = c.key
            n = 0
            m = min(len(k), len(toks) - i)
            while n < m and k[n] == toks[i + n]:
                n += 1
            if n < len(k):
                if not split:
                    break
                c = self._split(c, n)
            c.t = self.clock
            node, i = c, i + n
        return i, node

    def match(self, toks):
        return self._walk(toks)

    def insert(self, toks):
        """返回 (末端节点, 新增 token 数)。已有的部分不重复计数，调用方据此释放重复算出来的 KV。"""
        i, node = self._walk(toks)
        if i == len(toks):
            return node, 0
        leaf = Node(tuple(toks[i:]), node)
        leaf.t = self.clock
        node.children[leaf.key[0]] = leaf
        self.evictable += len(leaf.key)
        return leaf, len(leaf.key)

    def lock_node(self, node):
        while node is not self.root:
            if node.lock == 0:
                self.evictable -= len(node.key)
                self.protected += len(node.key)
            node.lock += 1
            node = node.parent

    def unlock_node(self, node):
        while node is not self.root:
            node.lock -= 1
            if node.lock == 0:
                self.evictable += len(node.key)
                self.protected -= len(node.key)
            node = node.parent

    def evict(self, num):
        """LRU 驱逐没加锁的叶子，直到腾出 num 个 token 或没得可驱。"""
        heap, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children.values())
            elif n is not self.root and n.lock == 0:
                heap.append((n.t, id(n), n))
        heapq.heapify(heap)
        freed = 0
        while heap and freed < num:
            _, _, n = heapq.heappop(heap)
            p = n.parent
            del p.children[n.key[0]]
            self.evictable -= len(n.key)
            freed += len(n.key)
            if not p.children and p is not self.root and p.lock == 0:
                heapq.heappush(heap, (p.t, id(p), p))
        self.pool.free += freed
        return freed


def dfs_weight_order(root, nodes):
    """base_prefix_cache._dfs_weight_order：权重 = 子树里挂着的等待请求数，重的子树先走，后序输出。"""
    at = {}
    for i, n in enumerate(nodes):
        at.setdefault(n, []).append(i)
    w = {n: len(v) for n, v in at.items()}

    def weigh(n):
        s = w.get(n, 0) + sum(weigh(c) for c in n.children.values())
        w[n] = s
        return s

    weigh(root)
    order = []

    def emit(n):
        for c in sorted(n.children.values(), key=lambda c: -w.get(c, 0)):
            emit(c)
        order.extend(at.get(n, ()))

    sys.setrecursionlimit(100000)
    emit(root)
    return order


# ---------------------------------------------------------------- 请求与池
class Pool:
    def __init__(self, cap):
        self.cap, self.free = cap, cap
        self.tree = None

    def ensure(self, n):
        """allocator.evict_to_free_tokens：空闲不够才驱逐。"""
        if self.free < n:
            self.tree.evict(n - self.free)
        return self.free >= n

    def alloc(self, n):
        if not self.ensure(n):
            raise RuntimeError(f'KV 池分配失败：要 {n}，空闲 {self.free}（准入预算出了漏洞）')
        self.free -= n


class Req:
    def __init__(self, rid, prompt, max_new, target, arr=0.0, priority=None):
        self.rid, self.prompt, self.max_new, self.target = rid, tuple(prompt), max_new, target
        self.arr, self.priority = arr, priority
        self.out = 0
        self.prefix_len = 0       # 已在树里且被本请求锁住的前缀长度
        self.node = None          # 锁住的末端节点
        self.owned = 0            # 本请求持有、还没进树的 KV
        self.extend_end = 0
        self.wait_key = None
        self.ttft = None
        self.end = None
        self.retractions = 0
        self.stain = False        # retracted_stain：重算的 token 单独计数
        self.aborted = False
        self.matched = 0          # num_matched_prefix_tokens，排序用

    def fill(self):
        return self.prompt + tuple(-1 - self.rid * 1_000_000 - j for j in range(self.out))

    def seqlen(self):
        return len(self.prompt) + self.out

    def finished(self):
        return self.aborted or self.out >= self.target or self.out >= self.max_new


# ---------------------------------------------------------------- PrefillAdder
class Adder:
    def __init__(self, s, running, ratio):
        self.s, self.pool, self.tree = s, s.pool, s.tree
        self.ratio = ratio
        self.rem_input = s.max_prefill_tokens
        self.rem_chunk = s.chunk
        self.total_offset = sum(self.offset(r) for r in running)
        self.current_offset = 0
        self.can_run, self.preempt_list = [], []
        self.new_chunked = None
        self.log_hit = self.log_input = self.reproc_input = 0

    def offset(self, r):
        """_get_running_request_total_token_offset"""
        return min(r.max_new - r.out, CLIP_MAX_NEW_TOKENS) * self.ratio

    def avail(self):
        return self.pool.free + self.tree.evictable

    @property
    def rem_total(self):
        return self.avail() - self.total_offset

    def budget_state(self):
        if not (self.rem_total > 0 and self.avail() - self.current_offset > 0):
            return NO_TOKEN
        if self.rem_input <= 0 or (self.rem_chunk is not None and self.rem_chunk <= 0):
            return OTHER
        return CONTINUE

    def _commit(self, r, extend, max_new):
        imm = extend + PAGE
        self.total_offset += imm + max_new
        self.current_offset += imm
        self.rem_input -= extend
        if self.rem_chunk is not None:
            self.rem_chunk -= extend
        self.log_hit += r.prefix_len
        self.log_input += extend
        if r.stain:
            self.reproc_input += extend

    def add_chunked_req(self, r):
        avail = min(self.rem_chunk, int(self.rem_total))
        if avail <= 0:
            avail = self.rem_chunk        # 单池续跑必须往前走，才能放掉它的 KV
        cand = len(r.fill()) - r.prefix_len
        truncated = cand > avail
        new_len = min(cand, avail)
        r.extend_end = r.prefix_len + new_len
        self.can_run.append(r)
        self._commit(r, new_len, 0 if truncated else min(r.max_new, CLIP_MAX_NEW_TOKENS))
        return r if truncated else None

    def add_one_req(self, r):
        max_new = min(max(r.max_new - r.out, 0), CLIP_MAX_NEW_TOKENS)
        extend = len(r.fill()) - r.prefix_len
        total = extend + max_new + PAGE
        self.tree.lock_node(r.node)                     # _lock_node：自己的前缀不算可驱逐
        ok = total < self.rem_total
        self.tree.unlock_node(r.node)
        if not ok:
            return NO_TOKEN
        if self.rem_chunk is None and self.can_run and extend >= self.rem_input:
            return OTHER
        max_new = min(r.max_new, CLIP_MAX_NEW_TOKENS)
        if self.rem_chunk is not None and extend > self.rem_chunk:
            extend = self.rem_chunk // PAGE * PAGE
            if extend <= 0:
                return OTHER
            max_new = 0
            self.new_chunked = r
        r.extend_end = r.prefix_len + extend
        self.tree.lock_node(r.node)
        self.can_run.append(r)
        self._commit(r, extend, max_new)
        return self.budget_state()

    def preempt_to_schedule(self, r, running, s):
        """schedule_policy.preempt_to_schedule：只抢优先级差 > 阈值的，低优先级、后到的先抢。"""
        cands = sorted((x for x in running if x not in self.preempt_list and not x.finished()),
                       key=lambda x: (x.priority, -x.wait_key[0], -x.wait_key[1]))
        need = len(r.fill()) - r.prefix_len + min(r.max_new, CLIP_MAX_NEW_TOKENS) - self.rem_total
        picked = []
        for x in cands:
            if r.priority - x.priority > s.preempt_threshold:
                picked.append(x)
                need -= self.offset(x)
                if need <= 0:
                    break
            else:
                break
        if not picked or need > 0:
            return False
        keep = [x for x in running if x not in picked]
        for i, x in enumerate(picked):
            self.total_offset -= self.offset(x)
            s.release_req(x, len(running) - i - 1)
            s.stats['preempted'] += 1
            s.log(f'Preempt req {x.rid} (priority {x.priority}) for req {r.rid} (priority {r.priority})')
        running[:] = keep
        self.preempt_list.extend(picked)
        return True


# ---------------------------------------------------------------- 调度器
class Scheduler:
    def __init__(self, a, srv):
        self.srv = srv
        L = ledger(a.model, a.gpu)
        cap = a.max_total_tokens or L['tokens']
        self.pool = Pool(cap)
        self.tree = Radix(self.pool)
        self.pool.tree = self.tree
        self.cap = cap
        self.chunk = a.chunked_prefill_size or L['chunk']
        if self.chunk <= 0:
            self.chunk = None
        self.max_prefill_tokens = a.max_prefill_tokens
        ctx = L['model'].context
        est = max(min(int(cap / ctx * 512), 4096), 2048)       # kv_cache_configurator 的估计
        self.max_running = min(a.max_running_requests, cap // 2) if a.max_running_requests else min(est, cap // 2)
        self.graph_max_bs = L['max_bs']
        self.policy = a.schedule_policy
        self.prio = a.enable_priority_scheduling
        self.preempt = self.prio and not a.disable_priority_preemption
        self.preempt_threshold = a.priority_scheduling_preemption_threshold
        init = min(INIT_NEW_TOKEN_RATIO * a.schedule_conservativeness, 1.0)
        self.r_init, self.r_min = init, min(init * MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
        self.r_decay = (self.r_init - self.r_min) / NEW_TOKEN_RATIO_DECAY_STEPS
        self.ratio = init
        self.waiting, self.running = [], []
        self.chunked = None
        self.batch_is_full = False
        self.seq = 0
        self.now = 0.0
        self.decode_ct = 0
        self.last_prefill_t = self.last_decode_t = 0.0
        self.gen_since = 0
        self.lines = []
        self.trace = []                # (t, ratio, usage, running)，画图用
        self.verbose = a.log
        self.stats = dict(prefill_steps=0, decode_steps=0, new_tok=0, hit_tok=0, reproc_tok=0, retract_events=0,
                          retracted=0, preempted=0, aborted=0, usage_sum=0.0, usage_n=0, max_running=0,
                          ratio_min=init)

    # -- 工具
    def log(self, msg):
        line = f'[t={self.now:8.3f}s] {msg}'
        self.lines.append(line)
        if self.verbose:
            print(line)

    def usage(self):
        return (self.cap - self.pool.free - self.tree.evictable) / self.cap

    def enqueue(self, r):
        """_add_request_to_queue：追加到队尾，打等待时间戳。"""
        if self.prio and r.priority is None:
            r.priority = -sys.maxsize - 1
        self.seq += 1
        r.wait_key = (self.now, self.seq)
        self.waiting.append(r)

    def match(self, r):
        """init_next_round_input：重新匹配前缀。最长只能匹配到 len-1，留一个 token 算 logits。"""
        f = r.fill()
        n, node = self.tree.match(f[:len(f) - 1])
        return n, node

    def checkpoint(self, r, upto):
        """把 [prefix_len, upto) 这段刚算好的 KV 插进树，锁转移到新末端。别人已经插过的部分属于重复计算，直接释放。"""
        node, new = self.tree.insert(r.fill()[:upto])
        moved = upto - r.prefix_len
        r.owned -= moved
        self.pool.free += moved - new
        self.tree.lock_node(node)
        if r.node is not None:
            self.tree.unlock_node(r.node)
        r.node, r.prefix_len = node, upto

    def release_finished(self, r):
        """release_kv_cache(checkpoint=True)：整条序列进树，解锁，留给后来者命中。"""
        kv = r.prefix_len + r.owned
        node, new = self.tree.insert(r.fill()[:kv])
        self.pool.free += r.owned - new
        r.owned = 0
        self.tree.unlock_node(r.node)
        r.node = None
        r.end = self.now

    def release_req(self, r, remaining):
        """schedule_batch.release_req + reset_for_retract：KV 不进树直接还池；输出保留，下次 prefill 把输入+已输出一起重算。"""
        self.pool.free += r.owned
        r.owned = 0
        if r.node is not None:
            self.tree.unlock_node(r.node)
        self.pool.ensure(remaining * RETRACT_DECODE_STEPS)
        r.node, r.prefix_len, r.extend_end = None, 0, 0
        r.retractions += 1
        r.stain = True

    # -- 排序
    def calc_priority(self):
        q = self.waiting
        pol = self.policy
        if pol == 'lpm' and len(q) > WAITING_QUEUE_PREFIX_MATCH_MAX:
            pol = 'fcfs'
        if pol == 'fcfs':
            if self.prio:
                q.sort(key=lambda r: (-r.priority, r.wait_key))
            return
        if pol == 'lof':
            q.sort(key=lambda r: ((-r.priority) if self.prio else 0, -r.max_new))
            return
        # cache-aware：先对真树匹配，再做批内去重
        depri = set()
        sim = Radix()
        for r in q:
            n, node = self.match(r)
            r.matched, r.last = n, node
            if n <= IN_BATCH_CHECK:
                f = r.fill()
                m, _ = sim._walk(f, split=False)
                if m >= IN_BATCH_DEPRIORITIZE:
                    depri.add(r.rid)
                else:
                    sim.insert(f)
        if pol == 'lpm':
            q.sort(key=lambda r: -r.matched if r.rid not in depri else math.inf)
        elif pol == 'dfs-weight':
            order = dfs_weight_order(self.tree.root, [r.last for r in q])
            q[:] = [q[i] for i in order]

    # -- get_new_batch_prefill
    def get_new_batch_prefill(self):
        if self.preempt or not self.running:
            self.batch_is_full = False
        if (self.batch_is_full or not self.waiting) and self.chunked is None:
            return None
        if self.max_running - len(self.running) <= 0 and self.chunked is None and not self.preempt:
            self.batch_is_full = True
            return None
        self.calc_priority()
        ad = Adder(self, self.running, self.ratio)
        if self.chunked is not None:
            r = self.chunked
            n, node = self.match(r)
            if n > r.prefix_len:                 # 别的请求已经把后面的部分算进树了
                self.tree.lock_node(node)
                self.tree.unlock_node(r.node)
                r.node, r.prefix_len = node, n
            self.chunked = ad.add_chunked_req(r)
        for r in self.waiting:
            if ad.budget_state() == OTHER:
                break
            if len(ad.can_run) >= self.max_running - len(self.running):
                self.batch_is_full = True
            if self.batch_is_full:
                if not self.preempt or not ad.preempt_to_schedule(r, self.running, self):
                    break
            r.prefix_len, r.node = self.match(r)
            res = ad.add_one_req(r)
            if res != CONTINUE:
                if res == NO_TOKEN:
                    self.batch_is_full = len(ad.can_run) > 0 or len(self.running) > 0
                break
        if not ad.can_run:
            for r in ad.preempt_list:
                self.enqueue(r)
            return None
        chosen = set(id(r) for r in ad.can_run)
        self.waiting = [r for r in self.waiting if id(r) not in chosen]
        for r in ad.preempt_list:
            self.enqueue(r)
        if ad.new_chunked is not None:
            self.chunked = ad.new_chunked
        ad.pending = sum(r.seqlen() for r in self.waiting)
        if self.chunked is not None:
            c = self.chunked
            ad.pending += c.seqlen() - c.extend_end
        return ad

    # -- update_running_batch
    def check_decode_mem(self, reqs):
        return self.pool.ensure(len(reqs) * PAGE)

    def update_running_batch(self):
        batch = self.running
        if not self.check_decode_mem(batch):
            old_free, old_ratio = self.pool.free, self.ratio
            order = sorted(range(len(batch)), key=lambda i: (batch[i].out, -len(batch[i].prompt)), reverse=True)
            retracted = []
            first = True
            while first or not self.check_decode_mem([batch[i] for i in order]):
                if len(order) == 1:
                    break
                first = False
                r = batch[order.pop()]
                self.release_req(r, len(order))
                retracted.append(r)
            if len(order) <= 1 and not self.check_decode_mem([batch[i] for i in order]):
                r = batch[order.pop()]
                self.release_req(r, 0)
                r.aborted = True
                r.end = self.now
                self.stats['aborted'] += 1
                self.log(f'retract_decode: aborted last request {r.rid} due to OOM')
            keep = sorted(order)
            self.running = [batch[i] for i in keep]
            rs = self.running
            self.ratio = min(1.0, (sum(r.out for r in rs) + RETRACT_DECODE_STEPS * len(rs))
                             / (sum(r.max_new for r in rs) + 1))
            self.stats['retract_events'] += 1
            self.stats['retracted'] += len(retracted)
            self.log(f'KV cache pool is full. Retract requests. #retracted_reqs: {len(retracted)}, '
                     f'#new_tokens_gained: {self.pool.free - old_free}, '
                     f'#new_token_ratio: {old_ratio:.4f} -> {self.ratio:.4f}')
            for r in retracted:
                self.enqueue(r)
            self.batch_is_full = False
        else:
            self.ratio = max(self.ratio - self.r_decay, self.r_min)
        self.stats['ratio_min'] = min(self.stats['ratio_min'], self.ratio)
        return self.running

    # -- 执行与结果处理
    def run_prefill(self, ad):
        parts = []
        for r in ad.can_run:
            ext = r.extend_end - r.prefix_len
            self.pool.alloc(ext)
            r.owned += ext
            parts.append((ext, r.prefix_len))
        dt = self.srv.t_prefill(parts)
        gap = self.now + dt - self.last_prefill_t
        self.last_prefill_t = self.now + dt
        st = self.stats
        st['prefill_steps'] += 1
        st['new_tok'] += ad.log_input
        st['hit_tok'] += ad.log_hit
        st['reproc_tok'] += ad.reproc_input
        self.log(f'Prefill batch, #new-seq: {len(ad.can_run)}, #new-token: {ad.log_input}, '
                 f'#cached-token: {ad.log_hit}, token usage: {self.usage():.2f}, '
                 f'#running-req: {len(self.running)}, #queue-req: {len(self.waiting)}, '
                 f'#pending-token: {ad.pending}, cuda graph: False, '
                 f'input throughput (token/s): {ad.log_input / gap if gap > 0 else 0:.2f}')
        self.now += dt
        for r in ad.can_run:
            if r is self.chunked:
                self.checkpoint(r, r.extend_end)
                continue
            self.checkpoint(r, r.extend_end)
            r.out += 1
            if r.ttft is None:
                r.ttft = self.now - r.arr
            if r.finished():
                self.release_finished(r)
            else:
                self.running.append(r)
        st['max_running'] = max(st['max_running'], len(self.running))

    def run_decode(self):
        batch = self.running
        if not batch:
            return
        self.pool.alloc(len(batch) * PAGE)
        for r in batch:
            r.owned += PAGE
        dt = self.srv.t_decode([r.prefix_len + r.owned for r in batch])
        self.now += dt
        self.decode_ct += 1
        st = self.stats
        st['decode_steps'] += 1
        st['usage_sum'] += self.usage()
        st['usage_n'] += 1
        self.trace.append((self.now, self.ratio, self.usage(), len(batch)))
        self.gen_since += len(batch)
        if self.decode_ct % DECODE_LOG_INTERVAL == 0:
            gap = self.now - self.last_decode_t
            self.log(f'Decode batch, #running-req: {len(batch)}, '
                     f'#token: {self.cap - self.pool.free - self.tree.evictable}, '
                     f'token usage: {self.usage():.2f}, cuda graph: {len(batch) <= self.graph_max_bs}, '
                     f'gen throughput (token/s): {self.gen_since / gap if gap > 0 else 0:.2f}, '
                     f'#queue-req: {len(self.waiting)}')
            self.last_decode_t, self.gen_since = self.now, 0
        keep = []
        for r in batch:
            r.out += 1
            if r.finished():
                self.release_finished(r)
            else:
                keep.append(r)
        if len(keep) < len(batch):
            self.batch_is_full = False
        self.running = keep

    def step(self):
        """get_next_batch_to_run + run_batch + process_batch_result。返回 False 表示这一轮空闲。"""
        ad = self.get_new_batch_prefill()
        if ad is not None:
            self.run_prefill(ad)
            return True
        if self.running:
            self.update_running_batch()
            self.run_decode()
            return True
        self.ratio = self.r_init       # on_idle 复位
        return False


# ---------------------------------------------------------------- 负载与运行
def make_shared(num, groups, prefix_len, suffix_len, out, clumped, seed):
    rng = random.Random(seed)
    sys_prompts = [[10_000_000 + g * 100_000 + j for j in range(prefix_len)] for g in range(groups)]
    gid = [i * groups // num for i in range(num)] if clumped else [i % groups for i in range(num)]
    reqs = []
    for i in range(num):
        suffix = [rng.randrange(1, 9_000_000) for _ in range(suffix_len)]
        reqs.append(Req(i, sys_prompts[gid[i]] + suffix, out, out))
    return reqs


def make_random(num, inp, out, ratio, rate, seed):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(num):
        n = rng.randint(max(int(inp * ratio), 1), inp)
        target = rng.randint(max(int(out * ratio), 1), out)
        reqs.append(Req(i, [rng.randrange(1, 9_000_000) for _ in range(n)], out, target, arr=t))
        if rate != math.inf:
            t += rng.expovariate(rate)
    return reqs


def simulate(a, reqs):
    srv = Server(a.model, a.gpu)
    s = Scheduler(a, srv)
    pending = sorted(reqs, key=lambda r: r.arr)
    k = 0
    while True:
        while k < len(pending) and pending[k].arr <= s.now + 1e-12:
            s.enqueue(pending[k])
            k += 1
        if not s.step():
            if k < len(pending):
                s.now = max(s.now, pending[k].arr)
                continue
            if not s.waiting and s.chunked is None:
                break
    return s


def summary(s, reqs):
    st = s.stats
    done = [r for r in reqs if r.end is not None and not r.aborted]
    tt = sorted(r.ttft for r in done)
    out_tok = sum(r.out for r in done)
    hit = st['hit_tok'] / max(st['hit_tok'] + st['new_tok'], 1)
    return dict(
        done=len(done), aborted=st['aborted'], time=s.now, out_tps=out_tok / s.now if s.now else 0,
        ttft_mean=sum(tt) / len(tt) if tt else 0, ttft_p99=tt[min(len(tt) - 1, int(len(tt) * 0.99))] if tt else 0,
        new_tok=st['new_tok'], hit_tok=st['hit_tok'], hit=hit, reproc=st['reproc_tok'],
        prefill_steps=st['prefill_steps'], decode_steps=st['decode_steps'],
        retract_events=st['retract_events'], retracted=st['retracted'], preempted=st['preempted'],
        usage=st['usage_sum'] / max(st['usage_n'], 1), max_running=st['max_running'], ratio_min=st['ratio_min'],
        max_running_cap=s.max_running, cap=s.cap, chunk=s.chunk)


def header(s, a):
    print(f'# {a.model} @ {a.gpu}：KV 池 {s.cap} token，chunked_prefill_size {s.chunk}，'
          f'max_prefill_tokens {s.max_prefill_tokens}，max_running_requests {s.max_running}，'
          f'new_token_ratio 初值 {s.r_init:.3f} 下限 {s.r_min:.4f} 每步降 {s.r_decay:.6f}')


def run_one(a, reqs, show=True):
    s = simulate(a, reqs)
    if show and not a.log and a.log_head:
        for line in s.lines[:a.log_head]:
            print(line)
    return s, summary(s, reqs)


def scenario_policy(a):
    print(f'# 共享前缀负载：{a.num} 条请求，{a.groups} 组 system prompt 各 {a.prefix_len} token，'
          f'私有后缀 {a.suffix_len} token，输出 {a.out}，{"同组扎堆到达" if a.clumped else "各组轮流到达"}，全部 t=0 发出')
    rows = []
    for pol in ['fcfs', 'lpm', 'dfs-weight']:
        a.schedule_policy = pol
        reqs = make_shared(a.num, a.groups, a.prefix_len, a.suffix_len, a.out, a.clumped, a.seed)
        s, m = run_one(a, reqs, show=False)
        rows.append((pol, m))
    header(s, a)
    print(f'{"policy":>11} {"prefill步":>8} {"新算token":>10} {"命中token":>10} {"命中率":>7} '
          f'{"TTFT均值s":>9} {"TTFT p99 s":>10} {"总时长s":>8}')
    for pol, m in rows:
        print(f'{pol:>11} {m["prefill_steps"]:>9} {m["new_tok"]:>11} {m["hit_tok"]:>11} {m["hit"]:>8.1%} '
              f'{m["ttft_mean"]:>10.3f} {m["ttft_p99"]:>10.3f} {m["time"]:>9.2f}')
    ideal = a.groups * a.prefix_len + a.num * a.suffix_len
    print(f'# 理想下限：每组前缀只算一次 = {a.groups} x {a.prefix_len} + {a.num} x {a.suffix_len} = {ideal} token')
    if a.num > WAITING_QUEUE_PREFIX_MATCH_MAX:
        print(f'# 队列 {a.num} > {WAITING_QUEUE_PREFIX_MATCH_MAX}：lpm 前几轮按 fcfs 排，'
              f'直到队列降到 {WAITING_QUEUE_PREFIX_MATCH_MAX} 以下')


def scenario_ratio(a):
    print(f'# 长输出负载：{a.num} 条请求，输入 [{int(a.inp * a.range_ratio)}, {a.inp}]，max_new_tokens {a.out}，'
          f'实际输出 [{int(a.out * a.range_ratio)}, {a.out}] 均匀，全部 t=0 发出')
    rows = []
    for c in [0.3, 1.0, 1.3]:
        a.schedule_conservativeness = c
        reqs = make_random(a.num, a.inp, a.out, a.range_ratio, math.inf, a.seed)
        s, m = run_one(a, reqs, show=False)
        rows.append((c, s, m))
        if c == 0.3:
            first = [l for l in s.lines if 'Retract' in l][:3]
    header(rows[1][1], a)
    print(f'{"conserv":>7} {"ratio初值":>8} {"ratio最低":>8} {"retract次":>9} {"退回请求":>8} {"重算token":>9} '
          f'{"decode usage":>12} {"峰值running":>10} {"总时长s":>8} {"输出tok/s":>9}')
    for c, s, m in rows:
        print(f'{c:>7} {s.r_init:>9.3f} {m["ratio_min"]:>9.4f} {m["retract_events"]:>10} {m["retracted"]:>9} '
              f'{m["reproc"]:>10} {m["usage"]:>12.2f} {m["max_running"]:>11} {m["time"]:>9.2f} {m["out_tps"]:>10.0f}')
    if first:
        print('# conservativeness 0.3 的前三条 retract 日志：')
        for l in first:
            print(l)


def scenario_priority(a):
    print('# 优先级抢占：max_running_requests=8，t=0 来 8 条低优先级（priority=0）长请求，'
          't=0.5s 来 1 条高优先级（priority=100）')
    for disable in [True, False]:
        a.enable_priority_scheduling = True
        a.disable_priority_preemption = disable
        a.max_running_requests = 8
        reqs = make_random(8, 2048, 1024, 1.0, math.inf, a.seed)
        for r in reqs:
            r.priority = 0
        vip = Req(8, [random.Random(1).randrange(1, 9_000_000) for _ in range(512)], 64, 64, arr=0.5, priority=100)
        reqs.append(vip)
        s, m = run_one(a, reqs, show=False)
        tag = '关闭抢占' if disable else '打开抢占'
        lows = [r for r in reqs if r.priority == 0]
        print(f'\n## {tag}：高优先级 TTFT {vip.ttft:.3f}s，完成于 {vip.end:.2f}s；'
              f'被抢占 {m["preempted"]} 次，低优先级重算 {m["reproc"]} token，'
              f'低优先级最晚完成 {max(r.end for r in lows):.2f}s')
        shown = [l for l in s.lines
                 if 'Preempt' in l or ('Prefill' in l and float(l.split('t=')[1].split('s]')[0]) >= 0.4)]
        for line in shown[:3]:
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', choices=['policy', 'ratio', 'priority', 'custom'], default='custom')
    ap.add_argument('--model', default='qwen3-8b')
    ap.add_argument('--gpu', default='h100-sxm')
    ap.add_argument('--max-total-tokens', type=int, default=None, help='缩小 KV 池，制造内存压力')
    ap.add_argument('--chunked-prefill-size', type=int, default=None, help='默认按显存档位，H100 为 8192；-1 关闭')
    ap.add_argument('--max-prefill-tokens', type=int, default=16384)
    ap.add_argument('--max-running-requests', type=int, default=None)
    ap.add_argument('--schedule-policy', default='fcfs', choices=['fcfs', 'lof', 'lpm', 'dfs-weight'])
    ap.add_argument('--schedule-conservativeness', type=float, default=1.0)
    ap.add_argument('--enable-priority-scheduling', action='store_true')
    ap.add_argument('--disable-priority-preemption', action='store_true')
    ap.add_argument('--priority-scheduling-preemption-threshold', type=int, default=10)
    ap.add_argument('--workload', choices=['shared', 'random'], default='shared')
    ap.add_argument('--num', type=int, default=None)
    ap.add_argument('--groups', type=int, default=16)
    ap.add_argument('--prefix-len', type=int, default=1024)
    ap.add_argument('--suffix-len', type=int, default=128)
    ap.add_argument('--interleaved', dest='clumped', action='store_false', help='各组轮流到达（默认同组扎堆）')
    ap.add_argument('--inp', type=int, default=1024)
    ap.add_argument('--out', type=int, default=None)
    ap.add_argument('--range-ratio', type=float, default=0.5)
    ap.add_argument('--request-rate', type=float, default=math.inf)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log', action='store_true', help='打印全部调度日志')
    ap.add_argument('--log-head', type=int, default=0, help='只打印前 N 行日志')
    a = ap.parse_args()

    if a.scenario == 'policy':
        a.num = a.num or 128
        a.out = a.out or 64
        if a.max_total_tokens is None:
            a.max_total_tokens = 60000
        scenario_policy(a)
        return
    if a.scenario == 'ratio':
        a.num = a.num or 400
        a.out = a.out or 1024
        if a.max_total_tokens is None:
            a.max_total_tokens = 120000
        scenario_ratio(a)
        return
    if a.scenario == 'priority':
        scenario_priority(a)
        return

    a.num = a.num or 100
    a.out = a.out or 128
    if a.workload == 'shared':
        reqs = make_shared(a.num, a.groups, a.prefix_len, a.suffix_len, a.out, a.clumped, a.seed)
    else:
        reqs = make_random(a.num, a.inp, a.out, a.range_ratio, a.request_rate, a.seed)
    s, m = run_one(a, reqs)
    header(s, a)
    for k, v in m.items():
        print(f'{k:>16}: {v:.4f}' if isinstance(v, float) else f'{k:>16}: {v}')


if __name__ == '__main__':
    main()
