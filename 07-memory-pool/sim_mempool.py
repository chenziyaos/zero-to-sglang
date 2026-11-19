"""SGLang 两级 KV 内存池的 CPU 复刻：req_to_token（请求 -> token 槽位）和 token_to_kv（槽位 -> KV 字节），
外加 page size 从 1 到 128 时分配、释放、前缀缓存和启动账本分别怎么变。

目的：不装 GPU，把下面几件事一步步打印出来，每一步都能对上源码行号。
  1. 一条请求的 KV 放在哪里：req_to_token[row, pos] 存的是槽位号，槽位号 // page_size 是页号，
     attention 后端拿 req_to_token[:, ::page_size] // page_size 当页表；
  2. 分页分配器的三段式 alloc_extend、decode 什么时候才要新页、释放时为什么以页为单位；
  3. page size 变大以后：尾页浪费、前缀缓存丢掉的尾巴、页表宽度、每步要新页的频率；
  4. 启动时 max_total_num_tokens、max_running_requests、req_to_token 表各是怎么算出来的；
  5. 几条 OOM 报错分别由哪一层抛出，该调哪个参数。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - ReqToTokenPool：int32 表 (size + 1, context_len + 4)，第 0 行留给 CUDA Graph 的 padding；
                    free_slots 从尾部弹出、释放时追加到尾部；每次分配 generation + 1
  - TokenToKVPoolAllocator（page 1）：free_pages = arange(1, size + 1)，槽位 0 留作 padding；
                    从头部取，释放追加到尾部；need_sort 时先进 release_pages，不够用时合并排序
  - PagedTokenToKVPoolAllocator：页 0 留作 padding；alloc_extend 先用 extend // ps + bs + 1 做上界，
                    上界超过空闲页时才算精确页数；三段式填槽位；alloc_decode 只在 seq_len % ps == 1 时取新页；
                    free 对槽位号 // ps 去重后整页归还，插回空闲表头部；need_sort 时进 staged_pages
  - 释放：release_kv_cache 只释放 [cache_protected_len, owned) 这一段，前缀树拿走的是向下取整到页的部分，
          超额分配的部分从 ceil_align(owned, ps) 开始还
  - 前缀缓存：RadixKey.page_aligned 把 key 向下取整到整页，不满一页的尾巴不进树
  - 启动账本：复用 ../02-ledger/sim_ledger.py；token 数向下取整到页；max_running_requests 按
             resolve_max_num_reqs；KV buffer 每层 (size + page_size, heads, head_dim)
不对齐的部分：
  - 不模拟 GPU 上的 Triton kernel，只按同样的规则逐请求算槽位
  - 前缀树只做「按整页匹配」这一件事，驱逐、锁、分裂留给 SGL-08
  - 不覆盖 SWA / Mamba / MLA 专用池、HiCache、DCP 与 page interleave 分片、投机解码的超额分配

用法：
    python3 sim_mempool.py                         # walk：两条请求走一遍两级映射，page 1 和 page 4 各一次
    python3 sim_mempool.py --scenario walk --page-size 16
    python3 sim_mempool.py --scenario pagesize     # 三种负载下 page 1/16/32/64/128 的代价
    python3 sim_mempool.py --scenario sizing       # 各模型 / GPU 的池子尺寸和启动日志
    python3 sim_mempool.py --scenario churn        # 反复分配释放后 KV 槽位的连续性，need_sort 的作用
    python3 sim_mempool.py --scenario errors       # 四条 OOM / 启动报错各自怎么触发
"""
import argparse
import math
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
from common.specs import KV_DTYPE, MODELS, SMI_MIB  # noqa: E402
from sim_ledger import ledger  # noqa: E402

GB = 1 << 30
EXTRA_CONTEXT = 4  # get_req_to_token_extra_context_len：不开投机解码时是 4


def cdiv(a, b):
    return -(-a // b)


def ceil_align(x, ps):
    return cdiv(x, ps) * ps


# ---------------------------------------------------------------- 第一级：请求 -> 槽位
class ReqToTokenPool:
    """memory_pool.py:376。表是 (size + 1) x max_context_len 的 int32，这里只存写过的位置。"""

    def __init__(self, size, max_context_len):
        self.size = size
        self.max_context_len = max_context_len
        self.rows = {}
        self.free_slots = list(range(1, size + 1))
        self.generation = [0] * (size + 1)

    def nbytes(self):
        return (self.size + 1) * self.max_context_len * 4

    def alloc_rows(self, n):
        if n > len(self.free_slots):
            return None
        if n == 0:
            return []
        sel = self.free_slots[-n:]           # 从尾部弹，O(n)
        del self.free_slots[-n:]
        for i in sel:
            self.generation[i] += 1
            self.rows[i] = []
        return sel

    def free_rows(self, rows):
        for i in rows:
            self.rows.pop(i, None)
        self.free_slots.extend(rows)          # 追加到尾部，下次最先被弹出

    def write(self, row, start, values):
        r = self.rows[row]
        if len(r) < start:
            r.extend([0] * (start - len(r)))
        r[start:start + len(values)] = values


# ---------------------------------------------------------------- 第二级：槽位 -> KV
class TokenAllocator:
    """allocator/token.py:28，page_size = 1。"""

    def __init__(self, size, need_sort=False):
        self.size, self.page_size, self.need_sort = size, 1, need_sort
        self.free = deque(range(1, size + 1))
        self.release = []

    def available_size(self):
        return len(self.free) + len(self.release)

    def merge_and_sort_free(self):
        if self.release:
            self.free = deque(sorted(list(self.free) + self.release))
            self.release = []

    def alloc(self, n):
        if self.need_sort and n > len(self.free):
            self.merge_and_sort_free()
        if n > len(self.free):
            return None
        return [self.free.popleft() for _ in range(n)]

    def free_slots(self, idx):
        if self.need_sort:
            self.release.extend(idx)
        else:
            self.free.extend(idx)              # 追加到尾部

    # page 1 时 extend / decode 都是 alloc(n)
    def alloc_extend(self, prefix_lens, seq_lens, last_locs):
        return self.alloc(sum(s - p for p, s in zip(prefix_lens, seq_lens)))

    def alloc_decode(self, seq_next, last_locs):
        return self.alloc(len(seq_next))


def num_new_pages_extend(prefix_lens, seq_lens, ps):
    """utils/common.py:4840 get_num_new_pages。"""
    return sum(cdiv(s, ps) - cdiv(p, ps) for p, s in zip(prefix_lens, seq_lens))


def num_new_pages_decode(seq_next, ps):
    return sum(1 for s in seq_next if s % ps == 1 or ps == 1)


class PagedAllocator:
    """allocator/paged.py:136。槽位号 = 页号 x ps + 页内偏移，页 0 留作 padding。"""

    def __init__(self, size, page_size, need_sort=False):
        self.page_size = page_size
        self.num_pages = size // page_size
        self.size = self.num_pages * page_size
        self.need_sort = need_sort
        self.free = deque(range(1, self.num_pages + 1))
        self.staged = []
        self.last_exact = False

    def available_size(self):
        return (len(self.free) + len(self.staged)) * self.page_size

    def merge_and_sort_free(self):
        if self.staged:
            self.free = deque(sorted(list(self.free) + self.staged))
            self.staged = []

    def alloc_extend(self, prefix_lens, seq_lens, last_locs):
        ps, bs = self.page_size, len(seq_lens)
        extend = sum(s - p for p, s in zip(prefix_lens, seq_lens))
        bound = extend // ps + bs + 1         # 上界：每条请求最多比 extend/ps 多一页
        if bound > len(self.free):
            self.merge_and_sort_free()
        self.last_exact = False
        if bound > len(self.free):
            self.last_exact = True            # 上界证明不了够用，才付出精确计数的代价
            if num_new_pages_extend(prefix_lens, seq_lens, ps) > len(self.free):
                return None
        out, take = [], []
        it = iter(self.free)
        for p, s, last in zip(prefix_lens, seq_lens, last_locs):
            ceil_p, floor_s = ceil_align(p, ps), s // ps * ps
            n1 = min(s, ceil_p) - p           # 第一段：填满前缀最后一页的空位，紧跟 last_loc
            out += [last + 1 + i for i in range(n1)]
            if p + n1 == s:
                continue
            for _ in range((floor_s - ceil_p) // ps):   # 第二段：整页
                pg = next(it)
                take.append(pg)
                out += [pg * ps + j for j in range(ps)]
            n3 = s - floor_s                  # 第三段：不满的尾页
            if n3:
                pg = next(it)
                take.append(pg)
                out += [pg * ps + j for j in range(n3)]
        for _ in take:
            self.free.popleft()
        return out

    def alloc_decode(self, seq_next, last_locs):
        ps = self.page_size
        if len(seq_next) > len(self.free):
            self.merge_and_sort_free()
        if num_new_pages_decode(seq_next, ps) > len(self.free):
            return None
        out = []
        for s, last in zip(seq_next, last_locs):
            if (s - 1) % ps == 0:             # 新 token 落在新页的第 0 格
                out.append(self.free.popleft() * ps)
            else:
                out.append(last + 1)
        return out

    def free_slots(self, idx):
        if not idx:
            return
        pages = sorted(set(i // self.page_size for i in idx))   # torch.unique 去重并排序
        if self.need_sort:
            self.staged.extend(pages)
        else:
            self.free.extendleft(reversed(pages))   # cat((pages, free_pages))：插回头部

    def alloc_whole(self, n_tokens):
        """alloc(need_size)：整页分配，返回连续页的全部槽位。"""
        n = n_tokens // self.page_size
        if n > len(self.free):
            self.merge_and_sort_free()
        if n > len(self.free):
            return None
        pages = [self.free.popleft() for _ in range(n)]
        return [p * self.page_size + j for p in pages for j in range(self.page_size)]


def make_allocator(size, ps, need_sort=False):
    return TokenAllocator(size, need_sort) if ps == 1 else PagedAllocator(size, ps, need_sort)


# ---------------------------------------------------------------- 极简前缀缓存：只做整页匹配
class PageStore:
    """RadixKey.page_aligned（radix_cache.py:147）：只有整页进树。驱逐和分裂见 SGL-08。"""

    def __init__(self, ps):
        self.ps = ps
        self.keys = {}

    def insert(self, tokens, slots):
        n = len(tokens) // self.ps * self.ps
        for k in range(self.ps, n + 1, self.ps):
            self.keys.setdefault(tuple(tokens[:k]), slots[:k])
        return n

    def match(self, tokens):
        n = len(tokens) // self.ps * self.ps
        for k in range(n, 0, -self.ps):
            hit = self.keys.get(tuple(tokens[:k]))
            if hit is not None:
                return hit
        return []


# ---------------------------------------------------------------- 把分配、写表、释放串起来
class Req:
    def __init__(self, rid, tokens):
        self.rid, self.tokens = rid, list(tokens)
        self.row = None
        self.prefix_len = 0
        self.seq_len = 0
        self.protected = 0        # cache_protected_len：前缀树拥有的部分
        self.allocated = 0        # kv_allocated_len


class Pool:
    def __init__(self, kv_size, ps, max_reqs, max_context, need_sort=False, store=None):
        self.ps = ps
        self.r2t = ReqToTokenPool(max_reqs, max_context)
        self.alloc = make_allocator(kv_size, ps, need_sort)
        self.store = store

    def extend(self, reqs, chunk_ends=None):
        """allocation.py:374 alloc_for_extend：先拿行，再拿槽位，最后写 req_to_token。"""
        for r in reqs:
            if r.row is None and self.store is not None:
                hit = self.store.match(r.tokens)
                r.prefix_len = r.protected = len(hit)
                r._hit = hit
        new = [r for r in reqs if r.row is None]
        rows = self.r2t.alloc_rows(len(new))
        if rows is None:
            raise RuntimeError('alloc_req_slots runs out of memory. Please set a smaller number for '
                               f'`--max-running-requests`. req_to_token_pool.available_size()='
                               f'{len(self.r2t.free_slots)}, num_reqs={len(new)}, ')
        for r, row in zip(new, rows):
            r.row = row
            if getattr(r, '_hit', None):
                self.r2t.write(row, 0, r._hit)
        ends = chunk_ends or [len(r.tokens) for r in reqs]
        pre = [r.seq_len if r.seq_len else r.prefix_len for r in reqs]
        last = [self.r2t.rows[r.row][p - 1] if p > 0 else -1 for r, p in zip(reqs, pre)]
        out = self.alloc.alloc_extend(pre, ends, last)
        if out is None:
            n = sum(e - p for p, e in zip(pre, ends))
            raise RuntimeError(f'Prefill out of memory. Try to lower your batch size.\n'
                               f'Try to allocate {n} tokens.\n'
                               f'Available tokens: {self.alloc.available_size()}')
        k = 0
        for r, p, e in zip(reqs, pre, ends):
            self.r2t.write(r.row, p, out[k:k + e - p])
            k += e - p
            r.seq_len = r.allocated = e
        return out

    def decode(self, reqs):
        """allocation.py:625 alloc_for_decode_default。"""
        seq_next = [r.seq_len + 1 for r in reqs]
        last = [self.r2t.rows[r.row][r.seq_len - 1] for r in reqs]
        out = self.alloc.alloc_decode(seq_next, last)
        if out is None:
            raise RuntimeError(f'Decode out of memory. Try to lower your batch size.\n'
                               f'Try to allocate {len(reqs)} tokens.\n'
                               f'Available tokens: {self.alloc.available_size()}')
        for r, loc in zip(reqs, out):
            r.tokens.append(10_000 + r.seq_len)
            self.r2t.write(r.row, r.seq_len, [loc])
            r.seq_len += 1
            r.allocated = r.seq_len
        return out

    def release(self, r, checkpoint=True):
        """common.py:418 release_kv_cache。返回 (进树的 token 数, 还给分配器的槽位)。"""
        row = self.r2t.rows[r.row]
        owned = r.seq_len
        kept = r.protected
        if checkpoint and self.store is not None:
            kept = self.store.insert(r.tokens[:owned], row[:owned])
            r.protected = max(r.protected, kept)
        freed = row[r.protected:owned]
        self.alloc.free_slots(freed)
        start = ceil_align(owned, self.ps) if self.ps > 1 else owned
        over = row[start:r.allocated]
        self.alloc.free_slots(over)
        self.r2t.free_rows([r.row])
        r.row = None
        return kept, freed


def page_table(row, seq_len, ps):
    """flashattention_backend.py:1317：req_to_token[:, ::ps] // ps。"""
    return [row[i] // ps for i in range(0, seq_len, ps)]


def fmt(xs, n=24):
    xs = list(xs)
    s = ' '.join(str(x) for x in xs[:n])
    return s + (f' …（共 {len(xs)}）' if len(xs) > n else '')


def runs(idx):
    if not idx:
        return 0
    return 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)


# ---------------------------------------------------------------- 场景
def scenario_walk(ps):
    print(f'# 池子：KV 槽位 64 个，page_size={ps}，req_to_token 4 行；A 的 prompt 10 token，B 的 prompt 7 token')
    store = PageStore(ps)
    pool = Pool(64, ps, 4, 32, store=store)
    sysp = list(range(100, 108))
    A = Req('A', sysp + [1, 2])
    B = Req('B', [7, 7, 7, 7, 7, 7, 7])
    fl = pool.alloc.free
    print(f'初始空闲表头：{fmt(fl, 8)}    req_to_token 空闲行：{pool.r2t.free_slots}')
    out = pool.extend([A, B])
    print('\n[1] prefill A、B（一批）')
    print(f'  out_cache_loc        {fmt(out)}')
    for r in (A, B):
        row = pool.r2t.rows[r.row]
        print(f'  {r.rid} 占 req_to_token 第 {r.row} 行（generation {pool.r2t.generation[r.row]}）：{fmt(row[:r.seq_len])}')
        if ps > 1:
            print(f'      页表 row[::ps]//ps = {page_table(row, r.seq_len, ps)}')
    print(f'  空闲表头：{fmt(pool.alloc.free, 8)}')

    print('\n[2] decode 3 步')
    for step in range(3):
        before = (len(pool.alloc.free) if ps > 1 else None)
        out = pool.decode([A, B])
        note = ''
        if ps > 1:
            note = f'   seq_len 变成 {A.seq_len}/{B.seq_len}，取新页 {before - len(pool.alloc.free)} 个'
        print(f'  step {step + 1}: out_cache_loc = {out}{note}')
    for r in (A, B):
        row = pool.r2t.rows[r.row]
        print(f'  {r.rid} 第 {r.row} 行：{fmt(row[:r.seq_len])}')

    print('\n[3] A 结束，release_kv_cache')
    kept, freed = pool.release(A)
    print(f'  A 共 {A.seq_len} token，进前缀树 {kept} 个' + ('（向下取整到页）' if ps > 1 else '') + f'，还给分配器的槽位：{freed}')
    if ps > 1:
        print(f'  按页归还：{sorted(set(i // ps for i in freed))}，插回空闲表头部')
    print(f'  空闲表头：{fmt(pool.alloc.free, 8)}    req_to_token 空闲行：{pool.r2t.free_slots}')

    C = Req('C', sysp + [5, 6, 9])
    out = pool.extend([C])
    print('\n[4] C 进来：和 A 共享 8 token 的系统提示词')
    print(f'  前缀命中 {C.protected} token，新算 {C.seq_len - C.protected} token；C 拿到第 {C.row} 行'
          f'（generation {pool.r2t.generation[C.row]}，刚被 A 释放的那一行）')
    print(f'  C 第 {C.row} 行：{fmt(pool.r2t.rows[C.row][:C.seq_len])}')
    print(f'  其中新分配的 out_cache_loc：{out}')


def lognormal(rng, median, sigma, lo, hi):
    return int(min(max(rng.lognormvariate(math.log(median), sigma), lo), hi))


WORKLOADS = {
    # 名称: (输入中位数, sigma, 输出中位数, sigma, 共享前缀长度)
    'chat': (300, 1.0, 200, 0.9, 1000),
    'short': (20, 0.6, 60, 0.7, 40),
    'rag': (4000, 0.5, 300, 0.6, 2500),
}


def scenario_pagesize(pool_tokens, n=20000, seed=7):
    print(f'# 合成负载各 {n} 条，长度取对数正态；池子按 Qwen3-8B @ H100 的 {pool_tokens:,} token 算')
    print('# 尾页浪费：在请求生命周期里随机取一个时刻，已分配槽位里没装 token 的比例')
    print('# 树丢尾：请求结束时 len % ps 个 token 不进前缀树；共享前缀命中：系统提示词能复用多少')
    print('# 页表：每条请求的页表项数（attention 每步要读的索引个数）；新页频率：decode 多少步取一次新页')
    for name, (im, isg, om, osg, shared) in WORKLOADS.items():
        rng = random.Random(seed)
        reqs = [(lognormal(rng, im, isg, 4, 32000), lognormal(rng, om, osg, 1, 4096)) for _ in range(n)]
        mi = sorted(i for i, _ in reqs)[n // 2]
        mo = sorted(o for _, o in reqs)[n // 2]
        print(f'\n[{name}] 输入中位数 {mi}，输出中位数 {mo}，共享系统提示词 {shared} token')
        print(f'  {"ps":>4} {"尾页浪费":>8} {"树丢尾 tok":>10} {"占比":>6} {"前缀命中":>8} '
              f'{"页表项":>7} {"新页频率":>8} {"可驻留条数":>10}')
        for ps in (1, 16, 32, 64, 128):
            r2 = random.Random(seed + 1)
            live = alloc = tail = total = pt = 0
            for i, o in reqs:
                L = i + r2.randint(0, o)
                live += L
                alloc += ceil_align(L, ps)
                pt += cdiv(L, ps)
                total += i + o
                tail += (i + o) % ps
            waste = 1 - live / alloc
            hit = shared // ps * ps
            resident = pool_tokens // ps * ps / (alloc / n)
            freq = '每步' if ps == 1 else f'1/{ps}'
            print(f'  {ps:>4} {waste:>8.1%} {tail / n:>10.1f} {tail / total:>6.2%} '
                  f'{hit:>4}/{shared:<4} {pt / n:>7.1f} {freq:>8} {resident:>10.0f}')


def scenario_sizing():
    cases = [('qwen3-8b', 'h100-sxm', 1), ('qwen3-8b', 'l40s', 1), ('qwen3-8b', 'rtx4090', 1),
             ('llama3.1-8b', 'h100-sxm', 1), ('llama3.1-70b', 'h100-sxm', 4), ('qwen3-235b-a22b', 'h100-sxm', 8)]
    print('# 账本来自 ../02-ledger/sim_ledger.py；token 数按 page_size 向下取整；KV buffer 每层多一页给 padding')
    print(f'{"模型":<18}{"GPU":<10}{"tp":>3}{"KB/tok":>8}{"max_total_tokens":>18}{"max_running":>12}'
          f'{"r2t 表 MiB":>11}{"满上下文条数":>12}{"FA 页表宽 ps=1/64":>20}')
    for key, gpu, tp in cases:
        L = ledger(key, gpu, tp)
        m = L['model']
        ctx = m.context + EXTRA_CONTEXT
        print(f'{m.name:<18}{gpu:<10}{tp:>3}{L["cell"] / 1024:>8.0f}{L["tokens"]:>18,}{L["running"]:>12,}'
              f'{L["r2t"] * 1024:>11.0f}{L["tokens"] / m.context:>12.1f}{ctx:>12,}/{cdiv(ctx, 64):<6,}')
    L = ledger('qwen3-8b', 'h100-sxm')
    m = L['model']
    heads = L['heads']
    print('\n# Qwen3-8B @ H100 的启动日志（page_size=1 与 64）')
    for ps in (1, 64):
        tok = L['tokens'] // ps * ps
        rows = tok + ps
        k = m.layers * rows * heads * m.head_dim * KV_DTYPE['bf16'] / GB
        print(f'  [ps={ps}] KV Cache is allocated. dtype: torch.bfloat16, #tokens: {tok}, '
              f'K size: {k:.2f} GB, V size: {k:.2f} GB')
        print(f'          每层 k_buffer 形状 ({rows}, {heads}, {m.head_dim})，共 {m.layers} 层；页数 {tok // ps:,}')
    print(f'  max_total_num_tokens={L["tokens"]}, chunked_prefill_size={L["chunk"]}, max_prefill_tokens=16384, '
          f'max_running_requests={L["running"]}, context_len={m.context}')
    est = max(min(int(L['tokens'] / m.context * 512), 4096), 2048)
    print(f'  resolve_max_num_reqs：clamp({L["tokens"]:,} / {m.context} x 512 = '
          f'{int(L["tokens"] / m.context * 512):,}, 2048, 4096) = {est}，再和 tokens // 2 取小')
    print(f'  req_to_token：({L["running"]} + 1) x ({m.context} + {EXTRA_CONTEXT}) x 4 B = {L["r2t"] * 1024:.0f} MiB'
          '，在 KV 池之外，从预留里出')
    for mr in (1024, 16384):
        L2 = ledger('qwen3-8b', 'h100-sxm', max_running=mr)
        print(f'  --max-running-requests {mr}：生效 {L2["running"]}，req_to_token {L2["r2t"] * 1024:.0f} MiB')


def scenario_churn(seed=3, steps=6000):
    print('# 一万槽位的池子，随机长度的请求不停进出（池子保持 85% 左右的占用），统计新请求 KV 槽位分成几段连续区间')
    print('# 段数 = PD 分离时这条请求的 KV 要拆成几次传输（group_concurrent_contiguous）')
    print(f'{"ps":>4} {"need_sort":>9} {"每条段数 p50":>12} {"p90":>6} {"每 100 token 段数":>16} {"合并排序次数":>12}')
    for ps in (1, 16):
        for ns in (False, True):
            rng = random.Random(seed)
            alloc = make_allocator(10_000, ps, ns)
            live = []
            seg, per100 = [], []
            merges = 0
            for _ in range(steps):
                if live and (alloc.available_size() < 2000 or rng.random() < 0.45):
                    idx = live.pop(rng.randrange(len(live)))
                    alloc.free_slots(idx)
                    continue
                n = rng.randint(20, 600)
                staged = len(getattr(alloc, 'staged', getattr(alloc, 'release', [])))
                out = alloc.alloc_extend([0], [n], [-1])
                if out is None:
                    continue
                if staged and not len(getattr(alloc, 'staged', getattr(alloc, 'release', []))):
                    merges += 1
                live.append(out)
                k = runs(out)
                seg.append(k)
                per100.append(k / n * 100)
            seg.sort()
            per100.sort()
            print(f'{ps:>4} {str(ns):>9} {seg[len(seg) // 2]:>12} {seg[len(seg) * 9 // 10]:>6} '
                  f'{per100[len(per100) // 2]:>16.1f} {merges:>12}')


def scenario_errors():
    print('[1] req_to_token 行用完：max_running_requests 设得比真实并发小，又绕过了准入')
    pool = Pool(1024, 16, 2, 256)
    try:
        pool.extend([Req(i, range(30)) for i in range(3)])
    except RuntimeError as e:
        print('  RuntimeError:', e)
    print('\n[2] prefill 槽位不够：64 token / ps 16 = 4 页，三条 20 token 的请求要 6 页')
    pool = Pool(64, 16, 8, 256)
    try:
        pool.extend([Req(i, range(20)) for i in range(3)])
    except RuntimeError as e:
        print('  RuntimeError:', e)
    print(f'  上界 extend//ps + bs + 1 = {60 // 16 + 3 + 1}，超过空闲页 4，于是算精确页数 '
          f'{num_new_pages_extend([0] * 3, [20] * 3, 16)}，仍不够，返回 None')
    print('\n[3] decode 跨页时没页可拿：两条请求都停在 32 token（页尾），池子已满')
    pool = Pool(64, 16, 8, 256)
    a, b = Req('a', range(32)), Req('b', range(32))
    pool.extend([a, b])
    try:
        pool.decode([a, b])
    except RuntimeError as e:
        print('  RuntimeError:', e)
    print(f'  check_decode_mem 本该提前发现：两条的 kv_committed_len % 16 == 0，各要一页，共 32 token，空闲 0 → 先 retract')
    print('\n[4] 启动时 KV 池为负：mem_fraction_static 太小')
    L = ledger('qwen3-8b', 'rtx4090', mem_frac=0.6)
    print(f'  Qwen3-8B @ RTX 4090，mem_fraction_static=0.6：权重 {L["w"]:.1f} GiB，留给 KV {L["rest"]:.1f} GiB')
    print('  RuntimeError: Not enough memory. Please try to increase --mem-fraction-static. '
          'Current value: mem_fraction_static=0.6')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='walk', choices=['walk', 'pagesize', 'sizing', 'churn', 'errors'])
    ap.add_argument('--page-size', type=int, default=None)
    a = ap.parse_args()
    if a.scenario == 'walk':
        for ps in ([a.page_size] if a.page_size else [1, 4]):
            scenario_walk(ps)
            print()
    elif a.scenario == 'pagesize':
        scenario_pagesize(ledger('qwen3-8b', 'h100-sxm')['tokens'])
    elif a.scenario == 'sizing':
        scenario_sizing()
    elif a.scenario == 'churn':
        scenario_churn()
    else:
        scenario_errors()


if __name__ == '__main__':
    main()
