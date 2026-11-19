"""多 LoRA：显存池吃掉多少 KV、调度时哪些请求被跳过、csgmv 怎么分段、饥饿怎么排空、异步加载省多少。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  lora/utils.py:172                       get_default_hidden_dim：qkv / o / gate_up / down 的输入输出维度
  lora/utils.py:330                       get_stacked_multiply：qkv ×3、gate_up ×2
  lora/utils.py:374                       ROW_PARALLELISM：o_proj / down_proj 的 A 按 TP 切输入维
  lora/mem_pool.py:424-492                get_lora_A_shape / get_lora_B_shape：(mlpb, r×c, in) 和 (mlpb, out/tp, r)
  lora/mem_pool.py:598                    init_buffers：每层每模块一块 torch.zeros，启动时一次分配
  model_executor/model_runner.py:762      maybe_init_lora_manager 在 load_model 之后、KV 池定尺寸之前
  lora/mem_pool.py:780                    prepare_lora_batch：先找空槽，再在「不在本批、没 pin」的里按策略挑受害者，
                                          有真 adapter 可踢就不踢 base（None）
  lora/eviction_policy.py                 LRU：mark_used 移到队尾；FIFO：只记第一次进入的顺序
  lora/lora_manager.py:427                validate_lora_batch：总数 ≤ mlpb，未 pin 的个数 ≤ mlpb - 已 pin 数
  managers/scheduler.py:4274              can_schedule_lora_req 不过就 continue（跳过，不 break）
  managers/scheduler.py:4481              can_schedule_lora_req：drainer → 已在跑 → 异步加载 / validate
  lora/backend/chunked_backend.py:197     _determine_chunk_size_for_tokens：≥256 → 128，≥64 → 32，否则 16
  lora/backend/chunked_backend.py:469-560 _get_permutation / _get_segments_info：按 adapter 稳定排序后切段
  kernels/lora/triton_ops/sgemm_lora_a.py triton 后端：每请求 cdiv(max_len, 16) 个块，短请求的块提前退出
  lora/lora_drainer.py                    LoRADrainer：等待超过阈值的 adapter 让剩余 token 最少的在跑 adapter
                                          停收新请求，max_new ≤ 1.2 × 剩余 才放行
  lora/lora_overlap_loader.py:33          try_overlap_load_lora：side stream 上异步拷，一次一个
  arg_groups/lora_hook.py                 --max-loaded-loras ≤ 2 × mlpb 才能开异步加载

  python3 lora_sim.py                         # memory：各 rank / 模块组合的 LoRA 池字节数，折成少了多少 KV token
  python3 lora_sim.py --scenario admit        # admit：一轮调度里谁进谁被跳过、pin 的影响、LRU vs FIFO 换入次数
  python3 lora_sim.py --scenario segments     # segments：csgmv 的排列与分段 vs triton 按请求切块
  python3 lora_sim.py --scenario drain        # drain：三个热 adapter 占满槽位时冷 adapter 等多久，开 drainer 前后
  python3 lora_sim.py --scenario overlap      # overlap：同步加载卡住整批 vs side stream 异步加载
"""
import argparse
import math
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))

from common.specs import MODELS  # noqa: E402
from sim_ledger import ledger  # noqa: E402
from sim_serve import Server  # noqa: E402

GiB = 1024 ** 3
INTER = {                         # config.json 的 intermediate_size，specs 里没有
    'qwen3-0.6b': 3072,
    'qwen3-8b': 12288,
    'llama3.1-8b': 14336,
    'llama3.1-70b': 28672,
}
STACK = {'qkv_proj': 3, 'gate_up_proj': 2}               # utils.py:330
ROW_PARALLEL = {'o_proj', 'down_proj'}                   # utils.py:374
ALL = ('qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj')
H2D_GBPS = 25.0                   # pinned 内存 H2D 的保守值
DRAIN_TOL = 1.2                   # lora_drainer.py DRAIN_SCHEDULE_TOLERANCE
MIN_CHUNK = 16


# ---------------------------------------------------------------- 显存
def hidden_dims(model, module):
    """utils.py:172 get_default_hidden_dim。"""
    m = MODELS[model]
    h, hd, nh, nkv, inter = m.hidden, m.head_dim, m.heads, m.kv_heads, INTER[model]
    return {
        'qkv_proj': (h, hd * (nh + 2 * nkv)),
        'o_proj': (hd * nh, h),
        'gate_up_proj': (h, 2 * inter),
        'down_proj': (inter, h),
    }[module]


def b_out_per_rank(model, module, tp):
    """mem_pool.py:492：列并行模块的 B 按 TP 切输出维；qkv 在 kv 头数 < tp 时每个 rank 复制 KV 头。"""
    m = MODELS[model]
    _, out = hidden_dims(model, module)
    if module in ROW_PARALLEL:
        return out
    if module == 'qkv_proj' and m.kv_heads < tp:
        return m.head_dim * m.heads // tp + 2 * m.head_dim
    return out // tp


def lora_shapes(model, module, rank, mlpb, tp=1):
    inp, _ = hidden_dims(model, module)
    c = STACK.get(module, 1)
    a_in = inp // tp if module in ROW_PARALLEL else inp
    return (mlpb, rank * c, a_in), (mlpb, b_out_per_rank(model, module, tp), rank)


def pool_bytes(model, rank, mlpb, targets=ALL, tp=1, dtype_bytes=2):
    """init_buffers：每层每模块 A、B 各一块，按 mlpb 个槽一次分配。"""
    per_layer = 0
    for mod in targets:
        a, b = lora_shapes(model, mod, rank, mlpb, tp)
        per_layer += (math.prod(a) + math.prod(b)) * dtype_bytes
    return per_layer * MODELS[model].layers


def adapter_bytes(model, rank, targets=ALL, tp=1):
    """一个 adapter 换入时拷的字节数（每个 TP rank）。"""
    return pool_bytes(model, rank, 1, targets, tp)


def scenario_memory():
    print('LoRA 池在 KV 池定尺寸之前分配（model_runner.py:762），多出来的字节直接从 KV 里扣。')
    print('bf16，每个 TP rank 的数字：')
    print(f'{"模型":<14}{"TP":>3}{"rank":>6}{"mlpb":>6}{"模块":>10}{"池 GiB":>9}{"单个 MiB":>10}'
          f'{"少 KV token":>13}{"占 KV 池":>9}')
    res = {}
    cases = [
        ('llama3.1-8b', 1, 16, 8, ALL), ('llama3.1-8b', 1, 64, 8, ALL),
        ('llama3.1-8b', 1, 128, 8, ALL), ('llama3.1-8b', 1, 64, 32, ALL),
        ('llama3.1-8b', 1, 64, 8, ('qkv_proj', 'o_proj')),
        ('qwen3-8b', 1, 64, 8, ALL),
        ('llama3.1-70b', 4, 64, 8, ALL), ('llama3.1-70b', 4, 256, 8, ALL),
    ]
    for model, tp, r, mlpb, tg in cases:
        L = ledger(model, 'h100-sxm', tp)
        pb = pool_bytes(model, r, mlpb, tg, tp)
        lost = int(pb / L['cell'])
        res[(model, tp, r, mlpb, tg)] = (pb, lost, L['tokens'])
        tname = 'all' if tg == ALL else 'attn'
        print(f'{model:<14}{tp:>3}{r:>6}{mlpb:>6}{tname:>10}{pb / GiB:>9.2f}'
              f'{adapter_bytes(model, r, tg, tp) / 2 ** 20:>10.1f}{lost:>13,}{lost / L["tokens"]:>9.1%}')
    print('   池按 --max-lora-rank 和 --lora-target-modules 的并集开，rank 8 的 adapter 也占 rank 128 的槽。')
    print('   rank 和 mlpb 是乘在一起的：rank 64 × 32 槽比 rank 64 × 8 槽多吃 4 倍。')
    print('   attn 只挂 q/k/v/o 时池子小很多，MLP 的 gate_up / down 才是大头。')
    return res


# ---------------------------------------------------------------- 准入与换入
class LRU:
    def __init__(self):
        self.order = OrderedDict()

    def mark_used(self, uid):
        if uid is not None:
            self.order.pop(uid, None)
            self.order[uid] = True

    def select_victim(self, cands):
        for uid in self.order:
            if uid in cands:
                return uid
        assert None in cands
        return None

    def remove(self, uid):
        self.order.pop(uid, None)


class FIFO(LRU):
    def mark_used(self, uid):
        if uid is not None and uid not in self.order:
            self.order[uid] = True


class Pool:
    """mem_pool.py:780 prepare_lora_batch 的槽位逻辑。"""

    def __init__(self, mlpb, policy='lru', pinned=()):
        self.mlpb = mlpb
        self.slots = ['EMPTY'] * mlpb
        self.policy = LRU() if policy == 'lru' else FIFO()
        self.pinned = set(pinned)
        self.loads = 0

    def prepare(self, cur):
        ordered = sorted(cur, key=lambda u: (u is not None, u or ''))
        for u in ordered:
            self.policy.mark_used(u)
        for u in ordered:
            if u in self.slots:
                continue
            if 'EMPTY' in self.slots:
                i = self.slots.index('EMPTY')
            else:
                cands = {s for s in self.slots if s not in cur and s not in self.pinned}
                if not cands:
                    raise ValueError('No available buffer slots found. Please ensure the number of '
                                     'active (pinned) loras is less than max_loras_per_batch.')
                use = (cands - {None}) or cands
                v = self.policy.select_victim(use)
                self.policy.remove(v)
                i = self.slots.index(v)
            self.slots[i] = u
            if u is not None:
                self.loads += 1


def validate_lora_batch(ids, mlpb, pinned):
    """lora_manager.py:427。"""
    if len(ids) > mlpb:
        return False
    if not pinned:
        return True
    in_batch = sum(1 for i in ids if i in pinned)
    return len(ids) - in_batch <= mlpb - len(pinned)


def schedule_pass(waiting, running, mlpb, pinned=()):
    """scheduler.py:4251-4342：不过 can_schedule_lora_req 就 continue，过了加进 running_loras。"""
    running = set(running)
    out = []
    for rid, lora in waiting:
        if lora in running:
            out.append((rid, lora, '进（已在跑）'))
        elif validate_lora_batch(running | {lora}, mlpb, set(pinned)):
            running.add(lora)
            out.append((rid, lora, '进（新占一个槽）'))
        else:
            out.append((rid, lora, '跳过'))
    return out, running


def zipf_batches(n_adapters, s, mlpb, n_batches, bs, rnd):
    w = [1 / (i + 1) ** s for i in range(n_adapters)]
    names = [f'a{i:02d}' for i in range(n_adapters)]
    out = []
    for _ in range(n_batches):
        cur = {None}
        for _ in range(bs):
            x = rnd.choices(names, w)[0]
            if x in cur or len(cur) < mlpb:
                cur.add(x)
        out.append(cur)
    return out


def scenario_admit(seed=0):
    mlpb = 4
    waiting = [('r1', 'B'), ('r2', 'C'), ('r3', 'D'), ('r4', 'B'), ('r5', 'A'), ('r6', None), ('r7', 'E')]
    print(f'1）--max-loras-per-batch {mlpb}（含 base），在跑 {{None, A}}，等待队列按到达顺序：')
    rows, run = schedule_pass(waiting, {None, 'A'}, mlpb)
    for rid, lora, why in rows:
        print(f'   {rid}  {str(lora):<5}{why}')
    print(f'   本轮之后在跑 {sorted(run, key=str)}。r3 被跳过，排在它后面的 r4 / r5 / r6 照样进，')
    print('   所以 LoRA 请求不会互相队头阻塞，但被跳过的那个可能一直被跳过，见 drain 场景。')
    print()
    print('2）A 被 pin（--lora-paths 里 pinned=true），num_pinned=1：')
    for ids in ({None, 'A', 'B'}, {None, 'A', 'B', 'C'}, {'B', 'C', 'D'}, {None, 'B', 'C', 'D'}):
        ok = validate_lora_batch(ids, mlpb, {'A'})
        print(f'   {str(sorted(ids, key=str)):<28}{"能放" if ok else "放不下"}')
    print('   未 pin 的（base 也算）最多 mlpb - 1 = 3 个。A 不在本批也占着槽，pin 一个等于永久少一个槽。')
    print('   pin 到 mlpb - 1 个时加载直接报错，防止 base 和未 pin 的 adapter 饿死。')
    print()
    rnd = random.Random(seed)
    print('3）32 个 adapter 按 Zipf 分布到达，每批 6 个请求，5000 批，统计换入次数：')
    print(f'{"Zipf s":>8}{"mlpb":>6}{"LRU 换入":>10}{"FIFO 换入":>11}{"每批平均 adapter":>18}')
    res = {}
    for s in (0.8, 1.2):
        for m in (8, 16):
            bs = zipf_batches(32, s, m, 5000, 6, rnd)
            lru, fifo = Pool(m, 'lru'), Pool(m, 'fifo')
            for b in bs:
                lru.prepare(b)
                fifo.prepare(b)
            avg = sum(len(b) - 1 for b in bs) / len(bs)
            res[(s, m)] = (lru.loads, fifo.loads)
            print(f'{s:>8}{m:>6}{lru.loads:>10}{fifo.loads:>11}{avg:>18.1f}')
    one = adapter_bytes('llama3.1-8b', 64)
    print(f'   llama3.1-8b rank 64 每次换入 {one / 2 ** 20:.0f} MiB，按 {H2D_GBPS:.0f} GB/s 是 '
          f'{one / H2D_GBPS / 1e6:.1f} ms，同步加载时这段加在当步前向前面。')
    print('   槽比每批用到的 adapter 多时，空出来的槽就是缓存，换入次数取决于淘汰谁。')
    print('   LRU 留住热的，FIFO 按进场顺序踢，热门 adapter 也会被踢出去再拷回来。')
    return rows, res


# ---------------------------------------------------------------- 分段
def chunk_size_for(num_tokens, max_chunk=16):
    """chunked_backend.py:197。"""
    if max_chunk <= MIN_CHUNK:
        return MIN_CHUNK
    c = 128 if num_tokens >= 256 else 32 if num_tokens >= 64 else 16
    return min(max_chunk, c)


def permutation(seq_w, seq_lens):
    """chunked_backend.py:469：逐 token 展开 adapter 编号，稳定排序。"""
    rows = [w for w, n in zip(seq_w, seq_lens) for _ in range(n)]
    perm = sorted(range(len(rows)), key=lambda i: rows[i])
    return perm, [rows[i] for i in perm]


def segments(w_sorted, chunk):
    """chunked_backend.py:510：同一 adapter 的连续 token 按 chunk 切段。"""
    wi, lens = [], []
    i = 0
    while i < len(w_sorted):
        j = i
        while j < len(w_sorted) and w_sorted[j] == w_sorted[i]:
            j += 1
        g = j - i
        n = (g + chunk - 1) // chunk
        wi += [w_sorted[i]] * n
        lens += [chunk] * (n - 1) + [g - (n - 1) * chunk]
        i = j
    indptr = [0]
    for x in lens:
        indptr.append(indptr[-1] + x)
    return wi, indptr


def triton_blocks(seq_lens, seq_w, mlpb=8, decode=False, block=16):
    """extend：sgemm_lora_a 的 grid = cdiv(max_len, 16) × bs，超出本请求长度的块直接返回。
    decode：triton_backend.py:469 _build_sgemm_routing 先按 adapter 排序，合成 mlpb 段，max_len = bs。"""
    if decode:
        cnt = [seq_w.count(i) for i in range(mlpb)]
        launched = math.ceil(len(seq_lens) / block) * mlpb
        active = sum(math.ceil(c / block) for c in cnt)
        return launched, active
    launched = math.ceil(max(seq_lens) / block) * len(seq_lens)
    active = sum(math.ceil(n / block) for n in seq_lens)
    return launched, active


def scenario_segments(seed=0):
    perm, w = permutation([0, 1, 0], [2, 1, 3])
    wi, ip = segments(w, 3)
    print('1）源码 docstring 的例子：adapter [0, 1, 0]，长度 [2, 1, 3]')
    print(f'   permutation {perm}   排序后 {w}')
    print(f'   chunk=3 切段 -> 每段 adapter {wi}  seg_indptr {ip}')
    print()
    rnd = random.Random(seed)
    print('2）几种 batch 下两种后端各发多少个 GEMM 块（shrink，按 token 维）：')
    print(f'{"batch":<34}{"token":>7}{"chunk":>6}{"csgmv 段":>9}{"triton 发射":>11}{"triton 干活":>11}')
    res = {}
    cases = [
        ('decode 64 请求 / 4 adapter', [1] * 64, [i % 4 for i in range(64)]),
        ('decode 64 请求 / 1 adapter', [1] * 64, [0] * 64),
        ('prefill 8 个各 512 / 3 adapter', [512] * 8, [i % 3 for i in range(8)]),
        ('prefill 1×8000 + 15 个短', [8000] + [rnd.randint(5, 60) for _ in range(15)],
         [rnd.randrange(4) for _ in range(16)]),
    ]
    for name, lens, ws in cases:
        n = sum(lens)
        c = chunk_size_for(n, 128)
        _, w = permutation(ws, lens)
        segs, _ = segments(w, c)
        la, ac = triton_blocks(lens, ws, decode=name.startswith('decode'))
        res[name] = (n, c, len(segs), la, ac)
        print(f'{name:<34}{n:>7}{c:>6}{len(segs):>9}{la:>11}{ac:>11}')
    print('   csgmv 先按 adapter 把 token 排到一起再切段，同一 adapter 的请求共享一次权重读取；')
    print('   triton 在 decode 时也先按 adapter 排序（mlpb=8 段），差别不大；extend 时按请求发块，')
    print('   一个长请求把 grid 撑到 cdiv(8000,16)=500 行，15 个短请求的 7000 多个块空转退出。')
    print('   --max-lora-chunk-size 默认 16，表里按 128 算；默认值下 prefill 段数会多 8 倍，')
    print('   长 prefill 为主的负载值得调大。')
    return res


# ---------------------------------------------------------------- 饥饿与排空
class Req:
    def __init__(self, rid, lora, arr, max_new):
        self.rid, self.lora, self.arr, self.max_new = rid, lora, arr, max_new
        self.out = 0
        self.start = None


class Drainer:
    """lora_drainer.py 逐行对应。"""

    def __init__(self, mlpb, thr):
        self.mlpb, self.thr = mlpb, thr
        self.stats = {}

    def st(self, a):
        return self.stats.setdefault(a, dict(n=0, wait=0.0, rem=0, drain=None))

    def update(self, waiting, running, now):
        for s in self.stats.values():
            s['n'], s['wait'], s['rem'] = 0, 0.0, 0
        for r in waiting:
            s = self.st(r.lora)
            s['n'] += 1
            s['wait'] = max(s['wait'], now - r.arr)
        for r in running:
            s = self.st(r.lora)
            s['rem'] = max(s['rem'], r.max_new - r.out)
        ids = {r.lora for r in running}
        if len(ids) >= self.mlpb:
            starving = {a for a, s in self.stats.items() if s['wait'] > self.thr and s['n'] > 0}
            draining_for = {s['drain'] for s in self.stats.values() if s['drain'] is not None}
            new = sorted(starving - draining_for, key=lambda a: -self.stats[a]['wait'])
            elig = {a for a in ids if self.st(a)['drain'] is None}
            for a in new:
                if not elig:
                    break
                v = min(sorted(elig, key=str), key=lambda x: self.st(x)['rem'])
                self.st(v)['drain'] = a
                elig.remove(v)
        for a, s in self.stats.items():
            if s['drain'] is not None and a not in ids:
                s['drain'] = None

    def can_schedule(self, r):
        s = self.st(r.lora)
        return s['drain'] is None or r.max_new <= s['rem'] * DRAIN_TOL


def run_drain(thr, horizon=60.0, step=0.02, seed=0, mlpb=4, max_running=128):
    """A、B、C 和 base 各自每 0.3 s 来一个请求、输出 120~400 token；D 在 t=2 s 来一个请求。"""
    rnd = random.Random(seed)
    arrivals = []
    for lora in ('A', 'B', 'C', None):
        t = rnd.random() * 0.3
        while t < horizon:
            arrivals.append(Req(len(arrivals), lora, t, rnd.randint(120, 400)))
            t += 0.3
    d = Req(len(arrivals), 'D', 2.0, 200)
    arrivals.append(d)
    arrivals.sort(key=lambda r: r.arr)
    dr = Drainer(mlpb, thr) if thr > 0 else None
    waiting, running, done = [], [], []
    t, ai = 0.0, 0
    while t < horizon:
        while ai < len(arrivals) and arrivals[ai].arr <= t:
            waiting.append(arrivals[ai])
            ai += 1
        if dr:
            dr.update(waiting, running, t)
        run_ids = {r.lora for r in running}
        keep = []
        for r in waiting:
            if len(running) >= max_running:
                keep.append(r)
                continue
            if dr and not dr.can_schedule(r):
                keep.append(r)
                continue
            if r.lora in run_ids or validate_lora_batch(run_ids | {r.lora}, mlpb, set()):
                run_ids.add(r.lora)
                r.start = t
                running.append(r)
            else:
                keep.append(r)
        waiting = keep
        for r in running:
            r.out += 1
        done += [r for r in running if r.out >= r.max_new]
        running = [r for r in running if r.out < r.max_new]
        t += step
    hot = [r.start - r.arr for r in done if r.lora in ('A', 'B', 'C') and r.start is not None]
    return d, sorted(hot)


def scenario_drain():
    print('mlpb=4，A、B、C、base 四路流量持续不断，槽位一直满；D 在 t=2 s 来一个请求。每步 20 ms。')
    print(f'{"--lora-drain-wait-threshold":<30}{"D 开始时刻":>12}{"D 等了":>10}{"热 adapter 等待 p50":>20}{"p99":>8}')
    res = {}
    for thr in (0.0, 1.0, 5.0):
        d, hot = run_drain(thr)
        p50 = hot[len(hot) // 2]
        p99 = hot[int(len(hot) * 0.99)]
        res[thr] = (d.start, p50, p99)
        st = f'{d.start:.2f} s' if d.start is not None else '没进'
        wt = f'{d.start - d.arr:.2f} s' if d.start is not None else '> 58 s'
        print(f'{thr:<30}{st:>12}{wt:>10}{p50 * 1e3:>17.0f} ms{p99:>7.2f}s')
    print('   0 是默认值，等于不开 drainer：A/B/C 永远有请求在跑，D 每轮都被跳过，一直饿着。')
    print('   开了以后，D 等过阈值，剩余 token 最少的在跑 adapter 停收新请求，等它跑空让出槽位。')
    print('   代价是被排空的那个 adapter 的新请求要多等一轮，p99 抬高。')
    return res


# ---------------------------------------------------------------- 异步加载
def scenario_overlap(model='llama3.1-8b', gpu='h100-sxm'):
    srv = Server(model=model, gpu=gpu)
    print(f'{model} / {gpu}：一个冷 adapter 第一次被调度，batch 里已有 bs 个请求在 decode。')
    print(f'{"rank":>6}{"拷贝 MiB":>10}{"加载 ms":>9}{"bs":>5}{"一步 ms":>9}'
          f'{"同步：整批多等":>16}{"异步：只有它等":>16}')
    res = {}
    for r in (16, 64, 128):
        b = adapter_bytes(model, r)
        load = b / (H2D_GBPS * 1e9)
        for bs in (32, 128):
            step = srv.t_decode([2000] * bs)
            res[(r, bs)] = (load, step)
            print(f'{r:>6}{b / 2 ** 20:>10.0f}{load * 1e3:>9.1f}{bs:>5}{step * 1e3:>9.2f}'
                  f'{load * 1e3:>13.1f} ms{math.ceil(load / step):>11} 步后进')
    print('   同步：prepare_lora_batch 在前向之前把权重拷进槽，这一步所有请求的 ITL 都加上加载时间。')
    print('   异步（--enable-lora-overlap-loading）：side stream 上拷，拷完前新请求留在队列，其余照常 decode。')
    print('   条件：--max-loaded-loras ≤ 2 × mlpb（CPU 端要 pin 住权重）；还在拷的 adapter 也算进槽位校验。')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='memory',
                    choices=['memory', 'admit', 'segments', 'drain', 'overlap'])
    a = ap.parse_args()
    {'memory': scenario_memory, 'admit': scenario_admit, 'segments': scenario_segments,
     'drain': scenario_drain, 'overlap': scenario_overlap}[a.scenario]()


if __name__ == '__main__':
    main()
