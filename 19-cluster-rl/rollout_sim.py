"""RL rollout 的账：GRPO 的长尾输出让同步 rollout 把 GPU 空着等最后几条，APRIL 式的超发 + 截停 + 回收能补回多少，
以及每个 RL step 之间那次权重同步要花多少时间。

一台推理引擎的模型（Qwen3-8B / 1xH100，单步耗时取自 ../03-bench/sim_serve.py 的 Server）：
  - 每个 RL step 这台引擎分到 32 个 prompt x 每组 8 条采样（GRPO 的 group），prompt 200-1000 token；
    输出长度按对数正态分布，组内共享一个难度偏移（难题整组都长），截断在 max_new_tokens=16384
  - 调度：prefill 优先，一批 prefill 不超过 chunk；同组 8 条共享 prompt，第一条付 prompt 的 prefill，
    其余命中前缀缓存只算 1 个 token（RadixCache，SGL-08）
  - 准入：已用 KV + 新请求 + 每条在跑请求预留 256 token 放得下才收；decode 一步放不下就 retract
    已输出最少的请求（schedule_batch.py retract_decode 的排序），被撤回的请求回到队首，再次准入时重算已输出的部分
  - 每个 RL step 开始前权重已经换过、前缀缓存已 flush（flush_cache_after_weight_update），所以上一步的缓存全部作废

两种 rollout 方式：
  sync     发 32 组，等 256 条全部生成完，交给训练
  april    超发 (1 + a) x 32 组，凑够 32 组完整的就 pause_generation（abort），没写完的采样带着已生成的部分
           进回收池，下一步先续写它们（APRIL, arXiv 2509.18521；sglang 文档 Easy To Postpone Generation 一节）。
           续写的 token 用的是新权重，同一条采样里混着两个版本，sglang 用 meta_info.weight_versions 按 token 标注
  drop     超发但不回收：凑够就截停，没写完的直接丢掉。吞吐和 april 接近，可进训练批的全是写得快的短采样，
           长度分布被截偏

权重同步（--sync-cost）：每卡要收的字节 / 有效带宽 + 调用次数 x 每次的固定延迟。有效带宽按 checkpoint-engine
README 的实测反推（每卡收到的字节 / Update(Broadcast) 时间），固定延迟是假设值，可以用参数改。

用法：
    python3 rollout_sim.py                  # 同步 rollout 的长尾 + 超发比例对比
    python3 rollout_sim.py --sync-cost      # 三个模型、四种权重同步路径的耗时
    python3 rollout_sim.py --json
"""
import argparse
import heapq
import json
import math
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))
import sim_serve  # noqa: E402
from common.specs import MODELS, weight_bytes  # noqa: E402

MAX_NEW = 16384
HEADROOM = 256


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pad(s, n):
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    return s + ' ' * max(0, n - w)


# ---------------------------------------------------------------- 采样
class Sample:
    __slots__ = ('gid', 'k', 'inp', 'out', 'gen', 'segs', 'done', 'join', 'gen0', 'retracts', 't_done')

    def __init__(self, gid, k, inp, out):
        self.gid, self.k, self.inp, self.out = gid, k, inp, out
        self.gen = 0                 # 已生成 token 数（在跑时由 引擎步数 - join + gen0 推出）
        self.segs = []               # [(权重版本, 从第几个输出 token 起)]
        self.done = False
        self.join = 0
        self.gen0 = 0
        self.retracts = 0
        self.t_done = None


class Group:
    __slots__ = ('gid', 'samples', 'born')

    def __init__(self, gid, samples, born):
        self.gid, self.samples, self.born = gid, samples, born

    @property
    def complete(self):
        return all(s.done for s in self.samples)


class LengthGen:
    def __init__(self, seed=3, G=8, median=1200, group_sigma=0.6, sigma=0.8, inp=(200, 1000)):
        self.rng = random.Random(seed)
        self.G, self.mu, self.gs, self.s, self.inp = G, math.log(median), group_sigma, sigma, inp
        self.gid = 0

    def group(self, born):
        rng = self.rng
        inp = rng.randint(*self.inp)
        shift = rng.gauss(0, self.gs)
        ss = []
        for k in range(self.G):
            o = int(math.exp(self.mu + shift + rng.gauss(0, self.s)))
            ss.append(Sample(self.gid, k, inp, max(16, min(MAX_NEW, o))))
        g = Group(self.gid, ss, born)
        self.gid += 1
        return g


# ---------------------------------------------------------------- 引擎
class Engine:
    """一个 RL step 内的一台引擎；step 之间重新建（前缀缓存已 flush）。"""

    def __init__(self, srv, version):
        self.srv, self.v = srv, version
        self.cap, self.chunk, self.max_running = srv.cap, srv.chunk, srv.max_running
        self.waiting = deque()
        self.running = set()
        self.heap = []
        self.d = 0                   # decode 步计数
        self.now = 0.0
        self.sum_gen = 0             # 在跑请求的已生成 token 之和
        self.sum_inp = 0             # 在跑请求的 prompt 之和（算注意力上下文用，每条都读自己的 prompt KV）
        self.grp_live = {}           # gid -> 在跑的组员数；prompt 的 KV 每组只占一份
        self.kv_prompt = 0
        self.cached = set()          # 本 step 里 prompt 已进前缀缓存的组
        self.busy_decode = self.busy_prefill = 0.0
        self.tokens = 0
        self.bs_time = 0.0           # sum(bs x 步时)，算时间加权的平均 batch
        self.retracts = 0
        self.recompute = 0           # retract / 回收后重算的输出 token
        self.prefill_tok = 0
        self._dcache = {}
        self.seq = 0

    def gen_of(self, s):
        return s.gen0 + self.d - s.join

    def used(self):
        return self.kv_prompt + self.sum_gen

    def submit(self, s, front=False):
        (self.waiting.appendleft if front else self.waiting.append)(s)

    def _enter(self, s):
        s.join, s.gen0 = self.d, s.gen
        if not s.segs or s.segs[-1][0] != self.v:
            s.segs.append((self.v, s.gen))
        self.running.add(s)
        self.sum_gen += s.gen
        self.sum_inp += s.inp
        n = self.grp_live.get(s.gid, 0)
        if n == 0:
            self.kv_prompt += s.inp
        self.grp_live[s.gid] = n + 1
        self.seq += 1
        heapq.heappush(self.heap, (s.join + s.out - s.gen, self.seq, s))

    def _leave(self, s):
        s.gen = self.gen_of(s)
        self.running.discard(s)
        self.sum_gen -= s.gen
        self.sum_inp -= s.inp
        n = self.grp_live[s.gid] - 1
        self.grp_live[s.gid] = n
        if n == 0:
            self.kv_prompt -= s.inp

    def _t_decode(self, bs, ctx):
        key = (bs, int(ctx) >> 6)
        t = self._dcache.get(key)
        if t is None:
            d = self.srv.s.decode(bs, ctx)
            t = self._dcache[key] = self.srv._t(d['flops'], d['bytes'])
        return t

    def step(self):
        """推进一步，返回这一步里写完的采样；没有可做的事返回 None。"""
        # prefill 优先
        if self.waiting and len(self.running) < self.max_running:
            parts, batch, new_kv, budget = [], [], 0, self.chunk
            while self.waiting and len(self.running) + len(batch) < self.max_running:
                s = self.waiting[0]
                first = s.gid not in self.cached and all(b.gid != s.gid for b in batch)
                share = self.grp_live.get(s.gid, 0) > 0 or any(b.gid == s.gid for b in batch)
                need = (s.inp if first else 1) + s.gen
                kv = s.gen + (0 if share else s.inp)
                if batch and need > budget:
                    break
                if self.used() + new_kv + kv + HEADROOM * (len(self.running) + len(batch) + 1) > self.cap:
                    break
                self.waiting.popleft()
                batch.append(s)
                parts.append((need, 0 if first else s.inp))
                new_kv += kv
                budget -= need
                self.recompute += s.gen
            if batch:
                t = self.srv.t_prefill(parts)
                self.now += t
                self.busy_prefill += t
                for s in batch:
                    self.cached.add(s.gid)
                    self.prefill_tok += parts[batch.index(s)][0]
                    self._enter(s)
                return []
        if not self.running:
            return None
        # decode 一步放不下就 retract 已输出最少的
        while self.used() + len(self.running) > self.cap and len(self.running) > 1:
            victim = min(self.running, key=lambda x: (self.gen_of(x), -x.inp, x.gid, x.k))   # 并列时按组号定，结果可复现
            self._leave(victim)
            victim.retracts += 1
            self.retracts += 1
            self.heap = [e for e in self.heap if e[2] is not victim]
            heapq.heapify(self.heap)
            self.submit(victim, front=True)
        bs = len(self.running)
        t = self._t_decode(bs, (self.sum_inp + self.sum_gen) / bs)
        self.now += t
        self.busy_decode += t
        self.bs_time += bs * t
        self.d += 1
        self.sum_gen += bs
        self.tokens += bs
        done = []
        while self.heap and self.heap[0][0] <= self.d:
            _, _, s = heapq.heappop(self.heap)
            self._leave(s)
            s.done = True
            s.t_done = self.now
            done.append(s)
        return done

    def abort_all(self):
        """pause_generation(mode=abort)：在跑的带着已生成的部分返回，排队的原样返回。"""
        out = []
        for s in sorted(self.running, key=lambda x: (x.gid, x.k)):
            self._leave(s)
            out.append(s)
        out.extend(self.waiting)
        self.waiting.clear()
        self.heap = []
        return out


# ---------------------------------------------------------------- 一串 RL step
def run(alpha=0.0, steps=8, P=32, G=8, seed=3, srv=None, trace=False, recycle=True):
    srv = srv or sim_serve.Server('qwen3-8b', 'h100-sxm')
    lg = LengthGen(seed, G)
    carry = []                       # 上一步没写完的组（含其中已写完的采样）
    ready = []                       # 已完整但超出本步需求的组，留给下一步
    rows, timeline = [], None
    for v in range(steps):
        eng = Engine(srv, v)
        groups = list(carry)
        target = P if alpha == 0 else int(round((1 + alpha) * P))
        while len(groups) + len(ready) < target:
            groups.append(lg.group(v))
        # 回收的先续写：没写完的采样排在队首
        for g in groups:
            for s in g.samples:
                if not s.done and g.born < v:
                    eng.submit(s)
        for g in groups:
            for s in g.samples:
                if not s.done and g.born == v:
                    eng.submit(s)
        need = P - len(ready)
        batch = list(ready)
        ready = []
        live = {g.gid: g for g in groups}
        fin_times = []
        bs_trace = []
        while True:
            done = eng.step()
            if done is None:
                break
            if trace and v == 0:
                bs_trace.append((eng.now, len(eng.running)))
            for s in done:
                fin_times.append(eng.now)
                g = live.get(s.gid)
                if g is not None and g.complete:
                    del live[s.gid]
                    if need > 0:
                        batch.append(g)
                        need -= 1
                    else:
                        ready.append(g)
            if alpha > 0 and need <= 0:
                break
        recycled = eng.abort_all()
        carry = list(live.values()) if recycle else []
        dropped = 0 if recycle else sum(s.gen for s in recycled)
        # 训练批的统计
        out_tok = sum(s.out for g in batch for s in g.samples)
        stale_tok = 0
        stale_samples = 0
        max_lag = 0
        for g in batch:
            for s in g.samples:
                segs = s.segs + [(None, s.out)]
                old = sum(segs[i + 1][1] - segs[i][1] for i in range(len(s.segs)) if segs[i][0] < v)
                stale_tok += old
                stale_samples += old > 0
                if s.segs:
                    max_lag = max(max_lag, v - s.segs[0][0])
        row = {
            'version': v, 'time': eng.now, 'groups_trained': len(batch),
            'samples_trained': sum(len(g.samples) for g in batch),
            'trained_out_tokens': out_tok, 'generated': eng.tokens,
            'decode_busy': eng.busy_decode, 'prefill_busy': eng.busy_prefill,
            'avg_bs': eng.bs_time / eng.busy_decode if eng.busy_decode else 0.0,
            'retracts': eng.retracts, 'recompute': eng.recompute, 'prefill_tok': eng.prefill_tok,
            'recycled_samples': sum(1 for s in recycled if s.gen > 0),
            'recycled_tokens': sum(s.gen for s in recycled),
            'carry_groups': len(carry), 'stale_token_share': stale_tok / out_tok if out_tok else 0.0,
            'stale_sample_share': stale_samples / max(1, sum(len(g.samples) for g in batch)),
            'max_lag': max_lag,
            'dropped_tokens': dropped,
            'out_mean': out_tok / max(1, sum(len(g.samples) for g in batch)),
            'truncated': sum(1 for g in batch for s in g.samples if s.out >= MAX_NEW),
        }
        if v == 0:
            n = len(fin_times)
            row['t50'] = fin_times[int(n * 0.5) - 1] if n else 0.0
            row['t90'] = fin_times[int(n * 0.9) - 1] if n else 0.0
            row['out_p50'] = pct([s.out for g in batch for s in g.samples], 0.5)
            row['out_max'] = max((s.out for g in batch for s in g.samples), default=0)
            if trace:
                timeline = bs_trace
        rows.append(row)
    res = summarize(rows)
    if trace:
        res['timeline'] = timeline
    return res


def summarize(rows):
    T = sum(r['time'] for r in rows)
    tok = sum(r['trained_out_tokens'] for r in rows)
    gen = sum(r['generated'] for r in rows)
    busy = sum(r['decode_busy'] for r in rows)
    return {
        'steps': rows, 'rollout_time': T, 'time_per_step': T / len(rows),
        'trained_tok_per_s': tok / T, 'gen_tok_per_s': gen / T,
        'avg_bs': sum(r['avg_bs'] * r['decode_busy'] for r in rows) / busy if busy else 0.0,
        'stale_token_share': sum(r['stale_token_share'] * r['trained_out_tokens'] for r in rows) / tok,
        'stale_sample_share': sum(r['stale_sample_share'] * r['samples_trained'] for r in rows) /
        sum(r['samples_trained'] for r in rows),
        'max_lag': max(r['max_lag'] for r in rows),
        'retracts': sum(r['retracts'] for r in rows),
        'recompute': sum(r['recompute'] for r in rows),
        'samples_trained': sum(r['samples_trained'] for r in rows),
        'dropped_tokens': sum(r['dropped_tokens'] for r in rows),
        'out_mean': tok / sum(r['samples_trained'] for r in rows),
    }


def compare(alphas=(0.0, 0.25, 0.5, 1.0), steps=8):
    out = {('april', a): run(a, steps, trace=(a == 0.0)) for a in alphas}
    out[('drop', 0.5)] = run(0.5, steps, recycle=False)
    return out


def name_of(k):
    mode, a = k
    if a == 0:
        return '同步'
    return f'超发 {a:.0%}' + ('' if mode == 'april' else ' 不回收')


def report_tail(R):
    r = R['steps'][0]
    print('第一步同步 rollout：32 组 x 8 条，Qwen3-8B / 1xH100')
    print(f'  输出长度中位数 {r["out_p50"]:,.0f} token，最长 {r["out_max"]:,}，截断在 {MAX_NEW:,} 的 {r["truncated"]} 条')
    print(f'  一半采样 {r["t50"]:.1f} s 写完，90% 在 {r["t90"]:.1f} s，最后一条在 {r["time"]:.1f} s；'
          f'最后 10% 的采样占了 {1 - r["t90"] / r["time"]:.0%} 的时间')
    tl = R.get('timeline') or []
    if tl:
        marks = []
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
            t = r['time'] * frac
            bs = next((b for tt, b in tl if tt >= t), 0)
            marks.append(f'{frac:.0%} 处 {bs}')
        print('  在跑 batch 随时间：' + '，'.join(marks))
    print(f'  decode 时间加权的平均 batch {r["avg_bs"]:.0f}，retract {r["retracts"]} 次、重算 {r["recompute"]:,} token')


def report_compare(out):
    print('同一串难度的 prompt，跑 8 个 RL step（时间只算 rollout，秒）')
    print('  ' + pad('方式', 20) + '每步 rollout  有效 token/s  平均 batch  训练批平均长度  含旧权重的采样  旧权重 token  最大滞后  丢弃 token')
    base = out[('april', 0.0)]
    for k, R in out.items():
        print(f'  {pad(name_of(k), 20)}{R["time_per_step"]:>12.1f}{R["trained_tok_per_s"]:>14,.0f}'
              f'{R["avg_bs"]:>12.0f}{R["out_mean"]:>16,.0f}{R["stale_sample_share"]:>16.0%}{R["stale_token_share"]:>14.0%}'
              f'{R["max_lag"]:>10}{R["dropped_tokens"]:>12,}')
    print('  有效 token/s = 进了训练批的采样的输出 token / rollout 时间（训练和权重同步的时间另算）')
    for k, R in out.items():
        if k[1] > 0:
            g = R['trained_tok_per_s'] / base['trained_tok_per_s'] - 1
            print(f'  {name_of(k)}：有效吞吐 {g:+.0%}，训练批平均长度 {base["out_mean"]:,.0f} → {R["out_mean"]:,.0f}，'
                  f'续写时重算 {R["recompute"]:,} token、retract {R["retracts"]} 次')


# ---------------------------------------------------------------- 权重同步
# checkpoint-engine README 的 Update(Broadcast) 实测：(模型, 卡, 总权重字节, 卡数, 秒)
CKPT_ENGINE = [
    ('GLM-4.5-Air BF16', '8xH800 TP8', 106e9 * 2, 8, 3.47),
    ('Qwen3-235B-A22B BF16', '8xH800 TP8', 235.1e9 * 2, 8, 6.22),
    ('DeepSeek-V3.1 FP8', '16xH20 TP16', 671e9, 16, 10.19),
    ('Kimi-K2 FP8', '16xH20 TP16', 1.04e12, 16, 14.36),
]


def tensor_count(m):
    """checkpoint 里的张量个数（近似）：每层注意力 4 个投影 + 2 个 norm（Qwen3 再加 q/k norm），
    稠密 MLP 3 个，每个路由专家 3 个外加 router；MLA 按 8 个投影算。FP8 每个线性层多一个 scale。"""
    attn = 8 if m.mla_kv_rank else 6
    dense_layers = m.layers - m.moe_layers
    per_dense = attn + 3
    per_moe = attn + 1 + 3 * m.experts + (3 if m.experts else 0)   # router + 专家 + 共享专家
    n = dense_layers * per_dense + m.moe_layers * per_moe + 3
    if m.native_dtype == 'fp8':
        n += m.layers * 4 + m.moe_layers * 3 * m.experts
    return n


def sync_cost(model, tp, nodes, bw_gbs, lat_tensor=30e-6, lat_call=20e-3, bucket_gb=1.0, finalize=0.5):
    m = MODELS[model]
    total = weight_bytes(m, m.native_dtype)
    per_gpu = total / tp
    nt = tensor_count(m)
    xfer = per_gpu / (bw_gbs * 1e9)
    nb = math.ceil(per_gpu / (bucket_gb * 2 ** 30))
    return {
        'model': m.name, 'tp': tp, 'nodes': nodes, 'bytes_per_gpu': per_gpu, 'tensors': nt, 'buckets': nb,
        # 逐张量 broadcast：一次调用里按张量循环，每个张量一次 NCCL 调用
        'per_tensor': lat_call + nt * lat_tensor + xfer + finalize,
        # flattened bucket：每桶一次调用、一次 broadcast
        'bucketed': nb * (lat_call + lat_tensor) + xfer + finalize,
        # 每个张量单独发一次请求（不打包、不分组的最差写法）
        'per_tensor_call': nt * (lat_call + lat_tensor) + xfer + finalize,
    }


def disk_cost(model, write_gbs=2.0, read_gbs=4.0, finalize=0.5):
    m = MODELS[model]
    total = weight_bytes(m, m.native_dtype)
    return total / (write_gbs * 1e9) + total / (read_gbs * 1e9) + finalize


def sync_table(bw_node=9.0, bw_multi=4.3, lat_tensor=30e-6, lat_call=20e-3, bucket_gb=1.0):
    cal = []
    for name, dev, total, n, t in CKPT_ENGINE:
        cal.append({'model': name, 'device': dev, 'gb_per_gpu': total / n / 1e9, 'seconds': t,
                    'eff_gbs': total / n / 1e9 / t})
    rows = []
    for model, tp, nodes in [('qwen3-8b', 1, 1), ('qwen3-235b-a22b', 8, 1), ('deepseek-v3', 16, 2)]:
        bw = bw_node if nodes == 1 else bw_multi
        r = sync_cost(model, tp, nodes, bw, lat_tensor, lat_call, bucket_gb)
        r['disk'] = disk_cost(model)
        r['bw'] = bw
        rows.append(r)
    return {'calibration': cal, 'rows': rows,
            'params': dict(bw_node=bw_node, bw_multi=bw_multi, lat_tensor=lat_tensor, lat_call=lat_call,
                           bucket_gb=bucket_gb)}


def report_sync(S):
    print('checkpoint-engine 实测反推的每卡有效带宽（Update Broadcast，权重从 CPU 内存出发，含 H2D）')
    print('  ' + pad('模型', 24) + pad('设备', 14) + '每卡 GB    秒   有效 GB/s')
    for c in S['calibration']:
        print(f'  {pad(c["model"], 24)}{pad(c["device"], 14)}{c["gb_per_gpu"]:>7.1f}{c["seconds"]:>6.2f}{c["eff_gbs"]:>11.1f}')
    p = S['params']
    print(f'估算（单机有效带宽 {p["bw_node"]} GB/s、跨机 {p["bw_multi"]} GB/s；每个张量的 NCCL 调用 {p["lat_tensor"] * 1e6:.0f} µs、'
          f'每次 HTTP 调用 {p["lat_call"] * 1e3:.0f} ms、桶 {p["bucket_gb"]} GiB、end_weight_update 0.5 s，后三个是假设值）')
    print('  ' + pad('模型', 18) + '卡   每卡 GB   张量数  桶数  逐张量一次调用  按桶  每张量一次调用  走磁盘')
    for r in S['rows']:
        print(f'  {pad(r["model"], 18)}{r["tp"]:>3}{r["bytes_per_gpu"] / 1e9:>10.1f}{r["tensors"]:>9,}{r["buckets"]:>6}'
              f'{r["per_tensor"]:>14.1f}{r["bucketed"]:>7.1f}{r["per_tensor_call"]:>16.1f}{r["disk"]:>8.0f}')
    print('  走磁盘按训练侧写 2 GB/s、推理侧读 4 GB/s 算整份 checkpoint，和卡数无关')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sync-cost', action='store_true')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--alpha', type=float, nargs='*', default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument('--bw', type=float, default=9.0, help='单机每卡有效带宽 GB/s')
    ap.add_argument('--lat-call', type=float, default=20.0, help='每次 HTTP 调用的固定延迟，毫秒')
    ap.add_argument('--bucket-gb', type=float, default=1.0)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.sync_cost:
        S = sync_table(bw_node=a.bw, lat_call=a.lat_call / 1e3, bucket_gb=a.bucket_gb)
        if a.json:
            print(json.dumps(S, ensure_ascii=False, indent=1))
        else:
            report_sync(S)
        return S
    out = compare(tuple(sorted(set([0.0] + a.alpha))), a.steps)
    if a.json:
        print(json.dumps({f'{k[0]}_{k[1]}': {kk: vv for kk, vv in v.items() if kk != 'timeline'} for k, v in out.items()},
                         ensure_ascii=False, indent=1))
        return out
    report_tail(out[('april', 0.0)])
    report_compare(out)
    return out


if __name__ == '__main__':
    main()
