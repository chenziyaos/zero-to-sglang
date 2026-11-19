"""HiCache 三层缓存：搬运和重算的账、写回策略、多轮对话命中率、L3 预取、页哈希。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  mem_cache/unified_radix_cache.py:570      write_through_threshold：write_through 为 1，selective 为 2
  mem_cache/unified_radix_cache.py:588      load_back_threshold = 10，主机命中不足 10 个 token 不搬
  mem_cache/unified_cache/unified_tree_core.py:1104  _inc_hit_count_and_check：命中计数到阈值就发 D2H 备份
  mem_cache/unified_cache/unified_tree_core.py:1788  evict_device_leaf：已备份降级到主机、未备份 write_through 直接删、
                                                    write_back 先备份再降级
  mem_cache/unified_cache/unified_tree_core.py:2366  _is_host_leaf：只有已离开显存的节点才能从主机池淘汰
  mem_cache/unified_radix_cache.py:1798     _execute_kv_backup：主机池不够先淘汰主机叶子，还不够就放弃备份
  managers/cache_controller.py:628-634      prefetch_capacity_limit：cache 模式留一半主机池给写回
  mem_cache/unified_radix_cache.py:2142     prefetch_from_storage：页对齐后短于 prefetch_threshold（256）不预取
  mem_cache/unified_radix_cache.py:2320     _start_storage_prefetch_wait：timeout = base + 页数 × per_page
  mem_cache/unified_radix_cache.py:2356     _can_terminate_prefetch：best_effort / wait_complete / timeout 三种停法
  mem_cache/unified_cache/storage_attachment.py:290  per_page = page_size / 1024 × prefetch_timeout_per_ki_token
  mem_cache/hybrid_cache/hybrid_cache_controller.py:404  extra_config 默认：threshold 256、base 1、per_ki 0.25
  managers/cache_controller.py:1419-1436    _page_backup / 预取按 128 页一批推进 completed_tokens
  mem_cache/utils.py:125-152                storage_namespace_seed / get_storage_hash_str：L3 键的命名空间种子
  mem_cache/cpp_utils/hash_binding.cpp      逐页 SHA256(上一页摘要 + token 的 uint32 小端字节)
  mem_cache/hicache_storage.py:728          HiCacheFile：文件名 = 页哈希 + _{model}_{tp_rank}_{tp_size}.bin

  python3 hicache_sim.py                         # cost：每 token 重算和搬运的微秒数、32K 前缀的 TTFT
  python3 hicache_sim.py --scenario multiturn    # multiturn：80 个会话轮询，五种配置的分层命中率
  python3 hicache_sim.py --scenario prefetch     # prefetch：三种预取停止策略在不同带宽下的 TTFT
  python3 hicache_sim.py --scenario hash         # hash：页哈希链、cache_salt 隔离、文件后端的键名
"""
import argparse
import hashlib
import os
import random
import struct
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))

from common.specs import GPUS, MODELS  # noqa: E402

GiB = 1 << 30
PAGE = 64                      # 最佳实践推荐的 page size
MFU = 0.5                      # prefill 能跑到峰值算力的比例，H100 上 40%~60% 常见
STORAGE_BATCH_PAGES = 128      # hicache_storage.py:24 STORAGE_BATCH_SIZE

# 每个 rank 实际拿到的有效带宽（GB/s）。换成自己机器的实测值再跑
LINKS = OrderedDict([
    ('L2 PCIe Gen5 x16', 50.0),     # 理论 64 GB/s 单向，pinned memcpy 实测 50 出头
    ('L2 PCIe Gen4 x16', 25.0),
    ('L3 RDMA 400G/卡', 40.0),       # 每卡一张 400Gbps 网卡，扣协议开销
    ('L3 3FS 40GiB/s÷8', 40 * GiB / 1e9 / 8),   # 3FS README：KVCache 客户端单机峰值 40 GiB/s，8 卡平分
    ('L3 本地NVMe÷8', 0.8),          # 一块 PCIe Gen4 NVMe 约 6~7 GB/s，8 个 rank 平分
])

# (模型, GPU, tp)：和 02-ledger 的切池口径一致
CONFIGS = [
    ('llama3.1-8b', 'h100-sxm', 1),
    ('llama3.1-70b', 'h100-sxm', 4),
    ('qwen3-235b-a22b', 'h100-sxm', 8),
    ('deepseek-v3', 'h200', 8),
]


def pool(model, gpu, tp):
    """复用 02-ledger：每个 rank 的 KV 池 token 数和每 token 字节数（MLA 每个 rank 存完整 latent）。"""
    from sim_ledger import ledger
    L = ledger(model, gpu, tp=tp)
    return L['tokens'], L['cell'], L['weight_dtype']


# ---------------------------------------------------------------- 搬运和重算的账

def prefill_ms(m, card, tp, new, ctx=0, weight_dtype='bf16', mfu=MFU):
    """new 个 token 接在 ctx 个已有 token 后面做 prefill，单个 rank 的耗时（Roofline 加 MFU 折扣）。

    线性层 2·active 次浮点/token；注意力 QK 和 PV 每对 (q, k) 每头 4·head_dim 次，因果掩码下
    新 token 平均看到 ctx + new/2 个 key。TP 把两部分都均分到各卡。
    """
    lin_tf = card.fp8_tflops if weight_dtype == 'fp8' and card.fp8_tflops else card.bf16_tflops
    lin = 2 * new * m.active_b * 1e9 / tp
    attn = 4 * new * (ctx + new / 2) * m.layers * m.heads * m.head_dim / tp
    t_compute = lin / (lin_tf * 1e12 * mfu) + attn / (card.bf16_tflops * 1e12 * mfu)
    wbytes = m.active_b * 1e9 * (1 if weight_dtype == 'fp8' else 2) / tp
    return max(t_compute, wbytes / (card.bw_tbs * 1e12)) * 1e3


def load_ms(tokens, cell, gbs):
    return tokens * cell / (gbs * 1e9) * 1e3


def pipeline_ms(load, compute, layers):
    """逐层重叠：第 i 层的 KV 到了才能算第 i 层（LayerDoneCounter.wait_until），同时第 i+1 层在搬。"""
    tl, tc = load / layers, compute / layers
    load_end = comp_end = 0.0
    for _ in range(layers):
        load_end += tl
        comp_end = max(comp_end, load_end) + tc
    return comp_end


def breakeven_tokens(m, card, tp, cell, gbs, weight_dtype, cap=1 << 20):
    """前缀从多长开始，往后任何长度从这条链路搬回来都比重算便宜。

    重算的曲线不是直线：很短时被读权重的下限托住，很长时注意力的平方项抬头，中间一段可能输给搬运。
    所以从 cap 往下按页扫，找最后一个搬运更贵的长度。返回 0 表示任何长度都划算，None 表示 cap 内都不划算。
    """
    if load_ms(cap, cell, gbs) > prefill_ms(m, card, tp, cap, 0, weight_dtype):
        return None
    n = cap
    while n >= PAGE:
        if load_ms(n, cell, gbs) > prefill_ms(m, card, tp, n, 0, weight_dtype):
            return n + PAGE
        n -= PAGE
    return 0


def scenario_cost(prefix=32768, new=2048):
    res = {}
    print(f'每 token 的代价（微秒/token，单个 rank）。重算按 MFU {MFU:.0%}，前缀很短时只剩线性层')
    head = f'  {"模型 × 卡":<28}{"cell B":>8}{"重算":>8}' + ''.join(f'{k:>18}' for k in LINKS)
    print(head)
    for model, gpu, tp in CONFIGS:
        m, card = MODELS[model], GPUS[gpu]
        _, cell, wd = pool(model, gpu, tp)
        re_us = prefill_ms(m, card, tp, 4096, 0, wd) / 4096 * 1e3
        row = {k: cell / (v * 1e9) * 1e6 for k, v in LINKS.items()}
        res[model] = dict(cell=cell, recompute_us=re_us, link_us=row, wd=wd)
        print(f'  {m.name + " " + str(tp) + "×" + card.name:<28}{cell:>8}{re_us:>8.1f}'
              + ''.join(f'{row[k]:>18.1f}' for k in LINKS))
    print()
    print('从哪条链路搬回来开始比重算快（前缀 token 数，0 = 任何长度都快，- = 1M 以内都不划算）')
    print(f'  {"模型":<20}' + ''.join(f'{k:>18}' for k in LINKS))
    for model, gpu, tp in CONFIGS:
        m, card = MODELS[model], GPUS[gpu]
        cell, wd = res[model]['cell'], res[model]['wd']
        be = {k: breakeven_tokens(m, card, tp, cell, v, wd) for k, v in LINKS.items()}
        res[model]['breakeven'] = be
        print(f'  {m.name:<20}' + ''.join(f'{"-" if b is None else b:>18}' for b in be.values()))
    print()
    print(f'一条 {prefix // 1024}K 前缀 + {new} 新 token 的请求，prefill 阶段耗时（ms）')
    print(f'  {"模型":<20}{"全部重算":>10}{"L2 串行":>10}{"L2 逐层重叠":>12}{"3FS 串行":>10}{"3FS 重叠":>10}')
    for model, gpu, tp in CONFIGS:
        m, card = MODELS[model], GPUS[gpu]
        cell, wd = res[model]['cell'], res[model]['wd']
        full = prefill_ms(m, card, tp, prefix + new, 0, wd)
        ext = prefill_ms(m, card, tp, new, prefix, wd)
        l2 = load_ms(prefix, cell, LINKS['L2 PCIe Gen5 x16'])
        l3 = load_ms(prefix, cell, LINKS['L3 3FS 40GiB/s÷8'])
        row = dict(full=full, l2_serial=l2 + ext, l2_overlap=pipeline_ms(l2, ext, m.layers),
                   l3_serial=l3 + ext, l3_overlap=pipeline_ms(l3, ext, m.layers))
        res[model]['ttft'] = row
        print(f'  {m.name:<20}{full:>10.0f}{row["l2_serial"]:>10.0f}{row["l2_overlap"]:>12.0f}'
              f'{row["l3_serial"]:>10.0f}{row["l3_overlap"]:>10.0f}')
    print()
    print('读法：L2 比重算快一个数量级，开 L2 基本只赚不赔；每卡一张 400G 网卡的 RDMA 和 L2 同一量级。')
    print('  3FS 按每节点 40 GiB/s 被 8 个 rank 平分后，稠密模型上仍然稳赢；DeepSeek-V3 激活参数少、')
    print('  又是 FP8，短前缀时重算每 token 比 3FS 搬运还便宜，前缀到几千 token、注意力项涨上来才回本。')
    print('  本地 NVMe 平分后每 rank 不到 1 GB/s，前缀要长到十几万 token 才回本。')
    return res


# ---------------------------------------------------------------- 多轮对话的分层命中

class TierSim:
    """按页模拟 HiRadixTree 的三层状态。

    键是 (会话, 页号)，同一会话的页构成一条链，前缀天然共享。LRU 时间戳从深到浅刷新，
    祖先总比后代新，等价于 radix tree 的“只淘汰叶子”。
    """

    def __init__(self, dev_pages, host_pages=0, policy='write_through', storage=False,
                 page=PAGE, prefetch_threshold=256, mode='cache'):
        self.dev_cap, self.host_cap = dev_pages, host_pages
        self.policy, self.storage, self.page = policy, storage, page
        self.threshold = 1 if policy == 'write_through' else 2       # unified_radix_cache.py:570
        self.prefetch_pages = -(-prefetch_threshold // page)
        self.prefetch_limit = host_pages // 2 if mode == 'cache' else int(0.9 * host_pages)
        self.dev = OrderedDict()        # 在显存里的页
        self.host_only = OrderedDict()  # 离开显存、只剩主机副本的页（H-leaf 候选）
        self.backed = set()             # 有主机副本的页，占主机池
        self.l3 = set()
        self.hits = {}
        self.st = dict(dev=0, host=0, l3=0, miss=0, d2h=0, l3w=0, dropped=0,
                       backup_fail=0, l3_host_capacity=0)

    # 主机池：只有离开显存的页能被淘汰（_is_host_leaf 要求 node.evicted）
    def _host_alloc(self, n, pinned=()):
        while len(self.backed) + n > self.host_cap:
            victim = next((k for k in self.host_only if k not in pinned), None)
            if victim is None:
                return False
            del self.host_only[victim]
            self.backed.discard(victim)
            self.hits.pop(victim, None)
        return True

    def _backup(self, k, pinned=()):
        if k in self.backed:
            return True
        if not self.host_cap or not self._host_alloc(1, pinned):
            self.st['backup_fail'] += 1
            return False
        self.backed.add(k)
        self.st['d2h'] += 1
        if self.storage and k not in self.l3:       # _finish_write_through_ack -> write_backup_storage
            self.l3.add(k)
            self.st['l3w'] += 1
        return True

    def _evict_dev(self, need, pinned):
        while len(self.dev) + need > self.dev_cap:
            k = next(k for k in self.dev if k not in pinned)
            del self.dev[k]
            if k not in self.backed and self.policy == 'write_back':
                self._backup(k, pinned)            # evict_device_leaf：write_back 先备份
            if k in self.backed:
                self.host_only[k] = True           # 降级成主机叶子
            else:
                self.st['dropped'] += 1            # write_through_unbacked_eviction / host_pressure
                self.hits.pop(k, None)

    def request(self, keys):
        """keys 是这条请求按顺序的全部页。返回 (显存命中, 主机命中, L3 命中, 重算) 页数。"""
        n = len(keys)
        i = 0
        while i < n and (keys[i] in self.dev or keys[i] in self.backed):
            i += 1
        matched = i
        dev_hit = sum(1 for k in keys[:matched] if k in self.dev)
        host_hit = matched - dev_hit
        l3_hit = 0
        if self.storage:
            j = matched
            while j < n and keys[j] in self.l3:
                j += 1
            run = j - matched
            if run >= self.prefetch_pages:
                want = min(run, self.prefetch_limit)
                pinned = set(keys)
                got = 0
                while got < want and self._host_alloc(1, pinned):
                    self.backed.add(keys[matched + got])
                    got += 1
                if got < self.prefetch_pages:      # allocate_storage_hit(min_tokens=threshold) 失败
                    for k in keys[matched:matched + got]:
                        self.backed.discard(k)
                    got = 0
                self.st['l3_host_capacity'] += run - got
                l3_hit = got
        miss = n - matched - l3_hit
        pinned = set(keys)
        self._evict_dev(n - dev_hit, pinned)
        for k in reversed(keys):                   # 深的先刷新，祖先最新
            self.host_only.pop(k, None)
            self.dev[k] = True
            self.dev.move_to_end(k)
        for idx, k in enumerate(keys):
            self.hits[k] = self.hits.get(k, 0) + 1
            if self.policy != 'write_back' and self.hits[k] >= self.threshold and k not in self.backed:
                self._backup(k, pinned)
        for name, v in (('dev', dev_hit), ('host', host_hit), ('l3', l3_hit), ('miss', miss)):
            self.st[name] += v
        return dev_hit, host_hit, l3_hit, miss


def multiturn_trace(clients=80, rounds=10, req_tokens=2048, page=PAGE, seed=0):
    """bench_multiturn 的形状：每轮每个会话追加 req_tokens，轮内随机顺序，轮间有 barrier。"""
    rng = random.Random(seed)
    per = req_tokens // page
    for r in range(rounds):
        order = list(range(clients))
        rng.shuffle(order)
        for c in order:
            yield r, c, [(c, p) for p in range((r + 1) * per)]


def run_multiturn(dev_pages, host_pages, policy, storage, on_request=None, **trace_kw):
    sim = TierSim(dev_pages, host_pages, policy, storage)
    per_round = {}
    for r, c, keys in multiturn_trace(**trace_kw):
        got = sim.request(keys)
        if on_request is not None:
            on_request(got)
        acc = per_round.setdefault(r, [0, 0, 0, 0])
        for i, v in enumerate(got):
            acc[i] += v
    return sim, per_round


def scenario_multiturn(model='llama3.1-8b', gpu='h100-sxm', tp=1, clients=80, rounds=10,
                       req_tokens=2048):
    m, card = MODELS[model], GPUS[gpu]
    dev_tokens, cell, wd = pool(model, gpu, tp)
    dev_pages = dev_tokens // PAGE
    host2 = (int(dev_tokens * 2.0) // PAGE + 1)
    host1 = (int(dev_tokens * 1.0) // PAGE + 1)
    total = clients * rounds * req_tokens
    print(f'{m.name} on {card.name}：KV 池 {dev_tokens:,} token = {dev_pages} 页（page {PAGE}）')
    print(f'负载：{clients} 个会话 × {rounds} 轮，每轮追加 {req_tokens} token；'
          f'最后一轮所有会话的上下文合计 {total:,} token，是显存池的 {total / dev_tokens:.1f} 倍')
    runs = OrderedDict([
        ('只有 GPU', (0, 'write_through', False)),
        ('GPU+主机 ratio1', (host1, 'write_through', False)),
        ('GPU+主机 wt', (host2, 'write_through', False)),
        ('GPU+主机 selective', (host2, 'write_through_selective', False)),
        ('GPU+主机 write_back', (host2, 'write_back', False)),
        ('GPU+主机+L3 wt', (host2, 'write_through', True)),
    ])
    l2_gbs, l3_gbs = LINKS['L2 PCIe Gen5 x16'], LINKS['L3 RDMA 400G/卡']
    res = OrderedDict()
    print(f'  {"配置":<20}{"显存命中":>9}{"主机命中":>9}{"L3命中":>8}{"重算":>8}{"D2H GB":>9}'
          f'{"丢弃页":>8}{"prefill s":>11}')
    for name, (hp, pol, sto) in runs.items():
        t = [0.0]

        def account(got):
            # prefill：新算的 token 接在命中前缀后面；搬运按串行计，逐层重叠能再省一点
            d, h, l3, miss = got
            t[0] += prefill_ms(m, card, tp, miss * PAGE, (d + h + l3) * PAGE, wd)
            t[0] += load_ms(h * PAGE, cell, l2_gbs) + load_ms(l3 * PAGE, cell, l3_gbs)

        sim, per_round = run_multiturn(dev_pages, hp, pol, sto, account, clients=clients,
                                       rounds=rounds, req_tokens=req_tokens)
        st = sim.st
        n = st['dev'] + st['host'] + st['l3'] + st['miss']
        t = t[0]
        res[name] = dict(st=dict(st), per_round=per_round, n=n, prefill_s=t / 1e3)
        print(f'  {name:<20}{st["dev"] / n:>9.1%}{st["host"] / n:>9.1%}{st["l3"] / n:>8.1%}'
              f'{st["miss"] / n:>8.1%}{st["d2h"] * PAGE * cell / 1e9:>9.0f}{st["dropped"]:>8}{t / 1e3:>11.1f}')
    print()
    print('逐轮命中率（命中 = 显存 + 主机 + L3，按页计）')
    print(f'  {"配置":<20}' + ''.join(f'{"R" + str(r + 1):>6}' for r in range(rounds)))
    for name, v in res.items():
        cells = []
        for r in range(rounds):
            d, h, l3, miss = v['per_round'][r]
            cells.append((d + h + l3) / (d + h + l3 + miss))
        v['round_hit'] = cells
        print(f'  {name:<20}' + ''.join(f'{x:>6.0%}' for x in cells))
    print()
    print('读法：轮询访问下，一个会话两次出现之间隔着其余所有会话这一轮的新 token。')
    print('  总上下文超过缓存容量以后，LRU 每次淘汰的恰好是下一个要用的会话，命中率一路往下掉。')
    print('  write_through 的主机池同时装着显存里那份的备份，有效容量约等于 max(显存, 主机)，')
    print('  所以 ratio 1 完全没用，ratio 2 也只多出一倍；write_back 只在页离开显存时才写主机，')
    print('  有效容量接近 显存 + 主机，代价是显存淘汰路径上要同步等 D2H 拷贝完成。')
    print('  selective 的二次命中门槛相当于一个准入过滤器，在超容量的循环访问里反而比纯 LRU 稳。')
    print('  只有 L3 的容量能装下全部会话，后几轮命中率继续上升。')
    return res


# ---------------------------------------------------------------- L3 预取停止策略

def prefetch_timeout_s(pages, page=PAGE, base=1.0, per_ki=0.25):
    return base + pages * (page / 1024 * per_ki)


def prefetch_plan(hit_tokens, cell, gbs, policy, queue_wait_s, page=PAGE, threshold=256,
                  base=1.0, per_ki=0.25):
    """L3 命中 hit_tokens 时，调度器什么时候放行这条请求、放行时拿到了多少 token。

    预取在请求入队时发起（scheduler.py:3441），调度器排到它时第一次问“好了没”（:4306）。
    completed_tokens 按 128 页一批推进，所以 best_effort 拿到的是整批。
    """
    key = hit_tokens // page * page
    if key < threshold:
        return dict(admit_s=queue_wait_s, fetched=0, reason='declined_too_short', timeout_s=None)
    pages = key // page
    batch_s = STORAGE_BATCH_PAGES * page * cell / (gbs * 1e9)
    full_s = key * cell / (gbs * 1e9)
    timeout = prefetch_timeout_s(pages, page, base, per_ki)

    def fetched_at(t):
        return min(key, int(t / batch_s) * STORAGE_BATCH_PAGES * page) if t < full_s else key

    if policy == 'best_effort' or full_s <= queue_wait_s:
        admit = queue_wait_s
    elif policy == 'wait_complete':
        admit = full_s
    elif policy == 'timeout':
        admit = min(full_s, queue_wait_s + timeout)   # 计时从第一次被拦下开始
    else:
        raise ValueError(policy)
    got = fetched_at(admit)
    return dict(admit_s=admit, fetched=got, reason='ok' if got == key else 'deadline',
                timeout_s=timeout)


def scenario_prefetch(model='deepseek-v3', gpu='h200', tp=8, hit=32768, new=2048):
    m, card = MODELS[model], GPUS[gpu]
    _, cell, wd = pool(model, gpu, tp)
    pages = hit // PAGE
    print(f'{m.name} {tp}×{card.name}：L3 命中 {hit} token（{pages} 页），新 token {new}，每 token {cell} B')
    print(f'默认 timeout = 1.0 + {pages} × {PAGE}/1024 × 0.25 = {prefetch_timeout_s(pages):.1f} s'
          '（设计文档写的 base 2、per_ki 0.1、上限 30 s 是旧值）')
    recompute = prefill_ms(m, card, tp, hit + new, 0, wd) / 1e3
    print(f'全部重算：{recompute * 1e3:.0f} ms')
    res = {}
    for gbs in (40.0, 5.37, 0.8, 0.1):
        print(f'  每 rank {gbs} GB/s，整段拉完 {hit * cell / gbs / 1e9 * 1e3:.0f} ms')
        print(f'    {"排队":>8}{"策略":>15}{"放行 ms":>10}{"拿到 token":>12}{"TTFT ms":>10}')
        for wait in (0.0, 0.2, 1.0):
            for pol in ('best_effort', 'timeout', 'wait_complete'):
                p = prefetch_plan(hit, cell, gbs, pol, wait)
                rest = hit - p['fetched']
                ttft = p['admit_s'] + prefill_ms(m, card, tp, rest + new, p['fetched'], wd) / 1e3
                res[(gbs, wait, pol)] = dict(p, ttft=ttft)
                print(f'    {wait * 1e3:>6.0f}ms{pol:>15}{p["admit_s"] * 1e3:>10.0f}{p["fetched"]:>12}'
                      f'{ttft * 1e3:>10.0f}')
    re_us = prefill_ms(m, card, tp, 4096, 0, wd) / 4096 * 1e3
    per_ki = 1024 * re_us / 1e6
    print()
    print('读法：带宽够的时候三种策略几乎一样，排队时间已经把预取盖住了。')
    print('  带宽变差时 best_effort 拿到的常常是 0 或一两批（128 页一批），等于白拉；')
    print('  timeout 的默认预算按每 1K token 给 0.25 s，远超重算的代价，慢存储上会把 TTFT 拖到几秒；')
    print('  wait_complete 只适合带宽有保障的 RDMA / 3FS 集群。')
    print(f'建议：每页等待的预算不要超过重算这一页的时间。本例重算 {re_us:.1f} us/token，')
    print(f'  对应 prefetch_timeout_per_ki_token ≈ {per_ki:.4f}；base 取一个调度周期的量级（0.05~0.2）。')
    res['suggest_per_ki'] = per_ki
    res['recompute_s'] = recompute
    return res


# ---------------------------------------------------------------- 页哈希

def namespace_seed(extra_key=None, cache_salt=None):
    """utils.py:125 storage_namespace_seed。"""
    if extra_key is None and cache_salt is None:
        return None
    h = hashlib.sha256(b'sglang-cache-namespace-v1')
    for part in (extra_key, cache_salt):
        if part is None:
            h.update(b'\x00')
            continue
        b = part.encode('utf-8')
        h.update(b'\x01' + len(b).to_bytes(8, 'little') + b)
    return h.hexdigest()


def page_hashes(tokens, page=PAGE, prior_hex=None, extra_key=None, cache_salt=None):
    """hash_binding.cpp：每页 SHA256(上一页 32 字节摘要 + token 的 uint32 小端)，最后不满一页按实际长度。"""
    if prior_hex is None:
        prior_hex = namespace_seed(extra_key, cache_salt)
    prior = bytes.fromhex(prior_hex) if prior_hex else None
    out = []
    for s in range(0, len(tokens), page):
        chunk = tokens[s:s + page]
        h = hashlib.sha256()
        if prior is not None:
            h.update(prior)
        h.update(struct.pack(f'<{len(chunk)}I', *chunk))
        prior = h.digest()
        out.append(h.hexdigest())
    return out


def file_key(page_hash, model_name, tp_rank=0, tp_size=1, is_mla=False, pp=None):
    """hicache_storage.py:728 HiCacheFile 的文件名。MLA 各 rank 内容一样，不带 tp 后缀，只有 rank 0 写。"""
    model = '-'.join(model_name.split('/'))
    shard = f'_{pp[1]}_{pp[0]}' if pp else ''
    suffix = f'_{model}{shard}' if is_mla else f'_{model}_{tp_rank}_{tp_size}{shard}'
    return f'{page_hash}{suffix}.bin'


def scenario_hash(page=4):
    sys_prompt = list(range(1000, 1010))            # 10 个 token 的系统提示
    a = sys_prompt + [7, 8]
    b = sys_prompt + [9, 9]
    ha, hb = page_hashes(a, page), page_hashes(b, page)
    hs = page_hashes(a, page, cache_salt='tenant-42')
    print(f'page={page}，请求 A、B 共享前 10 个 token，第 11 个开始不同')
    for i in range(max(len(ha), len(hb))):
        same = '相同' if i < len(ha) and i < len(hb) and ha[i] == hb[i] else '不同'
        print(f'  第 {i} 页  A {ha[i][:12]}  B {hb[i][:12]}  {same}  带 cache_salt 的 A {hs[i][:12]}')
    print('  第 2 页覆盖 token 8..11，A、B 在这一页里分叉，于是从这一页起整条链都不同；')
    print('  带 cache_salt 的请求种子不同，第 0 页就和别人不同，L3 里天然隔离。')
    print()
    print('文件后端的键名（默认目录 /tmp/hicache）：')
    print('  MHA tp=2 rank1 ', file_key(ha[0], 'meta-llama/Llama-3.1-8B', 1, 2))
    print('  MLA tp=8      ', file_key(ha[0], 'deepseek-ai/DeepSeek-V3', 3, 8, is_mla=True))
    print('  换了 tp_size，MHA 的键全变：L3 里存的 KV 是按 TP 切过的，换并行度不能复用。')
    return dict(a=ha, b=hb, salted=hs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='cost', choices=['cost', 'multiturn', 'prefetch', 'hash'])
    ap.add_argument('--model', default=None)
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--tp', type=int, default=None)
    ap.add_argument('--clients', type=int, default=80)
    ap.add_argument('--rounds', type=int, default=10)
    a = ap.parse_args()
    if a.scenario == 'cost':
        scenario_cost()
    elif a.scenario == 'multiturn':
        scenario_multiturn(a.model or 'llama3.1-8b', a.gpu or 'h100-sxm', a.tp or 1,
                           clients=a.clients, rounds=a.rounds)
    elif a.scenario == 'prefetch':
        scenario_prefetch(a.model or 'deepseek-v3', a.gpu or 'h200', a.tp or 8)
    else:
        scenario_hash()


if __name__ == '__main__':
    main()
