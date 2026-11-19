"""EPLB：冗余专家怎么复制、怎么摆到 GPU 上，以及摆完之后负载到底平了多少。

只用标准库。算法逐行移植自 sglang 8fabf5e 的 eplb/eplb_algorithms/deepseek.py
（那份文件又是从 deepseek-ai/EPLB 的 eplb.py 原样拷过来的），torch 张量换成 list。

  python3 eplb_sim.py                       # DeepSeek-V3 在 EP32 / EP72 上：不做 EPLB 和做 EPLB 的均衡度
  python3 eplb_sim.py --scenario example    # 复现 deepseek-ai/EPLB README 的例子，核对移植没走样
  python3 eplb_sim.py --scenario shift      # 统计窗口和实际负载对不上时，均衡度掉多少
  python3 eplb_sim.py --scenario locality   # 分层策略 vs 全局策略：一个 token 要跨几个节点
  python3 eplb_sim.py --scenario dispatch   # 一个逻辑专家有多个副本时，token 发给哪一个

口径：
  均衡度 = 每层各 GPU 收到的 token 数的 平均 / 最大，再对层取平均
  （eplb/expert_distribution.py:1159 compute_utilization_rate，带 1e-5 平滑）。
  1.0 表示完全均衡；MoE 层的耗时由最忙那张卡决定，所以 1 / 均衡度 ≈ 被拖慢的倍数。
  专家负载是合成的：专家热度 × 组热度，各取对数正态，参数在 make_load 里，
  不代表任何真实流量，只用来比较策略之间的相对好坏。
"""
import argparse
import math
import random
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------- 算法移植


def balanced_packing(weight: Sequence[float], num_packs: int) -> Tuple[List[int], List[int]]:
    """n 个带权物品装进 m 个箱子，每箱恰好 n/m 个，按重量降序贪心放进当前最轻、且没装满的箱子。"""
    n = len(weight)
    assert n % num_packs == 0
    per_pack = n // num_packs
    if per_pack == 1:
        return list(range(n)), [0] * n
    order = sorted(range(n), key=lambda i: -weight[i])           # torch sort descending
    pack_w = [0.0] * num_packs
    pack_n = [0] * num_packs
    pack_index = [-1] * n
    rank_in_pack = [-1] * n
    for item in order:
        pack = min((j for j in range(num_packs) if pack_n[j] < per_pack), key=pack_w.__getitem__)
        pack_index[item] = pack
        rank_in_pack[item] = pack_n[pack]
        pack_w[pack] += weight[item]
        pack_n[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: Sequence[float], num_phy: int) -> Tuple[List[int], List[int], List[int]]:
    """每多一个物理槽位，就给「当前单副本负载」最大的逻辑专家再加一个副本。"""
    num_log = len(weight)
    assert num_phy >= num_log
    phy2log = list(range(num_log)) + [0] * (num_phy - num_log)
    rank = [0] * num_phy
    logcnt = [1] * num_log
    for i in range(num_log, num_phy):
        j = max(range(num_log), key=lambda e: weight[e] / logcnt[e])   # 并列取下标最小
        phy2log[i] = j
        rank[i] = logcnt[j]
        logcnt[j] += 1
    return phy2log, rank, logcnt


def _inverse(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def rebalance_hierarchical(weight: Sequence[float], num_phy: int, num_groups: int, num_nodes: int,
                           num_gpus: int) -> Tuple[List[int], List[int]]:
    """单层版本的 rebalance_experts_hierarchical，返回 (phy2log, logcnt)。

    物理槽位 k 在 GPU k // (num_phy / num_gpus) 上。三步：
      1. 专家组按组内 token 总数装箱到节点（组不拆开，配合 DeepSeek-V3 的组限制路由）
      2. 每个节点内部按 replicate_experts 加副本
      3. 节点内的物理专家按「负载 / 副本数」装箱到 GPU
    num_groups=1、num_nodes=1 时就是全局策略。
    """
    E = len(weight)
    assert E % num_groups == 0 and num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0 and num_phy % num_gpus == 0
    gsize = E // num_groups
    groups_per_node = num_groups // num_nodes
    per_gpu = num_phy // num_gpus
    e_per_node = E // num_nodes
    phy_per_node = num_phy // num_nodes

    tok_group = [sum(weight[g * gsize:(g + 1) * gsize]) for g in range(num_groups)]
    gpack, grank = balanced_packing(tok_group, num_nodes)
    log2mlog = [(gpack[e // gsize] * groups_per_node + grank[e // gsize]) * gsize + e % gsize
                for e in range(E)]
    mlog2log = _inverse(log2mlog)

    pphy2mlog_all = []
    mlogcnt_all = []
    for n in range(num_nodes):
        tok_mlog = [weight[mlog2log[n * e_per_node + j]] for j in range(e_per_node)]
        phy2mlog, _, mlogcnt = replicate_experts(tok_mlog, phy_per_node)
        tok_phy = [tok_mlog[m] / mlogcnt[m] for m in phy2mlog]
        pack, rip = balanced_packing(tok_phy, num_gpus // num_nodes)
        phy2pphy = [pack[i] * per_gpu + rip[i] for i in range(phy_per_node)]
        pphy2phy = _inverse(phy2pphy)
        pphy2mlog_all += [phy2mlog[pphy2phy[k]] + n * e_per_node for k in range(phy_per_node)]
        mlogcnt_all += mlogcnt
    phy2log = [mlog2log[m] for m in pphy2mlog_all]
    logcnt = [mlogcnt_all[log2mlog[e]] for e in range(E)]
    return phy2log, logcnt


def rebalance(weight, num_phy, num_groups, num_nodes, num_gpus, algorithm='auto'):
    """eplb_algorithms/__init__.py compute_algorithm：组数能被节点数整除走分层，否则全局。"""
    if algorithm == 'auto':
        algorithm = 'hierarchical' if num_groups and num_groups % num_nodes == 0 else 'global'
    if algorithm == 'hierarchical':
        return rebalance_hierarchical(weight, num_phy, num_groups, num_nodes, num_gpus) + (algorithm,)
    return rebalance_hierarchical(weight, num_phy, 1, 1, num_gpus) + (algorithm,)


def trivial_map(num_log: int, num_phy: int) -> List[int]:
    """expert_location.py:108 init_trivial：物理槽位 i 放逻辑专家 i % num_log，冗余槽位依次放 0、1、2…"""
    return [i % num_log for i in range(num_phy)]


# ---------------------------------------------------------------- 负载与指标

def make_load(E=256, groups=8, layers=8, sigma_e=0.9, sigma_g=0.35, seed=0):
    """合成每层每个逻辑专家的相对热度。专家热度和组热度都取对数正态。"""
    r = random.Random(seed)
    out = []
    gsize = E // groups
    for _ in range(layers):
        gw = [math.exp(r.gauss(0, sigma_g)) for _ in range(groups)]
        w = [gw[e // gsize] * math.exp(r.gauss(0, sigma_e)) for e in range(E)]
        s = sum(w)
        out.append([x / s for x in w])
    return out


def sample_counts(p: Sequence[float], n: int, r: random.Random) -> List[float]:
    """n 次路由落到各专家的次数，用正态近似多项分布。"""
    return [max(0.0, n * x + math.sqrt(n * x * (1 - x)) * r.gauss(0, 1)) for x in p]


def perturb(p: Sequence[float], sigma: float, r: random.Random) -> List[float]:
    q = [x * math.exp(r.gauss(0, sigma)) for x in p]
    s = sum(q)
    return [x / s for x in q]


def gpu_load(count: Sequence[float], phy2log: Sequence[int], num_gpus: int) -> List[float]:
    """每个逻辑专家的 token 在它的副本之间平分（dynamic 分派、或静态表足够分散时的情形）。"""
    cnt = [0] * len(count)
    for e in phy2log:
        cnt[e] += 1
    per_gpu = len(phy2log) // num_gpus
    loads = [0.0] * num_gpus
    for k, e in enumerate(phy2log):
        loads[k // per_gpu] += count[e] / cnt[e]
    return loads


def balancedness(loads: Sequence[float]) -> float:
    return (sum(loads) / len(loads) + 1e-5) / (max(loads) + 1e-5)


def mean_balancedness(counts_per_layer, maps_per_layer, num_gpus) -> float:
    vals = [balancedness(gpu_load(c, m, num_gpus)) for c, m in zip(counts_per_layer, maps_per_layer)]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- 场景

def scenario_example():
    w = [[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
         [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]]
    expect = [[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
              [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]]
    print('deepseek-ai/EPLB README：12 个专家、4 组、4 个冗余，16 个副本摆到 2 节点 × 4 GPU')
    for layer, (row, exp) in enumerate(zip(w, expect)):
        phy2log, logcnt, algo = rebalance(row, 16, 4, 2, 8)
        ok = '一致' if phy2log == exp else '不一致'
        print(f'  第 {layer} 层（{algo}）phy2log = {phy2log}  与 README {ok}')
        loads = gpu_load(row, phy2log, 8)
        print(f'      每 GPU 负载 {[round(x) for x in loads]}  均衡度 {balancedness(loads):.3f}'
              f'（不加副本、按编号摆：{balancedness(gpu_load(row + [0] * 4, trivial_map(16, 16), 8)):.3f}）')


def _plans(p_layers, E, num_phy_red, groups, nodes, gpus):
    plans = {}
    plans['不做 EPLB，无冗余'] = [trivial_map(E, E)] * len(p_layers)
    plans['不做 EPLB，32 冗余按编号'] = [trivial_map(E, num_phy_red)] * len(p_layers)
    algo0 = algo1 = ''
    if E % gpus == 0:
        rows = [rebalance(p, E, groups, nodes, gpus) for p in p_layers]
        algo0 = rows[0][2]
        plans[f'EPLB 无冗余（{algo0}）'] = [x[0] for x in rows]
    rows = [rebalance(p, num_phy_red, groups, nodes, gpus) for p in p_layers]
    algo1 = rows[0][2]
    plans[f'EPLB 32 冗余（{algo1}）'] = [x[0] for x in rows]
    return plans


def scenario_balance(layers=8, tok_rank=128, topk=8):
    E, groups = 256, 8
    p_layers = make_load(E, groups, layers, seed=1)
    print('DeepSeek-V3：256 个路由专家、8 组、top-8；合成负载，每层独立')
    print(f'  「统计」= 用来做规划的长窗口平均；「新批次」= 每卡 {tok_rank} token 的一次 decode 实采\n')
    for ep, gpn in ((32, 8), (72, 8)):
        nodes = ep // gpn
        r = random.Random(ep)
        fresh = [sample_counts(p, ep * tok_rank * topk, r) for p in p_layers]
        stat = [[x * 1e6 for x in p] for p in p_layers]
        plans = _plans(stat, E, E + 32, groups, nodes, ep)
        print(f'EP{ep}（{nodes} 节点 × {gpn} 卡），每卡 {(E + 32) // ep if (E + 32) % ep == 0 else "-"} 个物理专家（含冗余）')
        print(f'  {"方案":<30}{"统计上":>8}{"新批次":>8}{"最忙卡 / 平均":>14}')
        for name, maps in plans.items():
            if E % ep and '无冗余' in name:
                continue
            b_stat = mean_balancedness(stat, maps, ep)
            b_new = mean_balancedness(fresh, maps, ep)
            print(f'  {name:<30}{b_stat:8.3f}{b_new:8.3f}{1 / b_new:13.2f}x')
        if E % ep:
            print(f'  （256 不能被 {ep} 整除，不加冗余时 EP{ep} 根本起不来，见 sim_parallel.py --scenario layout）')
        print()
    print('  读法：最忙卡 / 平均 就是 MoE 层因为负载不均被拖慢的倍数。')
    print('  冗余槽位按编号摆（init_trivial 的默认）几乎没用，副本给了 0~31 号专家，和热度无关。')


def scenario_shift(layers=8):
    E, groups, ep, nodes = 256, 8, 32, 4
    base = make_load(E, groups, layers, seed=1)
    r = random.Random(7)
    plan = [rebalance([x * 1e6 for x in p], E + 32, groups, nodes, ep)[0] for p in base]
    print('EP32 + 32 冗余，按「统计窗口」的负载规划；实际负载在窗口之后漂移')
    print('  漂移 = 每个专家热度乘 exp(N(0, σ))，σ 越大越不像规划时的样子\n')
    print(f'  {"σ":>6}{"按旧统计的方案":>14}{"按新负载重排":>14}{"不做 EPLB":>12}')
    for sigma in (0.0, 0.1, 0.3, 0.6, 1.0):
        new = [perturb(p, sigma, r) for p in base]
        cnt = [[x * 1e6 for x in p] for p in new]
        fresh_plan = [rebalance(c, E + 32, groups, nodes, ep)[0] for c in cnt]
        b_old = mean_balancedness(cnt, plan, ep)
        b_new = mean_balancedness(cnt, fresh_plan, ep)
        b_triv = mean_balancedness(cnt, [trivial_map(E, E + 32)] * layers, ep)
        print(f'  {sigma:6.1f}{b_old:14.3f}{b_new:14.3f}{b_triv:12.3f}')
    print('\n  漂移大到一定程度，旧方案会退回到不做 EPLB 的水平。sglang 的对策：')
    print('    --enable-eplb 按 --eplb-rebalance-num-iterations（默认 1000 次 forward）周期重排；')
    print('    --eplb-min-rebalancing-utilization-threshold 低于阈值才重排（默认 1.0 即总是重排）；')
    print('    prefill 和 decode 实例各自统计、各自规划（PD 分离时天然分开）。')


def _route_token(p, groups, topk_group, topk, r):
    """组限制路由的近似：按组热度无放回抽 topk_group 个组，再在这些组里按专家热度无放回抽 topk 个。"""
    E = len(p)
    gsize = E // groups
    gw = [sum(p[g * gsize:(g + 1) * gsize]) for g in range(groups)]
    chosen_g = []
    for _ in range(topk_group):
        tot = sum(gw[g] for g in range(groups) if g not in chosen_g)
        x = r.random() * tot
        for g in range(groups):
            if g in chosen_g:
                continue
            x -= gw[g]
            if x <= 0:
                chosen_g.append(g)
                break
        else:
            chosen_g.append(next(g for g in range(groups) if g not in chosen_g))
    cand = [e for g in chosen_g for e in range(g * gsize, (g + 1) * gsize)]
    picked = []
    for _ in range(topk):
        tot = sum(p[e] for e in cand if e not in picked)
        x = r.random() * tot
        for e in cand:
            if e in picked:
                continue
            x -= p[e]
            if x <= 0:
                picked.append(e)
                break
        else:
            picked.append(next(e for e in cand if e not in picked))
    return picked


def scenario_locality(n_tokens=2000):
    E, groups, ep, gpn = 256, 8, 32, 8
    nodes = ep // gpn
    p = make_load(E, groups, 1, seed=3)[0]
    stat = [x * 1e6 for x in p]
    num_phy = E + 32
    per_gpu = num_phy // ep
    maps = {
        '按编号摆（trivial）': trivial_map(E, num_phy),
        '分层策略（hierarchical）': rebalance(stat, num_phy, groups, nodes, ep, 'hierarchical')[0],
        '全局策略（global）': rebalance(stat, num_phy, groups, nodes, ep, 'global')[0],
    }
    r = random.Random(11)
    toks = [_route_token(p, groups, 4, 8, r) for _ in range(n_tokens)]
    print(f'EP32 = {nodes} 节点 × {gpn} 卡，DeepSeek-V3 组限制路由：先选 4 个组，再在组内选 8 个专家')
    print(f'  统计 {n_tokens} 个 token，每个 token 的 8 个专家（每个专家取离它最近的副本）分布在几个节点上\n')
    print(f'  {"放置":<26}{"平均节点数":>10}{"跨节点副本占比":>16}{"均衡度":>8}')
    for name, m in maps.items():
        node_of = {}
        for k, e in enumerate(m):
            node_of.setdefault(e, set()).add(k // per_gpu // gpn)
        touched = 0
        for t in toks:
            src = r.randrange(nodes)       # token 所在节点
            ns = set()
            for e in t:
                ns.add(src if src in node_of[e] else min(node_of[e]))
            touched += len(ns)
        spread = sum(1 for e, s in node_of.items() if len(s) > 1) / E
        b = balancedness(gpu_load(stat, m, ep))
        print(f'  {name:<26}{touched / n_tokens:10.2f}{spread:16.1%}{b:8.3f}')
    print('\n  分层策略把一个组整体留在一个节点里，token 最多碰 4 个组 → 最多 4 个节点；')
    print('  全局策略不管组，副本可能落到任意节点，均衡度最好，但跨节点流量变多。')
    print('  所以 DeepSeek 在 prefill（EP32，节点数整除组数）用分层，decode（EP144）用全局；')
    print('  sglang 的 auto 也是这个规则：8 组在 4 节点上走分层，在 9 节点（EP72）上只能走全局。')


def scenario_dispatch():
    """compute_logical_to_rank_dispatch_physical_map 的简化：每个源 rank 为每个逻辑专家固定挑一个副本。"""
    E, ep, gpn = 256, 32, 8
    num_phy = E + 32
    per_gpu = num_phy // ep
    p = make_load(E, 8, 1, seed=5)[0]
    stat = [x * 1e6 for x in p]
    m = rebalance(stat, num_phy, 8, ep // gpn, ep)[0]
    replicas = {}
    for k, e in enumerate(m):
        replicas.setdefault(e, []).append(k)
    r = random.Random(42)
    # static：同卡优先，其次同节点（仅当能缩小候选集），剩下的 rank 用 _fair_choices 均分
    table = [[-1] * E for _ in range(ep)]
    for e, cands in replicas.items():
        rest = []
        for rank in range(ep):
            same_gpu = [k for k in cands if k // per_gpu == rank]
            same_node = [k for k in cands if k // per_gpu // gpn == rank // gpn]
            if len(cands) == 1:
                table[rank][e] = cands[0]
            elif same_gpu:
                table[rank][e] = same_gpu[0]
            elif 0 < len(same_node) < len(cands):
                table[rank][e] = same_node[0]
            else:
                rest.append(rank)
        if rest:
            q, rem = divmod(len(rest), len(cands))
            choice = cands * q + r.sample(cands, rem)
            r.shuffle(choice)
            for rank, c in zip(rest, choice):
                table[rank][e] = c
    # 每个源 rank 发出的 token 量相同、专家分布相同
    load_static = [0.0] * ep
    for rank in range(ep):
        for e in range(E):
            load_static[table[rank][e] // per_gpu] += stat[e] / ep
    load_dyn = gpu_load(stat, m, ep)
    multi = sum(1 for c in replicas.values() if len(c) > 1)
    print(f'EP32 + 32 冗余：{multi} 个逻辑专家有多个副本')
    print('  static（默认）：每个源 rank 为每个逻辑专家查一张固定表，表在加载时按同卡 > 同节点 > 均分生成')
    print('  dynamic：每个 token 在副本之间轮转（rank_invariant 时按行号取模，否则随机）\n')
    print(f'  {"分派":<12}{"均衡度":>8}{"最忙卡 / 平均":>14}')
    for name, ld in (('static', load_static), ('dynamic', load_dyn)):
        b = balancedness(ld)
        print(f'  {name:<12}{b:8.3f}{1 / b:13.2f}x')
    print('\n  static 下同一节点内的源 rank 会挤到同一个本地副本上，副本之间不一定平分；')
    print('  EPLB 规划时假设副本平分负载，所以 static 的实际均衡度会比规划值略低。')
    print('  默认值（parallel_hook.py:750）：有 a2a 后端时 static；没有 a2a 时只能 dynamic，')
    print('  因为那时所有 EP rank 处理同一批 token 再求和，各 rank 必须挑同一个副本，')
    print('  static / lp 每个 rank 挑法不同，会直接报错。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='balance',
                    choices=['balance', 'example', 'shift', 'locality', 'dispatch'])
    a = ap.parse_args()
    globals()['scenario_' + a.scenario]()


if __name__ == '__main__':
    main()
