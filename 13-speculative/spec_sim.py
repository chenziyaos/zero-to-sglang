"""投机解码的数学和 SGLang 里的实现细节，纯 Python 复现。

只用标准库，CPU 上能跑。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/：
  srt/speculative/spec_utils.py:417-485      select_top_k_tokens：EAGLE-2 动态树每一层怎么挑要展开的节点
  srt/speculative/eagle_utils.py:103         organize_draft_results：从全部候选里挑 num_draft_tokens-1 个
  kernels/ops/speculative/spec_tree.py:53    建树：父节点定位、next_token / next_sibling、tree mask、positions
  kernels/ops/speculative/spec_tree.py:211   verify_tree_greedy：贪心树验证
  kernels/jit/include/sgl_kernel/speculative/sampling.cuh
                                             TreeSpeculativeSamplingTargetOnly：默认的采样树验证
  kernels/ops/speculative/reject_sampling.py 链式拒绝采样，以及 block verification（arXiv 2403.10444）
  srt/managers/scheduler_components/metrics_reporter.py:973   日志里的 accept len / accept rate
  srt/managers/tokenizer_manager.py:3164     每个请求 meta_info 里的 spec_accept_rate / spec_accept_length
  srt/speculative/ngram_worker.py、kernels/ops/speculative/ngram_corpus.py   NGRAM 的后缀匹配草稿

  python3 spec_sim.py                      # theory：接受率 α、草稿长度 k、草稿成本 c 和加速比
  python3 spec_sim.py --scenario sampling  # 四种验证规则的输出分布是否等于目标分布，接受率差多少
  python3 spec_sim.py --scenario tree      # EAGLE-2 动态树：建树、贪心验证、accept len 和 accept rate
  python3 spec_sim.py --scenario batch     # batch 和上下文变化时，投机解码从哪里开始变慢
  python3 spec_sim.py --scenario ngram     # 不要草稿模型的 NGRAM：抄写型输出和全新输出差多少
"""
import argparse
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from common.specs import GPUS, MODELS  # noqa: E402

# ---------------------------------------------------------------------------
# theory：Leviathan 2023 的期望 token 数和加速比
# ---------------------------------------------------------------------------


def expected_tokens(alpha, k):
    """每轮验证期望产出的 token 数（含目标模型最后补的那一个）。

    假设每个草稿 token 独立地以概率 alpha 被接受，第一次拒绝就停：
    E = 1 + α + α² + ... + α^k = (1 - α^(k+1)) / (1 - α)。
    """
    if alpha >= 1.0:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha, k, c):
    """草稿模型一次前向的耗时是目标模型的 c 倍，验证一次按目标模型一次前向算。

    这是 Leviathan 论文的 improvement factor，前提是验证 k+1 个 token 和 decode 1 个 token
    一样贵，也就是还在访存瓶颈区。batch 变大以后这个前提不成立，见 scenario_batch。
    """
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha, c, kmax=16):
    return max(range(1, kmax + 1), key=lambda k: speedup(alpha, k, c))


def alpha_from_accept_len(accept_len, k):
    """由线上看到的 accept len（每轮产出 token 数）和草稿长度 k 反推单 token 接受率。"""
    if accept_len >= k + 1:
        return 1.0
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if expected_tokens(mid, k) < accept_len:
            lo = mid
        else:
            hi = mid
    return lo


def sglang_accept_rate(accept_len, num_draft_tokens):
    """metrics_reporter 的口径：接受的草稿数 / (验证轮数 x (num_draft_tokens - 1))。"""
    return (accept_len - 1) / (num_draft_tokens - 1)


def scenario_theory():
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    ks = [1, 2, 3, 4, 5, 6, 8]
    print('每轮期望产出 token 数 E = (1-α^(k+1))/(1-α)（即 SGLang 日志里的 accept len）')
    print('  α \\ k ' + ''.join(f'{k:>7}' for k in ks))
    table = {}
    for a in alphas:
        row = [expected_tokens(a, k) for k in ks]
        table[a] = row
        print(f'  {a:<6}' + ''.join(f'{v:>7.2f}' for v in row))

    print('\n加速比 E/(k·c+1)，c=0.05（草稿一次前向约等于目标的 1/20，EAGLE 头的量级）')
    print('  α \\ k ' + ''.join(f'{k:>7}' for k in ks) + '   最优 k')
    sp = {}
    for a in alphas:
        row = [speedup(a, k, 0.05) for k in ks]
        sp[a] = row
        print(f'  {a:<6}' + ''.join(f'{v:>7.2f}' for v in row) + f'   {best_k(a, 0.05):>5}')

    print('\n最优 k 随草稿成本 c 的变化（c 越大，越不该猜长）')
    cs = [0.02, 0.05, 0.1, 0.2, 0.4]
    print('  α \\ c ' + ''.join(f'{c:>7}' for c in cs))
    bk = {}
    for a in alphas:
        row = [best_k(a, c) for c in cs]
        bk[a] = row
        print(f'  {a:<6}' + ''.join(f'{v:>7}' for v in row))

    print('\n从线上指标反推 α：accept len 是每轮产出 token 数，k = num_draft_tokens - 1（topk=1 链式）')
    obs = [(2.4, 3), (2.8, 3), (3.2, 5), (3.6, 7)]
    back = []
    print('  accept len  draft_tokens   反推 α   SGLang accept rate')
    for L, k in obs:
        a = alpha_from_accept_len(L, k)
        rate = sglang_accept_rate(L, k + 1)
        back.append((L, k, a, rate))
        print(f'  {L:>10}  {k + 1:>12}   {a:>6.3f}   {rate:>10.3f}')
    print('  accept rate 是“提出的草稿里有多少被接受”，它总是小于 α：草稿越长，排在后面的越难轮到。')
    return {'table': table, 'speedup': sp, 'best_k': bk, 'back': back}


# ---------------------------------------------------------------------------
# sampling：四种验证规则，Monte-Carlo 检查输出分布
# ---------------------------------------------------------------------------

def sample(dist, u):
    """逆 CDF 采样；dist 不必归一化。"""
    total = sum(dist)
    t = u * total
    acc = 0.0
    last = len(dist) - 1
    for i, x in enumerate(dist):
        if x <= 0:
            continue
        acc += x
        last = i
        if acc > t:
            return i
    return last


def chain_rejection_round(p, q, k, rng):
    """reject_sampling.py 的 token-wise 分支：草稿按 q 采样，coin·q < p 就接受（概率 min(1, p/q)）。

    第一次拒绝就停，从 max(p - q, 0) 归一化后补一个；全部接受则从 p 补一个。
    """
    out = []
    for _ in range(k):
        x = sample(q, rng.random())
        if 0 < q[x] <= 1 and rng.random() * q[x] < p[x]:
            out.append(x)
            continue
        out.append(sample([max(pi - qi, 0.0) for pi, qi in zip(p, q)], rng.random()))
        return out, len(out) - 1
    out.append(sample(p, rng.random()))
    return out, k


def block_verify_round(p, q, k, rng):
    """reject_sampling.py 的 BLOCK_VERIFICATION 分支（Sun 等, 2403.10444 Algorithm 2）。

    prefix_i = min(prefix_{i-1} · p(x_i)/q(x_i), 1)，从最长前缀往短扫，
    第一个通过的前缀整段接受，即使更早的某个 token 单看会被拒绝。
    """
    xs = [sample(q, rng.random()) for _ in range(k)]
    coins = [rng.random() for _ in range(k)]
    prefix, prefixes = 1.0, []
    for x in xs:
        prefix = min(prefix * p[x] / q[x], 1.0) if 0 < q[x] <= 1 else 0.0
        prefixes.append(prefix)
    num_accept, scale, norm = 0, 1.0, -1.0
    for step in range(k, 0, -1):
        pp = prefixes[step - 1]
        if coins[step - 1] < pp:
            accept_prob, residual = pp, -1.0
            if step < k and 0.0 < pp < 1.0:
                residual = sum(max(pp * pi - qi, 0.0) for pi, qi in zip(p, q))
                accept_prob = residual / (residual + 1.0 - pp)
            if coins[step - 1] < accept_prob:
                num_accept, scale, norm = step, pp, residual
                break
    out = xs[:num_accept]
    if num_accept == k:
        out.append(sample(p, rng.random()))
    else:
        out.append(sample([max(scale * pi - qi, 0.0) for pi, qi in zip(p, q)], rng.random()))
    return out, num_accept


def topk_tokens(dist, k):
    return sorted(range(len(dist)), key=lambda i: (-dist[i], i))[:k]


def tree_target_only_round(p, q, depth, width, rng, th_single=1.0, th_acc=1.0):
    """sampling.cuh 的 TreeSpeculativeSamplingTargetOnly，配一棵满树：每个节点的孩子是 q 的 top-width。

    这是 EAGLE 在 temperature > 0、没开 --speculative-use-rejection-sampling 时走的路径。
    草稿是按 top-k 挑的，不是按 q 采样的，核函数也只看目标概率 p：
      同一层的兄弟依次累加 prob_acc += p(x)，coin < prob_acc / th_acc 或 p(x) >= th_single 就接受；
      被拒的兄弟把 p(x) 记进 draft_probs，最后从 relu(p - draft_probs) 补一个。
    阈值都为 1 时，这等价于“按兄弟顺序对 p 做逆 CDF 采样”，输出分布严格等于 p。
    """
    out = []
    sib = topk_tokens(q, width)
    for _level in range(depth):
        coin = rng.random()
        prob_acc = 0.0
        removed = [0.0] * len(p)
        hit = None
        for x in sib:
            prob_acc += p[x]
            if p[x] > 0 and (coin < prob_acc / max(th_acc, 1e-9) or p[x] >= th_single):
                hit = x
                break
            removed[x] = p[x]
        if hit is None:
            out.append(sample([max(pi - ri, 0.0) for pi, ri in zip(p, removed)], rng.random()))
            return out, len(out) - 1
        out.append(hit)
        sib = topk_tokens(q, width)        # 上下文无关的玩具分布：每层的孩子都一样
    out.append(sample(p, rng.random()))
    return out, depth


def stream_stats(round_fn, p, n_tokens, rng):
    """反复调用一轮验证，把输出拼成一条 token 流，统计一元和二元分布。

    目标分布与上下文无关时，正确的验证规则产出的流必须是 i.i.d. p。
    """
    V = len(p)
    uni = [0] * V
    bi = [[0] * V for _ in range(V)]
    prev = None
    n = rounds = accepted = 0
    while n < n_tokens:
        toks, acc = round_fn(rng)
        rounds += 1
        accepted += acc
        for t in toks:
            uni[t] += 1
            if prev is not None:
                bi[prev][t] += 1
            prev = t
            n += 1
    tv_uni = 0.5 * sum(abs(uni[i] / n - p[i]) for i in range(V))
    nb = n - 1
    tv_bi = 0.5 * sum(abs(bi[i][j] / nb - p[i] * p[j]) for i in range(V) for j in range(V))
    return {'tv_uni': tv_uni, 'tv_bi': tv_bi, 'accept_len': n / rounds,
            'correct_per_round': accepted / rounds}


def scenario_sampling(n_tokens=120000, seed=0):
    rng = random.Random(seed)
    p = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]      # 目标模型在 temperature 下的分布
    q = [0.20, 0.40, 0.10, 0.05, 0.15, 0.10]      # 草稿模型的分布，和 p 有明显偏差
    alpha = sum(min(a, b) for a, b in zip(p, q))
    print(f'词表 6，p = {p}')
    print(f'          q = {q}')
    print(f'单 token 接受率：rejection sampling 是 Σmin(p,q) = {alpha:.2f}；'
          f'target-only 只拿 q 的 top-1（token 1），接受率是 p[1] = {p[1]:.2f}')
    print('\n规则                                        一元 TV   二元 TV   每轮产出   每轮接受草稿')
    cases = [
        ('链式 rejection sampling, k=3', lambda r: chain_rejection_round(p, q, 3, r)),
        ('block verification, k=3', lambda r: block_verify_round(p, q, 3, r)),
        ('target-only 树, 深 3 宽 1', lambda r: tree_target_only_round(p, q, 3, 1, r)),
        ('target-only 树, 深 3 宽 3', lambda r: tree_target_only_round(p, q, 3, 3, r)),
        ('target-only 树, 宽 3, threshold_acc=0.5', lambda r: tree_target_only_round(p, q, 3, 3, r, th_acc=0.5)),
        ('target-only 树, 宽 3, threshold_single=0.3', lambda r: tree_target_only_round(p, q, 3, 3, r, th_single=0.3)),
    ]
    res = {}
    for name, fn in cases:
        s = stream_stats(fn, p, n_tokens, rng)
        res[name] = s
        print(f'  {name:<42}{s["tv_uni"]:>8.4f}{s["tv_bi"]:>10.4f}{s["accept_len"]:>11.2f}'
              f'{s["correct_per_round"]:>14.2f}')
    print(f'\n参照：{n_tokens} 个样本的 TV 抽样噪声在 0.005 上下。超过 0.03 的那两行，输出分布已经不是 p 了。')
    print('理论值：链式 k=3 的每轮产出 = (1-α^4)/(1-α) = '
          f'{expected_tokens(alpha, 3):.2f}；宽 1 树 = (1-0.25^4)/0.75 = {expected_tokens(0.25, 3):.2f}；'
          f'宽 3 树每层接受概率 = p[1]+p[0]+p[4] = {p[1] + p[0] + p[4]:.2f}')

    peaked_p = [0.86, 0.06, 0.04, 0.02, 0.01, 0.01]
    peaked_q = [0.70, 0.12, 0.08, 0.05, 0.03, 0.02]
    a2 = sum(min(a, b) for a, b in zip(peaked_p, peaked_q))
    print(f'\n分布尖锐时（低 temperature 的常见情况）：p 的最大概率 {peaked_p[0]}，q 的最大概率 {peaked_q[0]}')
    print(f'  rejection sampling 接受率 Σmin(p,q) = {a2:.2f}，target-only top-1 接受率 = {peaked_p[0]:.2f}')
    res['peaked'] = (a2, peaked_p[0])
    res['alpha'] = alpha
    return res


# ---------------------------------------------------------------------------
# tree：EAGLE-2 动态树，按 SGLang 的张量布局逐步移植
# ---------------------------------------------------------------------------

def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def _mix(key, seed, salt):
    """确定性的整数哈希。不用内置 hash：含字符串时它受 PYTHONHASHSEED 影响，每次运行结果不同。"""
    h = 1469598103934665603 ^ (seed * 1000003 + salt)
    for t in key:
        h = ((h ^ (t + 1)) * 1099511628211) & ((1 << 64) - 1)
    return h


class ToyLM:
    """上下文取最后 order 个 token 的玩具语言模型。

    目标模型的 logits 由上下文哈希出的高斯数决定；草稿模型 = 目标 logits + 噪声，
    noise 越大草稿越不准。用哈希保证同一上下文永远给出同一分布。
    """

    def __init__(self, vocab=32, order=2, sharp=2.5, noise=0.0, seed=0, base=None):
        self.vocab, self.order, self.sharp = vocab, order, sharp
        self.noise, self.seed, self.base = noise, seed, base
        self.cache = {}

    def logits(self, ctx):
        key = tuple(ctx[-self.order:])
        if key not in self.cache:
            if self.base is None:
                r = random.Random(_mix(key, self.seed, 0))
                lg = [r.gauss(0, 1) * self.sharp for _ in range(self.vocab)]
            else:
                r = random.Random(_mix(key, self.seed, 1))
                lg = [x + r.gauss(0, 1) * self.noise for x in self.base.logits(ctx)]
            self.cache[key] = lg
        return self.cache[key]

    def dist(self, ctx):
        return softmax(self.logits(ctx))

    def argmax(self, ctx):
        lg = self.logits(ctx)
        return max(range(self.vocab), key=lambda i: (lg[i], -i))


def draft_tree(draft, ctx, steps, topk):
    """draft_forward（eagle_worker_v2.py:763）+ select_top_k_tokens 的逐元素版本。

    返回和 SGLang 同布局的 score_list / token_list / parents_list：
      第 0 层：topk 个候选，parents = [-1, 0, 1, ..., topk-1]
      第 i 层：当前 topk 个节点各取 topk 个孩子，累计路径概率 score x p，
              topk² 个里留 topk 个继续展开，parents = 留下的下标 + topk² (i-1) + topk
    第 0 层的 topk_p 来自 draft extend，之后每层一次草稿前向，最后一层不前向，
    所以一轮共 steps-1 次 draft_forward 内的前向。
    """
    p0 = draft.dist(ctx)
    t0 = topk_tokens(p0, topk)
    scores = [p0[t] for t in t0]
    score_list, token_list = scores[:], t0[:]
    parents_list = [[-1] + list(range(topk))]
    cur = [tuple(ctx) + (t,) for t in t0]
    forwards = 0
    for i in range(1, steps):
        forwards += 1
        expand, child = [], []
        for a in range(topk):
            pa = draft.dist(cur[a])
            for t in topk_tokens(pa, topk):
                expand.append(scores[a] * pa[t])
                child.append(t)
        cs = sorted(range(topk * topk), key=lambda j: (-expand[j], j))[:topk]
        parents_list.append([j + topk * topk * (i - 1) + topk for j in cs])
        cur = [cur[j // topk] + (child[j],) for j in cs]
        scores = [expand[j] for j in cs]
        score_list += expand
        token_list += child
    return score_list, token_list, parents_list, forwards


def organize_draft_results(score_list, token_list, parents_list, num_draft_tokens):
    """eagle_utils.py:103：全局取分数最高的 num_draft_tokens-1 个，下标升序排列。"""
    order = sorted(range(len(score_list)), key=lambda j: (-score_list[j], j))
    selected = sorted(order[:num_draft_tokens - 1])
    draft_tokens = [token_list[j] for j in selected]
    parent_list = [x for lst in parents_list[:-1] for x in lst]
    return parent_list, selected, draft_tokens


def build_tree(parent_list, selected, topk, seq_len=0):
    """spec_tree.py:53 的 Triton 核：父节点定位、兄弟链表、tree mask、positions。

    节点 0 是上一轮最后确认的 token（根），草稿 token 是 1..N-1。
    """
    n = len(selected) + 1
    next_token = [-1] * n
    next_sibling = [-1] * n
    parent = [-1] * n
    for i in range(n - 1, 0, -1):
        tb = selected[i - 1] // topk
        pos, found = 0, tb == 0
        if not found:
            ptok = parent_list[tb]
            for pp in range(n - 1):
                if selected[pp] == ptok:
                    pos, found = pp + 1, True
                    break
        if found:
            parent[i] = pos
            if next_token[pos] != -1:
                next_sibling[i] = next_token[pos]
            next_token[pos] = i
    depth = [0] * n
    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        j, d = i, 0
        while j != -1:
            mask[i][j] = 1
            if j == 0:
                break
            j = parent[j]
            d += 1
        depth[i] = d
    positions = [d + seq_len for d in depth]
    return {'next_token': next_token, 'next_sibling': next_sibling, 'parent': parent,
            'mask': mask, 'positions': positions, 'depth': depth}


def verify_greedy(tree, draft_tokens, target, ctx, steps):
    """spec_tree.py:211：每层取目标模型在“最后接受节点”处的 argmax，在孩子里找相同 token。"""
    node_ctx = {0: tuple(ctx)}
    cur, accepted = 0, []
    for _ in range(steps):
        want = target.argmax(node_ctx[cur])
        child = tree['next_token'][cur]
        hit = None
        while child != -1:
            if draft_tokens[child - 1] == want:
                hit = child
                break
            child = tree['next_sibling'][child]
        if hit is None:
            break
        accepted.append(want)
        node_ctx[hit] = node_ctx[cur] + (want,)
        cur = hit
    bonus = target.argmax(node_ctx[cur])
    return accepted + [bonus], len(accepted)


def spec_generate(target, draft, prompt, n_new, steps, topk, ndt):
    """一条请求跑完 n_new 个 token，返回输出和统计。"""
    if topk == 1:
        ndt = steps + 1             # speculative_hook.py:1301 的调整
    out = list(prompt)
    rounds = correct = forwards = 0
    while len(out) - len(prompt) < n_new:
        sl, tl, pl, f = draft_tree(draft, out, steps, topk)
        parent_list, selected, dt = organize_draft_results(sl, tl, pl, ndt)
        tree = build_tree(parent_list, selected, topk, seq_len=len(out))
        toks, c = verify_greedy(tree, dt, target, out, steps)
        out += toks
        rounds += 1
        correct += c
        forwards += f + 1            # + 1：verify 之后的 draft extend
    gen = out[len(prompt):len(prompt) + n_new]
    return gen, {'rounds': rounds, 'correct': correct, 'ndt': ndt, 'draft_forwards': forwards}


def greedy_generate(target, prompt, n_new):
    out = list(prompt)
    for _ in range(n_new):
        out.append(target.argmax(out))
    return out[len(prompt):]


TREE_CONFIGS = [(1, 1, 2), (3, 1, 4), (5, 1, 6), (5, 4, 8), (3, 4, 16), (6, 4, 32)]


def scenario_tree(noise=1.6, n_req=40, n_new=64, seed=0):
    target = ToyLM(seed=seed)
    draft = ToyLM(noise=noise, seed=seed, base=target)
    rng = random.Random(seed + 1)
    prompts = [[rng.randrange(target.vocab) for _ in range(8)] for _ in range(n_req)]

    # 先看一棵树长什么样
    ctx = prompts[0]
    sl, tl, pl, _ = draft_tree(draft, ctx, 3, 4)
    parent_list, selected, dt = organize_draft_results(sl, tl, pl, 8)
    tree = build_tree(parent_list, selected, 4, seq_len=len(ctx))
    print('steps=3, topk=4, num_draft_tokens=8 的一棵树（seq_len=8）')
    print(f'  候选总数 topk + (steps-1)·topk² = {len(sl)}，parent_list = {parent_list}')
    print(f'  selected（全局下标，升序）= {selected}')
    print('  节点  token  父节点  深度  position   累计概率   mask 行')
    for i in range(len(selected) + 1):
        tok = dt[i - 1] if i else '根'
        sc = f'{sl[selected[i - 1]]:.3f}' if i else '1'
        row = ''.join(str(x) for x in tree['mask'][i])
        print(f'  {i:>4}  {tok!s:>5}  {tree["parent"][i]:>6}  {tree["depth"][i]:>4}  '
              f'{tree["positions"][i]:>8}   {sc:>8}   {row}')

    print(f'\n{n_req} 条请求各生成 {n_new} 个 token，草稿噪声 σ={noise}，贪心验证')
    print('  (steps,topk,ndt)  每轮验证 token   accept len   accept rate   草稿前向/轮   和纯贪心一致')
    res = {}
    for steps, topk, ndt in TREE_CONFIGS:
        rounds = correct = fw = tokens = 0
        same = True
        for pr in prompts:
            gen, st = spec_generate(target, draft, pr, n_new, steps, topk, ndt)
            same &= gen == greedy_generate(target, pr, n_new)
            rounds += st['rounds']
            correct += st['correct']
            fw += st['draft_forwards']
            tokens += st['correct'] + st['rounds']
        ndt_eff = st['ndt']
        accept_len = tokens / rounds
        accept_rate = correct / (rounds * (ndt_eff - 1))
        res[(steps, topk, ndt)] = {'accept_len': accept_len, 'accept_rate': accept_rate,
                                   'same': same, 'ndt': ndt_eff}
        print(f'  ({steps},{topk},{ndt}){"":<{10 - len(str((steps, topk, ndt)))}}{ndt_eff:>14}'
              f'{accept_len:>13.2f}{accept_rate:>14.3f}{fw / rounds:>14.1f}{"是" if same else "否":>12}')
    print('  accept len 是每轮产出 token 数（含 bonus）；accept rate 按 metrics_reporter 口径，分母是 ndt-1。')
    print('  树越宽 accept rate 越低，这是口径决定的：16 个草稿最多只有一条路径能被接受。')
    return res


# ---------------------------------------------------------------------------
# batch：Roofline 下，验证 N 个 token 什么时候不再“免费”
# ---------------------------------------------------------------------------

# EAGLE-3 草稿头的量级（以 Llama-3.1-8B 为例）：一层 decoder（qkv 输入是 2·hidden）、
# 3·hidden -> hidden 的融合 fc、3.2 万词的草稿 lm_head，合计约 0.42B 参数
EAGLE3_HEAD_B = 0.42


def forward_ms(card, weight_b, tokens, n_req, ctx, m, kv_layers=None, flop_b=None):
    """一次前向的 Roofline 耗时：权重读一遍 + 每个请求的 KV 读一遍，和计算取大者。

    tokens 个 query 属于 n_req 个请求，同一请求的多个 query（树验证）共享同一段 KV，
    KV 只读一次（FlashInfer / FA3 的树验证都是这样组织的）。
    """
    layers = m.layers if kv_layers is None else kv_layers
    kv_per_tok = m.kv_bytes_per_token() * layers / m.layers
    wbytes = weight_b * 1e9 * 2
    kvbytes = n_req * ctx * kv_per_tok
    flops = 2 * tokens * (flop_b if flop_b is not None else weight_b) * 1e9 \
        + 4 * tokens * ctx * layers * m.heads * m.head_dim
    return max((wbytes + kvbytes) / (card.bw_tbs * 1e12), flops / (card.bf16_tflops * 1e12)) * 1e3


def spec_round_ms(card, m, batch, ctx, steps, topk, ndt, head_b=EAGLE3_HEAD_B):
    target_b = m.active_b - m.embed_b
    t_verify = forward_ms(card, target_b, batch * ndt, batch, ctx, m)
    # 草稿：verify 后的 extend 一次（每个请求 steps+1 个位置），加 draft_forward 里的 steps-1 次
    t_draft = forward_ms(card, head_b, batch * (steps + 1), batch, ctx, m, kv_layers=1)
    t_draft += (steps - 1) * forward_ms(card, head_b, batch * topk, batch, ctx, m, kv_layers=1)
    return t_verify, t_draft


def spec_speedup(card, m, batch, ctx, cfg, accept_len):
    steps, topk, ndt = cfg
    t_base = forward_ms(card, m.active_b - m.embed_b, batch, batch, ctx, m)
    tv, td = spec_round_ms(card, m, batch, ctx, steps, topk, ndt)
    return accept_len * t_base / (tv + td), t_base, tv, td


# 示例接受长度。换成自己服务的 sglang:spec_accept_length 再跑一遍
BATCH_CONFIGS = [((3, 1, 4), 2.8), ((5, 4, 8), 3.6), ((3, 4, 16), 3.4)]


def kv_pool_tokens(model, gpu):
    """复用 02-ledger 的切池逻辑，再按 1 层 EAGLE 草稿把 cell 放大（见 resolve_sim.py --scenario memory）。"""
    sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
    sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
    from sim_ledger import ledger
    L = ledger(model, gpu, tp=1, weight_dtype='bf16', kv_dtype='bf16')
    m = MODELS[model]
    return int(L['rest'] * (1 << 30) // (L['cell'] * (1 + 1 / m.layers)))


def scenario_batch(gpu='h100-sxm', model='llama3.1-8b', accept=None):
    card, m = GPUS[gpu], MODELS[model]
    configs = BATCH_CONFIGS if accept is None else [(c, accept) for c, _ in BATCH_CONFIGS]
    batches = [1, 4, 8, 16, 32, 64, 128, 256]
    res = {}
    pool = kv_pool_tokens(model, gpu)
    for ctx in (1024, 8192):
        print(f'{m.name} on {card.name}，上下文 {ctx}，Roofline 估算的吞吐加速比（>1 才划算；'
              f'- 表示 KV 池 {pool:,} token 放不下这么多请求）')
        print('  (steps,topk,ndt) L    ' + ''.join(f'{b:>7}' for b in batches))
        for cfg, L in configs:
            row = {b: spec_speedup(card, m, b, ctx, cfg, L)[0] for b in batches if b * ctx <= pool}
            res[(ctx, cfg)] = row
            print(f'  {str(cfg):<16} {L:<4} ' + ''.join(f'{row[b]:>7.2f}' if b in row else f'{"-":>7}'
                                                        for b in batches))
        print()
    b = 64
    sp, tb, tv, td = spec_speedup(card, m, b, 1024, (3, 4, 16), 3.4)
    print(f'拆开看 batch={b}、ctx=1024、(3,4,16)：普通 decode 一步 {tb:.2f} ms；'
          f'验证 {b}x16={b * 16} 个 token {tv:.2f} ms，草稿 {td:.2f} ms')
    print(f'  验证一次处理 {b * 16} 个 token，已经越过 H100 BF16 的拐点（约 {card.ridge:.0f} token），'
          '从读权重为主变成算力为主，猜错的 token 开始真金白银地花钱。')
    print('  长上下文时每步都要读整段 KV，而树验证的多个 query 共享这段 KV，额外 token 又变得便宜，'
          '所以 8K 上下文的表整体往右移；可惜单卡 KV 池只装得下 48 条 8K 请求，大 batch 根本到不了。')
    res['breakdown'] = (tb, tv, td)
    return res


# ---------------------------------------------------------------------------
# ngram：不需要草稿模型，从已有文本里找后缀匹配
# ---------------------------------------------------------------------------

def ngram_draft(history, max_depth, min_match, breadth, budget):
    """NGRAM 草稿的简化版：用最后 n 个 token（n 从大到小）在历史里找出现位置，
    取最近几处出现之后的 token 当候选，最多 breadth 条分支、合计 budget 个 token。

    sglang 的 C++ 实现（cpp_ngram）用 trie 记录频次、按 BFS 或概率扩展，思路一致。
    """
    n_hist = len(history)
    for n in range(min(max_depth, n_hist - 1), min_match - 1, -1):
        suffix = history[-n:]
        branches, seen = [], set()
        for start in range(n_hist - n - 1, -1, -1):
            if history[start:start + n] == suffix:
                cont = tuple(history[start + n:start + n + budget])
                if cont and cont[0] not in seen:
                    seen.add(cont[0])
                    branches.append(cont)
                    if len(branches) == breadth:
                        break
        if branches:
            per = max(1, budget // len(branches))
            return [b[:per] for b in branches]
    return []


def ngram_verify(branches, truth):
    """贪心验证：每条分支和真实续写比较最长公共前缀，取最长的那条。"""
    best = 0
    for b in branches:
        k = 0
        while k < len(b) and k < len(truth) and b[k] == truth[k]:
            k += 1
        best = max(best, k)
    return best


def make_copy_heavy(prompt, n, rng):
    """输出里大段复用 prompt 的片段，中间夹几个新 token：改代码、RAG 摘抄、格式转换都是这样。"""
    out = []
    while len(out) < n:
        s = rng.randrange(len(prompt) - 30)
        out += prompt[s:s + rng.randint(8, 30)]
        out += [rng.randrange(5000) for _ in range(rng.randint(1, 4))]
    return out[:n]


def scenario_ngram(seed=0, ndt=12, breadth=4):
    rng = random.Random(seed)
    res = {}
    print(f'NGRAM：num_draft_tokens={ndt}（_handle_ngram 的默认值），每轮最多 {breadth} 条分支')
    print('  输出类型       每轮产出 token   验证轮数 / 生成 token')
    for kind in ('copy', 'fresh'):
        rounds = tokens = 0
        for _ in range(20):
            prompt = [rng.randrange(5000) for _ in range(600)]
            out = make_copy_heavy(prompt, 300, rng) if kind == 'copy' else \
                [rng.randrange(5000) for _ in range(300)]
            pos = 0
            while pos < len(out):
                hist = prompt + out[:pos]
                br = ngram_draft(hist, max_depth=18, min_match=1, breadth=breadth, budget=ndt - 1)
                k = ngram_verify(br, out[pos:])
                pos += k + 1
                rounds += 1
                tokens += k + 1
        L = tokens / rounds
        res[kind] = L
        name = '抄写型（改代码）' if kind == 'copy' else '全新文本'
        print(f'  {name:<12}{L:>14.2f}{rounds / tokens:>18.2f}')
    print('  抄写型输出里每轮能白拿好几个 token；全新文本几乎匹配不上，NGRAM 只剩验证开销。')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='theory',
                    choices=['theory', 'sampling', 'tree', 'batch', 'ngram'])
    ap.add_argument('--gpu', default='h100-sxm')
    ap.add_argument('--model', default='llama3.1-8b')
    ap.add_argument('--accept-len', type=float, default=None, help='batch 场景统一使用的接受长度')
    ap.add_argument('--noise', type=float, default=1.6, help='tree 场景草稿模型的噪声')
    a = ap.parse_args()
    if a.scenario == 'theory':
        scenario_theory()
    elif a.scenario == 'sampling':
        scenario_sampling()
    elif a.scenario == 'tree':
        scenario_tree(noise=a.noise)
    elif a.scenario == 'batch':
        scenario_batch(a.gpu, a.model, a.accept_len)
    else:
        scenario_ngram()


if __name__ == '__main__':
    main()
