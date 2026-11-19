"""注意力后端手里的数据结构和数学：从 req_to_token 拼出 kernel 要的索引，再把两段注意力按 LSE 合起来。

只用标准库，数值用 Python float 算，和 GPU 上的 kernel 只差在精度和并行。

  python3 paged_attn.py                      # req_to_token → FlashInfer 的 CSR 和 FA3 的 page_table
  python3 paged_attn.py --scenario merge     # ragged（新 token）+ paged（前缀）按 LSE 合并 = 一次算完
  python3 paged_attn.py --scenario split     # split-K：把一条长 KV 切成几段，各算各的再 reduce
  python3 paged_attn.py --scenario page      # page_size 对前缀命中和尾页浪费的影响
  python3 paged_attn.py --scenario contract  # out_graph / in_graph 的契约检查
  python3 paged_attn.py --scenario hybrid    # 拆分 prefill/decode 后端时，每种 forward 走哪个

对应 sglang 8fabf5e（路径相对 python/sglang/srt/）：
  layers/attention/flashinfer_backend.py:486      kv_last_page_len 全 1，page_size 固定传 1
  layers/attention/flashinfer_backend.py:1357     ragged 段和 paged 段各自出 (o, lse)，再 merge_state
  layers/attention/flashattention_backend.py:819  page_table = req_to_token[:, ::page] // page
  layers/attention/triton_backend.py:431          get_num_kv_splits
  layers/attention/base_attn_backend.py:139       in_graph 不许 .item() / .cpu() / .tolist() / 动态 torch.empty
  layers/attention/hybrid_attn_backend.py:64      _select_backend
  mem_cache/radix_cache.py:381                    page_size > 1 时匹配长度截到 page 的整数倍
"""
import argparse
import ast
import math
import random
import textwrap
from typing import List, Sequence, Tuple

Vec = List[float]


# ---------------------------------------------------------------- 索引

def build_csr(req_to_token: Sequence[Sequence[int]], req_pool_indices: Sequence[int],
              seq_lens: Sequence[int]) -> Tuple[List[int], List[int]]:
    """FlashInfer 的 paged KV 用 CSR 描述：kv_indptr[i]..kv_indptr[i+1] 是第 i 条请求的 KV 槽位。

    sglang 传 page_size=1，每个「页」就是一个 token 槽位，所以 kv_indices 直接是
    req_to_token[req, :seq_len] 首尾相接，kv_last_page_len 全是 1。
    真实实现是一个 Triton kernel（create_flashinfer_kv_indices_triton），逻辑就是下面这个循环。
    """
    indptr, indices = [0], []
    for r, n in zip(req_pool_indices, seq_lens):
        indices.extend(req_to_token[r][:n])
        indptr.append(indptr[-1] + n)
    return indptr, indices


def page_table(req_to_token: Sequence[Sequence[int]], req_pool_indices: Sequence[int],
               seq_lens: Sequence[int], page: int) -> List[List[int]]:
    """FA3 / FA4 / FlashMLA 一类后端要的是二维页表：每行一条请求，每格一个物理页号。

    sglang 的 KV 池按 token 编号，page_size>1 时分配器保证同一页的 token 槽位连续且对齐，
    所以取每页第一个 token 的槽位除以 page 就是页号。page=1 时页表就是 req_to_token 本身。
    """
    out = []
    for r, n in zip(req_pool_indices, seq_lens):
        row = req_to_token[r][:n]
        out.append([row[i] // page for i in range(0, n, page)])
    return out


def check_page_aligned(row: Sequence[int], page: int) -> bool:
    """分配器的不变量：同一页里的槽位是 base, base+1, …，base 是 page 的倍数。"""
    for i in range(0, len(row), page):
        base = row[i]
        if base % page:
            return False
        for j in range(i, min(i + page, len(row))):
            if row[j] != base + (j - i):
                return False
    return True


# ---------------------------------------------------------------- 注意力数学

def dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def attend(q: Vec, ks: Sequence[Vec], vs: Sequence[Vec], scale: float) -> Tuple[Vec, float]:
    """一个 query 对一段 KV 的注意力，返回 (归一化后的输出, log-sum-exp)。空段返回 (0, -inf)。"""
    if not ks:
        return [0.0] * len(q), -math.inf
    s = [dot(q, k) * scale for k in ks]
    m = max(s)
    p = [math.exp(x - m) for x in s]
    z = sum(p)
    o = [sum(pi * v[d] for pi, v in zip(p, vs)) / z for d in range(len(vs[0]))]
    return o, m + math.log(z)


def merge_state(o1: Vec, lse1: float, o2: Vec, lse2: float) -> Tuple[Vec, float]:
    """两段的结果按各自的 softmax 分母加权：权重 = exp(lse_i - lse)。flashinfer.merge_state 做的就是这个。"""
    if lse1 == -math.inf:
        return o2, lse2
    if lse2 == -math.inf:
        return o1, lse1
    m = max(lse1, lse2)
    w1, w2 = math.exp(lse1 - m), math.exp(lse2 - m)
    lse = m + math.log(w1 + w2)
    return [(a * w1 + b * w2) / (w1 + w2) for a, b in zip(o1, o2)], lse


def extend_two_pass(qs, prefix_k, prefix_v, new_k, new_v, scale):
    """extend 时 FlashInfer 的做法：前缀在 KV 池里（paged，不加 mask），新 token 在手上（ragged，因果）。
    第 i 个新 token 看全部前缀加上新 token 0..i。两次 kernel 各出 (o, lse)，最后合并。"""
    out = []
    for i, q in enumerate(qs):
        o_p, l_p = attend(q, prefix_k, prefix_v, scale)
        o_r, l_r = attend(q, new_k[:i + 1], new_v[:i + 1], scale)
        out.append(merge_state(o_p, l_p, o_r, l_r)[0])
    return out


def extend_one_pass(qs, prefix_k, prefix_v, new_k, new_v, scale):
    ks, vs = list(prefix_k) + list(new_k), list(prefix_v) + list(new_v)
    n = len(prefix_k)
    return [attend(q, ks[:n + i + 1], vs[:n + i + 1], scale)[0] for i, q in enumerate(qs)]


def split_k(q, ks, vs, scale, splits: int) -> Tuple[Vec, float, List[Tuple[int, int]]]:
    """Flash-Decoding：KV 沿长度切成 splits 段，每段一个 CTA 独立算 (o, lse)，第二个 kernel 归并。"""
    n = len(ks)
    step = math.ceil(n / splits)
    o, lse, segs = [0.0] * len(q), -math.inf, []
    for s in range(0, n, step):
        segs.append((s, min(n, s + step)))
        o_s, l_s = attend(q, ks[s:s + step], vs[s:s + step], scale)
        o, lse = merge_state(o, lse, o_s, l_s)
    return o, lse, segs


def max_abs_diff(a: Sequence[Vec], b: Sequence[Vec]) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def rand_vecs(rng, n, d):
    return [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]


# ---------------------------------------------------------------- page_size

def prefix_hit(shared: int, page: int) -> int:
    """radix_cache.py:381：page_size>1 时匹配长度先截到 page 的整数倍。"""
    return shared // page * page


def page_stats(shared: int, prompt_lens: Sequence[int], out_len: int, page: int) -> dict:
    """一批请求共享 shared 个 token 的前缀。返回命中率和尾页浪费。

    命中：每条请求能复用 prefix_hit(shared, page) 个 token。
    浪费：每条请求最后一页没填满的槽位（prompt + 输出总长对 page 取余后补齐）。
    """
    hit = sum(prefix_hit(min(shared, L), page) for L in prompt_lens)
    total = sum(prompt_lens)
    waste = sum((-(L + out_len)) % page for L in prompt_lens)
    used = sum(L + out_len for L in prompt_lens)
    return dict(hit_rate=hit / total, waste=waste, waste_rate=waste / (used + waste))


# ---------------------------------------------------------------- 契约检查

BANNED_ATTR = {'item', 'cpu', 'tolist', 'numpy'}


def check_in_graph(src: str) -> List[str]:
    """扫 init_forward_metadata_in_graph 的函数体：graph 里录不进去的操作。

    规则来自 base_attn_backend.py:145 的 lint 注释：.item() / .cpu() / .tolist() 会让 CPU 等 GPU，
    capture 时直接报错或把当时的值烧进图里；torch.empty 的形状如果依赖本步数据，replay 时形状不会变。
    这里把「形状参数里出现了 forward_batch 的属性或 len()」当成动态形状。
    """
    tree = ast.parse(textwrap.dedent(src))
    bad = []  # (行号, 说明)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in BANNED_ATTR:
                bad.append((node.lineno, f'第 {node.lineno} 行 .{name}()：要同步 CPU，挪到 out_graph'))
            if name in ('empty', 'zeros', 'ones', 'arange') and _dynamic_shape(node):
                bad.append((node.lineno, f'第 {node.lineno} 行 torch.{name} 的形状随批次变：预分配到 init_cuda_graph_state'))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('int', 'float'):
            if node.args and isinstance(node.args[0], (ast.Attribute, ast.Subscript, ast.Call)):
                bad.append((node.lineno, f'第 {node.lineno} 行 {node.func.id}(张量)：隐式 .item()'))
    return [msg for _, msg in sorted(bad)]


def _dynamic_shape(call: ast.Call) -> bool:
    for a in call.args:
        for n in ast.walk(a):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in (
                    'forward_batch', 'fb'):
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'len':
                return True
    return False


GOOD_IN_GRAPH = '''
def init_forward_metadata_in_graph(self, forward_batch):
    bs = self.graph_bs
    torch.cumsum(forward_batch.seq_lens, dim=0, out=self.kv_indptr[1:bs + 1])
    create_kv_indices[(bs,)](self.req_to_token, forward_batch.req_pool_indices,
                             forward_batch.seq_lens, self.kv_indptr, self.kv_indices)
'''

BAD_IN_GRAPH = '''
def init_forward_metadata_in_graph(self, forward_batch):
    total = forward_batch.seq_lens.sum().item()
    kv_indices = torch.empty(total, dtype=torch.int32, device="cuda")
    max_len = int(forward_batch.seq_lens.max())
    lens = forward_batch.seq_lens.cpu().tolist()
    scratch = torch.empty(forward_batch.batch_size, 8, device="cuda")
'''


# ---------------------------------------------------------------- 拆分后端

def select_backend(mode: str, spec_attention_mode: str = 'prefill') -> str:
    """hybrid_attn_backend.py:64。spec_attention_mode 是 --speculative-attention-mode，默认 prefill。"""
    if mode in ('decode', 'idle'):
        return 'decode'
    if mode == 'target_verify':
        return 'decode' if spec_attention_mode == 'decode' else 'prefill'
    return 'prefill'          # extend、mixed 等其余模式都走 prefill 后端


# ---------------------------------------------------------------- 场景

def demo_pool(page: int = 1):
    """3 条请求，长度 5、3、7。page=1 时槽位随便散；page=4 时按页分配。"""
    if page == 1:
        rows = [[17, 3, 9, 40, 41], [8, 22, 23], [50, 51, 52, 53, 11, 12, 30]]
    else:
        rows = [[16, 17, 18, 19, 4], [8, 9, 10], [40, 41, 42, 43, 28, 29, 30]]
    req_to_token = {2: rows[0] + [0] * 3, 5: rows[1] + [0] * 5, 7: rows[2] + [0]}
    return req_to_token, [2, 5, 7], [5, 3, 7]


def scenario_csr():
    r2t, idx, lens = demo_pool(1)
    indptr, indices = build_csr(r2t, idx, lens)
    print('  page_size=1（FlashInfer 和 Triton 的默认）')
    for r, n in zip(idx, lens):
        print(f'    req_to_token[{r}][:{n}] = {r2t[r][:n]}')
    print(f'    kv_indptr  = {indptr}')
    print(f'    kv_indices = {indices}')
    print(f'    kv_last_page_len = {[1] * len(idx)}')
    print(f'    FA3 page_table（page=1，就是 req_to_token 本身）= {page_table(r2t, idx, lens, 1)}')
    print()
    r2t, idx, lens = demo_pool(4)
    print('  page_size=4（FA3 / FlashMLA / TRTLLM 一类后端要的页表）')
    for r, n in zip(idx, lens):
        row = r2t[r][:n]
        print(f'    req_to_token[{r}][:{n}] = {row}  对齐 {check_page_aligned(row, 4)}')
    print(f'    page_table（每页首个槽位 // 4）= {page_table(r2t, idx, lens, 4)}')
    print()
    print('  读法：sglang 的 KV 池永远按 token 编号，后端只是换一种方式「看」同一份 req_to_token。')
    print('  page=1 时 FlashInfer 拿到的 indices 长度等于总 token 数，plan 要在 CPU 上算 indptr，')
    print('  这是它元数据比 FA3 重的来源；FA3 的页表是一个定长二维张量，CUDA Graph 下只需原地改。')


def scenario_merge(seed: int = 0):
    rng = random.Random(seed)
    d, n_prefix, n_new = 16, 37, 9
    scale = 1 / math.sqrt(d)
    pk, pv = rand_vecs(rng, n_prefix, d), rand_vecs(rng, n_prefix, d)
    nk, nv = rand_vecs(rng, n_new, d), rand_vecs(rng, n_new, d)
    qs = rand_vecs(rng, n_new, d)
    two = extend_two_pass(qs, pk, pv, nk, nv, scale)
    one = extend_one_pass(qs, pk, pv, nk, nv, scale)
    print(f'  前缀命中 {n_prefix} 个 token（在 KV 池里），新来 {n_new} 个 token（因果）')
    print(f'  paged 段 + ragged 段 + merge_state  vs  一次算完：最大误差 {max_abs_diff(two, one):.2e}')
    o_p, l_p = attend(qs[0], pk, pv, scale)
    o_r, l_r = attend(qs[0], nk[:1], nv[:1], scale)
    _, l = merge_state(o_p, l_p, o_r, l_r)
    w = math.exp(l_p - l)
    print(f'  第 0 个新 token：lse_paged={l_p:.3f}  lse_ragged={l_r:.3f}  合并后 {l:.3f}，前缀权重 {w:.1%}')
    print()
    print('  读法：softmax 的分母可以分段求再合并，所以前缀和新 token 不必拼到一块连续显存里。')
    print('  FlashInfer extend 跑两个 kernel 加一次 merge（flashinfer_backend.py:1357）；')
    print('  FA3 一个 kernel 里同时吃页表和新 K/V。chunked prefill 的每一块、Flash-Decoding 的每一段，都是同一个公式。')


def scenario_split(seed: int = 1):
    rng = random.Random(seed)
    d, n = 16, 1000
    scale = 1 / math.sqrt(d)
    ks, vs = rand_vecs(rng, n, d), rand_vecs(rng, n, d)
    q = rand_vecs(rng, 1, d)[0]
    ref, ref_lse = attend(q, ks, vs, scale)
    print(f'  一个 decode query，KV 长 {n}')
    print(f'  {"splits":>8}{"每段长度":>10}{"最大误差":>12}{"中间结果 float 数":>20}')
    for s in (1, 2, 8, 16, 64):
        o, lse, segs = split_k(q, ks, vs, scale, s)
        err = max(abs(a - b) for a, b in zip(o, ref))
        print(f'  {s:>8}{segs[0][1] - segs[0][0]:>10}{err:>12.1e}{len(segs) * (d + 1):>20}')
    print()
    print('  读法：切多少段结果都一样（只差浮点舍入），代价是 splits×(d+1) 个 fp32 中间结果和一次 reduce。')
    print('  算 SM 占用和耗时的账见 attn_cost.py --scenario split。')
    print('  确定性推理要求切法与 batch 无关：triton_backend.py:431 在确定性模式下按固定 tile 切，')
    print('  不再按 batch 里的最长序列动态取 split，否则同一条请求在不同 batch 里归并顺序不同、末位会变。')


def scenario_page():
    rng = random.Random(7)
    shared = 1000                            # 一段 1000 token 的系统提示
    prompts = [shared + rng.randint(20, 400) for _ in range(256)]
    print(f'  256 条请求共享 {shared} token 的系统提示，各自再接 20~400 token，输出 200 token')
    print(f'  {"page":>6}{"可复用前缀":>10}{"前缀命中率":>11}{"尾页浪费槽位":>13}{"浪费占比":>9}')
    for page in (1, 16, 32, 64, 128):
        st = page_stats(shared, prompts, 200, page)
        print(f'  {page:>6}{prefix_hit(shared, page):>10}{st["hit_rate"]:>11.1%}'
              f'{st["waste"]:>13}{st["waste_rate"]:>9.2%}')
    print()
    print('  多轮对话：上一轮的回复也在缓存里，下一轮的前缀长度每次都不是整页')
    print(f'  {"page":>6}' + ''.join(f'{f"前缀 {L}":>11}' for L in (130, 515, 2047)))
    for page in (1, 16, 64, 128):
        print(f'  {page:>6}' + ''.join(f'{prefix_hit(L, page):>11}' for L in (130, 515, 2047)))
    print()
    print('  读法：page 越大，每条请求最多丢 page-1 个本可复用的 token，还要多占 page/2 个尾槽位。')
    print('  长 prompt 场景这点损失只有百分之几；短 prompt、高并发、前缀极碎的场景，page=128 能吃掉可观的命中率。')
    print('  TRTLLM / FA4 / FlashMLA 强制 page，这笔账要和 kernel 本身的提速一起算。')


def scenario_contract():
    for name, src in [('合规', GOOD_IN_GRAPH), ('违规', BAD_IN_GRAPH)]:
        bad = check_in_graph(src)
        print(f'  {name}的 init_forward_metadata_in_graph：{len(bad)} 处问题')
        for b in bad:
            print(f'    {b}')
    print()
    print('  读法：一步元数据分两半。out_graph 在 graph.replay() 之前跑，可以碰 CPU、可以算长度；')
    print('  in_graph 被录进图里，只能是固定形状的 GPU 操作，形状按捕获时的 bs 定死，靠预分配的缓冲区原地写。')
    print('  自己写后端时，CUDA Graph 下结果错而 eager 下对，第一个怀疑就是 in_graph 里藏了依赖本步数据的东西。')


def scenario_hybrid():
    modes = ['extend', 'decode', 'idle', 'target_verify', 'mixed']
    print('  --prefill-attention-backend fa3 --decode-attention-backend flashinfer 时，各 forward 走哪个')
    print(f'  {"forward_mode":<16}{"spec mode=prefill":>20}{"spec mode=decode":>20}')
    names = {'prefill': 'fa3', 'decode': 'flashinfer'}
    for m in modes:
        print(f'  {m:<16}{names[select_backend(m)]:>20}{names[select_backend(m, "decode")]:>20}')
    print()
    print('  读法：投机解码的 verify 一次要算 draft 个 token，默认当成小 prefill 交给 prefill 后端。')
    print('  拆分后端时两个后端各有一份元数据和 workspace，显存和 plan 开销都是两份。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='csr',
                    choices=['csr', 'merge', 'split', 'page', 'contract', 'hybrid'])
    s = ap.parse_args().scenario
    {'csr': scenario_csr, 'merge': scenario_merge, 'split': scenario_split, 'page': scenario_page,
     'contract': scenario_contract, 'hybrid': scenario_hybrid}[s]()


if __name__ == '__main__':
    main()
