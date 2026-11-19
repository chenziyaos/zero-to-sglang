"""TBO（two-batch overlap）：一个批怎么切成两半，以及切完以后到底快不快。

只用标准库。切分逻辑逐行移植自 sglang 8fabf5e 的 batch_overlap/two_batch_overlap.py，
阶段编排取自 batch_overlap/operations_strategy.py 的 DeepSeek 策略。

  python3 tbo_sim.py                        # 切分：decode 对半、extend 平衡切和两段切
  python3 tbo_sim.py --scenario decode      # decode 每层耗时：关 TBO vs 开 TBO，随每卡 batch 变化
  python3 tbo_sim.py --scenario prefill     # prefill：DeepEP normal 占 SM 之后还剩多少收益
  python3 tbo_sim.py --scenario stages      # 两个 micro-batch 在一层里的交错时间线

口径：
  耗时模型是屋顶线级别的粗估，用来看「什么时候赚、什么时候亏」，不用来预测绝对值。
  计算 = max(读权重, 算 FLOPs) + 读 KV + 每个 kernel 的固定开销；
  通信 = DeepEP low_latency 实测带宽（README，EP64 那一档）+ 每次调用的固定延迟。
  low_latency 的 hook 模式在后台收发、不占 SM，所以通信和另一个 micro-batch 的计算可以完全重叠。
  两个 micro-batch 在整个前向里错开推进（decode 的 tbo_delta_stages=2），
  填充和排空只在前向首尾各发生一次，摊到 61 层上忽略不计。
"""
import argparse
import os
import sys
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_parallel import LL_BW, deepep_wire_bytes, nearest  # noqa: E402

# ---------------------------------------------------------------- 切分逻辑（two_batch_overlap.py）

TBO_THRESHOLD = 0.48      # --tbo-token-distribution-threshold，exec_.py:1065


def split_balanced(arr: Sequence[int]) -> int:
    """_split_array_by_balanced_sum：从左往右累加，左右差第一次变大就停。"""
    total = sum(arr)
    left, best, min_diff = 0, 0, float('inf')
    for i in range(1, len(arr)):
        left += arr[i - 1]
        diff = abs(left - (total - left))
        if diff <= min_diff:
            min_diff, best = diff, i
        else:
            break
    return best


def split_cum_half(arr: Sequence[int]) -> int:
    """_split_array_by_cum_less_than_half：累加第一次超过 sum // 2 的那条序列的下标。"""
    left, half = 0, sum(arr) // 2
    for i, x in enumerate(arr):
        left += x
        if left > half:
            return i
    return 0


def two_chunk_enabled(arr: Sequence[int], threshold: float = TBO_THRESHOLD) -> bool:
    """_is_two_chunk_split_enabled：平衡切的左半边占比落在 [thr, 1-thr] 之外时，改为把一条序列切成两段。"""
    if not arr:
        return False
    vi = split_balanced(arr)
    left, total = sum(arr[:vi]), sum(arr)
    if not (left < total * threshold or left > total * (1 - threshold)):
        return False
    # child_a 的序列数不能多于它分到的 token 数（DSV4 compress planner 的约束）
    if split_cum_half(arr) + 1 > total // 2:
        return False
    return True


def split_extend(arr: Sequence[int], threshold: float = TBO_THRESHOLD):
    """返回 (方式, 切分序列下标, 切分 token 下标)。compute_split_seq_index + compute_split_token_index。"""
    if two_chunk_enabled(arr, threshold):
        return 'two-chunk', split_cum_half(arr), sum(arr) // 2
    i = split_balanced(arr)
    return 'balanced', i, sum(arr[:i])


def split_decode(bs: int, tokens_per_seq: int = 1):
    """decode / target_verify：按请求数对半，奇数时后一半多一个。"""
    i = bs // 2
    return i, i * tokens_per_seq


# ---------------------------------------------------------------- 耗时模型

class HW:
    """H800 / H100 量级，留了有效率折扣。"""
    hbm = 3.35e12 * 0.8          # B/s
    fp8 = 1979e12 * 0.55         # FLOP/s，DeepGEMM 在大 M 上的量级
    bf16 = 989e12 * 0.55
    sms = 132
    kernel_us = 3.0              # 一个 kernel 的固定开销（CUDA Graph 下的 launch + 尾部空转）
    comm_fixed_us = 15.0         # 一次 low_latency dispatch / combine 的固定延迟


class DSV3:
    hidden = 7168
    topk = 8
    expert_params = 3 * 7168 * 2048          # 一个路由专家
    shared_params = 3 * 7168 * 2048          # 1 个共享专家
    attn_params = 187e6                      # MLA 一层的权重（sim_parallel 按结构算出 11.4 GB / 61 层）
    kv_bytes = (512 + 64) * 2                # 一层每 token 的 latent KV，BF16
    kernels_per_layer = 12                   # 一个 micro-batch 一层大约这么多 kernel


def compute_us(b: int, ctx: int, local_experts: int, hw=HW, m=DSV3) -> float:
    """一个 micro-batch 在一层里的计算耗时（μs）。b = 本卡 token 数。"""
    if b == 0:
        return 0.0
    attn_w = m.attn_params / hw.hbm
    attn_f = 2 * m.attn_params * b / hw.fp8
    kv = b * ctx * m.kv_bytes / hw.hbm
    recv = b * m.topk                                     # 平均每卡收到的 (token, 专家) 对
    exp_w = local_experts * m.expert_params / hw.hbm
    exp_f = 2 * m.expert_params * recv / hw.fp8
    sh_w = m.shared_params / hw.hbm
    sh_f = 2 * m.shared_params * b / hw.fp8
    t = max(attn_w, attn_f) + kv + max(exp_w, exp_f) + max(sh_w, sh_f)
    return t * 1e6 + m.kernels_per_layer * hw.kernel_us


def comm_us(b: int, ep: int, hw=HW, m=DSV3) -> float:
    if b == 0:
        return 0.0
    d, c = deepep_wire_bytes(m.hidden)
    bd, bc = nearest(LL_BW, ep)
    return (b * m.topk * d / (bd * 1e9) + b * m.topk * c / (bc * 1e9)) * 1e6 + 2 * hw.comm_fixed_us


def decode_layer(b: int, ctx: int, ep: int, num_phy: int = 288):
    local = num_phy // ep
    C, M = compute_us(b, ctx, local), comm_us(b, ep)
    off = C + M
    b1, b2 = b // 2, b - b // 2
    C1, C2 = compute_us(b1, ctx, local), compute_us(b2, ctx, local)
    M1, M2 = comm_us(b1, ep), comm_us(b2, ep)
    on = max(C1 + C2, M1 + M2)
    return dict(C=C, M=M, off=off, on=on, C_half=C1 + C2, M_half=M1 + M2)


# ---------------------------------------------------------------- 场景

def scenario_split():
    print('decode：按请求数对半（compute_split_seq_index，bs // 2）')
    for bs in (1, 7, 64, 256):
        i, t = split_decode(bs)
        print(f'  bs={bs:<4} → A {i} 条 / B {bs - i} 条')
    i, t = split_decode(5, tokens_per_seq=4)
    print(f'  target_verify bs=5，每条 4 个 draft token → A {i} 条 {t} token / B {5 - i} 条 {20 - t} token')
    print(f'\nextend：先按序列边界找最平衡的切点；左半占比落在 [{TBO_THRESHOLD}, {1 - TBO_THRESHOLD:.2f}] 之外，')
    print('        改成「两段切」：在 sum // 2 处把一条长序列劈开，前半给 A、后半给 B')
    cases = [
        [1000, 1000, 1000, 1000],
        [4000, 100, 100, 100],
        [3000, 2000, 1000],
        [8192],
        [100, 200, 300, 400, 5000],
        [1],
    ]
    print(f'  {"extend_lens":<28}{"方式":<12}{"切在第几条":>10}{"A token":>9}{"B token":>9}')
    for arr in cases:
        how, si, ti = split_extend(arr)
        print(f'  {str(arr):<28}{how:<12}{si:>10}{ti:>9}{sum(arr) - ti:>9}')
    print('\n  [8192] 这种单条长 prompt 平衡切切不开（A 拿 0），两段切把它劈成 4096 + 4096；')
    print('  两段切时 B 要接着 A 的 KV 往下算，A 那段先写 KV，B 的 attention 读得到。')
    print('  [1] 只有 1 个 token：两段切会让 A 有 1 条序列却分到 0 个 token，违反 batch_size ≤ token 数，')
    print('  回退成按序列切，A 拿 0 条，整批交给 B，等于这一步没有 overlap。')


def scenario_decode(ctx: int = 2000, ep: int = 72):
    print(f'DeepSeek-V3 decode，EP{ep} + 288 个物理专家（每卡 {288 // ep} 个），平均上下文 {ctx}')
    print(f'  每卡 token 数 = 本卡 decode batch；DeepEP low_latency，RDMA 带宽取 README EP64 档\n')
    print(f'  {"每卡 bs":>8}{"计算":>8}{"通信":>8}{"关 TBO":>9}{"两半计算":>10}{"两半通信":>10}{"开 TBO":>9}{"变化":>8}')
    for b in (8, 16, 32, 64, 128, 256):
        r = decode_layer(b, ctx, ep)
        gain = r['off'] / r['on'] - 1
        print(f'  {b:>8}{r["C"]:8.0f}{r["M"]:8.0f}{r["off"]:9.0f}{r["C_half"]:10.0f}{r["M_half"]:10.0f}'
              f'{r["on"]:9.0f}{gain:+8.0%}')
    print('\n  单位 μs / 层。开 TBO 的耗时 = 两半计算之和 与 两半通信之和 里较大的那个。')
    print('  小 batch 时专家和 attention 都卡在读权重上，切成两半各读一遍，两半计算之和接近翻倍；')
    print('  通信又不够长，藏不住这多出来的计算，于是变慢。batch 上去以后计算按 token 线性涨，')
    print('  切开不吃亏，通信被整段藏进另一半的计算里。')
    print('  LMSYS 实测（H800，EP72 decode）：每卡 32 时 TBO 慢 27%，256 时快 25.5%；')
    print('  博客的结论是 batch 小于 64 到 128 时别开。模型方向一致，幅度偏乐观：')
    print('  它假设通信能完美藏住，也没算两个 micro-batch 各自一份的 kernel 尾部空转。')


def scenario_prefill(tokens: int = 16384, ep: int = 32, deepep_sms: int = 24, nodes_touched: float = 3.0,
                     avg_ctx: int = 1000):
    """prefill 走 DeepEP normal：kernel 自己占 SM，DeepGEMM 只能用剩下的。"""
    m, hw = DSV3, HW

    def C(n, sms):
        dense = 2 * (m.attn_params + m.expert_params * m.topk + m.shared_params) * n / hw.fp8
        attn = 2 * n * avg_ctx * 128 * (192 + 128) / hw.bf16          # MHA 形式的注意力，128 头
        return (dense + attn) * hw.sms / sms * 1e6

    def Mn(n):
        d, c = deepep_wire_bytes(m.hidden)
        # normal 模式按节点去重：一个 token 发到一个节点只走一次 RDMA，再由 NVLink 转给节点内的专家
        return n * nodes_touched * (d + c) / 50e9 * 1e6        # README normal EP32 档约 58 GB/s，打个折

    c_off, m_off = C(tokens, hw.sms), Mn(tokens)
    c_on, m_on = 2 * C(tokens // 2, hw.sms - deepep_sms), 2 * Mn(tokens // 2)
    off, on = c_off + m_off, max(c_on, m_on)
    print(f'DeepSeek-V3 prefill，EP{ep}（4 节点），每卡 {tokens} token，平均注意力上下文 {avg_ctx}')
    print(f'  DeepEP normal 占 {deepep_sms} 个 SM；一个 token 平均落到 {nodes_touched:.0f} 个节点'
          '（eplb_sim.py --scenario locality 的分层策略约 2.9）\n')
    print(f'  关 TBO：计算 {c_off / 1e3:.1f} ms + 通信 {m_off / 1e3:.1f} ms = {off / 1e3:.1f} ms / 层'
          '（normal 模式同步等通信，两者串行）')
    print(f'  开 TBO：计算只剩 {hw.sms - deepep_sms} 个 SM，两半计算 {c_on / 1e3:.1f} ms，'
          f'两半通信 {m_on / 1e3:.1f} ms → {on / 1e3:.1f} ms / 层，理想上限 {off / on - 1:+.0%}')
    node_off = tokens / (off * 61 / 1e6) * 8
    node_on = tokens / (on * 61 / 1e6) * 8
    print(f'\n  对账：按 61 层折算，每节点关 TBO {node_off:,.0f} tok/s、开 TBO {node_on:,.0f} tok/s；')
    print('  LMSYS 在 4 节点 EP32 实测每节点 5.0 万～5.8 万（开了 TBO）。量级对得上，开 TBO 那档偏乐观约 1.5 倍：')
    print('  模型只算了大 GEMM 和注意力，norm、RoPE、gate、量化这些小算子都没算，也假设通信被完美藏住。')
    print('  实测 TBO 收益 +27%～35%，差在 normal kernel 发送前要先算 layout、')
    print('  CPU 要等接收计数才能分配 buffer，这部分没法藏；两个 micro-batch 也不可能刚好一样长。')
    print('\n  operations_strategy.py:105 prefill 策略：deep_gemm_num_sms = 总 SM − DeepEP 的 num_sms，')
    print('  在 dispatch_a、combine_a 之后各让出一次，另一个 micro-batch 顶上去算。')
    print('  prefill 还有一个更实在的好处：一次只有半个批的激活在显存里。LMSYS 实测')
    print('  不开 TBO 时每卡 16384 token 会 OOM（只能跑 8192），开了能跑 16k。')


def scenario_stages():
    ops_decode = ['prepare_attn', 'attn_prepare', '|', 'attn_core', 'prepare_mlp', 'gate', 'select_experts', '|',
                  'dispatch_a', 'shared_experts', '|', 'dispatch_b', 'experts', 'combine_a', '|', 'combine_b',
                  '|', 'output', 'postprocess']
    stages, cur = [], []
    for op in ops_decode:
        if op == '|':
            stages.append(cur)
            cur = []
        else:
            cur.append(op)
    stages.append(cur)
    print('decode 策略（operations_strategy.py:136），| 是 YieldOperation，一层被切成 6 段：')
    for i, s in enumerate(stages):
        print(f'  段 {i}: {" → ".join(s)}')
    print('\n两个 micro-batch 轮流执行，B 落后 A 两段（tbo_delta_stages=2）：')
    n = len(stages)
    rows = []
    for t in range(n + 2):
        a = f'A.{t}' if t < n else '—'
        b = f'B.{t - 2}' if 0 <= t - 2 < n else '—'
        rows.append((t, a, b))
    for t, a, b in rows:
        print(f'  时刻 {t}: 先跑 {a:<5} 再跑 {b:<5}')
    print('\n  dispatch_a 发出后 A 让出，B 在算的时候 A 的 token 在网上飞；dispatch_b 只是等 hook 收尾。')
    print('  combine 同理。共享专家塞在 dispatch_a 和 dispatch_b 之间，正好填通信的空。')
    print('  DeepSeek 官方 decode 用的是 5 段流水，把 attention 也劈成两段，思路相同。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='split', choices=['split', 'decode', 'prefill', 'stages'])
    ap.add_argument('--ctx', type=int, default=2000)
    ap.add_argument('--ep', type=int, default=72)
    a = ap.parse_args()
    if a.scenario == 'decode':
        scenario_decode(a.ctx, a.ep)
    else:
        globals()['scenario_' + a.scenario]()


if __name__ == '__main__':
    main()
