"""SGLang decode CUDA Graph 的 CPU 复刻：捕获哪些 batch size、补齐浪费多少、默认 max_bs 和显存预留怎么算、
一次 replay 省下多少启动开销、补齐的那几行为什么必须指向 slot 0、哪些 batch 会掉回 eager。

目的：不装 GPU，把「CUDA Graph 开没开、开得对不对」这件事拆成能算的几项。
  1. 捕获列表：普通、投机解码、关补齐、mini-sglang 四种生成规则，以及 attn_tp / TBO 对齐过滤；
  2. 补齐浪费：真实 bs 落在两个桶之间时要补到上面那个桶，按 bs 分布算平均浪费；
  3. 默认值：按 GPU 显存分档给 chunked_prefill_size 和 decode max_bs，再推 mem_fraction_static；
  4. 启动开销：NVIDIA 博客的 3 组实测 + 本系列 cost.py 的 kernel 数，eager vs graph 每步差多少；
  5. 捕获顺序：从大到小捕获、共享一个内存池，为什么显存只按最大那张图算；
  6. 补齐策略：padding 行的 req_pool_indices / out_cache_loc 不清零会写坏谁；
  7. 掉回 eager 的条件。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - generate_decode_cuda_graph_batch_sizes（arg_groups/cuda_graph_hook.py:500）三条分支逐项一致，
    generate_prefill_cuda_graph_batch_sizes（:480）的 token 桶一致
  - get_batch_sizes_to_capture（runner/base_cuda_graph_runner.py:65）：ceil_align 后补 num_max_requests、
    按 mul_base 过滤、截到 req_to_token_pool.size；mul_base 来自 utils/common.py:4168
  - _pad_to_bucket（:166）：bisect_left 找第一个不小于 raw 的桶
  - handle_gpu_memory_settings（arg_groups/memory_hook.py:47）的显存分档和 reserved_mem 公式，
    reserve_for_graph_mb（:357）的 decode max_bs * 2 MB、prefill 每个桶 8 MB
  - PaddingPolicy（cuda_graph_buffer_registry.py:105）的 ZERO / FILL_SENTINEL / FOREACH_COPY 语义
  - mini-sglang engine/graph.py:49 的捕获列表
不对齐的部分：
  - bs 分布是合成的；显存按 nvidia-smi 报的 MiB 数手填
  - 不模拟 DP attention、PD 分离、MLA、VLM 对预留的额外调整，不模拟 post-capture KV sizing（默认关）
  - 启动开销用 cost.py 的假设值，要在自己机器上用 nsys 看 cudaLaunchKernel / cudaGraphLaunch 校准

用法：
    python3 cuda_graph_sim.py                      # buckets：四种捕获列表 + 对齐过滤
    python3 cuda_graph_sim.py --scenario waste     # 三种 bs 分布下的补齐浪费
    python3 cuda_graph_sim.py --scenario defaults  # 各 GPU 的默认 max_bs、chunked_prefill、mem_fraction_static
    python3 cuda_graph_sim.py --scenario launch    # 启动开销：NVIDIA 实测 + 每步省多少
    python3 cuda_graph_sim.py --scenario capture   # 捕获顺序和内存池
    python3 cuda_graph_sim.py --scenario padding   # 补齐行的几种策略
    python3 cuda_graph_sim.py --scenario fallback  # 哪些 batch 会掉回 eager
"""
import argparse
import bisect
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cost import PRESETS, gpu_ms, launch_ms, n_kernels, with_  # noqa: E402


def ceil_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ============================================================================= 捕获列表
def decode_bs(max_bs: int, mode: str = 'normal'):
    """arg_groups/cuda_graph_hook.py:500 generate_decode_cuda_graph_batch_sizes。"""
    if mode == 'nopad':
        bs = list(range(1, max_bs + 1))
    elif mode == 'normal':
        bs = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, max_bs + 1, 32))
    elif mode == 'spec':
        bs = (list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 65, 4)) + list(range(72, 257, 8))
              + list(range(272, max_bs + 1, 16)))
    elif mode == 'mini':
        # mini-sglang engine/graph.py:49；max_bs 按空闲显存 80GB 分 256 / 160
        return [1, 2, 4] + list(range(8, max_bs + 1, 8))
    else:
        raise ValueError(mode)
    bs = [b for b in bs if b <= max_bs]
    if max_bs not in bs:
        bs.append(max_bs)
    return bs


def prefill_tokens(max_tokens: int):
    """:480 generate_prefill_cuda_graph_batch_sizes，prefill 的 bs 是 token 数。"""
    s = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32)) + list(range(576, 1025, 64))
         + list(range(1280, 4097, 256)) + list(range(4608, max_tokens + 1, 512)))
    return [x for x in s if x <= max_tokens]


def to_capture(bs, pool_size: int, attn_tp: int = 1, gathered: bool = False, tbo: bool = False, width: int = 1):
    """runner/base_cuda_graph_runner.py:65 get_batch_sizes_to_capture。"""
    mul = 1
    if tbo:
        mul *= 2
    if gathered:
        mul *= attn_tp
    align_w = 1 if tbo else width
    nmax = ceil_align(pool_size, mul)
    bs = list(bs)
    if max(bs) > nmax:
        bs.append(nmax)
    bs = [b for b in bs if b * align_w % mul == 0 and b <= nmax]
    return sorted(set(bs)), mul


def pad_to(raw: int, buckets) -> int:
    assert raw <= buckets[-1], f'size {raw} exceeds max captured bucket {buckets[-1]}'
    return buckets[bisect.bisect_left(buckets, raw)]


def fmt(xs, head=10, tail=4):
    if len(xs) <= head + tail:
        return str(xs)
    return '[' + ', '.join(map(str, xs[:head])) + ', …, ' + ', '.join(map(str, xs[-tail:])) + f']（{len(xs)} 个）'


def scenario_buckets():
    print('== 捕获列表：四种生成规则 ==')
    for mode, mb in (('normal', 256), ('normal', 512), ('spec', 256), ('nopad', 64), ('mini', 256)):
        bs = decode_bs(mb, mode)
        gaps = [b - a for a, b in zip([0] + bs, bs)]
        print(f'  {mode:<7} max_bs={mb:<4} {fmt(bs)}，最大间距 {max(gaps)}')
    print('  投机解码的列表在小 bs 段更密：每条请求一步要验证多个 token，bs 补一行就多算 num_draft_tokens 行')
    print()
    print('== 对齐过滤：get_batch_sizes_to_capture ==')
    base = decode_bs(256)
    cases = [('默认（TP 下 attention 不需要 gathered buffer）', dict()),
             ('DP attention，attn_tp=4，需要 gathered buffer', dict(attn_tp=4, gathered=True)),
             ('two-batch overlap', dict(tbo=True)),
             ('TBO + gathered attn_tp=4', dict(tbo=True, attn_tp=4, gathered=True))]
    for name, kw in cases:
        cap, mul = to_capture(base, 4096, **kw)
        print(f'  {name}：mul_base={mul}，{fmt(cap, 8, 2)}')
    cap, _ = to_capture(base, 50)
    print(f'  --max-running-requests 50（req_to_token_pool 只有 50 行）：{fmt(cap, 12, 2)}，50 被补进来')
    cap, mul = to_capture(base, 50, attn_tp=4, gathered=True)
    print(f'  同上再加 attn_tp=4：50 向上对齐到 {ceil_align(50, mul)}，{fmt(cap, 12, 2)}')
    print()
    pt = prefill_tokens(8192)
    print(f'== prefill 的 token 桶（chunked_prefill_size=8192）：{fmt(pt, 8, 4)}')


# ============================================================================= waste
def scenario_waste():
    print('== 补齐浪费：真实 bs → 桶 ==')
    rng = random.Random(9)
    dists = {
        '均匀 1..256': [rng.randint(1, 256) for _ in range(20000)],
        '低并发 1..24': [rng.randint(1, 24) for _ in range(20000)],
        '对数均匀 1..256': [max(1, min(256, int(2 ** rng.uniform(0, 8)))) for _ in range(20000)],
    }
    print(f'  {"分布":<16}{"规则":<8}{"桶数":>5}{"平均补齐行":>11}{"浪费比例":>9}{"1..256 最坏":>11}')
    for dn, xs in dists.items():
        for mode in ('normal', 'spec', 'mini', 'nopad'):
            bs = decode_bs(256, mode)
            pads = [pad_to(x, bs) - x for x in xs]
            worst = max(range(1, 257), key=lambda r: (pad_to(r, bs) - r) / pad_to(r, bs))
            print(f'  {dn:<16}{mode:<8}{len(bs):>5}{sum(pads) / len(xs):>11.2f}'
                  f'{sum(pads) / sum(pad_to(x, bs) for x in xs) * 100:>8.1f}%'
                  f'{worst:>7}→{pad_to(worst, bs)}')
    c = PRESETS['qwen3-8b']
    print()
    print(f'  补齐的代价不是「多算的行数」本身：{c.model} decode 是带宽瓶颈，'
          f'bs 9 和 bs 12 的 GPU 时间 {gpu_ms(c, 9):.2f} vs {gpu_ms(c, 12):.2f} ms，几乎一样')
    print('  真正花钱的是：多出来的行也要读 KV（ZERO 策略下读的是 slot 0，长度 1，很便宜）、'
          '以及每多一个桶就多一张图的捕获时间和显存')
    print('  --disable-cuda-graph-padding 捕获 1..max_bs 每一个，max_bs=256 就是 256 张图，启动慢、占显存，只在 bs 很小的场景用')


# ============================================================================= defaults
GPU_MIB = [  # nvidia-smi --query-gpu=memory.total 的典型读数（MiB）
    ('T4', 15360), ('RTX 4090', 24564), ('L40S', 46068), ('A100-40G', 40960), ('A100-80G', 81920),
    ('H100-SXM', 81559), ('H20', 97871), ('H200', 143771), ('B200', 183359),
]


def tier(mib: float, tp: int):
    """memory_hook.py:89-170。返回 (chunked_prefill_size, decode max_bs)。"""
    if mib < 20 * 1024:
        return 2048, 8
    if mib < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if mib < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if mib < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def mem_fraction(mib: float, tp: int = 1, pp: int = 1, max_running=None, graph=True, chunk=None):
    """memory_hook.py:281-328 的非 post-capture 分支（非 MLA、非 VLM、非 PD、无 DP attention）。"""
    c0, max_bs = tier(mib, tp)
    chunk = chunk or c0
    if max_running is not None:
        max_bs = min(max_bs, max_running)
    act_tokens = max(chunk, 2048)
    parts = dict(base=512, act=act_tokens * 1.5, par=tp * pp / 8 * 1024,
                 decode_graph=max_bs * 2 if graph else 0,
                 prefill_graph=len(prefill_tokens(chunk)) * 8 if graph else 0)
    reserved = sum(parts.values())
    floor = mib > 60 * 1024 and reserved < 10 * 1024
    if mib > 60 * 1024:
        reserved = max(reserved, 10 * 1024)
    return round((mib - reserved) / mib, 3), reserved, parts, floor, chunk, max_bs


def scenario_defaults():
    print('== 默认值：按显存分档（memory_hook.py handle_gpu_memory_settings）==')
    print(f'  {"GPU":<10}{"MiB":>8}{"TP":>4}{"chunk":>7}{"max_bs":>8}{"预留 MiB":>10}{"其中图":>8}{"10G 保底":>9}{"mem_frac":>9}')
    for name, mib in GPU_MIB:
        for tp in (1, 8):
            if tp == 8 and mib < 40000:
                continue
            f, res, parts, floor, chunk, mb = mem_fraction(mib, tp)
            g = parts['decode_graph'] + parts['prefill_graph']
            print(f'  {name:<10}{mib:>8}{tp:>4}{chunk:>7}{mb:>8}{res:>10.0f}{g:>8.0f}{"是" if floor else "":>9}{f:>9.3f}')
    print('  读法：')
    print('  - 预留的大头是激活：chunked_prefill_size × 1.5 MB，chunk 8192 就是 12 GB；decode 图按 max_bs × 2 MB 算，256 才 0.5 GB；')
    print('  - 60 GB 以上的卡有 10 GB 保底，默认 chunk 下预留已经超过它，保底不起作用；')
    print('  - max_bs 默认会被 --max-running-requests 截断（memory_hook.py:182），反过来手动给 --cuda-graph-bs-decode 时 max_bs = max(列表)。')
    f1 = mem_fraction(24564, 1)[0]
    f2 = mem_fraction(24564, 1, max_running=8)[0]
    print(f'  4090 上 --max-running-requests 8：max_bs 48 → 8，mem_fraction_static {f1} → {f2}，只多 80 MB 给 KV 池')
    f3 = mem_fraction(81559, 1, chunk=2048)
    print(f'  H100 上手动 --chunked-prefill-size 2048：算出来的预留 {sum(f3[2].values()):.0f} MB < 10 GB，'
          f'保底生效，mem_fraction_static 停在 {f3[0]}，再调小 max_bs 也不变')


# ============================================================================= launch
def scenario_launch():
    print('== 启动开销 ==')
    print('  NVIDIA 技术博客（V100，20 个 2.9 µs 的小 kernel × 1000 步）：')
    rows = [('每个 kernel 后同步', 9.6), ('每步同步一次，kernel 连发', 3.8), ('CUDA Graph', 3.4)]
    for n, us in rows:
        print(f'    {n:<20} 每 kernel {us:.1f} µs，其中纯计算 2.9 µs，开销 {us - 2.9:.1f} µs')
    print('  这组数说明两件事：连发能把启动藏在前一个 kernel 后面；graph 再省掉的是每个 kernel 的 CPU 发射。'
          'kernel 越短、越多，graph 越赚')
    print()
    print(f'  {"preset":<15}{"kernel 数":>9}{"eager 启动":>11}{"graph 启动":>11}{"bs=1 GPU":>10}{"bs=1 瓶颈":>10}{"bs=64 GPU":>10}{"bs=64 瓶颈":>11}')
    for name in ('qwen3-0.6b', 'qwen3-8b', 'llama-70b-tp4', 'llama-70b-tp8'):
        c = PRESETS[name]
        e, g = launch_ms(c, False), launch_ms(c, True)
        g1, g64 = gpu_ms(c, 1), gpu_ms(c, 64)
        b1 = 'CPU' if e > g1 else 'GPU'
        b64 = 'CPU' if e > g64 else 'GPU'
        print(f'  {name:<15}{n_kernels(c):>9}{e:>10.2f}ms{g:>10.2f}ms{g1:>9.2f}ms{b1:>10}{g64:>9.2f}ms{b64:>11}')
    c = PRESETS['qwen3-0.6b']
    for op in (3.0, 6.0, 10.0):
        cc = with_(c, op_us=op)
        print(f'  qwen3-0.6b 每 kernel CPU 开销 {op:>4.1f} µs：eager 启动 {launch_ms(cc, False):.2f} ms，'
              f'bs=1 GPU {gpu_ms(cc, 1):.2f} ms → eager 下 GPU 利用率最多 {min(1, gpu_ms(cc, 1) / launch_ms(cc, False)) * 100:.0f}%')
    print('  PyTorch 博客给的口径一样：CPU 开销固定，batch 越小越吃亏；它的 Mask R-CNN 训练 1.70x、BERT 1.12x 也是这个规律')


# ============================================================================= capture
def scenario_capture():
    print('== 捕获顺序和内存池 ==')
    bs = decode_bs(256)
    act = {b: 0.6 + 0.012 * b for b in bs}            # 每张图中间激活的峰值（MB，假设值）
    # 每张图单独一个池：显存是各自峰值之和
    sep = sum(act.values())
    # 共享池、从小到大：池子随最大图增长，早先的小块被复用的机会少，碎片约按 30% 计（假设值）
    asc = max(act.values()) * 1.3
    # 共享池、从大到小：第一张就是最大的，后面的小图都能塞进它释放的块
    desc = max(act.values())
    print(f'  {len(bs)} 张图，单张中间激活 {min(act.values()):.1f}～{max(act.values()):.1f} MB（玩具数）')
    print(f'    各用各的池        {sep:8.1f} MB')
    print(f'    共享池，从小到大  {asc:8.1f} MB（碎片按 30% 估）')
    print(f'    共享池，从大到小  {desc:8.1f} MB')
    print('  decode_cuda_graph_runner.py:1358 按 reversed(capture_bs) 捕获，注释「Reverse so cuda graphs share memory better」；')
    print('  full_cuda_graph_backend.py 的 capture_session 用全局图内存池，每个 bs 捕获前跑两遍 warmup，'
          '把 kernel 加载、cuBLAS workspace 这类一次性开销付在图外')
    print('  mini-sglang 同样倒序捕获，并把第一张图的 pool 传给后面所有图（engine/graph.py）')
    print()
    print('  捕获日志怎么读（model_runner_components/cuda_graph_setup.py:757 / :786）：')
    print('    Capture target decode cuda graph begin. backend=full, num_tokens_per_req=1, avail mem=XX GB')
    print('    Capture target decode cuda graph end. bs=[…], elapsed=YY s, mem usage=ZZ GB, avail mem=WW GB.')
    print('  mem usage 就是图实际吃掉的显存，拿它和 memory_hook 的预留比：超了就要降 mem_fraction_static 或 max_bs')


# ============================================================================= padding
def scenario_padding():
    print('== 补齐行的几种策略（cuda_graph_buffer_registry.py PaddingPolicy）==')
    print('  真实 bs=5，桶 8，后 3 行是补齐行。上一次 replay 是 bs=8 的满批，静态缓冲尾部留着旧值')
    prev = dict(req_pool_indices=[3, 7, 9, 12, 14, 21, 22, 23], out_cache_loc=[40, 71, 95, 120, 141, 210, 219, 230],
                seq_lens=[10, 33, 51, 18, 9, 400, 260, 77], input_ids=[11, 12, 13, 14, 15, 16, 17, 18])
    live = dict(req_pool_indices=[3, 7, 9, 21, 14], out_cache_loc=[41, 72, 96, 501, 142],
                seq_lens=[11, 34, 52, 2, 10], input_ids=[31, 32, 33, 34, 35])
    # 请求 21 是刚接手 slot 21 的新请求；上一批的第 6 行也是 slot 21（旧请求已结束，槽位刚复用）
    pol = dict(req_pool_indices='ZERO', out_cache_loc='ZERO', seq_lens='FILL_SENTINEL(1)', input_ids='FOREACH_COPY')
    for k, p in pol.items():
        tail_new = {'ZERO': [0, 0, 0], 'FILL_SENTINEL(1)': [1, 1, 1], 'FOREACH_COPY': prev[k][5:]}[p]
        print(f'    {k:<17}{p:<17} 真实 {live[k]}  尾部 {tail_new}（不清就是 {prev[k][5:]}）')
    print()
    print('  如果 out_cache_loc 尾部不清：补齐行把新 KV 写进槽位 210、219、230。这些槽位上一步属于别的请求，')
    print('  现在可能已经还给分配器又分给了新请求，于是新请求的 KV 被写坏，没有任何报错，只是输出变怪')
    print('  如果 req_pool_indices 尾部不清：补齐行的 next token 写进 FutureMap 的 slot 21、22、23，'
          'slot 21 正被新请求使用，它下一步的输入被换掉（sim_overlap.py --scenario relay 复现了这一条）')
    print('  如果 seq_lens 尾部不清：注意力按 400、260、77 的长度去读 KV，白读几百个 token；'
          '填 1 则只读 slot 0 的一行')
    print('  positions 也是 ZERO：旧尾巴曾让 flashinfer 的 verify plan 越界（IMA，sglang issue #24361）')
    print('  input_ids 用 FOREACH_COPY 不清尾巴是安全的：补齐行的输出写进 slot 0，没人读')


# ============================================================================= fallback
def scenario_fallback():
    print('== 哪些 batch 会掉回 eager（decode_cuda_graph_runner.py:877 can_run_graph）==')
    cap = decode_bs(256)
    cases = [
        ('bs=37', dict(bs=37)),
        ('bs=300，超过 max_bs 256', dict(bs=300)),
        ('带 replace_embeds（请求自带 embedding 覆盖）', dict(bs=8, embeds=True)),
        ('投机解码，batch 的每请求 token 数 ≠ 捕获宽度', dict(bs=8, width=3, captured=4)),
        ('关补齐，bs=37 但只捕获了 1..32', dict(bs=37, nopad=True)),
    ]
    for name, kw in cases:
        bs = kw['bs']
        if kw.get('embeds'):
            why = '否：replace_embeds 按请求动态变化，图里没有这条路径'
        elif kw.get('width') and kw['width'] != kw['captured']:
            why = '否：uniform-width replay 不变式不成立'
        elif kw.get('nopad'):
            why = '否：关补齐后必须有完全相等的那张图'
        elif bs > cap[-1]:
            why = '否：超过最大桶'
        else:
            why = f'是：补到 {pad_to(bs, cap)}'
        print(f'  {name:<42} 能 replay？{why}')
    print('  掉回 eager 时日志 Decode batch 那一行的 cuda graph 字段是 False。线上看到它频繁为 False，'
          '先查 #running-req 是否经常超过 max_bs')
    for name in ('qwen3-0.6b', 'qwen3-8b'):
        c = PRESETS[name]
        e, g1 = launch_ms(c, False), gpu_ms(c, 1)
        hid = '，大部分被 GPU 前向盖住' if g1 > e else f' GPU 只要 {g1:.2f} ms，多出的启动直接加到 ITL 上'
        print(f'  {name} 掉回 eager：每步 CPU 发 kernel {e:.2f} ms，bs=1 时{hid}')


# ============================================================================= main
SCENARIOS = {
    'buckets': scenario_buckets, 'waste': scenario_waste, 'defaults': scenario_defaults, 'launch': scenario_launch,
    'capture': scenario_capture, 'padding': scenario_padding, 'fallback': scenario_fallback,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='buckets')
    args = ap.parse_args()
    SCENARIOS[args.scenario]()


if __name__ == '__main__':
    main()
