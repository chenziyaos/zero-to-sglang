"""SGLang overlap 调度的 CPU 复刻：把一步 decode 拆成「调度 → 发 kernel → GPU 前向 → D2H → 结果处理」，
用事件时间线比较普通循环和 overlap 循环，再把 overlap 带来的几个副作用一个个复现出来。

目的：不装 GPU，看清下面几件事。
  1. 普通循环里 GPU 每一步都要等 CPU 调度和结果处理，空转比例有多大；overlap 把这两段藏到下一步前向后面；
  2. 收益什么时候大：模型越小、TP 越大、batch 越小，GPU 一步越短，CPU 开销占比越高；
     eager 模式下 kernel 启动本身就比 GPU 慢，这时 overlap 救不了，要靠 CUDA Graph；
  3. overlap 让调度器提前一步做决定，于是：
     - 下一步的 input_ids 不能从 CPU 拿，只能在 GPU 上从 FutureMap 读（relay）；
     - 已经在这一步结束的请求还会被排进下一步，要么白算一行，要么提前剔掉（skip_finishing_decode）；
     - 同一条请求可能被释放两次，要有幂等保护；
     - 调度流写共享缓冲区要等上一步前向读完（WAR barrier）；
  4. 连续 prefill 时关 overlap 能让首批 TTFT 早一点，代价是吞吐。

对齐 sglang 主干 8fabf5e 的部分（文件:行号见 README）：
  - event_loop_normal / event_loop_overlap（scheduler.py:2028 / :2064）的执行顺序：
    overlap 里先 run_batch(N+1)，再 pop_and_process(N)，结果处理里 copy_done.synchronize() 才阻塞
  - is_disable_overlap_for_batch（:2139）：连续 extend 只在 SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP 时关
  - _reqs_finishing_inflight / update_running_batch（:4509 / :4524）：只剔「按长度必然结束」的请求；
    KV 不够且有在途结束的请求时，这一步不跑 decode，而不是 retract
  - FutureMap（overlap_utils.py:281）：按 req_pool_idx 索引的 GPU 缓冲，slot 0 是 padding 行
  - _apply_war_barrier（scheduler.py:2015）：有 read-done 事件就等事件，否则 wait_stream 整个前向
  - mini-sglang scheduler.py:138 的「overlap 可能让请求被释放两次，第二次跳过」
不对齐的部分：
  - 时间来自 cost.py 的粗估模型，CPU 各项是假设值，要用 nsys 校准
  - 一步只模拟一个 batch，不模拟 chunked prefill 的预算、grammar、投机解码的 delay_sample
  - kernel 时长按 GPU 总时长均分，真实的 attention、GEMM、all-reduce 长短不一

用法：
    python3 sim_overlap.py                         # timeline：普通 vs overlap 的前 4 步时间线
    python3 sim_overlap.py --scenario sweep        # 模型 × batch × eager/graph 的步长、GPU 空转、加速比
    python3 sim_overlap.py --scenario prefill      # 连续 prefill：关 overlap 换首批 TTFT
    python3 sim_overlap.py --scenario finishing    # skip_finishing_decode、KV 不够时跳一步、重复释放
    python3 sim_overlap.py --scenario relay        # FutureMap：从 CPU 拿 input_ids 会错一步；padding 行写哪
    python3 sim_overlap.py --scenario streams      # WAR barrier：粗粒度 vs 事件，空泡多大
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cost import PRESETS, Cost, gpu_ms, launch_ms, n_kernels, proc_ms, sched_ms, with_  # noqa: E402


# ============================================================================= 时间线模型
def forward_on_gpu(c: Cost, bs: int, t_cpu: float, gpu_free: float, graph: bool, g_ms: float = None):
    """CPU 从 t_cpu 开始发 kernel。返回 (CPU 发完的时刻, GPU 开始, GPU 结束, GPU 实际忙的时长)。

    eager：逐个 kernel 发，第 k 个 kernel 最早在它被发出来之后开始，也要等前一个 kernel 结束；
    graph：CPU 花 graph_us 拷静态输入并 cudaGraphLaunch，之后 GPU 一口气跑完。
    """
    g = gpu_ms(c, bs) if g_ms is None else g_ms
    if graph:
        cpu_end = t_cpu + launch_ms(c, True)
        start = max(cpu_end, gpu_free)
        return cpu_end, start, start + g, g
    n = n_kernels(c)
    op, dk = c.op_us / 1e3, g / n
    end = gpu_free
    start = None
    for k in range(n):
        issued = t_cpu + (k + 1) * op
        s = max(issued, end)
        start = s if start is None else start
        end = s + dk
    return t_cpu + n * op, start, end, g


def simulate(c: Cost, bs: int, steps: int, overlap: bool, graph: bool, trace: bool = False):
    """跑 steps 步 decode，返回总时长、GPU 忙碌比例、平均步长、吞吐，以及可选的事件列表。

    普通循环：sched → launch → 等 D2H → proc，下一步才开始调度；
    overlap 循环：sched(N) → launch(N) → 等 N-1 的 D2H → proc(N-1)，和 scheduler.py:2064 的顺序一致。
    """
    t, gpu_free, busy = 0.0, 0.0, 0.0
    ev = []
    pending = None   # overlap 下还没处理的上一批：(d2h 完成时刻, step)
    deliver = []     # 每个 token 对用户可见的时刻
    gpu_first = None
    for s in range(steps):
        t0 = t
        t += sched_ms(c, bs)
        ev.append(('cpu', 'S', t0, t, s))
        t_launch = t
        t, gs, ge, g = forward_on_gpu(c, bs, t, gpu_free, graph)
        ev.append(('cpu', 'L', t_launch, t, s))
        ev.append(('gpu', 'F', gs, ge, s))
        gpu_first = gs if gpu_first is None else gpu_first
        busy += g
        gpu_free = ge
        d2h_done = ge + c.d2h_ms
        if not overlap:
            if d2h_done > t:
                ev.append(('cpu', 'W', t, d2h_done, s))
            t = max(t, d2h_done)
            t1 = t
            t += proc_ms(c, bs)
            ev.append(('cpu', 'P', t1, t, s))
            deliver.append(t)
        else:
            if pending is not None:
                pd, ps = pending
                if pd > t:
                    ev.append(('cpu', 'W', t, pd, ps))
                t = max(t, pd)
                t1 = t
                t += proc_ms(c, bs)
                ev.append(('cpu', 'P', t1, t, ps))
                deliver.append(t)
            pending = (d2h_done, s)
    if overlap and pending is not None:
        pd, ps = pending
        t = max(t, pd)
        t1 = t
        t += proc_ms(c, bs)
        ev.append(('cpu', 'P', t1, t, ps))
        deliver.append(t)
    # 首尾各去掉 3 步：开头有预热，overlap 的最后一步没有下一步要发，处理得更早
    k = 3 if len(deliver) > 8 else 0
    step = (deliver[-1 - k] - deliver[k]) / (len(deliver) - 1 - 2 * k)
    span = gpu_free - gpu_first
    return dict(total=t, step=step, tput=bs / step * 1e3, gpu_busy=busy / span if span else 1.0,
                first_token=deliver[0], events=ev if trace else None)


def draw(ev, upto: float, width: int = 96):
    """把事件画成两条 ASCII 泳道。"""
    scale = width / upto
    lanes = {'cpu': [' '] * width, 'gpu': ['.'] * width}
    for lane, ch, a, b, s in ev:
        if a >= upto:
            continue
        i0, i1 = int(a * scale), max(int(a * scale) + 1, int(min(b, upto) * scale))
        mark = ch if lane == 'cpu' else str(s % 10)
        for i in range(i0, min(i1, width)):
            lanes[lane][i] = mark
    return ''.join(lanes['cpu']), ''.join(lanes['gpu'])


# ============================================================================= timeline
def scenario_timeline():
    c = PRESETS['qwen3-8b']
    bs = 32
    print(f'== 时间线：{c.model} TP{c.tp} {c.gpu}，bs={bs}，CUDA Graph 开 ==')
    print(f'  一步的账（cost.py 的估算）：GPU 前向 {gpu_ms(c, bs):.2f} ms，调度 {sched_ms(c, bs):.2f} ms，'
          f'graph replay {launch_ms(c, True):.2f} ms，D2H {c.d2h_ms:.2f} ms，结果处理 {proc_ms(c, bs):.2f} ms')
    rs = {}
    for ov in (False, True):
        r = simulate(c, bs, 12, ov, True, trace=True)
        rs[ov] = r
    upto = 4.2 * rs[False]['step']
    for ov in (False, True):
        r = rs[ov]
        cpu, gpu = draw(r['events'], upto)
        name = 'overlap' if ov else '普通循环'
        print(f'\n  {name}：稳态步长 {r["step"]:.2f} ms，GPU 忙碌 {r["gpu_busy"] * 100:.1f}%，'
              f'吞吐 {r["tput"]:.0f} tok/s')
        print(f'    CPU |{cpu}|')
        print(f'    GPU |{gpu}|')
    print('\n  图例：S 调度  L 发 kernel / graph replay  W 阻塞等 D2H（copy_done.synchronize）  P 结果处理；'
          'GPU 道上的数字是第几步，点是空转')
    sp = rs[False]['step'] / rs[True]['step']
    print(f'  overlap 稳态加速 {sp:.2f}x。第 1 个 token 可见时刻：普通 {rs[False]["first_token"]:.2f} ms，'
          f'overlap {rs[True]["first_token"]:.2f} ms')
    print('  overlap 先调度并发出第 N+1 步，再处理第 N 步；这段 CPU 活只要比 GPU 前向短，就完全躲在等待里，不推迟出 token')


# ============================================================================= sweep
def idle(r):
    return max(0.0, (1 - r['gpu_busy']) * 100)


def scenario_sweep():
    print('== 扫描：模型 × batch × eager/graph，普通循环 vs overlap ==')
    print('  步长单位 ms；idle = GPU 空转比例；加速 = 普通步长 / overlap 步长')
    hdr = f'  {"preset":<15}{"bs":>5}{"mode":>7}{"GPU前向":>9}{"发kernel":>9}{"普通":>8}{"idle":>7}' \
          f'{"overlap":>9}{"idle":>7}{"加速":>7}'
    print(hdr)
    for name in ('qwen3-0.6b', 'qwen3-8b', 'llama-70b-tp4', 'llama-70b-tp8'):
        c = PRESETS[name]
        for bs in (1, 16, 64, 256):
            for graph in (False, True):
                a = simulate(c, bs, 24, False, graph)
                b = simulate(c, bs, 24, True, graph)
                mode = 'graph' if graph else 'eager'
                print(f'  {name:<15}{bs:>5}{mode:>7}{gpu_ms(c, bs):>9.2f}{launch_ms(c, graph):>9.2f}'
                      f'{a["step"]:>8.2f}{idle(a):>6.0f}%{b["step"]:>9.2f}'
                      f'{idle(b):>6.0f}%{a["step"] / b["step"]:>6.2f}x')
        print()
    print('  读法：')
    c6 = PRESETS['qwen3-0.6b']
    print(f'  - eager 下 qwen3-0.6b 每步发 {n_kernels(c6)} 个 kernel，CPU 发完要 {launch_ms(c6, False):.2f} ms，'
          f'bs 小时比 GPU 前向还长：GPU 一直在等下一个 kernel，overlap 藏得住调度和结果处理，藏不住启动；')
    c7 = PRESETS['llama-70b-tp8']
    print(f'  - 70B TP8 每层多两个 all-reduce，一步 {n_kernels(c7)} 个 kernel、eager 启动 {launch_ms(c7, False):.1f} ms，'
          '已经接近 GPU 前向，GPU 再快一点（更大 TP、FP8）就会变成启动瓶颈；')
    print('  - graph 下启动只剩一次 replay，剩下的 CPU 开销正好被 overlap 藏住，两者叠加才把 GPU 喂满；')
    print('  - overlap 的收益看「调度 + 结果处理」和 GPU 前向的比值。CPU 这两项每条请求都有一份，随 bs 线性涨；'
          'GPU 前向在带宽瓶颈区几乎不随 bs 涨，所以 8B、70B 上 bs 越大 overlap 越赚，0.6B 前向本身涨得快，比值反而下降。')
    print('  和 LMSYS v0.4 博客的结论同向：小模型、大 TP 收益最大；具体倍数取决于你机器上 CPU 那几项的真实值。')


# ============================================================================= prefill
def scenario_prefill():
    print('== 连续 prefill：SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP ==')
    print('  两批 prefill 背靠背（P1、P2），比较 P1 的 TTFT 和两批跑完的总时长。prefill 调度按每 token 0.2 µs 计'
          '（前缀匹配、分配 KV 都和 token 数成正比，假设值）；')
    print('  prefill 按 eager 发 kernel 计（主干 prefill 默认 breakable graph，在注意力处断开，启动开销介于 eager 和 full graph 之间）')
    print(f'  {"preset":<13}{"每批 token":>10}{"P1 GPU":>9}{"TTFT 开":>9}{"TTFT 关":>9}{"差":>8}{"总时长 开":>11}{"总时长 关":>11}')
    for name in ('qwen3-0.6b', 'qwen3-8b'):
        c = PRESETS[name]
        for toks in (128, 512, 2048, 8192):
            res = {}
            for disable in (False, True):
                sched = c.sched_base_ms + toks * 0.0002
                g = gpu_ms(c, 1, tokens=toks)
                t = sched
                t, _, ge1, _ = forward_on_gpu(c, 1, t, 0.0, False, g)
                t += sched                          # 规划 P2（is_disable_overlap_for_batch 之前就做完了）
                if disable:                         # 先 pop_and_process(P1)，再发 P2
                    t = max(t, ge1 + c.d2h_ms) + c.proc_base_ms
                    ttft = t
                    t, _, ge2, _ = forward_on_gpu(c, 1, t, ge1, False, g)
                else:                               # 先发 P2，再处理 P1
                    t, _, ge2, _ = forward_on_gpu(c, 1, t, ge1, False, g)
                    t = max(t, ge1 + c.d2h_ms) + c.proc_base_ms
                    ttft = t
                end = max(t, ge2 + c.d2h_ms) + c.proc_base_ms
                res[disable] = (ttft, end, g)
            (t_on, e_on, g), (t_off, e_off, _) = res[False], res[True]
            print(f'  {name:<13}{toks:>10}{g:>9.2f}{t_on:>9.2f}{t_off:>9.2f}{t_on - t_off:>8.2f}{e_on:>11.2f}{e_off:>11.2f}')
    print('  读法：overlap 开着时，P1 的结果要等 P2 的 kernel 全发完才处理。P1 的 GPU 比「发 P2」长，这段时间就藏住了，TTFT 不变；')
    print('  短 prompt 时 P1 早算完，TTFT 被 P2 的启动拖长。关掉以后先处理 P1，代价是这段时间 GPU 空着、总时长变长。')
    print('  所以主干默认不关（environ.py:697 默认 False），源码注释也只说「可能略微影响吞吐」。')


# ============================================================================= finishing
def scenario_finishing():
    print('== 在途结束的请求：skip_finishing_decode ==')
    rng = random.Random(9)
    n = 2000
    for label, p_len in (('bench_serving random（ignore_eos，全按长度结束）', 1.0),
                         ('对话负载（约 25% 撞 max_new_tokens，其余 EOS）', 0.25)):
        lens = [rng.randint(16, 512) for _ in range(n)]
        by_len = [rng.random() < p_len for _ in range(n)]
        useful = sum(lens)
        naive = n                                       # 每条请求结束后还被排进下一步一次
        skip = sum(1 for b in by_len if not b)           # 只有 EOS / stop 结束的猜不到
        print(f'  {label}：{n} 条，有效 decode {useful} 行')
        print(f'    普通循环      多算 0 行')
        print(f'    overlap 不剔  多算 {naive} 行（{naive / useful * 100:.2f}%），每条多占 1 个 KV 槽位一步')
        print(f'    overlap 剔掉  多算 {skip} 行（{skip / useful * 100:.2f}%）：next_output_finishes_by_length '
              f'只认 len(output_ids)+1 >= max_new_tokens')
    print('  多算的这一行结果在 batch_result_processor.py:1022 被跳过，多分配的槽位由 release_kv_cache 一起还。')

    print('\n  KV 不够时：retract 还是跳一步（update_running_batch，scheduler.py:4524）')
    c = PRESETS['qwen3-8b']
    bs, finishing, ctx = 128, 3, 1500
    step = simulate(c, bs, 12, True, True)['step']
    re_ms = gpu_ms(c, 1, tokens=ctx)
    print(f'    {c.model}，running {bs} 条，空闲槽位比 {bs} 少几个，但有 {finishing} 条请求本步按长度结束、各占 {ctx} 个 token')
    print(f'    retract 1 条：这条以后要重算 {ctx} token 的 prefill ≈ {re_ms:.1f} ms GPU，还要重新排队')
    print(f'    跳一步：本轮不发 decode，在途那一步结束后 {finishing * ctx} 个槽位回来，代价是剩下 {bs - finishing} 条多等'
          f' ≈ {step:.2f} ms')

    print('\n  重复释放：overlap 不剔、也没有幂等保护时')
    freed, double = set(), 0
    for rid in range(50):
        for _ in range(2):                              # 第 N 步处理时释放一次，第 N+1 步处理时又释放一次
            if rid in freed:
                double += 1
            freed.add(rid)
    print(f'    50 条请求触发 {double} 次重复 free。KV 槽位会进两次空闲表，之后被两条请求同时拿到。')
    print('    mini-sglang 在 _process_last_data 里看到请求已结束就跳过第二次 free（scheduler.py:138 的注释）；'
          'sglang 靠 finished() / is_retracted 判断跳过，并把槽位归还集中在 release_kv_cache。')


# ============================================================================= relay
def toy_next(tok: int, rid: int) -> int:
    """玩具模型：下一个 token 只取决于上一个 token 和请求编号。"""
    return (31 * tok + 7 + rid) % 1000


def scenario_relay():
    print('== FutureMap relay：overlap 下一步的 input_ids 从哪来 ==')
    reqs = {1: 11, 2: 22, 3: 33}                      # req_pool_idx -> prompt 最后一个 token
    steps = 5
    ref = {i: [] for i in reqs}
    for i, t in reqs.items():
        for _ in range(steps):
            t = toy_next(t, i)
            ref[i].append(t)

    # 正确做法：GPU 上的 output_tokens_buf[req_pool_idx]，同一条 forward 流上先写后读
    buf = [0] * 8
    got = {i: [] for i in reqs}
    for i, t in reqs.items():
        buf[i] = t
    for _ in range(steps):
        idx = list(reqs)
        inp = [buf[i] for i in idx]                    # resolve_forward_inputs：output_tokens_buf[req_pool_indices]
        out = [toy_next(x, i) for x, i in zip(inp, idx)]
        for i, o in zip(idx, out):
            buf[i] = o                                 # publish / stash 写回同一个 slot
            got[i].append(o)
    print(f'  GPU relay：{got[1]} …  和普通循环一致：{got == ref}')

    # 错误做法：从 CPU 侧最近处理过的结果拿。overlap 下第 N 步的结果此时还没处理，拿到的是 N-1
    cpu_last = dict(reqs)
    processed = dict(reqs)
    bad = {i: [] for i in reqs}
    pending = None
    for _ in range(steps):
        inp = {i: processed[i] for i in reqs}
        out = {i: toy_next(inp[i], i) for i in reqs}
        if pending is not None:
            processed.update(pending)                  # 处理的是上一步的结果
        pending = out
        for i in reqs:
            bad[i].append(out[i])
            cpu_last[i] = out[i]
    first = next(k for k, (a, b) in enumerate(zip(ref[1], bad[1])) if a != b)
    print(f'  CPU 拿：  {bad[1]} …  从第 {first + 1} 步开始错，每个 token 都重复喂了一遍旧输入')
    print('  所以 run_batch 里把 batch.input_ids 置 None（scheduler.py 注释「Next-iter input_ids relayed via future_map」），'
          '避免有人误用 CPU 上的旧值')

    # padding 行：bs=3 补到 4
    print('\n  CUDA Graph 补齐：真实 bs=3，graph 按 4 捕获，第 4 行写到哪个 slot')
    for policy in ('ZERO', 'stale'):
        buf = [0] * 8
        live = {1: 101, 2: 202, 5: 505}                # slot 5 是刚复用的新请求 E
        for i, t in live.items():
            buf[i] = t
        # 上一次 replay 的第 4 行属于已结束的 D：input_ids 尾部还是 D 的旧 token（FOREACH_COPY 不清尾巴），
        # 如果 req_pool_indices 尾部也不清，它还指着 D 的 slot 5，而 slot 5 刚分给了 E
        pad_idx = 0 if policy == 'ZERO' else 5
        idx = list(live) + [pad_idx]
        inp = [buf[i] for i in live] + [404]
        outs = [toy_next(x, i) for x, i in zip(inp, idx)]
        for i, o in zip(idx, outs):                    # 同一个 kernel 里并行写，后写的行覆盖先写的
            buf[i] = o
        e_ok = buf[5] == toy_next(505, 5)
        print(f'    req_pool_indices 尾部 {policy:<5}：padding 行写 slot {pad_idx}，E 的下一个输入 '
              f'{"正确" if e_ok else f"被覆盖成 {buf[5]}（应为 {toy_next(505, 5)}）"}')
    print('  cuda_graph_buffer_registry.py 把 req_pool_indices、out_cache_loc 的尾部设成 ZERO：'
          'slot 0 是 KV 池和 FutureMap 共同的 padding 行（memory_pool.py:400 多留一行），写进去没人读')


# ============================================================================= streams
def scenario_streams():
    print('== 三条流和 WAR barrier ==')
    print('  schedule_stream：调度器为下一步做 H2D、写 req_to_token / out_cache_loc 这类共享缓冲')
    print('  forward_stream：前向，开头先 wait_stream(schedule_stream)；前向开头一小段读这些共享缓冲（准备注意力元数据）')
    print('  copy_stream：等前向写完 next_token_ids 再 D2H，记 copy_done；前向流从不等它，它是叶子')
    print()
    print('  下一步的 schedule_stream 写什么时候能开始：')
    print(f'  {"preset":<15}{"bs":>5}{"前向":>8}{"元数据读":>9}{"prep":>7}{"无屏障冲突":>11}{"粗粒度空泡":>11}{"事件空泡":>9}')
    for name in ('qwen3-0.6b', 'qwen3-8b', 'llama-70b-tp8'):
        c = PRESETS[name]
        for bs in (8, 128):
            g = gpu_ms(c, bs)
            read = min(0.03, 0.05 * g)                 # 元数据读取只占前向开头一小段（假设值）
            prep = 0.02 + 0.0002 * bs                  # 下一步在调度流上的 GPU 活（假设值）
            # CPU 发出下一步 prep 的时刻：本步 graph replay 之后再调度一遍，通常远早于前向结束
            issue = launch_ms(c, True) + sched_ms(c, bs)
            hazard = issue < read                      # 无屏障时，prep 可能在读完之前写
            coarse = prep                              # prep 排在本步前向之后，下一步前向要等它
            fine = max(0.0, max(issue, read) + prep - g)
            print(f'  {name:<15}{bs:>5}{g:>8.2f}{read:>9.3f}{prep:>7.3f}{"可能" if hazard else "否":>11}'
                  f'{coarse / (g + coarse) * 100:>10.1f}%{fine / (g + fine) * 100:>8.1f}%')
    print('  粗粒度：schedule_stream.wait_stream(forward_stream)，下一步的 prep 排在整个前向之后，每步多一个 prep 的空泡；')
    print('  事件：等 decode graph 里「元数据准备完」那个点记的事件（_publish_read_done），prep 和本步前向重叠，空泡没了。')
    print('  上表「无屏障冲突」一列全是「否」：正常情况下 CPU 发得没那么快。但只要有一次 CPU 比 GPU 快（小模型、GPU 被别的流拖住），'
          '就会写坏还在读的元数据，所以屏障必须在；SGLANG_FORCE_COARSE_WAR_BARRIER=1 可以退回粗粒度排查问题。')


# ============================================================================= main
SCENARIOS = {
    'timeline': scenario_timeline, 'sweep': scenario_sweep, 'prefill': scenario_prefill,
    'finishing': scenario_finishing, 'relay': scenario_relay, 'streams': scenario_streams,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='timeline')
    args = ap.parse_args()
    SCENARIOS[args.scenario]()


if __name__ == '__main__':
    main()
