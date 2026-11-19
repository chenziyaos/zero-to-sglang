"""调参前先算账：显存分档、mem_fraction_static、max_running_requests、准入松紧、chunk、CUDA Graph 上限，各动一下会换来什么。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  arg_groups/memory_hook.py:89-170、:281-330   显存分档、reserved_mem、mem_fraction_static（复用 ../09 cuda_graph_sim.mem_fraction）
  arg_groups/memory_hook.py:357                reserve_for_graph_mb：decode max_bs x 2 MB，prefill 每个桶 8 MB
  mem_cache/kv_cache_configurator.py:3119      resolve_max_num_reqs：clamp(tokens / context x 512, 2048, 4096)，
                                               不超过 tokens // 2，手动给的值被压低时打 reduced 警告
  mem_cache/memory_pool.py                     ReqToTokenPool：int32 [(max_running + 1), context + 4]
  managers/schedule_policy.py:795、:1405       准入：在跑请求按 min(剩余输出, 4096) x new_token_ratio 预留（复用 ../05 调度器）
  managers/scheduler.py:4591                   retract 之后 new_token_ratio 被拉回
  managers/scheduler.py:1272                   available_gpu_mem 是图捕获之后量的
  ../06、../09                                 chunk 与 mixed 的时间线、eager 与 graph 的步长
时间和显存都是模型推算，用来比方向和量级；绝对值以自己机器的日志为准。

  python3 tune_sim.py                          # tier：各 GPU 的默认 chunk、max_bs、预留、mem_fraction_static
  python3 tune_sim.py --scenario frac          # frac 每 +0.01 换多少 KV，启动日志的 available_gpu_mem 落在哪一档
  python3 tune_sim.py --scenario running       # max_running_requests 的默认值、被压低的条件、req_to_token 的显存
  python3 tune_sim.py --scenario admit         # 早停负载下 max_tokens 和 conservativeness 怎么影响 usage 与吞吐
  python3 tune_sim.py --scenario admit --dump  # 同上，再把一次运行的调度日志写到 sim_admit.log，交给 parse_log.py
  python3 tune_sim.py --scenario chunk         # chunk 大小与 mixed：KV 容量、长请求 TTFT、ITL 尾部
  python3 tune_sim.py --scenario graph         # bs 超过 decode 图上限时掉回 eager 的代价，以及调大上限的显存账
"""
import argparse
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
for p in ('', '01-first-server', '02-ledger', '03-bench', '05-scheduler', '06-chunked-prefill', '09-overlap-cudagraph'):
    sys.path.insert(0, os.path.join(ROOT, p))

from common.specs import MODELS, SMI_MIB  # noqa: E402
from sim_ledger import ledger, cell_bytes  # noqa: E402
import cuda_graph_sim as G  # noqa: E402
from cost import PRESETS, gpu_ms, launch_ms, with_  # noqa: E402
from sim_overlap import simulate as step_sim  # noqa: E402

GiB = 1 << 30
AVAIL_BANDS = ((5.0, '偏紧，长 prompt、logprob 请求易 OOM'), (8.0, '合适'), (10.0, '略松，可再 +0.01'),
               (math.inf, '偏松，KV 池少拿了'))


def band(avail):
    for hi, name in AVAIL_BANDS:
        if avail < hi:
            return name
    return AVAIL_BANDS[-1][1]


# ============================================================================= tier
def scenario_tier():
    print('== 默认值从哪来：按 nvidia-smi 显存分档（memory_hook.py handle_gpu_memory_settings）==')
    print(f'  {"GPU":<10}{"MiB":>8}{"TP":>4}{"chunk":>7}{"max_bs":>8}{"激活":>8}{"图":>6}{"并行":>6}'
          f'{"预留 MiB":>10}{"10G 保底":>9}{"frac":>7}')
    for name, mib in G.GPU_MIB:
        for tp in (1, 8):
            if tp == 8 and mib < 40000:
                continue
            f, res, parts, floor, chunk, mb = G.mem_fraction(mib, tp)
            g = parts['decode_graph'] + parts['prefill_graph']
            print(f'  {name:<10}{mib:>8}{tp:>4}{chunk:>7}{mb:>8}{parts["act"]:>8.0f}{g:>6.0f}{parts["par"]:>6.0f}'
                  f'{res:>10.0f}{"是" if floor else "":>9}{f:>7.3f}')
    print('  读法：预留 = 512 + 激活（chunk x 1.5 MB）+ 并行（tp x pp / 8 GB）+ 图；frac = (显存 - 预留) / 显存')
    print('  激活是大头。手动改 chunk 会连带改 frac，手动给 frac 则上面这套全部跳过')


# ============================================================================= frac
def avail_estimate(L, gpu_mib, tp=1):
    """启动日志里 available_gpu_mem 的估计：预留部分减去图实际占用（按 reserve_for_graph_mb 估）。"""
    _, _, parts, _, _, _ = G.mem_fraction(gpu_mib, tp)
    graph_gib = (parts['decode_graph'] + parts['prefill_graph']) / 1024
    return L['slack'] - graph_gib


def scenario_frac(model='qwen3-8b', gpu='h100-sxm', tp=1):
    mib = SMI_MIB[gpu]
    base = ledger(model, gpu, tp)
    print(f'== mem_fraction_static：{MODELS[model].name} on {tp} x {gpu}，自动值 {base["auto"]} ==')
    print(f'  {"frac":>6}{"KV 池 GiB":>10}{"KV token":>11}{"比自动值":>9}{"预计 avail GB":>14}  档位')
    rows = []
    for f in [base['auto']] + [round(x / 100, 2) for x in range(84, 94)]:
        L = ledger(model, gpu, tp, mem_frac=f)
        av = avail_estimate(L, mib, tp)
        rows.append((f, L, av))
        d = L['tokens'] / base['tokens'] - 1
        print(f'  {f:>6.3f}{L["rest"]:>10.2f}{L["tokens"]:>11,}{d:>+9.1%}{av:>14.2f}  {band(av)}')
    step = ledger(model, gpu, tp, mem_frac=0.89)['tokens'] - ledger(model, gpu, tp, mem_frac=0.88)['tokens']
    cell = cell_bytes(MODELS[model], tp, 'bf16', 0)[0]
    print(f'  每 +0.01 多 {step:,} token（{base["pre"] * 0.01:.2f} GiB / {cell / 1024:.0f} KB 每 token）')
    ok = [f for f, _, av in rows if 5.0 <= av < 8.0]
    if ok:
        print(f'  落在 5 - 8 GB 的 frac：{ok[0]:.2f} - {ok[-1]:.2f}。从自动值往上调到 {ok[0]:.2f} 起步，'
              f'压测稳定再 +0.01，出现 OOM 退回上一档')
    print('  预计 avail = 加载前空闲 x (1 - frac) - 图实际占用。日志里的 available_gpu_mem 是图捕获之后量的，')
    print('  和这里同一个口径；差出来的部分是 NCCL buffer、cuBLAS workspace 这类本模型没算的东西')
    return rows


# ============================================================================= running
def resolve_max_num_reqs(tokens, context, requested=None):
    """kv_cache_configurator.py:3119。返回 (值, 是否打 reduced 警告)。"""
    est = max(min(int(tokens / context * 512), 4096), 2048)
    if requested is not None:
        v = min(requested, tokens // 2)
        return v, v < requested
    return min(est, tokens // 2), False


def req_to_token_gib(running, context):
    return (running + 1) * (context + 4) * 4 / GiB


def scenario_running():
    print('== max_running_requests：默认值和被压低的条件（kv_cache_configurator.py resolve_max_num_reqs）==')
    print(f'  {"场景":<28}{"KV token":>10}{"context":>9}{"手动给":>8}{"结果":>7}{"reduced 警告":>13}{"req_to_token":>13}')
    cases = [('qwen3-8b / H100 默认', 'qwen3-8b', 'h100-sxm', None, None),
             ('qwen3-8b / H100 给 1024', 'qwen3-8b', 'h100-sxm', None, 1024),
             ('qwen3-8b / 4090 默认', 'qwen3-8b', 'rtx4090', None, None),
             ('qwen3-8b / 4090 给 4096', 'qwen3-8b', 'rtx4090', None, 4096),
             ('llama-8b / H100 默认（128k）', 'llama3.1-8b', 'h100-sxm', None, None),
             ('小池子 6000 token 给 4096', 'qwen3-8b', 'h100-sxm', 6000, 4096)]
    for name, mk, gk, tok, req in cases:
        L = ledger(mk, gk)
        tokens = tok or L['tokens']
        ctx = MODELS[mk].context
        v, warn = resolve_max_num_reqs(tokens, ctx, req)
        print(f'  {name:<28}{tokens:>10,}{ctx:>9}{req if req else "-":>8}{v:>7}{"是" if warn else "":>13}'
              f'{req_to_token_gib(v, ctx):>11.2f} G')
    print('  读法：')
    print('  - 默认值几乎总是 2048 - 4096，它只是并发上限，真正卡并发的是 KV 池和准入；')
    print('  - 手动给的值只会被压到 KV token 的一半，压低时日志打 max_running_requests was reduced；')
    print('  - req_to_token 表按 (max_running + 1) x (context + 4) 个 int32 开，128k 上下文、4096 并发就是 2 GB，')
    print('    这块显存在 frac 留出的余量里，调大 max_running 等于从激活余量里拿')
    print('  - 什么时候该手动调小：要压 ITL 上限、或者 decode 显存不够反复 retract；调大只在日志里 #running-req 长期顶格时有用')


# ============================================================================= admit
def _sched():
    import sim_scheduler as S
    return S


def ns(**kw):
    d = dict(model='qwen3-8b', gpu='h100-sxm', max_total_tokens=None, chunked_prefill_size=None,
             max_prefill_tokens=16384, max_running_requests=None, schedule_policy='fcfs',
             schedule_conservativeness=1.0, enable_priority_scheduling=False, disable_priority_preemption=False,
             priority_scheduling_preemption_threshold=10, log=False, log_head=0)
    d.update(kw)
    return argparse.Namespace(**d)


def early_eos(num=800, inp=1500, lo=50, hi=400, max_new=None, seed=0):
    """全部 t=0 到达；输入 [inp/2, inp]，实际输出 [lo, hi] 后 EOS，max_new 是客户端给的 max_tokens（None 表示没给）。"""
    S = _sched()
    rng = random.Random(seed)
    reqs = []
    for i in range(num):
        n = rng.randint(inp // 2, inp)
        target = rng.randint(lo, hi)
        reqs.append(S.Req(i, [rng.randrange(1, 9_000_000) for _ in range(n)], max_new or (1 << 30), target))
    return reqs


ADMIT_CAP = 80000


def admit_rows():
    S = _sched()
    rows = []
    for mn in (512, 4096, None):
        for c in (0.3, 1.0):
            s, m = S.run_one(ns(max_total_tokens=ADMIT_CAP, schedule_conservativeness=c), early_eos(max_new=mn),
                             show=False)
            rows.append((mn, c, s, m))
    return rows


def retract_rows():
    S = _sched()
    rows = []
    for c in (0.3, 1.0, 1.3):
        reqs = S.make_random(400, 1024, 1024, 0.5, math.inf, 0)
        s, m = S.run_one(ns(max_total_tokens=120000, schedule_conservativeness=c), reqs, show=False)
        rows.append((c, s, m))
    return rows


def scenario_admit(dump=False):
    print(f'== 准入松紧（早停负载）：800 条请求 t=0 到达，输入 750 - 1500，实际输出 50 - 400 后 EOS，KV 池 {ADMIT_CAP} token ==')
    print(f'  {"max_tokens":>11}{"conserv":>8}{"decode usage":>13}{"峰值 running":>12}{"总时长 s":>9}{"输出 tok/s":>10}'
          f'{"TTFT 均值 s":>11}{"retract":>8}')
    rows = admit_rows()
    for mn, c, s, m in rows:
        print(f'  {mn if mn else "未设置":>11}{c:>8}{m["usage"]:>13.3f}{m["max_running"]:>12}{m["time"]:>9.2f}'
              f'{m["out_tps"]:>10.0f}{m["ttft_mean"]:>11.2f}{m["retract_events"]:>8}')
    print('  读法：调度器按 min(max_tokens, 4096) x new_token_ratio 给每条在跑请求留输出空间。')
    print('  max_tokens 远大于实际输出时，留出来的空间一直空着，usage 上不去，新请求却进不来；')
    print('  conservativeness 0.3 把 ratio 初值从 0.7 压到 0.21，追回一部分；把 max_tokens 设准追回得更多。没设 max_tokens 时按 4096 截断估，和 4096 几乎一样')
    print()
    print('== 准入太松（长输出负载）：400 条，输入 512 - 1024，max_tokens 1024，实际输出 512 - 1024，KV 池 120000 ==')
    print(f'  {"conserv":>8}{"retract 次":>11}{"退回请求":>9}{"重算 token":>11}{"decode usage":>13}{"输出 tok/s":>10}')
    for c, s, m in retract_rows():
        print(f'  {c:>8}{m["retract_events"]:>11}{m["retracted"]:>9}{m["reproc"]:>11,}{m["usage"]:>13.3f}{m["out_tps"]:>10.0f}')
    print('  读法：实际输出接近 max_tokens 时，0.3 让调度器塞进太多请求，decode 中途池子满，只能退回请求重算 prefill。')
    print('  这时该往 1.3 调。两种负载方向相反，所以 conservativeness 没有通用最优值，要看日志：usage 低且排队就往下，retract 频繁就往上')
    if dump:
        mn, c, s, m = next(r for r in rows if r[0] == 4096 and r[1] == 1.0)   # 调参前的样子
        path = os.path.join(HERE, 'sim_admit.log')
        with open(path, 'w') as fh:
            fh.write('# tune_sim.py --scenario admit --dump 生成：05 调度器模拟，max_tokens=4096、conservativeness=1.0。\n')
            fh.write(f"server_args={{'schedule_conservativeness': {c}, 'schedule_policy': 'fcfs', "
                     f"'max_running_requests': None, 'chunked_prefill_size': {m['chunk']}}}\n")
            fh.write('\n'.join(s.lines) + '\n')
        print(f'  已写 {os.path.basename(path)}（{len(s.lines)} 行），用 python3 parse_log.py sim_admit.log 读')


# ============================================================================= chunk
def chunk_rows(model='qwen3-8b', gpu='h100-sxm'):
    import sim_chunked as C
    rows = []
    for ch in (2048, 4096, 8192, 16384):
        kt = C.kv_tokens(model, gpu, ch)
        for mixed in (False, True):
            sim, cap = C.build(model, gpu, ch, mixed)
            reqs = sim.run(C.wl_one())
            t1 = max(r.end for r in reqs if r.kind == 'short')
            m = C.metrics(reqs, sim, (3.0, t1))
            rows.append((ch, mixed, kt, m))
    return rows


def scenario_chunk():
    print('== chunked_prefill_size：64 路 decode 中途插进一条 32k prompt（qwen3-8b / H100，复用 ../06）==')
    print(f'  {"chunk":>6}{"mixed":>6}{"frac":>7}{"KV token":>10}{"长 TTFT ms":>11}{"ITL p99":>9}{"ITL max":>9}{"块数":>5}')
    rows = chunk_rows()
    for ch, mixed, kt, m in rows:
        print(f'  {ch:>6}{"开" if mixed else "关":>6}{kt["frac"]:>7.3f}{kt["tokens"]:>10,}{m["long_ttft"]:>11.0f}'
              f'{m["itl_p99"]:>9.1f}{m["itl_max"]:>9.1f}{m["long_chunks"]:>5.0f}')
    print('  读法：')
    print('  - mixed 关：各块背靠背跑，decode 停顿约等于整条 prompt 的 prefill（ITL max 约 1.5 s），chunk 大小几乎不改；')
    print('  - mixed 开：decode 搭在每块上，最长停顿约等于一块的耗时，chunk 越小 ITL max 越小，但每步都背着 prefill，p99 抬高；')
    print('  - chunk 同时决定激活预留：H100 上 8192 → 2048 因为 10 GB 保底多出约 7% KV，16384 则少约 24%；')
    print('  - Prefill out of memory 时官方建议先降到 4096 或 2048，这是最便宜的一招')


# ============================================================================= graph
def graph_rows():
    out = []
    for name in ('qwen3-0.6b', 'qwen3-8b', 'llama-70b-tp8'):
        for op in (6.0, 15.0):
            c = with_(PRESETS[name], op_us=op)
            for bs in (64, 384):
                e = step_sim(c, bs, 60, True, False)['step']
                g = step_sim(c, bs, 60, True, True)['step']
                out.append((name, op, bs, launch_ms(c, False), gpu_ms(c, bs), e, g))
    return out


def scenario_graph():
    print('== decode 图上限：bs 超过 --cuda-graph-max-bs-decode 就走 eager（overlap 开，复用 ../09 的时间线）==')
    print(f'  {"preset":<15}{"每 kernel µs":>12}{"bs":>5}{"eager 发射":>10}{"GPU 前向":>9}{"eager 步长":>10}{"graph 步长":>10}{"慢多少":>8}')
    for name, op, bs, la, g1, e, g in graph_rows():
        print(f'  {name:<15}{op:>12.0f}{bs:>5}{la:>9.2f}ms{g1:>8.2f}ms{e:>9.2f}ms{g:>9.2f}ms{round((e / g - 1) * 100) + 0:>7d}%')
    print('  读法：overlap 下 CPU 发射和 GPU 前向并行，步长约等于两者取大。')
    print('  bs 大、模型大、TP 小时 GPU 前向更长，eager 发射被盖住；TP 8 把前向切薄，或 CPU 慢（每 kernel 15 µs）时发射露出来')
    print()
    print('== 调大上限的显存账（reserve_for_graph_mb：decode 每 bs 2 MB）==')
    m = MODELS['qwen3-8b']
    cell = cell_bytes(m, 1, 'bf16', 0)[0]
    for top in (320, 384, 512):
        extra = (top - 256) * 2
        print(f'  qwen3-8b / H100：256 → {top}，多预留 {extra} MB，约 {int(extra * 2 ** 20 // cell):,} 个 KV token'
              f'（池子共 {ledger("qwen3-8b", "h100-sxm")["tokens"]:,}）')
    print('  上限按日志里 cuda graph=False 那些行的 #running-req p90 取，向上取到 32 的倍数；超过 256 后桶间距 16，补齐浪费也不大')
    print('  大 TP 上官方建议可以给到 512 或 768，同时把 frac 降一点')


# ============================================================================= main
SCENARIOS = dict(tier=scenario_tier, frac=scenario_frac, running=scenario_running, admit=scenario_admit,
                 chunk=scenario_chunk, graph=scenario_graph)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='tier')
    ap.add_argument('--dump', action='store_true', help='admit 场景：把一次运行的调度日志写到 sim_admit.log')
    a = ap.parse_args()
    if a.scenario == 'admit':
        scenario_admit(a.dump)
    else:
        SCENARIOS[a.scenario]()


if __name__ == '__main__':
    main()
