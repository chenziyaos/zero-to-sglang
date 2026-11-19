"""读 SGLang 服务日志，统计稳态指标，按规则给出该动哪个参数。

只用标准库。日志格式按 sglang 8fabf5e，路径相对 python/sglang/srt/：
  entrypoints/engine.py:311                    server_args={...}（resolved_dict 的 dict repr）
  managers/scheduler.py:1272                   max_total_num_tokens=…, available_gpu_mem=X GB（图捕获之后量的）
  model_executor/model_runner.py:1303          Load weight begin. avail mem=X GB
  model_executor/model_runner_components/cuda_graph_setup.py
                                               Capture … decode CUDA graph end. bs=[…, 256]
  managers/scheduler_components/metrics_reporter.py:754、:963-1040
                                               Prefill batch / Decode batch，字段由逗号分隔
  managers/scheduler_components/pool_stats_observer.py:104-150
                                               token usage / full token usage + swa token usage / mamba usage
  managers/scheduler.py:4591                   KV cache pool is full. Retract requests. …
  managers/schedule_batch.py:3512、:3536        retract_decode: aborted request / aborted last request
  mem_cache/kv_cache_configurator.py:3167      max_running_requests was reduced from …
  managers/scheduler.py:2763、:2792             Capping max_new_tokens / Request prompt exceeds the KV memory budget
  mem_cache/allocation.py:239、:613             Prefill out of memory / Decode out of memory
诊断阈值取自 docs/docs/advanced_features/hyperparameter_tuning.mdx，个别阈值是本目录的经验值，见 RULES。

  python3 parse_log.py sample_before.log           # 调参前的日志：指标 + 诊断 + 建议参数
  python3 parse_log.py sample_after.log            # 调参后
  python3 parse_log.py sample_before.log --json    # 机器可读
  python3 parse_log.py sim_admit.log               # tune_sim.py --dump 出来的模拟日志也能读（[t=…s] 前缀）
"""
import argparse
import ast
import json
import math
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------- 阈值
RULES = dict(
    usage_good=0.90,         # 官方：token usage > 0.9 算用满
    queue_low=100,           # 官方：#queue-req 健康区间 100 - 2000
    queue_high=2000,
    idle_share=0.8,          # 本目录：八成以上 decode 行 queue=0，判为客户端没压满
    loose_share=0.5,         # 本目录：一半以上 decode 行 usage < 0.9 且 queue > 0，判为准入太保守
    retract_per_min=1.0,     # 官方：大约每分钟一次以内可以接受
    avail_ok=(5.0, 8.0),     # 官方：available_gpu_mem 5 - 8 GB 合适
    avail_high=10.0,         # 官方：10 - 20 GB 偏高
    avail_target=7.0,        # 本目录：往上调 frac 时把余量收到 7 GB 附近，留 1 GB 给波动
    graph_false_share=0.2,   # 本目录：两成以上 decode 行掉回 eager 才值得调
    lpm_queue=128,           # schedule_policy.py:314 WAITING_QUEUE_PREFIX_MATCH_MAX
    cap_share=0.3,           # 本目录：三成以上 decode 行顶着 max_running_requests
    accept_len_low=1.6,      # 本目录：平均接受长度低于这个，投机解码多半不赚
)
CUDA_CTX_GIB = 0.6           # 加载权重之前 CUDA context 吃掉的显存，和 01/02 的口径一致

# ---------------------------------------------------------------- 正则
TS_REAL = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\] ')
TS_SIM = re.compile(r'^\[t=\s*([\d.]+)s\] ')
STARTUP = re.compile(r'max_total_num_tokens=(\d+), chunked_prefill_size=(-?\d+), max_prefill_tokens=(\d+), '
                     r'max_running_requests=(\d+), context_len=(\d+), available_(?:gpu|cpu)_mem=([\d.]+) GB')
ARGS_DICT = re.compile(r'server_args=(\{.*\})\s*$')
ARGS_REPR = re.compile(r'server_args=ServerArgs\((.*)\)\s*$')
LOAD_BEGIN = re.compile(r'Load weight begin\. avail mem=([\d.]+) GB')
DECODE_CAPTURE = re.compile(r'Capture \w+ decode \w+ graph end\. bs=\[([^\]]*)\]')
BATCH = re.compile(r'(Prefill|Decode) batch(?: \[\d+\])?, (.*)$')
RETRACT = re.compile(r'KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+), #new_tokens_gained: (-?\d+)'
                     r'(?:, #mamba_num_gained: -?\d+)?, #new_token_ratio: ([\d.]+) -> ([\d.]+)')
ABORT_LAST = re.compile(r'retract_decode: aborted last request (\S+) due to OOM')
ABORT_REQ = re.compile(r'retract_decode: aborted request (\S+): (.*)$')
REDUCED = re.compile(r'max_running_requests was reduced from the requested (\d+) to (\d+)')
CAPPING = re.compile(r'Capping max_new_tokens of request (\S+) to SGLANG_MAX_NEW_TOKENS_LIMIT=(\d+) '
                     r'\(requested: (\w+)\)')
KV_BUDGET = re.compile(r'Request prompt exceeds the KV memory budget: input_len=(\d+)')
OOM = re.compile(r'(Prefill|Decode) out of memory\.')

ARG_KEYS = ('mem_fraction_static', 'max_running_requests', 'max_queued_requests', 'chunked_prefill_size',
            'max_prefill_tokens', 'schedule_policy', 'schedule_conservativeness', 'cuda_graph_max_bs_decode',
            'tp_size', 'dp_size', 'enable_mixed_chunk', 'context_length', 'page_size', 'stream_interval',
            'speculative_algorithm', 'kv_cache_dtype', 'disable_radix_cache', 'swa_full_tokens_ratio',
            'model_path')


# ---------------------------------------------------------------- 解析
def parse_ts(line):
    """返回 (秒, 去掉前缀的正文)。没有时间前缀时秒为 None。"""
    m = TS_SIM.match(line)
    if m:
        return float(m.group(1)), line[m.end():]
    m = TS_REAL.match(line)
    if m:
        t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        return t.timestamp(), line[m.end():]
    return None, line


def _num(v):
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_fields(body):
    """'a: 1, token usage: 0.5, cuda graph: True' -> dict。metrics_reporter 的字段之间是 ', '，值里不含逗号。"""
    out = {}
    for part in body.split(', '):
        if ': ' not in part:
            continue
        k, v = part.split(': ', 1)
        out[k.strip()] = _num(v.strip())
    return out


def parse_server_args(body):
    """两种写法都认：8fabf5e 的 dict repr，以及旧版的 ServerArgs(k=v, …)。只保留 ARG_KEYS。"""
    m = ARGS_DICT.search(body)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            return {k: d[k] for k in ARG_KEYS if k in d}
        except (ValueError, SyntaxError):
            text = m.group(1)
            out = {}
            for k in ARG_KEYS:
                mm = re.search(r"'%s': ('[^']*'|[^,}]+)" % k, text)
                if mm:
                    out[k] = _literal(mm.group(1))
            return out
    m = ARGS_REPR.search(body)
    if m:
        out = {}
        for mm in re.finditer(r"(\w+)=('(?:[^'\\]|\\.)*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\}|[^,]+)", m.group(1)):
            if mm.group(1) in ARG_KEYS:
                out[mm.group(1)] = _literal(mm.group(2))
        return out
    return None


def _literal(s):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def usage_of(f):
    """返回 (全注意力池 usage, SWA 池 usage 或 None, mamba usage 或 None)。"""
    full = f.get('token usage', f.get('full token usage'))
    return full, f.get('swa token usage'), f.get('mamba usage')


def graph_of(f):
    for k, v in f.items():
        if k.endswith(' graph') and isinstance(v, bool):
            return v
    return None


def parse(lines):
    R = dict(args={}, startup=None, load_begin=None, graph_max_bs=None, prefill=[], decode=[], retract=[],
             abort_last=[], abort_req=[], reduced=None, capping=[], kv_budget=[], oom=[], t0=None, t1=None)
    for raw in lines:
        line = raw.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        t, body = parse_ts(line)
        m = ARGS_DICT.search(body) or ARGS_REPR.search(body)
        if m:
            R['args'] = parse_server_args(body) or {}
            continue
        m = STARTUP.search(body)
        if m:
            R['startup'] = dict(max_total=int(m.group(1)), chunk=int(m.group(2)), max_prefill=int(m.group(3)),
                                max_running=int(m.group(4)), context_len=int(m.group(5)),
                                avail=float(m.group(6)))
            continue
        m = LOAD_BEGIN.search(body)
        if m and R['load_begin'] is None:
            R['load_begin'] = float(m.group(1))
            continue
        m = DECODE_CAPTURE.search(body)
        if m:
            nums = re.findall(r'\d+', m.group(1))
            if nums:
                R['graph_max_bs'] = int(nums[-1])
            continue
        m = BATCH.search(body)
        if m:
            f = parse_fields(m.group(2))
            f['t'] = t
            (R['prefill'] if m.group(1) == 'Prefill' else R['decode']).append(f)
            if t is not None:
                R['t0'] = t if R['t0'] is None else R['t0']
                R['t1'] = t
            continue
        m = RETRACT.search(body)
        if m:
            R['retract'].append(dict(t=t, n=int(m.group(1)), gained=int(m.group(2)),
                                     ratio_old=float(m.group(3)), ratio_new=float(m.group(4))))
            continue
        m = ABORT_LAST.search(body)
        if m:
            R['abort_last'].append(m.group(1))
            continue
        m = ABORT_REQ.search(body)
        if m:
            R['abort_req'].append((m.group(1), m.group(2)))
            continue
        m = REDUCED.search(body)
        if m:
            R['reduced'] = (int(m.group(1)), int(m.group(2)))
            continue
        m = CAPPING.search(body)
        if m:
            R['capping'].append((m.group(1), int(m.group(2)), m.group(3)))
            continue
        m = KV_BUDGET.search(body)
        if m:
            R['kv_budget'].append(int(m.group(1)))
            continue
        m = OOM.search(body)
        if m:
            R['oom'].append(m.group(1))
    return R


# ---------------------------------------------------------------- 统计
def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(R):
    """只用 #running-req > 0 的 decode 行，空转的行不算稳态。"""
    rows = [f for f in R['decode'] if f.get('#running-req', 0) > 0]
    S = dict(n=len(rows), n_prefill=len(R['prefill']))
    if not rows:
        return S
    run = [f['#running-req'] for f in rows]
    q = [f.get('#queue-req', 0) for f in rows]
    use = [usage_of(f)[0] for f in rows]
    swa = [usage_of(f)[1] for f in rows if usage_of(f)[1] is not None]
    mamba = [usage_of(f)[2] for f in rows if usage_of(f)[2] is not None]
    g = [graph_of(f) for f in rows]
    tput = [f.get('gen throughput (token/s)') for f in rows if f.get('gen throughput (token/s)') is not None]
    acc = [f['accept len'] for f in rows if 'accept len' in f]
    S.update(running_p50=pct(run, 50), running_p90=pct(run, 90), running_max=max(run),
             queue_p50=pct(q, 50), queue_max=max(q), queue_zero=sum(1 for x in q if x == 0) / len(q),
             queue_over_lpm=sum(1 for x in q if x > RULES['lpm_queue']) / len(q),
             usage_p50=pct([u for u in use if u is not None], 50),
             usage_max=max(u for u in use if u is not None) if any(u is not None for u in use) else None,
             loose=sum(1 for u, x in zip(use, q) if u is not None and u < RULES['usage_good'] and x > 0) / len(q),
             swa_p50=pct(swa, 50), mamba_p50=pct(mamba, 50),
             tput_p50=pct(tput, 50), accept_p50=pct(acc, 50))
    known = [(r, x) for r, x in zip(run, g) if x is not None]
    if known:
        false = [r for r, x in known if not x]
        true = [r for r, x in known if x]
        S.update(graph_false=len(false) / len(known), graph_true_max=max(true) if true else None,
                 graph_false_min=min(false) if false else None, graph_false_p90=pct(false, 90) if false else None)
    st = R['startup']
    if st:
        S['at_cap'] = sum(1 for r in run if r >= st['max_running']) / len(run)
    span = (R['t1'] - R['t0']) if R['t0'] is not None and R['t1'] is not None else 0
    S['span_s'] = span
    S['retracts'] = len(R['retract'])
    S['retracted_reqs'] = sum(r['n'] for r in R['retract'])
    S['retract_per_min'] = len(R['retract']) / (span / 60) if span > 0 else (math.inf if R['retract'] else 0.0)
    return S


# ---------------------------------------------------------------- 诊断
def finding(code, level, msg, flags=None):
    return dict(code=code, level=level, msg=msg, flags=flags or {})


def gpu_total_gib(R, gpu_gib=None):
    if gpu_gib:
        return gpu_gib
    if R['load_begin'] is not None:
        return R['load_begin'] + CUDA_CTX_GIB
    return None


def floor2(x):
    return math.floor(x * 100 + 1e-9) / 100


def diagnose(R, S=None, gpu_gib=None):
    S = S or stats(R)
    A, st = R['args'], R['startup']
    out = []

    # 1. 报错类，先看
    if R['oom']:
        kinds = sorted(set(R['oom']))
        if 'Prefill' in kinds:
            cur = (st or {}).get('chunk') or A.get('chunked_prefill_size')
            nxt = 4096 if not cur or cur > 4096 else 2048
            out.append(finding('OOM_PREFILL', 'error',
                               f'出现 Prefill out of memory {R["oom"].count("Prefill")} 次：一次 prefill 的激活超出余量。'
                               f'先把 chunk 从 {cur} 降到 {nxt}，长 prompt 会多切几块',
                               {'--chunked-prefill-size': nxt}))
        if 'Decode' in kinds:
            out.append(finding('OOM_DECODE', 'error',
                               f'出现 Decode out of memory {R["oom"].count("Decode")} 次：先降 --max-running-requests，'
                               f'不行再把 --mem-fraction-static 降到 0.8 或 0.7',
                               {'--max-running-requests': '降低', '--mem-fraction-static': 0.8}))
    if R['abort_last'] or R['abort_req']:
        out.append(finding('RETRACT_ABORT', 'error',
                           f'retract 之后仍有请求被中止（last request {len(R["abort_last"])} 条，'
                           f'无 KV 备份 {len(R["abort_req"])} 条），客户端会看到 500/503。'
                           f'池子对这批请求太小：降 --max-running-requests 或给请求设准 max_tokens',
                           {'--max-running-requests': '降低'}))
    if R['kv_budget']:
        mx = max(R['kv_budget'])
        cap = st['max_total'] if st else None
        tail = f'，池子只有 {cap} token' if cap else ''
        out.append(finding('KV_BUDGET', 'error',
                           f'{len(R["kv_budget"])} 条请求的 prompt 放不进空池（最长 input_len={mx}{tail}），直接 400。'
                           f'要么调小 --context-length 在入口拒掉，要么加 KV（frac、FP8 KV、加 TP）',
                           {'--context-length': '≤ 池容量'}))

    # 2. 启动时的显存余量
    if st:
        avail, total = st['avail'], gpu_total_gib(R, gpu_gib)
        frac = A.get('mem_fraction_static')
        lo, hi = RULES['avail_ok']
        if avail >= RULES['avail_high'] and frac and total:
            target = floor2(frac + (avail - RULES['avail_target']) / total)
            gain = (target - frac) * total
            out.append(finding('AVAIL_HIGH', 'warn',
                               f'启动时 available_gpu_mem={avail:.2f} GB，官方建议 5 - 8 GB。mem_fraction_static '
                               f'{frac} -> {target} 能多给 KV 约 {gain:.1f} GB；改完按 +0.01 步进压测，到 OOM 前一档停',
                               {'--mem-fraction-static': target}))
        elif avail >= RULES['avail_high']:
            out.append(finding('AVAIL_HIGH', 'warn', f'available_gpu_mem={avail:.2f} GB 偏高，'
                               f'日志里没找到 mem_fraction_static，无法给出具体值'))
        elif avail > hi:
            out.append(finding('AVAIL_SLACK', 'info', f'available_gpu_mem={avail:.2f} GB，略高于 8 GB，'
                               f'可以再试 +0.01'))
        elif avail < lo:
            flags = {'--mem-fraction-static': floor2(frac - 0.02)} if frac else {}
            out.append(finding('AVAIL_LOW', 'warn', f'available_gpu_mem={avail:.2f} GB 低于 5 GB，长 prompt 或'
                               f'带 logprob 的请求容易 OOM，frac 往下收 0.01 - 0.02', flags))
        else:
            out.append(finding('AVAIL_OK', 'ok', f'available_gpu_mem={avail:.2f} GB，在 5 - 8 GB 内'))
    if R['reduced']:
        a, b = R['reduced']
        out.append(finding('RUNNING_REDUCED', 'info', f'max_running_requests 请求 {a}，被 KV 容量压到 {b}'
                           f'（上限是 max_total_num_tokens // 2）'))

    if not S.get('n'):
        out.append(finding('NO_DECODE', 'info', '日志里没有 #running-req > 0 的 Decode batch 行，无法判断稳态'))
        return out

    # 3. 负载是否压满
    idle = S['queue_zero'] >= RULES['idle_share']
    if idle:
        out.append(finding('QUEUE_IDLE', 'info',
                           f'{S["queue_zero"]:.0%} 的 decode 行 #queue-req=0：请求发得不够快，服务端没被压满。'
                           f'这时 usage 低说明不了参数问题，先把客户端并发提上去再调'))
    elif S['queue_p50'] > RULES['queue_high']:
        out.append(finding('QUEUE_DEEP', 'warn',
                           f'#queue-req 中位数 {S["queue_p50"]:.0f}，超过 2000：调度器每轮都要扫长队列，'
                           f'排队时延失控。加实例或在入口限流，--max-queued-requests 让超出的请求直接 503',
                           {'--max-queued-requests': int(RULES['queue_high'])}))

    # 4. 准入松紧
    cap_bound = st is not None and S.get('at_cap', 0) >= RULES['cap_share']
    cons = A.get('schedule_conservativeness', 1.0)
    rpm = S['retract_per_min']
    if cap_bound and S['loose'] >= RULES['loose_share']:
        out.append(finding('RUNNING_CAP', 'warn',
                           f'{S["at_cap"]:.0%} 的 decode 行 #running-req 顶在 max_running_requests='
                           f'{st["max_running"]}，usage 还没满：上限是它，不是准入。调大要算 req_to_token 的显存',
                           {'--max-running-requests': st['max_running'] * 2}))
    elif rpm > RULES['retract_per_min']:
        jump = [r for r in R['retract'] if r['ratio_new'] > r['ratio_old']]
        tail = (f'，ratio 被拉回 {jump[-1]["ratio_new"]:.2f}，说明输出比估计的长' if jump else '')
        nxt = 1.3 if cons < 1.3 else round(cons + 0.3, 1)
        out.append(finding('RETRACT_FREQ', 'warn',
                           f'{S["span_s"] / 60:.1f} 分钟内 retract {S["retracts"]} 次（{rpm:.1f} 次/分），'
                           f'退回 {S["retracted_reqs"]} 条请求{tail}。准入偏松，被退的请求要重算 prefill',
                           {'--schedule-conservativeness': nxt}))
    elif not idle and S['loose'] >= RULES['loose_share']:
        if cons > 0.3:
            out.append(finding('ADMIT_TIGHT', 'warn',
                               f'{S["loose"]:.0%} 的 decode 行 token usage < 0.9 且队列里有请求，usage 中位数 '
                               f'{S["usage_p50"]:.2f}：准入按 max_new_tokens 留的余量太多，多半是请求的 max_tokens '
                               f'很大而实际早早停了。conservativeness {cons} -> 0.3，或让客户端把 max_tokens 设准',
                               {'--schedule-conservativeness': 0.3}))
        else:
            out.append(finding('ADMIT_TIGHT', 'warn',
                               f'conservativeness 已是 {cons}，usage 仍只有 {S["usage_p50"]:.2f}：'
                               f'看请求的 max_tokens 是否没设（按 4096 截断估），或 SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION'))
    elif rpm > 0:
        out.append(finding('RETRACT_OK', 'ok', f'retract {S["retracts"]} 次（{rpm:.1f} 次/分），官方认为每分钟一次以内可以接受'))
    if not idle and S['usage_p50'] is not None and S['usage_p50'] >= RULES['usage_good'] and rpm <= RULES['retract_per_min']:
        out.append(finding('USAGE_OK', 'ok', f'token usage 中位数 {S["usage_p50"]:.2f}，KV 池用满了'))

    # 5. 混合池
    if S.get('swa_p50') is not None and S['usage_p50'] is not None:
        f_, s_ = S['usage_p50'], S['swa_p50']
        ratio = A.get('swa_full_tokens_ratio')
        if s_ >= RULES['usage_good'] and f_ < 0.6:
            out.append(finding('SWA_BIND', 'warn', f'SWA 池 usage {s_:.2f}、全注意力池 {f_:.2f}：SWA 池先满。'
                               f'调大 --swa-full-tokens-ratio（当前 {ratio}，上限 1.0）', {'--swa-full-tokens-ratio': '调大'}))
        elif f_ >= RULES['usage_good'] and s_ < 0.6:
            out.append(finding('FULL_BIND', 'info', f'全注意力池 usage {f_:.2f}、SWA 池 {s_:.2f}：'
                               f'SWA 池有富余，可调小 --swa-full-tokens-ratio 把显存还给全注意力池',
                               {'--swa-full-tokens-ratio': '调小'}))

    # 6. CUDA Graph
    if S.get('graph_false') is not None and S['graph_false'] >= RULES['graph_false_share']:
        top = R['graph_max_bs'] or S.get('graph_true_max')
        if top and S['graph_false_min'] is not None and S['graph_false_min'] > top:
            want = int(math.ceil(S['graph_false_p90'] / 32) * 32)
            extra_mb = (want - top) * 2
            out.append(finding('GRAPH_FALLBACK', 'warn',
                               f'{S["graph_false"]:.0%} 的 decode 行 cuda graph=False，这些行 #running-req 都大于 '
                               f'{top}（decode 图的最大 bs）：超过就走 eager。--cuda-graph-max-bs-decode 提到 {want}，'
                               f'多捕获的图约占 {extra_mb} MB，frac 相应少给一点',
                               {'--cuda-graph-max-bs-decode': want}))
        else:
            out.append(finding('GRAPH_OTHER', 'info',
                               f'{S["graph_false"]:.0%} 的 decode 行 cuda graph=False，但 bs 不超过已捕获的上限：'
                               f'查投机解码宽度、请求自带 embedding 或 --disable-cuda-graph'))

    # 7. 调度策略
    pol = A.get('schedule_policy')
    if pol in ('lpm', 'hrrn') and S['queue_over_lpm'] >= 0.5:
        out.append(finding('LPM_FALLBACK', 'info',
                           f'schedule_policy={pol}，但 {S["queue_over_lpm"]:.0%} 的时间队列超过 128 条，'
                           f'这时调度器退回 FCFS（schedule_policy.py:314），前缀排序没生效。'
                           f'前缀复用靠 router 的 cache-aware 路由更稳'))

    # 8. 其他
    if R['capping']:
        lim = R['capping'][0][1]
        out.append(finding('CAPPING', 'info', f'{len(R["capping"])} 条请求的 max_new_tokens 被 '
                           f'SGLANG_MAX_NEW_TOKENS_LIMIT={lim} 截断'))
    if S.get('accept_p50') is not None and S['accept_p50'] < RULES['accept_len_low']:
        out.append(finding('SPEC_LOW', 'info', f'投机解码平均接受长度 {S["accept_p50"]:.2f}，低于 '
                           f'{RULES["accept_len_low"]}，多半不赚，对照 SGL-13 的收益账'))
    return out


def merged_flags(fs):
    flags = {}
    for f in fs:
        for k, v in f['flags'].items():
            flags.setdefault(k, v)
    return flags


# ---------------------------------------------------------------- 输出
def fmt(v, spec='.2f'):
    return '-' if v is None else format(v, spec)


def report(R, S, fs):
    st, A = R['startup'], R['args']
    print('== 启动 ==')
    if A:
        keys = [k for k in ARG_KEYS if k in A and k != 'model_path']
        print('  ' + '，'.join(f'{k}={A[k]}' for k in keys))
    if st:
        print(f'  max_total_num_tokens={st["max_total"]}，chunk={st["chunk"]}，max_running={st["max_running"]}，'
              f'context_len={st["context_len"]}，available_gpu_mem={st["avail"]:.2f} GB'
              + (f'，decode 图最大 bs={R["graph_max_bs"]}' if R['graph_max_bs'] else ''))
    print('== 稳态（#running-req > 0 的 decode 行）==')
    if not S.get('n'):
        print('  无')
    else:
        print(f'  {S["n"]} 行 decode、{S["n_prefill"]} 行 prefill，跨度 {S["span_s"]:.0f} s')
        print(f'  #running-req p50/p90/max = {S["running_p50"]:.0f}/{S["running_p90"]:.0f}/{S["running_max"]}；'
              f'#queue-req p50/max = {S["queue_p50"]:.0f}/{S["queue_max"]}，queue=0 占 {S["queue_zero"]:.0%}')
        print(f'  token usage p50/max = {fmt(S["usage_p50"])}/{fmt(S["usage_max"])}'
              + (f'，swa usage p50 = {S["swa_p50"]:.2f}' if S.get('swa_p50') is not None else '')
              + f'；usage<0.9 且有排队占 {S["loose"]:.0%}')
        if S.get('graph_false') is not None:
            print(f'  cuda graph=False 占 {S["graph_false"]:.0%}' +
                  (f'（这些行 running ≥ {S["graph_false_min"]}）' if S['graph_false_min'] is not None else ''))
        print(f'  gen throughput p50 = {fmt(S["tput_p50"], ".0f")} token/s；retract {S["retracts"]} 次，'
              f'{S["retract_per_min"]:.2f} 次/分')
    print('== 诊断 ==')
    tag = dict(error='[错误]', warn='[调整]', info='[提示]', ok='[正常]')
    for f in fs:
        print(f'  {tag[f["level"]]} {f["code"]}：{f["msg"]}')
    flags = merged_flags(fs)
    print('== 建议改动 ==')
    if not flags:
        print('  无')
    for k, v in flags.items():
        print(f'  {k} {v}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log')
    ap.add_argument('--gpu-gib', type=float, default=None, help='单卡显存 GiB；不给就用 Load weight begin + 0.6')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    with open(a.log, encoding='utf-8', errors='replace') as fh:
        R = parse(fh)
    S = stats(R)
    fs = diagnose(R, S, a.gpu_gib)
    if a.json:
        json.dump(dict(stats=S, findings=fs, flags=merged_flags(fs)), sys.stdout, ensure_ascii=False, indent=1,
                  default=str)
        print()
    else:
        report(R, S, fs)


if __name__ == '__main__':
    main()
