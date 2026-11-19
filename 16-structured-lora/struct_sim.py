"""结构化输出：mask 有多大、填 mask 要多久、语法队列什么时候卡住、overlap 怎么排、思考预算怎么强制收尾。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  constrained/xgrammar_backend.py:66        _allocate_token_bitmask：每行 ceil(V/32) 个 int32，初值 -1，pinned 内存
  constrained/outlines_backend.py:58        allocate_vocab_mask：每行 V 个 bool，fill_(1) 后把允许的位置清零
  constrained/base_grammar_backend.py:88    fill_vocab_mask_batched：默认逐行顺序填
  constrained/base_grammar_backend.py:205   ThreadPoolExecutor()：编译线程池，默认 min(32, cpu + 4)
  constrained/base_grammar_backend.py:304   get_cached_or_future_value：命中返回 copy，不命中 submit 一个编译
  constrained/grammar_manager.py:148        process_req_with_grammar：json > regex > ebnf > structural_tag 取 key，
                                            不命中进 grammar_queue；同一个 key 并发不命中会各编各的
  constrained/grammar_manager.py:211        get_ready_grammar_requests：轮询 5 ms，等满 10000 轮判超时，
                                            TP 间 ready 取交集、failed 取并集，超时结果也写进缓存
  utils.py:89 (python/sglang/)              convert_json_schema_to_str：json.dumps，不排序 key
  constrained/torch_ops/token_filter_torch_ops.py:25  set_token_filter_torch：按 int32 字聚合后置位或清位
  speculative/spec_utils.py:628             traverse_tree：DFS 草稿树，accept / fill / rollback(1)
  managers/tp_worker.py:728                 overlap 下有语法的 batch 推迟采样，mask 在 GPU 前向期间填
  managers/scheduler.py:2139                is_disable_overlap_for_batch：投机 + 语法需要同步时关掉这一步的 overlap
  constrained/reasoner_grammar_backend.py:154-300  ReasonerGrammarObject：思考段计数、KMP 匹配 think_end、
                                            预算用完只放行下一个 think_end token、思考段屏蔽 excluded token
  utils/token_sequence_matcher.py:18        TokenSequenceMatcher：KMP 前缀函数

  python3 struct_sim.py                      # mask：各模型每行 mask 字节、H2D 量、逐行填 mask 和 GPU 前向谁长
  python3 struct_sim.py --scenario fsm       # fsm：玩具语法的 token mask、accept / rollback、草稿树 traverse_tree
  python3 struct_sim.py --scenario queue     # queue：编译线程池排队、重复编译、key 顺序导致的不命中、超时时长
  python3 struct_sim.py --scenario overlap   # overlap：同步、推迟采样、投机同步三种时间线下每步多少 ms
  python3 struct_sim.py --scenario think     # think：思考预算、多 token 的 think_end、excluded token、回滚
"""
import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '03-bench'))

from common.specs import MODELS  # noqa: E402
from sim_serve import Server  # noqa: E402

VOCAB = OrderedDict([            # 词表大小，取自各模型 config.json 的 vocab_size
    ('qwen3-8b', 151936),
    ('llama3.1-8b', 128256),
    ('deepseek-v3', 129280),
    ('gpt-oss-120b', 201088),
])
POLL_INTERVAL = 0.005            # environ.py:410 SGLANG_GRAMMAR_POLL_INTERVAL
MAX_POLL = 10000                 # environ.py:411 SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
MAX_ROLLBACK_TOKENS = 200        # xgrammar_backend.py:63
# 每行 mask 的 CPU 耗时（秒）。XGrammar 论文：JSON < 40 µs/token，XML / DSL 类 CFG < 200 µs/token；
# llguidance 自报 128k 词表约 50 µs/token。这里取上限，实际随语法和词表变
FILL_COST = OrderedDict([('JSON 40µs', 40e-6), ('CFG 200µs', 200e-6)])
M32 = 0xFFFFFFFF
H2D_GBPS = 25.0                  # pinned 内存 H2D 实测带宽的保守值，PCIe Gen5 x16 能到 50 上下


# ---------------------------------------------------------------- mask 尺寸
def mask_bytes(vocab, bs, kind='bitmask'):
    """xgrammar / llguidance 用 int32 位图，outlines 用 bool。"""
    if kind == 'bitmask':
        return bs * math.ceil(vocab / 32) * 4
    return bs * vocab


def fill_time(bs, per_row, batched_threads=1):
    """fill_vocab_mask_batched 默认逐行；llguidance 全是 GuidanceGrammar 时才走原生批量。"""
    return math.ceil(bs / batched_threads) * per_row


def scenario_mask(model='qwen3-8b', gpu='h100-sxm'):
    print('1）每行 mask 多大（字节）：')
    print(f'{"模型":<16}{"V":>8}{"位图 int32":>12}{"bool":>10}{"bs=256 位图":>14}{"bs=256 bool":>14}')
    res = {}
    for m, v in VOCAB.items():
        b, o = mask_bytes(v, 1), mask_bytes(v, 1, 'bool')
        res[m] = (b, o)
        print(f'{m:<16}{v:>8}{b:>12}{o:>10}{mask_bytes(v, 256) / 1e6:>12.2f}MB'
              f'{mask_bytes(v, 256, "bool") / 1e6:>12.2f}MB')
    h2d = mask_bytes(VOCAB[model], 256) / (H2D_GBPS * 1e9)
    print(f'   位图比 bool 小 8 倍。每步拷到 GPU 的就是这一块：{model} bs=256 是 '
          f'{mask_bytes(VOCAB[model], 256) / 1e6:.2f} MB，')
    print(f'   pinned + non_blocking 按 {H2D_GBPS:.0f} GB/s 算 {h2d * 1e3:.2f} ms，还能和前向重叠。瓶颈在 CPU 上把它填出来。')
    print()
    srv = Server(model=model, gpu=gpu)
    print(f'2）逐行填 mask 和 GPU 一步 decode 谁长（{model} / {gpu}，ctx 2000）：')
    hdr = ''.join(f'{k:>16}' for k in FILL_COST)
    print(f'{"bs":>6}{"decode ms":>12}{hdr}')
    rows = {}
    for bs in (1, 8, 32, 64, 128, 256):
        t = srv.t_decode([2000] * bs)
        fills = [fill_time(bs, c) for c in FILL_COST.values()]
        rows[bs] = (t, fills)
        cells = ''.join(f'{f * 1e3:>10.2f} ({f / t:>3.0%})' for f in fills)
        print(f'{bs:>6}{t * 1e3:>12.2f}{cells}')
    print('   括号里是填 mask 时间占这一步 decode 的比例。overlap 打开时 mask 在 GPU 前向期间填，')
    print('   比例 < 100% 就藏住了；超过 100% 的部分直接加到每步延迟上。复杂 CFG 在大 batch 下会超。')
    return res, rows


# ---------------------------------------------------------------- 玩具语法
class ToyRegex:
    """一串 (字符集, '1' 或 '+')，够表达 {"n":[0-9]+} 这类形状。状态是 (第几项, 本项已吃几个)。"""

    def __init__(self, items):
        self.items = items

    def step(self, state, ch):
        i, k = state
        while i < len(self.items):
            cs, rep = self.items[i]
            if ch in cs:
                return (i, k + 1) if rep == '+' else (i + 1, 0)
            if rep == '+' and k >= 1:
                i, k = i + 1, 0
                continue
            return None
        return None

    def walk(self, state, s):
        for ch in s:
            state = self.step(state, ch)
            if state is None:
                return None
        return state

    def is_final(self, state):
        i, k = state
        n = len(self.items)
        return i == n or (i == n - 1 and self.items[i][1] == '+' and k >= 1)


def set_token_filter(row, token_ids, is_allowed=True, reset=True):
    """torch_ops/token_filter_torch_ops.py:25：reset 时整行先置 -1（屏蔽模式）或 0（放行模式），再按字聚合置位/清位。"""
    if reset:
        for w in range(len(row)):
            row[w] = 0 if is_allowed else M32
    words = {}
    for t in token_ids:
        words[t // 32] = words.get(t // 32, 0) | (1 << (t % 32))
    for w, bits in words.items():
        row[w] = (row[w] | bits) & M32 if is_allowed else row[w] & ~bits & M32


def bit(row, t):
    return (row[t // 32] >> (t % 32)) & 1


class ToyGrammar:
    """接口照 BaseGrammarObject：accept_token / rollback / is_terminated / fill_vocab_mask。"""

    def __init__(self, regex, vocab, eos):
        self.re, self.vocab, self.eos = regex, vocab, eos
        self.hist = [(0, 0)]
        self.done = False

    @property
    def state(self):
        return self.hist[-1]

    def allowed(self):
        out = []
        for tid, s in enumerate(self.vocab):
            if tid == self.eos:
                if self.re.is_final(self.state):
                    out.append(tid)
            elif s and self.re.walk(self.state, s) is not None:
                out.append(tid)
        return out

    def fill_vocab_mask(self, mask, idx):
        set_token_filter(mask[idx], self.allowed(), is_allowed=True)

    def accept_token(self, tid):
        if tid == self.eos:
            assert self.re.is_final(self.state)
            self.done = True
            self.hist.append(self.state)
            return
        nxt = self.re.walk(self.state, self.vocab[tid])
        assert nxt is not None, f'token {self.vocab[tid]!r} 不合法'
        self.hist.append(nxt)

    def rollback(self, k):
        assert k <= MAX_ROLLBACK_TOKENS and k < len(self.hist)
        for _ in range(k):
            self.hist.pop()
        self.done = False

    def is_terminated(self):
        return self.done


DIGITS = set('0123456789')
TOY_RE = [({'{'}, '1'), ({'"'}, '1'), ({'n'}, '1'), ({'"'}, '1'), ({':'}, '1'), (DIGITS, '+'), ({'}'}, '1')]
TOY_VOCAB = ['{', '{"', '"', 'n', '":', '":1', '1', '12', '3', '}', ' ', 'a', '"n"', '}\n', '<eos>']


def new_mask(bs, vocab):
    return [[M32] * math.ceil(vocab / 32) for _ in range(bs)]


def traverse_tree(next_token, next_sibling, draft, grammar, mask, vocab=None):
    """spec_utils.py:628 原样搬：父节点 mask 里有这个 token 才 accept，往下走完 rollback(1)。"""
    visited = []

    def dfs(curr, parent_pos):
        if curr == 0:
            accepted = True
        else:
            tok = draft[curr]
            accepted = not (vocab and tok >= vocab) and bit(mask[parent_pos], tok) != 0
        if accepted:
            if curr != 0:
                grammar.accept_token(draft[curr])
            visited.append(curr)
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(mask, curr)
                if next_token[curr] != -1:
                    dfs(next_token[curr], curr)
            if curr != 0:
                grammar.rollback(1)
        if next_sibling[curr] != -1:
            dfs(next_sibling[curr], parent_pos)

    dfs(0, -1)
    return visited


def scenario_fsm():
    re_ = ToyRegex(TOY_RE)
    eos = TOY_VOCAB.index('<eos>')
    g = ToyGrammar(re_, TOY_VOCAB, eos)
    print('1）语法 {"n":[0-9]+}，词表 15 个 token。逐步 accept，每步可选的 token：')
    path = ['{"', 'n', '":1', '12', '}', '<eos>']
    trace = []
    for s in path:
        allow = [TOY_VOCAB[t] for t in g.allowed()]
        trace.append(allow)
        print(f'   状态 {str(g.state):<8} 可选 {allow}  -> 选 {s!r}')
        g.accept_token(TOY_VOCAB.index(s))
    print(f'   terminated={g.is_terminated()}。注意「":1」一个 token 跨了三个语法单元，')
    print('   mask 判的是整个 token 的字符串能不能被语法吃下，所以跨边界的 token 也能放行。')
    print()
    g.rollback(3)
    print(f'2）rollback(3) 后状态回到 {g.state}，可选 {[TOY_VOCAB[t] for t in g.allowed()]}')
    print(f'   xgrammar 最多回滚 {MAX_ROLLBACK_TOKENS} 个 token，投机解码拒掉的草稿靠它撤回。')
    print()
    # 草稿树：0 是根（已确定的最后一个 token），1..6 是草稿
    #        0
    #      /   \
    #   1:'{"'  4:'{'
    #     |       |
    #   2:'n'   5:'a'   <- 不合法
    #     |
    #   3:'":1'  6:'":'（3 的兄弟）
    T = TOY_VOCAB.index
    draft = [-1, T('{"'), T('n'), T('":1'), T('{'), T('a'), T('":')]
    next_token = [1, 2, 3, -1, 5, -1, -1]
    next_sibling = [-1, 4, -1, 6, -1, -1, -1]
    g2 = ToyGrammar(re_, TOY_VOCAB, eos)
    mask = new_mask(len(draft), len(TOY_VOCAB))
    vis = traverse_tree(next_token, next_sibling, draft, g2, mask)
    print('3）草稿树 traverse_tree：')
    for n in range(1, len(draft)):
        print(f'   节点 {n} {TOY_VOCAB[draft[n]]!r:<6} {"通过" if n in vis else "被父节点的 mask 拒掉"}')
    print(f'   遍历完语法状态回到根：{g2.state}，历史长度 {len(g2.hist)}。每个节点都填了自己的 mask 行，')
    print('   verify 时目标模型在这些行上采样，被拒的分支不会被接受。')
    return trace, vis, g2


# ---------------------------------------------------------------- 编译队列
def schema_key(schema_obj):
    """protocol.py 调 convert_json_schema_to_str，就是不带 sort_keys 的 json.dumps。"""
    return ('json', json.dumps(schema_obj))


def simulate_compile(arrivals, workers, compile_s, dedup=False):
    """arrivals：[(t, key)]。线程池 FIFO，编译完写缓存。返回每个请求拿到语法的时间和总编译次数。"""
    free_at = [0.0] * workers
    cache_ready = {}            # key -> 编译完成时刻
    inflight = {}               # dedup 时用：key -> 完成时刻
    done, compiles = [], 0
    for t, key in sorted(arrivals):
        if key in cache_ready and cache_ready[key] <= t:
            done.append(t)
            continue
        if dedup and key in inflight:
            done.append(inflight[key])
            continue
        w = min(range(workers), key=lambda i: free_at[i])
        start = max(t, free_at[w])
        end = start + compile_s(key)
        free_at[w] = end
        compiles += 1
        done.append(end)
        inflight[key] = min(inflight.get(key, end), end)
        cache_ready[key] = min(cache_ready.get(key, end), end)
    waits = sorted(d - t for d, (t, _) in zip(done, sorted(arrivals)))
    return waits, compiles


def tp_sync(per_rank):
    """grammar_manager.py:273-282：ready 取交集，failed 取并集。"""
    ready = set.intersection(*[r for r, _ in per_rank])
    failed = set.union(*[f for _, f in per_rank])
    return ready, failed


def timeout_seconds(step_s, overlap=True):
    """每轮 wait_ct += 1。源码注释写 max(轮询间隔, 前向时间)；不开 overlap 时 CPU 先轮询再等前向，两者相加。"""
    per = max(POLL_INTERVAL, step_s) if overlap else POLL_INTERVAL + step_s
    return MAX_POLL * per


def scenario_queue(seed=0):
    rnd = random.Random(seed)
    cpu = os.cpu_count() or 4
    workers = min(32, cpu + 4)
    print(f'1）编译线程池：ThreadPoolExecutor() 默认 min(32, cpu + 4)，本机 {cpu} 核 -> {workers} 个线程。')
    print('   同一个 schema 第一次出现时，同时到达的请求各自 submit 一次编译，缓存要等结果取回才写：')
    print(f'{"负载":<34}{"编译次数":>8}{"p50 等待":>10}{"最慢等待":>10}{"假如去重":>10}')
    res = {}
    cases = [
        ('64 个同 schema 同时到达', [(0.0, 'A')] * 64, 0.3),
        ('64 个同 schema 1 s 内均匀到达', [(i / 64, 'A') for i in range(64)], 0.3),
        ('16 个不同 schema 各 4 个', [(0.0, f'S{i % 16}') for i in range(64)], 0.3),
        ('64 个同 schema，大 schema 编 2 s', [(0.0, 'A')] * 64, 2.0),
    ]
    for name, arr, c in cases:
        w, n = simulate_compile(arr, workers, lambda k, c=c: c)
        wd, nd = simulate_compile(arr, workers, lambda k, c=c: c, dedup=True)
        res[name] = (n, w[len(w) // 2], w[-1], nd, wd[-1])
        print(f'{name:<34}{n:>8}{w[len(w) // 2]:>9.2f}s{w[-1]:>9.2f}s{wd[-1]:>9.2f}s')
    print('   「假如去重」是给 in-flight 的 future 加一张表后的结果，主干没有这一层。上线新 schema')
    print('   之前先用一个请求预热，或者让网关把相同 schema 的首批请求串行化，效果一样。')
    print()
    a = {'type': 'object', 'properties': {'n': {'type': 'integer'}, 's': {'type': 'string'}}}
    b = {'properties': {'s': {'type': 'string'}, 'n': {'type': 'integer'}}, 'type': 'object'}
    ka, kb = schema_key(a), schema_key(b)
    print('2）缓存 key 是 json.dumps 的原样字符串，不排序：')
    print(f'   {ka[1]}')
    print(f'   {kb[1]}')
    print(f'   语义相同，key 相同？{ka == kb}。客户端每次随机字段顺序，缓存就永远不命中。')
    res['key_equal'] = ka == kb
    print()
    ranks = [({0, 1, 2}, set()), ({0, 2}, set()), ({0, 1, 2}, {3})]
    ready, failed = tp_sync(ranks)
    print(f'3）TP=3 的三份本地结果 {ranks}')
    print(f'   同步后 ready={ready}（交集，最慢的 rank 说了算）failed={failed}（并集，一个 rank 超时就全体放弃）')
    res['tp'] = (ready, failed)
    print()
    print('4）多久算超时（wait_ct ≥ 10000）：')
    srv = Server(model='qwen3-8b')
    for bs in (1, 64, 256):
        t = srv.t_decode([2000] * bs)
        print(f'   bs={bs:<4} 一步 {t * 1e3:6.2f} ms   overlap {timeout_seconds(t):7.1f} s   '
              f'非 overlap {timeout_seconds(t, False):7.1f} s')
    print(f'   空载时 10000 × 5 ms = {timeout_seconds(0):.0f} s。超时后 key 被写成')
    print('   InvalidGrammarObject("Grammar preprocessing timed out") 进缓存，之后同一个 schema 的请求')
    print('   立刻失败、不再编译，直到 flush_cache 清掉缓存。')
    res['timeout_idle'] = timeout_seconds(0)
    _ = rnd
    return res


# ---------------------------------------------------------------- overlap 时间线
def step_time(fwd, fill, sched=0.0005, mode='overlap'):
    """一步 decode 的墙钟时间（秒）。
    sync：前向完再填 mask、再采样；overlap：tp_worker 推迟采样，mask 在前向期间由 CPU 填；
    spec_sync：投机 + grammar_needs_sync 时这一步关 overlap，等同 sync。"""
    sample = 0.0002
    if mode == 'overlap':
        return max(fwd, sched + fill) + sample
    return sched + fwd + fill + sample


def scenario_overlap(model='qwen3-8b', gpu='h100-sxm'):
    srv = Server(model=model, gpu=gpu)
    print(f'{model} / {gpu}，ctx 2000，每步 ms（括号是相对无语法的增幅）：')
    print(f'{"bs":>5}{"无语法":>9}{"JSON 同步":>16}{"JSON overlap":>18}{"CFG 同步":>16}{"CFG overlap":>18}')
    res = {}
    for bs in (8, 64, 128, 256):
        fwd = srv.t_decode([2000] * bs)
        base = step_time(fwd, 0)
        row = [base]
        cells = ''
        for per in FILL_COST.values():
            f = fill_time(bs, per)
            for mode in ('sync', 'overlap'):
                t = step_time(fwd, f, mode=mode)
                row.append(t)
                cells += f'{t * 1e3:>9.2f} (+{t / base - 1:>4.0%})'
        res[bs] = row
        print(f'{bs:>5}{base * 1e3:>9.2f}{cells}')
    print()
    print('什么时候 overlap 会被关掉：')
    print('   投机解码 + 有语法：verify 后的 accept 要等语法确认，scheduler 在 decode 且结果队列非空时')
    print('   关掉这一步的 overlap（eagle / standalone / dflash 支持语法 overlap，NGRAM 一直同步）。')
    print('   SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1：连续 prefill 不 overlap。')
    return res


# ---------------------------------------------------------------- 思考预算
class TokenSequenceMatcher:
    """utils/token_sequence_matcher.py:18，KMP。"""

    def __init__(self, pattern):
        if not pattern:
            raise ValueError('pattern must contain at least one token')
        self.p = list(pattern)
        pl = [0] * len(self.p)
        matched = 0
        for i in range(1, len(self.p)):
            while matched > 0 and self.p[i] != self.p[matched]:
                matched = pl[matched - 1]
            if self.p[i] == self.p[matched]:
                matched += 1
            pl[i] = matched
        self.pl = pl

    def __len__(self):
        return len(self.p)

    def advance(self, matched, tok):
        while matched > 0 and tok != self.p[matched]:
            matched = self.pl[matched - 1]
        if tok == self.p[matched]:
            matched += 1
        return matched


class Reasoner:
    """ReasonerGrammarObject 的状态机。tokens_after_end == -1 表示还在思考，≥0 表示进入正文。"""

    def __init__(self, think_end_ids, inner=None, max_think=-1, excluded=None, strict=True,
                 require_reasoning=True, reasoning_in_grammar=False):
        self.think_end_ids = list(think_end_ids)
        self.matcher = TokenSequenceMatcher(think_end_ids)
        self.grammar = inner
        self.max_think_tokens = max_think
        self.excluded = excluded
        self.enable_token_filter = strict and (excluded is not None or max_think >= 0)
        self.reasoning_in_grammar = reasoning_in_grammar
        self.tokens_in_think, self.tokens_after_end = (0, -1) if require_reasoning else (-1, 0)
        self.matched = 0
        self.hist = []

    def is_thinking(self):
        return self.tokens_in_think >= 0 and self.tokens_after_end == -1

    def is_generation(self):
        return self.tokens_after_end >= 0

    def grammar_active(self):
        return self.grammar is not None and (self.reasoning_in_grammar or self.is_generation())

    def can_think_more(self):
        return self.max_think_tokens < 0 or self.tokens_in_think + self.matched < self.max_think_tokens

    def transfer_state(self, tok):          # :154
        if self.is_thinking():
            prev = self.matched
            self.hist.append(prev)
            m = self.matcher.advance(prev, tok)
            if m == len(self.matcher):
                self.matched = 0
                self.tokens_after_end = 0
            else:
                self.tokens_in_think += prev + 1 - m
                self.matched = m
        elif self.is_generation():
            self.tokens_after_end += 1

    def rollback_state(self):               # :169
        if self.is_thinking():
            if self.hist:
                prev = self.hist.pop()
                self.tokens_in_think -= prev + 1 - self.matched
                self.matched = prev
        elif self.is_generation():
            if self.tokens_after_end == 0 and self.hist:
                self.tokens_after_end = -1
                self.matched = self.hist.pop()
            elif self.tokens_after_end > 0:
                self.tokens_after_end -= 1

    def accept_token(self, tok):            # :184
        if self.grammar_active():
            self.grammar.accept_token(tok)
        self.transfer_state(tok)

    def rollback(self, k):                  # :211
        after = k if self.reasoning_in_grammar else min(k, max(0, self.tokens_after_end))
        if self.grammar is not None and after > 0:
            self.grammar.rollback(after)
        for _ in range(k):
            self.rollback_state()

    def fill_vocab_mask(self, mask, idx):   # :262（不含 reasoning_in_grammar 分支）
        if self.is_thinking():
            if not self.enable_token_filter:
                return
            if self.can_think_more():
                if self.excluded is not None:
                    set_token_filter(mask[idx], self.excluded, is_allowed=False)
            else:
                set_token_filter(mask[idx], [self.think_end_ids[self.matched]], is_allowed=True)
            return
        if self.is_generation() and self.grammar is not None:
            self.grammar.fill_vocab_mask(mask, idx)


THINK_VOCAB = TOY_VOCAB + ['想', '算', '先', '再', '</', 'think', '>', '<tool_call>', '<|im_end|>']


def run_think(think_want, answer_want, max_think, excluded_names=('<tool_call>', '<|im_end|>')):
    """think_want / answer_want 是模型在思考段、正文段的首选。首选被 mask 掉时，
    logits 变 -inf，采样落到允许集合里概率最高的那个：只剩一个就是它，否则取 FALLBACK 里第一个允许的。"""
    T = THINK_VOCAB.index
    eos = T('<eos>')
    inner = ToyGrammar(ToyRegex(TOY_RE), THINK_VOCAB, eos)
    r = Reasoner([T('</'), T('think'), T('>')], inner=inner, max_think=max_think,
                 excluded=[T(x) for x in excluded_names])
    rows, ti, ai = [], 0, 0
    for _ in range(40):
        mask = new_mask(1, len(THINK_VOCAB))
        r.fill_vocab_mask(mask, 0)
        thinking = r.is_thinking()
        w = think_want[ti] if thinking else answer_want[ai]
        tok = T(w)
        if not bit(mask[0], tok):
            ok = [i for i in range(len(THINK_VOCAB)) if bit(mask[0], i)]
            fb = [T(x) for x in FALLBACK if bit(mask[0], T(x))]
            tok = ok[0] if len(ok) == 1 else fb[0]
        if thinking:
            ti = min(ti + 1, len(think_want) - 1)
        else:
            ai += 1
        r.accept_token(tok)
        rows.append(('思考' if thinking else '正文', w, THINK_VOCAB[tok],
                     r.tokens_in_think, r.matched, r.tokens_after_end))
        if inner.is_terminated():
            break
    return rows, r


FALLBACK = ['想', '算']


def scenario_think():
    think_want = ['想', '先', '<tool_call>', '</', '再', '想', '算', '想', '算', '想', '</', 'think', '>']
    answer_want = ['{"', 'n', '":1', '3', '}', '<eos>']
    print('think_end 是三个 token「</ think >」，预算 max_thinking_tokens=6，屏蔽 <tool_call> 和 <|im_end|>。')
    print('「想要」是模型自己的首选，「实际」是 mask 之后采到的 token：')
    rows, r = run_think(think_want, answer_want, 6)
    print(f'{"步":>3}  {"阶段":<4}{"想要":<14}{"实际":<14}{"思考计数":>8}{"已匹配":>7}{"正文计数":>9}')
    for i, (ph, w, a, n, m, after) in enumerate(rows):
        flag = '' if w == a else '  <- 被改写'
        print(f'{i:>3}  {ph:<4}{w:<14}{a:<14}{n:>8}{m:>7}{after:>9}{flag}')
    print('   第 3 步「</」后面跟的是「再」，KMP 把匹配退回 0，「</」和「再」一起记进思考计数（3 -> 5）。')
    print('   第 2 步模型想调工具，<tool_call> 在思考段被屏蔽，只能继续想。')
    print('   计数 + 已匹配到 6 以后，每步只放行 think_end 的下一个 token，模型被逼着收尾。')
    print()
    T = THINK_VOCAB.index
    r2 = Reasoner([T('</'), T('think'), T('>')], max_think=-1)
    seq = [T('想'), T('</'), T('think'), T('>'), T('{')]
    snaps = []
    for t in seq:
        r2.accept_token(t)
        snaps.append((r2.tokens_in_think, r2.matched, r2.tokens_after_end))
    r2.rollback(3)
    print(f'回滚：accept 5 个 token 后状态 {snaps[-1]}，rollback(3) 后 '
          f'{(r2.tokens_in_think, r2.matched, r2.tokens_after_end)}，和第 2 个 token 之后的 {snaps[1]} 一致。')
    print('   投机解码拒掉跨过 think_end 的草稿时，状态能退回思考段。')
    return rows, r, snaps, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='mask', choices=['mask', 'fsm', 'queue', 'overlap', 'think'])
    ap.add_argument('--model', default='qwen3-8b')
    ap.add_argument('--gpu', default='h100-sxm')
    a = ap.parse_args()
    if a.scenario == 'mask':
        scenario_mask(a.model, a.gpu)
    elif a.scenario == 'fsm':
        scenario_fsm()
    elif a.scenario == 'queue':
        scenario_queue()
    elif a.scenario == 'overlap':
        scenario_overlap(a.model, a.gpu)
    else:
        scenario_think()


if __name__ == '__main__':
    main()
