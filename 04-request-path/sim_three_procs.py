#!/usr/bin/env python3
"""sim_three_procs.py — 用多进程队列复现 SGLang 的三进程请求路径。

对应主干 8fabf5e：

    HTTP/TokenizerManager  --scheduler_input-->  Scheduler
            ^                                            |
            +--------tokenizer_ipc---- Detokenizer <-----+
                         detokenizer_ipc

本脚本只用标准库，CPU 上能跑。模型计算用 sleep 代替（prefill 5 ms，decode 一步 2 ms），
重点不是性能数字，而是消息在三个进程之间怎么流：

1. 三进程拓扑和三条 ZMQ 通道（这里用 multiprocessing.Queue 代替）。
2. 流式节拍：stream_interval、首 token 立即发、非流式每 50 个 token 强制推一次。
3. 增量反分词：5 个 token 的上下文窗口、surr/read 偏移、不完整 UTF-8 暂不提交。
4. batch_notify_size：攒够一批请求再一次性唤醒，让出一次事件循环。
5. 每条消息带上逐跳时间戳，最后算 TTFT/ITL，和 SGL-03 的客户端口径对齐。
6. Detokenizer 的 DecodeStatus 容量有限，--max-states 调小可以复现 issue #2812。

运行：

    python3 sim_three_procs.py                       # 6 条流式请求
    python3 sim_three_procs.py --requests 20 --max-new 60
    python3 sim_three_procs.py --stream-interval 4
    python3 sim_three_procs.py --nonstream 2         # 两条非流式，看 50 token 强制推送
    python3 sim_three_procs.py --max-states 2        # 复现状态被挤掉后的报错
"""

import argparse
import multiprocessing as mp
import queue
import random
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 模拟用的时间尺度（毫秒）。真实数字取决于 GPU，这里只保证事件顺序正确。
PREFILL_MS = 5.0
DECODE_MS = 2.0
TOKENIZE_MS = 0.4
DETOK_MS = 0.2

# schedule_batch.py: INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5
SURR_WINDOW = 5
# output_streamer.py: SGLANG_FORCE_STREAM_INTERVAL 默认 50
DEFAULT_FORCE_INTERVAL = 50
# tokenizer_manager.py: SGLANG_REQUEST_STATE_WAIT_TIMEOUT 默认 4 秒
STATE_WAIT_TIMEOUT_S = 4.0


# ---------------------------------------------------------------------------
# 线上消息：对应 io_struct.py 里的 msgspec.Struct。这里用 dataclass + pickle 队列。
# ---------------------------------------------------------------------------

@dataclass
class TokenizedReq:
    """对应 TokenizedGenerateReqInput：分词之后进 scheduler 的消息。"""
    rid: str
    input_ids: List[int]
    output_plan: List[int]          # 每一步要吐出的 token id（模拟模型输出）
    stream: bool
    stream_interval: int
    stop: Optional[str]
    stamps: dict = field(default_factory=dict)


@dataclass
class BatchTokenIDOutput:
    """对应 BatchTokenIDOutput：scheduler 一批输出给 detokenizer。"""
    rids: List[str]
    decode_ids: List[List[int]]     # 每条请求本次新增的解码窗口片段
    read_offsets: List[int]         # 片段里 prompt 尾巴的长度（首个包才非零）
    finished: List[bool]
    stamps: dict = field(default_factory=dict)


@dataclass
class BatchStrOutput:
    """对应 BatchStrOutput：detokenizer 转回文本后给 tokenizer。"""
    rids: List[str]
    texts: List[str]
    finished: List[bool]
    stamps: dict = field(default_factory=dict)


@dataclass
class AbortReq:
    rid: str


# ---------------------------------------------------------------------------
# 假分词器：字节即 token。
# ASCII 字符一个字节一个 token；CJK 字符三个字节拆成三个 token（id 带高位标记），
# 这样一个汉字跨三个 decode step 到达，可以真实触发不完整 UTF-8 的暂存逻辑。
# ---------------------------------------------------------------------------

FRAG_BIT = 1 << 24


def encode_text(text: str) -> List[int]:
    ids = []
    for b in text.encode("utf-8"):
        ids.append(b if b < 128 else FRAG_BIT | b)
    return ids


def decode_ids(ids: List[int]) -> str:
    raw = bytes(i & 0xFF for i in ids)
    # errors="replace"：不完整的 UTF-8 解成 U+FFFD，和 HF tokenizer 的表现一致。
    return raw.decode("utf-8", errors="replace")


def is_chinese_char(cp: int) -> bool:
    # transformers streamers.py 里的原始区间判断
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F
    )


def find_printable_text(text: str) -> str:
    """sglang utils.find_printable_text 的同构实现。"""
    if text.endswith("\n"):
        return text
    if len(text) > 0 and is_chinese_char(ord(text[-1])):
        return text
    if len(text) > 1 and is_chinese_char(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]


# ---------------------------------------------------------------------------
# 请求内容生成：英文单词流里定期插一个汉字，保证多字节切分一定会发生。
# ---------------------------------------------------------------------------

def make_plan(rng: random.Random, max_new: int) -> List[int]:
    """返回输出 token 序列：英文单词流里每隔 9~13 个输出位置插一个汉字（3 个字节 token）。"""
    cjk = "推理引擎调度批处理显存请求旅程"
    out: List[int] = []
    steps = 0
    next_cjk = rng.randint(8, 12)
    while steps < max_new:
        if steps >= next_cjk:
            frags = encode_text(rng.choice(cjk))
            out.extend(frags)
            steps += len(frags)
            next_cjk = steps + rng.randint(8, 12)
            continue
        word_len = min(rng.randint(2, 6), max_new - steps - 1)
        if word_len <= 0:
            out.append(ord(rng.choice(string.ascii_uppercase)))
            steps += 1
            continue
        for _ in range(word_len):
            out.append(ord(rng.choice(string.ascii_uppercase)))
        steps += word_len
        if steps < max_new:
            out.append(ord(" "))
            steps += 1
    return out[:max_new]


def expected_text(ids: List[int]) -> str:
    return decode_ids(ids)


# ---------------------------------------------------------------------------
# Scheduler 进程：事件循环 = ingest_requests -> get_next_batch -> run_batch -> stream
# ---------------------------------------------------------------------------

class RunningReq:
    def __init__(self, msg: TokenizedReq):
        self.rid = msg.rid
        self.plan = msg.output_plan
        self.step = 0
        self.stream = msg.stream
        self.stream_interval = msg.stream_interval
        self.stop = msg.stop
        # 增量反分词状态（在真实代码里属于 Req，由 scheduler 侧维护）
        self.surr_offset: Optional[int] = None
        self.read_offset = 0
        self.full_decode: List[int] = []   # prompt 尾巴 + 全部输出
        self.send_token_offset = 0         # 已送给 detok 的输出 token 数
        self.finished_output = False


def scheduler_proc(in_q, detok_q, force_interval):
    waiting: List[TokenizedReq] = []
    running: Dict[str, RunningReq] = {}
    prompt_of: Dict[str, List[int]] = {}

    while True:
        # 1) ingest_requests：非阻塞地把输入队列一次抽干
        drained = 0
        while True:
            try:
                msg = in_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if isinstance(msg, AbortReq):
                waiting = [m for m in waiting if m.rid != msg.rid]
                running.pop(msg.rid, None)
                prompt_of.pop(msg.rid, None)
                continue
            msg.stamps["sched_recv"] = time.perf_counter()
            waiting.append(msg)
            prompt_of[msg.rid] = msg.input_ids

        # 2) prefill：新请求先进 running（真实代码里还会受 KV 预算限制）
        for msg in waiting:
            time.sleep(PREFILL_MS / 1000)
            msg.stamps["prefill_done"] = time.perf_counter()
            running[msg.rid] = RunningReq(msg)
        waiting.clear()

        # 3) decode 一步：所有在跑请求各吐一个 token
        if not running:
            if drained == 0:
                time.sleep(0.002)
            continue
        time.sleep(DECODE_MS / 1000)

        acc_rids, acc_ids, acc_offsets, acc_finished = [], [], [], []
        done_rids = []
        for rid, req in list(running.items()):
            req.step += 1
            output_len = min(req.step, len(req.plan))
            finished = output_len >= len(req.plan)

            # OutputStreamer.accept 的判定
            if finished:
                should = True
            elif req.stream:
                num_unsent = output_len - req.send_token_offset
                should = (
                    req.stream_interval == 1
                    or num_unsent >= req.stream_interval
                    or (req.send_token_offset == 0 and num_unsent > 0)
                )
                if should and req.stop:
                    tail = expected_text(req.plan[:output_len])
                    should = not any(
                        tail.endswith(req.stop[:k]) for k in range(1, len(req.stop) + 1)
                    )
            else:
                should = output_len % force_interval == 0

            if not should:
                continue

            # 组装增量反分词窗口
            first = req.surr_offset is None
            if first:
                req.read_offset = len(prompt_of[rid])
                req.surr_offset = max(req.read_offset - SURR_WINDOW, 0)
                req.full_decode = prompt_of[rid][req.surr_offset:] + req.plan[:output_len]
                payload = list(req.full_decode)
                read_offset = req.read_offset - req.surr_offset
            else:
                payload = req.plan[req.send_token_offset:output_len]
                read_offset = 0
            req.send_token_offset = output_len

            acc_rids.append(rid)
            acc_ids.append(payload)
            acc_offsets.append(read_offset)
            acc_finished.append(finished)
            if finished:
                req.finished_output = True
                done_rids.append(rid)

        if acc_rids:
            out = BatchTokenIDOutput(
                acc_rids, acc_ids, acc_offsets, acc_finished,
                {"sched_send": time.perf_counter()},
            )
            detok_q.put(out)
        for rid in done_rids:
            running.pop(rid, None)
            prompt_of.pop(rid, None)

        # 全部请求跑完且没有新输入时退出（由主控发哨兵，见 main）
        # 退出判断放在 main 的超时回收里，这里保持循环。


# ---------------------------------------------------------------------------
# Detokenizer 进程：同步 zmq 事件循环 = sock_recv -> dispatch -> sock_send
# ---------------------------------------------------------------------------

class DecodeStatus:
    def __init__(self, decoded_text, decode_ids, read_offset):
        self.decoded_text = decoded_text
        self.decode_ids = list(decode_ids)
        self.surr_offset = 0
        self.read_offset = read_offset
        self.sent_offset = 0

    @property
    def decoded_text_len(self):
        return len(self.decoded_text)


class LimitedCapacityDict(dict):
    """detokenizer_manager.LimitedCapacityDict：满了挤掉最老的（issue #2812）。"""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity and key not in self:
            oldest = next(iter(self))
            del self[oldest]
        super().__setitem__(key, value)


def detokenizer_proc(detok_q, tok_q, max_states):
    statuses: LimitedCapacityDict = LimitedCapacityDict(max_states)

    while True:
        recv = detok_q.get()
        recv_t = time.perf_counter()
        rids, texts_out, finished_out = [], [], []

        # 初始化 / 扩展状态，算出 read 窗口和 surr 窗口
        read_windows, surr_windows, finished_map = [], {}, {}
        for i, rid in enumerate(recv.rids):
            if rid not in statuses:
                statuses[rid] = DecodeStatus("", recv.decode_ids[i], recv.read_offsets[i])
            else:
                statuses[rid].decode_ids.extend(recv.decode_ids[i])
            s = statuses[rid]
            read_windows.append(list(s.decode_ids[s.surr_offset:]))
            surr_windows[rid] = list(s.decode_ids[s.surr_offset:s.read_offset])
            finished_map[rid] = recv.finished[i]

        # 批量 decode（真实代码默认走 tokenizer 的批量解码）
        read_text = {rid: decode_ids(ids) for rid, ids in
                     ((recv.rids[i], read_windows[i]) for i in range(len(recv.rids)))}

        for rid in recv.rids:
            s = statuses.get(rid)
            if s is None:
                raise RuntimeError(
                    f"Decode status not found for request {rid}. It may have been evicted "
                    "from the decode status due to memory pressure. Set "
                    "SGLANG_DETOKENIZER_MAX_STATES bigger. See issue #2812."
                )
            surr_text = decode_ids(surr_windows[rid])
            new_text = read_text[rid][len(surr_text):]
            finished = finished_map[rid]
            time.sleep(DETOK_MS / 1000)

            if not finished:
                # 与 detokenizer_manager._decode_batch_token_id_output 同构
                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("�"):
                    s.decoded_text += new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    emit = new_text[pending:] if pending else new_text
                else:
                    # 不完整 UTF-8：只发可打印前缀，偏移不提交，下一轮带着后续字节重试
                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    emit = printable[pending:] if pending else printable
            else:
                # finish：把已提交文本和最后一段拼起来，从 sent_offset 切增量。
                # 上一轮若有「可打印但未提交」的 pending 文本，它已经发过，这里跳过。
                full = s.decoded_text + new_text
                emit = full[s.sent_offset:]
                del statuses[rid]

            if emit:
                rids.append(rid)
                texts_out.append(emit)
                finished_out.append(finished)
            else:
                # 线上空串也会随 BatchStrOutput 走一遍（output_strs 与 rids 等长），
                # tokenizer 侧照样 append 到 out_list 并唤醒请求；这里保持一致。
                rids.append(rid)
                texts_out.append("")
                finished_out.append(finished)

        if rids:
            tok_q.put(BatchStrOutput(rids, texts_out, finished_out,
                                     {"detok_recv": recv_t, "detok_send": time.perf_counter()}))


# ---------------------------------------------------------------------------
# Tokenizer 侧（主进程内多线程，模拟 asyncio 里的每条请求一个 task）
# ---------------------------------------------------------------------------

@dataclass
class ReqState:
    out_list: list = field(default_factory=list)
    event: threading.Event = field(default_factory=threading.Event)
    finished: bool = False


def request_worker(idx, args, sched_q, reply_q, state_lock, states, records):
    rid = f"req-{idx:03d}"
    rng = random.Random(args.seed + idx)
    stream = idx >= args.nonstream
    plan = make_plan(rng, args.max_new)
    expected = expected_text(plan)

    state = ReqState()
    with state_lock:
        states[rid] = state

    # 分词（主进程，CPU 开销不挡 GPU）
    time.sleep(TOKENIZE_MS / 1000)
    prompt = f"PROMPT{idx:02d} "
    t0 = time.perf_counter()
    msg = TokenizedReq(
        rid, encode_text(prompt), plan, stream, args.stream_interval,
        args.stop, {"created": t0, "tokenize_done": time.perf_counter()},
    )
    sched_q.put(msg)

    # _stream_one_response：等事件，排空 out_list
    chunks, chunk_ts, held_chunks, hops = [], [], 0, []
    assembled = ""
    while True:
        if not state.event.wait(timeout=STATE_WAIT_TIMEOUT_S):
            raise TimeoutError(f"{rid}: 等待输出超时（{STATE_WAIT_TIMEOUT_S}s）")
        with state_lock:
            batch = state.out_list
            state.out_list = []
            state.event.clear()
            finished = state.finished
        for text, fin, stamps in batch:
            if text == "":
                held_chunks += 1
            chunks.append(text)
            chunk_ts.append(time.perf_counter() - t0)
            assembled += text
        if finished:
            break

    # 流式：拼起来必须逐字节等于目标；非流式只在最后拿全文，不做拼接断言
    if stream and assembled != expected:
        raise AssertionError(f"{rid} 反分词结果不一致：{assembled!r} != {expected!r}")

    ttft = chunk_ts[0] if chunk_ts else 0.0
    itls = [b - a for a, b in zip(chunk_ts, chunk_ts[1:])]
    n_nonempty = sum(1 for c in chunks if c)
    records[rid] = {
        "stream": stream,
        "chunks": len(chunks),
        "nonempty_chunks": n_nonempty,
        "held_empty": held_chunks,
        "tokens": len(plan),
        "ttft_ms": ttft * 1000,
        "itl_ms": [x * 1000 for x in itls],
        "text": assembled,
    }


def receiver_thread(tok_q, state_lock, states, batch_notify_size):
    """handle_loop：async_sock_recv -> _handle_batch_output -> 分组 set event。"""
    while True:
        out = tok_q.get()
        pending = []
        for rid, text, fin in zip(out.rids, out.texts, out.finished):
            with state_lock:
                st = states.get(rid)
                if st is None:
                    continue
                st.out_list.append((text, fin, out.stamps))
                if fin:
                    st.finished = True
                pending.append(st)
                # 攒够一批才唤醒，并让出一次事件循环（asyncio.sleep(0)）
                if len(pending) >= batch_notify_size:
                    for s in pending:
                        s.event.set()
                    pending = []
                    time.sleep(0)
        for s in pending:
            s.event.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--nonstream", type=int, default=0,
                    help="前 N 条请求走非流式（默认全部流式）")
    ap.add_argument("--stream-interval", type=int, default=1)
    ap.add_argument("--batch-notify-size", type=int, default=16)
    ap.add_argument("--force-interval", type=int, default=DEFAULT_FORCE_INTERVAL)
    ap.add_argument("--max-states", type=int, default=1 << 16)
    ap.add_argument("--stop", default=None, help="停止串，演示 streamer 的前缀暂留")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 用 fork：spawn 要重新 import 整个模块，六个请求线程同时灌队列时，
    # 子进程启动可能被整体推迟数秒（玩具模拟里的调度假象），fork 没有这个问题。
    ctx = mp.get_context("fork")
    sched_q = ctx.Queue()    # scheduler_input_ipc_name: tokenizer -> scheduler
    detok_q = ctx.Queue()    # detokenizer_ipc_name:     scheduler -> detokenizer
    tok_q = ctx.Queue()      # tokenizer_ipc_name:       detokenizer -> tokenizer

    sched = ctx.Process(target=scheduler_proc,
                        args=(sched_q, detok_q, args.force_interval), daemon=True)
    detok = ctx.Process(target=detokenizer_proc,
                        args=(detok_q, tok_q, args.max_states), daemon=True)
    sched.start()
    detok.start()

    states: Dict[str, ReqState] = {}
    records: Dict[str, dict] = {}
    state_lock = threading.Lock()
    recv = threading.Thread(target=receiver_thread,
                            args=(tok_q, state_lock, states, args.batch_notify_size),
                            daemon=True)
    recv.start()

    t_start = time.perf_counter()
    threads = []
    for i in range(args.requests):
        th = threading.Thread(target=request_worker,
                              args=(i, args, sched_q, tok_q, state_lock, states, records))
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=30)
    elapsed = (time.perf_counter() - t_start) * 1000

    # 检查 detokenizer 是否因为状态被挤掉而死
    alive = detok.is_alive()
    sched.terminate()
    detok.terminate()

    print(f"# {args.requests} 条请求（流式 {args.requests - args.nonstream}，"
          f"非流式 {args.nonstream}），每条最多 {args.max_new} token，"
          f"stream_interval={args.stream_interval}，batch_notify_size={args.batch_notify_size}")
    if not alive:
        print("# detokenizer 进程已退出：DecodeStatus 容量太小，状态被挤掉，见 issue #2812")
        print("# 调大 --max-states（线上对应 SGLANG_DETOKENIZER_MAX_STATES）后重试")
        return

    print(f"{'rid':>9} {'模式':<4} {'chunk':>5} {'非空':>4} {'空chunk':>7} "
          f"{'TTFT ms':>8} {'ITL中位 ms':>10}")
    for rid in sorted(records):
        r = records[rid]
        itls = sorted(r["itl_ms"])
        med = itls[len(itls) // 2] if itls else 0.0
        print(f"{rid:>9} {'流' if r['stream'] else '非流':<4} {r['chunks']:>5} "
              f"{r['nonempty_chunks']:>4} {r['chunks'] - r['nonempty_chunks']:>7} "
              f"{r['ttft_ms']:>8.1f} {med:>10.1f}")
    print(f"# 全部完成用时 {elapsed:.0f} ms（prefill {PREFILL_MS} ms/请求，"
          f"decode {DECODE_MS} ms/批，时间是 sleep 出来的，只看事件顺序）")

    # 挑一条流式请求打印前几个 chunk，看汉字如何跨三步到达
    rid = next(k for k in sorted(records) if records[k]["stream"])
    r = records[rid]
    text = r["text"]
    cjk_at = [i for i, ch in enumerate(text) if is_chinese_char(ord(ch))][:2]
    print(f"# {rid} 前 60 字：{text[:60]!r}")
    if cjk_at:
        pos = cjk_at[0]
        before = text[:pos]
        # 找到包含这个汉字的 chunk 边界（空 chunk 对应被暂存的中间字节）
        print(f"# 第一个汉字在第 {pos} 个字符；它的 3 个 UTF-8 字节分 3 个 decode step 到达，"
              f"前两个 step 末尾是不完整序列，只发可打印前缀（可能为空），第三个 step 才提交整个汉字")


if __name__ == "__main__":
    main()
