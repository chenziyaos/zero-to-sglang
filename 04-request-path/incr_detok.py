#!/usr/bin/env python3
"""incr_detok.py — 把增量反分词从三进程模拟里单独拎出来，逐条打印偏移和提交决定。

为什么需要它：一个汉字 3 个 UTF-8 字节，可能跨 3 个 decode step 到达。
detokenizer 不能看到半个字符就往外吐，否则用户屏幕上会先出现 �。
sglang 的做法是维护 surr/read 两个偏移，末尾是不完整 UTF-8 时只发可打印前缀、
偏移不提交，下一轮带着新字节重解（detokenizer_manager.py:310）。

运行：python3 incr_detok.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_three_procs import (  # noqa: E402
    SURR_WINDOW,
    decode_ids,
    encode_text,
    find_printable_text,
)


class DecodeStatus:
    """和 sim_three_procs.detokenizer_proc 里同一个状态机，去掉进程部分。"""

    def __init__(self, prompt_ids):
        self.decoded_text = ""
        self.surr_offset = 0
        self.read_offset = len(prompt_ids)
        self.decode_ids = list(prompt_ids)
        self.sent_offset = 0

    @property
    def decoded_text_len(self):
        return len(self.decoded_text)

    def feed(self, new_ids, finished=False):
        """喂一批新 token，返回 (发出的文本, 是否提交偏移, 末尾新文本)。"""
        self.decode_ids.extend(new_ids)
        surr_ids = self.decode_ids[self.surr_offset:self.read_offset]
        read_ids = self.decode_ids[self.surr_offset:]
        surr_text = decode_ids(surr_ids)
        read_text = decode_ids(read_ids)
        new_text = read_text[len(surr_text):]

        if finished:
            full = self.decoded_text + new_text
            emit = full[self.sent_offset:]
            return emit, True, new_text

        pending = self.sent_offset - self.decoded_text_len
        if new_text and not new_text.endswith("�"):
            # 干净文本：提交，窗口前移
            self.decoded_text += new_text
            self.surr_offset = self.read_offset
            self.read_offset = len(self.decode_ids)
            self.sent_offset = self.decoded_text_len
            return (new_text[pending:] if pending else new_text), True, new_text

        # 不完整 UTF-8：只发可打印前缀，偏移原地不动，下一轮重试
        printable = find_printable_text(new_text)
        self.sent_offset = self.decoded_text_len + len(printable)
        return (printable[pending:] if pending else printable), False, new_text


def main():
    # prompt 尾巴取 5 个 token 作为上下文窗口（INIT_INCREMENTAL_DETOKENIZATION_OFFSET=5）
    prompt = encode_text("问题: ")
    output = encode_text("Hello 世界")  # 空格之后，'世' 3 字节，'界' 3 字节

    st = DecodeStatus(prompt)
    print(f"prompt={prompt!r}（read_offset 初始为 {st.read_offset}，"
          f"surr 窗口取末尾 {SURR_WINDOW} 个 token）")
    print(f"output={output!r}")
    print()
    print(f"{'step':>4} {'新到 token':>22} {'new_text':>12} {'提交?':>5}  发出")

    streamed = ""
    step = 0
    # 首包：prompt 尾巴 + 第一个输出 token（scheduler 首包的语义）
    first_n = 1
    emitted, committed, new_text = st.feed(output[:first_n])
    step += 1
    streamed += emitted
    raw = ",".join(f"{b:#05x}" for b in [i & 0xFF for i in output[:first_n]])
    print(f"{step:>4} {raw:>22} {new_text!r:>12} {'是' if committed else '否':>5}  {emitted!r}")

    i = first_n
    while i < len(output):
        chunk = output[i:i + 1]  # 一个 decode step 一个 token
        emitted, committed, new_text = st.feed(chunk, finished=(i + 1 >= len(output)))
        streamed += emitted
        i += 1
        step += 1
        raw = ",".join(f"{b:#05x}" for b in [j & 0xFF for j in chunk])
        print(f"{step:>4} {raw:>22} {new_text!r:>12} {'是' if committed else '否':>5}  {emitted!r}")

    print()
    print(f"拼起来：{streamed!r}")
    expected = decode_ids(output)
    print(f"目标：  {expected!r}")
    assert streamed == expected, "增量反分词结果不等于一次性解码"
    print("一致。关键看 step 7/8（'世' 的前两个字节）：发出为空，偏移不提交，step 9 带着第 3 个字节重解，整字一次发出。")


if __name__ == "__main__":
    main()
