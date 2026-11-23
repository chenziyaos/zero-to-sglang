# SGL-04 请求旅程

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/04-request-path.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-04｜原理｜请求旅程：从 HTTP 到流式返回。

验证基线：sglang 主干 `8fabf5e`。两个脚本只用标准库，CPU 上能跑。

| 文件 | 作用 |
|---|---|
| `sim_three_procs.py` | 用 multiprocessing 三个进程 + 三条队列，复现 tokenizer → scheduler → detokenizer → tokenizer 的消息流，含 stream_interval、非流式强制推送、batch_notify_size、增量反分词偏移和 DecodeStatus 容量淘汰 |
| `incr_detok.py` | 把增量反分词状态机单独拎出来，逐 step 打印 surr/read 偏移、提交决定和发出文本，专门看清一个汉字怎么跨 3 个 decode step 到达 |

模拟里模型计算用 sleep 代替（prefill 5 ms/请求，decode 2 ms/批），数字没有性能意义，只保证事件顺序和线上一致：首 token 立即推、之后按 stream_interval 攒批、不完整 UTF-8 暂不提交、finish 时发尾包。

## 运行

```bash
# 1. 默认：6 条流式请求，stream_interval=1，每个 decode step 都推
python3 sim_three_procs.py

# 2. 攒 4 个 token 推一次，ITL 中位约变成 4 倍
python3 sim_three_procs.py --stream-interval 4 --requests 3

# 3. 两条非流式：中间每 10 个 token 强制推一次（线上默认 50），最后一次给全文
python3 sim_three_procs.py --nonstream 2 --force-interval 10 --requests 4 --max-new 40

# 4. 把 DecodeStatus 容量调到 2，复现 issue #2812：最老请求的状态被挤掉，detokenizer 报错退出
python3 sim_three_procs.py --requests 8 --max-states 2

# 5. 单独看增量反分词：'世' 的前两个字节到达时发出为空、偏移不提交
python3 incr_detok.py
```

## 期望输出（默认）

```
# 6 条请求（流式 6，非流式 0），每条最多 48 token，stream_interval=1，batch_notify_size=16
      rid 模式   chunk   非空  空chunk  TTFT ms   ITL中位 ms
  req-000 流       48     42       6     36.9        2.1
  req-001 流       48     42       6     36.5        2.1
  ...
# req-000 前 60 字：'XI HEX VX 程SN AC HQT RG 请HOSIZ YZ WNK 调YK '
# 第一个汉字在第 10 个字符；它的 3 个 UTF-8 字节分 3 个 decode step 到达，前两个 step 末尾是不完整序列，只发可打印前缀（可能为空），第三个 step 才提交整个汉字
```

三个观察点：

1. **chunk 数等于输出 token 数，但非空 chunk 更少。** 48 个 token 产生 48 个 chunk，其中 6 个空串，对应 3 个汉字各有前两个字节落在不完整 UTF-8 序列里。空 chunk 在线上也会走一遍 BatchStrOutput，tokenizer 侧照样唤醒请求，只是文本没有增量。
2. **ITL 中位等于一个 decode step。** `--stream-interval 4` 时变成约 8 ms（4 个 step），首 token 仍然在第一步就推（streamer 对首 token 有特例），TTFT 不随 interval 变大。
3. **`--max-states 2` 时 detokenizer 进程退出。** 报错信息和线上 issue #2812 同款：Decode status not found，调大 `SGLANG_DETOKENIZER_MAX_STATES`（默认 65536）。

`incr_detok.py` 的关键输出：

```
step               新到 token     new_text   提交?  发出
   7                  0x0e4          '�'       否   ''
   8                  0x0b8          '�'       否   ''
   9                  0x096          '世'       是   '世'
```

step 7、8 的偏移原地不动，step 9 把三个字节放在一起重解，整字一次发出。这就是用户屏幕上不会先闪一个 � 的原因。
