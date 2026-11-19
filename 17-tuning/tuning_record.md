# 调参记录

一次只改一类参数，每次改完跑同一份压测、留同一段日志，再填一行。表头之外的内容是 sample_before.log → sample_after.log 那次的示范，换成自己的数。

## 0. 固定条件

| 项 | 值 |
|---|---|
| 模型 / 精度 | Qwen/Qwen3-8B，bf16 权重，KV auto |
| 硬件 | 1 × H100 80GB SXM，nvidia-smi 81559 MiB |
| sglang 版本 | 8fabf5e |
| 压测命令 | `python3 -m sglang.benchmark.serving --backend sglang-oai --dataset-name sharegpt --num-prompts 4000 --request-rate 30 --disable-ignore-eos --extra-request-body '{"max_tokens": 4096}'`（必须 --disable-ignore-eos，否则每条请求吐满、准入诊断失真；sglang-oai 后端下 extra-request-body 的 max_tokens 覆盖数据集长度） |
| 负载说明 | prompt 300 - 900，实际输出 50 - 500 后 EOS；客户端 max_tokens 4096（调参前） |
| SLO | TTFT p99 ＜ 3 s，ITL p99 ＜ 60 ms |
| 日志开关 | 默认 `--log-level info`，Decode batch 每 40 步一行（`--decode-log-interval`） |

不改的东西写在这里：压测命令、请求分布、客户端并发。它们一变，前后数据就不能比。

## 1. 每轮记录

| 轮 | 改了什么 | 启动 avail GB | max_total_num_tokens | running p50 | usage p50 | queue p50 | graph=False | retract/分 | gen tok/s p50 | TTFT p99 | ITL p99 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 默认 + lpm | 12.26 | 365,312 | 271 | 0.74 | 270 | 65% | 0.10 | 12,080 | | | parse_log：AVAIL_HIGH、ADMIT_TIGHT、GRAPH_FALLBACK、LPM_FALLBACK |
| 1 | frac 0.83 → 0.89；policy → fcfs | | | | | | | | | | | 只动显存和策略，看 KV 多了多少 |
| 2 | conservativeness 1.0 → 0.3；客户端 max_tokens → 768 | | | | | | | | | | | running 预计涨到 360 左右 |
| 3 | cuda-graph-max-bs-decode 256 → 384 | 7.27 | 399,744 | 364 | 0.91 | 110 | 0% | 0.20 | 13,177 | | | 全部 [正常]，停 |

图上限没有照第 0 轮建议的 320 给：第 2 轮放松准入后 running 会上去，按改完以后的 p90 取才对。所以第 3 轮要在第 2 轮的日志上重新跑 parse_log。

## 2. 每轮必做

1. 启动日志里抄 `max_total_num_tokens`、`available_gpu_mem`、decode 图的 `bs=[…]` 最后一个数。
2. 压测跑满 10 分钟以上，去掉前 1 分钟预热。
3. `python3 parse_log.py server.log > round_N.txt`，`--json` 结果一起留档。
4. bench_serving 的 TTFT / ITL p99 抄进表。
5. 有 `[错误]` 级别的诊断先处理，不进下一轮。

## 3. 回退条件

- 出现 Prefill / Decode out of memory：退回上一轮的 frac 或 chunk。
- retract 超过每分钟 1 次且 TTFT p99 变差：conservativeness 往回调 0.2。
- ITL p99 超 SLO：先看 graph=False 比例，再考虑 `--max-running-requests` 封顶。
