# SGL-18 可观测与 profiling

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/18-observability.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-18｜实战｜可观测与 profiling：从指标到 kernel。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

一条慢请求从上往下追四层：`/metrics` 看整体（scrape_metrics.py）→ ReqTimeStats 看单条请求和它进来时的服务端状态（reqstats.py）→ torch profiler trace 看一步前向里 GPU 在等谁（trace_summary.py）→ nsys 按层看 kernel（profile.sh nsys）。

| 文件 | 内容 |
|---|---|
| scrape_metrics.py | 解析 Prometheus exposition 文本：标签、NaN / +Inf、多 rank 求和；按 histogram_quantile 的规则从桶估分位数并报出所在桶；两次抓取做差只看中间窗口，窗口长度从 `scheduler_stage_seconds_total` 各阶段之和推出；输出吞吐、TTFT / ITL / TPOT / E2E / 排队分位数、调度循环时间去向、CUDA graph 命中率，按诊断码给结论；能直接抓 URL，`--json` 输出机器可读结果 |
| reqstats.py | 读 `--enable-request-time-stats-logging` 打出的 ReqTimeStats 行（unified、PD prefill、PD decode 三种），拆成排队 + 前向；自动对齐日志 asctime 与 entry_time 的时区差；最慢几条请求对回同一时刻的 Prefill / Decode batch 行；按进入时刻分桶，找出 cuda graph 掉成 False 的时间段 |
| trace_summary.py | 读 torch profiler 导出的 `*.trace.json.gz`：每步墙钟、GPU 忙碌、GPU 空闲比例，kernel 排行，CPU 侧 `scheduler.*` 跨度占比，launch 到开跑的延迟；判定 LAUNCH_BOUND / CPU_GAP / GPU_BOUND；给两份 trace 时做对比 |
| profile.sh | 真机上采一次完整数据：`serve` 带指标和逐请求日志起服务，`load` 抓两次 `/metrics` 夹一次压测并中途抓 trace，`analyze` 跑上面三个脚本，`nsys` / `nsys-start` 按层抓 kernel |
| whatif.py | 定位到容量以后先算账：同一份到达序列，比原样、开 mixed、chunk 2048 + mixed、两副本轮询、限流 10 条/s 五种办法下的 TTFT、ITL、前向和 decode batch 峰值 |
| make_fixtures.py | 用前面章节的模拟器造夹具，输出确定，重跑字节级一致 |
| metrics_t0.txt / metrics_t1.txt | 测试夹具：45 s 与 90 s 两次 `/metrics` 抓取 |
| server_reqstats.log | 测试夹具：1002 条 ReqTimeStats 加 Prefill / Decode batch 行 |
| trace_eager.json.gz / trace_graph.json.gz | 测试夹具：Qwen3-0.6B bs 16 的 10 步 decode，一份 eager、一份 CUDA graph 回放，发射露出来的情形 |
| trace_8b_eager.json.gz / trace_8b_graph.json.gz | 测试夹具：夹具负载那台 Qwen3-8B 在峰值 bs 206 时的 10 步 decode，eager 和 graph 各一份，GPU 前向盖住发射的情形 |
| test_obs.py | 41 个边界用例 |

夹具的负载：Qwen3-8B / 1×H100，输入 1024-2048、输出 128-256；0-45 s 每秒 8 条，45-90 s 每秒 14 条；decode 图上限 160。后半段到达率上来，prefill 占掉 65% 的 GPU 时间，decode batch 涨到 206 并超过 160 掉回 eager。指标和日志里排队、变慢和 cuda graph False 同时出现，看上去像是 eager 拖慢了前向；8B 的那对 trace 说明 eager 在这里几乎不花钱，真正的原因是容量。相关性靠指标和日志找，因果要靠 trace 定。

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `parse` / `quantile` | exposition 格式见 prometheus_client `exposition.generate_latest`；分位数插值同 Prometheus `bucketQuantile`；指标定义在 `srt/observability/metrics_collector.py:365-1290`（调度器）、`:1803-2075`（tokenizer） |
| 窗口长度 / 调度阶段 | `srt/observability/scheduler_stage_metrics.py:41` 阶段枚举、`:128` profiler 跨度名 `scheduler.{stage}` |
| `EAGER_DECODE` | `cuda_graph_passes_total` 的 mode 定义在 `srt/observability/metrics_collector.py:1527-1533`；decode 计数在 `srt/managers/scheduler_components/batch_result_processor.py:994`，prefill 在 `srt/managers/scheduler_components/metrics_reporter.py:807` |
| `NO_DEVICE_TIMER` | `forward_execution_seconds_total` 要 `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` |
| `UTILIZATION_ZERO` | `srt/managers/scheduler_components/metrics_reporter.py:1307` 的 TODO |
| `BUCKET_COARSE` | 可配的只有 `--bucket-time-to-first-token`、`--bucket-inter-token-latency`、`--bucket-e2e-request-latency`；queue_time 桶写死在 `metrics_collector.py:838` |
| `REQ` / `KV` | `srt/managers/schedule_batch.py:2206-2225` 前缀，`srt/observability/req_time_stats.py:1099-1200` 正文 |
| `PREFILL_RETRY` | attempts 即 `prefill_attempt_count`（`schedule_batch.py:1449`），只在 PD prefill 的乐观预填里自增：`srt/disaggregation/prefill.py:531`、`:542`、`:1843` |
| `BATCH` / `FIELD` | `srt/managers/scheduler_components/metrics_reporter.py:753`、`:962` |
| `STEP` | `srt/utils/profile_utils.py:462`（`detailed_annotations` 打开后带 g_sq / g_sk） |
| `scheduler.*` 颜色 | `srt/utils/nvtx_utils.py:60-64` |
| `profile.sh load` | `/start_profile` 参数见 `docs/docs/developer_guide/benchmark_and_profiling.mdx:182-260`；start_step 处理见 `srt/managers/scheduler_components/profiler_manager.py:143` |
| `profile.sh nsys` | `benchmark_and_profiling.mdx:466-551`；`--disable-cuda-graph` 已弃用（`srt/server_args.py:411`），改用 `--cuda-graph-backend-decode/prefill disabled`（`srt/arg_groups/fields/exec_.py:616-629`）；`--enable-layerwise-nvtx-marker` 在同文件 `:715` |
| `whatif.py` | mixed 的开关在 `srt/managers/scheduler.py:1379`，拼批在 `:4440`，带 logprob 的批不混（SGL-06）；多副本轮询对应仓库根下 `sgl-model-gateway/src/policies/round_robin.rs`（SGL-12） |
| `make_fixtures.py` | 复用 `../03-bench/sim_serve.py`、`../09-overlap-cudagraph/cost.py`、`../02-ledger` |

## 运行

```bash
cd zero-to-sglang/18-observability
python3 scrape_metrics.py metrics_t0.txt                       # 单次抓取：启动以来的累计
python3 scrape_metrics.py metrics_t0.txt metrics_t1.txt        # 两次抓取：只看中间 45 s
python3 scrape_metrics.py metrics_t0.txt metrics_t1.txt --json
python3 scrape_metrics.py http://127.0.0.1:30000/metrics --interval 60   # 真机：现抓两次
python3 reqstats.py server_reqstats.log --top 3 --bin 10
python3 trace_summary.py trace_eager.json.gz trace_graph.json.gz          # 0.6B：eager 吃亏
python3 trace_summary.py trace_8b_eager.json.gz trace_8b_graph.json.gz --top 4   # 8B bs 206：eager 不吃亏
python3 whatif.py                                              # 几种容量办法各改善多少
python3 make_fixtures.py                                       # 重造夹具
python3 -m unittest test_obs -v                                # 约 1 秒

# 真机（需要 GPU 和 sglang）
bash profile.sh serve        # 终端 1
bash profile.sh load         # 终端 2，结束后
bash profile.sh analyze
```

## 期望输出

```text
$ python3 scrape_metrics.py metrics_t0.txt metrics_t1.txt（节选）
== 窗口 ==
  两次抓取之差，时长 45.0 s（scheduler_stage_seconds_total 各阶段之和）
== 吞吐 ==
  完成请求 475 条，10.56 条/s；输出 2028 token/s（按完成请求计）
  实时：prefill 计算 22443 token/s，decode 2365 token/s，前缀命中 0 token/s
== 延迟（直方图估计）==
  指标                  样本        均值       p50       p90       p99   p99 所在桶
  TTFT               643    605 ms    592 ms    1.58 s    1.96 s   (1, 2]
  ITL             106412     70 ms     30 ms    165 ms    388 ms   (0.2, 0.4]
  TPOT（逐请求）      475     68 ms     68 ms     79 ms     96 ms   (0.08, 0.1]
  E2E                475   13.49 s   14.14 s   19.01 s   29.44 s   (20, 40]
  排队               643    508 ms    496 ms    1.48 s    1.95 s   (1, 2]
  prefill 前向       643     97 ms     87 ms    194 ms    323 ms   (0.201681, 0.326723]
== 调度循环的时间去向（scheduler_stage_seconds_total）==
  process_batch_result        41.77 s   92.8%
  run_batch                    2.44 s    5.4%
  get_next_batch_to_run        0.70 s    1.6%
== CUDA graph ==
  decode_cuda_graph 316，decode_none 384，prefill_cuda_graph 405；decode 走 eager 54.9%
== 诊断 ==
  [调整] QUEUE_DOMINANT：TTFT p99 1.96 s，其中排队 p99 1.95 s（100%）。…
  [提示] ITL_SPIKES：ITL p99 388 ms 是逐请求 TPOT p99 96 ms 的 4.1 倍：…
  [提示] EAGER_DECODE：55% 的 decode 前向没走 CUDA graph（cuda_graph_passes_total mode=decode_none），…吃不吃亏看 GPU 前向和 CPU 发射谁长…
  [提示] NO_DEVICE_TIMER：sglang:forward_execution_seconds_total 不在，fwd_occupancy 是 NaN：…
  [提示] UTILIZATION_ZERO：sglang:utilization 恒为 0：…

$ python3 reqstats.py server_reqstats.log --top 3 --bin 10（节选）
== 分布（秒）==
                 p50     p90     p99     max
  排队          0.00    1.09    1.61    1.71
  前向          9.41   16.57   19.38   20.57
  合计          9.73   17.19   19.98   21.45
  最慢 1% 的请求里，排队占 5%
== 最慢的几条，和它进来时的服务端状态 ==
  rid=3990ea27a157… 输入 1188 输出 253：排队 1.71 s + 前向 19.74 s = 21.45 s，进入于 06:01:08
    最近的 Decode 行：#running-req 206，#queue-req 15，token usage 0.94，cuda graph False
    最近的 Prefill 行：#queue-req 22，#pending-token 34578
== 按进入时刻分桶 ==
    t s    请求    排队 p50    排队 max    前向 p50  running  queue  usage  graph
   50.0   147      0.00      0.00     14.65      141      0   0.64  True
   60.0   165      0.88      1.71     14.88      207     26   0.95  False
   70.0   132      0.90      1.65     12.93      206     12   0.94  False
== 诊断 ==
  [提示] SLOW_BY_FORWARD：最慢 1% 的请求里排队只占 5%，慢在前向：…
  [调整] EAGER_WINDOW：3 个时间桶里最近一行 Decode batch 是 cuda graph: False，从第 60.0 s 起；这段时间进来的请求前向中位数 12.93 s，之前是 2.92 s。前向变慢也可能只是 batch 变大、prefill 插得多，eager 是不是原因要抓 trace 看

$ python3 trace_summary.py trace_eager.json.gz trace_graph.json.gz（节选）
== trace_eager.json.gz ==
  稳态每步：墙钟 2.98 ms，GPU 忙 1.23 ms，GPU 空闲 59%，每步 398 个 kernel
  kernel 排行（共 12.25 ms GPU 时间，<3 µs 的小 kernel 占个数的 86%）
    attention_decode               280 次      5.93 ms   48.4%  均    21.2 µs
    gate_up_proj_gemm              280 次      1.11 ms    9.1%  均     4.0 µs
    lm_head_gemm                    10 次      0.98 ms    8.0%  均    98.1 µs
  CPU 侧（占跨度）：scheduler.run_batch 81%，…；发射调用 40%，同步等待 0%
  [调整] LAUNCH_BOUND：GPU 空闲 59%，71% 的 kernel 在 launch 返回后 2 µs 内就开跑：…
== trace_graph.json.gz ==
  稳态每步：墙钟 1.26 ms，GPU 忙 1.23 ms，GPU 空闲 3%，每步 398 个 kernel
  CPU 侧（占跨度）：scheduler.process_batch_result 62%，…；发射调用 1%，同步等待 42%
  [提示] GPU_BOUND：CUDA graph 回放，GPU 空闲 3%：GPU 是瓶颈，往下看 kernel 排行
== 对比 ==
  步长比 2.36x，GPU 忙碌时间比 1.00x（kernel 本身没变，差在空隙）

$ python3 trace_summary.py trace_8b_eager.json.gz trace_8b_graph.json.gz --top 4（节选）
== trace_8b_eager.json.gz ==
  稳态每步：墙钟 18.40 ms，GPU 忙 18.40 ms，GPU 空闲 0%，每步 510 个 kernel
    attention_decode               360 次    121.59 ms   66.1%  均   337.7 µs
  [提示] GPU_BOUND：GPU 空闲 0%：GPU 是瓶颈，往下看 kernel 排行，对照 Roofline 判断是带宽还是算力
== trace_8b_graph.json.gz ==
  稳态每步：墙钟 18.63 ms，GPU 忙 18.40 ms，GPU 空闲 1%，每步 510 个 kernel
== 对比 ==
  步长比 0.99x，GPU 忙碌时间比 1.00x（kernel 本身没变，差在空隙）

$ python3 whatif.py
同一份到达序列，只看 45-90 s 进来的请求（秒；ITL 毫秒）
  办法                卡  TTFT p99  排队 p99   ITL p99   ITL max  前向 p50   E2E p99  prefill 占比  bs max  超图上限
  原样                 1      1.74      1.62       272      1711     12.47     20.43           65%     208       55%
  开 mixed             1      0.48      0.24       225       240      6.58     12.39           67%     166        2%
  chunk 2048 + mixed   1      0.62      0.50        63        63      5.62     10.17           68%     136        0%
  两副本轮询           2      0.08      0.00        92       268      2.49      3.92           32%      39        0%
  限流 10 条/s         1      0.14      0.00       166       396      3.77      5.38           46%      50        0%
```

时间都是模型推算，用来比方向和量级。夹具按 8fabf5e 的格式构造，不是实机录制；拿到自己的实机数据以后，以实机为准。

## 真机采集的几个坑

- `sglang:utilization` 在 8fabf5e 上恒为 0，`fwd_occupancy` 不设 `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` 是 NaN，两者都别直接拿来做告警。
- `/start_profile` 不给 `output_dir` 时写到 `SGLANG_TORCH_PROFILER_DIR`，再没有就写 `/tmp`。trace 抓满 `num_steps` 后才落盘，`analyze` 找不到文件时先等一会儿。
- 想按层看 kernel 只能在 eager 下：CUDA graph 回放的 kernel 不再发 NVTX。`profile.sh nsys` 关掉了图，测到的步长比线上慢，只用来看层内 kernel 的相对占比。
- 多 TP rank 时每个 rank 一份 trace，先看 TP-0；`merge_profiles: true` 能合成一份，文件会大很多。
