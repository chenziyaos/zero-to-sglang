# SGL-17 参数调优手册

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/17-tuning.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-17｜实战｜参数调优手册：从日志倒推配置。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| parse_log.py | 读服务日志：server_args（dict 和旧版 ServerArgs repr）、启动行、decode 图 bs、Prefill / Decode batch 字段（含 SWA、mamba、投机接受长度）、retract、abort、OOM、max_running 被压低、max_new_tokens 截断；算稳态分位数，按 24 种诊断码给诊断和建议参数，`--json` 输出机器可读结果 |
| tune_sim.py | 调参前先算账：各 GPU 的默认 chunk / max_bs / 预留 / frac；frac 每 +0.01 换多少 KV、启动余量落在哪一档；max_running_requests 的默认值和被压低的条件、req_to_token 的显存；早停负载和长输出负载下 max_tokens 与 conservativeness 的方向；chunk 与 mixed 对 KV、TTFT、ITL 的影响；decode 图上限不够时掉回 eager 的代价和调大的显存账 |
| sample_before.log / sample_after.log | 测试夹具：按 8fabf5e 的日志格式构造，数值取 H100 + Qwen3-8B 的推算量级，不是实机录制 |
| sim_admit.log | `tune_sim.py --scenario admit --dump` 写出的模拟调度日志，parse_log 也能读 |
| tuning_record.md | 调参记录模板，带一次示范 |
| test_tuning.py | 83 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `ARGS_DICT` / `parse_server_args` | `srt/entrypoints/engine.py:173` _server_args_for_logging、`:311` |
| `STARTUP` | `srt/managers/scheduler.py:1272`，available_gpu_mem 在图捕获之后量 |
| `DECODE_CAPTURE` | `srt/model_executor/model_runner_components/cuda_graph_setup.py:752-791` |
| `BATCH` / `parse_fields` / `usage_of` | `srt/managers/scheduler_components/metrics_reporter.py:745-776`、`:963-1040`；`srt/managers/scheduler_components/pool_stats_observer.py:104-150` |
| `RETRACT` | `srt/managers/scheduler.py:4590-4601` |
| `ABORT_LAST` / `ABORT_REQ` | `srt/managers/schedule_batch.py:3512`、`:3536` |
| `REDUCED` / `resolve_max_num_reqs` | `srt/mem_cache/kv_cache_configurator.py:3119-3172` |
| `CAPPING` / `KV_BUDGET` | `srt/managers/scheduler.py:2763`、`:2792` |
| `OOM` | `srt/mem_cache/allocation.py:239`、`:613` |
| `RULES` | `docs/docs/advanced_features/hyperparameter_tuning.mdx` 的阈值；`lpm_queue` 来自 `srt/managers/schedule_policy.py:314` |
| `scenario_tier` / `avail_estimate` | `srt/arg_groups/memory_hook.py:89-170`、`:281-330`、`:357` reserve_for_graph_mb；复用 `../09-overlap-cudagraph/cuda_graph_sim.py` |
| `scenario_frac` | `srt/mem_cache/kv_cache_configurator.py:2990` _profile_available_bytes；复用 `../02-ledger/sim_ledger.py` |
| `scenario_admit` | `srt/managers/schedule_policy.py:795`、`:1405`；`srt/managers/scheduler.py:4591`；复用 `../05-scheduler/sim_scheduler.py` |
| `scenario_chunk` | 复用 `../06-chunked-prefill/sim_chunked.py` |
| `scenario_graph` | 复用 `../09-overlap-cudagraph/sim_overlap.py` |

## 运行

```bash
cd zero-to-sglang/17-tuning
python3 parse_log.py sample_before.log          # 调参前：指标 + 诊断 + 建议参数
python3 parse_log.py sample_after.log           # 调参后：全部 [正常]
python3 parse_log.py sample_before.log --json   # 机器可读
python3 parse_log.py server.log --gpu-gib 80    # 自己的日志；没有 Load weight begin 行时手给显存
python3 tune_sim.py                             # 各 GPU 的默认值
python3 tune_sim.py --scenario frac             # frac 每 +0.01 换多少 KV
python3 tune_sim.py --scenario running          # max_running_requests 的默认值和压低条件
python3 tune_sim.py --scenario admit            # 准入松紧的两个方向
python3 tune_sim.py --scenario admit --dump     # 再写出 sim_admit.log
python3 parse_log.py sim_admit.log
python3 tune_sim.py --scenario chunk            # chunk 与 mixed
python3 tune_sim.py --scenario graph            # 图上限与 eager 回退
python3 -m unittest test_tuning -v              # 约 8 秒
```

## 期望输出

```text
$ python3 parse_log.py sample_before.log（节选）
== 稳态（#running-req > 0 的 decode 行）==
  120 行 decode、30 行 prefill，跨度 600 s
  #running-req p50/p90/max = 271/307/336；#queue-req p50/max = 270/439，queue=0 占 3%
  token usage p50/max = 0.74/0.86；usage<0.9 且有排队占 97%
  cuda graph=False 占 65%（这些行 running ≥ 259）
  gen throughput p50 = 12080 token/s；retract 1 次，0.10 次/分
== 诊断 ==
  [调整] AVAIL_HIGH：启动时 available_gpu_mem=12.26 GB，官方建议 5 - 8 GB。mem_fraction_static 0.83 -> 0.89 能多给 KV 约 4.8 GB；…
  [调整] ADMIT_TIGHT：97% 的 decode 行 token usage < 0.9 且队列里有请求，usage 中位数 0.74：…
  [调整] GRAPH_FALLBACK：65% 的 decode 行 cuda graph=False，这些行 #running-req 都大于 256（decode 图的最大 bs）：…
  [提示] LPM_FALLBACK：schedule_policy=lpm，但 89% 的时间队列超过 128 条，这时调度器退回 FCFS …
== 建议改动 ==
  --mem-fraction-static 0.89
  --schedule-conservativeness 0.3
  --cuda-graph-max-bs-decode 320

$ python3 parse_log.py sample_after.log（节选）
  #running-req p50/p90/max = 364/373/381；#queue-req p50/max = 110/211，queue=0 占 2%
  token usage p50/max = 0.91/0.97；usage<0.9 且有排队占 36%
  cuda graph=False 占 0%
  gen throughput p50 = 13177 token/s；retract 2 次，0.20 次/分
== 诊断 ==
  [正常] AVAIL_OK：available_gpu_mem=7.27 GB，在 5 - 8 GB 内
  [正常] RETRACT_OK：retract 2 次（0.2 次/分），官方认为每分钟一次以内可以接受
  [正常] USAGE_OK：token usage 中位数 0.91，KV 池用满了

$ python3 tune_sim.py --scenario frac（节选）
    frac  KV 池 GiB   KV token     比自动值   预计 avail GB  档位
   0.830     50.35    366,669    +0.0%         12.48  偏松，KV 池少拿了
   0.880     54.31    395,449    +7.8%          8.53  略松，可再 +0.01
   0.890     55.10    401,205    +9.4%          7.74  合适
   0.920     57.47    418,473   +14.1%          5.37  合适
   0.930     58.26    424,229   +15.7%          4.58  偏紧，长 prompt、logprob 请求易 OOM
  每 +0.01 多 5,756 token（0.79 GiB / 144 KB 每 token）

$ python3 tune_sim.py --scenario running（节选）
  场景                            KV token  context     手动给     结果   reduced 警告 req_to_token
  qwen3-8b / H100 默认             366,669    40960       -   4096                    0.63 G
  llama-8b / H100 默认（128k）       415,401   131072       -   2048                    1.00 G
  小池子 6000 token 给 4096            6,000    40960    4096   3000            是       0.46 G

$ python3 tune_sim.py --scenario admit
   max_tokens conserv decode usage  峰值 running    总时长 s  输出 tok/s  TTFT 均值 s retract
          512     0.3        0.878          65    63.38      2895      28.71       1
          512     1.0        0.869          64    63.58      2885      29.04       0
         4096     0.3        0.746          54    67.34      2724      31.25       0
         4096     1.0        0.626          46    72.38      2535      34.25       0
          未设置     0.3        0.744          54    67.40      2722      31.27       0
          未设置     1.0        0.622          45    72.61      2526      34.33       0
   conserv  retract 次     退回请求   重算 token decode usage  输出 tok/s
       0.3         22       22     28,604        0.790      5404
       1.0          7        7      5,897        0.760      5372
       1.3          0        0          0        0.749      5351

$ python3 tune_sim.py --scenario chunk
   chunk mixed   frac  KV token  长 TTFT ms  ITL p99  ITL max   块数
    2048     关  0.874   391,995       1499     13.6   1512.1   16
    2048     开  0.874   391,995       1533    110.7    123.5   17
    4096     关  0.874   391,995       1503     13.6   1509.7    8
    4096     开  0.874   391,995       1516    176.1    246.6    9
    8192     关  0.830   366,669       1499     13.6   1508.5    4
    8192     开  0.830   366,669       1510     18.3    479.1    5
   16384     关  0.677   278,601       1497     13.6   1507.9    2
   16384     开  0.677   278,601       1505     13.6    889.5    3

$ python3 tune_sim.py --scenario graph（节选）
  preset          每 kernel µs   bs  eager 发射   GPU 前向  eager 步长  graph 步长     慢多少
  qwen3-0.6b               15   64     5.97ms    3.47ms     7.14ms     3.47ms    106%
  qwen3-8b                  6  384     3.06ms   29.10ms    29.10ms    29.10ms      0%
  llama-70b-tp8            15  384    19.29ms   15.63ms    24.30ms    15.63ms     55%
  qwen3-8b / H100：256 → 384，多预留 256 MB，约 1,820 个 KV token（池子共 366,669）
```

时间和显存都是模型推算，用来比方向和量级。两份 sample 日志是按格式构造的夹具；拿到自己的实机日志以后，以实机为准。

## 真机压测的一个坑

`python3 -m sglang.benchmark.serving` 默认给每条请求带 `ignore_eos=True`（serving.py:289、:628），每条请求都吐满 output_len，准入估算里 ratio 恒为 1，conservativeness 几乎测不出差别。测早停负载要加 `--disable-ignore-eos`，并让请求里的 max_tokens 与实际输出分开，例如：

```bash
python3 -m sglang.benchmark.serving --backend sglang-oai --port 30000 \
  --dataset-name sharegpt --num-prompts 4000 --request-rate 30 \
  --disable-ignore-eos --extra-request-body '{"max_tokens": 4096}'
python3 parse_log.py server.log
```
