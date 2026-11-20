# SGL-09 Overlap 调度与 CUDA Graph

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/09-overlap-cudagraph.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-09｜原理｜Overlap 调度与 CUDA Graph：把 CPU 开销藏到 GPU 后面。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| cost.py | 一步 decode 的时间账：Roofline 粗估 GPU 前向，加上调度、结果处理、kernel 启动、D2H 几项 CPU 开销 |
| sim_overlap.py | 事件时间线：普通循环 vs overlap 循环；连续 prefill、在途结束、FutureMap relay、WAR barrier 几个副作用 |
| cuda_graph_sim.py | CUDA Graph：捕获列表、对齐过滤、补齐浪费、按显存分档的默认值和预留、启动开销、捕获顺序、补齐行策略、掉回 eager 的条件 |
| test_overlap.py | 18 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/srt/`。

| 本目录 | sglang |
|---|---|
| `simulate(overlap=False)` | `managers/scheduler.py:2028` event_loop_normal |
| `simulate(overlap=True)` | `managers/scheduler.py:2064` event_loop_overlap：先 run_batch(N+1)，再 pop_and_process(N) |
| 结果处理前的阻塞 W | `managers/scheduler_components/batch_result_processor.py:268` 等处的 `copy_done.synchronize()` |
| `scenario_prefill` | `managers/scheduler.py:2139` is_disable_overlap_for_batch；`environ.py:697` SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP |
| `scenario_finishing` 剔除 | `managers/scheduler.py:4509` _reqs_finishing_inflight、`:4524` update_running_batch；`managers/schedule_batch.py:1668` next_output_finishes_by_length |
| `scenario_finishing` 跳一步 | `managers/scheduler.py:4524` KV 不够且有在途结束时返回 None，不 retract |
| `scenario_relay` GPU relay | `managers/overlap_utils.py:113` resolve_forward_inputs、`:281` FutureMap、`:649` publish |
| `scenario_relay` slot 0 | `mem_cache/memory_pool.py:400` 多留的 padding 行 |
| `scenario_streams` | `managers/scheduler.py:2015` _apply_war_barrier；`model_executor/runner/decode_cuda_graph_runner.py:715` _publish_read_done |
| `decode_bs` | `arg_groups/cuda_graph_hook.py:500` generate_decode_cuda_graph_batch_sizes |
| `prefill_tokens` | `arg_groups/cuda_graph_hook.py:480` generate_prefill_cuda_graph_batch_sizes |
| `to_capture` | `model_executor/runner/base_cuda_graph_runner.py:65` get_batch_sizes_to_capture；`utils/common.py:4168` 对齐倍数 |
| `pad_to` | `model_executor/runner/base_cuda_graph_runner.py:166` _pad_to_bucket |
| `tier` | `arg_groups/memory_hook.py:89-170` 按显存分档 |
| `mem_fraction` | `arg_groups/memory_hook.py:281-328` 预留公式、`:357` reserve_for_graph_mb |
| `scenario_capture` | `model_executor/runner/decode_cuda_graph_runner.py:1358` reversed(capture_bs)；`model_executor/runner_backend/full_cuda_graph_backend.py:127` capture_session |
| `scenario_padding` | `model_executor/cuda_graph_buffer_registry.py:105` PaddingPolicy |
| `scenario_fallback` | `model_executor/runner/decode_cuda_graph_runner.py:877` can_run_graph |
| `decode_bs(mode='mini')` | mini-sglang `python/minisgl/engine/graph.py:49` |
| 重复释放保护 | mini-sglang `python/minisgl/scheduler/scheduler.py:138` |

## 运行

```bash
cd zero-to-sglang/09-overlap-cudagraph
python3 sim_overlap.py                          # 普通 vs overlap 的时间线
python3 sim_overlap.py --scenario sweep         # 模型 × batch × eager/graph
python3 sim_overlap.py --scenario prefill       # 连续 prefill 关 overlap 的 TTFT 和吞吐
python3 sim_overlap.py --scenario finishing     # 在途结束、跳一步、重复释放
python3 sim_overlap.py --scenario relay         # FutureMap relay 和 padding 行
python3 sim_overlap.py --scenario streams       # WAR barrier
python3 cuda_graph_sim.py                       # 捕获列表和对齐过滤
python3 cuda_graph_sim.py --scenario waste      # 补齐浪费
python3 cuda_graph_sim.py --scenario defaults   # 各 GPU 的默认 max_bs 和 mem_fraction_static
python3 cuda_graph_sim.py --scenario launch     # 启动开销
python3 cuda_graph_sim.py --scenario capture    # 捕获顺序和内存池
python3 cuda_graph_sim.py --scenario padding    # 补齐行策略
python3 cuda_graph_sim.py --scenario fallback   # 掉回 eager
python3 -m unittest test_overlap -v
```

全部场景都在几秒内跑完。

## 期望输出

```text
$ python3 sim_overlap.py
  一步的账（cost.py 的估算）：GPU 前向 7.95 ms，调度 0.38 ms，graph replay 0.08 ms，D2H 0.02 ms，结果处理 0.41 ms
  普通循环：稳态步长 8.83 ms，GPU 忙碌 90.7%，吞吐 3623 tok/s
  overlap：稳态步长 7.95 ms，GPU 忙碌 100.0%，吞吐 4026 tok/s
  overlap 稳态加速 1.11x。第 1 个 token 可见时刻：普通 8.83 ms，overlap 8.83 ms

$ python3 sim_overlap.py --scenario sweep（节选）
  qwen3-0.6b         1  eager     0.52     2.39    2.82    81%     2.80    81%  1.01x
  qwen3-0.6b         1  graph     0.52     0.08    1.04    48%     0.52     0%  1.98x
  qwen3-8b          64  graph     9.87     0.08   11.14    11%     9.87     0%  1.13x
  llama-70b-tp8    256  graph    13.11     0.08   16.69    21%    13.11     0%  1.27x

$ python3 sim_overlap.py --scenario relay
  GPU relay：[349, 827, 645, 3, 101] …  和普通循环一致：True
  CPU 拿：  [349, 349, 827, 827, 645] …  从第 2 步开始错
    req_pool_indices 尾部 ZERO ：padding 行写 slot 0，E 的下一个输入 正确
    req_pool_indices 尾部 stale：padding 行写 slot 5，E 的下一个输入 被覆盖成 536（应为 667）

$ python3 cuda_graph_sim.py --scenario defaults（节选）
  GPU            MiB  TP  chunk  max_bs    预留 MiB     其中图   10G 保底 mem_frac
  RTX 4090     24564   1   4096      48      7280     496             0.704
  H100-SXM     81559   1   8192     256     13904     976             0.830
  H100-SXM     81559   8   8192     512     15312    1488             0.812
  B200        183359   1  16384     512     26832    1616             0.854
```

## 校准

模拟里 CPU 那几项（调度、结果处理、每个 kernel 的发射开销、graph replay）都是假设值，换到自己的机器上要量：

1. nsys 抓 decode 稳态：`nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --delay 60 --duration 30 python3 -m sglang.launch_server ...`，同时用 bench_serving 打稳定负载。看 GPU 行上 decode 步之间的空隙，和 CPU 行上 run_batch 之外的时间。
2. 加 `--disable-overlap-schedule` 再抓一次，两次的步长差就是 overlap 藏住的那部分。
3. 加 `--cuda-graph-backend-decode disabled` 再抓一次，数一步里 cudaLaunchKernel 的个数和总耗时，填回 cost.py 的 kernels_per_layer 和 op_us。
4. 启动日志里 `Capture target decode cuda graph end ... mem usage=` 是图实际占的显存，和 `mem_fraction` 算出来的 decode_graph 预留对比。

只看吞吐的话，用同一份 bench_serving 参数在开、关 `--disable-overlap-schedule` 下各跑一次，比较 output throughput 和 ITL；模拟给的倍数只能说明趋势，以实测为准。
