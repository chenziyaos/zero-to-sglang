# SGL-15 PD 分离

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/15-pd-disagg.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-15｜原理｜PD 分离：prefill 和 decode 拆到两组卡上。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| pd_sim.py | 每条请求在每个 rank 上要搬的 KV 字节，四种链路的整段耗时和 chunk 流水后暴露的尾巴；描述符计数：src、dst 两边同时连续才合成一段，need_sort 开关对碎片段数的影响，异构 TP 走 slice 时描述符数暴涨；按 TTFT / TPOT SLO 算 P:D 配比和每卡 goodput；两卡合并部署和 1P1D 的事件模拟（长 prompt 混合、短请求、TCP）；decode 准入预算（每个活跃请求预留 512、FIFO 队头阻塞、retract 回来的优先）；一条请求的分块发送边界、staging 格子、KVPoll 取 MIN，以及 early 和 prefill_complete 两种策略各占槽多久 |
| resolve_sim.py | `--disaggregation-mode` 相关参数解析链的移植：mooncake_tcp 改写、decode 侧默认关 radix、extra_slots 默认值、prefill_complete 不支持的特性清单、staging 的后端限制、role switch 的拒绝清单、host receive 检查；Mooncake 传输线程池大小，以及它和文档公式的出入 |
| test_pd.py | 60 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `rank_mapping` | `srt/disaggregation/common/conn.py:1432` _resolve_rank_mapping；D 侧 TP 小且是 MLA 时只从一个 P rank 取，其余 rank 发 dummy |
| `group_contiguous` | `srt/disaggregation/common/utils.py:132` group_concurrent_contiguous |
| `descriptors` | `srt/disaggregation/mooncake/conn.py:1028` _send_kvcache_generic（每段每层一个，MHA 的 K、V 各一个）；`:1656` send_kvcache_slice（每个 token 槽每层一个） |
| `head_slice` / `staging_layout` / `grid_*` | `srt/disaggregation/common/staging_buffer.py:773` compute_head_slice_params、`:809` compute_staging_layout、`:860` staging_grid_tokens / compute_grid_segments |
| `send_end` / `early_send_end` | `srt/disaggregation/prefill.py:1528` _send_kv_chunk、`:1460` maybe_send_cached_prefix_chunk |
| `DecodeAdmission` / `max_admit` | `srt/disaggregation/decode.py:511` _required_admission_tokens、`:1336` pop_preallocated、`:2030-2140` _need_space_for_single_req / _active_req_count / _allocatable_token_budgets |
| `kvpoll_min` | `srt/disaggregation/utils.py:300` _all_reduce_polls |
| `TokenAllocator(need_sort=True)` | `srt/mem_cache/allocator/token.py`：free 的页先进 release_pages，不够分配时再合并排序 |
| `resolve` | `srt/arg_groups/disagg.py:55-185` 参数和默认值；`srt/arg_groups/pd_disaggregation_hook.py:23` handle_pd_disaggregation、`:199` _validate_prefill_complete |
| `thread_pool` | `srt/disaggregation/mooncake/conn.py:328` 传输线程池；`srt/environ.py:755-782` 队列数、超时、心跳 |
| `run_colo` | 复用 06-chunked-prefill 的调度器 |

## 运行

```bash
cd zero-to-sglang/15-pd-disagg
python3 pd_sim.py                           # 32K 请求的 KV 字节、各链路耗时、流水尾巴、描述符数
python3 pd_sim.py --input 8192              # 换输入长度
python3 pd_sim.py --scenario ratio          # 四种负载的 P:D 配比和每卡 goodput
python3 pd_sim.py --scenario colo           # 合并部署 vs 1P1D，三种负载（约 1 秒）
python3 pd_sim.py --scenario admission      # decode 准入预算和队头阻塞
python3 pd_sim.py --scenario timeline       # 一条请求的发送边界、KVPoll、两种分配策略
python3 resolve_sim.py                      # 16 条命令行的解析结果和告警
python3 resolve_sim.py --scenario threads   # 传输线程池大小
python3 -m unittest test_pd -v              # 约 2 秒
```

## 期望输出

```text
$ python3 pd_sim.py（节选）
  模型 × 卡                      cell B  每rank MB prefill ms       NVLink 机内     RDMA 400G/卡     RDMA 200G/卡    TCP 100GbE÷8
  Llama-3.1-8B 1×H100         131072      4295       1419    28.6/7.2       107.4/26.8      214.7/53.7     3579.1/2417.7
  Llama-3.1-70B 4×H100         81920      2684       2743    17.9/4.5        67.1/16.8      134.2/33.6     2237.0/559.2
  DeepSeek-V3 8×H200           70272      2303       1320    15.4/3.8        57.6/14.4      115.1/28.8     1918.9/732.3
     2048 token  prefill     134 ms  KV    168 MB  暴露尾巴   4.2 ms
   131072 token  prefill   18733 ms  KV  10737 MB  暴露尾巴  16.8 ms
  跑了一阵，need_sort=True（PD 的设置）：平均 18 段、最多 26 段 -> 1152 个
  跑了一阵，need_sort=False：平均 71 段、最多 85 段 -> 4544 个
  P TP2 -> D TP1 走 send_kvcache_slice：每个 P rank 4096 × 64 = 262,144 个，每个只有 1024 B

$ python3 pd_sim.py --scenario ratio
  负载                   输入/输出           SLO   P 上限 rps    D 并发   D 上限 rps    P:D  PD rps/卡   合并 rps/卡   PD/合并
  聊天              1024/256         1s/50ms       14.6     233       37.1   2.54      2.62       2.20    1.19
  RAG 摘要          8192/256         2s/50ms        1.5      46        6.3   4.27      0.30       0.27    1.11
  代码补全长输出          512/1024      0.5s/30ms       29.4     203        8.9   0.30      1.70       1.73    0.98
  长上下文 agent     32768/512         4s/50ms        0.2      12        0.8   4.67      0.04       0.04    0.84

$ python3 pd_sim.py --scenario colo（节选）
  部署                  短TTFT p50  短TTFT p99    长TTFT  ITL p50  ITL p99  ITL max out tok/s    prefill 占时
  合并部署 ×2                    33       1430     1424      7.1     36.2   1629.5      1412        每卡 19%
  1P1D RDMA 400G/卡           38       1560     1473      7.6      9.5      9.7      1410       P 卡 39%
  （短请求 40 req/s）
  合并部署 ×2                    12         22      nan      9.8     24.3     32.5     17309        每卡 12%
  1P1D RDMA 400G/卡           16         33      nan     14.3     15.4     15.5     16540       P 卡 24%
  （第一种负载，链路换成 TCP）
  1P1D TCP 100GbE÷8        8068      14755     9944      7.8      9.1      9.1      1151       P 卡 32%

$ python3 pd_sim.py --scenario admission（节选）
  waiting_queue   0 条：能再预分配  35 条 4096 token 的请求（起步预留 512 × 100 = 51,200 token）
  waiting_queue 300 条：能再预分配   2 条 4096 token 的请求（起步预留 512 × 400 = 204,800 token）
  放进 0 条，队头 100000 token 要 100,512，预算只有 90,000；后面 10 条 2000 token 的请求本来放得下，也在等。
  同样的请求换个顺序（大请求排最后）：放进 10 条。

$ python3 pd_sim.py --scenario timeline（节选）
  中间块  extend_end  10192  发 [     0,  10176)  扣下 16 个不满一页的 token
  P 排队   10 s：early 占槽  11.53 s（    231 k token·s）  prefill_complete 占槽  0.04 s

$ python3 resolve_sim.py（节选）
D 默认                          no_radix: False -> True  extra_slots: None -> 0
P mooncake_tcp                backend: mooncake_tcp -> mooncake  ib_device: mlx5_0 -> None  MC_FORCE_TCP=1
P prefill_complete TP4        启动失败：... does not yet support attention tensor parallelism greater than one

$ python3 resolve_sim.py --scenario threads
  CPU 核数       代码里的线程数      文档公式
      64             4         6
     128             8        12
     192            12        12
```
