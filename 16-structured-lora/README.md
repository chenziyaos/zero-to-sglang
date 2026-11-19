# SGL-16 结构化输出与 LoRA

对应飞书文档：SGL-16｜原理｜结构化输出与多 LoRA：约束解码和 adapter 批处理的代价。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| struct_sim.py | 各模型每行 token mask 的字节数（int32 位图 vs bool）、bs=256 的 H2D 量，逐行填 mask 和一步 decode 谁长；玩具正则语法的逐 token mask、跨语法单元的 token、rollback、投机草稿树 traverse_tree；编译线程池排队、同 schema 并发不命中导致的重复编译、json.dumps 不排序导致的缓存不命中、TP 间 ready 交集 / failed 并集、超时时长随 batch 变长；同步与推迟采样两种时间线下每步 ms；思考预算、多 token think_end 的 KMP 匹配、思考段屏蔽工具调用 token、跨 think_end 回滚 |
| lora_sim.py | LoRA 池按 rank × mlpb × 模块一次分配，折成少了多少 KV token；一轮调度里谁进谁被跳过、pin 占槽、LRU 和 FIFO 在 Zipf 流量下的换入次数；csgmv 的排列与分段 vs triton 按请求发块；冷 adapter 在热流量下饿死，开 drainer 前后的等待；同步加载和 side stream 异步加载的代价 |
| resolve_sim.py | grammar backend 默认值、strict thinking 和 grammar backend / UNO 的冲突、`check_lora_server_args` 和 `check_lora_speculative_compatibility` 的移植、pinned adapter 上限；请求级 `json_schema` / `max_thinking_tokens` 在不同启动参数下的结果 |
| test_struct_lora.py | 76 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `mask_bytes` | `srt/constrained/xgrammar_backend.py:66` _allocate_token_bitmask、`srt/constrained/outlines_backend.py:58` allocate_vocab_mask |
| `fill_time` | `srt/constrained/base_grammar_backend.py:88` fill_vocab_mask_batched，默认逐行 |
| `set_token_filter` | `srt/constrained/torch_ops/token_filter_torch_ops.py:25` |
| `traverse_tree` | `srt/speculative/spec_utils.py:628` |
| `simulate_compile` / `tp_sync` / `timeout_seconds` | `srt/constrained/base_grammar_backend.py:205` 线程池、`:304` get_cached_or_future_value；`srt/constrained/grammar_manager.py:148` process_req_with_grammar、`:211` get_ready_grammar_requests |
| `schema_key` | `utils.py:89` convert_json_schema_to_str |
| `step_time` | `srt/managers/tp_worker.py:728` 推迟采样；`srt/managers/scheduler.py:2139` is_disable_overlap_for_batch |
| `TokenSequenceMatcher` / `Reasoner` | `srt/utils/token_sequence_matcher.py:18`；`srt/constrained/reasoner_grammar_backend.py:154-300` |
| `pool_bytes` / `lora_shapes` | `srt/lora/utils.py:172`、`:330`、`:374`；`srt/lora/mem_pool.py:424-492`、`:598` init_buffers |
| `Pool` / `LRU` / `FIFO` | `srt/lora/mem_pool.py:780` prepare_lora_batch；`srt/lora/eviction_policy.py` |
| `validate_lora_batch` / `schedule_pass` | `srt/lora/lora_manager.py:427`；`srt/managers/scheduler.py:4274`、`:4481` can_schedule_lora_req |
| `chunk_size_for` / `permutation` / `segments` | `srt/lora/backend/chunked_backend.py:197`、`:469`、`:510` |
| `triton_blocks` | `srt/lora/backend/triton_backend.py:469` _build_sgemm_routing；`srt/kernels/lora/triton_ops/sgemm_lora_a.py` |
| `Drainer` / `run_drain` | `srt/lora/lora_drainer.py`；`srt/managers/scheduler.py:2404` 阈值 > 0 才创建 |
| `scenario_overlap`（lora_sim） | `srt/lora/lora_overlap_loader.py:33` try_overlap_load_lora |
| `resolve` | `srt/arg_groups/serving_hook.py:239` handle_grammar_backend；`srt/arg_groups/lora_hook.py:20`、`:33`、`:240`；`srt/arg_groups/speculative_hook.py:452` _handle_uno；`srt/constrained/base_grammar_backend.py:436`；`srt/lora/lora_manager.py:378` |
| `request_fate` | `srt/managers/tokenizer_manager.py:905`；`srt/constrained/grammar_manager.py:148` |

## 运行

```bash
cd zero-to-sglang/16-structured-lora
python3 struct_sim.py                       # 各模型 mask 字节、填 mask vs decode
python3 struct_sim.py --model llama3.1-8b   # 换模型
python3 struct_sim.py --scenario fsm        # 玩具语法、rollback、草稿树
python3 struct_sim.py --scenario queue      # 编译排队、重复编译、key 顺序、超时
python3 struct_sim.py --scenario overlap    # 同步 vs 推迟采样每步 ms
python3 struct_sim.py --scenario think      # 思考预算和强制收尾
python3 lora_sim.py                         # LoRA 池吃掉多少 KV
python3 lora_sim.py --scenario admit        # 调度跳过、pin、LRU vs FIFO
python3 lora_sim.py --scenario segments     # csgmv 分段 vs triton 发块
python3 lora_sim.py --scenario drain        # 冷 adapter 饿死与 drainer
python3 lora_sim.py --scenario overlap      # 同步加载 vs 异步加载
python3 resolve_sim.py                      # 20 条命令行的解析结果
python3 resolve_sim.py --scenario request   # 请求级参数的结果
python3 -m unittest test_struct_lora -v     # 约 1 秒
```

## 期望输出

```text
$ python3 struct_sim.py --scenario overlap（节选）
qwen3-8b / h100-sxm，ctx 2000，每步 ms（括号是相对无语法的增幅）：
   bs      无语法         JSON 同步      JSON overlap          CFG 同步       CFG overlap
    8     7.96     8.78 (+ 10%)     7.96 (+  0%)    10.06 (+ 26%)     7.96 (+  0%)
   64    15.00    18.06 (+ 20%)    15.00 (+  0%)    28.30 (+ 89%)    15.00 (+  0%)
  128    23.05    28.67 (+ 24%)    23.05 (+  0%)    49.15 (+113%)    26.30 (+ 14%)
  256    39.15    49.89 (+ 27%)    39.15 (+  0%)    90.85 (+132%)    51.90 (+ 33%)

$ python3 struct_sim.py --scenario queue（节选）
负载                                    编译次数    p50 等待      最慢等待      假如去重
64 个同 schema 同时到达                       64     1.50s     2.40s     0.30s
64 个同 schema 1 s 内均匀到达                  20     0.00s     0.65s     0.30s
16 个不同 schema 各 4 个                     64     1.50s     2.40s     0.60s
64 个同 schema，大 schema 编 2 s             64    10.00s    16.00s     2.00s

$ python3 struct_sim.py --scenario think（节选）
  步  阶段  想要            实际                思考计数    已匹配     正文计数
  2  思考  <tool_call>   想                    3      0       -1  <- 被改写
  3  思考  </            </                   3      1       -1
  4  思考  再             再                    5      0       -1
  6  思考  算             </                   6      1       -1  <- 被改写
  7  思考  想             think                6      2       -1  <- 被改写
  8  思考  算             >                    6      0        0  <- 被改写

$ python3 lora_sim.py（节选）
模型             TP  rank  mlpb        模块    池 GiB    单个 MiB   少 KV token   占 KV 池
llama3.1-8b     1    16     8       all     0.62      80.0        5,120     1.2%
llama3.1-8b     1    64     8       all     2.50     320.0       20,480     4.9%
llama3.1-8b     1    64    32       all    10.00     320.0       81,920    19.7%
llama3.1-8b     1    64     8      attn     0.81     104.0        6,656     1.6%
llama3.1-70b    4   256     8       all    25.47    3260.0      333,824    79.9%

$ python3 lora_sim.py --scenario segments（节选）
batch                               token chunk  csgmv 段  triton 发射  triton 干活
decode 64 请求 / 4 adapter               64    32        4         32          4
prefill 8 个各 512 / 3 adapter         4096   128       32        256        256
prefill 1×8000 + 15 个短               8544   128       68       8000        541

$ python3 lora_sim.py --scenario drain
--lora-drain-wait-threshold         D 开始时刻      D 等了    热 adapter 等待 p50     p99
0.0                                     没进    > 58 s                7 ms   0.01s
1.0                                 8.98 s    6.98 s                7 ms   4.69s
5.0                                13.42 s   11.42 s                7 ms   7.21s

$ python3 resolve_sim.py（节选）
lora-paths 两个                 grammar: None -> xgrammar  enable_lora: None -> True  overlap: None -> False
                                WARN --enable-lora is set to True because --lora-paths is provided.
overlap 无 max_loaded          启动失败：Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: 16
pinned 8 个, mlpb 8            启动失败：Failed to load LoRA adapter a7 as a pinned adapter. ...
UNO + strict                  启动失败：UNO does not support --enable-strict-thinking because it requires grammar decoding.
```
