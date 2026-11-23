# SGL-16 结构化输出与 LoRA

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/16-structured-lora.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-16｜原理｜结构化输出与多 LoRA：约束解码和 adapter 批处理的代价。

验证基线：sglang 主干 `8fabf5e`。模拟器和 `check_adapter.py` 只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体；`train_lora.py` 要装 `requirements-lora.txt`。

| 文件 | 内容 |
|---|---|
| struct_sim.py | 各模型每行 token mask 的字节数（int32 位图 vs bool）、bs=256 的 H2D 量，逐行填 mask 和一步 decode 谁长；玩具正则语法的逐 token mask、跨语法单元的 token、rollback、投机草稿树 traverse_tree；编译线程池排队、同 schema 并发不命中导致的重复编译、json.dumps 不排序导致的缓存不命中、TP 间 ready 交集 / failed 并集、超时时长随 batch 变长；同步与推迟采样两种时间线下每步 ms；思考预算、多 token think_end 的 KMP 匹配、思考段屏蔽工具调用 token、跨 think_end 回滚 |
| lora_sim.py | LoRA 池按 rank × mlpb × 模块一次分配，折成少了多少 KV token；一轮调度里谁进谁被跳过、pin 占槽、LRU 和 FIFO 在 Zipf 流量下的换入次数；csgmv 的排列与分段 vs triton 按请求发块；冷 adapter 在热流量下饿死，开 drainer 前后的等待；同步加载和 side stream 异步加载的代价 |
| resolve_sim.py | grammar backend 默认值、strict thinking 和 grammar backend / UNO 的冲突、`check_lora_server_args` 和 `check_lora_speculative_compatibility` 的移植、pinned adapter 上限；请求级 `json_schema` / `max_thinking_tokens` 在不同启动参数下的结果 |
| train_lora.py | 用 PEFT 在 `data/city_country.jsonl` 上训一个 LoRA adapter（城市 → 一行 JSON），只对 completion 算 loss，打印留出样本训练前后的输出；默认用随机权重的小 Llama 只验证流程，`--base Qwen/Qwen3-0.6B` 训真模型 |
| check_adapter.py | 上线前检查：只读 `adapter_config.json` 和 safetensors 头部，对照 vLLM 和 SGLang 加载时的校验（rank 上限、DoRA、bias、modules_to_save、加词表、目标模块），给出两边的启动参数；只用标准库 |
| test_lora_adapter.py | 13 个用例：12 个测 `check_adapter`，1 个在装了 torch / transformers / peft 时用小模型训一遍再检查 |
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

## 自己训一个 LoRA adapter

`serve_so_lora.sh` 和下面的模拟器都假设手里已经有 adapter。这一节从零训一个，再检查它能不能被两边加载。训练环境和推理环境分开装：

```bash
pip install -r ../requirements-lora.txt                     # torch、transformers、peft，CPU 就行
python3 train_lora.py                                       # 随机权重的小 Llama，几秒，只验证流程
python3 train_lora.py --base Qwen/Qwen3-0.6B --steps 80 --lr 1e-3 --out loras/qwen-city-r8
python3 train_lora.py --base Qwen/Qwen3-0.6B --rank 64 --out loras/qwen-city-r64   # 换 rank，给多 LoRA 实验凑不同档位
python3 check_adapter.py loras/*                            # 上线前检查，只要标准库
python3 test_lora_adapter.py
```

任务是让模型把「城市：巴黎」答成一行 JSON。底座不会自发输出这个格式，训完的 adapter 会，所以加不加 adapter 一眼就能看出差别。下面是 Qwen3-0.6B 在 Apple M4 Pro 的 CPU 上实跑的输出（float32，batch 8，80 步）：

```
底座 Qwen/Qwen3-0.6B，LoRA r=8 alpha=16 模块 ['q_proj', 'k_proj', 'v_proj', 'o_proj']
可训练参数 2,293,760 / 总参数 598,343,680（0.383%）
  step    0  loss 2.3322
  step   13  loss 0.0050
  step   26  loss 0.0008
  step   39  loss 0.0013
  step   52  loss 0.0008
  step   65  loss 0.0029
  step   78  loss 0.0014
  step   79  loss 0.0005
训练 80 步，用时 73.3 s，adapter 存到 loras/qwen-city-r8
留出的 4 条：loss 2.338 → 0.057，完全答对 0% → 75%
探针 '城市：巴西利亚'
  训练前 '时间：2023年'
  训练后 '{"city": "巴西利亚", "country": "巴西"}'
```

留出的 4 个城市里答对 3 个。可训练参数只占 0.38%，adapter 文件 9.2 MB。`check_adapter.py` 对这个目录的检查结果：

```
== qwen-city-r8
  底座      Qwen/Qwen3-0.6B
  r / alpha 8 / 16
  声明模块  ['v_proj', 'k_proj', 'o_proj', 'q_proj']
  实际模块  ['k_proj', 'o_proj', 'q_proj', 'v_proj']（224 个张量）
  vLLM    可以加载
  SGLang  可以加载
  vLLM    --enable-lora --max-lora-rank 8 --lora-modules qwen-city-r8=/绝对路径/loras/qwen-city-r8
  SGLang  --enable-lora --max-lora-rank 8 --lora-target-modules k_proj o_proj q_proj v_proj --lora-paths qwen-city-r8=/绝对路径/loras/qwen-city-r8
```

几个容易踩的点，检查脚本都会报出来：

- vLLM 的 `--max-lora-rank` 只能取 1、8、16、32、64、128、256、320、512，默认 16（`vllm/config/lora.py:27`）。rank 64 的 adapter 不改这个参数会加载失败。
- SGLang 不传 `--max-lora-rank` 时，取启动时所有 adapter 里最大的 r（`lora/lora_manager.py:896`）；之后热加载更大 rank 的 adapter 会报 incompatible。
- 两边都不支持 DoRA。vLLM 的 `modules_to_save` 只接受 `classifier` / `score`，所以把 `lm_head` 或 `embed_tokens` 也一起训的 adapter 上不了 vLLM。
- 训练时往词表里加了 token（目录里有 `added_tokens.json`、id 超出底座词表），SGLang 拒绝加载。
- SGLang 的 `--lora-target-modules` 会把 q/k/v_proj 归一成 `qkv_proj`、gate/up_proj 归一成 `gate_up_proj`；只写 `qkv_proj` 而 adapter 还训了 `o_proj`，同样加载失败。

把训好的几个目录放进 `$LORA_DIR`，就能接着跑 `serve_so_lora.sh lora`、`lora_bench` 和 `sgl_lora`。GPU 上的加载和压测本节没有实测。

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
