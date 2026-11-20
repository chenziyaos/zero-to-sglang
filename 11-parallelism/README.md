# SGL-11 并行

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/11-parallelism.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-11｜原理｜并行：TP、DP Attention、EP 怎么组合。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| sim_parallel.py | 并行参数解析（attn_tp、moe_tp、rank 三维展开、各项校验），按结构累加的每卡显存与 KV 容量，FP8 块整除检查，每层通信字节（TP all-reduce / DP gather / DeepEP），DP gather 的 MAX_LEN / SUM_LEN 选择，DeepEP low_latency buffer 大小 |
| eplb_sim.py | EPLB 算法的纯 Python 移植（balanced_packing、replicate_experts、分层 / 全局两种策略），在合成负载上比较均衡度、负载漂移、跨节点数、static / dynamic 分派 |
| tbo_sim.py | TBO 的切分逻辑移植（decode 对半、extend 平衡切和两段切），decode / prefill 的屋顶线耗时模型，decode 策略的 6 段交错时间线 |
| test_parallel.py | 28 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/srt/`。

| 本目录 | sglang |
|---|---|
| `Par` 的字段 | `arg_groups/fields/parallel.py`：`:45` tp_size、`:59` pp_size、`:73` dp_size、`:81` attn_dp_size、`:104` attn_cp_size、`:112` moe_dp_size、`:194` enable_dp_attention（已废弃，等价 attn_dp=N）、`:199` enable_dp_lm_head、`:281` ep_size、`:289` moe_dense_tp_size |
| `resolve` 里的 attn_tp / moe_tp | `runtime_context.py:73` attn_tp = tp // attn_dp // attn_cp；`:154` moe_tp = tp // ep // moe_dp |
| `rank_layout` | `runtime_context.py:79`，按 (dp, cp, tp) 展开，tp 变化最快；`:127-133` moe 的 rank 划分 |
| `resolve` 的校验 | `arg_groups/parallel_hook.py:162` tp 能被 attn_dp×attn_cp 整除；`:287` dp 和 attn_dp 不能同时大于 1；`arg_groups/validation_hook.py:207` moe_dense_tp 只能是 1 或 tp |
| a2a 后端强制 ep = tp | `arg_groups/overrides.py:1578`、`:1612` |
| DP attention 的显存调整 | `arg_groups/parallel_hook.py:318` schedule_conservativeness × 0.3、`:327` chunked_prefill_size //= attn_dp |
| `kv_heads_per_gpu` | `configs/model_config.py:1718` max(1, kv_heads // (tp // dcp)) |
| `components` 的切法 | `models/deepseek_v2.py`：注意力头按 attn_tp 切，fused_qkv_a 是 ReplicatedLinear；`:874-897` a2a 下共享专家不切；`:2676` moe_dense_tp=1 时稠密 MLP 不切；`:2954` embedding 走 get_embedding_tp_kwargs（`layers/vocab_parallel_embedding.py:166`）；`:3302` lm_head |
| 冗余专家 | `models/deepseek_v2.py:751`、`:988` num_experts += ep_num_redundant_experts；`arg_groups/fields/exec_.py:917` |
| `shard` 场景 | `layers/quantization/fp8.py:593-609` 块量化 128 整除检查 |
| `padding_mode` | `layers/dp_attention.py:152` DpPaddingMode，`:165` 选择规则 |
| `dp_gather_bytes` | `layers/dp_attention.py:1080` dp_scatter、`:1111` can_use_dp_reduce_scatter |
| `deepep_wire_bytes` / `ll_buffer_bytes` | `layers/moe/token_dispatcher/deepep.py:204` buffer 申请、`:400-409` num_max_dispatch_tokens_per_rank 默认 128、上限 1024、`:746` LowLatency；DeepEP v1.2.1 的 LowLatencyLayout |
| DeepEP normal 关 CUDA Graph | `arg_groups/moe_hook.py:437` |
| `balanced_packing` / `replicate_experts` / `rebalance_hierarchical` | `eplb/eplb_algorithms/deepseek.py`（与 deepseek-ai/EPLB 的 eplb.py 相同） |
| `rebalance(..., 'auto')` | `eplb/eplb_algorithms/__init__.py` compute_algorithm：组数能被节点数整除走分层，否则全局 |
| `trivial_map` | `eplb/expert_location.py:108` init_trivial、`:495` 冗余槽位按 i % num_log 补齐 |
| `scenario_dispatch` | `eplb/expert_location.py:615` compute_logical_to_rank_dispatch_physical_map（seed 42，同卡 > 同节点 > _fair_choices 均分）；`eplb/expert_location_dispatch.py:132` dynamic |
| static / dynamic 默认值 | `arg_groups/parallel_hook.py:750` handle_eplb_and_dispatch：有 a2a 用 static，没有 a2a 只能 dynamic；开 EPLB 自动把 recorder 设成 stat |
| `balancedness` | `eplb/expert_distribution.py:1159` (avg + 1e-5) / (max + 1e-5) |
| 周期重排 | `eplb/eplb_manager.py:225` _check_rebalance_needed；`--eplb-rebalance-num-iterations` 默认 1000 |
| `split_decode` | `batch_overlap/two_batch_overlap.py:83` compute_split_seq_index |
| `two_chunk_enabled` | `batch_overlap/two_batch_overlap.py:102`；阈值 `arg_groups/fields/exec_.py:1065` 默认 0.48，设 0 关掉两段切 |
| `split_balanced` / `split_cum_half` | `batch_overlap/two_batch_overlap.py:157`、`:142` |
| `split_extend` 的 token 下标 | `batch_overlap/two_batch_overlap.py:291` compute_split_token_index |
| `scenario_stages` / `scenario_prefill` | `batch_overlap/operations_strategy.py:136` decode（tbo_delta_stages=2，5 个 yield）、`:105` prefill（deep_gemm_num_sms = 总 SM − DeepEP num_sms） |

## 运行

```bash
cd zero-to-sglang/11-parallelism
python3 sim_parallel.py                       # 各种并行组合的每卡显存、KV 复制倍数、KV 容量
python3 sim_parallel.py --scenario layout     # rank 编号和参数校验
python3 sim_parallel.py --scenario shard      # FP8 块大小 128 的整除检查，稠密层最优 TP
python3 sim_parallel.py --scenario comm       # 每层通信字节和耗时，对账 DeepEP README
python3 sim_parallel.py --scenario padding    # DP gather 的 MAX_LEN / SUM_LEN
python3 sim_parallel.py --scenario deepep     # DeepEP low_latency buffer
python3 eplb_sim.py                           # EP32 / EP72 上做不做 EPLB 的均衡度
python3 eplb_sim.py --scenario example        # 复现 deepseek-ai/EPLB README 的例子
python3 eplb_sim.py --scenario shift          # 负载漂移以后旧方案掉多少
python3 eplb_sim.py --scenario locality       # 分层 vs 全局：一个 token 碰几个节点
python3 eplb_sim.py --scenario dispatch       # 多副本时 static / dynamic 分派
python3 tbo_sim.py                            # TBO 切分
python3 tbo_sim.py --scenario decode          # decode 开关 TBO 的每层耗时
python3 tbo_sim.py --scenario prefill         # prefill 开 TBO 的上限和吞吐对账
python3 tbo_sim.py --scenario stages          # 两个 micro-batch 的交错时间线
python3 -m unittest test_parallel -v
```

全部场景都在几秒内跑完。

## 期望输出

```text
$ python3 sim_parallel.py（节选）
DeepSeek-V3  fp8 权重  H200
  配置                             attn  dense     共享     路由  emb+lm    合计GB   KV/tok    复制     每卡 tok     整组 tok
  TP8                             2.2   0.15   0.32   81.7    0.46    84.9    68.6K    8x       497k       497k
  TP8 + attn_dp8                 11.4   0.15   0.32   81.7    3.71    97.3    68.6K    1x       321k      2564k
  DP8 + DeepEP EP8               11.4   1.19   2.55   81.7    3.71   100.6    68.6K    1x       274k      2191k

$ python3 sim_parallel.py --scenario shard（节选）
  DeepSeek-V3 稠密 MLP        9216      4608      2304      1152       576 ✗
  Qwen3-235B 专家              768       384       192 ✗      96 ✗      48 ✗

$ python3 sim_parallel.py --scenario comm（节选）
  TP8                       attn 后 all-reduce + MoE 后 all-reduce       51.38       343
  DP-attn8 + MoE TP8 均匀     MAX_LEN：gather + scatter                 25.69       171
  DP8 + DeepEP EP8          dispatch(FP8) + combine(BF16)            22.25       193
     EP   实测 dispatch       按带宽反推    实测 combine       按带宽反推
      8          77μs        77μs         114μs       116μs
     64         173μs       176μs         314μs       319μs

$ python3 eplb_sim.py --scenario example（节选）
  第 0 层（hierarchical）phy2log = [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1]  与 README 一致
  第 1 层（hierarchical）phy2log = [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]  与 README 一致

$ python3 eplb_sim.py（节选）
  不做 EPLB，无冗余                      0.396   0.398         2.51x
  EPLB 32 冗余（hierarchical）         0.875   0.850         1.18x
  EPLB 32 冗余（global）               0.984   0.934         1.07x     （EP72）

$ python3 eplb_sim.py --scenario locality（节选）
  按编号摆（trivial）                   3.10           12.5%   0.362
  分层策略（hierarchical）              2.90            0.0%   0.772
  全局策略（global）                    3.63            2.3%   0.995

$ python3 tbo_sim.py --scenario decode（节选）
     每卡 bs      计算      通信    关 TBO      两半计算      两半通信    开 TBO      变化
         8     195      61      256       383        91      383    -33%
        32     215     154      369       403       184      403     -8%
       256     530    1020     1551       630      1050     1050    +48%
```

## 校准

1. DeepEP 带宽：`sim_parallel.py --scenario comm` 用 README 的 low_latency 带宽和实测时延互相反推，EP8 到 EP256 误差都在 5% 以内，说明消息字节（dispatch 7392 B、combine 14336 B）口径对。自己的集群用 DeepEP 仓库的 `tests/test_low_latency.py` 跑一遍，把带宽填回 `LL_BW`。
2. EPLB 移植：`eplb_sim.py --scenario example` 两层的 phy2log 与 deepseek-ai/EPLB README 逐位一致。真实负载用 `--expert-distribution-recorder-mode stat` 启动，调 `/start_expert_distribution_record`、`/dump_expert_distribution_record` 导出，替换 `make_load`。
3. TBO：decode 方向和 LMSYS 博客一致（小 batch 亏、大 batch 赚，拐点在 32 到 64 之间），幅度偏乐观；prefill 关 TBO 时折算每节点约 5.2 万 tok/s，和博客 4 节点 EP32 实测的 5.0 万～5.8 万同量级。上线前用 `--enable-two-batch-overlap` 开关各跑一次 bench_serving 比较。
4. FP8 整除：Qwen3-235B-A22B-FP8 在 TP8 上专家中间维切到 192，报 block_n = 128 不能整除，和 QwenLM/Qwen3 discussion #1674 的现象一致；改 `--ep 8` 或 TP4 即可。
5. 显存：启动日志里的 `Load weight end ... mem usage=` 对 `合计GB`，`max_total_num_tokens=` 对 `每卡 tok`。`mem_fraction_static` 和 CUDA Graph 预留不同，每卡 token 会差 10% 到 20%，权重应该差在 1 GB 以内。
