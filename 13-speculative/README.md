# SGL-13 投机解码

对应飞书文档：SGL-13｜原理｜投机解码：EAGLE、MTP、NGRAM 怎么选、怎么调。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| spec_sim.py | 接受率 α、草稿长度 k、草稿成本 c 和加速比；从日志的 accept len 反推 α；链式 rejection sampling、block verification、target-only 树验证和两个 threshold 的输出分布对比；EAGLE-2 动态树的建树（parent_list、selected、tree mask、positions、next_token / next_sibling）和贪心验证；batch 与上下文变化时的 Roofline 加速比；NGRAM 后缀匹配草稿 |
| resolve_sim.py | `--speculative-*` 的解析链移植：大小写、NEXTN 别名、block verification、adaptive、EAGLE 家族和 NGRAM 分支的先后顺序、自动选参、各种报错；每个请求每步预留的 KV 槽位；草稿层摊进 KV cell 后池子缩水多少 |
| adaptive_sim.py | adaptive 投机解码的移植：默认四个 batch 槽位、EMA 加上下迟滞的 steps 切换、按 CUDA graph bs 补齐后路由到槽位；三段接受率变化下 adaptive 和固定 steps 的效率对比 |
| test_spec.py | 40 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `expected_tokens` / `alpha_from_accept_len` / `sglang_accept_rate` | Leviathan 2023 的期望公式；`srt/managers/scheduler_components/metrics_reporter.py:973` accept len / accept rate 的口径；`srt/managers/tokenizer_manager.py:3164` 每请求的 spec_accept_rate / spec_accept_length |
| `chain_rejection_round` / `block_verify_round` | `kernels/ops/speculative/reject_sampling.py` 链式拒绝采样和 block verification（arXiv 2403.10444） |
| `tree_target_only_round` | `kernels/jit/include/sgl_kernel/speculative/sampling.cuh` TreeSpeculativeSamplingTargetOnly，含 threshold_single / threshold_acc |
| `draft_tree` | `srt/speculative/eagle_worker_v2.py` draft_forward 和 `srt/speculative/spec_utils.py:417` select_top_k_tokens |
| `organize_draft_results` | `srt/speculative/eagle_utils.py:103` |
| `build_tree` / `verify_greedy` | `kernels/ops/speculative/spec_tree.py:53` 建树、`:211` verify_tree_greedy |
| `ngram_draft` | `srt/speculative/ngram_worker.py`、`kernels/ops/speculative/ngram_corpus.py`（真实实现是 C++ trie，这里只保留后缀匹配的思路） |
| `resolve` 主干 | `srt/arg_groups/speculative_hook.py:124` handle_speculative_decoding、`:84` 别名、`:1113` _handle_eagle_family、`:1326` _handle_ngram、`:1433` _init_adaptive_speculative_params、`:1471` _auto_choose_speculative_params |
| `adaptive_unsupported_reason` / `AdaptiveStepSlot` / `AdaptiveSpeculativeParams` | `srt/speculative/adaptive_spec_params.py` |
| `alloc_len_per_decode` | `srt/mem_cache/allocation_sizing.py:28` get_alloc_len_per_decode，调度器按 2 倍预留 |
| `scenario_memory` 的 cell 放大 | `srt/model_executor/pool_configurator.py:259` |

## 运行

```bash
cd zero-to-sglang/13-speculative
python3 spec_sim.py                         # α、k、c 和加速比；从 accept len 反推 α
python3 spec_sim.py --scenario sampling     # 六种验证规则的输出分布和接受长度（约 5 秒）
python3 spec_sim.py --scenario tree         # 动态树的结构，六组 (steps,topk,ndt) 的 accept len
python3 spec_sim.py --scenario batch        # batch 和上下文变化时的 Roofline 加速比
python3 spec_sim.py --scenario ngram        # 抄写型输出和全新输出
python3 resolve_sim.py                      # 16 条命令行的解析结果和告警
python3 resolve_sim.py --scenario memory    # KV 预留和池子缩水
python3 adaptive_sim.py                     # 三段接受率下 adaptive 的 steps 轨迹
python3 adaptive_sim.py --scenario route    # batch 怎么路由到四个槽位
python3 -m unittest test_spec -v            # 约 10 秒
```

## 期望输出

```text
$ python3 spec_sim.py（节选）
加速比 E/(k·c+1)，c=0.05
  α \ k       1      2      3      4      5      6      8   最优 k
  0.6      1.52   1.78   1.89   1.92   1.91   1.87   1.77       4
  0.8      1.71   2.22   2.57   2.80   2.95   3.04   3.09       8
  accept len  draft_tokens   反推 α   SGLang accept rate
         2.4             4    0.665        0.467
         3.6             8    0.750        0.371

$ python3 spec_sim.py --scenario sampling（节选）
规则                                        一元 TV   二元 TV   每轮产出
  链式 rejection sampling, k=3                  0.0016    0.0041       2.66
  block verification, k=3                     0.0016    0.0061       2.87
  target-only 树, 深 3 宽 3                      0.0007    0.0050       2.46
  target-only 树, 宽 3, threshold_acc=0.5       0.3011    0.4423       4.00

$ python3 spec_sim.py --scenario tree（节选）
  (steps,topk,ndt)  每轮验证 token   accept len   accept rate   和纯贪心一致
  (3,1,4)              4         2.26         0.419           是
  (5,4,8)              8         2.71         0.244           是
  (3,4,16)            16         3.01         0.134           是

$ python3 spec_sim.py --scenario batch（节选，上下文 1024）
  (steps,topk,ndt) L          1      4      8     16     32     64    128    256
  (3, 1, 4)        2.8     2.40   2.40   2.41   2.42   2.43   2.45   2.47   2.28
  (5, 4, 8)        3.6     2.81   2.82   2.83   2.85   2.87   2.62   1.87   1.43
  (3, 4, 16)       3.4     2.91   2.92   2.92   2.93   2.20   1.40   0.98   0.75

$ python3 spec_sim.py --scenario ngram
  抄写型（改代码）   4.48     全新文本   1.00

$ python3 resolve_sim.py（节选）
topk=1 却写 draft_tokens=16
  -> EAGLE steps=3 topk=1 draft_tokens=4
     WARN speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1
只写 topk 不写 steps
  -> 启动失败 AssertionError
adaptive + topk 4（被静默关掉）
  -> EAGLE steps=5 topk=4 draft_tokens=8 adaptive=False

$ python3 resolve_sim.py --scenario memory（节选）
  (5, 4, 8)                20      128      512
  不开投机 415,401 token；EAGLE 1 层草稿摊进 cell 后 402,813 token（少 3.0%）
    (3, 4, 16)   page=64   49,152 token，占池子 12.20%

$ python3 adaptive_sim.py（节选）
  固定 steps=1 / 3 / 5 / 7   1.440 / 1.762 / 1.790 / 1.768
  adaptive                   1.900
```

## 校准

1. 接受长度：服务日志里定期打印 `accept len: x.xx, accept rate: 0.xx`，请求返回的 `meta_info` 里有 `spec_accept_length`。用 `spec_sim.alpha_from_accept_len(L, ndt-1)` 反推 α（topk=1 链式），再到理论表里看当前 steps 是否接近最优。
2. 解析链：起服务时日志里的 `Max running requests is reset to 48`、`speculative_num_draft_tokens is adjusted`、`speculative_adaptive disabled` 应和 `resolve_sim.py` 一致。换版本先 diff `speculative_hook.py`。
3. 加速比：Roofline 是上限。实测用 `python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --max-concurrency N` 对开关投机的两个服务分别压 1 / 4 / 16 / 64 并发，看加速比从哪里掉到 1 以下，和 `--scenario batch` 的拐点对照。
4. KV 池：启动日志 `max_total_num_tokens=` 开投机后应比不开少约 1/(L+1)（L 为目标模型层数，1 层草稿）。STANDALONE 草稿模型层数多时少得多。
