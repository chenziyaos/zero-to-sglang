# SGL-10 注意力后端

对应飞书文档：SGL-10｜原理｜注意力后端：FlashInfer、FA3、Triton、TRTLLM 怎么选、怎么接进 CUDA Graph。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| chooser.py | 启动参数解析里和注意力后端有关的几步按源码顺序重写：模型 override、确定性推理、自动选择、page_size 约束、各后端校验和回退；附文档支持矩阵筛选 |
| attn_cost.py | 注意力这层的账：decode 算术强度（MHA / GQA / MLA）、FlashInfer Tensor Core 开关、split-K 的 SM 占用、prefill 注意力 FLOPs 占比、decode 一步里 KV 读取占比 |
| paged_attn.py | 后端手里的数据结构和数学：req_to_token 到 CSR / 页表、ragged + paged 两段按 LSE 合并、split-K 归并、page_size 对前缀命中的影响、in_graph 契约检查、拆分后端的分派 |
| test_attention.py | 26 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/srt/`。

| 本目录 | sglang |
|---|---|
| `resolve` 第 1 步 | `arg_groups/pipeline.py:225` handle_model_specific_adjustments；`arg_groups/model_overrides/deepseek_v2.py:102`（DSA）、`:178`（DeepSeek V3 在 SM100 填 trtllm_mla） |
| `resolve` 第 2 步 | `arg_groups/overrides.py:1093` _deterministic_attention_backend；`arg_groups/choices.py:113` 确定性可用后端 |
| `default_backend` | `arg_groups/model_override_base.py:277` get_default_attn_backend |
| `no_spec_or_topk_one` | `utils/common.py:3960` is_no_spec_infer_or_topk_one |
| `Platform` 的几个判定 | `utils/common.py:287-330` |
| `resolve` 第 3 步 | `arg_groups/overrides.py:1127` _attention_backend_default，`:1137` 打印默认后端日志 |
| `resolve` 第 4–9 步 | `arg_groups/overrides.py:1144-1301`（attention_hook 里按 `:120 → :184 → :190 → :196 → :198 → :239 → :241` 的顺序挂载） |
| `resolve` 第 10 步 | `arg_groups/overrides.py:1341` _page_size_default |
| `resolve` 第 11 步 | `layers/attention/attention_registry.py:240` FA3 平台断言，trtllm_mha / hpc_ops 只认非 MLA |
| 投机参数的自动值 | `arg_groups/speculative_hook.py:1206` _auto_choose_speculative_params，晚于注意力后端选择 |
| `tc_tile_fill` 和默认开关 | `layers/attention/flashinfer_backend.py:2670` should_use_tensor_core、`:394` decode_use_tensor_cores |
| `decode_attn_us` 的 split | `layers/attention/triton_backend.py:327` max_kv_splits、`:431` get_num_kv_splits；`arg_groups/fields/exec_.py:416` 参数定义 |
| `build_csr` | `layers/attention/flashinfer_backend.py:486` kv_last_page_len 全 1；`:1847`、`:2403` page_size 传 1 |
| `extend_two_pass` | `layers/attention/flashinfer_backend.py:1357`、`:1503` ragged 与 paged 两段 merge_state |
| `page_table` | `layers/attention/flashattention_backend.py:819-828` |
| `prefix_hit` | `mem_cache/radix_cache.py:381` 匹配长度截到 page 整数倍 |
| `check_in_graph` | `layers/attention/base_attn_backend.py:117` out_graph、`:139` in_graph 的 lint 契约 |
| `select_backend` | `layers/attention/hybrid_attn_backend.py:64` _select_backend；`arg_groups/fields/spec.py:129` speculative_attention_mode 默认 prefill |

## 运行

```bash
cd zero-to-sglang/10-attention-backend
python3 chooser.py                        # 常见 GPU × 模型的默认后端
python3 chooser.py --scenario spec        # 投机 topk、page_size 怎么改变默认值
python3 chooser.py --scenario rules       # 手动指定后端时被改写或拒绝
python3 chooser.py --scenario matrix      # 按功能需求筛后端
python3 attn_cost.py                      # decode 注意力的算术强度
python3 attn_cost.py --scenario tc        # Tensor Core 开关和 MMA 填充率
python3 attn_cost.py --scenario split     # split-K 与 SM 占用
python3 attn_cost.py --scenario prefill   # prefill 注意力 FLOPs 占比
python3 attn_cost.py --scenario share     # decode 一步里 KV 读取占比
python3 paged_attn.py                     # CSR 和页表
python3 paged_attn.py --scenario merge    # ragged + paged 按 LSE 合并
python3 paged_attn.py --scenario split    # split-K 归并
python3 paged_attn.py --scenario page     # page_size 与前缀命中
python3 paged_attn.py --scenario contract # out_graph / in_graph 契约
python3 paged_attn.py --scenario hybrid   # 拆分后端的分派
python3 -m unittest test_attention -v
```

全部场景都在几秒内跑完。

## 期望输出

```text
$ python3 chooser.py（节选）
                    Qwen3-8B                        gpt-oss-120b                    DeepSeek-V3
  A100              flashinfer             page=1   triton                 page=1   triton                 page=1
  H100/H20          fa3                    page=1   fa3                    page=1   fa3                    page=1
  H100 + CUDA 12.1  flashinfer             page=1   triton                 page=1   triton                 page=1
  B200              trtllm_mha             page=64  trtllm_mha             page=64  trtllm_mla             page=64

$ python3 chooser.py --scenario spec（节选）
    EAGLE topk 未设             fa3                    page=1      （H100）
    EAGLE topk=1, page=16     flashinfer             page=16
    EAGLE topk 未设             flashinfer             page=1      （B200）

$ python3 attn_cost.py（节选）
  Qwen3-8B                   32/8     4      4.0     8.0        8.0
  Qwen3-235B-A22B            64/4    16     16.0    32.0       32.0
  DeepSeek-V3           128/latent     -    241.8   483.6      483.6

$ python3 attn_cost.py --scenario split（节选）
     1   32768       1     8     6%     826.3      1.0x
     1   32768      17   136   100%      53.3     15.5x
    64    4096       8  4096   100%     410.0      1.0x

$ python3 attn_cost.py --scenario share（节选）
     1    4096    4,096     5.65    0.23         4%
    64    4096  262,144     5.65   14.42        72%

$ python3 paged_attn.py --scenario merge
  paged 段 + ragged 段 + merge_state  vs  一次算完：最大误差 1.11e-16

$ python3 paged_attn.py --scenario page（节选）
    page     可复用前缀      前缀命中率       尾页浪费槽位     浪费占比
       1      1000      82.9%            0    0.00%
     128       896      74.3%        15120    4.03%
```

## 校准

模拟给的是结构和量级，后端之间的真实快慢只能在自己的卡上量。

1. 看实际选中的后端：启动日志里搜 `Attention backend not specified. Use ... backend by default.`（`arg_groups/overrides.py:1137`）；显式指定时不打这句，以参数为准。先用 `chooser.py` 推一遍，对不上说明版本或平台判定不同。
2. 后端对比：同一个模型、同一份 bench_serving 参数，分别加 `--attention-backend fa3`、`flashinfer`、`triton` 各跑一次（B200 上加 `trtllm_mha`），比较 TTFT、ITL 和 output throughput。Hopper 上 FA3 与 FlashInfer 的差距和 flashinfer 版本强相关，参考 sglang issue #17411。
3. decode 注意力吃了多少带宽：`nsys profile --cuda-graph-trace=node` 抓稳态，按 kernel 名过滤注意力 kernel（FlashInfer 的 BatchDecode / BatchPrefill 系列，FA3 的 flash_fwd 系列，Triton 的 stage1 / stage2），用「本步读的 KV 字节 ÷ kernel 耗时」对比 HBM 峰值，填回 `attn_cost.py` 的 `bw_eff`。
4. split 的效果：Triton 后端改 `--triton-attention-num-kv-splits`（默认 8），在 bs=1、长上下文下比较 ITL。
5. page_size：同一份带共享前缀的负载，分别用 `--page-size 1` 和 `64` 跑，看日志和 `/metrics` 里的 cache hit rate 与吞吐，和 `paged_attn.py --scenario page` 的估算对比。
