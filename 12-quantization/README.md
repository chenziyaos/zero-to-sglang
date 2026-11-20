# SGL-12 量化

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/12-quantization.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-12｜原理｜量化：FP8、INT4 与 KV 量化怎么选。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| quant_sim.py | E4M3 / E5M2 / E2M1 格点的纯 Python 实现；十种权重格式（FP8 逐张量 / 逐通道 / 128×128 块 / UE8M0 块、INT8 逐张量 / 逐通道、INT4 g128 对称 / 零点、MXFP4、NVFP4）的误差；激活量化和 SmoothQuant；AWQ 的通道缩放搜索；GPTQ（含 desc_act）的纯 Python 实现；FP8 / FP4 KV 对 attention 输出的影响 |
| resolve_sim.py | `--quantization` 的解析链移植：注册表、override、兼容表、算力和 dtype 检查；Marlin 支持的类型和形状检查（按 TP 切分后的 MLP）；FP8 线性层在各代 GPU 上走哪条 kernel；`--kv-cache-dtype` 解析和 k_scale / v_scale 兜底 |
| mem_sim.py | 每种格式的真实位宽和 checkpoint 体积（和 Hugging Face 文件对账）；权重格式 × KV 格式的 KV 池容量（复用 02-ledger）；W4A16 / W8A16 / W8A8 的 decode Roofline 加速比随 batch 的衰减 |
| test_quant.py | 30 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/srt/`。

| 本目录 | sglang |
|---|---|
| `registry` | `layers/quantization/__init__.py:62` BASE_QUANTIZATION_METHODS；裸 `gptq` 只在 NPU / CPU / XPU 分支注册，CUDA 上没有；mxfp4 只在 cpu / cuda / gfx95 / xpu 加入 |
| `verify` 主干 | `configs/model_config.py:2018` _verify_quantization：rocm_supported / optimized / compatible 三张表，`:545` REQUANTIZATION_METHODS；`:1745` _parse_quant_hf_config；`:1856` _parse_modelopt_quant_config |
| `find_override` | 依次调用各方法的 override_quantization_method：`layers/quantization/awq/awq.py:333`（`:402` is_awq_marlin_compatible）、`gptq/gptq.py:414`（`:44` check_marlin_format、`:459` is_gptq_marlin_compatible）、`moe_wna16.py:164`、`base_config.py:223` _modelopt_override_quantization_method |
| `MIN_CAP` / `ACT_DTYPES` | 各 Config 的 get_min_capability / get_supported_act_dtypes；检查在 `model_loader/loader.py:259`（算力）、`:266`（dtype）；AWQConfig 只收 float16（`awq/awq.py:105`） |
| 在线量化 | `model_loader/weight_utils.py:269` get_quant_config：fp8 的 get_config_filenames 为空，BF16 模型加 `--quantization fp8` 直接构造 Fp8Config |
| `marlin_types` / `marlin_supported` | `layers/quantization/marlin_utils.py:68` query_marlin_supported_quant_types、`:104` _check_marlin_supported；group 只收 -1 / 32 / 64 / 128 |
| `marlin_shape` | `marlin_utils.py:161` verify_marlin_supports_shape（out % 64、in % 128、in % group）；AWQ 在 `awq/schemes/awq_marlin.py:49` create_weights 时检查，GPTQ 在 `hardware_backend/gpu/quantization/gptq_kernels.py:89` 加载后检查 |
| `fp8_block_ok` | `layers/quantization/fp8_utils.py:2852` validate_fp8_block_shape |
| `fp8_linear_path` | `layers/quantization/fp8.py:508` Fp8LinearMethod 选 Marlin（`fp8_utils.py:2794` can_auto_enable_marlin_fp8，80 ≤ sm < 89）；`fp8.py:1142` process_weights_after_loading 选逐通道 / 逐张量；`fp8_utils.py:397` cutlass_fp8_supported；`:704` dispatch_w8a8_block_fp8_linear；`:1170` _dispatch_auto_backend；`:1207` SM120 默认 cutlass |
| `kv_cache_dtype` | `mem_cache/kv_cache_dtype.py` configure_kv_cache_dtype |
| `kv_scales` | `layers/quantization/kv_cache.py:52` BaseKVCacheMethod.process_weights_after_loading |
| 参数 | `arg_groups/fields/model.py:177` quantization、`:185` quantization_param_path、`:197` kv_cache_dtype、`:233` modelopt_quant；`arg_groups/fields/exec_.py:276` fp8_gemm_runner_backend |
| `per_block_cast_to_fp8` 的口径 | `fp8_utils.py:2394`（128×128，amax 下限 1e-4，除 448）、`:2411` ceil_to_ue8m0 |

## 运行

```bash
cd zero-to-sglang/12-quantization
python3 quant_sim.py                          # 十种权重格式的误差
python3 quant_sim.py --scenario formats       # E4M3 / E5M2 / E2M1 格点
python3 quant_sim.py --scenario activation    # 激活量化、SmoothQuant、FP8 动态量化
python3 quant_sim.py --scenario awq           # AWQ 的 α 搜索
python3 quant_sim.py --scenario gptq          # RTN / GPTQ / GPTQ+desc_act
python3 quant_sim.py --scenario kv            # FP8 / FP4 KV 对 attention 输出的影响
python3 resolve_sim.py                        # 22 个 checkpoint × 参数 × GPU 组合的解析结果
python3 resolve_sim.py --scenario marlin      # Marlin 类型和 TP 切分后的形状检查
python3 resolve_sim.py --scenario fp8path     # FP8 线性层在各代 GPU 上走哪条 kernel
python3 resolve_sim.py --scenario kv          # --kv-cache-dtype 和 scale 兜底
python3 mem_sim.py                            # 位宽和 checkpoint 体积对账
python3 mem_sim.py --scenario kv              # 权重格式 × KV 格式的 KV 池容量
python3 mem_sim.py --scenario decode          # decode 加速比随 batch 的衰减
python3 -m unittest test_quant -v
```

gptq 和 kv 场景是纯 Python 矩阵运算，各要十几秒，其余都在一两秒内。

## 期望输出

```text
$ python3 quant_sim.py（节选）
  格式                          位/参数     相对误差  SQNR dB
  FP8 per-tensor              8.000     2.62%     31.6
  FP8 per-channel             8.008     2.54%     31.9
  INT8 per-tensor             8.000     6.47%     23.8
  INT8 per-channel            8.008     1.52%     36.4
  INT4 g128 对称（GPTQ）          4.125    15.70%     16.1
  INT4 g128 零点（AWQ）           4.156    11.73%     18.6
  NVFP4 g16                   4.500     9.46%     20.5

$ python3 quant_sim.py --scenario gptq
  bit     分组      RTN     GPTQ  GPTQ+desc_act
    4     整行   12.14%    8.99%          7.29%
    3      g32   20.66%   16.25%         13.50%

$ python3 quant_sim.py --scenario kv
  K 离群通道幅度  E4M3 scale=1    E4M3 校准  E5M2 scale=1    FP4 块16
               8        6.83%     5.88%       11.54%    22.60%
             600       88.08%    46.51%       49.15%    46.89%

$ python3 resolve_sim.py（节选）
[H100] AWQ 4bit，不加参数
    -> awq_marlin
[H100] AWQ，--quantization awq --dtype bfloat16
    -> ERROR bfloat16 is not supported for quantization method awq. Supported dtypes: ['float16']
[A100] GPTQ，--quantization gptq
    -> ERROR Unknown quantization method: gptq.
[H100] FP8 块量化，--quantization awq
    -> ERROR Quantization method specified in the model config (fp8) does not match ...

$ python3 resolve_sim.py --scenario marlin（节选）
  Qwen2.5-72B（BF16 原版）
    TP2  Marlin 不通过  FP8 块 不通过  down_proj: input_size_per_partition = 14784 不能被 min_thread_k = 128 整除
  Qwen2.5-72B（官方 AWQ / GPTQ，intermediate 补到 29696）
    TP8  Marlin 通过   FP8 块 通过

$ python3 mem_sim.py（节选）
  hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4  估算   39.79 GB  实际   39.77 GB  偏差  +0.1%
  Qwen/Qwen3-8B-FP8                                    估算    9.44 GB  实际    9.44 GB  偏差  -0.0%

$ python3 mem_sim.py --scenario decode（节选）
  batch           1      8     16     32     64    128    256    512   拐点 batch
  w4a16        3.88   3.85   3.82   3.77   3.32   1.70   1.00   1.00       54     （A10）
```

## 校准

1. 解析链：在真实环境里用 `python3 -m sglang.launch_server --model-path <checkpoint> [--quantization X]` 起服务，日志里的 `The model is convertible to awq_marlin during runtime`、`Using CLI-specified quantization`、报错文本都应和 `resolve_sim.py` 的输出逐字一致。换了 sglang 版本先 diff `configs/model_config.py` 的三张表。
2. checkpoint 体积：`mem_sim.py` 的口径（Linear 按位宽，scale 摊入，embedding / lm_head 留 BF16）在四个真实仓库上偏差 0.1% 以内。自己量化的模型如果大出很多，多半是 lm_head 或 MoE gate 被一起量化或没量化，查 `modules_to_not_convert` / `ignore`。
3. 误差数字：`quant_sim.py` 用的是合成权重和激活，只比较相对大小。真实模型用 llm-compressor 或 AutoAWQ 量化后跑 `python3 -m sglang.test.few_shot_gsm8k` 和一个长推理任务（AIME / GPQA），GSM8K 不掉不代表长推理不掉，FP4 KV 就是例子。
4. decode 加速：Roofline 是上限。Marlin 论文在 A10 上 batch 16 到 32 之前接近 4 倍、64 到 128 回落，比理想拐点早；端到端 vLLM 最高 2.8 倍。实测用 `python3 -m sglang.benchmark.one_batch --batch-size 1 8 32 128 --input-len 1024 --output-len 64` 对比 BF16 和量化版的 decode 吞吐。
5. KV 容量：启动日志 `max_total_num_tokens=` 对 `mem_sim.py --scenario kv` 的 token 数。BF16 换 FP8 KV 应该正好两倍，差得多说明 mem_fraction_static 或 CUDA Graph 预留也变了。
