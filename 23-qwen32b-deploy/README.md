# SGL-23 Qwen3-32B 部署规划

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/23-qwen32b-deploy.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-23｜实战｜Qwen3-32B：单卡到两卡的 SGLang 部署，以及迁到 Qwen3.8-27B。文档先算 KV 能放多少 token，再按卡的架构选精度和卡数。本目录的 plan.py 对应文档第 2、3、7、8 章：输入卡型，复算 sglang 默认的 mem_fraction_static，算出每卡权重、KV 预算和能放的 token 数，选出精度和 TP，再给出启动命令。加 `--qwen38` 改算 Qwen3.8-27B 的 GDN 状态槽和 KV 怎么分。

验证基线：sglang 主干 `8fabf5e`。plan.py 只用标准库，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| plan.py | 默认 mem_fraction_static 复算、每卡显存账、KV token 数、精度和卡数选择、FP8 分块切分校验、batch 1 decode 理论上限、启动命令；Qwen3.8-27B 的 `--mamba-full-memory-ratio` 平衡值和并发估算 |
| smoke.py | 7.3 节的冒烟：关思考、开思考、工具调用三项，检查 reasoning_content 和 tool_calls 是否分流干净 |
| bench.sh | 第 6 章的压测：预热一轮，再按并发 1 到 64 扫 6144 进 2048 出，汇总输出吞吐、TTFT 和 ITL 中位数 |
| test_plan.py | 25 个用例 |

## 和文档的对应

| 本目录 | 文档 |
|---|---|
| `kv_bytes_per_token`、`CKPT_GB` | 第 2 章：64 层、8 个 KV 头、head_dim 128，KV 每 token BF16 256 KiB。checkpoint 取 Hugging Face 文件体积：BF16 65.52 GB、FP8 34.32 GB、AWQ 19.33 GB |
| `capacity_defaults`、`default_mem_fraction` | 2.2 节：按 memory_hook.py 复算默认 chunked_prefill_size、CUDA Graph 档数和预留显存，显存大于 60 GiB 时预留不低于 10 GiB |
| `kv_budget_mib`、`kv_tokens` | 2.3 节每卡显存账：加载前空闲显存乘 mem_fraction_static，减去每卡权重 |
| `fp8_kernel`、`fp8_tp_ok` | 第 3 章：FP8 在不同架构上走哪条内核，TP 切开后每片要是 128 的整数倍 |
| `choose`、`TUNED_CHUNK` | 第 7 章选卡：先少卡，再按架构偏好的精度，最后才把 chunked_prefill_size 压到 2048 |
| `command` | 第 4、5 章：parser、YaRN、EAGLE3 草稿参数 |
| `QWEN_SPEED_H20` | 第 6 章引用的 Qwen 官方 H20 速度基准 |
| `q38_*` | 第 8 章：每请求状态字节、每 token KV 字节、配比公式和剩余显存的拆分 |

这些数字是按 config.json、checkpoint 体积和 sglang 默认值公式推出来的估算，用来判断方向。真实容量以启动日志里的 `max_total_num_tokens` 为准。

## 运行

```bash
cd zero-to-sglang/23-qwen32b-deploy
python3 plan.py --gpu h20-96
python3 plan.py --gpu h100 --n 1 --precision bf16
python3 plan.py --gpu rtx4090
python3 plan.py --gpu rtx4090 --n 1 --precision awq
python3 plan.py --gpu a100-80 --target-tokens 131072 --context 131072
python3 plan.py --qwen38 --gpu h200 --spec38 eagle
python3 -m unittest test_plan -v

# 服务起来以后
python3 smoke.py --url http://127.0.0.1:30000
PORT=30000 OUT=results ./bench.sh
```

第一条在 H20 96G 上选 FP8 单卡：sglang 推出 mem_fraction_static 0.858，预留 13.6 GiB，每卡权重 32.0 GiB，KV 约 49 GiB，能放约 20 万 token，末尾附 Qwen 官方 batch 1 实测 46 tok/s。第二条强制 H100 跑 BF16 单卡：默认参数下 KV 只有约 1.8 万 token，plan.py 自动把 chunk 压到 2048 也只到约 3.2 万，差一点放不下一条 32K 请求，`--context-length` 降到 16384，并提示加卡。第三条是 4090 自动找到的方案：两卡 AWQ TP2，加 `--chunked-prefill-size 2048`，约 8.3 万 token。第四条强制单卡，KV 只剩约 5 千 token，`--context-length` 会被压到 4096。第五条打开 YaRN，命令里会带 factor 4.0 的 rope_scaling。最后一条算 Qwen3.8-27B 在 H200 上开 EAGLE 时的配比 4.59，和 SGLang cookbook 的推荐值一致，同时打印默认 0.9 和平衡值下各能并发多少请求。

冒烟三项都过再压测。bench.sh 的 1 并发那一行，输出吞吐和 Qwen 官方 H20 表里 6144 输入那一档对着看，FP8 单卡差得太多，先看启动日志里 FP8 是不是走了 DeepGEMM，再看 CUDA Graph 有没有抓上。
