# zero-to-sglang

“从 0 开始 SGLang”飞书系列的配套代码。每章一个目录，目录名和文档章节号一一对应，索引页是 SGL-00。

系列文档的 Markdown 版放在 [`book/`](book/README.md)，从索引页 SGL-00 读起；各章末尾有上一章、下一章的链接。

## 版本基线

- SGLang 主干提交 `8fabf5e`（2026-10-28），发行版 `v0.5.22`（2026-10-19）
- 官方镜像 `lmsysorg/sglang:latest`，CUDA 13 环境
- CPU 练习只依赖 Python 3.10 及以上和 numpy

## 文件命名约定

| 前缀或后缀 | 含义 | 运行环境 |
|---|---|---|
| `sim_*.py` | 模拟器或计算器，复现主干的关键数据结构和账本 | CPU |
| `gpu_*.py` | 需要真实模型和 GPU 的实验 | GPU |
| `*.sh` | 启动、压测或部署脚本 | GPU |
| `test_*.py` | 模拟器的单元测试，`python3 -m pytest` 或直接运行 | CPU |

## 目录

| 目录 | 章节 | 内容 |
|---|---|---|
| `common/` | 全系列 | 模型结构和 GPU 规格表 |
| `01-first-server/` | SGL-01 第一个服务 | 启动脚本、冒烟测试 |
| `02-ledger/` | SGL-02 推理账本 | 显存与 Roofline 计算器 |
| `03-bench/` | SGL-03 指标与压测 | 压测脚本、结果统计 |
| `04-request-path/` | SGL-04 请求旅程 | 三进程消息流模拟 |
| `05-scheduler/` | SGL-05 调度器 | 调度循环与策略模拟器 |
| `06-chunked-prefill/` | SGL-06 Chunked prefill | 切块对 TTFT、ITL 的影响 |
| `07-memory-pool/` | SGL-07 内存池与页表 | 两级映射与页式分配器 |
| `08-radix-attention/` | SGL-08 RadixAttention | 基数树：匹配、插入、加锁、驱逐 |
| `09-overlap-cudagraph/` | SGL-09 Overlap 与 CUDA Graph | 流水时间线模拟 |
| `10-attention-backend/` | SGL-10 注意力后端 | 后端选择表与微基准 |
| `11-parallelism/` | SGL-11 TP、DP Attention、EP | 并行方案账本 |
| `12-quantization/` | SGL-12 量化 | 量化误差与显存对比 |
| `13-speculative/` | SGL-13 投机解码 | 接受率与加速比 |
| `14-hicache/` | SGL-14 HiCache | 分层缓存命中模拟 |
| `15-pd-disagg/` | SGL-15 PD 分离 | P、D 配比与 KV 传输账本 |
| `16-structured-lora/` | SGL-16 结构化输出与 LoRA | token mask 与多 LoRA 批处理 |
| `17-tuning/` | SGL-17 参数调优手册 | 日志解析与调参记录模板 |
| `18-observability/` | SGL-18 可观测与 profiling | 指标抓取与 trace 分析 |
| `19-cluster-rl/` | SGL-19 集群与 RL rollout | 路由策略模拟 |
| `20-contrib/` | SGL-20 前沿复盘（读新 PR 的工具） | check_pr.py 标题 / 规模 / CODEOWNERS 自查、find_symbol.sh、replay_utilization.py、PR 模板与 #22713 修法补丁 |
| `21-deepseek/` | SGL-21 DeepSeek 开源与 SGLang | check_config.py 推 V4.1 注意力后端与校验 Engram 配置、align_bench.py 对齐 DeepInfer 复测测法 |
| `22-dsv41-deploy/` | SGL-22 DeepSeek-V4.1-Flash 部署 | plan.py 按卡型分档、算每卡显存账并给启动命令，host_setup.sh 消费级主机设置，bench_h20.sh H20 压测 |
| `23-qwen32b-deploy/` | SGL-23 Qwen3-32B 部署 | plan.py 按卡型选精度和卡数、复算默认 mem-fraction 和 KV 容量并给启动命令，含 Qwen3.8-27B 状态槽配比；smoke.py、bench.sh |

目录会随章节发布逐个填满，还没发布的章节目录里只有一份 README 写明计划。

## 快速试跑

```bash
cd zero-to-sglang/02-ledger
python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
python3 sim_ledger.py --model llama3.1-70b --gpu h100-sxm --tp 4 --kv-dtype fp8
```

## 参考

- SGLang：https://github.com/sgl-project/sglang
- Mini-SGLang：https://github.com/sgl-project/mini-sglang
- zero-to-sglang 课程：https://github.com/datawhalechina/zero-to-sglang
- 官方文档：https://docs.sglang.io/
