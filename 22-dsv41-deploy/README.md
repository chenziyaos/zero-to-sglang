# SGL-22 DeepSeek-V4.1-Flash 部署规划

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/22-dsv41-deploy.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-22｜实战｜DeepSeek-V4.1-Flash：消费级显卡与 H20 上的 SGLang 部署。文档把可用机器分成五档，从 8 卡 H20 原生部署一直到消费级单卡加 KT-Kernel 混合推理。本目录的 plan.py 对应文档第 2、3、5 章：输入卡型、卡数和主机内存，算出机器落在哪一档、每卡显存怎么分，再给出和文档一致的启动命令。有公开实测的配置（8×H20-3e、4×RTX PRO 6000、2×RTX 5070 Ti）会把实测行一起打出来，自己压完可以直接对照。

验证基线：sglang 主干 `8fabf5e`。plan.py 只用标准库，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| plan.py | 分档、每卡显存账（骨干、Engram、剩给 KV 的空间和能装的 token 数）、KT 路径每层常驻专家数和单流 decode 上限、GGUF 量化档选择、公开实测对照、启动命令 |
| host_setup.sh | 3.1 节的三项主机设置：THP shmem 改 advise、min_free_kbytes 留 8 GB、关 swap，需要 root |
| bench_h20.sh | 4.7 节的压测：预热一轮，再按并发 1 到 64 扫 8192 进 1024 出，最后把每档的输出吞吐、TTFT 和 ITL 中位数汇成一张表 |
| smoke.py | 7.3 节的冒烟：17×19 要答 323，带工具定义时要解析出 get_weather，content 里不能混原始标记 |
| test_plan.py | 18 个用例 |

## 和文档的对应

| 本目录 | 文档 |
|---|---|
| `BACKBONE_GB`、`ENGRAM_GB`、`kv_bytes` | 第 2 章：510 GB checkpoint 拆成骨干约 307 GB 和 Engram 约 203 GB。全局 KV 每 token 在 SM100/SM103 上约 890 字节，其他架构走旧布局，按约 1790 字节估 |
| `native_budget` | 2.3 节每卡显存表，Engram 留显存和放主机两种算法 |
| `classify` | 第 5 章选卡路线图。原生两档按每卡账判断：Engram 留显存后每卡 KV 还有 5 GB 以上算第 1 档，Engram 放主机才够算第 2 档。4 张及以上 SM120 卡、主机内存 128 GB 以上，按 3.4 节的社区配方归第 2 档 |
| `kt_gpu_experts`、`decode_ceiling` | 3.2、3.3 节：每层常驻 GPU 的专家数，每 token 要从主机内存读约 4.5 GB 路由专家权重 |
| `GGUF` | 3.5 节 Unsloth 动态 GGUF 表 |
| `PUBLIC`、`PUBLIC_NEAR` | 3.1、3.4、4.2 节引用的公开实测。H20 96G 没有同卡数据，借 H20-3e 的数并提示带宽差 |
| `command` | 3.4 节 4×RTX PRO 6000 compose、4.3 节 TP8 加 DSpark、4.4 节 TP8 吞吐配置、4.5 节 TP4 加 Engram 放主机 |
| smoke.py、bench_h20.sh | 7.3 节：服务起来先跑冒烟，再压测，结果和 4.2 节 8×H20-3e 的实测比 |

这些数字是按 config.json 和 checkpoint 体积推出来的估算值，用来判断方向。真正上线以启动日志里的显存占用和压测结果为准。

## 运行

```bash
cd zero-to-sglang/22-dsv41-deploy
python3 plan.py --gpu h20-96 --n 8
python3 plan.py --gpu h20-96 --n 4 --host-mem 384
python3 plan.py --gpu b200 --n 2 --host-mem 512
python3 plan.py --gpu rtx-pro-6000 --n 4 --host-mem 128
python3 plan.py --gpu rtx5070ti --n 2 --host-mem 512 --mem-bw 410
python3 plan.py --gpu rtx4090 --n 1 --host-mem 128
python3 -m unittest test_plan.py -v

# 服务起来以后
python3 smoke.py --url http://127.0.0.1:30000
PORT=30000 OUT=results ./bench_h20.sh
```

第一条落在第 1 档：Engram 留显存时每卡还剩约 20.7 GB 给 KV，默认给出 DSpark 低延迟命令，加 `--mode throughput` 换成 Engram 放主机、64 并发的配置。输出末尾附 sglang #46310 在 8×H20-3e 上的实测（8K 输入、8 并发 984 tok/s），并提示 96G 版带宽低约 17%。第三条值得对比着看：两张 B200 显存合计 360 GB，比骨干大，但 Engram 再进显存后每卡 KV 不到 5 GB，只能把 Engram 放主机，落在第 2 档。第四条是四张 RTX PRO 6000 走 SGLang 原生 TP4，Engram 放 NVMe 加 64 GiB 内存缓存，128 GB 内存就够，附社区实测 8K 输入单流 196 tok/s、8 并发 705 tok/s。第五条给出主机内存带宽对应的单流 decode 上限约 91 tok/s，实测通常只有一半多。

冒烟两项都过再压测。bench_h20.sh 末尾的表里，8 并发那一行的输出吞吐到 8×H20-3e 实测 984 tok/s 的六成以上算正常。不到四成，先查注意力是不是落到了 Triton 回退，再看 DSpark 的平均接受长度是不是低于 3。
