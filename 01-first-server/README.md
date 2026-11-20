# SGL-01 第一个服务

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/01-first-server.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-01｜上手｜第一个服务：安装、启动、发请求和读懂启动日志。

验证基线：sglang 主干 `8fabf5e`、发行版 `v0.5.22`。日志格式和默认启发式都按这个提交写，升级后先拿 `sample_startup.log` 和一份新日志各跑一遍，看正则还能不能匹配。

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `estimate_startup.py` | CPU | 按主干默认启发式预测启动日志里的关键数字 |
| `serve.sh` | GPU | 起 Qwen3-8B 服务，日志同时落到 `logs/` |
| `parse_startup_log.py` | CPU | 从启动日志抽数字，和预测对账；也统计运行期的缓存命中和排队 |
| `smoke_test.py` | 任意 | 只用标准库，从健康检查到流式生成过一遍，失败即非零退出 |
| `sample_startup.log` | — | 按真实格式构造的测试夹具，数值是推算量级，不是实机录制 |

## 推荐顺序

先预测，再启动，再对账，最后发请求。预测和日志对不上，比服务起不来更值得查。

```bash
# 1. 启动前：预测 KV 池能装多少 token（CPU 上跑）
python3 estimate_startup.py --model qwen3-8b --gpu h100-sxm

# 2. 启动（GPU 机器上；24 GB 卡先换小模型）
bash serve.sh
MODEL=Qwen/Qwen3-0.6B bash serve.sh

# 3. 看到 "The server is fired up and ready to roll!" 后，另开终端对账
python3 parse_startup_log.py logs/serve-*.log --model qwen3-8b --gpu h100-sxm

# 4. 冒烟测试
python3 smoke_test.py --base http://127.0.0.1:30000
```

没有 GPU 也能把第 1、3 步跑通：

```bash
python3 parse_startup_log.py sample_startup.log --model qwen3-8b --gpu h100-sxm
```

## 期望输出

`estimate_startup.py --model qwen3-8b --gpu h100-sxm`：

```
Qwen3-8B on 1 x H100 SXM（nvidia-smi 81559 MiB）
[参数推导]
  chunked_prefill_size      8192
  decode cuda graph max_bs  256（36 个捕获档）
  prefill cuda graph 档数   58
  graph 预留                976 MB
  reserved_mem              13904 MB
  mem_fraction_static       0.83
[预测日志]
  Load weight begin. avail mem=79.05 GB
  Load weight end. ... avail mem=63.79 GB, mem usage=15.26 GB.
  KV Cache is allocated. #tokens: 366669, K size: 25.18 GB, V size: 25.18 GB
  max_total_num_tokens=366669, chunked_prefill_size=8192, max_running_requests=4096, context_len=40960
```

日志里写的 `GB` 实际按 2^30 字节算，也就是 GiB，对账时别再换算一次。

`parse_startup_log.py sample_startup.log --model qwen3-8b --gpu h100-sxm` 的对账段：

```
[对账]
  CUDA 上下文等（反推）       0.72 GiB
  mem_fraction_static         0.83（取自 server_args 行）
  权重 预测 15.26 GB，实际 15.31 GB
  max_total_num_tokens 预测 365959，实际 365312，偏差 +0.2%
```

解析脚本用日志里的 `Load weight begin. avail mem` 反推 CUDA 上下文占用，再用 `server_args` 行里的 `mem_fraction_static` 重算，剩下的偏差只来自权重实际大小和 KV 池内部的固定开销。实机偏差在 2% 以内算正常，超过 10% 一般是下面几种情况：

- 手动传了 `--mem-fraction-static` 以外的容量参数，比如 `--max-total-tokens`
- 开了投机解码（draft 模型和它的 KV 池也要显存）、DP Attention 或多模态
- 模型是 MLA（DeepSeek 系）或带滑窗层（Gemma、gpt-oss），每 token KV 的算法不同，见 SGL-02

`smoke_test.py` 六步全绿时大致是：

```
[OK] 就绪检查  /ready=200 /health=200
[OK] 模型信息  model=qwen3 is_generation=True reasoning_parser=qwen3 tool_call_parser=qwen
[OK] OpenAI chat 非流式  finish=stop prompt=... completion=... reasoning[:30]='...' content[:40]='...'
[OK] OpenAI chat 流式  TTFT=... ms  平均 ITL=... ms  chunks=... completion_tokens=...
[OK] 原生 /generate 流式  chunks=16 累计文本=True（默认 incremental_streaming_output=False）
[OK] 原生 /generate logprob  finish={'type': 'length', 'length': 4} ...
```

TTFT 和 ITL 只看量级。单请求时 decode 每步要把约 16 GB 权重读一遍，H100 带宽 3.35 TB/s，下限约 5 ms，实测 ITL 落在 6～8 ms 属于正常。正经测性能用 SGL-03 的压测脚本。

## 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动报 ValueError，提示增大 `--mem-fraction-static` | 加载权重后的剩余显存小于留给激活和 Graph 的部分，KV 预算算出来是负数 | 换小模型、加 `--tp`，或确认激活够用后调高 `--mem-fraction-static` |
| CUDA Graph 捕获阶段 OOM | 留给捕获的显存不够 | 调低 `--mem-fraction-static` 或 `--cuda-graph-max-bs-decode`（主干已拆成 `-decode` 和 `-prefill` 两个参数，旧名 `--cuda-graph-max-bs` 会因前缀歧义被 argparse 拒绝） |
| `/ready` 一直 503 | 预热请求没通过，状态没有置为 Up | 看日志里预热那条 `/generate` 的报错 |
| `/health` 503 但进程还在 | 20 秒内 Scheduler 没有回包，可能卡死，也可能被大 batch 占满 | 看最近的 `Decode batch` 日志和 `#queue-req` |
| 原生流式 `/generate` 文本重复 | 默认每条返回累计文本 | 客户端只取新增部分，或启动时加 `--incremental-streaming-output` |
