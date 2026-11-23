# SGL-03 指标与压测

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/03-bench.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-03｜上手｜指标与压测：TTFT、TPOT、ITL 与 goodput。

验证基线：sglang 主干 `8fabf5e`。两个 Python 脚本只用标准库，CPU 上能跑；`bench.sh` 需要一个已启动的服务。

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `bench.sh` | GPU | 调 `python -m sglang.benchmark.serving` 做请求率扫描（开环）和并发扫描（闭环），结果追加进 `results/*.jsonl` |
| `sim_stats.py` | CPU | 读 serving 的 `--output-details` 结果，按主干口径重算分位数、goodput、ITL 尖峰、稳态窗口，找最大可承受负载 |
| `sim_serve.py` | CPU | 连续批处理服务的离散事件模拟，输出和 serving 同格式的 JSONL，没卡也能把上面的分析走一遍 |

`sim_serve.py` 的服务端模型：容量取自 `../02-ledger/sim_ledger.py`，单步耗时取 `../02-ledger/sim_roofline.py` 的下限除以利用率（decode 按 MBU 70%，prefill 按 MFU 55%）再加 0.3 ms 固定开销，调度是 sglang 默认的 prefill 优先。它的用途是看清曲线形状，不能拿来预测绝对数字。

## 推荐顺序

```bash
# 1. 没卡：先用模拟器把一条请求率扫描跑出来（约 4 秒，文件约 20 MB，不入库）
python3 sim_serve.py --sweep 2 4 8 12 16 18 20 24 --output-file sweep.jsonl

# 2. 换一组更严的 SLO 重算，看最大可承受负载怎么变
python3 sim_stats.py sweep.jsonl --goodput ttft:500 tpot:30 --target 0.95

# 3. 看某一档的完整分位数和 ITL 尖峰
python3 sim_stats.py sweep.jsonl --detail 12

# 4. prefill 插队与 mixed chunk
python3 sim_serve.py --sweep 8 12 16 18 20 --enable-mixed-chunk --output-file mixed.jsonl

# 5. 闭环：固定并发
python3 sim_serve.py --max-concurrency 64 --output-file c64.jsonl
python3 sim_stats.py c64.jsonl --detail c=64

# 6. 有卡：先起服务（../01-first-server/serve.sh），再压
bash bench.sh rate
bash bench.sh conc
python3 sim_stats.py results/rate-*.jsonl --detail all
```

## 期望输出

`sim_serve.py --sweep 2 4 8 12 16 18 20 24`：

```
# Qwen3-8B on 1 x H100 SXM: KV 池 366,669 token, chunk 8192, 请求上限 4096, MBU 70%, MFU 55%
# 单步耗时：decode b=1 6.82 ms，b=64 11.39 ms，prefill 8192 token 264.6 ms
        负载  req/s out tok/s     并发  TTFT p50  TTFT p99  TPOT p50  TPOT p99  ITL p99  ITL max  goodput    达标率
         2   2.04       523    4.0        29        58       7.6       8.4     36.5     95.4     2.04 100.0%
         4   4.07      1041    8.9        29        58       8.5       9.8     37.1    154.2     4.07 100.0%
         8   8.07      2066   23.3        29        58      11.3      13.1     67.4    184.5     8.07 100.0%
        12  11.96      3063   49.9        29        87      16.3      19.4     99.6    332.4    11.96 100.0%
        16  15.63      4000  111.7        29       117      28.6      32.8    191.1    418.2    15.63 100.0%
        18  17.18      4398  181.9        29       117      45.2      51.2    257.8    489.4    16.60  96.6%
        20  17.90      4584  250.7       829      2516      57.2      60.1    347.7    738.6     3.17  17.7%
        24  18.12      4638  318.2      3721     10192      58.2      60.0    377.0   1104.3     0.33   1.8%
# SLO ttft≤2000ms tpot≤50ms，达标率门槛 99%：最大可承受负载 16（goodput 15.63 req/s）
# 输出吞吐在 18 之后增长不到 5%，再往上加负载只会推高排队
```

三件事：输出吞吐在 18 req/s 附近变平，TTFT 却在 20 req/s 一下跳了两个数量级；TPOT 的 p99 和 p50 只差 15% 左右，ITL 的 p99 却是 p50 的十倍；达标率从 96.6% 掉到 17.7% 只隔了一档。

`--goodput ttft:500 tpot:30 --target 0.95` 把最大可承受负载从 16 拉回到 12。

`--detail 12`：

```
## 负载 12：完成 1000，时长 83.6 s，输入 12252 tok/s，输出 3063 tok/s，平均并发 49.9
      指标      mean       p50       p90       p95       p99
    TTFT      35.6      29.4      58.4      58.4      87.5
    TPOT      16.2      16.3      18.5      18.8      19.4
     ITL      16.2      10.7      39.4      41.0      99.6
     E2E    4168.2    4196.5    4745.5    4818.4    4992.6
# ITL 尖峰：超过中位数 3 倍（32.1 ms）的间隔占 11.9%，却占了生成时间的 43.0%；p99/p50 = 9.3，最长一次 332 ms
# goodput 11.96 req/s，全部达标 100.0%（ttft 100.0%，tpot 100.0%）
```

mixed chunk 打开后同一组请求率：

```
        负载  req/s out tok/s     并发  TTFT p50  TTFT p99  TPOT p50  TPOT p99  ITL p99  ITL max  goodput    达标率
         8   8.07      2066   21.2        30        59      10.3      11.7     30.2     87.9     8.07 100.0%
        12  11.98      3067   41.1        31        89      13.4      15.4     59.6    117.7    11.98 100.0%
        16  15.73      4028   74.1        32       119      18.3      20.8     61.0    119.0    15.73 100.0%
        18  17.51      4482  100.4        33       119      22.7      25.1     90.5    148.3    17.51 100.0%
        20  19.18      4910  136.6        60       148      29.0      32.4     92.1    149.8    19.18 100.0%
```

闭环并发扫描（每档 1000 条，`--max-concurrency 16/32/64/128/256`）：

```
  inf c=16   6.41      1640   15.9       233       466       7.9       8.8      8.0    240.6     6.41 100.0%
  inf c=32   9.74      2493   31.5       466       931      10.0      11.8      9.3    707.2     9.74 100.0%
  inf c=64  13.31      3407   62.8       931      1862      14.1      17.8     11.9   1640.5    13.31 100.0%
 inf c=128  16.30      4173  125.4      1862      3725      22.4      29.7     17.0   3507.0     8.35  51.2%
 inf c=256  18.36      4701  250.4      3725      7450      39.0      53.6     27.3   7240.0     2.94  16.0%
```

固定长度、同时发出的闭环里请求成批同步，TTFT 被成批的 prefill 拉长，ITL 的 p99 反而很低。这是固定长度加 rate=inf 的人为现象；把 `--random-range-ratio` 改成 0.5 后 c=64 的 TTFT p50 只剩 25 ms。

## 用到真实结果上要注意

- serving 的 `--output-details` 不写逐请求的发出时刻，`sim_stats.py --steady` 只对模拟结果有效；真实压测要稳态口径，用 `python -m sglang.benchmark.steady_state_serving`
- 真实结果里没有逐请求 e2e，`sim_stats.py` 按 `ttft + sum(itls)` 还原，和 serving 内部的 latency 有毫秒级出入
- 用投机解码时 serving 的 ITL 会按重新分词的 token 数摊，口径见文档第 3 章
- 过载档的 TTFT 取决于 `--num-prompts`：同样 24 req/s，500 条时 p50 是 117 ms，2000 条时是 9.8 s。超过容量后的延迟数字只说明「过载了」，不要拿来比较
