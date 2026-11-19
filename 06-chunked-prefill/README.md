# SGL-06 Chunked prefill

对应飞书文档：SGL-06｜原理｜Chunked prefill：切块大小、mixed chunk 与 TTFT / ITL 的取舍。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，单步耗时复用 `../03-bench/sim_serve.py` 的 Roofline 模型，KV 池容量复用 `../02-ledger/sim_ledger.py`，GPU 分档复用 `../01-first-server/estimate_startup.py`，显存预留复刻 `memory_hook.py` 的公式。

| 文件 | 作用 |
|---|---|
| `sim_chunked.py` | chunked prefill 模拟：PrefillAdder 的 chunk 预算（含 mixed 时扣掉的 decode 行），chunked_req 优先续跑，`--enable-mixed-chunk` 把 decode 行拼进 prefill 批，`--prefill-decode-interval` 在 prefill 之间强插 decode，以及不同 chunk 对显存预留和 KV 池的影响。日志格式和线上 `Prefill batch` / `Decode batch` 一致 |

## 和源码的对应

| 模拟里的函数 | sglang 源码 | 说明 |
|---|---|---|
| `Sim.step` | `managers/scheduler.py:3954` get_next_batch_to_run | 有 prefill 就先跑 prefill，没有才 decode |
| interval 逻辑 | `managers/scheduler.py:1403` _should_defer_prefill、`:1416` _arm_prefill_decode_interval | 每跑一个 extend 批（含 MIXED）重新上膛；running 少于 min_bs 时清零 |
| `Sim.prefill` | `managers/scheduler.py:4139` _get_new_batch_prefill_raw | 没有 chunked_req 且（批满或 waiting 空）时直接返回 |
| chunked_req 续跑 | `managers/scheduler.py:4241`–`4249` | 续跑排在所有新请求前面，所以默认模式下各块背靠背 |
| mixed 拼批 | `managers/scheduler.py:4440`–`4485`、`managers/schedule_batch.py:3274` mix_with_running | 有 logprob / input_embeds / beam 时不拼；KV 不够时 decode 行坐着等，不 retract |
| `Adder` | `managers/schedule_policy.py:617` PrefillAdder、`mem_cache/prefill_budget.py:60` PrefillBudget | rem_input = 16384 − num_mixed（`:649`），rem_chunk −= num_mixed（`:664`） |
| `Adder.add_chunked_req` | `managers/schedule_policy.py:1150` | 本块 = min(rem_chunk, remaining_total)；截断块不预留输出 |
| `Adder.add_one_req` | `managers/schedule_policy.py:1405`、`:1602` _select_prefill_admission | 关掉 chunk 时第二条以后的超长请求返回 OTHER；开 chunk 时超出部分截断成 new_chunked |
| 中间块不出 token | `managers/batch_result_processor.py:427` | 只有最后一块采样，TTFT 从最后一块算 |
| `reserved_mb` / `kv_tokens` | `arg_groups/memory_hook.py:297`–`325`、`:357` reserve_for_graph_mb | act = max(chunk, 2048)；关 chunk 时 act = max(max_prefill, 2048) 且不抓 prefill graph；大于 60G 的卡至少留 10G |
| 日志 | `managers/scheduler_components/metrics_reporter.py:754`、`load_inquirer.py:62` | #pending-token = waiting 的总长 + chunked_req 剩余部分 |

## 运行

```bash
# 1. 64 条短请求 decode 中途来一条 32k 长 prompt：默认 / mixed × 各种 chunk
python3 sim_chunked.py

# 2. 再看更小的块
python3 sim_chunked.py --chunks 512 1024

# 3. 不开 mixed，用 --prefill-decode-interval 在块之间插 decode
python3 sim_chunked.py --scenario interval

# 4. 泊松负载：8 req/s 短请求，每 6 s 一条 32k 长 prompt，跑 60 s
python3 sim_chunked.py --scenario poisson

# 5. chunk 对显存预留和 KV 池的影响，四档 GPU
python3 sim_chunked.py --scenario memory

# 6. 看日志：默认和 mixed 各一次
python3 sim_chunked.py --scenario trace --chunked-prefill-size 8192
python3 sim_chunked.py --scenario trace --chunked-prefill-size 8192 --enable-mixed-chunk
```

## 期望输出

场景 1：

```
# 64 条短请求（输入 1024、输出 512）t=0 进来，t=3s 来一条 32768 token 的长 prompt（输出 64）
# ITL 只统计 t=3s 之后、短请求全部结束之前的 decode 间隔；长请求的块数 = 它被切成几次 prefill
    模式  chunk    长请求块数   长TTFT ms  短TTFT p99  ITL p50  ITL p99  ITL max    停顿占比 prefill步  decode步
    默认    off        1       1497       1861     12.4     13.6   1507.6   22.9%        6      511
    默认   2048       16       1499       1870     12.4     13.6   1512.1   22.9%       48      511
    默认   4096        8       1503       1865     12.4     13.6   1509.7   22.9%       24      511
    默认   8192        4       1499       1862     12.4     13.6   1508.5   22.9%       12      511
    默认  16384        2       1497       1861     12.4     13.6   1507.9   22.9%        6      511
 mixed   2048       17       1533       1881     12.4    110.7    123.5   24.4%       50      494
 mixed   4096        9       1516       1871     12.4    176.1    246.6   23.0%       26      502
 mixed   8192        5       1510       1866     12.4     18.3    479.1   22.9%       14      506
 mixed  16384        3       1505       1865     12.4     13.6    889.5   22.8%        8      508
# 参考：64 路 decode 一步 11.9 ms；32768 token 整段 prefill 1494 ms
```

场景 2：

```
    默认    512       64       1519       1880     12.4     13.6   1526.5   23.0%      192      511
    默认   1024       32       1513       1861     12.4     13.6   1516.9   22.9%       96      511
 mixed    512       74       1665       2014     12.5     29.8     30.4    0.0%      211      437
 mixed   1024       35       1574       1925     12.4     59.4     62.4   21.9%      102      476
```

场景 3：

```
 chunk    N min_bs   长TTFT ms  ITL p99  ITL max    停顿占比    被推迟轮数
  2048    0      0       1499     13.6   1512.1   22.9%        0
  2048    2      0       1845    120.9    141.2   24.9%       96
  2048    8      0       4086    121.4    141.7   30.7%      384
  2048   32      0      13153     67.7    130.2   20.0%     1536
  2048    8     80       1499     13.6   1512.1   22.9%        0
  8192    0      0       1499     13.6   1508.5   22.9%        0
  8192    2      0       1569     13.6    496.3   23.3%       24
  8192    8      0       1768     13.6    496.3   23.0%       96
  8192   32      0       3851    244.1    496.8   27.9%      384
  8192    8     80       1499     13.6   1508.5   22.9%        0
# min_bs=80 大于在跑的 64 条，interval 每次都被清零，等于没开
```

场景 4（最后一行来自加 `--chunks 512` 再跑一次）：

```
    模式  chunk  短TTFT p50  短TTFT p99   长TTFT ms  ITL p50  ITL p99  ITL max    停顿占比 out tok/s
    默认    off         39       1739       1501     10.9     71.0   2088.7   46.6%      1897
    默认   2048         39       1503       1506     10.9     71.2   2095.0   47.1%      1897
    默认   8192         39       1672       1502     10.9     71.0   2089.6   46.9%      1897
 mixed   2048         40       1517       1564     10.5    104.1    126.9   27.3%      1901
 mixed   8192         40       1672       1720     10.6    181.7    482.1   26.0%      1900
 mixed  16384         39       1742       1775     10.6     59.5    892.1   25.8%      1900
 mixed    512         50       1533       1564      9.6     30.1     32.2    4.4%      1905
```

场景 5（摘录）：

```
      GPU  chunk   激活token  Graph MB    预留 MB   10G下限   frac  KV tokens      相对默认
  rtx4090  2048       2048       432     4144          0.831     30,442   +245.4%
  rtx4090  4096*      4096       496     7280          0.704      8,813     +0.0%
  rtx4090  8192       8192       560    13488          0.451          0   -100.0%
     l40s  2048       2048       400     4112          0.911    183,374    +13.6%
     l40s  4096*      4096       464     7248          0.843    161,395     +0.0%
     l40s 16384      16384       656    25872          0.438     30,488    -81.1%
 h100-sxm  2048       2048       848    10240      触发  0.874    391,995     +6.9%
 h100-sxm  4096       4096       912    10240      触发  0.874    391,995     +6.9%
 h100-sxm  8192*      8192       976    13904          0.830    366,669     +0.0%
 h100-sxm 16384      16384      1104    26320          0.677    278,601    -24.0%
 h100-sxm   off      16384       512    25728          0.685    283,206    -22.8%
     h200  8192*      8192       976    13904          0.903    808,172     +0.0%
     h200 16384      16384      1104    26320          0.817    720,624    -10.8%
# * 是该档 GPU 的默认 chunk；KV tokens 为 0 表示权重放不下或池子为负
```

场景 6 默认模式：4 个 prefill 批背靠背，每批 `#new-token: 8192`，`#pending-token` 24576 → 16384 → 8192 → 0，短请求 #0 的 decode 间隔（ms）是 `11 11 11 11 11 1509 13 13`。

场景 6 mixed：5 个批，前 4 批 `#new-token: 8128`（8192 减去 64 行 decode），最后一批 256，每批带 `[MIXED +64 decode]`，短请求 #0 的间隔是 `11 11 11 11 11 264 336 407 479 18 13`。

六个观察点：

1. **只开 chunk，decode 照样停 1.5 秒。** chunked_req 续跑排在所有新请求前面，被切成的块一块接一块地跑，中间没有 decode。不管块多大，ITL max 都约等于整段 prefill 的 1.5 s。chunk 本身只做两件事：限住单批激活显存，让别的新请求能在块的余量里一起进来。
2. **开 mixed 后，停顿上限变成一块的时间。** 每个 prefill 批顺带跑一步 decode，ITL max 从 1.5 s 降到 479 ms（8192）乃至 30 ms（512），基本和块大小成正比。代价是长请求 TTFT 慢 1% 到 11%，块越小越贵，因为每块都要重读一遍权重。
3. **后面的块比前面慢。** 第 k 块的注意力要读前 k−1 块的 KV，8192 的四块依次是 264 / 336 / 407 / 479 ms。ITL max 由最后一块决定，prompt 越长越明显。
4. **p99 看不到稀有的长停顿。** mixed 8192 的 ITL p99 只有 18.3 ms，16384 是 13.6 ms，比 2048 的 110.7 ms 还好看，因为停顿次数太少，落不进前 1%。判断 chunk 时看 ITL max 和停顿占比。
5. **interval 是不开 mixed 时的替代。** N 轮 decode 插在两块之间，ITL max 降到一块的时间，但长请求 TTFT 被拉长（2048、N=32 时 13 s）。min_bs 大于在跑的请求数，这个开关就失效。
6. **chunk 越大，KV 池越小。** 预留按 1.5 × chunk 加 prefill graph 算。H100 上 2048 和 4096 被 10G 下限拉平，16384 比默认少 24% 的 KV；L40S 上 16384 少 81%；RTX 4090 用 8192 直接起不来，这就是分档默认值的来历。

模拟是单线程、无 overlap、page_size=1，时间数字只用来排序和对比，不代表真实性能。
