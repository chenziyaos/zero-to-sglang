# SGL-05 调度器

对应飞书文档：SGL-05｜原理｜调度器：waiting、running 与 KV 预算。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，单步耗时复用 `../03-bench/sim_serve.py` 的 Roofline 模型，KV 池容量复用 `../02-ledger/sim_ledger.py`。

| 文件 | 作用 |
|---|---|
| `sim_scheduler.py` | 调度器模拟：waiting / running 两条队列，PrefillAdder 预算，fcfs / lof / lpm / dfs-weight 排序（含批内前缀去重），chunked prefill 续跑，token 级 radix 前缀缓存，new_token_ratio 衰减，decode 时 KV 不够就 retract，优先级抢占。日志格式和线上 `Prefill batch` / `Decode batch` / `KV cache pool is full` 一致 |

## 和源码的对应

| 模拟里的函数 | sglang 源码 | 说明 |
|---|---|---|
| `Scheduler.step` | `managers/scheduler.py:2029` event_loop_normal、`:3954` get_next_batch_to_run | 先试 prefill，凑不出来才 decode；空闲时复位 ratio（`:5311` on_idle） |
| `Scheduler.get_new_batch_prefill` | `managers/scheduler.py:4139` _get_new_batch_prefill_raw | chunked_req 优先续跑；行数满、预算满或第一次 NO_TOKEN 就停 |
| `Adder` | `managers/schedule_policy.py:617` PrefillAdder、`mem_cache/prefill_budget.py:62` PrefillBudget | remaining_total = 空闲 + 可驱逐 − total_offset |
| `Adder.add_chunked_req` | `managers/schedule_policy.py:1150` | 截断时不预留输出 |
| `Adder.add_one_req` | `managers/schedule_policy.py:1405`、`:1602` _select_prefill_admission | 输入 + 输出预留 + 1 页 ≥ remaining_total 返回 NO_TOKEN |
| `Adder.budget_state` | `managers/schedule_policy.py:884` | NO_TOKEN / OTHER / CONTINUE |
| `Adder.preempt_to_schedule` | `managers/schedule_policy.py:1717` | 优先级差 > 阈值（默认 10）才抢 |
| `Scheduler.calc_priority` | `managers/schedule_policy.py:213`、`:314`、`:356` | 队列 > 128（`:137`）时 lpm 退回 fcfs；批内去重阈值 32（`:110`、`:117`） |
| `dfs_weight_order` | `mem_cache/base_prefix_cache.py:456` | 子树里等待请求越多越先走 |
| `Scheduler.checkpoint` | `mem_cache/common.py:271` checkpoint_kv_cache | prefill 后立即把算好的部分插进树 |
| `Scheduler.update_running_batch` | `managers/scheduler.py:4524`、日志 `:4591` | KV 不够先 retract，再按公式抬高 ratio |
| `Scheduler.check_decode_mem` | `managers/schedule_batch.py:3439` | page_size=1 时每条请求要 1 个 token |
| retract 顺序 | `managers/schedule_batch.py:3458` retract_decode、`:2367` release_req、`:2056` reset_for_retract | 输出最少的先退，同样少时输入最长的先退 |
| `r_init / r_min / r_decay` | `managers/scheduler_components/new_token_ratio_tracker.py:14` | 0.7 × conservativeness，600 步降到 0.14 倍 |

## 运行

```bash
# 1. 共享前缀，同组扎堆到达：fcfs 和 dfs-weight 把同组 8 条塞进同一批，各算一遍前缀；lpm 的批内去重只放 1 条
python3 sim_scheduler.py --scenario policy

# 2. 各组轮流到达：第一批里每组只有 1 条，去重没有用武之地，三种策略差不多
python3 sim_scheduler.py --scenario policy --interleaved

# 3. 队列 400 条超过 128：lpm 前几轮退回 fcfs，差距缩小
python3 sim_scheduler.py --scenario policy --num 400

# 4. conservativeness 0.3 / 1.0 / 1.3，长输出把 KV 池压满
python3 sim_scheduler.py --scenario ratio

# 5. 优先级抢占：关 / 开两次对比
python3 sim_scheduler.py --scenario priority

# 6. 自定义：看前 12 行调度日志
python3 sim_scheduler.py --workload random --num 400 --out 1024 --max-total-tokens 120000 \
    --schedule-conservativeness 0.3 --log-head 12
```

## 期望输出

场景 1：

```
# 共享前缀负载：128 条请求，16 组 system prompt 各 1024 token，私有后缀 128 token，输出 64，同组扎堆到达，全部 t=0 发出
     policy prefill步    新算token    命中token     命中率   TTFT均值s TTFT p99 s     总时长s
       fcfs        17      131200       16384    11.1%      2.011      3.735      4.77
        lpm         8       57728       93184    61.7%      1.002      1.653      2.68
 dfs-weight        17      131200       16384    11.1%      2.011      3.735      4.77
# 理想下限：每组前缀只算一次 = 16 x 1024 + 128 x 128 = 32768 token
```

场景 2、3：

```
# --interleaved
       fcfs         5       35072      113536    76.4%      0.826      1.007      2.03
        lpm         6       41216      109312    72.6%      0.718      1.182      2.21
 dfs-weight         6       41216      109312    72.6%      0.718      1.182      2.21
# --num 400
       fcfs        18      140131      356352    71.8%      2.732      5.719      6.74
        lpm        16      121978      374390    75.4%      2.648      5.204      6.23
 dfs-weight        18      136448      361501    72.6%      2.730      5.615      6.64
```

场景 4：

```
conserv  ratio初值  ratio最低  retract次     退回请求   重算token decode usage  峰值running     总时长s   输出tok/s
    0.3     0.210    0.0294         22        22      28604         0.79         117     56.50       5404
    1.0     0.700    0.0980          7         7       5897         0.76         110     56.84       5372
    1.3     0.910    0.1274          0         0          0         0.75         104     57.06       5351
# conservativeness 0.3 的前三条 retract 日志：
[t=   6.154s] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 1289, #new_token_ratio: 0.1266 -> 0.2910
```

场景 5：

```
## 关闭抢占：高优先级 TTFT 8.217s，完成于 9.14s；被抢占 0 次，低优先级重算 0 token，低优先级最晚完成 8.70s
## 打开抢占：高优先级 TTFT 0.021s，完成于 1.01s；被抢占 1 次，低优先级重算 5 token，低优先级最晚完成 9.14s
[t=   0.506s] Preempt req 7 (priority 0) for req 8 (priority 100)
[t=   1.006s] Prefill batch, #new-seq: 1, #new-token: 5, #cached-token: 2048, ...
```

五个观察点：

1. **同组扎堆时 fcfs 命中率只有 11%。** 同一批里 8 条请求共享同一个 1024 token 前缀，可前缀要等这一批算完才进 radix 树，批内彼此看不到，于是前缀被算了 8 遍。lpm 的批内去重把命中 waiting 队列小树超过 32 token 的请求挪到队尾，一批只放每组 1 条，下一批就能全部命中。dfs-weight 在 8fabf5e 里不做这个去重，所以和 fcfs 一样。
2. **到达顺序变了，结论就变了。** 轮流到达时 fcfs 反而最好，因为第一批天然每组 1 条。lpm 的价值取决于负载形状，压测时用 `--interleaved` 和默认各跑一遍再下结论。
3. **队列超过 128 条，lpm 前几轮按 fcfs 排。** 场景 3 的差距从 50 个百分点缩到 4 个。线上积压越深，lpm 越不起作用，这是源码里有意为之的开销保护。
4. **conservativeness 越小，retract 越多。** 0.3 时 ratio 从 0.21 起步，衰减到 0.029 后准入几乎只看输入长度，KV 池很快被 decode 撑爆，22 次 retract 浪费了 28604 token 的重算。每次 retract 后 ratio 被公式抬到 0.29 左右，随后又衰减，形成锯齿。本负载下三档吞吐只差 1%，可 TTFT 和重算量差很多。
5. **抢占的代价是被抢请求重算。** 高优先级 TTFT 从 8.2 s 降到 0.02 s。被抢的 req 7 前缀已经 checkpoint 到树里，回来时只重算 5 token；如果前缀被驱逐了，就是整段重算。

模拟是单线程、无 overlap、page_size=1，时间数字只用来排序和对比，不代表真实性能。
