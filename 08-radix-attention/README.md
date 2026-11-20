# SGL-08 RadixAttention

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/08-radix-attention.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-08｜原理｜RadixAttention：基数树的匹配、插入、加锁与驱逐。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU 和 sglang 本体。

| 文件 | 作用 |
|---|---|
| `radix_core.py` | 经典 `RadixCache` 的逐行复刻：`RadixKey`（命名空间、整页 child_key）、`TreeNode`、匹配时分裂、插入时 priority 取 max 和 hit_count 去重、inc/dec_lock_ref 两本账、只删叶子的堆驱逐，外加七种驱逐策略和一个按页分配的分配器 |
| `sim_radix.py` | 把 match → 加锁 → 分配 → checkpoint → decode → release 串成最小调度器，跑七个场景：完整生命周期、命中率随池子大小、五种驱逐策略、出队顺序、n 采样预热、page size、四类泄漏 |
| `test_radix.py` | 11 个边界用例：不满一页不进树、分裂保值、chunked 插入不重复计 hit、两本账、驱逐级联、空闲不变式 |

## 和源码的对应

路径都相对 `python/sglang/srt/`。

| 模拟里的函数 | sglang 源码 | 说明 |
|---|---|---|
| `RadixKey` | `mem_cache/radix_cache.py:59` | token 序列 + extra_key（lora_id 等）+ cache_salt |
| `RadixKey.page_aligned` / `match` | `radix_cache.py:147`、`:178`、`:182` | 匹配长度和 key 都向下取整到整页；源码用倍增 + 二分找第一个不同的位置 |
| `RadixKey.child_key` | `radix_cache.py:226` child_key_at | page 1 用单个 token，page > 1 用整页 tuple；命名空间拼进 key，不同 LoRA 分到不同子树 |
| `TreeNode` | `radix_cache.py:250` | 根节点 lock_ref = 1、priority = -sys.maxsize（reset `:343`） |
| `match_prefix` / `_split_node` | `radix_cache.py:364`、`:651`、`:677` | 沿路刷新 last_access_time；停在节点中间时分裂，新节点继承 lock_ref / priority / hit_count |
| `insert` | `radix_cache.py:428`、`:705` _insert_helper | priority 沿路径取 max；只有超过 inserted_len 的部分才 hit_count + 1；新叶子计入 evictable |
| `Engine.checkpoint` | `radix_cache.py:450` _insert_cache、`:510` checkpoint；`mem_cache/common.py:271` | 插入、释放和树里重复的槽位、重新匹配、改写 req_to_token 行、cache_protected_len 前移、锁换到新节点；split_prompt 再插一次 prompt，优先级 + 1 |
| `inc_lock_ref` / `dec_lock_ref` | `radix_cache.py:582`、`:597` | 0 → 1 时 len(key) 从 evictable 挪到 protected，1 → 0 挪回；dec 没有 >= 0 断言 |
| `evict` / `evictable_leaves` | `radix_cache.py:552`、`:790`、`:780` | 可驱逐叶子建小根堆，弹出即 free，父节点无子无锁时压回堆 |
| `make_policy` | `mem_cache/evict_policy.py:16`–`:84` | lru / lfu / fifo / mru / priority / tlru / slru；tlru 在源码里只有 UnifiedRadixCache 支持 |
| `Engine.admit` | `mem_cache/common.py:193`、`:221` match_kv_cache；`managers/schedule_policy.py:1695` | 匹配上限 input_len - 1（`managers/schedule_batch.py:1773`），准入提交时 req.lock = tree_cache.lock(last_node) |
| `Engine._reserve` | `mem_cache/allocator/base.py:155` evict_to_free_tokens；`mem_cache/common.py:283` | 差多少驱逐多少 |
| `Engine.release` | `mem_cache/common.py:418` release_kv_cache | 结束时 checkpoint，再还 [cache_protected_len, 已分配)，最后解锁 |
| `Engine.invariant` | `managers/scheduler_components/invariant_checker.py:79`、`:297` | available + evictable + protected + session_held + uncached == total |
| `run(order='lpm')` | `managers/schedule_policy.py:314` | 等待队列超过 128 条时 lpm 退回 fcfs |
| `run(order='dedup')` | `managers/schedule_policy.py:356` _compute_prefix_matches | 批内去重：和等待队列里排在前面的请求共享 >= 32 token 的推后 |
| `run(warmup=True)` | `managers/tokenizer_manager.py` n > 1 分支（约 `:2130`） | 先发一条 max_new_tokens=0 的请求把 prompt 放进树 |
| leak [2] 的报错 | `managers/scheduler.py:5366`–`:5370`、`invariant_checker.py:452` | 空闲时检查，`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` 默认开 |
| leak [3] 的报错 | `invariant_checker.py:350` _check_kv_page_invariants | 需要 `SGLANG_CHECK_KV_PAGE_INVARIANTS=1` |
| leak [4] 的报错 | `mem_cache/unified_cache/unified_tree_core.py:3088` | Unified 树的 sanity_check 查 lock_ref < 0，经典树没有；只在 `SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK=1` 且 SWA / Mamba 混合模型时调用（`invariant_checker.py:484`） |

线上默认走的是 `UnifiedRadixCache`（`mem_cache/registry.py:83`），树核默认 Rust 实现（`environ.py` 的 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND`）。经典 `RadixCache` 和它语义一致、代码短得多，适合读。

## 运行

```bash
# 1. 源码自带的小例子 + 两条请求的完整生命周期，默认跑 ps=1 和 ps=4
python3 sim_radix.py

# 2. 只看某个 page size
python3 sim_radix.py --scenario walk --page-size 4

# 3. 四种负载 × 池子大小的命中率
python3 sim_radix.py --scenario hitrate

# 4. lru / lfu / slru / priority / tlru 在三种负载下的命中率和 prefill 尾延迟
python3 sim_radix.py --scenario policy

# 5. fcfs / lpm / 批内去重
python3 sim_radix.py --scenario order

# 6. n = 8 采样要不要预热
python3 sim_radix.py --scenario nsample

# 7. page size 对命中和节点数的影响
python3 sim_radix.py --scenario pagesize

# 8. 忘记解锁、忘记释放、重复释放、重复解锁
python3 sim_radix.py --scenario leak

# 9. 单元测试
python3 -m unittest test_radix -v
```

policy 场景最慢，几分钟左右，其余都在一分钟内。

## 期望输出

场景 1（ps=1，第 3 步）：

```
[3] B 准入：和 A 共享 12 token 系统提示词，匹配停在 A 的节点中间 -> 分裂
  [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011] len=12 lock=2 hit=1 prio=0
    [1, 2, 3, 4, 5] len=5 lock=1 hit=1 prio=0
  B 命中 12，B row 前 12 个槽位就是树里的：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  available 236 + evictable 0 + protected 17 + uncached 3 = 256 == total 256
```

场景 3：

```
负载            输入 token       8K      16K      32K      64K     128K     256K       无限大
fewshot      1,488,090    73.7%    95.1%    94.9%    94.9%    94.9%    94.9%     94.9%
chat           629,494    28.9%    39.9%    54.4%    82.8%    90.0%    90.0%     90.0%
selfcons       729,656    99.8%    99.8%    99.8%    99.8%    99.8%    99.8%     99.8%
unique         503,637     0.0%     0.0%     0.0%     0.0%     0.0%     0.0%      0.0%
```

场景 4（shift 和 chat 两组，节选）：

```
[shift] 热点迁移，池子 4.6K（只放得下 4 个模板多一点），每批 2 条，2400 条
  策略             命中率     P90     P99    超 SLO    驱逐 token    迁移后命中   旧模板残留
  lru           93.6%      57     944     0.0%     212,031      92.7%         0
  lfu           62.0%     948     959     0.0%     851,856      45.3%     1,800
  slru          95.3%      56      60     0.0%     178,731      95.3%         0

[chat] 多轮对话 80 用户 × 10 轮，池子 90K，每批 16 条，800 条
  策略             命中率     P90     P99    超 SLO    驱逐 token
  lru           73.8%    1654    2584    16.1%     344,116
  lfu           65.0%    1544    2256    22.6%     430,497
  tlru          72.2%    1037    2498    10.4%     357,567
```

场景 5（节选）：

```
      池子     顺序    队列      命中率  prefill token    驱逐 token
      8K   fcfs   128     0.0%        991,606   1,057,345
      8K    lpm   800    14.8%        845,206     899,490
      8K    lpm   128    94.8%         52,006      93,441
     20K  dedup   128    95.8%         41,206     112,744
```

场景 8（第 2、4 种）：

```
[2] 忘记释放（release 漏了 free [cache_protected_len, 已分配)）
  available 88 + evictable 36 + protected 0 + uncached 0 = 124 != total 128
[4] 重复解锁（dec_lock_ref 调了两次）
  available 88 + evictable 36 + protected 0 + uncached 4 = 128 == total 128
  B 还在跑，evict(全部) 删掉 36 个 token（可驱逐叶子 1 个），B 正在用的前缀槽位 [4, 5, 6, 7]… 已经回到空闲表
```

数字是模拟结果，用来看趋势和量级。真实调度器有 chunked prefill 预算、retract、HiCache 主机层，命中率和尾延迟要拿自己的负载在 `/metrics` 上复核（`sglang:cache_hit_rate`、`sglang:token_usage`）。
