# SGL-07 内存池与页表

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/07-memory-pool.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-07｜原理｜内存池与页表：req_to_token、token_to_kv 与 page size。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑。KV 池容量复用 `../02-ledger/sim_ledger.py`，模型规格复用 `../common/specs.py`。

| 文件 | 作用 |
|---|---|
| `sim_mempool.py` | 两级映射的逐行复刻：`ReqToTokenPool`（请求 → 行）、`TokenAllocator` / `PagedAllocator`（行里的槽位 → 页），alloc_extend 的三段式、decode 只在跨页时取新页、按页归还插回表头、前缀树按整页截断、need_sort 合并排序。另带 page size 扫描、启动参数推算、长时间抖动后的碎片统计和四类报错复现 |

## 和源码的对应

| 模拟里的函数 | sglang 源码 | 说明 |
|---|---|---|
| `ReqToTokenPool` | `mem_cache/memory_pool.py:376` | int32 表，形状 (size+1, context_len+4)，第 0 行留给 CUDA graph 的 padding |
| `alloc_rows` / `free_rows` | `memory_pool.py:437`、`:454` | 从 free_slots 尾部弹出、还回尾部；generation 单调递增 |
| `TokenAllocator` | `mem_cache/allocator/token.py:28` | page_size=1，空闲表 arange(1, size+1)，从头取 |
| `PagedAllocator.alloc_extend` | `mem_cache/allocator/paged.py:204`、`:41` alloc_extend_naive | 三段：补满上一页的空位 → 整页 → 新尾页；先用上界 extend//ps + bs + 1 粗估，不够再算精确页数 |
| `PagedAllocator.alloc_decode` | `paged.py:279`、`:117` | 只有 seq_len % ps == 1 的请求取新页 |
| `PagedAllocator.free_slots` | `paged.py:328`、`:375` _release_page_ids | unique(idx // ps) 后插回空闲表头部；need_sort 时先进 staged，下次不够才合并排序 |
| `num_new_pages_*` | `utils/common.py:4840` get_num_new_pages | extend：ceil(seq/ps) − ceil(prefix/ps) |
| need_sort | `kv_cache_configurator.py:2805` | 只在 PD 分离的 prefill / decode 侧打开 |
| `Pool.extend` | `mem_cache/allocation.py:374` alloc_for_extend、`:175`、`:327` | 先拿 req_to_token 行，再分配 KV 槽位，最后写表 |
| `Pool.decode` | `allocation.py:625` alloc_for_decode_default、`:572` | decode 预估按 bs × ps 保守算 |
| `Pool.release` | `mem_cache/common.py:418` release_kv_cache、`:465` _release_overallocated | checkpoint 之后只还 [cache_protected_len, owned) |
| `PageStore.insert` | `mem_cache/radix_cache.py:147` RadixKey.page_aligned、`:510` | 进树的 key 向下取整到页，尾部零头不进树 |
| `page_table` | `layers/attention/flashattention_backend.py:1317` | 页表 = req_to_token[:, ::ps] // ps |
| sizing | `pool_configurator.py:637`、`:94`、`kv_cache_configurator.py:3119`、`memory_pool.py:2567` | token 数按页向下取整；max_running 推算；KV buffer 每层多一页 |

## 运行

```bash
# 1. 两条请求走一遍：prefill → decode → 释放 → 新请求复用前缀；默认跑 ps=1 和 ps=4
python3 sim_mempool.py

# 2. 换个 page size 看同一流程
python3 sim_mempool.py --scenario walk --page-size 16

# 3. 三种负载 × 五档 page size：尾页浪费、树丢尾、前缀命中、页表项、可驻留条数
python3 sim_mempool.py --scenario pagesize

# 4. 六组模型 / GPU 的池子大小、max_running、req_to_token 开销和启动日志
python3 sim_mempool.py --scenario sizing

# 5. 长时间进出之后槽位的连续性（影响 PD 分离的传输段数）
python3 sim_mempool.py --scenario churn

# 6. 四类内存报错复现
python3 sim_mempool.py --scenario errors
```

## 期望输出

场景 1（ps=4 部分）：

```
[1] prefill A、B（一批）
  out_cache_loc        4 5 6 7 8 9 10 11 12 13 16 17 18 19 20 21 22
  A 占 req_to_token 第 3 行（generation 1）：4 5 6 7 8 9 10 11 12 13
      页表 row[::ps]//ps = [1, 2, 3]
  B 占 req_to_token 第 4 行（generation 1）：16 17 18 19 20 21 22
      页表 row[::ps]//ps = [4, 5]

[2] decode 3 步
  step 1: out_cache_loc = [14, 23]   seq_len 变成 11/8，取新页 0 个
  step 2: out_cache_loc = [15, 24]   seq_len 变成 12/9，取新页 1 个
  step 3: out_cache_loc = [28, 25]   seq_len 变成 13/10，取新页 1 个

[3] A 结束，release_kv_cache
  A 共 13 token，进前缀树 12 个（向下取整到页），还给分配器的槽位：[28]
  按页归还：[7]，插回空闲表头部

[4] C 进来：和 A 共享 8 token 的系统提示词
  C 第 3 行：4 5 6 7 8 9 10 11 28 29 30
```

场景 3（short 负载部分）：

```
[short] 输入中位数 20，输出中位数 59，共享系统提示词 40 token
    ps     尾页浪费    树丢尾 tok     占比     前缀命中     页表项     新页频率      可驻留条数
     1     0.0%        0.0  0.00%   40/40      61.0       每步       6010
    16    10.8%        7.5  7.54%   32/40       4.3     1/16       5359
    32    19.9%       15.5 15.59%   32/40       2.4     1/32       4813
    64    33.6%       32.6 32.90%    0/40       1.4     1/64       3989
   128    55.6%       66.5 66.99%    0/40       1.1    1/128       2669
```

场景 5：

```
  ps need_sort     每条段数 p50    p90   每 100 token 段数       合并排序次数
   1     False           33     95             13.9            0
   1      True            2      5              0.8          261
  16     False           13     25              4.6            0
  16      True            2      4              0.8          272
```

数字是模拟结果，用来看趋势和量级。真实部署里前缀树还会驱逐、请求长度分布也不同，结论要拿自己的负载复核。
