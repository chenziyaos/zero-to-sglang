# SGL-14 HiCache

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/14-hicache.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-14｜原理｜HiCache：显存、主机内存、远端存储三层 KV 缓存。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| hicache_sim.py | 四个模型配置下每 token 重算和搬运的代价、各条链路的回本前缀长度、逐层重叠的流水线时延；按页模拟 GPU / 主机 / L3 三层的 TierSim（write_through、write_through_selective、write_back，主机池只能淘汰离开显存的页，L3 预取受阈值和半个主机池限制）；bench_multiturn 形状的多轮负载；三种预取停止策略在不同带宽和排队时间下的 TTFT；页哈希链、namespace 种子和文件后端的键名 |
| resolve_sim.py | `--enable-hierarchical-cache` 相关参数的解析链移植：ratio 默认值、fraction 什么时候被关掉、布局和 IO 后端互相改写、mooncake 不收 layer_first、buffer_only 的三条规矩、decode 实例和 DCP 的限制；主机池大小的完整算法（10 GiB 预留、按 rank 平分、auto-sizing、多补一页、两条告警和一条报错）；decode 实例的 ratio |
| test_hicache.py | 43 个边界用例 |

## 和源码的对应

路径相对 `python/sglang/`。

| 本目录 | sglang |
|---|---|
| `TierSim` 的备份门槛 | `srt/mem_cache/unified_radix_cache.py:570` write_through 1 次、selective 2 次；`srt/mem_cache/unified_cache/unified_tree_core.py:1104` _inc_hit_count_and_check，write_back 不在插入时备份 |
| `TierSim._evict_dev` | `unified_tree_core.py:1788` evict_device_leaf、`:1825` 没有主机副本时整棵子树丢弃；`unified_radix_cache.py:1044` write_back 淘汰前同步备份 |
| `TierSim._host_alloc` | `unified_tree_core.py:2366` _is_host_leaf 要求页已离开显存；`unified_radix_cache.py:1798` 主机池不够时先 evict_host，再不够就跳过备份 |
| `TierSim._backup` 写 L3 | `unified_radix_cache.py:1864` _finish_write_through_ack 里调 write_backup_storage |
| `TierSim.request` 的 L3 段 | `unified_radix_cache.py:2142` prefetch_from_storage 的阈值和页对齐；`srt/managers/cache_controller.py:628` prefetch_capacity_limit |
| `prefetch_timeout_s` / `prefetch_plan` | `srt/mem_cache/unified_cache/storage_attachment.py:290` 每页预算、`unified_radix_cache.py:2320` 计时起点、`:2356` _can_terminate_prefetch、`:2452` check_prefetch_progress；`srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:404` 默认 threshold 256、base 1、per_ki 0.25；`srt/mem_cache/hicache_storage.py:24` 每批 128 页 |
| `namespace_seed` / `page_hashes` | `srt/mem_cache/utils.py:125` storage_namespace_seed、`:142` get_storage_hash_str；sgl-kernel 的 hash_binding.cpp（SHA256(上一页摘要 + uint32 小端 token)） |
| `file_key` | `srt/mem_cache/hicache_storage.py:728` HiCacheFile 的键后缀 |
| `resolve` | `srt/arg_groups/pipeline.py:77` / `:272`；`srt/arg_groups/hicache_hook.py:18` handle_hicache、`:100` ratio 默认值、`:129` DCP、`:171` 布局和 IO、`:197` 存储和布局、`:228` unified memory、`:247` buffer_only |
| `decode_ratio` | `srt/mem_cache/kv_cache_builder.py:164` BACKUP_ONLY_HICACHE_RATIO、`:244` decode 实例加载模型后定 ratio |
| `host_budget_bytes` / `auto_ratio` | `srt/mem_cache/pool_host/base.py:28` 10 GiB 预留、`:84` 按 rank 平分；`srt/mem_cache/hicache_auto_size.py` 0.95 余量、全局取 MIN |
| `host_pool_tokens` / `check_alloc` | `srt/mem_cache/pool_host/base.py:171` HostKVCache.__init__、`:237` _allocate_host_buffer |

## 运行

```bash
cd zero-to-sglang/14-hicache
python3 hicache_sim.py                         # 重算和搬运每 token 的代价、回本前缀、32K 前缀的 prefill 时延
python3 hicache_sim.py --scenario multiturn    # 80 会话 × 10 轮，六种缓存配置的命中率（约 1 秒）
python3 hicache_sim.py --scenario prefetch     # 三种预取策略 × 四种带宽 × 三种排队时间
python3 hicache_sim.py --scenario hash         # 页哈希链、cache_salt 隔离、文件后端键名
python3 resolve_sim.py                         # 17 条命令行的解析结果和告警
python3 resolve_sim.py --scenario size         # 8 卡机器上主机池怎么算出来（2048 GiB 内存）
python3 resolve_sim.py --scenario size --host-gib 512
python3 -m unittest test_hicache -v            # 约 1 秒
```

## 期望输出

```text
$ python3 hicache_sim.py（节选）
  模型 × 卡                        cell B      重算  L2 PCIe Gen5 x16 ... L3 3FS 40GiB/s÷8  L3 本地NVMe÷8
  Llama-3.1-8B 1×H100 SXM       131072    34.5               2.6 ...             24.4          163.8
  DeepSeek-V3 8×H200             70272    11.4               1.4 ...             13.1           87.8
从哪条链路搬回来开始比重算快（前缀 token 数）
  Llama-3.1-8B          L2 0   RDMA 0   3FS 0      本地NVMe 248064
  DeepSeek-V3           L2 0   RDMA 0   3FS 7424   本地NVMe 155392
一条 32K 前缀 + 2048 新 token 的请求（ms）
  模型                        全部重算     L2 串行     L2 逐层重叠    3FS 串行    3FS 重叠
  Llama-3.1-8B              1769       226         142       940       804
  DeepSeek-V3                938       135          90       518       430

$ python3 hicache_sim.py --scenario multiturn（节选）
  配置                       显存命中     主机命中    L3命中      重算   D2H GB     丢弃页  prefill s
  只有 GPU                  12.7%     0.0%    0.0%   87.3%        0  116440      318.0
  GPU+主机 ratio1           12.7%     0.0%    0.0%   87.3%     1031       0      318.0
  GPU+主机 wt               12.7%    26.0%    0.0%   61.3%      724       0      235.1
  GPU+主机 selective        12.7%    34.1%    0.0%   53.2%       90   60188      214.4
  GPU+主机 write_back       12.7%    40.0%    0.0%   47.3%      504       0      187.6
  GPU+主机+L3 wt            12.7%    26.0%   43.1%   18.2%      215       0       89.7

$ python3 hicache_sim.py --scenario prefetch（节选）
默认 timeout = 1.0 + 512 × 64/1024 × 0.25 = 9.0 s
全部重算：938 ms
  每 rank 5.37 GB/s，整段拉完 429 ms
       200ms    best_effort       200        8192      1027
       200ms        timeout       429       32768       518
  每 rank 0.1 GB/s，整段拉完 23027 ms
         0ms        timeout      9000        8192      9827
         0ms  wait_complete     23027       32768     23116
建议：本例重算 11.4 us/token，对应 prefetch_timeout_per_ki_token ≈ 0.0117

$ python3 resolve_sim.py（节选）
page_first + direct               mem_layout=page_first_direct  io_backend=direct
mooncake + layer_first            mem_layout=page_first  io_backend=kernel
buffer_only + write_back          ratio=1.2  write_policy=write_through
DCP + MHA 模型                      启动失败：HiCache with --dcp-size > 1 is only supported for MLA models

$ python3 resolve_sim.py --scenario size --host-gib 512（节选）
  默认（ratio 2，fraction 0.8）：每 rank 预算 62.8 × 0.8 = 50.2 GiB，ratio 2.000 -> 0.999
    WARN HiCache kv host pool (1250176 tokens) is smaller than the device pool (1251703 tokens);L2 cache effectiveness is reduced.
  --hicache-size 120：... Not enough host memory available. Requesting 120.00 GB but only have 67.38 GB free.
```

## 校准

1. 链路带宽：L2 先用 `nvidia-smi topo -m` 确认每张卡挂在哪个 CPU 和 PCIe 交换芯片下，再用 nvbandwidth 或 pinned 内存上的 `cudaMemcpyAsync` 测 H2D / D2H。8 张卡同时拷时每卡带宽常常只有单卡的一半到七成，按同时拷的数值改 `LINKS`。L3 用存储自带的压测工具在 8 个进程并发下测读带宽，除以 8 得到每 rank 的数。
2. 重算代价：在不开 HiCache 的服务上单独发一条 32K 前缀的请求测 TTFT，反推实际 MFU，改 `MFU`。重算的每 token 代价决定了 L3 什么时候划算、预取超时该给多少。
3. 主机池大小：启动日志里的 `HiCache auto-sizing: ratio 2.000 -> x` 和 `smaller than the device pool` 告警应和 `resolve_sim.py --scenario size --host-gib <free -g 的 available>` 一致；容器里以 cgroup 的 memory.max 减去当前用量为准。
4. 命中来源：开 `--enable-metrics` 后看 `sglang:cached_tokens_total{cache_source=device|host|storage}` 三条曲线的占比，和 `--scenario multiturn` 的显存 / 主机 / L3 命中对照；`sglang:hicache_host_used_tokens` 长期贴着 `sglang:hicache_host_total_tokens` 说明主机池偏小。
5. 预取：`sglang:storage_prefetch_unfulfilled_tokens_total` 按 reason 拆开看，`deadline` 多说明超时预算不够或存储太慢，`host_capacity` 多说明主机池的一半装不下预取；`sglang:prefetch_bandwidth` 直方图就是实测的 L3 读带宽，可以直接回填 `LINKS`。
6. 负载形状：`benchmark/hicache/bench_multiturn.py --num-clients 80 --num-rounds 10 --request-length 2048 --enable-round-barrier` 和 `--scenario multiturn` 的负载形状一致。对只开 radix、开 L2、开 L2+L3 三种服务各跑一次，比较逐轮 TTFT。
