# SGL-19 集群与 RL rollout

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/19-cluster-rl.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-19｜实战｜集群与 RL rollout：路由、权重同步与长尾。

验证基线：sglang 主干 `8fabf5e`（网关在仓库根下的 `sgl-model-gateway/`）。脚本只用标准库，CPU 上能跑，不依赖 GPU、torch 和 sglang 本体。

多副本前面加一层网关以后，有两类问题前面章节没碰过。一类是在线服务的：请求发给哪台，前缀缓存才不白建（sim_router.py）。另一类是 RL 训练的：rollout 被最长的那几条采样拖住，每一步之间还要把新权重灌进推理引擎（rollout_sim.py）。

| 文件 | 内容 |
|---|---|
| sim_router.py | 4 台 Qwen3-8B 副本 + 一个网关的事件模拟。负载是 8 个应用的多轮会话（system prompt 1000-3000 token，Zipf 热度，每会话 2-10 轮）。副本带按块近似的 RadixCache、prefill 优先调度和 KV 准入；网关按 8fabf5e 的 `src/policies/*.rs` 实现 round_robin、power_of_two、consistent_hashing、prefix_hash、cache_aware 五种策略。`--sweep` 扫 cache_aware 的阈值和树大小，`--remove` 跑到一半摘一台，`--ring` 算哈希环扩缩容搬走多少 key |
| rollout_sim.py | 一台推理引擎上的 GRPO rollout：32 个 prompt × 8 条采样，输出长度对数正态、组内共享难度。比较同步 rollout、APRIL 式超发 + 截停 + 回收、超发但不回收三种做法的每步时间、有效吞吐、训练批里旧权重 token 的比例。`--sync-cost` 用 checkpoint-engine 的实测反推有效带宽，估三种规模的模型按张量、按桶、每张量一次调用、走磁盘四种权重同步路径的耗时 |
| test_cluster.py | 39 个用例 |

## 和源码的对应

网关路径相对 `sgl-model-gateway/src/`，sglang 路径相对 `python/sglang/srt/`。

| 本目录 | 源码 |
|---|---|
| `RoundRobin` | `policies/round_robin.rs:31` |
| `PowerOfTwo` | `policies/power_of_two.rs:81-131`：有轮询来的 token 负载就比 token，否则比在途数，`<=` 取第一台；负载每 5 s 从 `/v1/loads?include=core` 拉一次（`core/worker_manager.rs:240`、`:362-401`，只在配了 power_of_two 时才拉；`main.rs:775`） |
| `ConsistentHashing` | `policies/consistent_hashing.rs:105-160`：X-SMG-Target-Worker → X-SMG-Routing-Key → authorization / x-forwarded-for / cookie → 随机；环在 `core/worker_registry.rs:31`、`:44`，150 个虚节点，blake3（这里换成标准库的 blake2b）；Python 启动器 `router_args.py:20-34` 的 `--policy` 能选，Rust 二进制 `main.rs:150` 的列表里没有 |
| `PrefixHash` | `policies/prefix_hash.rs:107-197`，`load_ok` 在 `:117`；HTTP 路径传 `tokens: None`，策略返回 NoTokens（`routers/http/router.rs:134-180`、`prefix_hash.rs:211-214`），所以只在 gRPC 路径生效 |
| `CacheAware` | `policies/cache_aware.rs:387-511`；失衡判定 `:412-413`，匹配率阈值 `:444`，匹配到的台已不在时删租户、退回第一台健康的 `:489-511`；策略拿到的 worker 已按 `is_available()` 过滤（`router.rs:151`、`core/worker.rs:229`）；文件头注释说低匹配时去「树最小」的台，代码 `:444` 之后走的是最少在途数 |
| `ApproxTree` | `policies/tree.rs`：insert `:362`、prefix_match_with_counts `:531`（last_tenant 优先 `:571-598`）、evict_tenant_by_size `:718`、remove_tenant `:823`；按字符计大小 |
| `Worker.load` | `core/worker.rs:784` 在途请求数，`:1147` WorkerLoadGuard 在请求结束时减一 |
| `Worker.drain` | 重试参数 `main.rs` 的 `--retry-max-retries 5` 等，熔断 `--cb-*` |
| CLI 默认 vs 库默认 | `main.rs`：cache_threshold 0.3、balance_abs 64、balance_rel 1.5、eviction_interval 120、max_tree_size 2^26（Python 启动器 `router_args.py:59-63` 同，只有 eviction 是 60）；`policies/mod.rs:106` 的 `CacheAwareConfig::default()` 是 0.5 / 32 / 1.1 / 30 / 10000 |
| `Engine.step` 的 retract | `managers/schedule_batch.py` 的 `retract_decode`，调度器入口 `managers/scheduler.py:4558` |
| `Engine.abort_all` | `/pause_generation`（`entrypoints/http_server.py:1831`，`managers/scheduler.py:5902`），mode 有 abort / retract / in_place，见 `docs/docs/advanced_features/sglang_for_rl.mdx:640-655` |
| `segs` 权重版本 | `meta_info.weight_versions`，`sglang_for_rl.mdx:600-610`；版本号由 `/update_weight_version` 或各更新接口的 `weight_version` 参数设 |
| 每步 flush | `flush_cache_after_weight_update`，有在跑请求时返回失败 |
| `sync_cost` 的按桶 | `model_executor/model_runner_components/weight_updater.py:275`、`:354` 的 flattened_bucket，`weight_sync/tensor_bucket.py:19` FlattenedTensorBucket；接口字段见 `managers/io_struct.py:1970-2187` |
| `CKPT_ENGINE` | MoonshotAI/checkpoint-engine README 的 Benchmark 表 |

## 运行

```bash
cd zero-to-sglang/19-cluster-rl
python3 sim_router.py                 # 五种策略对比，约 17 s
python3 sim_router.py --sweep         # cache_aware 调参，约 33 s
python3 sim_router.py --remove        # 第 90 s 摘掉一台
python3 sim_router.py --ring          # 哈希环
python3 sim_router.py --policy cache_aware --lib-defaults --json
python3 rollout_sim.py                # rollout 长尾 + 超发对比，约 1 s
python3 rollout_sim.py --sync-cost    # 权重同步
python3 rollout_sim.py --sync-cost --lat-call 5 --bucket-gb 2
python3 -m unittest test_cluster -v   # 约 4 s
```

## 期望输出

```text
$ python3 sim_router.py
负载：1469 个会话、8813 条请求，prompt 中位数 3,548 token、最长 8,860，最热的应用占 36% 的会话；4 台 Qwen3-8B / H100，每台 KV 池 366,669 token
五种策略，同一份负载（秒）
  策略                      命中率  prefill 千 token  TTFT p50  TTFT p99  E2E p99  忙碌 max/mean  会话跨台
  round_robin                  58%            13,913      0.06      2.04     36.6           1.01       96%
  power_of_two                 58%            13,743      0.09      2.80     34.9           1.01       97%
  consistent_hashing           80%             6,431      0.03      2.95     34.5           1.11        0%
  prefix_hash                  78%             7,064      0.03      0.22     26.5           1.01       28%
  cache_aware（CLI 默认）      81%             6,119      0.03      0.21     25.9           1.05       22%
  cache_aware（库默认）        82%             5,869      0.03      0.19     23.9           1.02       22%
  cache_aware 分支：命中 93%、低匹配走最少在途 0%、失衡走最少在途 7%；网关树估的匹配率 81%，副本上实际命中 81%

$ python3 sim_router.py --remove
第 90 s 摘掉 worker1，手上的请求由网关重试到其余三台
  策略                  摘前命中  摘后命中  摘后 TTFT p99  重试请求  会话跨台
  round_robin                70%       65%           0.20        38       95%
  consistent_hashing         84%       87%           0.11        49        9%
  cache_aware                85%       85%           0.23        55       23%

$ python3 sim_router.py --ring
  变化    环（150 虚节点）  取模 hash % N   理想
  4->5             20.6%          79.7%   20.0%
  4->3             24.2%          74.9%   25.0%
  8->9             10.7%          88.7%   11.1%

$ python3 rollout_sim.py（节选）
  一半采样 25.7 s 写完，90% 在 88.1 s，最后一条在 217.5 s；最后 10% 的采样占了 59% 的时间
  方式                每步 rollout  有效 token/s  平均 batch  训练批平均长度  含旧权重的采样  旧权重 token  最大滞后  丢弃 token
  同步                       211.1         2,507          47           2,067              0%            0%         0           0
  超发 50%                   138.6         3,519         116           1,905             30%           49%         3           0
  超发 50% 不回收            121.9         2,638         142           1,256              0%            0%         0   1,138,116

$ python3 rollout_sim.py --sync-cost（节选）
  模型              卡   每卡 GB   张量数  桶数  逐张量一次调用  按桶  每张量一次调用  走磁盘
  Qwen3-8B            1      16.4      327    16           2.3    2.6             8.9      13
  Qwen3-235B-A22B     8      58.8   37,039    55           8.2    8.1           748.9     353
  DeepSeek-V3        16      41.9   90,064    40          13.0   11.1          1814.2     504
```

## 读数时留意

- 模拟器的单步耗时来自 `../03-bench/sim_serve.py` 的 Roofline 模型，绝对值和真机有出入，策略之间的相对关系和量级可信。
- prefix_hash 这一行假设走 gRPC。HTTP 路由传给策略的 tokens 是 None（router.rs:176），策略直接返回 None，路由回 503 no_available_workers（router.rs:287-291），每条请求都会失败。
- HTTP 路由只给 cache_aware 和 manual 挂 in-flight 计数（router.rs:300-302），power_of_two 在 HTTP 下的回退比较拿到的都是 0；模拟器里 power_of_two 用的是每 5 s 轮询一次的 token 负载。
- 摘台实验的到达率是每秒 4.8 个会话：每秒 8 个时剩下三台接不住，几种策略一起塌，看不出差别。
- rollout 的「有效吞吐」只算 rollout 时间。超发换来的吞吐要用训练端的 off-policy 修正（重要性采样截断等）来付，这部分不在这里算。
