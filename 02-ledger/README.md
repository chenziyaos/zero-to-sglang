# SGL-02 推理账本

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/02-ledger.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-02｜上手｜推理账本：显存、KV 与 Roofline。

验证基线：sglang 主干 `8fabf5e`。两个脚本都只用标准库，CPU 上能跑。模型结构和 GPU 规格从 `../common/specs.py` 读，启发式（chunk 档位、decode max_bs）复用 `../01-first-server/estimate_startup.py`。

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `sim_ledger.py` | CPU | 按主干切池逻辑算每卡账本：权重、预留、KV 池、token 数、请求上限，覆盖 GQA、MLA、SWA 和 TP 下 KV 头重复 |
| `sim_roofline.py` | CPU | 算 decode 一步、prefill 一个 chunk 的 FLOP、字节和时间下限，判断落在拐点哪一侧；读 one_batch 结果算 MBU、MFU |
| `sample_one_batch.jsonl` | — | 按 one_batch 输出格式构造的夹具，数值是量级推算，不是实机录制 |

## 推荐顺序

先算账本，再看 Roofline，最后上卡实测。账本告诉你能装多少请求，Roofline 告诉你装进去以后一步最快多快，实测告诉你离下限差多少。

```bash
# 1. 十个典型场景的对照表
python3 sim_ledger.py --compare

# 2. 单个场景的完整账本
python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
python3 sim_ledger.py --model gpt-oss-120b --gpu h100-sxm
python3 sim_ledger.py --model gpt-oss-120b --gpu h100-sxm --swa-ratio 0.25 --mem-fraction-static 0.9 --max-running-requests 256

# 3. Roofline：decode 随 batch、prefill 随 chunk 的变化
python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm
python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm --ctx 1024 --kv-dtype fp8
python3 sim_roofline.py --model qwen3-235b-a22b --gpu h100-sxm --tp 8 --weight-dtype fp8
python3 sim_roofline.py --model deepseek-v3 --gpu h200 --tp 8 --ctx 2048

# 4. GPU 上实测，再回来算利用率
python -m sglang.benchmark.one_batch --model-path Qwen/Qwen3-8B \
    --batch-size 1 8 32 64 --input-len 1024 --output-len 32 --result-filename result.jsonl
python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm --jsonl result.jsonl
```

只想看利用率怎么算、手边没卡：

```bash
python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm --jsonl sample_one_batch.jsonl
```

## 期望输出

`sim_ledger.py --compare`：

```
场景            模型                       卡     权重   KV      f    权重GiB   KV池GiB  KB/tok     tokens    满上下文条数
稠密 GQA 基线     Qwen3-8B            1xH100   bf16 bf16  0.830    15.26    50.35   144.0    366,669       9.0
KV 换 FP8      Qwen3-8B            1xH100   bf16  fp8  0.830    15.26    50.35    72.0    733,338      17.9
70B 四卡        Llama-3.1-70B       4xH100   bf16 bf16  0.819    32.88    31.86    80.0    417,650       3.2
70B 八卡        Llama-3.1-70B       8xH100   bf16 bf16  0.812    16.44    47.75    40.0  1,251,703       9.5
MoE BF16 八卡   Qwen3-235B-A22B     8xH100   bf16 bf16  0.812    54.74     9.45    47.0    210,787       5.1
MoE FP8 八卡    Qwen3-235B-A22B     8xH100    fp8 bf16  0.812    27.37    36.82    47.0    821,398      20.1
MoE FP8 四卡    Qwen3-235B-A22B     4xH100    fp8 bf16  0.819    54.74    10.00    47.0    223,132       5.4
SWA 默认比例      gpt-oss-120b        1xH100  mxfp4 bf16  0.830    60.71     4.90    64.8     79,219       0.6
SWA 比例 0.25   gpt-oss-120b        1xH100  mxfp4 bf16  0.830    60.71     4.90    45.0    114,075       0.9
MLA 八卡 H200   DeepSeek-V3         8xH200    fp8 bf16  0.886    78.11    45.75    68.6    699,040       4.3
```

几行值得停下来看：

- 70B 从四卡到八卡，卡数翻倍，token 数是 3 倍。权重减半让 KV 池从 31.86 涨到 47.75 GiB，同时每卡 KV 头从 2 降到 1，每 token 字节也减半。
- Qwen3-235B 只有 4 个 KV 头，TP8 时每个头在两张卡上重复存，每 token 47 KB 和 TP4 一样，加卡不省 KV。
- 同一个 235B，BF16 权重时 KV 池只剩 9.45 GiB，换 FP8 权重变成 36.82 GiB，token 数接近 4 倍。
- gpt-oss-120b 单卡 H100 的 KV 池不到 5 GiB，一条满上下文请求都装不下。

`sim_ledger.py --model gpt-oss-120b --gpu h100-sxm`：

```
gpt-oss-120b on 1 x H100 SXM  weight=mxfp4 kv=bf16
  mem_fraction_static    0.83
[每卡账本，GiB]
  可见显存                 79.65
  CUDA 上下文等             0.60
  权重                     60.71
  激活与 Graph 预留        13.44   其中 req_to_token 表 1.00
  KV 池                     4.90
[KV 池]
  每卡 KV 头 8（共 8）
  全注意力层 18 层 36.0 KB/token，滑窗层 18 层 36.0 KB/token，ratio 0.8
  每 token 计价             64.8 KB
  max_total_num_tokens      79,219
  swa_layer_tokens          63,375   按平均 4096 token 驻留时实际要用约 21,286（34%）
  max_running_requests       2,048
  满上下文 131072 的请求能装 0.6 条，单条长请求会被截断或拒绝
  平均 4096 token 的请求能同时驻留 19 条
```

SWA 池默认按全注意力池的 0.8 倍开，可窗口只有 128，实际只用到三分之一。把 `--swa-full-tokens-ratio` 降到 0.25、`--mem-fraction-static` 提到 0.9、`--max-running-requests` 压到 256（req_to_token 表从 1.00 GiB 降到 0.13 GiB）之后：

```
  每 token 计价             45.0 KB
  max_total_num_tokens     243,011
  swa_layer_tokens          60,752   按平均 4096 token 驻留时实际要用约 31,606（52%）
  max_running_requests         256
  满上下文 131072 的请求能装 1.9 条
  平均 4096 token 的请求能同时驻留 59 条
```

提高 `--mem-fraction-static` 是拿激活余量换 KV，前提是 CUDA Graph 捕获和最大 chunk 的 prefill 都还能过，上卡后必须跑一遍最长输入确认不 OOM。

`sim_roofline.py --model qwen3-8b --gpu h100-sxm`：

```
Qwen3-8B on 1 x H100 SXM  weight=bf16 kv=bf16  拐点 295 FLOP/B
  KV 池 366,669 token，上下文 4096 时最多驻留 89 个请求
[decode，每请求上下文 4096]
    batch   权重GB    KV GB   强度   下限ms   tok/s      瓶颈
        1    15.14     0.60      1     4.70      213      带宽
        8    15.14     4.83      7     5.96     1342      带宽
       32    15.14    19.33     16    10.29     3111      带宽
       64    15.14    38.65     21    16.06     3986      带宽
      128    15.14    77.31     24    27.60     4638      带宽  超出 KV 池
      256    15.14   154.62     26    50.67     5052      带宽  超出 KV 池
      512    15.14   309.24     28    96.83     5288      带宽  超出 KV 池
[prefill，单个 chunk]
    tokens   强度   下限ms   tok/s     瓶颈
        64     64     4.52     14156     带宽
       256    257     4.53     56519     带宽
      1024   1034    15.98     64063     算力
      2048   2088    32.59     62834     算力
      4096   4253    67.69     60512     算力
      8192   8797   145.38     56347     算力
     16384  18639   330.79     49530     算力
  prefill 从 295 token 起转为算力瓶颈
```

decode 在任何 batch 下都是带宽瓶颈：权重读一次可以被 batch 摊薄，KV 却是每个请求各读各的，batch 越大 KV 占比越高，强度卡在 30 以内上不去。上下文缩到 1024、KV 换 FP8 后（`--ctx 1024 --kv-dtype fp8`），同一张卡 batch 512 的下限是 16.06 ms，吞吐 31,887 tok/s，是上表 batch 64 的 8 倍。

prefill 的 tok/s 在 1024 之后反而下降，是因为注意力 FLOP 随 chunk 长度平方增长。

MoE 的 decode 多一列命中专家数（`--model qwen3-235b-a22b --tp 8 --weight-dtype fp8`）：

```
    batch   权重GB    KV GB   强度   下限ms   tok/s      瓶颈  命中专家/层
        1     2.70     0.20      2     0.86     1157      带宽       8.0
        8    12.37     1.58      4     4.16     1921      带宽      51.6
       32    25.71     6.31      7     9.56     3348      带宽     111.8
       64    28.85    12.62     11    12.38     5170      带宽     125.9
      128    29.30    25.23     16    16.28     7863      带宽     128.0
```

batch 1 只读 8 个专家，batch 8 就要读 51.6 个，单步时间翻了将近 5 倍。MoE 模型小 batch 的 decode 延迟对 batch 很敏感，这是稠密模型没有的现象。

`sim_roofline.py --jsonl sample_one_batch.jsonl`：

```
 batch     in   out  decode ms    下限 ms    MBU  prefill s    MFU
     1   1024    32       6.20     4.56    74%     0.0291    55%
     8   1024    32       6.60     4.88    74%     0.2153    59%
    32   1024    32       8.10     5.98    74%     0.8604    59%
    64   1024    32      10.20     7.45    73%     1.7215    59%
```

夹具数值是推算的量级，不代表实测。H100 上稠密模型 decode MBU 在 60%～80%、prefill MFU 在 40%～60%，属于正常区间。明显偏低时依次查：是否没开 CUDA Graph（小 batch decode 被 launch 开销拖住）、注意力后端是否走了慢路径、GPU 是否降频（`nvidia-smi -q -d CLOCK`）。

## 和启动日志对账

`sim_ledger.py` 的 `max_total_num_tokens` 对应日志里的 `KV Cache is allocated. #tokens:` 和 `max_total_num_tokens=` 两行。带滑窗层的模型多一行：

```
Use sliding window memory pool. full_layer_tokens=…, swa_layer_tokens=…
```

`full_layer_tokens` 对 `max_total_num_tokens`，`swa_layer_tokens` 对同名一行。偏差在 2% 以内正常；超过 10% 先看 SGL-01 README 里列的几种情况。

## 模型的边界

- 只算 HBM 读写和矩阵乘的下限，不算 kernel launch、TP 通信、采样、调度。实测一定比下限慢，比值就是利用率。
- MoE 命中专家数按路由均匀独立算。真实路由有偏，命中数比公式少，实际权重读取更少。
- MXFP4 在 Hopper 上按 BF16 算力计，FP8 权重按 FP8 Tensor Core 计（W8A8）。
- 不覆盖投机解码的草稿层、DP Attention、PP、多模态、HiCache 和 `--max-total-tokens` 上限。
