# SGL-25 Embedding 与 Rerank 服务

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/25-embedding-rerank.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-25｜专题｜Embedding 与 Rerank 服务：只有 prefill 的引擎怎么配。

验证基线：sglang 主干 `8fabf5e`，vLLM 对照用主干 `0d6f92a`。`sim_pool.py`、`pool_client.py` 和测试只用标准库，CPU 上能跑；`serve_pool.sh` 需要一张 GPU 和装好的 sglang，0.6B 模型 24 GB 显存足够，`rr_qwen` 要用 SGLang 源码里的 jinja 模板，把源码克隆到 `$SGLANG_SRC`。

| 文件 | 内容 |
|---|---|
| sim_pool.py | 四个场景。models：六个模型的结构、池化方式、两家的启动参数和接口，附 vLLM 的默认开关作对照。ledger：只有 prefill 时一步的时间、读权重和算力的分界。rerank：一次查询里 bi-encoder、cross-encoder、decoder reranker 的在线计算量，以及共享前缀能省多少。mrl：Matryoshka 截维对向量库存储的影响，两家对 `dimensions` 的校验和起服时怎么声明 |
| pool_client.py | 标准库客户端：embed、search（bi-encoder 召回）、rerank（`/v1/rerank`）、bench（并发压测；`--rerank-docs` 压 rerank，补上 SGLang 没有的 rerank 压测） |
| serve_pool.sh | 真机实验：五种起法（含 `--prefill-only-disable-kv-cache`）、`sglang.bench_serving --backend sglang-embedding`、rerank 压测、客户端演示。没有在 GPU 上实跑过 |
| data/docs.txt | 12 段推理服务相关的短文，给 search 和 rerank 当候选 |
| test_sim_pool.py | 16 个用例，同 vLLM 那一章；客户端对着本地假服务端测两家响应格式的解析 |

## 和源码的对应

路径相对 `python/sglang/`，文档路径相对仓库根目录。

| 本目录 | sglang |
|---|---|
| `sglang_flags`：Qwen3-Embedding 要 `--is-embedding` | `srt/configs/model_config.py:2378` is_generation_model：Qwen3ForCausalLM 不在非生成名单里，不加 flag 就按生成模型起；`docs/docs/supported-models/embedding_models.mdx:8` |
| `sglang_flags`：XLM-R 自动切 embedding | `srt/configs/model_config.py:2398`；`srt/configs/embedding_model_spec.py` 的架构表，`srt/arg_groups/model_hook.py:841` 打日志 |
| `sglang_flags`：bge 用 triton 后端、手动关 radix 和 chunked prefill | `docs/docs/supported-models/rerank_models.mdx` 的启动命令；`embedding_model_spec.py` 里只有 Gemma 系的 spec 会自动关这两项 |
| `sglang_flags`：Qwen3-Reranker 不能加 `--is-embedding` | `docs/docs/supported-models/rerank_models.mdx:117`；`srt/entrypoints/openai/serving_rerank.py:88` 按模板文字判断走 decoder 打分还是 cross-encoder |
| Qwen3 的池化方式 | `srt/models/qwen3.py:594`：`Pooler(pooling_type=PoolingType.LAST, normalize=True)` |
| `serve_pool.sh emb_nokv` | `srt/mem_cache/kv_cache_configurator.py:812`：要求 `--is-embedding`、`--chunked-prefill-size -1`、`--disable-radix-cache`、FA3/FA4 prefill 后端，否则启动报错 |
| `scenario_mrl`、`serve_pool.sh emb` 的 `--json-model-override-args` | `srt/managers/tokenizer_manager.py:1485` 没有 is_matryoshka 就拒绝 dimensions，再校验范围和 matryoshka_dimensions 列表；`srt/configs/model_config.py:919` is_matryoshka 读 config；`srt/layers/pooler.py:258` 先截维再归一化 |
| 接口列表 | `srt/entrypoints/http_server.py`：`/encode` 993、`/classify` 1006、`/v1/embeddings` 1883、`/v1/classify` 1899、`/v1/score` 2050、`/v1/rerank` 2102 |
| rerank 响应格式 | `srt/entrypoints/openai/protocol.py:1729` RerankResponse，接口直接返回列表 |
| `serve_pool.sh bench_emb` | `benchmark/serving.py:981` sglang-embedding 后端，`:999` 打 `/v1/embeddings` |

## 运行

```bash
python3 sim_pool.py --scenario models
python3 sim_pool.py --scenario ledger --model qwen3-emb-0.6b --doc-len 512
python3 sim_pool.py --scenario rerank --docs 100 --query-len 32 --doc-len 512
python3 sim_pool.py --scenario mrl --vectors 10000000
python3 test_sim_pool.py

# 有 GPU 时（未实测）
bash serve_pool.sh emb                       # 终端 1
bash serve_pool.sh client                    # 终端 2
bash serve_pool.sh bench_emb 512
bash serve_pool.sh emb_nokv                  # 换成不写 KV 的起法，再压一遍对比
bash serve_pool.sh rr_bge
bash serve_pool.sh bench_rr 100
```

`ledger` 场景按 vLLM 的每步 token 预算排批；SGLang 的对应参数是 `--chunked-prefill-size` 和 `--max-prefill-tokens`，本章没有移植它的批预算逻辑，只看一步的算力和读权重时间。

## 期望输出

```
模型                  架构                                  注意力  池化        矩阵乘参数  FLOP/token@512
Qwen3-Embedding-0.6B  Qwen3ForCausalLM                      因果    LAST             440 M          0.94 G
Qwen3-Embedding-4B    Qwen3ForCausalLM                      因果    LAST            3633 M          7.42 G
Qwen3-Embedding-8B    Qwen3ForCausalLM                      因果    LAST            6946 M         14.04 G
bge-m3                XLMRobertaModel                       双向    CLS              302 M          0.65 G
bge-reranker-v2-m3    XLMRobertaForSequenceClassification   双向    CLASSIFY         302 M          0.65 G
Qwen3-Reranker-0.6B   Qwen3ForCausalLM                      因果    CLASSIFY         440 M          0.94 G

vLLM 对池化模型的默认值（vllm/config/model.py:2200、:2246）：
  Qwen3-Embedding-0.6B   chunked prefill 开  prefix caching 开
  Qwen3-Embedding-4B     chunked prefill 开  prefix caching 开
  Qwen3-Embedding-8B     chunked prefill 开  prefix caching 开
  bge-m3                 chunked prefill 关  prefix caching 关
  bge-reranker-v2-m3     chunked prefill 关  prefix caching 关
  Qwen3-Reranker-0.6B    chunked prefill 开  prefix caching 开
  双向注意力不能分块、也不能复用前缀；因果注意力配 LAST 池化两者都能开，配 MEAN / CLS 就不行。
  启动后如果 KV spec 带 non_causal，引擎还会再强制关一遍（vllm/v1/engine/core.py:274）。

启动参数：
  Qwen3-Embedding-0.6B
    vLLM    vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-0.6B --is-embedding
  Qwen3-Embedding-4B
    vLLM    vllm serve Qwen/Qwen3-Embedding-4B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-4B --is-embedding
  Qwen3-Embedding-8B
    vLLM    vllm serve Qwen/Qwen3-Embedding-8B --runner pooling
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-8B --is-embedding
  bge-m3
    vLLM    vllm serve BAAI/bge-m3 （不用加，架构本身就是池化模型）
    SGLang  python3 -m sglang.launch_server --model-path BAAI/bge-m3 --attention-backend triton --disable-radix-cache --chunked-prefill-size -1
  bge-reranker-v2-m3
    vLLM    vllm serve BAAI/bge-reranker-v2-m3 （不用加，架构本身就是池化模型）
    SGLang  python3 -m sglang.launch_server --model-path BAAI/bge-reranker-v2-m3 --is-embedding --attention-backend triton --disable-radix-cache --chunked-prefill-size -1
  Qwen3-Reranker-0.6B
    vLLM    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' --chat-template examples/pooling/score/template/qwen3_reranker.jinja
    SGLang  python3 -m sglang.launch_server --model-path Qwen/Qwen3-Reranker-0.6B --disable-radix-cache --chat-template examples/chat_template/qwen3_reranker.jinja

接口：
  vLLM    /v1/embeddings、/v2/embed、/pooling；/score、/v1/score；/rerank、/v1/rerank、/v2/rerank；/classify
  SGLang  /v1/embeddings、/encode；/v1/rerank；/v1/score；/classify、/v1/classify
```

一次查询的 rerank 账。SGLang 文档给 Qwen3-Reranker 的启动参数带 `--disable-radix-cache`，所以最后一行的前缀复用在 SGLang 上默认拿不到；想要这部分收益，要去掉这个开关并自己验证打分不变：

```
一次查询：query 32 token，候选 100 篇，每篇 512 token；H100 SXM，MFU 假设 40%
做法                                  在线 token     GFLOP   算力下限 ms
bi-encoder：只算 query 向量                   32      28.3         0.072
cross-encoder：bge-reranker-v2-m3         54,800   36050.2        91.128
decoder：Qwen3-Reranker-0.6B              61,800   58821.0       148.688
decoder + 前缀全部命中                    53,880   51282.8       129.633
bi-encoder 的文档向量离线算好：100 篇 × 512 token 只在入库时算一次，查询时只剩 100 次点积。
cross-encoder 每个 (query, 文档) 对都要整段过一遍模型，所以先用向量召回几百篇，再只对前几十篇 rerank。
Qwen3-Reranker 的模板把指令和 query 放在文档前面，同一次查询的 100 个对共享前 93 个 token，按 16 token 一块能复用 80 个。
  这一行是上限：要求前缀块已经算完并留在缓存里，同一步里一起排进去的请求未必互相命中，实际命中率看
  vLLM 的 prefix cache 命中率指标。SGLang 文档给 Qwen3-Reranker 的启动参数带 --disable-radix-cache，这时一个都不复用。
```

截维：

```
10,000,000 个向量的原始存储（不含索引结构）：
  维度   fp32 GB   fp16 GB
  4096     163.8      81.9
  2560     102.4      51.2
  1024      41.0      20.5
   512      20.5      10.2
   256      10.2       5.1
   128       5.1       2.6
截维要模型训练时用过 Matryoshka（MRL），否则召回会明显变差。两家都要求模型 config 先声明支持：
  vLLM    config 里要有 is_matryoshka 或 matryoshka_dimensions，否则请求直接 400（vllm/pooling_params.py:246）。
          Qwen3-Embedding 的 config.json 两个都没有，启动时加 --hf-overrides '{"is_matryoshka": true}'。
  SGLang  同样要 is_matryoshka 或 matryoshka_dimensions（srt/configs/model_config.py:919），否则拒绝请求；
          再校验 1 到 hidden_size、以及在 matryoshka_dimensions 列表里（srt/managers/tokenizer_manager.py:1485）。
          启动时加 --json-model-override-args '{"is_matryoshka": true}'；池化后先截维再做 L2 归一化（srt/layers/pooler.py:258）。
```

## 模拟器的假设

- 参数量按 config.json 逐项累加，只算参与矩阵乘的部分；词表 embedding 是查表，不进每步的读写和 FLOP。
- 注意力 FLOP 用 4 × 层数 × 头数 × head_dim × 可见长度，因果取平均可见 (L+1)/2，双向取 L。
- MFU 默认 0.4，是假设值。
- Qwen3-Reranker 模板的 token 数（61 / 4 / 9）是用 Qwen3 tokenizer 对默认 instruction 实际数出来的。
