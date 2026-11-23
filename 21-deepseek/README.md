# SGL-21 DeepSeek 组件的离线检查

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-sglang/blob/main/colab/21-deepseek.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-21｜前沿｜DeepSeek 开源如何改写 SGLang：从 kernel 到 DeepInfer。文档按时间线讲 DeepSeek 的模型、kernel 和 DeepInfer 引擎怎样一步步改变 SGLang 的工作重心；本目录的两个脚本对应文档第 4、5 章：起 V4.1 服务前先离线算出启动参数会被解析成什么，复测 DeepInfer 前先列出两份配置哪里没对齐。

验证基线：sglang 主干 `8fabf5e`。只用标准库，不依赖 GPU、torch 和 sglang 本体。

| 文件 | 内容 |
|---|---|
| check_config.py | `engram` 校验 `--engram-config`，报错文字与上游一致；`backend` 推出 `--dsv4-attn-backend auto` 最后落到 flashmla 还是 flashmla_mega，并列出原因；`plan` 把后端、Engram、有界重放、DSpark 加 PD 的约束合在一起给一份核对结果 |
| align_bench.py | 对比两份压测配置的六项：prefill 计时窗口、输入是否重复、Engram 放置、草稿模型冗余专家数、草稿 token 数、验证方式 |
| data/deepinfer_readme.json、data/lmsys_sglang.json | LMSYS 复测说明里两侧的测法，草稿 token 数和验证方式博客没给，故意留空，脚本会报缺失 |
| test_deepseek.py | 13 个用例 |

## 和源码的对应

| 本目录 | 源码 |
|---|---|
| `check_engram` | `python/sglang/srt/arg_groups/engram_config.py:41` `EngramConfig.from_dict`：peer_read 要求 placement 为 gpu，per_node 要求 placement 为 cpu，旧值 collective 按 auto 读 |
| `resolve_backend` | `python/sglang/srt/arg_groups/deepseek_v4_hook.py:54` `_default_dsv4_attention_backend`，注意力 TP 的判断在同文件 `:42`，开启 prefill CP 时不算注意力 TP |
| flashmla_mega 的硬约束 | 同文件 `:173`，要求 DeepSeek-V4.1 加 SM100/SM103 |
| 有界重放 | `python/sglang/srt/arg_groups/fields/exec_.py:168`、`:173`，decoder 一侧对支持的配置默认开 |
| 六项测法 | LMSYS「More Than DeepInfer」博客（2026-10-24）的复测说明 |

## 运行

```bash
cd zero-to-sglang/21-deepseek
python3 check_config.py backend --model deepseek_v41 --sm 100 --tp 4 --attn-dp 1
python3 check_config.py plan --tp 4 --attn-dp 4 --engram '{"placement":"cpu","lookup":"peer_read"}' --rl
python3 align_bench.py data/deepinfer_readme.json data/lmsys_sglang.json
python3 -m unittest test_deepseek.py -v
```

第一条会落到 flashmla：非 PD 部署下注意力 TP 为 4，MegaAttention 要把每卡补到 64 头，所以 auto 不选它。把 `--attn-dp` 改成 4 或加 `--disagg decode` 再跑一次，结果换成 flashmla_mega。第二条的 Engram 配置启动时会直接报错。
