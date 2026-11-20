# SGL-20 读新 PR 与提 PR 的工具

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-sglang/blob/main/colab/20-contrib.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

对应飞书文档：SGL-20｜前沿｜42 天主干复盘：缓存寻址、PD 状态与投机解码的组合边界。文档按“原来卡在哪、引入了什么机制、代价落在哪”读主干的新改动；本目录的脚本用来在本地克隆里定位这些改动涉及的符号、自查自己的 PR。

验证基线：sglang 主干 `8fabf5e`。脚本只用标准库（`find_symbol.sh` 只用 grep），不依赖 GPU、torch 和 sglang 本体；`plan --repo`、`find_symbol.sh` 需要一份本地 sglang 克隆。

前面十九篇都在「用」sglang。这一篇换到另一侧：拿到一个真实 issue，从入口一路读到出错的那一行，改掉，配上测试，按仓库的规矩提出去，并且让它真的能被合。例子是 issue #22713：`sglang:utilization` 这个 gauge 在开源版里永远是 0：计算它的那段代码只在 `max_running_requests_under_SLO` 有值时才执行，而这个属性没有任何地方赋值，连 KV 占用那一项也一起被跳过了。半年里有六个 PR 碰过它，合进去的只有一个加 TODO 注释的。

| 文件 | 内容 |
|---|---|
| check_pr.py | 提 PR 前的自查。`title` 按 `.github/workflows/pr-title.yml` 的正则和硬件标签规则校验标题并给出中文原因；`plan` 读 `git diff --numstat`，报规模、动了几块代码、每个文件要哪组 CODEOWNERS 批、Merge Oncall 是谁、镜像单测在不在；`owners` 单查路径；`case` 回放 #22713 的六个 PR；`stats` 统计 300 个最近关闭的 PR |
| replay_utilization.py | 用 ast 从源码里取出 `_calculate_utilization`，在桩对象上对比修前（`git show 8fabf5e:…`）和修后（工作区）的输出，不需要 torch |
| find_symbol.sh | 给一个函数名，列定义、真正的调用点（排除注释和测试）、镜像单测路径（找不到同名时给近似文件）、已有测试里谁提到它 |
| pr_template.md | 照 sglang 官方模板填好的一份 PR 描述，带中文批注 |
| upstream/utilization_fix.patch | 对 8fabf5e 的修法：`metrics_reporter.py` 改 8 行，新增 77 行 CPU 单测 |
| data/CODEOWNERS.8fabf5e | 上游 CODEOWNERS，行号与原文一致（已退役的 /sgl-router 条目换成注释占位） |
| data/prs_closed_20261025_1030.json | 300 个 PR 的快照：编号、标题、作者身份、创建 / 关闭 / 合并时间 |
| data/case_22713.json | issue #22713 与六个相关 PR 的文件改动、结局和关闭原因 |
| data/numstat_utilization_fix.txt | 上面那份修法的 numstat |
| test_contrib.py | 23 个用例；标题部分直接用上游 `pr-title.test.cjs` 的 31 条合法、54 条非法例子，装了 node 时还会把同一批标题交给上游 JS 正则逐条对答案；回放部分直接拿补丁里的前后两段源码跑 |

## 和源码的对应

| 本目录 | 源码 |
|---|---|
| `TITLE_RE`、`BRANDS` | `.github/workflows/pr-title.yml:29-42`，用例 `.github/scripts/pr-title.test.cjs`；检查由 #44165 在 2026-10-14 引入。JS 的 `$` 只匹配串尾，Python 的 `$` 会放过末尾一个换行，所以这里用 `\Z` |
| `_pat_regex`、`owners_of` | `.github/CODEOWNERS`，gitignore 语义，后出现的匹配覆盖先出现的 |
| `ONCALL` | `.github/MAINTAINER.md` 的 Merge Oncall 表 |
| `mirror_test` | `test/registered/unit/README.md:9-13`：`srt/X/Y.py` 对应 `unit/X/test_Y.py` |
| `plan` 的覆盖率提示 | 同一 README：`diff-cover coverage.xml --compare-branch=origin/main --fail-under=60` |
| `plan` 的 400 / 1000 行 | Google eng-practices「Small CLs」：100 行左右合适，1000 行通常太大 |
| 被修的函数 | `python/sglang/srt/managers/scheduler_components/metrics_reporter.py:1303` `_calculate_utilization`，调用点 `:893`、`:1139` |
| 指标字段 | `python/sglang/srt/observability/metrics_collector.py`：`QueueCount :60`、`num_running_reqs :82`、`token_usage :97`、`utilization :154` |
| 单测写法 | 参照 `test/registered/unit/observability/test_prealloc_token_usage.py:64` 用 `SchedulerMetricsReporter.__new__` 绕过构造函数 |

## 运行

```bash
cd zero-to-sglang/20-contrib
python3 check_pr.py title '[Fix] observability: Fall back to max_running_requests for sglang:utilization'
python3 check_pr.py plan --title '[Fix] observability: ...' --numstat data/numstat_utilization_fix.txt --repo ../../sglang
python3 check_pr.py owners python/sglang/srt/managers/scheduler_pp_mixin.py
python3 check_pr.py case
python3 check_pr.py stats
bash find_symbol.sh _calculate_utilization ../../sglang
python3 replay_utilization.py ../../sglang   # 先在克隆里 git apply 补丁
python3 -m unittest test_contrib -v      # 约 0.1 s

# 在你自己的分支上
git diff --numstat origin/main...HEAD | python3 check_pr.py plan --title "$(git log -1 --format=%s)" --numstat - --repo .
# 想用仓库里最新的 CODEOWNERS
python3 check_pr.py plan ... --codeowners .github/CODEOWNERS
```

上游补丁的单测要在 sglang 开发环境里跑（`pip install -e "python"` 之后）：

```bash
cd sglang && git apply ../zero-to-sglang/20-contrib/upstream/utilization_fix.patch
pytest test/registered/unit/managers/scheduler_components/test_metrics_reporter.py -v
```

## 期望输出

```text
$ python3 replay_utilization.py ../../sglang
场景                        8fabf5e   工作区
48 / 64 在跑，KV 6%         0.0       0.75
8 在跑，KV 72%              0.0       0.8
设了 SLO 上限 16，8 在跑    0.5       0.5
prefill 实例                -1        -1

$ python3 check_pr.py case（节选）
  PR     作者身份         开      关      结果          文件     行   块  owner组  今天的标题 测试
  #22881 CONTRIBUTOR  04-15  06-10  未合并          3    91   2       2  不过    无
  #23212 CONTRIBUTOR  04-20  08-24  未合并          7    85   3       3  不过    无
  #24811 COLLABORATOR 05-09  05-10  未合并          2     2   2       2  不过    无
  #25441 COLLABORATOR 05-16  05-16  合并           2     2   2       2  不过    无
  #27465 NONE         06-06  09-13  未合并          1    14   1       1  不过    无
  #31461 NONE         07-16  07-27  未合并          2   115   1       1  不过    有

$ python3 check_pr.py plan ...
规模  2 个文件，+85 / -8，源码 1 个、测试 1 个
范围  1 块：srt/managers
      需要 1 组 CODEOWNERS 各至少一人批准
结论  可以提

$ python3 check_pr.py stats
  合并 242，关闭未合并 58
  从开到合并（小时）：p25 2.3  中位 10.2  p75 28.4  p90 52.6
  标题过 pr-title 检查：合并的 242/242，没合并的 47/58
  COLLABORATOR     183      18     91%
  CONTRIBUTOR       57      25     70%
  NONE               1      15      6%
```

## 读数时要记住的边界

- 「今天的标题」一列是用 2026-10-14 才上线的规则回看，不是那几个 PR 当年被拒的原因。它们真正卡住的地方在 `note` 里：混入无关改动、没有单测、CI 红了没人跟。
- `stats` 只是一周的快照，作者身份和合并率高度相关，但因果方向是双向的：常合 PR 的人才会被加成协作者。
- CODEOWNERS 每周都在变。正式提之前用 `--codeowners` 换成你本地主干上的那份。
- `find_symbol.sh` 用纯文本匹配，动态分发（`getattr`、注册表、`super()` 链）只能找到字面出现的地方；它是读源码的起点，不是调用图。
- 镜像规则是新测试的约定。8fabf5e 上 1065 个单测文件里，严格按 `test_<源文件名>` 对上源码的只有 173 个，很多老测试叫 `test_xxx_unit.py` 之类，所以脚本会给「近似」文件。
