# 一份填好的 PR 描述：issue #22713 的聚焦修法

骨架照抄 sglang 8fabf5e 的 `.github/pull_request_template.md`，五节一节不少。正文用英文写，因为审的人和以后翻历史的人都看英文；`<!-- -->` 里的中文是给你看的批注，提交时删掉。

标题（先在本地跑 `python3 check_pr.py title '...'`，过了再贴）：

```text
[Fix] observability: Fall back to max_running_requests for sglang:utilization
```

<!-- 类型用 Fix，scope 写 observability：gauge 定义在 srt/observability，审的人看到 scope 就知道该谁看。
     不要写 [Bugfix]、fix(metrics):、fix: 这几种，它们在 #27465、#23212、#22881 里出现过，今天的 pr-title 检查都会拦下。 -->

---

## Motivation

Fixes #22713.

`sglang:utilization` is computed in `SchedulerMetricsReporter._calculate_utilization` as
`max(num_running_reqs / max_running_requests_under_SLO, token_usage / 0.9)`.
`max_running_requests_under_SLO` has no setter in the open-source codebase (the two read sites carry a TODO from #25441),
so the guarded branch never runs and the gauge stays at its default 0.0, even the KV term is skipped. A router or autoscaler that reads
this gauge (it is also exported through `/v1/loads`) sees an idle replica while the running batch is already at `max_running_requests`.

<!-- Motivation 只写三件事：现在算的是什么、为什么错、谁受影响。给出函数名和 issue 号，审的人不用再去翻。
     不要写「起个服务就能看到」——#27465 的测试一节就只写了这句，挂了 98 天。 -->

## Modifications

- `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`: in the non-prefill branch, use `max_running_requests_under_SLO`
  only when it is set and positive; otherwise fall back to `scheduler.max_running_requests`. Prefill workers still report `-1`.
  The formula and the 0.9 KV headroom are unchanged.
- `test/registered/unit/managers/scheduler_components/test_metrics_reporter.py` (new, CPU-only, registered in `base-a-test-cpu`):
  7 cases covering the fallback, KV-dominated load, idle, SLO cap set, non-positive SLO cap, decode worker, prefill worker.

Out of scope: wiring a real setter for `max_running_requests_under_SLO`, and the speculative-decoding tracing that was bundled into #23212.

<!-- 「Out of scope」这一行最值钱：明说哪些事不做，等于提前回答了「为什么不顺手把 X 也修了」。
     #23212 被关的原因就是把投机解码的 tracing 塞进了同一个 PR。 -->

## Accuracy Tests

Not applicable: no kernel, model forward or sampling code is touched; only a Prometheus gauge changes.

## Speed Tests and Profiling

Not applicable: one `getattr` and one comparison per metrics log interval, off the forward path.

<!-- 不相关就写不相关，并给一句理由。空着不写，审的人会以为你忘了。 -->

## Checklist

- [x] Format your code according to the Format code with pre-commit. (`pre-commit run --all-files` clean)
- [x] Add unit tests according to the Run and add unit tests.
      `pytest test/registered/unit/managers/scheduler_components/test_metrics_reporter.py -v` → 7 passed
      `diff-cover coverage.xml --compare-branch=origin/main --fail-under=60` → changed lines covered
- [ ] Update documentation according to Write documentations. (no user-facing flag or API changes)
- [ ] Provide accuracy and speed benchmark results. (not applicable, see above)
- [x] Follow the SGLang code style guidance.

Manual check on 1×GPU, Qwen3-8B, `--max-running-requests 64`, 48 concurrent short requests:

```text
before  sglang:utilization  0.0    (branch skipped)
after   sglang:utilization  0.75   (48 / 64)
```

<!-- 单测证明逻辑对，手工这一段证明它在真服务上也对；两者都要。
     上面这组数字是示意：提交前用你自己机器上 curl :30000/metrics | grep utilization 的真实输出替换。
     没打勾的项保留，并在括号里写为什么不适用，不要删。 -->

## Review and Merge Process

1. Ping Merge Oncalls: Scheduler area (`managers/`) — @merrymercy @hnyls2002 @cctry.
2. CODEOWNERS: `/python/sglang/srt/managers` (CODEOWNERS line 55); the test file has no owner.
3. Trigger CI with `/tag-and-rerun-ci` (or ask someone with permission).
4. After green CI and approval, ask a Merge Oncall to merge.

<!-- 第 1、2 步的人名不要凭感觉写，用 `python3 check_pr.py owners <改动的文件>` 查：
     它按 CODEOWNERS「后出现的匹配覆盖先出现的」规则算，再按 MAINTAINER.md 的 Merge Oncall 表取最长前缀。
     @ 一次就够；24 小时没动静再在 PR 里礼貌提醒一次，不要私聊轰炸。 -->
