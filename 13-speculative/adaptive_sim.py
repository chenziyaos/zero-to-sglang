"""Adaptive 投机解码：SGLang 怎么按接受长度在线切换 num_steps。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  speculative/adaptive_spec_params.py   DEFAULT_ADAPTIVE_CONFIG、AdaptiveStepSlot（EMA + 迟滞）、
                                        AdaptiveSpeculativeParams（按 batch size 路由到槽位）
  arg_groups/speculative_hook.py:1433   _init_adaptive_speculative_params：起始 steps 取候选并集的中位

  python3 adaptive_sim.py                     # trace：接受率分三段变化，看 steps 怎么跟
  python3 adaptive_sim.py --scenario route    # route：batch size 怎么路由到 1 / 8 / 32 / 64 四个槽位
"""
import argparse
import bisect
import json
import math
import random

DEFAULT_ADAPTIVE_CONFIG = {
    '1': {'candidate_steps': [1, 3, 5, 7], 'up_hysteresis': 0.0, 'down_hysteresis': -0.25,
          'ceiling_coeff': 0},
    '8': {'candidate_steps': [0, 1, 3], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0,
          'ceiling_coeff': 0},
    '32': {'candidate_steps': [0, 1], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0,
           'ceiling_coeff': 0},
    '64': {'candidate_steps': [0], 'up_hysteresis': 0.0, 'down_hysteresis': 0.0,
           'ceiling_coeff': 0},
}


def load_config(cfg=None):
    """_load_adaptive_config：cfg 可以是 dict、json 路径或 None。非数字 key 是对所有槽位生效的全局项。"""
    if cfg is None:
        cfg = DEFAULT_ADAPTIVE_CONFIG
    elif isinstance(cfg, str):
        with open(cfg) as f:
            cfg = json.load(f)
    bs_entries = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue
        steps = entry.get('candidate_steps')
        if not isinstance(steps, list) or not steps or not all(isinstance(s, int) and s >= 0 for s in steps):
            raise ValueError(f'BS {key}: candidate_steps must be a list of non-negative ints, got {steps!r}')
        bs_entries[int(key)] = entry
    if not bs_entries:
        raise ValueError('speculative_adaptive_config must contain at least one integer-string BS key')
    return cfg, bs_entries


def resolve_candidate_steps(cfg=None):
    _, bs_entries = load_config(cfg)
    return sorted({s for e in bs_entries.values() for s in e['candidate_steps']})


class AdaptiveStepSlot:
    """一个 batch size 档位的状态：EMA 平滑的接受草稿数 + 上下迟滞。"""

    def __init__(self, initial_steps, cfg):
        self.candidate_steps = sorted(set(cfg['candidate_steps']))
        self.ema_alpha = cfg.get('ema_alpha', 0.2)
        self.update_interval = cfg.get('update_interval', 5)
        self.warmup_batches = cfg.get('warmup_batches', 10)
        self.down_hysteresis = cfg.get('down_hysteresis', -0.25)
        self.up_hysteresis = cfg.get('up_hysteresis', 0.0)
        self.ceiling_coeff = cfg.get('ceiling_coeff', 0)
        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req):
        if not num_correct_drafts_per_req:
            return False
        if self.current_steps > 0:
            avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
            self.ema_accept_len = (1 - self.ema_alpha) * self.ema_accept_len + self.ema_alpha * avg
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute()

    def _recompute(self):
        old = self.current_steps
        idx = self.candidate_steps.index(old)
        old_idx = idx
        if old == 0:
            # 关掉投机一段时间后，每个更新周期都试探下一档；没有历史就从该档的中性 EMA 开始
            idx = min(idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[idx]
            if target > 0 and self.ema_accept_len < 0:
                self.ema_accept_len = float(target - 1)
            return self._apply(old, target)
        while idx > 0:
            prev = self.candidate_steps[idx - 1]
            drop = (0.5 if prev == 0 else prev - 0.5) + self.down_hysteresis
            if self.ema_accept_len <= drop:
                idx -= 1
            else:
                break
        if idx == old_idx:
            while idx < len(self.candidate_steps) - 1:
                if self.ema_accept_len > self.candidate_steps[idx] - 0.5 + self.up_hysteresis:
                    idx += 1
                else:
                    break
        target = self.candidate_steps[idx]
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if ceiling < target <= old:
                while idx > 0 and self.candidate_steps[idx] > ceiling:
                    idx -= 1
                target = self.candidate_steps[idx]
        return self._apply(old, target)

    def _apply(self, old, target):
        if target != old:
            self.current_steps = target
            return True
        return False


class AdaptiveSpeculativeParams:
    def __init__(self, initial_steps, cfg=None, cuda_graph_bs=None):
        cfg, bs_entries = load_config(cfg)
        self._bs_list = sorted(bs_entries)
        self._slots = {bs: AdaptiveStepSlot(initial_steps, {**cfg, **e}) for bs, e in sorted(bs_entries.items())}
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    @property
    def candidate_steps(self):
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def slot_key(self, batch_size):
        return self._find_closest_bs(self._pad(batch_size))

    def get_steps_for_batch(self, batch_size):
        return self._slots[self.slot_key(batch_size)].current_steps

    def on_verify_complete(self, num_correct, batch_size):
        slot = self._slots[self.slot_key(batch_size)]
        return slot.current_steps if slot.update(num_correct) else None

    def _pad(self, bs):
        if self._cuda_graph_bs is None:
            return bs
        i = bisect.bisect_left(self._cuda_graph_bs, bs)
        return self._cuda_graph_bs[i] if i < len(self._cuda_graph_bs) else bs

    def _find_closest_bs(self, target):
        i = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, i)]


# ---------------------------------------------------------------------------

def draw_correct(alpha, steps, rng):
    """链式草稿：逐个以概率 alpha 接受，第一次失败就停。返回接受的草稿数（不含 bonus）。"""
    n = 0
    while n < steps and rng.random() < alpha:
        n += 1
    return n


def round_cost(steps, c):
    """一轮的相对耗时：目标验证一次记 1，每步草稿记 c；steps=0 就是普通 decode。"""
    return 1.0 + c * steps


PHASES = [(0.85, 150, '抄写/格式化输出'), (0.55, 150, '一般对话'), (0.25, 150, '高温度创作')]


def run_trace(policy, phases=PHASES, batch=1, c=0.08, seed=0, record=False):
    """policy 是固定 steps 的整数，或 'adaptive'。返回产出 token / 耗时，以及每轮的 steps 轨迹。"""
    rng = random.Random(seed)
    ad = AdaptiveSpeculativeParams(initial_steps=3) if policy == 'adaptive' else None
    tokens = cost = 0.0
    trace = []
    for alpha, rounds, _ in phases:
        for _ in range(rounds):
            steps = ad.get_steps_for_batch(batch) if ad else policy
            correct = [draw_correct(alpha, steps, rng) for _ in range(batch)]
            tokens += sum(x + 1 for x in correct) / batch
            cost += round_cost(steps, c)
            if ad:
                ad.on_verify_complete(correct, batch)
            if record:
                ema = ad._slots[ad.slot_key(batch)].ema_accept_len if ad else None
                trace.append((alpha, steps, ema))
    return tokens / cost, trace


def oracle_steps(alpha, c, cands=(1, 3, 5, 7)):
    def eff(k):
        return (1 - alpha ** (k + 1)) / (1 - alpha) / round_cost(k, c)
    return max(cands, key=eff)


def scenario_trace(c=0.08, seed=0):
    print(f'batch=1，草稿成本 c={c}（每步草稿相当于目标前向的 {c:.0%}），三段接受率各 150 轮')
    eff, trace = run_trace('adaptive', c=c, seed=seed, record=True)
    print('  轮次   α     当前 steps   EMA(接受草稿数)')
    switches = []
    prev = None
    for i, (alpha, steps, ema) in enumerate(trace):
        if steps != prev:
            switches.append((i, alpha, steps, ema))
            print(f'  {i:4d}  {alpha:.2f}   {steps:6d}       {ema:6.2f}')
            prev = steps
    print('\n  每段的理论最优 steps（候选 1/3/5/7）：' +
          '，'.join(f'α={a} -> {oracle_steps(a, c)}' for a, _, _ in PHASES))
    res = {'adaptive': eff, 'switches': switches}
    print('\n  策略         产出 token / 单位耗时')
    for k in (1, 3, 5, 7):
        e, _ = run_trace(k, c=c, seed=seed)
        res[k] = e
        print(f'  固定 steps={k}       {e:.3f}')
    print(f'  adaptive           {eff:.3f}')
    print("""
  EMA 的 alpha 是 0.2，大约 5 轮记忆；前 10 轮 warmup 不动，之后每 5 轮评估一次，
  一次评估可以连跳几档（while 循环），所以换段后 10 到 20 轮就跟上了。
  向下的阈值是 prev-0.5-0.25，比向上的 cur-0.5 低 0.25，这是迟滞，防止在两档之间来回切。
  高接受率那一段它在 3 和 5 之间摆，到不了理论最优的 7：EMA 记的是"接受了几个草稿"，
  这个数被当前 steps 截断（只猜 3 个就最多接受 3 个），α=0.85 时 steps=3 的期望只有 2.18，
  steps=5 也只有 3.15，够不到升到 7 的阈值 4.5。规则天然偏保守，草稿便宜、α 很高的场景
  可以自己写 config 把候选改成 [3,5,7] 或者调低 up_hysteresis。
  即便如此，它跟住了三段的变化，总效率仍然高于任何一个固定 steps。""")
    return res


def scenario_route():
    ad = AdaptiveSpeculativeParams(initial_steps=3, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    print(f'候选 steps 并集 {ad.candidate_steps}，起始 steps 取中位 = '
          f'{ad.candidate_steps[len(ad.candidate_steps) // 2]}')
    print('  batch   补齐到 CUDA graph bs   槽位   该槽位候选 steps   初始 steps')
    out = {}
    for bs in [1, 3, 7, 8, 9, 20, 31, 33, 48, 64, 100]:
        padded = ad._pad(bs)
        key = ad.slot_key(bs)
        slot = ad._slots[key]
        out[bs] = (padded, key, slot.current_steps)
        print(f'  {bs:5d}   {padded:12d}            {key:4d}   {str(slot.candidate_steps):16s}   {slot.current_steps}')
    print("""
  路由分两步：先 bisect_left 补齐到不小于 batch 的 CUDA graph 尺寸，再 bisect_right-1 找不超过它的槽位。
  batch 9 会补到 12，落进 8 号槽位；batch 33 补到 48，落进 32 号槽位。
  起始 steps=3 不在 32 号槽位的候选 [0,1] 里，构造函数取候选中位，所以是 1；64 号槽位只有 0，
  也就是 batch 过了 64 就不投机。每个槽位各自维护 EMA，batch 在档位间跳来跳去时互不污染。""")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='trace', choices=['trace', 'route'])
    ap.add_argument('--c', type=float, default=0.08)
    a = ap.parse_args()
    if a.scenario == 'trace':
        scenario_trace(a.c)
    else:
        scenario_route()


if __name__ == '__main__':
    main()
