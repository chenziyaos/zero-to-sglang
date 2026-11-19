"""Roofline 账本：prefill 和 decode 一步要搬多少字节、算多少 FLOP，落在拐点的哪一侧。

模型：一步前向的时间下限 = max(FLOP / 峰值算力, 字节 / HBM 带宽)。只计 HBM 读写和矩阵乘，
不计 kernel launch、通信、采样和调度开销，所以实测一定比下限慢。实测 / 下限 就是利用率：
decode 看 MBU（带宽利用率），prefill 看 MFU（算力利用率）。

decode 一步读的字节：
  权重：稠密部分全读；MoE 只读这一步被选中的专家，batch 为 b 时每层期望命中
        E x (1 - (1 - k/E)^b) 个专家（假设路由均匀独立）
  KV  ：每个请求读自己全部上下文；滑窗层最多读 window 个 token
prefill 一个 chunk：权重读一遍（MoE 在几千 token 下基本命中全部专家），KV 写一遍，
        注意力 FLOP 按因果掩码算一半

用法：
    python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm
    python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm --ctx 1024 --kv-dtype fp8
    python3 sim_roofline.py --model qwen3-235b-a22b --gpu h100-sxm --tp 8 --weight-dtype fp8
    python3 sim_roofline.py --model deepseek-v3 --gpu h200 --tp 8 --ctx 2048
    python3 sim_roofline.py --model qwen3-8b --gpu h100-sxm --jsonl result.jsonl   # 用 one_batch 实测算 MBU/MFU
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common.specs import GPUS, KV_DTYPE, MODELS, WEIGHT_DTYPE  # noqa: E402
from sim_ledger import ledger  # noqa: E402

GB = 1e9


def experts_hit(E, k, b):
    """b 个 token 各选 k 个专家（共 E 个）时，一层里被至少选中一次的专家数期望。"""
    return E * (1 - (1 - k / E) ** b)


def peak_tflops(gpu, weight_dtype):
    # FP8 权重走 FP8 Tensor Core（W8A8）；MXFP4 在 Hopper 上反量化后按 BF16 算
    return gpu.fp8_tflops if weight_dtype == 'fp8' and gpu.fp8_tflops else gpu.bf16_tflops


class Step:
    def __init__(self, key, gpu_key, tp=1, weight_dtype=None, kv_dtype='bf16'):
        self.m = m = MODELS[key]
        self.g = GPUS[gpu_key]
        self.tp = tp
        self.wd = weight_dtype or m.native_dtype
        self.e_kv = KV_DTYPE[kv_dtype]
        self.dense_b, self.exp_b = WEIGHT_DTYPE[self.wd]
        self.peak = peak_tflops(self.g, self.wd) * 1e12
        self.bw = self.g.bw_tbs * 1e12
        # 每 token 计算用到的参数（扣掉输入 embedding 查表），拆成稠密部分和路由专家部分
        routed_active = m.topk * m.moe_layers * m.expert_b
        self.dense_active = (m.active_b - routed_active - m.embed_b) * 1e9
        self.dense_all = (m.params_b - m.routed_b - m.embed_b) * 1e9
        self.compute_params = (m.active_b - m.embed_b) * 1e9

    # ---- KV 与注意力 ----
    def kv_per_layer_token(self):
        m = self.m
        if m.mla_kv_rank:
            return (m.mla_kv_rank + m.mla_rope_dim) * self.e_kv      # 每卡完整一份 latent
        return max(1, m.kv_heads // self.tp) * 2 * m.head_dim * self.e_kv

    def kv_read(self, ctx):
        """一个请求在 decode 一步里读的 KV 字节（每卡）。"""
        m = self.m
        swa_ctx = min(ctx, m.window) if m.swa_layers else 0
        return self.kv_per_layer_token() * (m.full_layers * ctx + m.swa_layers * swa_ctx)

    def attn_flops(self, q_tokens, ctx_tokens):
        """q_tokens 个 query 对 ctx_tokens 个 key 做注意力的 FLOP（每卡，QK 和 PV 两次乘加）。"""
        m = self.m
        heads = m.heads / self.tp
        if m.mla_kv_rank:
            # 吸收权重后：QK 在 rank+rope 维上做，PV 在 rank 维上做
            per = 2 * heads * (m.mla_kv_rank + m.mla_rope_dim) + 2 * heads * m.mla_kv_rank
        else:
            per = 4 * heads * m.head_dim
        swa_ctx = min(ctx_tokens, m.window) if m.swa_layers else 0
        return per * q_tokens * (m.full_layers * ctx_tokens + m.swa_layers * swa_ctx)

    # ---- 一步 decode ----
    def decode(self, bs, ctx):
        m = self.m
        w = self.dense_active * self.dense_b
        if m.experts:
            hit = experts_hit(m.experts, m.topk, bs)
            w += hit * m.moe_layers * m.expert_b * 1e9 * self.exp_b
        else:
            hit = 0
        w /= self.tp
        kv = bs * self.kv_read(ctx)
        fl = 2 * self.compute_params / self.tp * bs + self.attn_flops(bs, ctx)
        return self._bound(fl, w + kv, extra=dict(weights=w, kv=kv, hit=hit))

    # ---- 一个 prefill chunk（单请求从 0 开始）----
    def prefill(self, n):
        m = self.m
        w = self.dense_all * self.dense_b
        if m.experts:
            w += experts_hit(m.experts, m.topk, n) * m.moe_layers * m.expert_b * 1e9 * self.exp_b
        w /= self.tp
        kv = n * self.kv_per_layer_token() * m.layers
        fl = 2 * self.compute_params / self.tp * n + self.attn_flops(n, n) / 2
        return self._bound(fl, w + kv)

    def _bound(self, fl, by, extra=None):
        t_c, t_m = fl / self.peak, by / self.bw
        d = dict(flops=fl, bytes=by, ai=fl / by, t=max(t_c, t_m), t_c=t_c, t_m=t_m,
                 bound='算力' if t_c > t_m else '带宽')
        d.update(extra or {})
        return d


def report(s, ctx, kv_dtype):
    m, g = s.m, s.g
    L = ledger(next(k for k, v in MODELS.items() if v is m), next(k for k, v in GPUS.items() if v is g),
               s.tp, s.wd, kv_dtype)
    cap = L['tokens'] // ctx if L['tokens'] else 0
    ridge = s.peak / s.bw
    print(f'{m.name} on {s.tp} x {g.name}  weight={s.wd} kv={kv_dtype}  拐点 {ridge:.0f} FLOP/B')
    print(f'  KV 池 {L["tokens"]:,} token，上下文 {ctx} 时最多驻留 {cap} 个请求')
    print(f'[decode，每请求上下文 {ctx}]')
    print('    batch   权重GB    KV GB   强度   下限ms   tok/s      瓶颈' + ('  命中专家/层' if m.experts else ''))
    for bs in (1, 8, 32, 64, 128, 256, 512):
        d = s.decode(bs, ctx)
        mark = '' if bs <= cap else '  超出 KV 池'
        hit = f'{d["hit"]:10.1f}' if m.experts else ''
        print(f'    {bs:5d}  {d["weights"] / GB:7.2f}  {d["kv"] / GB:7.2f}  {d["ai"]:5.0f}  {d["t"] * 1e3:7.2f}'
              f'  {bs / d["t"]:7.0f}  {d["bound"]:>6}{hit}{mark}')
    print('[prefill，单个 chunk]')
    print('    tokens   强度   下限ms   tok/s     瓶颈')
    for n in (64, 256, 1024, 2048, 4096, 8192, 16384):
        d = s.prefill(n)
        print(f'    {n:6d}  {d["ai"]:5.0f}  {d["t"] * 1e3:7.2f}  {n / d["t"]:8.0f}  {d["bound"]:>5}')
    # 稠密部分的强度：decode 约等于 batch，prefill 约等于 token 数（BF16 时每参数 2 FLOP 对 2 字节）
    lo, hi = 1, 1 << 16
    while lo < hi:
        mid = (lo + hi) // 2
        if s.prefill(mid)['bound'] == '算力':
            hi = mid
        else:
            lo = mid + 1
    print(f'  prefill 从 {lo} token 起转为算力瓶颈')


def mbu_from_jsonl(s, path):
    """读 sglang.benchmark.one_batch 的 result.jsonl，算 decode MBU 和 prefill MFU。"""
    print(f'{"batch":>6}{"in":>7}{"out":>6}{"decode ms":>11}{"下限 ms":>9}{"MBU":>7}{"prefill s":>11}{"MFU":>7}')
    for line in open(path):
        r = json.loads(line)
        bs, i, o = r['batch_size'], r['input_len'], r['output_len']
        d = s.decode(bs, i + o // 2)            # 取 decode 过程的平均上下文
        mbu = d['bytes'] / r['median_decode_latency'] / s.bw
        p = s.prefill(i)
        mfu = p['flops'] * bs / r['prefill_latency'] / s.peak
        print(f'{bs:6d}{i:7d}{o:6d}{r["median_decode_latency"] * 1e3:11.2f}{d["t"] * 1e3:9.2f}{mbu:7.0%}'
              f'{r["prefill_latency"]:11.4f}{mfu:7.0%}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--gpu', default='h100-sxm', choices=GPUS)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--weight-dtype', choices=WEIGHT_DTYPE, help='默认取官方 checkpoint 的精度')
    ap.add_argument('--kv-dtype', default='bf16', choices=KV_DTYPE)
    ap.add_argument('--ctx', type=int, default=4096, help='decode 时每个请求的上下文长度')
    ap.add_argument('--jsonl', help='one_batch 输出的 result.jsonl')
    a = ap.parse_args()
    s = Step(a.model, a.gpu, a.tp, a.weight_dtype, a.kv_dtype)
    if a.jsonl:
        mbu_from_jsonl(s, a.jsonl)
    else:
        report(s, a.ctx, a.kv_dtype)


if __name__ == '__main__':
    main()
