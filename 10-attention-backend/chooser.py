"""注意力后端是怎么被选出来的：把 sglang 启动参数解析里和注意力后端有关的几步按原顺序重写一遍。

对应 sglang 8fabf5e（路径相对 python/sglang/srt/）：
  arg_groups/pipeline.py:225-241         调用顺序：模型 override → 确定性推理 → 后端兼容处理
  arg_groups/model_overrides/deepseek_v2.py:99,178  DSA 模型填 dsa；DeepSeek V3/R1 在 SM100 上填 trtllm_mla
  arg_groups/overrides.py:1093           _deterministic_attention_backend
  arg_groups/overrides.py:1127           _attention_backend_default
  arg_groups/model_override_base.py:277  get_default_attn_backend
  arg_groups/overrides.py:1144-1301      page_size 约束、TRTLLM MLA 校验、FA3 fp8_e5m2 回退、FA4 page 128
  arg_groups/overrides.py:1341           _page_size_default：没人设过就是 1
  layers/attention/attention_registry.py 构造时的断言（FA3 只认 SM80/SM90 等）

只用标准库。GPU 用 (SM 主版本, CUDA 版本) 描述，和 utils/common.py:287-330 的判定函数一致。

  python3 chooser.py                    # 常见 GPU × 模型的默认后端
  python3 chooser.py --scenario spec    # 投机解码 topk 和 page_size 怎么改变默认值
  python3 chooser.py --scenario rules   # 手动指定后端时被改写或拒绝的几种情况
  python3 chooser.py --scenario matrix  # 按功能需求从文档支持矩阵里筛后端
"""
import argparse
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple


# ---------------------------------------------------------------- 平台

@dataclass(frozen=True)
class Platform:
    name: str
    major: int = 9               # SM 主版本：A100=8, L40S=8(8.9), H100/H20=9, B200=10, RTX 5090=12
    cuda: Tuple[int, int] = (12, 8)
    hip: bool = False
    mps: bool = False
    has_flashinfer: bool = True

    # utils/common.py:287-330
    @property
    def hopper_cuda123(self) -> bool:
        return not self.hip and self.major == 9 and self.cuda >= (12, 3)

    @property
    def sm100(self) -> bool:
        return not self.hip and self.major == 10 and self.cuda >= (12, 8)

    @property
    def sm120(self) -> bool:
        return not self.hip and self.major == 12 and self.cuda >= (12, 8)

    @property
    def blackwell(self) -> bool:
        return not self.hip and self.major in (10, 11, 12) and self.cuda >= (12, 8)


PLATFORMS = {
    'a100': Platform('A100', 8, (12, 4)),
    'l40s': Platform('L40S', 8, (12, 4)),
    'h100': Platform('H100/H20', 9, (12, 4)),
    'h100-cu121': Platform('H100 + CUDA 12.1', 9, (12, 1)),
    'b200': Platform('B200', 10, (12, 8)),
    'rtx5090': Platform('RTX 5090', 12, (12, 8)),
    'mi300x': Platform('MI300X', 9, (0, 0), hip=True),
}


# ---------------------------------------------------------------- 模型

@dataclass(frozen=True)
class ModelCfg:
    name: str
    mla: bool = False
    kv_heads: int = 8
    attention_sinks: bool = False    # gpt-oss 一类带 sink 的模型，FlashInfer 不支持
    asymmetric_kv: bool = False      # K、V 行宽不同（v_head_dim != qk_head_dim）
    deepseek_v3: bool = False        # DeepseekV3ForCausalLM 一族（非 DSA）
    dsa: bool = False                # DeepSeek V3.2 / GLM-5 这类稀疏注意力


MODELS = {
    'qwen3-8b': ModelCfg('Qwen3-8B'),
    'llama3.1-70b': ModelCfg('Llama-3.1-70B'),
    'gpt-oss-120b': ModelCfg('gpt-oss-120b', attention_sinks=True),
    'deepseek-v3': ModelCfg('DeepSeek-V3', mla=True, kv_heads=128, deepseek_v3=True),
    'deepseek-v3.2': ModelCfg('DeepSeek-V3.2', mla=True, kv_heads=128, dsa=True),
}


# ---------------------------------------------------------------- 启动参数

@dataclass(frozen=True)
class Args:
    attention_backend: Optional[str] = None
    prefill_attention_backend: Optional[str] = None
    decode_attention_backend: Optional[str] = None
    page_size: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    speculative_eagle_topk: Optional[int] = None
    enable_deterministic_inference: bool = False
    kv_cache_dtype: str = 'auto'


DETERMINISTIC_CHOICES = ['ascend', 'fa3', 'fa4', 'flashinfer', 'intel_xpu', 'triton']  # choices.py:113


@dataclass
class Result:
    args: Args
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def prefill(self) -> str:
        return self.args.prefill_attention_backend or self.args.attention_backend

    @property
    def decode(self) -> str:
        return self.args.decode_attention_backend or self.args.attention_backend


def no_spec_or_topk_one(a: Args) -> bool:
    """utils/common.py:3960 is_no_spec_infer_or_topk_one。注意它读的是此刻的 page_size。"""
    return a.speculative_eagle_topk is None or (
        a.speculative_eagle_topk == 1 and a.page_size in (1, None))


def default_backend(p: Platform, m: ModelCfg, a: Args) -> str:
    """model_override_base.py:277 get_default_attn_backend 的逐行翻译（不含 OOT 平台和 Whisper）。"""
    if not m.mla:
        if p.hopper_cuda123 and no_spec_or_topk_one(a):
            return 'fa3'           # issue #17411：flashinfer 0.6.1 在 Hopper 上退化，先默认 FA3
        if p.sm100 and no_spec_or_topk_one(a) and (
                a.speculative_algorithm is None or a.speculative_eagle_topk is not None
                or str(a.speculative_algorithm).upper() == 'DFLASH'):
            return 'fa4' if m.asymmetric_kv else 'trtllm_mha'
        if p.hip:
            return 'aiter'
        if p.mps:
            return 'torch_native'
        return 'flashinfer' if p.has_flashinfer and not m.attention_sinks else 'triton'
    if p.hopper_cuda123:
        return 'fa3'
    if p.sm100:
        return 'flashinfer'
    if p.hip:
        return 'aiter' if m.kv_heads in (16, 128) else 'triton'
    if p.mps:
        return 'torch_native'
    return 'triton'


def _unset(a: Args) -> bool:
    return a.attention_backend is None and a.prefill_attention_backend is None \
        and a.decode_attention_backend is None


def _uses(a: Args, name: str, prefill=True) -> bool:
    return name in (a.attention_backend, a.decode_attention_backend) or \
        (prefill and a.prefill_attention_backend == name)


def resolve(p: Platform, m: ModelCfg, a: Args) -> Result:
    r = Result(a)

    def set_(why, **kw):
        r.args = replace(r.args, **kw)
        r.log.append(f'{why}: ' + ', '.join(f'{k}={v}' for k, v in kw.items()))

    # 1. 模型 override（pipeline.py:225 handle_model_specific_adjustments）
    if m.dsa and _unset(r.args):
        set_('deepseek_v2.py:102 DSA 模型', attention_backend='dsa')
    elif m.deepseek_v3 and p.sm100 and _unset(r.args):
        set_('deepseek_v2.py:185 DeepSeek V3 在 SM100', attention_backend='trtllm_mla')

    # 2. 确定性推理（overrides.py:1093），排在自动选择前面
    if r.args.enable_deterministic_inference:
        if r.args.attention_backend is None:
            if p.sm100 or p.sm120:
                b = 'triton' if (m.deepseek_v3 or m.dsa) else 'flashinfer'
            else:
                b = 'fa3'
            set_('overrides.py:1093 确定性推理', attention_backend=b)
        elif r.args.attention_backend not in DETERMINISTIC_CHOICES:
            r.error = f'确定性推理不支持 {r.args.attention_backend}，只支持 {DETERMINISTIC_CHOICES}'
            return r

    # 3. 拆分后端相同就合并；都没设就自动选（overrides.py:1127）
    a = r.args
    if a.prefill_attention_backend is not None and \
            a.prefill_attention_backend == a.decode_attention_backend:
        set_('overrides.py:1129 prefill 与 decode 相同', attention_backend=a.prefill_attention_backend)
    elif a.attention_backend is None:
        set_('overrides.py:1133 自动选择', attention_backend=default_backend(p, m, a))

    # 4. page_size 约束（overrides.py:1144）
    a, ps = r.args, r.args.page_size
    if _uses(a, 'flashmla', prefill=False):
        ps = 64
    if _uses(a, 'trtllm_mla', prefill=False) and ps not in (32, 64):
        ps = 64
    if _uses(a, 'cutedsl_mla') and ps not in (32, 64):
        ps = 64
    if _uses(a, 'trtllm_mha') and ps not in (16, 32, 64, 128):
        ps = 64
    if _uses(a, 'hpc_ops') and ps != 64:
        ps = 64
    if ps != a.page_size:
        set_('overrides.py:1144 后端要求的页大小', page_size=ps)

    # 5. TRTLLM MLA 校验（overrides.py:1204）
    if _uses(r.args, 'trtllm_mla', prefill=False):
        if not p.blackwell:
            r.error = 'TRTLLM MLA 只支持 Blackwell'
            return r
        if r.args.kv_cache_dtype not in ('fp8_e4m3', 'bf16', 'auto'):
            r.error = 'TRTLLM MLA 的 KV 只支持 fp8_e4m3 / bf16 / auto'
            return r

    # 6. CuteDSL MLA 只做 decode，prefill 默认补 trtllm_mla（overrides.py:1233）
    if _uses(r.args, 'cutedsl_mla'):
        if r.args.prefill_attention_backend == 'cutedsl_mla':
            r.error = 'CuteDSL MLA 只支持 decode'
            return r
        if not p.sm100:
            r.error = 'CuteDSL MLA 只支持 SM100'
            return r
        if r.args.prefill_attention_backend is None:
            set_('overrides.py:1233 CuteDSL 补 prefill', prefill_attention_backend='trtllm_mla')

    # 7. TRTLLM MHA 平台检查（attention_hook.py:198）
    if 'trtllm_mha' in (r.prefill, r.decode) and not (p.major in (9, 10, 12) and not p.hip):
        r.error = 'TRTLLM MHA 需要 SM90 / SM100 / SM120'
        return r

    # 8. FA3 遇到 fp8_e5m2 回退 Triton（overrides.py:1263）
    if r.args.attention_backend == 'fa3' and r.args.kv_cache_dtype == 'fp8_e5m2':
        set_('overrides.py:1263 FA3 只认 fp8_e4m3', attention_backend='triton')

    # 9. FA4 在 SM100 非 MLA 模型上强制 page 128（overrides.py:1274）
    if _uses(r.args, 'fa4') and not m.mla and p.sm100 and (r.args.speculative_eagle_topk or 0) <= 1:
        set_('overrides.py:1274 FA4 非 MLA', page_size=128)

    # 10. 没人设过 page_size 就是 1（overrides.py:1341）
    if r.args.page_size is None:
        set_('overrides.py:1341 默认页大小', page_size=1)   # MUSA 和 AITER 5D 布局例外，这里不模拟

    # 11. 构造后端时的断言（attention_registry.py）
    for b in {r.prefill, r.decode}:
        if b == 'fa3' and not p.hip and not ((p.major == 8 and not m.mla) or p.major == 9):
            r.error = 'attention_registry.py:240 FA3 需要 SM80（非 MLA）或 SM90'
        if b in ('trtllm_mha', 'hpc_ops') and m.mla:
            r.error = f'attention_registry.py {b} 只能用于非 MLA 模型'
    return r


def show(r: Result) -> str:
    if r.error:
        return f'报错：{r.error}'
    pd = r.prefill if r.prefill == r.decode else f'{r.prefill} / {r.decode}'
    return f'{pd:<22} page={r.args.page_size}'


# ---------------------------------------------------------------- 文档支持矩阵

# docs.sglang.io advanced_features/attention_backend，8fabf5e 同期版本。
# 列：native page>1, FP8 KV, FP4 KV, spec topk=1, spec topk>1, SWA, 多模态
MHA_MATRIX = {
    'flashinfer':   ('Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N'),
    'fa3':          ('Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y'),
    'fa4':          ('128', 'N', 'Y', 'Y', 'Y', 'Y', 'Y'),
    'triton':       ('N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'),
    'torch_native': ('N', 'Y', 'Y', 'N', 'N', 'N', 'Y'),
    'flex_attention': ('N', 'N', 'Y', 'N', 'N', 'N', 'N'),
    'trtllm_mha':   ('16/32/64', 'Y', 'Y', 'Y', 'N', 'Y', 'N'),
    'hpc_ops':      ('64', 'Y', 'N', 'N', 'N', 'N', 'N'),
    'aiter':        ('Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y'),
}
MHA_COLS = ('page', 'fp8', 'fp4', 'topk1', 'topk_n', 'swa', 'mm')

# 列：page, FP8 KV, FP4 KV, chunked prefix, spec topk=1, spec topk>1
MLA_MATRIX = {
    'flashinfer':  ('1', 'N', 'Y', 'Y', 'Y', 'N'),
    'flashmla':    ('64', 'Y', 'Y', 'Y', 'Y', 'N'),
    'cutlass_mla': ('128', 'Y', 'Y', 'Y', 'Y', 'N'),
    'trtllm_mla':  ('32/64', 'Y', 'Y', 'Y', 'Y', 'N'),
    'cutedsl_mla': ('32/64', 'Y', 'N', 'Y', 'Y', 'N'),
    'fa3':         ('n/a', 'N', 'N', 'Y', 'Y', 'page1'),
    'triton':      ('n/a', 'N', 'N', 'N', 'Y', 'page1'),
    'fa4':         ('1', 'N', 'Y', 'Y', 'N', 'N'),
}
MLA_COLS = ('page', 'fp8', 'fp4', 'chunked', 'topk1', 'topk_n')


def candidates(mla: bool, **need) -> List[str]:
    """need 里写要的功能，比如 fp8=True, topk_n=True。page=True 表示要原生 page>1。"""
    table, cols = (MLA_MATRIX, MLA_COLS) if mla else (MHA_MATRIX, MHA_COLS)
    out = []
    for name, row in table.items():
        ok = True
        for k, want in need.items():
            if not want:
                continue
            v = row[cols.index(k)]
            if k == 'page':
                ok &= v not in ('N', '1', 'n/a')
            else:
                ok &= v != 'N'
        if ok:
            out.append(name)
    return out


# ---------------------------------------------------------------- 场景

def scenario_default():
    print('  不加任何注意力参数时，各 GPU × 模型拿到的后端（prefill / decode，page_size）')
    print(f'  {"":<18}' + ''.join(f'{MODELS[k].name:<32}' for k in MODELS))
    for pk, p in PLATFORMS.items():
        row = ''.join(f'{show(resolve(p, MODELS[mk], Args())):<32}' for mk in MODELS)
        print(f'  {p.name:<18}{row}')
    print()
    print('  几个要看懂的格子：')
    print('    H100 + CUDA 12.1 掉到 flashinfer：Hopper 判定要求 CUDA ≥ 12.3')
    print('    gpt-oss 在 A100 上是 triton：模型带 attention sink，FlashInfer 不支持')
    print('    DeepSeek-V3 在 B200 上是 trtllm_mla 且 page=64：模型 override 先填了后端，页大小再被后端约束改写')
    print('    RTX 5090（SM120）不走 trtllm_mha：默认分支只认 SM100，落到 flashinfer')


def scenario_spec():
    p, m = PLATFORMS['h100'], MODELS['qwen3-8b']
    print('  H100 + Qwen3-8B，投机解码参数对默认后端的影响')
    cases = [
        ('不开投机', Args()),
        ('EAGLE topk 未设', Args(speculative_algorithm='EAGLE')),
        ('EAGLE topk=1', Args(speculative_algorithm='EAGLE', speculative_eagle_topk=1)),
        ('EAGLE topk=1, page=16', Args(speculative_algorithm='EAGLE', speculative_eagle_topk=1, page_size=16)),
        ('EAGLE topk=4', Args(speculative_algorithm='EAGLE', speculative_eagle_topk=4)),
        ('EAGLE topk=4, 指定 fa3', Args(attention_backend='fa3', speculative_algorithm='EAGLE',
                                       speculative_eagle_topk=4)),
    ]
    for name, a in cases:
        print(f'    {name:<26}{show(resolve(p, m, a))}')
    print()
    p = PLATFORMS['b200']
    print('  B200 + Qwen3-8B')
    for name, a in [('不开投机', Args()),
                    ('EAGLE topk 未设', Args(speculative_algorithm='EAGLE')),
                    ('EAGLE topk=1', Args(speculative_algorithm='EAGLE', speculative_eagle_topk=1)),
                    ('EAGLE topk=4', Args(speculative_algorithm='EAGLE', speculative_eagle_topk=4))]:
        print(f'    {name:<26}{show(resolve(p, m, a))}')
    print()
    print('  topk 未设的两种结局：投机参数的自动值（speculative_hook.py:1206）在注意力后端选完之后才填。')
    print('  H100 分支把 topk=None 当成没开投机，选 fa3；B200 分支额外要求 topk 已确定，退到 flashinfer。')
    print('  Llama 架构的自动值是 (steps=5, topk=4, draft=8)，所以 H100 上 Llama + EAGLE 不写 topk，')
    print('  拿到的是「按 topk=1 选的 fa3」配「实际 topk=4」。想要确定的结果，三个投机参数和后端都显式写。')


def scenario_rules():
    cases = [
        ('h100', 'qwen3-8b', Args(attention_backend='fa3', kv_cache_dtype='fp8_e5m2'), 'FA3 + fp8_e5m2'),
        ('a100', 'deepseek-v3', Args(attention_backend='fa3'), 'A100 上给 MLA 指定 FA3'),
        ('b200', 'qwen3-8b', Args(attention_backend='trtllm_mha', page_size=1), 'trtllm_mha + page 1'),
        ('b200', 'qwen3-8b', Args(attention_backend='fa4'), 'B200 + FA4（非 MLA）'),
        ('h100', 'deepseek-v3', Args(attention_backend='trtllm_mla'), 'H100 上指定 trtllm_mla'),
        ('b200', 'deepseek-v3', Args(decode_attention_backend='cutedsl_mla'), 'decode 用 cutedsl_mla'),
        ('b200', 'deepseek-v3', Args(prefill_attention_backend='fa4', decode_attention_backend='fa4'),
         'prefill 与 decode 都写 fa4'),
        ('h100', 'qwen3-8b', Args(prefill_attention_backend='fa3', decode_attention_backend='flashinfer'),
         '拆分：prefill fa3, decode flashinfer'),
        ('h100', 'qwen3-8b', Args(enable_deterministic_inference=True), '确定性推理，不指定后端'),
        ('b200', 'deepseek-v3', Args(enable_deterministic_inference=True), 'B200 + DeepSeek 确定性'),
        ('h100', 'qwen3-8b', Args(enable_deterministic_inference=True, attention_backend='trtllm_mha'),
         '确定性 + trtllm_mha'),
        ('a100', 'qwen3-8b', Args(attention_backend='hpc_ops', page_size=16), 'hpc_ops + page 16'),
    ]
    for pk, mk, a, name in cases:
        r = resolve(PLATFORMS[pk], MODELS[mk], a)
        print(f'  {name:<34}{show(r)}')
        for line in r.log:
            if '自动选择' not in line and '默认页大小' not in line:
                print(f'      {line}')


def scenario_matrix():
    print('  MHA 模型，按需求筛后端（来源：官方文档支持矩阵）')
    for desc, need in [('FP8 KV', dict(fp8=True)),
                       ('FP8 KV + page>1', dict(fp8=True, page=True)),
                       ('EAGLE topk>1 + page>1', dict(topk_n=True, page=True)),
                       ('多模态 + FP8 KV', dict(mm=True, fp8=True)),
                       ('FP4 KV + topk>1', dict(fp4=True, topk_n=True))]:
        print(f'    {desc:<26}{", ".join(candidates(False, **need))}')
    print('  MLA 模型')
    for desc, need in [('FP8 KV', dict(fp8=True)),
                       ('EAGLE topk>1', dict(topk_n=True)),
                       ('FP8 KV + chunked prefix', dict(fp8=True, chunked=True))]:
        print(f'    {desc:<26}{", ".join(candidates(True, **need))}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='default', choices=['default', 'spec', 'rules', 'matrix'])
    s = ap.parse_args().scenario
    {'default': scenario_default, 'spec': scenario_spec, 'rules': scenario_rules,
     'matrix': scenario_matrix}[s]()


if __name__ == '__main__':
    main()
