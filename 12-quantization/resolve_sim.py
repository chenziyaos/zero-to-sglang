"""启动时量化方法怎么定下来：命令行、checkpoint 配置、GPU 三方对账的纯 Python 移植。

只用标准库，CPU 上能跑。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  configs/model_config.py:2018      _verify_quantization：checkpoint 覆盖、兼容表、不匹配报错、ROCm 白名单
  configs/model_config.py:1745      _parse_quant_hf_config：quant_method 缺省或为 modelopt 时按 quant_algo 解析
  layers/quantization/__init__.py   各平台注册表（CUDA 上没有裸 gptq）
  layers/quantization/awq/awq.py:333    awq -> awq_marlin 的覆盖规则
  layers/quantization/gptq/gptq.py:414  gptq -> gptq_marlin 的覆盖规则
  layers/quantization/moe_wna16.py:164  只在用户显式要求时生效
  model_loader/loader.py:259        最低算力和激活 dtype 检查
  layers/quantization/marlin_utils.py   Marlin 支持的类型、group、形状
  layers/quantization/fp8_utils.py:2852 validate_fp8_block_shape
  layers/quantization/fp8.py:508、:1142 FP8 线性层走哪条路
  mem_cache/kv_cache_dtype.py、layers/quantization/kv_cache.py  KV dtype 和 k/v scale 的兜底

  python3 resolve_sim.py                      # 十几种 checkpoint x 参数 x GPU 组合，最后落到哪个方法或报什么错
  python3 resolve_sim.py --scenario marlin    # Marlin 的类型 / group / 形状约束，按 TP 切完以后还能不能跑
  python3 resolve_sim.py --scenario fp8path   # 同一个 FP8 checkpoint 在不同 GPU 上走哪条 kernel
  python3 resolve_sim.py --scenario kv        # --kv-cache-dtype 的解析和 k/v scale 的兜底
"""
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 平台
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Platform:
    name: str
    kind: str = 'cuda'          # cuda / hip / cpu / npu / xpu
    cap: int = 90               # major*10+minor；ROCm 上 torch 返回 gfx 架构号，MI300X 是 94
    cuda: Tuple[int, int] = (12, 8)
    deepgemm: bool = True       # 装了 deep_gemm 且 SGLANG_ENABLE_JIT_DEEPGEMM 没关
    flashinfer: bool = True


PLATFORMS = {
    'v100': Platform('V100', cap=70),
    't4': Platform('T4', cap=75),
    'a100': Platform('A100', cap=80),
    'a10': Platform('A10', cap=86),
    'l40s': Platform('L40S', cap=89),
    'l40s-cu121': Platform('L40S + CUDA 12.1', cap=89, cuda=(12, 1)),
    'h100': Platform('H100', cap=90),
    'h100-nodg': Platform('H100 无 DeepGEMM', cap=90, deepgemm=False),
    'b200': Platform('B200', cap=100),
    'rtx5090': Platform('RTX 5090', cap=120),
    'mi300x': Platform('MI300X', kind='hip', cap=94),
}


class QuantError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 注册表和三张名单（configs/model_config.py:2018 起）
# ---------------------------------------------------------------------------
BASE = ['fp8', 'mxfp8', 'minimax_m3_nvfp4', 'blockwise_int8', 'modelopt', 'modelopt_fp8',
        'modelopt_fp4', 'nvfp4_online', 'modelopt_mixed', 'w8a8_int8', 'w8a8_fp8', 'awq',
        'awq_marlin', 'bitsandbytes', 'gguf', 'gptq_marlin', 'moe_wna16', 'compressed-tensors',
        'w4afp8', 'petit_nvfp4', 'quark', 'quark_mxfp4', 'auto-round', 'auto-round-int8',
        'modelslim', 'quark_int4fp8_moe', 'humming', 'mxfp_w4a8']

ROCM_OK = {'awq', 'fp8', 'compressed_tensors', 'compressed-tensors', 'w8a8_fp8', 'petit_nvfp4',
           'quark', 'mxfp4', 'mxfp8', 'auto-round', 'auto-round-int8', 'quark_int4fp8_moe',
           'quark_mxfp4'}
OPTIMIZED = {'fp8', 'marlin', 'modelopt_fp8', 'modelopt_fp4', 'modelopt_mixed', 'nvfp4_online',
             'gptq_marlin_24', 'gptq_marlin', 'awq_marlin', 'compressed_tensors',
             'compressed-tensors', 'experts_int8', 'w8a8_int8', 'w8a8_fp8', 'moe_wna16', 'w4afp8',
             'petit_nvfp4', 'quark', 'modelslim', 'humming', 'quark_mxfp4', 'auto-round',
             'minimax_m3_nvfp4'}
COMPATIBLE = {
    'modelopt_fp8': ['modelopt'],
    'modelopt_fp4': ['modelopt', 'fp8'],
    'modelopt_mixed': ['modelopt'],
    'nvfp4_online': ['fp8', 'modelopt_fp8'],
    'minimax_m3_nvfp4': ['mxfp8'],
    'petit_nvfp4': ['modelopt'],
    'w8a8_int8': ['compressed-tensors', 'compressed_tensors'],
    'w8a8_fp8': ['compressed-tensors', 'compressed_tensors'],
    'auto-round-int8': ['compressed-tensors', 'compressed_tensors'],
}
REQUANT = ['quark_mxfp4']


def registry(p: Platform) -> List[str]:
    """layers/quantization/__init__.py：BASE 加上各平台的补充项。"""
    reg = list(BASE)
    if p.kind in ('cpu', 'cuda', 'xpu') or (p.kind == 'hip' and p.cap == 95):
        reg.append('mxfp4')
    if p.kind in ('npu', 'cpu', 'xpu'):
        reg.append('gptq')          # CUDA 上的裸 GPTQ kernel 已经删掉了
    return reg


# 最低算力（get_min_capability）和支持的激活 dtype（get_supported_act_dtypes）
MIN_CAP = {'awq': 75, 'awq_marlin': 80, 'gptq_marlin': 80, 'fp8': 80, 'mxfp8': 100,
           'w8a8_fp8': 89, 'w8a8_int8': 75, 'modelopt_fp8': 89, 'modelopt_fp4': 80,
           'modelopt_mixed': 80, 'compressed-tensors': 70, 'moe_wna16': 70, 'mxfp4': 80}
ACT_DTYPES = {'awq': ['float16'], 'gptq': ['float16']}
DEFAULT_ACT = ['bfloat16', 'float16']

# ---------------------------------------------------------------------------
# Marlin（layers/quantization/marlin_utils.py）
# ---------------------------------------------------------------------------
MIN_THREAD_N, MIN_THREAD_K = 64, 128
GROUPS = [-1, 32, 64, 128]
FP4_GROUPS = [16, 32]


def marlin_types(has_zp: bool, cap: int) -> List[str]:
    """query_marlin_supported_quant_types：算力不到 8.0 一个都没有。"""
    if cap < 80:
        return []
    if has_zp:
        return ['uint4']                                   # AWQ 风格：无符号 + 运行时零点
    return ['uint4b8', 'uint8b128', 'float8_e4m3fn', 'float4_e2m1f']   # GPTQ 风格：对称 + 偏置


def marlin_supported(qtype: str, group: Optional[int], has_zp: bool, cap: int) -> Tuple[bool, str]:
    if qtype not in marlin_types(has_zp, cap):
        return False, f'Marlin does not support weight_bits = {qtype} (device_capability = {cap}, zp = {has_zp})'
    allowed = FP4_GROUPS if qtype == 'float4_e2m1f' else GROUPS
    if group is None or group not in allowed:
        return False, f'Marlin does not support group_size = {group}. Only group_sizes = {allowed}'
    return True, ''


def marlin_shape(out_part: int, in_part: int, in_full: int, group: int) -> Tuple[bool, str]:
    """verify_marlin_supports_shape；group=-1 时按整行算。"""
    g = in_full if group == -1 else group
    if out_part % MIN_THREAD_N:
        return False, f'output_size_per_partition = {out_part} 不能被 min_thread_n = 64 整除'
    if in_part % MIN_THREAD_K:
        return False, f'input_size_per_partition = {in_part} 不能被 min_thread_k = 128 整除'
    if g < in_full and in_part % g:
        return False, f'input_size_per_partition = {in_part} 不能被 group_size = {g} 整除'
    return True, ''


GPTQ_TYPE = {(4, True): 'uint4b8', (8, True): 'uint8b128'}
AWQ_TYPE = {4: 'uint4', 8: 'uint8'}


def awq_marlin_compatible(cfg: dict, p: Platform) -> bool:
    if p.kind != 'cuda' or cfg.get('quant_method', '').lower() != 'awq':
        return False
    bits, group, zp = cfg.get('bits'), cfg.get('group_size'), cfg.get('zero_point')
    if bits is None or group is None or zp is None or bits not in AWQ_TYPE:
        return False
    return marlin_supported(AWQ_TYPE[bits], group, zp, p.cap)[0]


def gptq_marlin_compatible(cfg: dict, p: Platform) -> bool:
    if cfg.get('quant_method', '').lower() != 'gptq':
        return False
    bits, group, sym, desc = (cfg.get(k) for k in ('bits', 'group_size', 'sym', 'desc_act'))
    if None in (bits, group, sym, desc) or (bits, sym) not in GPTQ_TYPE:
        return False
    return marlin_supported(GPTQ_TYPE[(bits, sym)], group, False, p.cap)[0]


def marlin_format(cfg: dict) -> bool:
    return cfg.get('checkpoint_format') == 'marlin' or cfg.get('is_marlin_format', False)


def moe_wna16_compatible(cfg: dict, p: Platform) -> bool:
    qm, bits = cfg.get('quant_method', '').lower(), cfg.get('bits')
    if qm == 'gptq':
        return not cfg.get('desc_act') and bits in (4, 8)
    return qm == 'awq' and bits == 4 and p.cap >= 75


# ---------------------------------------------------------------------------
# checkpoint 覆盖：按注册表顺序问一遍，第一个给出答案的说了算
# ---------------------------------------------------------------------------
def modelopt_override(cfg: dict, user: Optional[str]) -> Optional[str]:
    if user == 'nvfp4_online' or user in REQUANT:
        return None
    algo = cfg.get('quant_algo', '').upper()
    if user == 'modelopt':
        if 'FP4' in algo:
            return 'modelopt_fp4'
        if algo == 'FP8':
            return 'modelopt_fp8'
    if cfg.get('quant_method') in ('modelopt_fp8', 'modelopt_fp4'):
        return cfg['quant_method']
    return None


def find_override(cfg: dict, user: Optional[str], p: Platform, log: List[str]) -> Optional[str]:
    if modelopt_override(cfg, user):
        return modelopt_override(cfg, user)
    if cfg.get('quant_method') == 'modelopt_mixed':
        return 'modelopt_mixed'
    if awq_marlin_compatible(cfg, p):
        if user in (None, 'marlin', 'awq_marlin'):
            log.append('info: The model is convertible to awq_marlin during runtime.')
            return 'awq_marlin'
        if user == 'awq':
            log.append('info: 能走 awq_marlin，但你指定了 awq，强制用 awq')
    if not marlin_format(cfg) and gptq_marlin_compatible(cfg, p):
        if user in (None, 'marlin', 'gptq_marlin'):
            log.append('info: The model is convertible to gptq_marlin during runtime.')
            return 'gptq_marlin'
        if user == 'gptq':
            log.append('info: 能走 gptq_marlin，但你指定了 gptq，强制用 gptq')
    if user == 'moe_wna16' and moe_wna16_compatible(cfg, p):
        return 'moe_wna16'
    return None


def parse_hf(cfg: Optional[dict]) -> Optional[dict]:
    """_parse_quant_hf_config 和 _parse_modelopt_quant_config 的主干。"""
    if cfg is None:
        return None
    cfg = dict(cfg)
    if 'quant_method' not in cfg or cfg['quant_method'] == 'modelopt':
        algo = cfg.get('quant_algo')
        if algo and 'FP4' in algo:
            cfg['quant_method'] = 'modelopt_fp4'
        elif algo == 'FP8':
            cfg['quant_method'] = 'modelopt_fp8'
    return cfg


@dataclass
class Resolved:
    method: Optional[str]
    log: List[str] = field(default_factory=list)


def verify(cli: Optional[str], ckpt: Optional[dict], p: Platform, dtype: str = 'bfloat16',
           draft: bool = False) -> Resolved:
    """_verify_quantization + loader.py 的算力 / dtype 检查。出错抛 QuantError，文本对齐源码。"""
    log: List[str] = []
    q = cli.lower() if cli else None
    cfg = parse_hf(ckpt)
    if cfg is not None:
        qm = cfg.get('quant_method', q or '').lower()
        if q not in REQUANT:
            o = find_override(cfg, q, p, log)
            if o:
                qm = q = o
        if q is None:
            q = qm
        elif q != qm:
            if q in COMPATIBLE and qm in COMPATIBLE[q]:
                log.append(f'info: Using CLI-specified quantization ({q}) which is compatible '
                           f'with HF config quant_method ({qm}).')
            elif draft:
                log.append(f'info: 草稿模型用 checkpoint 自己的量化方法 {qm}')
                q = qm
            elif q in REQUANT:
                log.append(f'info: Requantizing from {qm} to {q}，精度要重新评测')
            else:
                raise QuantError('Quantization method specified in the model config '
                                 f'({qm}) does not match the quantization method specified '
                                 f'in the `quantization` argument ({q}).')
    if q is None:
        return Resolved(None, log)
    if q not in registry(p):
        raise QuantError(f'Unknown quantization method: {q}.')
    if p.kind == 'hip' and q not in ROCM_OK:
        raise QuantError(f'{q} quantization is currently not supported in ROCm.')
    if q not in OPTIMIZED and not (q in ('mxfp4', 'mxfp8') and p.cap == 100):
        log.append(f'warning: {q} quantization is not fully optimized yet.')
    if p.kind == 'cuda':
        need = MIN_CAP.get(q, 0)
        if p.cap < need:
            raise QuantError(f'The quantization method {q} is not supported for the current GPU. '
                             f'Minimum capability: {need}. Current capability: {p.cap}.')
    ok = ACT_DTYPES.get(q, DEFAULT_ACT)
    if dtype not in ok:
        raise QuantError(f'{dtype} is not supported for quantization method {q}. '
                         f'Supported dtypes: {ok}')
    return Resolved(q, log)


# 常见 checkpoint 的 quantization_config（取自 Hugging Face 上对应仓库的 config.json）
CKPT = {
    'awq': {'quant_method': 'awq', 'bits': 4, 'group_size': 128, 'zero_point': True, 'version': 'gemm'},
    'gptq': {'quant_method': 'gptq', 'bits': 4, 'group_size': 128, 'sym': True, 'desc_act': False},
    'gptq-asym': {'quant_method': 'gptq', 'bits': 4, 'group_size': 128, 'sym': False, 'desc_act': False},
    'gptq-3bit': {'quant_method': 'gptq', 'bits': 3, 'group_size': 128, 'sym': True, 'desc_act': True},
    'gptq-g256': {'quant_method': 'gptq', 'bits': 4, 'group_size': 256, 'sym': True, 'desc_act': False},
    'fp8-block': {'quant_method': 'fp8', 'fmt': 'e4m3', 'activation_scheme': 'dynamic',
                  'weight_block_size': [128, 128]},
    'compressed-tensors': {'quant_method': 'compressed-tensors', 'format': 'float-quantized'},
    'modelopt-fp4': {'quant_algo': 'NVFP4', 'kv_cache_quant_algo': 'FP8'},
}

CASES = [
    # (说明, 平台, --quantization, checkpoint, --dtype)
    ('AWQ 4bit，不加参数', 'h100', None, 'awq', 'float16'),
    ('AWQ，--quantization awq', 'h100', 'awq', 'awq', 'float16'),
    ('AWQ，--quantization awq --dtype bfloat16', 'h100', 'awq', 'awq', 'bfloat16'),
    ('AWQ，T4 上不加参数', 't4', None, 'awq', 'float16'),
    ('AWQ，V100 上不加参数', 'v100', None, 'awq', 'float16'),
    ('AWQ，MI300X 上不加参数', 'mi300x', None, 'awq', 'float16'),
    ('GPTQ 4bit 对称，不加参数', 'a100', None, 'gptq', 'float16'),
    ('GPTQ，--quantization gptq', 'a100', 'gptq', 'gptq', 'float16'),
    ('GPTQ 非对称', 'a100', None, 'gptq-asym', 'float16'),
    ('GPTQ 3bit', 'a100', None, 'gptq-3bit', 'float16'),
    ('GPTQ group 256', 'a100', None, 'gptq-g256', 'float16'),
    ('GPTQ，MI300X 上不加参数', 'mi300x', None, 'gptq', 'float16'),
    ('FP8 块量化，不加参数', 'h100', None, 'fp8-block', 'bfloat16'),
    ('FP8 块量化，A100 上', 'a100', None, 'fp8-block', 'bfloat16'),
    ('FP8 块量化，--quantization awq', 'h100', 'awq', 'fp8-block', 'bfloat16'),
    ('FP8 块量化，--quantization fp8', 'h100', 'fp8', 'fp8-block', 'bfloat16'),
    ('BF16 模型，--quantization fp8', 'h100', 'fp8', None, 'bfloat16'),
    ('BF16 模型，V100 上 --quantization fp8', 'v100', 'fp8', None, 'float16'),
    ('BF16 模型，A100 上 --quantization w8a8_fp8', 'a100', 'w8a8_fp8', None, 'bfloat16'),
    ('compressed-tensors，--quantization w8a8_fp8', 'h100', 'w8a8_fp8', 'compressed-tensors', 'bfloat16'),
    ('ModelOpt NVFP4，不加参数', 'b200', None, 'modelopt-fp4', 'bfloat16'),
    ('ModelOpt NVFP4，--quantization modelopt', 'b200', 'modelopt', 'modelopt-fp4', 'bfloat16'),
]


def run_case(plat: str, cli: Optional[str], ck: Optional[str], dtype: str) -> Tuple[str, List[str]]:
    try:
        r = verify(cli, CKPT.get(ck) if ck else None, PLATFORMS[plat], dtype)
        return r.method or '不量化', r.log
    except QuantError as e:
        return 'ERROR ' + str(e), []


def scenario_resolve():
    for desc, plat, cli, ck, dtype in CASES:
        out, log = run_case(plat, cli, ck, dtype)
        print(f'[{PLATFORMS[plat].name}] {desc}')
        print(f'    -> {out}')
        for line in log:
            print(f'       {line}')
    print()
    print('  要点：不加 --quantization 是最稳的。AWQ / GPTQ checkpoint 在 8.0 以上的卡上会自动改走 Marlin；')
    print('  手动写 gptq 反而会撞上 CUDA 注册表里已经没有裸 GPTQ 的报错。AWQ 原生 kernel 只收 float16，')
    print('  所以显式写 awq 再配 bfloat16 会在加载前报 dtype 错。checkpoint 已经量化过时，--quantization')
    print('  只能写兼容表里允许的名字，写别的直接报不匹配。')
    print('  GPTQ 非对称、3 bit、group 256 这几类 Marlin 吃不下的 checkpoint，在 CUDA 上没有退路：')
    print('  override 不改名，quant_method 停在 gptq，注册表里找不到，直接 Unknown。要么重新量化成')
    print('  4/8 bit 对称 group 128，要么换 AWQ；MoE 模型还可以显式写 --quantization moe_wna16。')


# ---------------------------------------------------------------------------
# Marlin 和 FP8 块量化在 TP 切分以后的形状检查
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dense:
    name: str
    hidden: int
    inter: int


DENSE = {
    'qwen2.5-72b-bf16': Dense('Qwen2.5-72B（BF16 原版）', 8192, 29568),
    'qwen2.5-72b-awq': Dense('Qwen2.5-72B（官方 AWQ / GPTQ，intermediate 补到 29696）', 8192, 29696),
    'llama3.1-70b': Dense('Llama-3.1-70B', 8192, 28672),
    'qwen3-32b': Dense('Qwen3-32B', 5120, 25600),
}


def mlp_shards(m: Dense, tp: int) -> Dict[str, Tuple[int, int, int, List[int]]]:
    """MLP 两个线性层在 TP 下每卡的 (out_part, in_part, in_full, 合并前各段 out)。

    gate_up_proj 是列并行，两段 out 各切 tp 份再拼起来；down_proj 是行并行，切的是 in。
    """
    return {
        'gate_up_proj': (2 * m.inter // tp, m.hidden, m.hidden, [m.inter // tp] * 2),
        'down_proj': (m.hidden, m.inter // tp, m.inter, [m.hidden]),
    }


def fp8_block_ok(out_parts: List[int], in_part: int, in_full: int, tp: int,
                 block=(128, 128)) -> Tuple[bool, str]:
    """validate_fp8_block_shape：行并行查 in % block_k，列并行或合并矩阵查每段 out % block_n。"""
    bn, bk = block
    if tp > 1 and in_full // in_part == tp and in_part % bk:
        return False, f'input_size_per_partition = {in_part} 不能被 block_k = {bk} 整除'
    split = tp > 1 and in_full == in_part
    if split or len(out_parts) > 1:
        for o in out_parts:
            if o % bn:
                return False, f'output_partition_size = {o} 不能被 block_n = {bn} 整除'
    return True, ''


def shape_report(key: str, tps=(1, 2, 4, 8), group: int = 128):
    m = DENSE[key]
    rows = []
    for tp in tps:
        if m.inter % tp:
            rows.append((tp, False, f'intermediate {m.inter} 不能被 {tp} 整除', False, ''))
            continue
        mk, mmsg, fk, fmsg = True, '', True, ''
        for name, (o, i, full, parts) in mlp_shards(m, tp).items():
            ok, msg = marlin_shape(o, i, full, group)
            if mk and not ok:
                mk, mmsg = False, f'{name}: {msg}'
            ok, msg = fp8_block_ok(parts, i, full, tp)
            if fk and not ok:
                fk, fmsg = False, f'{name}: {msg}'
        rows.append((tp, mk, mmsg, fk, fmsg))
    return rows


def scenario_marlin():
    print('Marlin 支持的权重类型（query_marlin_supported_quant_types）')
    for cap in (75, 80, 90):
        print(f'  sm{cap}  有零点(AWQ): {marlin_types(True, cap) or "无"}   '
              f'无零点(GPTQ): {marlin_types(False, cap) or "无"}')
    print()
    print('checkpoint 能不能转 Marlin（A100）')
    a100 = PLATFORMS['a100']
    for k in ('awq', 'gptq', 'gptq-asym', 'gptq-3bit', 'gptq-g256'):
        cfg = CKPT[k]
        ok = awq_marlin_compatible(cfg, a100) if cfg['quant_method'] == 'awq' else gptq_marlin_compatible(cfg, a100)
        desc = ', '.join(f'{x}={cfg[x]}' for x in ('bits', 'group_size', 'sym', 'zero_point', 'desc_act') if x in cfg)
        print(f'  {k:<10} {desc:<52} {"可以" if ok else "不行"}')
    print()
    print('MLP 按 TP 切完以后的形状检查（group 128 的 Marlin / 128x128 的 FP8 块）')
    for key in DENSE:
        print(f'  {DENSE[key].name}')
        for tp, mk, mmsg, fk, fmsg in shape_report(key):
            mtxt = '通过' if mk else '不通过'
            ftxt = '通过' if fk else '不通过'
            print(f'    TP{tp:<2} Marlin {mtxt:<4} FP8 块 {ftxt:<4} {mmsg or fmsg}')
    print()
    print('  29568 = 2^7 x 231，切 2 份以后每份 14784 = 128 x 115.5，group 128 就对不齐了。')
    print('  Qwen 官方把 AWQ / GPTQ 版的 intermediate 补零到 29696 = 1024 x 29，切到 TP8 每份 3712 = 128 x 29。')
    print('  自己从 BF16 原版量化 72B 时要么同样补零，要么只能 TP1。FP8 块量化同理。')


# ---------------------------------------------------------------------------
# FP8 线性层的执行路径
# ---------------------------------------------------------------------------
def cutlass_fp8_supported(p: Platform) -> bool:
    if p.kind != 'cuda':
        return False
    if p.cap >= 90:
        return p.cuda >= (12, 0)
    if p.cap == 89:
        return p.cuda >= (12, 4)
    return False


def fp8_linear_path(p: Platform, block: bool, serialized: bool = True,
                    act_scheme: str = 'dynamic', force_marlin: bool = False) -> str:
    """Fp8LinearMethod 的选择：Marlin W8A16 / 块量化各后端 / 逐通道 CUTLASS / 逐张量。"""
    if p.kind == 'cuda' and p.cap < 80:
        return '报错：fp8 最低算力 80'
    use_marlin = p.kind == 'cuda' and (force_marlin or 80 <= p.cap < 89)
    if use_marlin:
        return 'Marlin W8A16：权重 FP8 存、kernel 里反量化，激活不量化'
    act = '激活按 token 动态量化' if act_scheme == 'dynamic' else '激活用静态 input_scale'
    if block:
        if p.cap >= 90 and p.deepgemm and p.cap != 120:
            return 'DeepGEMM 块量化 GEMM，激活 1x128 分组量化'
        if p.cap in (100, 103) and p.flashinfer:
            return 'FlashInfer TRTLLM 块量化 GEMM'
        if p.cap == 120:
            return 'CUTLASS 块量化 GEMM（SM120 auto 直接选 cutlass）'
        return 'Triton 块量化 GEMM，激活 1x128 分组量化'
    if not serialized:
        if cutlass_fp8_supported(p):
            return f'加载时逐通道量化权重，CUTLASS scaled_mm，{act}'
        return f'加载时逐张量量化权重，torch._scaled_mm，{act}'
    if cutlass_fp8_supported(p) and p.cap // 10 == 12:
        return f'SM120：逐通道 scale 合并成最大值，逐张量 GEMM，{act}'
    if cutlass_fp8_supported(p):
        return f'逐通道 scale，CUTLASS scaled_mm，{act}'
    return f'逐张量 scale（按最大值重量化），torch._scaled_mm，{act}'


def scenario_fp8path():
    plats = ['v100', 'a100', 'a10', 'l40s-cu121', 'l40s', 'h100', 'h100-nodg', 'b200', 'rtx5090']
    for title, kw in (('块量化 checkpoint（如 Qwen3-8B-FP8、DeepSeek-V3）', dict(block=True)),
                      ('逐张量 checkpoint（如 ModelOpt / llm-compressor 导出的 FP8）', dict(block=False)),
                      ('BF16 模型 + --quantization fp8（在线量化）', dict(block=False, serialized=False))):
        print(title)
        for k in plats:
            print(f'  {PLATFORMS[k].name:<18} {fp8_linear_path(PLATFORMS[k], **kw)}')
        print()
    print('  8.0 到 8.8 之间的卡（A100、A10、3090）没有 FP8 Tensor Core，SGLang 自动改走 Marlin W8A16：')
    print('  显存和带宽按 1 字节算，计算还是 BF16，所以只在 decode 带宽受限时赚。SGLANG_FORCE_FP8_MARLIN=1')
    print('  可以在 Hopper 上强制这条路，用来对比。L40S 是 sm89，CUDA 低于 12.4 时 CUTLASS 不可用，')
    print('  退回逐张量 scale，精度会略差。')


# ---------------------------------------------------------------------------
# KV cache dtype 和 scale
# ---------------------------------------------------------------------------
def kv_cache_dtype(arg: str, p: Platform, model_dtype: str = 'bfloat16',
                   ckpt_kv_algo: Optional[str] = None, torch_fp4: bool = True) -> str:
    """configure_kv_cache_dtype 的主干。"""
    fnuz = p.kind == 'hip'
    if arg == 'auto':
        if isinstance(ckpt_kv_algo, str) and ckpt_kv_algo.upper() == 'FP8':
            return 'float8_e4m3fnuz' if fnuz else 'float8_e4m3fn'
        return model_dtype
    if arg == 'fp8_e5m2':
        if p.kind == 'cpu':
            raise QuantError('--kv-cache-dtype fp8_e5m2 is not supported on CPU.')
        return 'float8_e4m3fnuz' if fnuz else 'float8_e5m2'
    if arg == 'fp8_e4m3':
        return 'float8_e4m3fnuz' if fnuz else 'float8_e4m3fn'
    if arg == 'mxfp8':
        return 'float8_e4m3fn'
    if arg in ('bf16', 'bfloat16'):
        return 'bfloat16'
    if arg in ('nvfp4', 'fp4_mx_block16'):
        if not torch_fp4:
            raise QuantError(f'--kv-cache-dtype={arg} requires torch.float4_e2m1fn_x2 support. '
                             'Please use PyTorch 2.8.0+ with CUDA 12.8+.')
        return 'float4_e2m1fn_x2（会打印 accuracy drop 警告）'
    raise QuantError(f'Unsupported kv_cache_dtype: {arg}.')


def kv_scales(k: float = -1.0, v: float = -1.0, fnuz: bool = False) -> Tuple[float, float]:
    """BaseKVCacheMethod.process_weights_after_loading：-1 表示 checkpoint 里没有。"""
    if k > 0 and v > 0:
        ks, vs = k, v
    elif k <= 0 and v <= 0:
        return 1.0, 1.0
    else:
        ks = vs = max(k, v)
    if fnuz:
        ks, vs = ks * 2, vs * 2
    return ks, vs


def scenario_kv():
    h100, mi = PLATFORMS['h100'], PLATFORMS['mi300x']
    print('--kv-cache-dtype 解析结果')
    rows = [('auto', h100, None), ('auto', h100, 'FP8'), ('fp8_e4m3', h100, None),
            ('fp8_e5m2', h100, None), ('fp8_e4m3', mi, None), ('fp8_e5m2', mi, None),
            ('nvfp4', h100, None), ('int8', h100, None)]
    for arg, p, algo in rows:
        src = f'，checkpoint 声明 kv_cache_quant_algo={algo}' if algo else ''
        try:
            out = kv_cache_dtype(arg, p, ckpt_kv_algo=algo)
        except QuantError as e:
            out = 'ERROR ' + str(e)
        print(f'  {arg:<9} [{p.name}{src}] -> {out}')
    print()
    print('k_scale / v_scale 兜底（-1 表示 checkpoint 里没有）')
    for k, v, fnuz in ((-1, -1, False), (0.12, 0.08, False), (0.12, -1, False), (0.12, 0.08, True)):
        print(f'  载入 k={k:<5} v={v:<5} {"ROCm fnuz" if fnuz else "CUDA     "} -> 实际用 {kv_scales(k, v, fnuz)}')
    print()
    print('  没有 scale 时按 1.0 算，等于假设 K、V 的绝对值都不超过 448（E4M3）。多数模型成立，')
    print('  K 里有大离群通道的模型会饱和，见 quant_sim.py --scenario kv。ROCm 的 fnuz 格式最大值')
    print('  只有 240，所以同一份 scale 要乘 2。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='resolve', choices=['resolve', 'marlin', 'fp8path', 'kv'])
    a = ap.parse_args()
    {'resolve': scenario_resolve, 'marlin': scenario_marlin, 'fp8path': scenario_fp8path,
     'kv': scenario_kv}[a.scenario]()


if __name__ == '__main__':
    main()
