"""SGL-22：给一台机器算出 DeepSeek-V4.1-Flash 落在哪一档、每卡显存怎么分、启动命令怎么写。

只用标准库。数字和文档第 2、3、5 章一致：
  checkpoint 510 GB = 骨干约 307 GB（FP8 稠密 + FP4 路由专家）+ Engram 两张 FP8 表约 203 GB
  GPU 上非专家部分约 12 GB，单个 FP4 路由专家约 18.8 MB，40 层，每 token 激活 6 个
  全局 KV 每 token：SM100/SM103 走 V4.1 原生格式约 890 字节，其他架构按约 1790 字节估
这些都是按 config.json 和 checkpoint 体积推算的估算值，上线前以实测为准。

用法：
  python3 plan.py --gpu h20-96 --n 8
  python3 plan.py --gpu h20-96 --n 4 --host-mem 384
  python3 plan.py --gpu rtx-pro-6000 --n 4 --host-mem 128
  python3 plan.py --gpu rtx5070ti --n 2 --host-mem 512 --mem-bw 410
  python3 plan.py --gpu rtx4090 --n 1 --host-mem 128
"""
import argparse
import sys
from dataclasses import dataclass

BACKBONE_GB = 307.0
ENGRAM_GB = 203.0
NONEXPERT_GB = 12.0
EXPERT_GB = 0.0188
LAYERS = 40
TOPK = 6
KV_BYTES_PER_TOKEN = 890          # SM100/SM103 上的 V4.1 原生 KV 格式
KV_BYTES_PER_TOKEN_LEGACY = 1790  # 其他架构走旧布局，按 vLLM fp8_ds_mla 的体积估
MIN_KV_GB = 5.0          # 每卡至少留给 KV 和 CUDA Graph 的空间
ROUTED_READ_GB = TOPK * LAYERS * EXPERT_GB  # 每 token 读的专家权重，约 4.5 GB


@dataclass(frozen=True)
class GPU:
    name: str
    sm: int          # 架构代号，AMD gfx950 记为 950
    vram: float      # GB
    bw: float        # 显存带宽 TB/s
    fp8: float       # FP8 稠密 TFLOPS，0 表示没有 FP8 张量核


# 厂商公开规格，数据中心卡取 SXM/OAM 版本
GPUS = {
    'b300': GPU('B300', 103, 288, 8.0, 4500),
    'b200': GPU('B200', 100, 180, 8.0, 4500),
    'mi355x': GPU('MI355X', 950, 288, 8.0, 5000),
    'h200': GPU('H200', 90, 141, 4.8, 989),
    'h100': GPU('H100', 90, 80, 3.35, 989),
    'h800': GPU('H800', 90, 80, 3.35, 989),
    'h20-141': GPU('H20 141G', 90, 141, 4.8, 296),
    'h20-96': GPU('H20 96G', 90, 96, 4.0, 296),
    'a100-80': GPU('A100 80G', 80, 80, 2.0, 0),
    'l40s': GPU('L40S', 89, 48, 0.86, 362),
    'rtx-pro-6000': GPU('RTX PRO 6000 96G', 120, 96, 1.79, 0),
    'rtx5090': GPU('RTX 5090', 120, 32, 1.79, 0),
    'rtx5080': GPU('RTX 5080', 120, 16, 0.96, 0),
    'rtx5070ti': GPU('RTX 5070 Ti', 120, 16, 0.90, 0),
    'rtx4090': GPU('RTX 4090', 89, 24, 1.0, 0),
    'rtx3090': GPU('RTX 3090', 86, 24, 0.94, 0),
}
NATIVE_SM = (90, 100, 103, 950)   # dev-dsv41 镜像和 FlashMLA 稀疏路径覆盖的架构
KT_MIN_SM = 89                    # 社区 KT 仓库：SM90 及以上原生，SM89 走 Triton 回退
SM120_NATIVE_MIN_N = 4            # 社区 4×RTX PRO 6000 配方：SGLang 原生 TP4，Engram 走 NVMe 加内存缓存
SM120_NATIVE_MIN_HOST = 128

# Unsloth 动态 GGUF：量化名、文件 GB、内存加显存下限 GB
GGUF = [('UD-Q4_K_XL', 500.7, 305), ('UD-IQ3_XXS', 394.7, 256),
        ('UD-Q2_K_XL', 334.8, 176), ('UD-IQ1_S', 233.0, 128)]


def kv_bytes(gpu):
    return KV_BYTES_PER_TOKEN if gpu.sm in (100, 103) else KV_BYTES_PER_TOKEN_LEGACY


def native_budget(gpu, n, engram_on_gpu, mem_frac):
    """原生 TP 时每卡的显存账，单位 GB。"""
    usable = gpu.vram * mem_frac
    backbone = BACKBONE_GB / n
    engram = ENGRAM_GB / n if engram_on_gpu else 0.0
    kv = usable - backbone - engram
    return {'usable': usable, 'backbone': backbone, 'engram': engram, 'kv': kv,
            'kv_mtokens': max(kv, 0) * 1e9 / kv_bytes(gpu) / 1e6}


# 已公开配置里每层常驻 GPU 的专家数
KT_KNOWN = {
    'rtx-pro-6000': (90, '社区 RTX PRO 6000 仓库'),
    'rtx5090': (10, 'KTransformers 官方 V4-Flash 配置'),
    'rtx5080': (1, '社区 16 GB 仓库'),
    'rtx5070ti': (1, '社区 16 GB 仓库'),
}


# 公开实测：(卡型, 卡数) -> (来源和测法, [(负载, 总输出 tok/s, 备注)])
PUBLIC = {
    ('h20-141', 8): ('sglang issue #46310，8×H20-3e，SGLang 02b9836，TP8/CP8 + DSpark 5，c8 固定并发，输出 1024', [
        ('8K 输入 c8', '984', 'TTFT 1.17 s，TPOT 5.56 ms'),
        ('16K 输入 c8', '685', 'TTFT 2.02 s，TPOT 8.05 ms'),
        ('32K 输入 c8', '396', 'TTFT 4.12 s，TPOT 14.1 ms'),
        ('64K 输入 c8', '217', 'TTFT 8.00 s，TPOT 27.0 ms'),
    ]),
    ('rtx-pro-6000', 4): ('社区 4×RTX PRO 6000 配方，TP4/EP4 + DSpark 5，Engram 走 NVMe 加 64 GiB 内存缓存，输出 8192', [
        ('512 输入 c1', '201', 'TTFT 0.26 s'),
        ('8K 输入 c1', '196', 'TTFT 1.12 s'),
        ('8K 输入 c8', '705', '每请求 89.9，TTFT 5.32 s'),
        ('64K 输入 c1', '232', 'TTFT 8.92 s'),
        ('128K 输入 c1', '203', 'TTFT 18.6 s'),
    ]),
    ('rtx5070ti', 2): ('社区 KT 仓库，2×EPYC 7452 + 512 GB DDR4，单流', [
        ('单流 decode', '37 至 50', '35K prefill 1736 tok/s'),
    ]),
}
# 没有同卡实测时借用的近邻数据：(卡型, 卡数) -> (借用的键, 差异说明)
PUBLIC_NEAR = {
    ('h20-96', 8): (('h20-141', 8), 'H20 96G 显存带宽 4.0 TB/s，比 H20-3e 的 4.8 TB/s 低约 17%，decode 按比例打折'),
}


def kt_gpu_experts(gpu, mem_frac=0.90):
    """KT 混合路径下每层能常驻 GPU 的专家数上限（估算）。"""
    kv_reserve = 7.0 if gpu.vram >= 48 else 5.0
    room = gpu.vram * mem_frac - NONEXPERT_GB - kv_reserve
    return max(int(room / (LAYERS * EXPERT_GB)), 0)


def decode_ceiling(mem_bw_gbs):
    """KT 路径单流 decode 的主机内存带宽上限，tok/s。"""
    return mem_bw_gbs / ROUTED_READ_GB


def classify(gpu, n, host_mem, mem_frac=0.90):
    """返回 (档位, 说明列表)。档位 1-5 和文档路线图自上而下一致。

    原生两档按每卡显存账判断：Engram 留显存后每卡 KV 还有 MIN_KV_GB 以上算第 1 档，
    只有 Engram 放主机才够算第 2 档。
    """
    notes = []
    if (gpu.sm == 120 and n >= SM120_NATIVE_MIN_N
            and native_budget(gpu, n, False, mem_frac)['kv'] >= MIN_KV_GB):
        if host_mem >= SM120_NATIVE_MIN_HOST:
            notes.append('社区配方，不在官方镜像验证矩阵：Engram 放 NVMe 加 64 GiB 内存缓存，'
                         '主机内存 256 GB 以上才考虑整份放内存')
            return 2, notes
        notes.append(f'SM120 多卡原生路径要求主机内存至少 {SM120_NATIVE_MIN_HOST} GB')
    if gpu.sm not in NATIVE_SM:
        if gpu.vram * n >= BACKBONE_GB:
            notes.append(f'SM{gpu.sm} 不在 dev-dsv41 镜像和 FlashMLA 稀疏路径的支持范围')
    elif gpu.fp8 == 0:
        notes.append(f'{gpu.name} 没有 FP8 张量核')
    elif native_budget(gpu, n, True, mem_frac)['kv'] >= MIN_KV_GB:
        return 1, notes
    elif native_budget(gpu, n, False, mem_frac)['kv'] >= MIN_KV_GB:
        if host_mem >= 256:
            return 2, notes
        notes.append(f'主机内存 {host_mem} GB 放不下 Engram（约 {ENGRAM_GB:.0f} GB），至少 256 GB')
    if gpu.sm >= KT_MIN_SM and 16 <= gpu.vram <= 96 and host_mem >= 256:
        if host_mem < 420:
            notes.append('主机内存不足 420 GiB，要开专家缓存，未命中的专家从 NVMe 读，单流约降到 15 至 20 tok/s')
        if gpu.sm == 89:
            notes.append('SM89 注意力走 Triton 回退，比 SM120 慢')
        return 3, notes
    if gpu.sm < KT_MIN_SM and host_mem >= 256:
        notes.append(f'SM{gpu.sm} 低于社区 KT 仓库的最低要求 SM89')
    if gpu.vram * n + host_mem >= 128:
        return 4, notes
    return 5, notes


def command(tier, gpu, n, mode):
    if tier == 2 and gpu.sm == 120:
        return '\n'.join([
            '# 社区 4×RTX PRO 6000 配方，见文档 3.4 节',
            'docker compose up --build -d',
            '# 实测候选配置（mem 0.95，上下文 524288）',
            'docker compose -f compose.yaml -f compose.measured.yaml up --build -d'])
    if tier in (1, 2):
        lines = ['sglang serve /models/DeepSeek-V4.1-Flash \\', f'  --tp {n} \\',
                 '  --reasoning-parser auto --tool-call-parser auto \\']
        if tier == 1 and mode == 'latency':
            lines += ['  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \\',
                      '  --mem-fraction-static 0.88 \\', '  --max-running-requests 8 \\']
        else:
            mrr = 64 if tier == 1 else 16
            lines += ["  --engram-config '{\"placement\": \"cpu\"}' \\",
                      '  --mem-fraction-static 0.90 \\', f'  --max-running-requests {mrr} \\']
            lines += ['  --schedule-policy fcfs \\'] if tier == 1 else ['  --cuda-graph-max-bs 16 \\']
        lines.append('  --host 0.0.0.0 --port 30000')
        return '\n'.join(lines)
    if tier == 3:
        return '\n'.join([
            '# 社区仓库按卡自动分配常驻专家，见文档 3.1 节',
            'MODEL=/models/dsv41-1017 ./setup.sh',
            'MODEL=/models/dsv41-1017 ./serve.sh'])
    if tier == 4:
        return '# llama.cpp 加 Unsloth 动态 GGUF，Engram 走 mmap，见文档 3.5 节'
    return '# 用托管 API，或等 SSD Expert Pack 出 V4.1 的 pack'


def report(a):
    gpu = GPUS[a.gpu]
    tier, notes = classify(gpu, a.n, a.host_mem)
    names = {1: 'SGLang 原生，整份进显存', 2: 'SGLang 原生，Engram 放主机',
             3: 'SGLang + KT-Kernel 混合', 4: '非 SGLang：llama.cpp GGUF', 5: '托管 API'}
    out = [f'{a.n}×{gpu.name}，显存合计 {gpu.vram * a.n:.0f} GB，主机内存 {a.host_mem} GB',
           f'档位：{tier}（{names[tier]}）']
    out += [f'  注意：{x}' for x in notes]
    if tier in (1, 2):
        rows = [(True, 0.88), (False, 0.88)] if tier == 1 else [(False, 0.90)]
        for on_gpu, mf in rows:
            b = native_budget(gpu, a.n, on_gpu, mf)
            where = 'Engram 留显存' if on_gpu else 'Engram 放主机'
            flag = '' if b['kv'] >= MIN_KV_GB else f'，低于 {MIN_KV_GB:.0f} GB，batch 只能压到个位数'
            out.append(f'  {where}（mem-fraction-static {mf}）：每卡骨干 {b["backbone"]:.1f} GB，'
                       f'Engram {b["engram"]:.1f} GB，KV {b["kv"]:.1f} GB，'
                       f'约 {b["kv_mtokens"]:.0f}M token{flag}')
        if gpu.sm not in (100, 103):
            out.append(f'  KV 按每 token {kv_bytes(gpu)} 字节估，非 SM100 走旧布局')
        if gpu.sm == 90:
            out.append('  SM90 没有原生 FP4，专家走 W4A16；W4A8 开关先和默认后端对比再保留')
    elif tier == 3:
        if a.gpu in KT_KNOWN:
            k, src = KT_KNOWN[a.gpu]
            out.append(f'  每层常驻 GPU 的专家：{k} 个（{src}），显存上限估算约 {kt_gpu_experts(gpu)} 个')
        else:
            out.append(f'  每层常驻 GPU 的专家：显存上限估算约 {kt_gpu_experts(gpu)} 个')
        if a.mem_bw:
            out.append(f'  主机内存带宽 {a.mem_bw} GB/s，单流 decode 理论上限约 '
                       f'{decode_ceiling(a.mem_bw):.0f} tok/s，实测通常只有一半多')
    elif tier == 4:
        pool = gpu.vram * a.n + a.host_mem
        pick = next(q for q in GGUF if pool >= q[2])
        out.append(f'  内存加显存 {pool:.0f} GB，可选最高 {pick[0]}（文件 {pick[1]} GB）')
    key, near = (a.gpu, a.n), None
    if key not in PUBLIC and key in PUBLIC_NEAR:
        key, near = PUBLIC_NEAR[key]
    if key in PUBLIC:
        src, rows = PUBLIC[key]
        out.append(f'公开实测（{src}）：')
        out += [f'  {w}：{t} tok/s，{x}' for w, t, x in rows]
        if near:
            out.append(f'  注意：这是近邻卡型的数据，{near}')
    out.append('启动命令：')
    out.append(command(tier, gpu, a.n, a.mode))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True, choices=sorted(GPUS))
    p.add_argument('--n', type=int, default=1, help='卡数')
    p.add_argument('--host-mem', type=int, default=0, help='主机内存 GB')
    p.add_argument('--mem-bw', type=float, default=0, help='主机内存带宽 GB/s，只用于 KT 档')
    p.add_argument('--mode', choices=['latency', 'throughput'], default='latency')
    a = p.parse_args(argv)
    print('\n'.join(report(a)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
