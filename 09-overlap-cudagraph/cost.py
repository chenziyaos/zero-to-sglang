"""一步 decode 的时间账：GPU 前向、CPU 调度、CPU 结果处理、kernel 启动。

sim_overlap.py 和 cuda_graph_sim.py 共用。GPU 时间按 Roofline 粗估（权重 + KV 读带宽、算力取大），
CPU 侧的几项是假设值，默认数字的量级和 LMSYS v0.4 博客「未优化的引擎可能有一半时间花在 CPU 开销上」一致。
自己的机器上要用 nsys 或 torch profiler 量一遍再改（见 README「校准」一节）。
"""
import os
import sys
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common.specs import GPUS, MODELS, weight_bytes  # noqa: E402


@dataclass(frozen=True)
class Cost:
    model: str = 'qwen3-8b'
    gpu: str = 'h100-sxm'
    tp: int = 1
    ctx: int = 1024              # decode 时每条请求的平均上下文长度
    bw_eff: float = 0.75         # 实际能吃到的 HBM 带宽比例
    flop_eff: float = 0.5
    kernels_per_layer: int = 14  # norm、qkv、rope、attn、o_proj、norm、gate_up、act、down、残差等
    op_us: float = 6.0           # eager 模式每个 kernel 的 CPU 开销：PyTorch 分发 + cudaLaunchKernel
    graph_us: float = 80.0       # CUDA Graph 一次 replay 的 CPU 开销：拷静态输入 + 注意力元数据 + cudaGraphLaunch
    allreduce_us: float = 12.0   # TP > 1 时每次 all-reduce 的延迟，每层两次
    sched_base_ms: float = 0.25  # 调度：收请求、组 batch、分配 KV、前缀匹配
    sched_per_req_ms: float = 0.004
    proc_base_ms: float = 0.15   # 结果处理：追加 token、查停止条件、流式输出、释放
    proc_per_req_ms: float = 0.008
    d2h_ms: float = 0.02         # next_token_ids 拷回 CPU


def gpu_ms(c: Cost, bs: int, tokens: int = None) -> float:
    """一次前向的 GPU 时间。tokens 默认等于 bs（decode）；prefill 传总 token 数。"""
    m, g = MODELS[c.model], GPUS[c.gpu]
    tokens = bs if tokens is None else tokens
    w = (weight_bytes(m, 'bf16') - m.embed_b * 1e9 * 2) / c.tp
    kv = bs * c.ctx * m.kv_bytes_per_token() / c.tp
    mem = (w + kv) / (g.bw_tbs * 1e12 * c.bw_eff)
    flops = 2 * m.active_b * 1e9 * tokens / c.tp
    comp = flops / (g.bf16_tflops * 1e12 * c.flop_eff)
    comm = (2 * m.layers * c.allreduce_us * 1e-6) if c.tp > 1 else 0.0
    return max(mem, comp) * 1e3 + comm * 1e3


def n_kernels(c: Cost) -> int:
    m = MODELS[c.model]
    return m.layers * (c.kernels_per_layer + (2 if c.tp > 1 else 0)) + 6   # 首尾 embedding、lm_head、采样


def launch_ms(c: Cost, graph: bool) -> float:
    """CPU 把一步前向的 kernel 发出去要多久。"""
    return c.graph_us / 1e3 if graph else n_kernels(c) * c.op_us / 1e3


def sched_ms(c: Cost, bs: int) -> float:
    return c.sched_base_ms + c.sched_per_req_ms * bs


def proc_ms(c: Cost, bs: int) -> float:
    return c.proc_base_ms + c.proc_per_req_ms * bs


PRESETS = {
    'qwen3-0.6b': Cost(model='qwen3-0.6b'),
    'qwen3-8b': Cost(model='qwen3-8b'),
    'llama-70b-tp4': Cost(model='llama3.1-70b', tp=4),
    'llama-70b-tp8': Cost(model='llama3.1-70b', tp=8),
}


def with_(c: Cost, **kw) -> Cost:
    return replace(c, **kw)
