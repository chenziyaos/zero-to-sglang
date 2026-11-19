"""模型结构和 GPU 规格表，全系列练习共用。

数字取自各模型 Hugging Face config.json、模型卡和厂商公开规格页（稠密 BF16，不含稀疏加速）。
改动或新增条目时，在注释里写清出处。

口径约定：
  params_b    总参数量（十亿），含 embedding 和 lm_head
  active_b    每 token 参与计算的参数量；稠密模型等于 params_b
  embed_b     输入 embedding 表的参数量。decode 每步只按 token id 取几行，不整表读取，
              算带宽时从 active_b 里扣掉。tie_word_embeddings=True 的模型、或 active_b
              已经不含输入 embedding 的模型（gpt-oss 模型卡口径）填 0
  expert_b    单个路由专家的参数量（十亿），稠密模型为 0
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float
    active_b: float
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    mla_kv_rank: int = 0      # MLA 的压缩 KV 维度，非 MLA 为 0
    mla_rope_dim: int = 0     # MLA 解耦 RoPE 的维度
    context: int = 32768      # max_position_embeddings，决定 req_to_token 每行多长
    embed_b: float = 0.0
    experts: int = 0          # 路由专家数
    topk: int = 0             # 每 token 激活的路由专家数
    moe_layers: int = 0       # 带路由专家的层数
    expert_b: float = 0.0
    swa_layers: int = 0       # 滑动窗口层数，其余为全注意力层
    window: int = 0
    native_dtype: str = 'bf16'  # 官方 checkpoint 的权重精度

    @property
    def full_layers(self) -> int:
        return self.layers - self.swa_layers

    @property
    def routed_b(self) -> float:
        """全部路由专家的参数量（十亿）。"""
        return self.experts * self.moe_layers * self.expert_b

    def kv_bytes_per_token(self, dtype_bytes: float = 2) -> float:
        """不切分时，每个 token 在所有层上占用的 KV 字节数（SWA 层也按全长计）。"""
        if self.mla_kv_rank:
            # MLA 每层只存压缩后的 latent 和 RoPE key，K、V 共用
            return self.layers * (self.mla_kv_rank + self.mla_rope_dim) * dtype_bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


MODELS = {
    # Qwen/Qwen3-0.6B config.json；tie_word_embeddings=True
    'qwen3-0.6b': Model('Qwen3-0.6B', 0.6, 0.6, 28, 1024, 16, 8, 128, 151936, context=40960),
    # Qwen/Qwen3-8B config.json；按 config 逐层累加为 8,190,735,360 个参数
    'qwen3-8b': Model('Qwen3-8B', 8.19, 8.19, 36, 4096, 32, 8, 128, 151936, context=40960,
                      embed_b=0.622),
    # meta-llama/Llama-3.1-8B config.json
    'llama3.1-8b': Model('Llama-3.1-8B', 8.0, 8.0, 32, 4096, 32, 8, 128, 128256, context=131072,
                         embed_b=0.525),
    # meta-llama/Llama-3.1-70B config.json
    'llama3.1-70b': Model('Llama-3.1-70B', 70.6, 70.6, 80, 8192, 64, 8, 128, 128256, context=131072,
                          embed_b=1.051),
    # Qwen/Qwen3-235B-A22B config.json：94 层，4 个 KV 头，128 专家取 8，moe_intermediate_size 1536，
    # 单专家 3×4096×1536 = 18.87M，路由专家合计 227.1B
    'qwen3-235b-a22b': Model('Qwen3-235B-A22B', 235.1, 22.2, 94, 4096, 64, 4, 128, 151936,
                             context=40960, embed_b=0.622, experts=128, topk=8, moe_layers=94,
                             expert_b=0.018874),
    # deepseek-ai/DeepSeek-V3 config.json：kv_lora_rank 512，qk_rope_head_dim 64；
    # 前 3 层稠密，其余 58 层 256 路由专家取 8，moe_intermediate_size 2048，单专家 44.04M。
    # 官方 checkpoint 是 FP8（128×128 块缩放），不含 MTP 层
    'deepseek-v3': Model('DeepSeek-V3', 671, 37, 61, 7168, 128, 128, 128, 129280,
                         mla_kv_rank=512, mla_rope_dim=64, context=163840, embed_b=0.927,
                         experts=256, topk=8, moe_layers=58, expert_b=0.044040,
                         native_dtype='fp8'),
    # openai/gpt-oss-120b config.json 与模型卡（arXiv 2508.10925）：
    # 36 层，滑窗层与全注意力层交替各 18 层，窗口 128；8 个 KV 头，head_dim 64；128 专家取 4，
    # 单专家 3×2880×2880 加偏置 = 24.89M。模型卡口径：总参 116.83B，激活 5.13B（不含输入
    # embedding），MLP 114.71B；MoE 权重 MXFP4（4.25 bit/参数），其余 BF16，checkpoint 60.8 GiB
    'gpt-oss-120b': Model('gpt-oss-120b', 116.83, 5.13, 36, 2880, 64, 8, 64, 201088,
                          context=131072, experts=128, topk=4, moe_layers=36,
                          expert_b=0.024892, swa_layers=18, window=128, native_dtype='mxfp4'),
}


# 权重精度：(非专家部分字节/参数, 路由专家字节/参数)
# mxfp4 只量化路由专家，attention、embedding、router 保持 BF16，对应 gpt-oss 官方 checkpoint
WEIGHT_DTYPE = {
    'bf16': (2, 2),
    'fp8': (1, 1),
    'int4': (0.5, 0.5),
    'mxfp4': (2, 4.25 / 8),
}
KV_DTYPE = {'bf16': 2, 'fp8': 1}


def weight_bytes(m: Model, dtype: str) -> float:
    """整模型权重字节数（未切分）。"""
    dense_b, expert_b = WEIGHT_DTYPE[dtype]
    routed = m.routed_b * 1e9
    return (m.params_b * 1e9 - routed) * dense_b + routed * expert_b


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float            # HBM 带宽，TB/s
    bf16_tflops: float       # 稠密 BF16 Tensor Core 峰值
    fp8_tflops: float

    @property
    def ridge(self) -> float:
        """BF16 Roofline 拐点：每字节需要多少 FLOP 才能吃满算力。"""
        return self.bf16_tflops / self.bw_tbs


GPUS = {
    'a100-80g': GPU('A100 80GB SXM', 80, 2.039, 312, 0),
    'h100-sxm': GPU('H100 SXM', 80, 3.35, 989, 1979),
    'h200': GPU('H200', 141, 4.8, 989, 1979),
    'l40s': GPU('L40S', 48, 0.864, 362, 733),
    'rtx4090': GPU('RTX 4090', 24, 1.008, 165, 330),
}

# nvidia-smi --query-gpu=memory.total 的实际读数（MiB），比标称容量略小
SMI_MIB = {'a100-80g': 81920, 'h100-sxm': 81559, 'h200': 143771, 'l40s': 46068, 'rtx4090': 24564}
