"""启动参数解析：一条 --speculative-* 命令行，SGLang 最后到底用了哪组 (steps, topk, draft_tokens)。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  arg_groups/speculative_hook.py:124    handle_speculative_decoding：大写、别名、block verification、adaptive 的先后顺序
  arg_groups/speculative_hook.py:84     _resolve_speculative_algorithm_alias：NEXTN -> EAGLE
  arg_groups/speculative_hook.py:1113   _handle_eagle_family：max_running 48、MTP 架构自动填草稿路径、自动选参、
                                        rejection sampling 的约束、topk=1 时 draft_tokens 强制等于 steps+1、
                                        topk>1 且 page>1 的后端白名单
  arg_groups/speculative_hook.py:1326   _handle_ngram：draft_tokens 默认 12，topk = max_bfs_breadth
  arg_groups/speculative_hook.py:1433   _init_adaptive_speculative_params：adaptive 先于算法分支执行
  arg_groups/speculative_hook.py:1471   _auto_choose_speculative_params：按模型架构给默认值
  speculative/adaptive_spec_params.py   adaptive_unsupported_reason：哪些组合会把 adaptive 静默关掉
  mem_cache/allocation_sizing.py:28     get_alloc_len_per_decode：每个请求每步要预留多少 KV 槽位
  model_executor/pool_configurator.py:259  EAGLE / STANDALONE 草稿层的 KV 摊进目标模型每 token 的 cell

  python3 resolve_sim.py                     # resolve：十几条典型命令行的解析结果和告警
  python3 resolve_sim.py --scenario memory   # memory：投机解码多吃的 KV 槽位和 KV 池缩水
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))

from common.specs import MODELS  # noqa: E402
from adaptive_sim import resolve_candidate_steps  # noqa: E402

# _handle_eagle_family 里"MTP 层就在目标 checkpoint 里"的架构：不给草稿路径就用 model_path
MTP_IN_CKPT = {
    'DeepseekV32ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepseekV4ForCausalLM',
    'Glm4MoeForCausalLM', 'Glm4MoeLiteForCausalLM', 'GlmMoeDsaForCausalLM',
    'BailingMoeForCausalLM', 'BailingMoeV2ForCausalLM', 'BailingMoeV2_5ForCausalLM',
    'MistralLarge3ForCausalLM', 'PixtralForConditionalGeneration',
    'HYV3ForCausalLM', 'HYV4ForCausalLM', 'Qwen4ExpForConditionalGeneration',
}
AUTO_314 = {
    'DeepseekV32ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepseekV2ForCausalLM',
    'GptOssForCausalLM', 'Glm4MoeForCausalLM', 'Glm4MoeLiteForCausalLM', 'GlmMoeDsaForCausalLM',
    'BailingMoeForCausalLM', 'BailingMoeV2ForCausalLM', 'BailingMoeV2_5ForCausalLM',
    'MistralLarge3ForCausalLM', 'PixtralForConditionalGeneration',
    'MiMoV2ForCausalLM', 'MiMoV2FlashForCausalLM',
}
PAGE_TREE_BACKENDS = ('flashinfer', 'fa3', 'triton')

DEFAULTS = dict(
    algorithm=None, draft_model_path=None, num_steps=None, eagle_topk=None,
    num_draft_tokens=None, max_running_requests=None, use_rejection_sampling=False,
    use_block_verification=False, accept_threshold_single=1.0, accept_threshold_acc=1.0,
    adaptive=False, adaptive_config=None, enable_mixed_chunk=False, ngram_max_bfs_breadth=10,
    # 运行环境
    arch='LlamaForCausalLM', model_path='model', device='cuda', is_hip=False,
    attention_backend='flashinfer', page_size=1, dp_attention=False,
    enable_multi_layer_eagle=False, enable_two_batch_overlap=False, token_map=None,
)

MIXED_CHUNK_OK = {'EAGLE', 'EAGLE3', 'DFLASH', 'DSPARK'}   # spec_info.py:147 supports_mixed_chunk；STANDALONE 不在里面，NGRAM 在 _handle_ngram 里无条件关掉


def auto_choose(a):
    """_auto_choose_speculative_params。"""
    if a['algorithm'] == 'STANDALONE':
        return 3, 1, 4
    if a['arch'] == 'LlamaForCausalLM':
        return 5, 4, 8
    if a['arch'] in AUTO_314:
        return 3, 1, 4
    if a['arch'] in ('Grok1ForCausalLM', 'Grok1VForCausalLM'):
        return 5, 4, 8
    return 3, 1, 4


def adaptive_unsupported_reason(a):
    if a['algorithm'] not in ('EAGLE', 'EAGLE3'):
        return f"speculative_algorithm={a['algorithm']} (only EAGLE/EAGLE3 are supported)"
    if a['eagle_topk'] is not None and a['eagle_topk'] != 1:
        return f"speculative_eagle_topk={a['eagle_topk']} (only topk=1 is supported)"
    if a['dp_attention']:
        return 'attention DP is not supported'
    if a['enable_multi_layer_eagle']:
        return 'enable_multi_layer_eagle=True is not supported'
    if a['enable_two_batch_overlap']:
        return 'enable_two_batch_overlap=True is not supported'
    return None


def _eagle_family(a, log):
    if a['algorithm'] == 'STANDALONE' and a['dp_attention']:
        raise ValueError('Currently standalone speculative decoding does not support dp attention.')
    if a['max_running_requests'] is None:
        a['max_running_requests'] = 48
        log.append('WARN Max running requests is reset to 48 for speculative decoding.')
    if a['enable_mixed_chunk'] and a['algorithm'] not in MIXED_CHUNK_OK:
        a['enable_mixed_chunk'] = False
        log.append(f"WARN Mixed chunked prefill is disabled: {a['algorithm']} does not support it.")
    if a['arch'] in MTP_IN_CKPT:
        if a['draft_model_path'] is None:
            a['draft_model_path'] = a['model_path']
            log.append('INFO draft_model_path <- model_path（MTP 层在目标 checkpoint 里）')
        elif a['arch'] not in ('MistralLarge3ForCausalLM', 'PixtralForConditionalGeneration'):
            log.append('WARN DeepSeek MTP does not require setting speculative_draft_model_path.')
    if not a['adaptive'] and a['num_steps'] is None:
        if a['eagle_topk'] is not None or a['num_draft_tokens'] is not None:
            # 源码是一句 assert：只给 topk 或 draft_tokens、不给 steps，启动直接 AssertionError
            raise AssertionError('num_steps 为空时 eagle_topk 和 num_draft_tokens 也必须为空')
        a['num_steps'], a['eagle_topk'], a['num_draft_tokens'] = auto_choose(a)
        log.append(f"INFO auto params ({a['num_steps']}, {a['eagle_topk']}, {a['num_draft_tokens']})")
    if a['attention_backend'] == 'trtllm_mha' and a['eagle_topk'] > 1:
        raise ValueError('trtllm_mha backend only supports topk = 1 for speculative decoding.')
    if (a['is_hip'] and not a['use_rejection_sampling'] and a['algorithm'] == 'EAGLE'
            and a['token_map'] is None and a['eagle_topk'] == 1
            and a['accept_threshold_single'] == 1.0 and a['accept_threshold_acc'] == 1.0):
        a['use_rejection_sampling'] = True
        log.append('INFO ROCm: enabling speculative_use_rejection_sampling by default.')
    if a['use_rejection_sampling']:
        if a['algorithm'] not in ('EAGLE', 'EAGLE3'):
            raise NotImplementedError('--speculative-use-rejection-sampling is only supported for '
                                      'EAGLE / EAGLE3 / NEXTN')
        if a['eagle_topk'] != 1:
            raise ValueError('--speculative-use-rejection-sampling requires --speculative-eagle-topk=1.')
        if a['accept_threshold_single'] != 1.0 or a['accept_threshold_acc'] != 1.0:
            raise ValueError('--speculative-use-rejection-sampling is incompatible with '
                             '--speculative-accept-threshold-single / -acc')
    if a['eagle_topk'] == 1 and a['num_draft_tokens'] != a['num_steps'] + 1:
        log.append('WARN speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 '
                   'when speculative_eagle_topk == 1')
        a['num_draft_tokens'] = a['num_steps'] + 1
    if a['eagle_topk'] > 1 and a['page_size'] > 1 and a['attention_backend'] not in PAGE_TREE_BACKENDS:
        raise ValueError('speculative_eagle_topk > 1 with page_size > 1 is only supported on '
                         f'{PAGE_TREE_BACKENDS}')


def _ngram(a, log):
    if a['device'] not in ('cuda', 'cpu', 'xpu'):
        raise ValueError('Ngram speculative decoding only supports CUDA, CPU, or XPU devices.')
    if a['max_running_requests'] is None:
        a['max_running_requests'] = 48
        log.append('WARN Max running requests is reset to 48 for speculative decoding.')
    a['enable_mixed_chunk'] = False
    a['eagle_topk'] = a['ngram_max_bfs_breadth']
    if a['num_draft_tokens'] is None:
        a['num_draft_tokens'] = 12
        log.append('WARN speculative_num_draft_tokens is set to 12 by default for ngram.')
    if a['num_steps'] is None:
        a['num_steps'] = a['num_draft_tokens'] // a['eagle_topk']
    if a['eagle_topk'] > 1 and a['page_size'] > 1 and a['attention_backend'] != 'flashinfer':
        raise ValueError('ngram topk > 1 with page_size > 1 is only supported for flashinfer')
    if a['dp_attention']:
        raise ValueError('Currently ngram speculative decoding does not support dp attention.')


def resolve(**kw):
    """返回 (解析后的参数, 日志)。非法组合抛出和源码同类型的异常。"""
    a = dict(DEFAULTS)
    unknown = set(kw) - set(a)
    if unknown:
        raise TypeError(f'unknown args {unknown}')
    a.update(kw)
    log = []
    if a['algorithm'] is not None:
        a['algorithm'] = a['algorithm'].upper()
        if a['algorithm'] == 'NEXTN':
            a['algorithm'] = 'EAGLE'
            log.append('INFO NEXTN -> EAGLE')
    if a['use_block_verification']:
        if a['algorithm'] not in ('EAGLE', 'EAGLE3', 'DFLASH', 'DSPARK'):
            raise ValueError('--speculative-use-block-verification only supports EAGLE / EAGLE3 / '
                             'NEXTN / DFLASH / DSPARK.')
        if a['device'] != 'cuda':
            raise ValueError('--speculative-use-block-verification only supports CUDA or ROCm.')
        if a['algorithm'] in ('EAGLE', 'EAGLE3') and not a['use_rejection_sampling']:
            a['use_rejection_sampling'] = True
            log.append('INFO block verification -> use_rejection_sampling=True')
    # adaptive 在算法分支之前：它先把 steps 定下来，后面的自动选参就不会再碰
    if a['adaptive']:
        reason = adaptive_unsupported_reason(a)
        if reason:
            log.append(f'WARN speculative_adaptive disabled: {reason}. Falling back to static params.')
            a['adaptive'] = False
        else:
            cands = resolve_candidate_steps(a['adaptive_config'])
            if a['eagle_topk'] is None:
                a['eagle_topk'] = 1
            if a['num_steps'] is None:
                a['num_steps'] = cands[len(cands) // 2]
            if a['num_steps'] not in cands:
                raise ValueError(f"--speculative-num-steps={a['num_steps']} is not in the adaptive "
                                 f'config candidate_steps {cands}.')
            a['num_draft_tokens'] = a['num_steps'] + 1
            log.append(f"INFO adaptive: candidates {cands}, start steps={a['num_steps']}")
    if a['algorithm'] in ('EAGLE', 'EAGLE3', 'STANDALONE'):
        _eagle_family(a, log)
    elif a['algorithm'] == 'NGRAM':
        _ngram(a, log)
    return a, log


def alloc_len_per_decode(steps, topk, ndt, page_size=1, has_draft_kv=True):
    """allocation_sizing.get_alloc_len_per_decode。调度器每步按 2 倍这个数预留（double buffer）。"""
    steps = steps or 1
    topk = topk or 1
    if page_size == 1 or topk == 1 or not has_draft_kv:
        return max(steps * topk, ndt)
    pages_per_branch = ((page_size - 1) + steps + page_size - 1) // page_size
    return max(pages_per_branch * page_size * topk, ndt)


CASES = [
    ('Llama-3.1-8B + EAGLE3，什么都不填', dict(algorithm='eagle3', draft_model_path='d')),
    ('DeepSeek-V3 + NEXTN，什么都不填', dict(algorithm='nextn', arch='DeepseekV3ForCausalLM')),
    ('Qwen3 + EAGLE3，什么都不填', dict(algorithm='EAGLE3', arch='Qwen3ForCausalLM', draft_model_path='d')),
    ('文档示例 3/4/16', dict(algorithm='EAGLE3', draft_model_path='d', num_steps=3, eagle_topk=4,
                         num_draft_tokens=16)),
    ('topk=1 却写 draft_tokens=16', dict(algorithm='EAGLE', draft_model_path='d', num_steps=3,
                                       eagle_topk=1, num_draft_tokens=16)),
    ('只写 topk 不写 steps', dict(algorithm='EAGLE', draft_model_path='d', eagle_topk=1)),
    ('rejection sampling + topk 4', dict(algorithm='EAGLE', draft_model_path='d', num_steps=3,
                                         eagle_topk=4, num_draft_tokens=8, use_rejection_sampling=True)),
    ('block verification，topk 1', dict(algorithm='EAGLE3', draft_model_path='d', num_steps=3,
                                       eagle_topk=1, num_draft_tokens=4, use_block_verification=True)),
    ('STANDALONE + 文档示例 4/2/7', dict(algorithm='STANDALONE', draft_model_path='d', num_steps=4,
                                        eagle_topk=2, num_draft_tokens=7)),
    ('topk 4 + page 64 + flashmla', dict(algorithm='EAGLE', arch='DeepseekV3ForCausalLM', num_steps=3,
                                          eagle_topk=4, num_draft_tokens=8, page_size=64,
                                          attention_backend='flashmla')),
    ('adaptive，默认配置', dict(algorithm='EAGLE', draft_model_path='d', adaptive=True)),
    ('adaptive + steps 4', dict(algorithm='EAGLE', draft_model_path='d', adaptive=True, num_steps=4)),
    ('adaptive + topk 4（被静默关掉）', dict(algorithm='EAGLE', draft_model_path='d', adaptive=True,
                                         num_steps=5, eagle_topk=4, num_draft_tokens=8)),
    ('NGRAM，什么都不填', dict(algorithm='ngram')),
    ('NGRAM，draft 16 bfs 10（文档示例）', dict(algorithm='NGRAM', num_draft_tokens=16,
                                             ngram_max_bfs_breadth=10)),
    ('ROCm + EAGLE topk 1', dict(algorithm='EAGLE', draft_model_path='d', num_steps=3, eagle_topk=1,
                                num_draft_tokens=4, is_hip=True)),
]


def scenario_resolve():
    out = {}
    for name, kw in CASES:
        try:
            a, log = resolve(**kw)
            res = (a['algorithm'], a['num_steps'], a['eagle_topk'], a['num_draft_tokens'])
            out[name] = dict(ok=True, params=res, log=log, args=a)
            print(f'{name}\n  -> {res[0]} steps={res[1]} topk={res[2]} draft_tokens={res[3]} '
                  f"max_running={a['max_running_requests']} rejection={a['use_rejection_sampling']} "
                  f"adaptive={a['adaptive']}")
        except (ValueError, AssertionError, NotImplementedError) as e:
            out[name] = dict(ok=False, error=f'{type(e).__name__}: {e}')
            print(f'{name}\n  -> 启动失败 {type(e).__name__}: {e}')
            continue
        for line in log:
            print(f'     {line}')
    print("""
  三条最常踩的：
  1. topk=1 时 num_draft_tokens 只能是 steps+1，填别的会被改掉，日志里只有一行 WARN。
  2. 只填一部分参数（比如只给 topk）会在 assert 上崩；要么全填三个，要么全不填交给自动选参。
  3. adaptive 只认 EAGLE/EAGLE3 + topk 1，组合不对时只打一行 WARN 就退回静态参数，服务照常启动。""")
    return out


def scenario_memory(model='llama3.1-8b', gpu='h100-sxm'):
    from sim_ledger import ledger
    m = MODELS[model]
    print('每个请求每步预留的 KV 槽位（get_alloc_len_per_decode，调度器再乘 2 做 double buffer）')
    print(f'  {"(steps,topk,ndt)":18s} {"page=1":>8s} {"page=16":>8s} {"page=64":>8s}')
    rows = {}
    for cfg in [(3, 1, 4), (5, 4, 8), (3, 4, 16), (6, 4, 32), (5, 8, 64)]:
        vals = [alloc_len_per_decode(*cfg, page_size=p) for p in (1, 16, 64)]
        rows[cfg] = vals
        print(f'  {str(cfg):18s}' + ''.join(f'{v:>9d}' for v in vals))
    L = ledger(model, gpu, tp=1, weight_dtype='bf16', kv_dtype='bf16')
    tokens = L['tokens']
    draft_layers = 1
    cell_spec = L['cell'] * (1 + draft_layers / m.layers)
    tokens_spec = int(L['rest'] * (1 << 30) // cell_spec)
    print(f'\n{m.name} 单卡 H100，KV 池：不开投机 {tokens:,} token；EAGLE 1 层草稿摊进 cell 后 {tokens_spec:,} token'
          f'（少 {100 * (1 - tokens_spec / tokens):.1f}%，即 1/{m.layers + 1}）')
    print('  48 个并发请求（投机解码默认的 max_running_requests）在途预留：')
    reserve = {}
    for cfg in [(3, 1, 4), (3, 4, 16)]:
        for p in (1, 64):
            r = 48 * 2 * alloc_len_per_decode(*cfg, page_size=p)
            reserve[(cfg, p)] = r
            print(f'    {str(cfg):12s} page={p:<3d} {r:>7,} token，占池子 {100 * r / tokens_spec:.2f}%')
    print("""
  草稿模型自己的 KV 不另开池子：EAGLE 草稿层直接用目标模型分配器给的下标，所以每 token 的 cell
  按草稿层数 / 目标层数等比例变大。1 层草稿对 32 层目标只贵 3%；STANDALONE 的小模型如果有 28 层，
  cell 几乎翻倍，KV 池直接腰斩（源码注释里的前提是草稿和目标每层 KV 尺寸一样）。
  topk>1 叠加大 page 时，每个分支都要按整页预留：(5,4,8) 在 page 64 下每步预留 512 个槽位，
  是 page 1 的 25 倍；(3,4,16) 配 page 64，48 个请求的在途预留吃掉 12% 的 KV 池。""")
    return dict(alloc=rows, tokens=tokens, tokens_spec=tokens_spec, reserve=reserve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='resolve', choices=['resolve', 'memory'])
    ap.add_argument('--model', default='llama3.1-8b')
    ap.add_argument('--gpu', default='h100-sxm')
    a = ap.parse_args()
    if a.scenario == 'resolve':
        scenario_resolve()
    else:
        scenario_memory(a.model, a.gpu)


if __name__ == '__main__':
    main()
