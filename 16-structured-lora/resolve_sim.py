"""启动参数解析：结构化输出和 LoRA 相关的命令行，SGLang 最后改写了哪些参数、拒掉哪些组合。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  arg_groups/serving_hook.py:239              handle_grammar_backend：没指定就用 xgrammar
  constrained/base_grammar_backend.py:436     grammar_backend=none 加 strict thinking 直接拒绝
  arg_groups/speculative_hook.py:452          _handle_uno：UNO 不能和 strict thinking、Multi-LoRA 同开
  managers/tokenizer_manager.py:905           请求带 max_thinking_tokens 但没开 strict thinking 时返回错误
  arg_groups/lora_hook.py:33                  check_lora_server_args：enable_lora 推断、overlap 上限、
                                              lora_paths 解析、target modules、max_loaded、chunk、drain
  arg_groups/lora_hook.py:20                  _check_v2_lora_modes：triton_v2 不支持的并行模式
  arg_groups/lora_hook.py:240                 check_lora_speculative_compatibility：和投机解码的组合
  lora/lora_manager.py:378                    启动时加载 pinned adapter：最多 max_loras_per_batch - 1 个

  python3 resolve_sim.py                      # resolve：十几条典型命令行的解析结果和告警
  python3 resolve_sim.py --scenario request   # request：请求级参数（json_schema / max_thinking_tokens）在不同启动参数下的命运
"""
import argparse

ENV = dict(                          # environ.py
    SGLANG_EXPERIMENTAL_LORA_OPTI=False,
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM=False,
    SGLANG_RAGGED_VERIFY_MODE='static',
    SGLANG_GRAMMAR_POLL_INTERVAL=0.005,
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS=10000,
    SGLANG_ENABLE_DELAY_SAMPLE=False,
)

DEFAULTS = dict(
    grammar_backend=None, reasoning_parser=None, enable_strict_thinking=False,
    enable_lora=None, lora_paths=None, max_loras_per_batch=8, max_loaded_loras=None,
    max_lora_rank=None, lora_target_modules=None, lora_backend='csgmv', lora_eviction_policy='lru',
    max_lora_chunk_size=16, lora_drain_wait_threshold=0.0, enable_lora_overlap_loading=None,
    moe_runner_backend='auto', speculative_moe_runner_backend=None,
    speculative_algorithm=None, speculative_adaptive=False, speculative_num_draft_tokens=None,
    speculative_draft_model_path=None, uno_lora_path=None, enable_deterministic_inference=False,
    device='cuda', attn_dp_size=1, enable_pdmux=False, enable_two_batch_overlap=False, is_hip=False,
)

GRAMMAR_BACKENDS = ('xgrammar', 'outlines', 'llguidance', 'none')     # arg_groups/choices.py:131
LORA_BACKENDS = ('triton', 'csgmv', 'ascend', 'torch_native', 'triton_v2')
LORA_SPEC = ('EAGLE', 'EAGLE3', 'DFLASH', 'DSPARK')


class StructArgError(ValueError):
    pass


def _v2_modes(a, env, option):
    """lora_hook.py:20。"""
    for bad, mode in ((a['attn_dp_size'] > 1, 'DP-attention with --attn-dp-size > 1'),
                      (a['enable_pdmux'], 'PD-multiplexing'),
                      (a['enable_two_batch_overlap'], 'two-batch overlap'),
                      (env['SGLANG_EXPERIMENTAL_LORA_OPTI'], 'SGLANG_EXPERIMENTAL_LORA_OPTI'),
                      (a['is_hip'], 'ROCm')):
        if bad:
            raise StructArgError(f'{option} does not yet support {mode}')


def _spec_compat(a, env):
    """lora_hook.py:240。NEXTN 在这之前已经被改名成 EAGLE。"""
    alg = a['speculative_algorithm']
    if alg in ('NGRAM', None):
        return
    if alg not in LORA_SPEC:
        raise StructArgError('LoRA is only compatible with NGRAM, EAGLE, NEXTN, EAGLE3, DFLASH, or DSPARK '
                             f'speculative decoding, not {alg}.')
    mode = env['SGLANG_RAGGED_VERIFY_MODE']
    for bad, why in ((alg == 'DSPARK' and mode != 'static', f'does not support SGLANG_RAGGED_VERIFY_MODE={mode!r}'),
                     (a['speculative_adaptive'], 'does not support --speculative-adaptive'),
                     ('experimental_sgl_trtllm' in (a['moe_runner_backend'], a['speculative_moe_runner_backend']),
                      'does not support the experimental_sgl_trtllm MoE runner'),
                     (env['SGLANG_ENABLE_OVERLAP_PLAN_STREAM'], 'does not support SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1')):
        if bad:
            raise StructArgError(f'LoRA with EAGLE/NEXTN/EAGLE3 speculative decoding {why}.')


def parse_lora_paths(paths):
    """lora_hook.py:76-136。返回 [(name, path, pinned)]。"""
    if paths is None:
        return []
    if isinstance(paths, dict):
        return [(k, v, False) for k, v in paths.items()]
    if not isinstance(paths, list):
        raise StructArgError(f'Invalid type for --lora-paths: {type(paths)}. Expected a list or a dictionary.')
    out = []
    for p in paths:
        if isinstance(p, str):
            name, path = p.split('=', 1) if '=' in p else (p, p)
            out.append((name, path, False))
        elif isinstance(p, dict):
            if 'lora_name' not in p or 'lora_path' not in p:
                raise StructArgError("When providing LoRA paths as a list of dict, each dict should contain "
                                     f"'lora_name' and 'lora_path' keys. Got: {p}")
            out.append((p['lora_name'], p['lora_path'], p.get('pinned', False)))
        else:
            raise StructArgError(f'Invalid type for item in --lora-paths list: {type(p)}. Expected a string or a dictionary.')
    return out


def _uno(a):
    """speculative_hook.py:452，只保留和本章相关的几条。"""
    if not a['device'].startswith('cuda'):
        raise StructArgError('UNO only supports CUDA.')
    if a['speculative_draft_model_path'] is not None:
        raise StructArgError('UNO reuses the target model and does not accept --speculative-draft-model-path.')
    if a['uno_lora_path'] is None:
        raise StructArgError('UNO requires --uno-lora-path.')
    if a['enable_deterministic_inference']:
        raise StructArgError('UNO does not support --enable-deterministic-inference because its sampling path '
                             'does not use per-request seeds.')
    if a['enable_strict_thinking']:
        raise StructArgError('UNO does not support --enable-strict-thinking because it requires grammar decoding.')
    w = a['speculative_num_draft_tokens']
    if w is None or int(w) < 1:
        raise StructArgError('UNO requires --speculative-num-draft-tokens to be a positive integer denoting '
                             'the linear width or tree verify width Q.')
    if a['enable_lora'] or a['lora_paths']:
        raise StructArgError('UNO does not support public Multi-LoRA serving.')
    a['enable_lora_overlap_loading'] = False


def resolve(env=None, **kw):
    bad = set(kw) - set(DEFAULTS)
    if bad:
        raise TypeError(f'unknown args: {sorted(bad)}')
    env = dict(ENV, **(env or {}))
    a = dict(DEFAULTS, **kw)
    warns, set_env = [], {}

    # serving_hook.py:239
    if a['grammar_backend'] is None:
        a['grammar_backend'] = 'xgrammar'
    if a['grammar_backend'] not in GRAMMAR_BACKENDS:
        raise StructArgError(f"Invalid grammar backend: {a['grammar_backend']}")
    if a['speculative_algorithm'] == 'UNO':
        _uno(a)

    # lora_hook.py:33
    if a['max_loras_per_batch'] <= 0:
        raise StructArgError('max_loras_per_batch must be positive')
    if a['lora_paths']:
        if a['enable_lora'] is None:
            a['enable_lora'] = True
            warns.append('--enable-lora is set to True because --lora-paths is provided.')
        elif a['enable_lora'] is False:
            warns.append('--enable-lora is set to False, any provided lora_paths will be ignored.')
    if a['enable_lora']:
        if a['lora_backend'] not in LORA_BACKENDS:
            raise StructArgError(f"invalid choice: {a['lora_backend']}")
        if a['lora_backend'] == 'triton_v2' and not a['moe_runner_backend'].startswith('lora'):
            _v2_modes(a, env, '--lora-backend triton_v2')
        if a['enable_lora_overlap_loading'] is None:
            a['enable_lora_overlap_loading'] = False
        if a['enable_lora_overlap_loading']:
            lim = a['max_loras_per_batch'] * 2
            if a['max_loaded_loras'] is None or a['max_loaded_loras'] > lim:
                raise StructArgError('Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, '
                                     f'so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: {lim}')
        _spec_compat(a, env)
        a['lora_paths'] = parse_lora_paths(a['lora_paths'])
        if a['lora_target_modules']:
            a['lora_target_modules'] = set(a['lora_target_modules'])
            if 'all' in a['lora_target_modules'] and len(a['lora_target_modules']) != 1:
                raise StructArgError("If 'all' is specified in --lora-target-modules, it should be the only module specified.")
        if not (a['lora_paths'] or (a['max_lora_rank'] and a['lora_target_modules'])):
            raise StructArgError('When no initial --lora-paths is provided, you need to specify both --max-lora-rank '
                                 'and --lora-target-modules for LoRA initialization.')
        if a['max_loaded_loras'] is not None:
            if a['max_loaded_loras'] < a['max_loras_per_batch']:
                raise StructArgError('max_loaded_loras should be greater than or equal to max_loras_per_batch. '
                                     f"max_loaded_loras={a['max_loaded_loras']}, max_loras_per_batch={a['max_loras_per_batch']}")
            if len(a['lora_paths']) > a['max_loaded_loras']:
                raise StructArgError('The number of LoRA paths should not exceed max_loaded_loras. '
                                     f"max_loaded_loras={a['max_loaded_loras']}, lora_paths={len(a['lora_paths'])}")
        c = a['max_lora_chunk_size']
        if c is not None and not (16 <= c <= 128 and c & (c - 1) == 0):
            raise StructArgError('--max-lora-chunk-size must be a power of 2 between 16 and 128.')
        if a['lora_drain_wait_threshold'] < 0.0:
            raise StructArgError('--lora-drain-wait-threshold must be non-negative.')
        # lora_manager.py:378：启动时逐个 load_lora_adapter，pinned 占满 mlpb-1 个槽位后再 pin 就失败
        n_pinned = 0
        for name, _, pinned in a['lora_paths']:
            if pinned and n_pinned >= a['max_loras_per_batch'] - 1:
                raise StructArgError(f'Failed to load LoRA adapter {name} as a pinned adapter. It is not allowed to pin '
                                     'all slots in the LoRA memory pool to avoid starvation for unpinned adapters and base models.')
            n_pinned += int(pinned)
        if a['attn_dp_size'] > 1 and any(not p for _, _, p in a['lora_paths']):
            raise StructArgError('DP-attention LoRA requires pinned adapters')     # lora_manager.py:341

    # base_grammar_backend.py:436，模型加载后建 grammar backend 时才检查
    if a['enable_strict_thinking'] and a['grammar_backend'] == 'none':
        raise StructArgError("--enable-strict-thinking requires a grammar backend that supports token filtering, "
                             "but grammar_backend='none' was specified.")
    if a['enable_strict_thinking'] and not a['reasoning_parser']:
        warns.append('没有 --reasoning-parser，think_end_ids 为空，ReasonerGrammarBackend 不会挂上，strict thinking 不生效')
    return a, set_env, warns


def request_fate(a, json_schema=False, max_thinking_tokens=None):
    """单条请求在给定启动参数下的结果。tokenizer_manager.py:905 + grammar_manager.py:148。"""
    if max_thinking_tokens is not None and not a['enable_strict_thinking']:
        return '400：max_thinking_tokens requires the server to be launched with --enable-strict-thinking'
    if json_schema and a['grammar_backend'] == 'none':
        return '中止：Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported ' \
               'when the server is launched with --grammar-backend none'
    parts = []
    if json_schema:
        parts.append(f"{a['grammar_backend']} 编译 schema，进 grammar 队列")
    if a['enable_strict_thinking'] and a['reasoning_parser']:
        parts.append('思考段按 think_end 序列约束' + (f'，{max_thinking_tokens} token 后强制收尾' if max_thinking_tokens else ''))
    return '；'.join(parts) or '普通采样'


P = dict(lora_paths=['sql=/ckpt/sql', 'chat=/ckpt/chat'])
CASES = [
    ('默认', dict(), None),
    ('grammar none', dict(grammar_backend='none'), None),
    ('strict + none', dict(grammar_backend='none', enable_strict_thinking=True, reasoning_parser='qwen3'), None),
    ('strict 无 parser', dict(enable_strict_thinking=True), None),
    ('lora-paths 两个', dict(P), None),
    ('enable-lora False + paths', dict(P, enable_lora=False), None),
    ('只开 enable-lora', dict(enable_lora=True), None),
    ('动态加载 r64 all', dict(enable_lora=True, max_lora_rank=64, lora_target_modules=['all']), None),
    ('all + q_proj', dict(enable_lora=True, max_lora_rank=64, lora_target_modules=['all', 'q_proj']), None),
    ('overlap 无 max_loaded', dict(P, enable_lora_overlap_loading=True), None),
    ('overlap max_loaded 16', dict(P, enable_lora_overlap_loading=True, max_loaded_loras=16), None),
    ('max_loaded 4 < mlpb 8', dict(P, max_loaded_loras=4), None),
    ('chunk 48', dict(P, max_lora_chunk_size=48), None),
    ('pinned 8 个, mlpb 8', dict(lora_paths=[dict(lora_name=f'a{i}', lora_path=f'/c/{i}', pinned=True) for i in range(8)]), None),
    ('EAGLE3 + LoRA', dict(P, speculative_algorithm='EAGLE3'), None),
    ('EAGLE + LoRA + plan stream', dict(P, speculative_algorithm='EAGLE'), dict(SGLANG_ENABLE_OVERLAP_PLAN_STREAM=True)),
    ('STANDALONE + LoRA', dict(P, speculative_algorithm='STANDALONE'), None),
    ('triton_v2 + DPA', dict(P, lora_backend='triton_v2', attn_dp_size=2), None),
    ('UNO + strict', dict(speculative_algorithm='UNO', uno_lora_path='/c/uno', speculative_num_draft_tokens=4,
                          enable_strict_thinking=True, reasoning_parser='qwen3'), None),
    ('UNO + lora-paths', dict(P, speculative_algorithm='UNO', uno_lora_path='/c/uno', speculative_num_draft_tokens=4), None),
]
SHOW = ['grammar_backend', 'enable_lora', 'enable_lora_overlap_loading']
SHORT = dict(grammar_backend='grammar', enable_lora='enable_lora', enable_lora_overlap_loading='overlap')


def scenario_resolve():
    out = {}
    for name, kw, env in CASES:
        try:
            a, set_env, warns = resolve(env, **kw)
            out[name] = (a, set_env, warns)
            given = dict(DEFAULTS, **kw)
            changed = [f'{SHORT[k]}: {given[k]} -> {a[k]}' for k in SHOW if a[k] != given[k]]
            if set_env:
                changed += [f'{k}={v}' for k, v in set_env.items()]
            print(f'{name:<30}{"  ".join(changed) or "不改写"}')
            for w in warns:
                print(f'{"":<32}WARN {w}')
        except StructArgError as e:
            out[name] = e
            print(f'{name:<30}启动失败：{e}')
    return out


def scenario_request():
    launches = [('默认', dict()), ('grammar none', dict(grammar_backend='none')),
                ('strict + qwen3', dict(enable_strict_thinking=True, reasoning_parser='qwen3'))]
    reqs = [('普通请求', dict()), ('json_schema', dict(json_schema=True)),
            ('max_thinking 512', dict(max_thinking_tokens=512)),
            ('json + max_thinking', dict(json_schema=True, max_thinking_tokens=512))]
    res = {}
    for ln, lkw in launches:
        a, _, _ = resolve(**lkw)
        print(f'启动参数：{ln}')
        for rn, rkw in reqs:
            res[(ln, rn)] = request_fate(a, **rkw)
            print(f'  {rn:<20}{res[(ln, rn)]}')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='resolve', choices=['resolve', 'request'])
    a = ap.parse_args()
    scenario_resolve() if a.scenario == 'resolve' else scenario_request()


if __name__ == '__main__':
    main()
