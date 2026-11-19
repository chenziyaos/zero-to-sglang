"""启动参数解析：一条 --disaggregation-mode 命令行，SGLang 最后改写了哪些参数、拒掉哪些组合、传输线程开几个。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  arg_groups/disagg.py:55-185                 PD 相关参数和默认值
  arg_groups/pd_disaggregation_hook.py:23     handle_pd_disaggregation：mooncake_tcp 改写、host receive 检查、
                                              decode 侧 radix 开关、extra_slots 默认值、staging 后端检查、
                                              role switch 拒绝清单
  arg_groups/pd_disaggregation_hook.py:199    _validate_prefill_complete：prefill_complete 不支持的特性
  disaggregation/mooncake/conn.py:328         传输线程池：min(max(4, int(0.5 × cpu) // 8), 12)
  environ.py:755-782                          队列数、bootstrap / waiting 超时、心跳

  python3 resolve_sim.py                      # resolve：十几条典型命令行的解析结果和告警
  python3 resolve_sim.py --scenario threads   # threads：不同 CPU 核数下传输线程池多大，和文档公式对比
"""
import argparse

ENV = dict(                          # environ.py
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=None,
    SGLANG_DISAGGREGATION_QUEUE_SIZE=4,
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300,
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5.0,
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=2,
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300,
    SGLANG_DISAGG_STAGING_BUFFER=False,
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096,
    SGLANG_RUST_SERVER=False,
)

DEFAULTS = dict(
    disaggregation_mode='null', disaggregation_decode_allocation_policy='early',
    disaggregation_transfer_backend='mooncake', disaggregation_ib_device=None,
    disaggregation_bootstrap_port=8998, disaggregation_decode_enable_radix_cache=False,
    disaggregation_decode_retraction_backup=None, num_reserved_decode_tokens=512,
    disaggregation_decode_extra_slots=None, disaggregation_decode_polling_interval=1,
    disaggregation_decode_host_receive_threshold=1.0, enable_pd_role_switch=False,
    optimistic_prefill_attempts=0, disable_radix_cache=False, max_running_requests=None,
    tp_size=1, dp_size=1, enable_dp_attention=False, pp_size=1, attn_cp_size=1, dcp_size=1,
    ep_size=1, moe_a2a_backend='none', speculative_algorithm=None, enable_hisparse=False,
    enable_hierarchical_cache=False, hicache_storage_backend=None, hicache_write_policy='write_through',
    enable_pdmux=False, decoupled_spec_role='null', encoder_only=False, language_only=False,
)


class PDArgError(ValueError):
    pass


def _validate_prefill_complete(a, env):
    """pd_disaggregation_hook.py:199。"""
    bad = []
    if a['disaggregation_mode'] not in ('prefill', 'decode'):
        bad.append('aggregated serving')
    if a['disaggregation_transfer_backend'] != 'mooncake':
        bad.append('transfer backends other than Mooncake')
    if a['pp_size'] != 1 or a['attn_cp_size'] != 1 or a['dcp_size'] != 1:
        bad.append('pipeline or attention context parallelism')
    attn_dp = a['dp_size'] if a['enable_dp_attention'] else 1
    if a['tp_size'] != attn_dp:
        bad.append('attention tensor parallelism greater than one')
    if a['enable_pdmux'] or a['decoupled_spec_role'] != 'null':
        bad.append('PD multiplexing or decoupled speculation')
    if a['enable_pd_role_switch']:
        bad.append('runtime PD role switching')
    if a['encoder_only'] or a['language_only']:
        bad.append('encoder disaggregation')
    if a['enable_hisparse']:
        bad.append('HiSparse')
    if a['disaggregation_decode_host_receive_threshold'] < 1:
        bad.append('decode host KV buffering')
    if a['disaggregation_mode'] == 'decode' and (
            a['disaggregation_decode_enable_radix_cache'] or a['enable_hierarchical_cache']):
        bad.append('decode radix cache or decode HiCache')
    if (a['disaggregation_mode'] == 'prefill' and a['enable_hierarchical_cache']
            and (a['hicache_storage_backend'] is not None or a['hicache_write_policy'] != 'write_back')):
        bad.append('prefill HiCache other than L2 write_back')
    if env['SGLANG_DISAGG_STAGING_BUFFER']:
        bad.append('staging buffers')
    if env['SGLANG_RUST_SERVER']:
        bad.append('the Rust bootstrap registry')
    if bad:
        raise PDArgError('--disaggregation-decode-allocation-policy prefill_complete does not yet support '
                         + ', '.join(bad))


def resolve(env=None, **kw):
    """返回 (解析后的参数, 设置的环境变量, 告警列表)。抛 PDArgError 表示启动失败。"""
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown args {sorted(unknown)}')
    env = dict(ENV, **(env or {}))
    a = dict(DEFAULTS, **kw)
    set_env, warns = {}, []
    mode = a['disaggregation_mode']

    if a['disaggregation_transfer_backend'] == 'mooncake_tcp':
        set_env['MC_FORCE_TCP'] = '1'
        a['disaggregation_transfer_backend'] = 'mooncake'
        a['disaggregation_ib_device'] = None

    if a['disaggregation_decode_allocation_policy'] == 'prefill_complete':
        _validate_prefill_complete(a, env)
        if mode == 'prefill' and a['optimistic_prefill_attempts'] == 0:
            a['optimistic_prefill_attempts'] = 1

    if mode == 'prefill' and a['dcp_size'] > 1:
        warns.append('DCP on a PD prefill server ... usually adds communication overhead')

    thr = a['disaggregation_decode_host_receive_threshold']
    if not 0 <= thr <= 1:
        raise PDArgError('--disaggregation-decode-host-receive-threshold must be between 0 and 1')
    if thr < 1:
        if a['enable_hisparse'] or a['enable_pd_role_switch']:
            raise PDArgError('Decode host receive does not yet support HiSparse or role switching')
        if a['disaggregation_decode_retraction_backup'] == 'cpu_tensor':
            raise PDArgError('Decode host KV buffering requires host_pool retraction')
        a['disaggregation_decode_retraction_backup'] = 'host_pool'

    if mode == 'decode' and a['dcp_size'] > 1 and a['disaggregation_transfer_backend'] not in (
            'mooncake', 'nixl', 'mori', 'ascend', 'fake'):
        raise PDArgError('PD decode DCP requires --disaggregation-transfer-backend mooncake, nixl, mori, '
                         f'ascend, or fake ..., got {a["disaggregation_transfer_backend"]!r}.')

    if mode == 'decode':
        if a['disaggregation_decode_enable_radix_cache']:
            if a['enable_hisparse']:
                raise PDArgError('--disaggregation-decode-enable-radix-cache is incompatible with --enable-hisparse')
            if a['disaggregation_transfer_backend'] == 'fake':
                raise PDArgError('--disaggregation-decode-enable-radix-cache is incompatible with '
                                 '--disaggregation-transfer-backend fake')
            if a['enable_dp_attention']:
                warns.append('EXPERIMENTAL: Decode radix cache with DP attention.')
            a['disable_radix_cache'] = False
            warns.append('EXPERIMENTAL: Radix cache is enabled for decode server')
        else:
            a['disable_radix_cache'] = True
            warns.append('Radix cache is disabled for decode server')
        if a['disaggregation_decode_extra_slots'] is None:
            extra = 0
            if a['max_running_requests'] is not None:
                dp = a['dp_size'] if a['enable_dp_attention'] else 1
                per_worker = max(1, a['max_running_requests'] // dp)
                if per_worker <= 32:
                    extra = per_worker * 2
            a['disaggregation_decode_extra_slots'] = extra
    elif mode == 'prefill':
        if a['disaggregation_transfer_backend'] == 'fake':
            raise PDArgError("Prefill server does not support 'fake' as the transfer backend")

    if mode in ('prefill', 'decode'):
        if env['SGLANG_DISAGG_STAGING_BUFFER'] and a['disaggregation_transfer_backend'] not in ('mooncake', 'nixl'):
            raise PDArgError("SGLANG_DISAGG_STAGING_BUFFER requires disaggregation_transfer_backend='mooncake' "
                             f"or 'nixl', got '{a['disaggregation_transfer_backend']}'.")
        if a['enable_pd_role_switch']:
            bad = []
            if a['ep_size'] > 1:
                bad.append(f'expert parallelism (--ep-size {a["ep_size"]})')
            if a['moe_a2a_backend'] != 'none':
                bad.append(f'MoE all-to-all (--moe-a2a-backend {a["moe_a2a_backend"]})')
            if a['pp_size'] > 1:
                bad.append(f'pipeline parallelism (--pp-size {a["pp_size"]})')
            if a['dcp_size'] > 1:
                bad.append(f'decode context parallelism (--dcp-size {a["dcp_size"]})')
            if a['speculative_algorithm'] is not None:
                bad.append(f'speculative decoding (--speculative-algorithm {a["speculative_algorithm"]})')
            if bad:
                raise PDArgError('--enable-pd-role-switch does not rebuild role-specific state for the following '
                                 'features: ' + ', '.join(bad) + '.')
    return a, set_env, warns


def thread_pool(cpu, env_override=None, queue_size=4):
    """mooncake/conn.py:328。返回 (线程数, 文档公式的值)。线程数小于队列数时启动断言失败。"""
    n = env_override if env_override is not None else min(max(4, int(0.5 * cpu) // 8), 12)
    doc = min(max(4, int(0.75 * cpu) // 8), 12)
    if n < queue_size:
        raise PDArgError(f'The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={n} must be '
                         f'greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={queue_size}.')
    return n, doc


P, D = dict(disaggregation_mode='prefill'), dict(disaggregation_mode='decode')
CASES = [
    ('P 默认', dict(P), None),
    ('D 默认', dict(D), None),
    ('D 开 radix', dict(D, disaggregation_decode_enable_radix_cache=True), None),
    ('D 小批 max_running 16', dict(D, max_running_requests=16), None),
    ('D max_running 128', dict(D, max_running_requests=128), None),
    ('D DP8 max_running 256', dict(D, max_running_requests=256, dp_size=8, enable_dp_attention=True, tp_size=8), None),
    ('P mooncake_tcp', dict(P, disaggregation_transfer_backend='mooncake_tcp', disaggregation_ib_device='mlx5_0'), None),
    ('P fake 后端', dict(P, disaggregation_transfer_backend='fake'), None),
    ('P prefill_complete TP1', dict(P, disaggregation_decode_allocation_policy='prefill_complete'), None),
    ('P prefill_complete TP4', dict(P, disaggregation_decode_allocation_policy='prefill_complete', tp_size=4), None),
    ('D prefill_complete + radix', dict(D, disaggregation_decode_allocation_policy='prefill_complete',
                                        disaggregation_decode_enable_radix_cache=True), None),
    ('P prefill_complete + staging', dict(P, disaggregation_decode_allocation_policy='prefill_complete'),
     dict(SGLANG_DISAGG_STAGING_BUFFER=True)),
    ('P nixl + staging', dict(P, disaggregation_transfer_backend='nixl'), dict(SGLANG_DISAGG_STAGING_BUFFER=True)),
    ('P mori + staging', dict(P, disaggregation_transfer_backend='mori'), dict(SGLANG_DISAGG_STAGING_BUFFER=True)),
    ('D role switch + EP8', dict(D, enable_pd_role_switch=True, ep_size=8, moe_a2a_backend='deepep'), None),
    ('D host receive 0.9', dict(D, disaggregation_decode_host_receive_threshold=0.9), None),
]
SHOW = ['disaggregation_transfer_backend', 'disaggregation_ib_device', 'disable_radix_cache', 'disaggregation_decode_extra_slots',
        'optimistic_prefill_attempts', 'disaggregation_decode_retraction_backup']
SHORT = dict(disaggregation_transfer_backend='backend', disaggregation_ib_device='ib_device', disable_radix_cache='no_radix',
             disaggregation_decode_extra_slots='extra_slots', optimistic_prefill_attempts='optimistic',
             disaggregation_decode_retraction_backup='backup')


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
        except PDArgError as e:
            out[name] = e
            print(f'{name:<30}启动失败：{e}')
    return out


def scenario_threads():
    print(f'{"CPU 核数":>8}{"代码里的线程数":>14}{"文档公式":>10}')
    res = {}
    for cpu in (32, 64, 96, 128, 192, 224, 256):
        n, doc = thread_pool(cpu)
        res[cpu] = (n, doc)
        print(f'{cpu:>8}{n:>14}{doc:>10}')
    print('代码：min(max(4, int(0.5 × cpu) // 8), 12)；文档写的是 0.75。线程数是每个 TP rank 的，')
    print('一台 8 卡机器上 8 个 rank 各开一份。队列数默认 4，线程数手动设成小于 4 会在启动时断言失败。')
    try:
        thread_pool(128, env_override=2)
    except PDArgError as e:
        res['bad'] = e
        print(f'SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=2：启动失败：{e}')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='resolve', choices=['resolve', 'threads'])
    a = ap.parse_args()
    scenario_resolve() if a.scenario == 'resolve' else scenario_threads()


if __name__ == '__main__':
    main()
