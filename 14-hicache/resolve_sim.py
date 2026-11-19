"""启动参数解析：一条 --enable-hicache 命令行，SGLang 最后用了什么布局、IO 后端、主机池多大。

只用标准库。逻辑按 sglang 8fabf5e 重写，路径相对 python/sglang/srt/：
  arg_groups/pipeline.py:77,272          ratio 默认值先于 dummy model 边界，handle_hicache 在后
  arg_groups/hicache_hook.py:100         handle_hicache_ratio_default：cache 模式 2.0、buffer_only 1.2，
                                         显式给 ratio 或 size 时关掉自动缩放（fraction=None）
  arg_groups/hicache_hook.py:18          handle_hicache：外部 linker 互斥检查、何时整段跳过、各步顺序
  arg_groups/hicache_hook.py:171         resolve_layout_io_compatibility：page_first_direct 和 kernel、page_first 和 direct
  arg_groups/hicache_hook.py:197         resolve_storage_layout_compatibility：mooncake / npu_memcache 不收 layer_first
  arg_groups/hicache_hook.py:228         resolve_unified_memory_layout_io
  arg_groups/hicache_hook.py:247         validate_hicache_host_memory_mode：buffer_only 的三条规矩
  arg_groups/hicache_hook.py:129         resolve_hicache_dcp_compatibility：DCP 只支持 MLA
  mem_cache/kv_cache_builder.py:164,244  decode 实例的 ratio：备份专用 host_pool 0.2（且至少装下一个最长请求），否则 2.0
  mem_cache/hicache_auto_size.py         auto_size_hicache：预算 × fraction × 0.95 ÷ 设备字节，取 min 后全局 MIN
  mem_cache/pool_host/base.py:28,84      预留 10 GiB，剩下按每机 rank 数平分
  mem_cache/pool_host/base.py:171        HostKVCache：size 换算成 token、多补一页、按页取整、两条告警和一条报错

  python3 resolve_sim.py                    # resolve：十几条典型命令行的解析结果和告警
  python3 resolve_sim.py --scenario size    # size：一台 8 卡机器的主机池怎么算出来
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '02-ledger'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))

GiB = 1 << 30
RESERVE = 10 * GiB          # pool_host/base.py:28 HICACHE_HOST_MEMORY_RESERVE_BYTES
SLACK = 0.05                # hicache_auto_size.py _ALLOCATION_SLACK_FRACTION
BACKUP_ONLY_RATIO = 0.2     # kv_cache_builder.py:164

DEFAULTS = dict(
    enable_hierarchical_cache=False, hicache_host_memory_mode='cache', hicache_ratio=None,
    hicache_size=0, hicache_host_memory_fraction=0.8, hicache_write_policy='write_through',
    hicache_io_backend='kernel', hicache_mem_layout='page_first', hicache_storage_backend=None,
    hicache_storage_prefetch_policy='timeout',
    disaggregation_mode='null', disaggregation_decode_enable_offload_kvcache=False,
    disaggregation_decode_retraction_backup=None, disaggregation_decode_host_receive_threshold=1.0,
    enable_unified_cache_external_linker=False, enable_unified_memory=False,
    dcp_size=1, use_mla=False, speculative_algorithm=None, enable_lmcache=False, enable_hisparse=False,
)


class HiCacheArgError(ValueError):
    pass


def resolve(**kw):
    """返回 (解析后的参数, 告警列表)。抛 HiCacheArgError / NotImplementedError 表示启动失败。"""
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown args {sorted(unknown)}')
    a = dict(DEFAULTS, **kw)
    warns = []

    # pipeline.py:77 handle_hicache_ratio_default
    f = a['hicache_host_memory_fraction']
    if f is not None and not 0 < f <= 1:
        raise HiCacheArgError('--hicache-host-memory-fraction must be in (0, 1].')
    explicit = a['hicache_ratio'] is not None or a['hicache_size'] > 0
    if a['hicache_ratio'] is None and a['disaggregation_mode'] != 'decode':
        a['hicache_ratio'] = 1.2 if a['hicache_host_memory_mode'] == 'buffer_only' else 2.0
    if explicit:
        a['hicache_host_memory_fraction'] = None

    # pipeline.py:272 handle_hicache
    if a['enable_unified_cache_external_linker']:
        if a['enable_hierarchical_cache']:
            raise HiCacheArgError('--enable-unified-cache-external-linker and '
                                  '--enable-hierarchical-cache are mutually exclusive.')
        if a['hicache_storage_backend'] is not None:
            raise HiCacheArgError('--enable-unified-cache-external-linker does not use '
                                  '--hicache-storage-backend.')
        return a, warns
    active = (a['enable_hierarchical_cache'] or a['disaggregation_decode_enable_offload_kvcache']
              or (a['disaggregation_mode'] == 'decode'
                  and a['disaggregation_decode_retraction_backup'] in (None, 'host_pool')))
    if not active:
        return a, warns

    _validate_mode(a, warns)
    if a['disaggregation_decode_host_receive_threshold'] < 1:
        a['hicache_mem_layout'] = 'layer_first'

    # Step 1 layout <-> io
    if a['hicache_mem_layout'] == 'page_first_direct' and a['hicache_io_backend'] == 'kernel':
        a['hicache_io_backend'] = 'direct'
        warns.append('Kernel io backend does not support page first direct layout, '
                     'switching to direct io backend')
    if a['hicache_mem_layout'] == 'page_first' and a['hicache_io_backend'] == 'direct':
        a['hicache_mem_layout'] = 'page_first_direct'
        warns.append('Page first layout is not supported with direct IO backend, '
                     'switching to page first direct layout')

    # Step 2 storage <-> layout
    sb = a['hicache_storage_backend']
    if not (sb == 'mooncake' and a['dcp_size'] > 1) and sb in ('mooncake', 'npu_memcache') \
            and a['hicache_mem_layout'] == 'layer_first':
        io = a['hicache_io_backend']
        new = {'direct': 'page_first_direct', 'kernel': 'page_first'}.get(io, a['hicache_mem_layout'])
        a['hicache_mem_layout'] = new
        warns.append('Mooncake/Ascend MemCache storage backend does not support layer_first layout, '
                     f'switching to {new} layout for {io} io backend')
    if a['enable_unified_memory'] and a['hicache_mem_layout'] == 'page_first_direct':
        a['hicache_mem_layout'], a['hicache_io_backend'] = 'page_first', 'kernel'
        warns.append('--enable-unified-memory: switching to the page_first layout with the kernel io backend')
    if a['disaggregation_decode_host_receive_threshold'] < 1 and a['hicache_mem_layout'] != 'layer_first':
        raise HiCacheArgError(f"The resolved HiCache storage layout {a['hicache_mem_layout']!r} "
                              'cannot share the layer_first decode host pool used by KV transfer')

    # Step 3 DCP
    if a['dcp_size'] > 1 and a['enable_hierarchical_cache']:
        if sb is not None and a['hicache_host_memory_mode'] == 'buffer_only':
            raise NotImplementedError('HiCache L3 with DCP does not support buffer_only host memory.')
        if a['speculative_algorithm'] not in (None, 'DSPARK'):
            raise NotImplementedError('HiCache with --dcp-size > 1 only supports DSPARK speculative decoding')
        if a['enable_lmcache'] or a['enable_hisparse']:
            raise NotImplementedError('lmcache / hisparse with --dcp-size > 1 is not supported')
        if not a['use_mla']:
            raise NotImplementedError('HiCache with --dcp-size > 1 is only supported for MLA models')
    return a, warns


def _validate_mode(a, warns):
    mode = a['hicache_host_memory_mode']
    if mode not in ('cache', 'buffer_only'):
        raise HiCacheArgError(f"hicache_host_memory_mode must be 'cache' or 'buffer_only', got {mode!r}")
    if a['hicache_size'] <= 0 and a['hicache_ratio'] is None and a['disaggregation_mode'] != 'decode':
        raise HiCacheArgError('requires a host pool size: pass --hicache-size or --hicache-ratio.')
    if mode == 'cache':
        return
    if a['hicache_storage_backend'] is None:
        raise HiCacheArgError('--hicache-host-memory-mode buffer_only requires a storage backend')
    if a['hicache_write_policy'] == 'write_back':
        a['hicache_write_policy'] = 'write_through'
        warns.append('(info) --hicache-host-memory-mode buffer_only resolves '
                     '--hicache-write-policy write_back to write_through')
    if a['disaggregation_mode'] == 'decode':
        raise HiCacheArgError('--hicache-host-memory-mode buffer_only is not supported on decode instances')


def decode_ratio(backend, enable_hicache, hicache_size, max_context, device_tokens, page=64,
                 receive_threshold=1.0):
    """kv_cache_builder.py:244：decode 实例在加载模型后才定 ratio。"""
    if backend == 'host_pool' and not enable_hicache:
        ratio = BACKUP_ONLY_RATIO
        if hicache_size == 0:
            need = -(-min(max_context - 1, device_tokens) // page) * page
            if receive_threshold < 1:
                need *= 2
            ratio = max(ratio, need / device_tokens)
        return ratio
    return 2.0


def host_budget_bytes(available_bytes, ranks_per_host):
    """pool_host/base.py:84：可用内存（受 cgroup 限制）先扣 10 GiB，再按本机 rank 数平分。"""
    return (available_bytes - RESERVE) // ranks_per_host


def auto_ratio(requested, device_bytes, available_bytes, ranks_per_host, fraction=0.8):
    """hicache_auto_size.py：fraction 为 None（显式给了 ratio/size）时不动。返回 (ratio, 每 rank 预算字节)。"""
    if fraction is None:
        return requested, None
    budget = int(host_budget_bytes(available_bytes, ranks_per_host) * fraction)
    ratio = min(requested, budget * (1 - SLACK) / device_bytes)
    if ratio <= 0:
        raise HiCacheArgError('No host memory is left for HiCache after the 10 GiB reserve; '
                              'set --hicache-ratio or --hicache-size explicitly.')
    return ratio, budget


def host_pool_tokens(device_tokens, size_per_token, page, ratio=None, hicache_size_gb=0):
    """pool_host/base.py:171。返回 (token 数, 告警或 None)。"""
    if hicache_size_gb > 0:
        size = int(hicache_size_gb * 1e9 // size_per_token)
    else:
        size = int(device_tokens * ratio)
    page_num = size // page + 1
    size = page_num * page
    warn = None
    if size <= device_tokens:
        warn = (f'HiCache kv host pool ({size} tokens) is smaller than the device pool '
                f'({device_tokens} tokens);L2 cache effectiveness is reduced.')
    return size, warn


def check_alloc(size, size_per_token, budget_bytes):
    req = size * size_per_token
    if req > budget_bytes:
        raise HiCacheArgError(f'Not enough host memory available. Requesting {req / 1e9:.2f} GB '
                              f'but only have {budget_bytes / 1e9:.2f} GB free. '
                              'Please reduce the size of the hierarchical cache.')
    return req


CASES = [
    ('默认 cache 模式', dict(enable_hierarchical_cache=True)),
    ('buffer_only 没给存储', dict(enable_hierarchical_cache=True, hicache_host_memory_mode='buffer_only')),
    ('buffer_only + write_back', dict(enable_hierarchical_cache=True, hicache_host_memory_mode='buffer_only',
                                      hicache_storage_backend='mooncake', hicache_write_policy='write_back')),
    ('page_first_direct + kernel', dict(enable_hierarchical_cache=True, hicache_mem_layout='page_first_direct')),
    ('page_first + direct', dict(enable_hierarchical_cache=True, hicache_io_backend='direct')),
    ('mooncake + layer_first', dict(enable_hierarchical_cache=True, hicache_storage_backend='mooncake',
                                    hicache_mem_layout='layer_first')),
    ('mooncake + layer_first + direct', dict(enable_hierarchical_cache=True, hicache_storage_backend='mooncake',
                                             hicache_mem_layout='layer_first', hicache_io_backend='direct')),
    ('hf3fs + layer_first', dict(enable_hierarchical_cache=True, hicache_storage_backend='hf3fs',
                                 hicache_mem_layout='layer_first')),
    ('显式 --hicache-size 30', dict(enable_hierarchical_cache=True, hicache_size=30)),
    ('fraction 1.5', dict(enable_hierarchical_cache=True, hicache_host_memory_fraction=1.5)),
    ('没开 hicache', dict(hicache_storage_backend='file')),
    ('decode 实例默认', dict(disaggregation_mode='decode')),
    ('decode + buffer_only', dict(disaggregation_mode='decode', enable_hierarchical_cache=True,
                                  hicache_host_memory_mode='buffer_only', hicache_storage_backend='mooncake')),
    ('decode 接收阈值 0.5', dict(disaggregation_mode='decode', disaggregation_decode_host_receive_threshold=0.5)),
    ('DCP + MHA 模型', dict(enable_hierarchical_cache=True, dcp_size=2)),
    ('DCP + MLA + EAGLE', dict(enable_hierarchical_cache=True, dcp_size=2, use_mla=True,
                               speculative_algorithm='EAGLE')),
    ('linker 和 hicache 同开', dict(enable_hierarchical_cache=True, enable_unified_cache_external_linker=True)),
]

SHOW = ('hicache_ratio', 'hicache_host_memory_fraction', 'hicache_write_policy', 'hicache_mem_layout',
        'hicache_io_backend')


def scenario_resolve():
    out = {}
    for name, kw in CASES:
        try:
            a, warns = resolve(**kw)
            out[name] = (a, warns)
            short = '  '.join(f'{k.replace("hicache_", "")}={a[k]}' for k in SHOW)
            print(f'{name:<34}{short}')
            for w in warns:
                print(f'{"":<36}WARN {w}')
        except (HiCacheArgError, NotImplementedError) as e:
            out[name] = e
            print(f'{name:<34}启动失败：{e}')
    return out


def scenario_size(model='llama3.1-70b', gpu='h100-sxm', tp=8, host_gib=2048, page=64):
    from sim_ledger import ledger
    L = ledger(model, gpu, tp=tp)
    dev, cell = L['tokens'], L['cell']
    dev_bytes = dev * cell
    avail = host_gib * GiB
    print(f'{L["model"].name} tp={tp}，每 rank KV 池 {dev:,} token × {cell} B = {dev_bytes / GiB:.1f} GiB')
    print(f'整机可用内存 {host_gib} GiB，8 个 rank')
    res = {}
    for label, kw in (('默认（ratio 2，fraction 0.8）', dict(fraction=0.8)),
                      ('fraction 0.3（同机还跑别的）', dict(fraction=0.3))):
        ratio, budget = auto_ratio(2.0, dev_bytes, avail, 8, **kw)
        size, warn = host_pool_tokens(dev, cell, page, ratio=ratio)
        res[label] = (ratio, budget, size, warn)
        print(f'  {label}：每 rank 预算 {(avail - RESERVE) / 8 / GiB:.1f} × {kw["fraction"]} = {budget / GiB:.1f} GiB，'
              f'ratio {2.0:.3f} -> {ratio:.3f}，主机池 {size:,} token = {size * cell / GiB:.1f} GiB')
        if warn:
            print(f'    WARN {warn}')
    for gb in (30, 120):
        size, warn = host_pool_tokens(dev, cell, page, hicache_size_gb=gb)
        budget = host_budget_bytes(avail, 8)
        try:
            check_alloc(size, cell, budget)
            msg = 'OK'
        except HiCacheArgError as e:
            msg = str(e)
        res[f'size{gb}'] = (size, warn, msg)
        print(f'  --hicache-size {gb}：{size:,} token（{size / dev:.2f} 倍显存池），整机 {gb * 8} GB；{msg}')
        if warn:
            print(f'    WARN {warn}')
    print('  --hicache-size 按每个 rank 算，单位是 1e9 字节；8 卡写 30 就是整机 240 GB。')
    print('  auto-sizing 只在没给 ratio 和 size 时生效，日志里的 “HiCache auto-sizing: ratio 2.000 -> x” 就是它。')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='resolve', choices=['resolve', 'size'])
    ap.add_argument('--host-gib', type=int, default=2048)
    a = ap.parse_args()
    if a.scenario == 'resolve':
        scenario_resolve()
    else:
        scenario_size(host_gib=a.host_gib)


if __name__ == '__main__':
    main()
