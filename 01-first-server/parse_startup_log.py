"""从 SGLang 启动日志里抽出关键数字，和 estimate_startup.py 的预测对账（CPU 即可运行）。

    python3 parse_startup_log.py logs/serve-20261028-153000.log
    python3 parse_startup_log.py sample_startup.log --model qwen3-8b --gpu h100-sxm

匹配的日志格式取自 sglang 主干 8fabf5e：
  model_runner.py      Load weight begin / Load weight end
  memory_pool.py       KV Cache is allocated
  cuda_graph_setup.py  Capture ... CUDA graph begin / end
  scheduler.py         max_total_num_tokens=...
  metrics_reporter.py  Prefill batch / Decode batch
多卡时每个 rank 都会打印一部分行，这里只取第一次出现的值。
"""
import argparse
import os
import re
import sys
import unicodedata

PATTERNS = {
    'frac': r"server_args=ServerArgs\(.*?mem_fraction_static=([\d.]+)",
    'pre_load': r'Load weight begin\. avail mem=([\d.]+) GB',
    'weight_time': r'Load weight end\. elapsed=([\d.]+) s',
    'after_load': r'Load weight end\..*?avail mem=([\d.]+) GB',
    'weights': r'Load weight end\..*?mem usage=([\d.]+) GB',
    'kv_tokens': r'KV Cache is allocated\. dtype: \S+, #tokens: (\d+)',
    'k_size': r'KV Cache is allocated\..*?K size: ([\d.]+) GB',
    'kv_size_mla': r'KV Cache is allocated\..*?KV size: ([\d.]+) GB',
    'pool_end': r'Memory pool end\. avail mem=([\d.]+) GB',
    'prefill_graph': r'Capture \w+ prefill CUDA graph end\. elapsed=([\d.]+) s, mem usage=([\d.]+) GB',
    'decode_graph': r'Capture \w+ (?:decode|verify) CUDA graph end\. bs=\[.*?\], elapsed=([\d.]+) s, mem usage=([\d.]+) GB',
    'max_total': r'max_total_num_tokens=(\d+)',
    'chunk': r'max_total_num_tokens=\d+, chunked_prefill_size=(-?\d+)',
    'max_running': r'max_running_requests=(\d+)',
    'context_len': r'context_len=(\d+)',
    'avail_after': r'available_gpu_mem=([\d.]+) GB',
}
PREFILL = re.compile(r'Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), '
                     r'#running-req: (\d+), #queue-req: (\d+).*?input throughput \(token/s\): ([\d.]+)')
DECODE = re.compile(r'Decode batch.*?#running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*?'
                    r'cuda graph: (True|False), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)')


def parse(text):
    out = {}
    for k, p in PATTERNS.items():
        m = re.search(p, text)
        if m:
            out[k] = tuple(float(x) for x in m.groups()) if len(m.groups()) > 1 else float(m.group(1))
    out['prefill'] = [tuple(float(x) if '.' in x else int(x) for x in m.groups()) for m in PREFILL.finditer(text)]
    out['decode'] = [m.groups() for m in DECODE.finditer(text)]
    out['ready'] = 'The server is fired up and ready to roll!' in text
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--model', help='给出时调用 estimate_startup 对账')
    ap.add_argument('--gpu')
    ap.add_argument('--tp', type=int, default=1)
    a = ap.parse_args()
    r = parse(open(a.log, errors='ignore').read())

    def pad(s, n=28):
        w = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in s)
        return s + ' ' * max(n - w, 1)

    def show(name, key, fmt='{}'):
        v = r.get(key)
        print('  ' + pad(name) + (fmt.format(*v) if isinstance(v, tuple) else fmt.format(v) if v is not None else '（未找到）'))

    print('[启动阶段]')
    show('加载前空闲', 'pre_load', '{:.2f} GB')
    show('权重占用', 'weights', '{:.2f} GB')
    show('权重加载耗时', 'weight_time', '{:.1f} s')
    show('KV 池 token 数', 'kv_tokens', '{:.0f}')
    show('K 池大小（V 同）', 'k_size', '{:.2f} GB')
    show('prefill graph 耗时/显存', 'prefill_graph', '{:.1f} s / {:.2f} GB')
    show('decode graph 耗时/显存', 'decode_graph', '{:.1f} s / {:.2f} GB')
    show('max_total_num_tokens', 'max_total', '{:.0f}')
    show('chunked_prefill_size', 'chunk', '{:.0f}')
    show('max_running_requests', 'max_running', '{:.0f}')
    show('context_len', 'context_len', '{:.0f}')
    show('启动完成后空闲', 'avail_after', '{:.2f} GB')
    print('  ' + pad('就绪') + f'{"是" if r["ready"] else "否（没看到 fired up 这一行）"}')

    if r['prefill'] or r['decode']:
        print('[运行期]')
        if r['prefill']:
            new = sum(p[1] for p in r['prefill'])
            hit = sum(p[2] for p in r['prefill'])
            print(f'  prefill 批次 {len(r["prefill"])}，新算 token {new}，命中缓存 token {hit}，'
                  f'命中率 {hit / max(new + hit, 1):.1%}')
        if r['decode']:
            usage = [float(d[2]) for d in r['decode']]
            no_graph = sum(d[3] == 'False' for d in r['decode'])
            peak_q = max(int(d[5]) for d in r['decode'])
            print(f'  decode 日志 {len(r["decode"])} 条，token usage 峰值 {max(usage):.2f}，'
                  f'未走 CUDA Graph {no_graph} 条，排队峰值 {peak_q}')

    if a.model and a.gpu and r.get('max_total'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from estimate_startup import SMI_MIB, estimate
        # 用日志里真实的“加载前空闲”反推 CUDA 上下文占用，把这一项误差先消掉
        ctx = SMI_MIB[a.gpu] / 1024 - r['pre_load'] if 'pre_load' in r else 0.6
        e = estimate(a.model, a.gpu, a.tp, mem_frac=r.get('frac'), cuda_ctx_gib=ctx)
        err = (e['tokens'] - r['max_total']) / r['max_total']
        print('[对账]')
        print('  ' + pad('CUDA 上下文等（反推）') + f'{ctx:.2f} GiB')
        if 'frac' in r:
            print('  ' + pad('mem_fraction_static') + f"{r['frac']}（取自 server_args 行）")
        if 'weights' in r:
            print(f'  权重 预测 {e["weights"]:.2f} GB，实际 {r["weights"]:.2f} GB')
        print(f'  max_total_num_tokens 预测 {e["tokens"]}，实际 {r["max_total"]:.0f}，偏差 {err:+.1%}')
        print('  偏差超过 10% 时先看：是否手动设了 --mem-fraction-static、是否开了投机解码或 DP Attention、'
              '模型是否 MLA 或带滑窗层')


if __name__ == '__main__':
    main()
