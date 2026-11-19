"""SGL-21：起 DeepSeek-V4.1 服务前，离线检查启动参数会被解析成什么。

只用标准库，规则抄自 sglang 8fabf5e：
  engram    -> python/sglang/srt/arg_groups/engram_config.py:41-70  EngramConfig.from_dict
  backend   -> python/sglang/srt/arg_groups/deepseek_v4_hook.py:42-69  _default_dsv4_attention_backend
  replay    -> python/sglang/srt/arg_groups/fields/exec_.py:168-173  两侧有界重放开关

用法：
  python3 check_config.py engram '{"placement":"cpu","lookup":"peer_read"}'
  python3 check_config.py backend --model deepseek_v41 --sm 100 --tp 4 --attn-dp 1
  python3 check_config.py plan --model deepseek_v41 --sm 100 --tp 4 --attn-dp 4 \
      --engram '{"placement":"gpu","lookup":"peer_read"}' --rl
"""
import argparse
import json
import sys

PLACEMENT = ('gpu', 'cpu')
REPLICATION = ('none', 'per_node')
LOOKUP = ('auto', 'all_reduce', 'reduce_scatter', 'all_to_all', 'peer_read')
CHOICES = {'placement': PLACEMENT, 'replication': REPLICATION, 'lookup': LOOKUP}
DEFAULT = {'placement': 'gpu', 'replication': 'none', 'lookup': 'auto'}


def check_engram(raw):
    """返回解析后的 dict；非法时抛 ValueError，报错文字与上游一致。"""
    if raw.get('lookup') == 'collective':  # 旧值，按 auto 读
        raw = {**raw, 'lookup': 'auto'}
    invalid = [f'{k}={v!r}' for k, v in raw.items() if v not in CHOICES.get(k, ())]
    if invalid:
        raise ValueError(f"--engram-config: invalid {', '.join(invalid)}; choices: {CHOICES}")
    cfg = {**DEFAULT, **raw}
    if cfg['lookup'] == 'peer_read' and cfg['placement'] != 'gpu':
        raise ValueError("--engram-config: lookup 'peer_read' reads GPU shards; it needs placement 'gpu'.")
    if cfg['replication'] == 'per_node' and cfg['placement'] != 'cpu':
        raise ValueError("--engram-config: replication 'per_node' needs placement 'cpu'.")
    return cfg


def attn_tp_size(tp, attn_dp=1, attn_cp=1):
    # derive_attn_tp_size：注意力 TP = TP / (DP × CP)
    return tp // (attn_dp * attn_cp)


def resolve_backend(model, device='cuda', sm=100, kv_dtype='fp8_e4m3', disagg='null',
                    tp=1, attn_dp=1, attn_cp=1, prefill_cp=False, layout=None):
    """--dsv4-attn-backend auto 的解析结果和原因。"""
    attn_tp = False if prefill_cp else attn_tp_size(tp, attn_dp, attn_cp) > 1
    reasons = []
    if model != 'deepseek_v41':
        reasons.append(f'模型是 {model}，MegaAttention 只支持 deepseek_v41')
    if device != 'cuda':
        reasons.append(f'设备是 {device}')
    if sm not in (100, 103):
        reasons.append(f'SM{sm} 不是 SM100/SM103')
    if kv_dtype != 'fp8_e4m3':
        reasons.append(f'KV 精度是 {kv_dtype}，需要 fp8_e4m3')
    if disagg == 'null' and attn_tp:
        reasons.append(f'非 PD 部署且注意力 TP={attn_tp_size(tp, attn_dp, attn_cp)}>1，'
                       'MegaAttention 会把每卡补到 64 头')
    if layout and layout.lower() not in ('auto', 'v41', 'v41_new'):
        reasons.append(f'SGLANG_DSV4_KV_LAYOUT={layout}')
    return ('flashmla' if reasons else 'flashmla_mega'), reasons


def plan(a):
    out = []
    backend, why = resolve_backend(a.model, a.device, a.sm, a.kv_dtype, a.disagg,
                                   a.tp, a.attn_dp, a.attn_cp, a.prefill_cp, a.layout)
    out.append(f'dsv4 注意力后端：{backend}')
    out += [f'  原因：{w}' for w in why]
    if a.engram:
        try:
            cfg = check_engram(json.loads(a.engram))
            out.append(f'engram：{cfg}')
            if cfg['placement'] == 'cpu':
                out.append('  表在主机内存：KV 容量变大，查表走 PCIe，比较 max_total_num_tokens 和 decode 吞吐')
        except ValueError as e:
            out.append(f'engram：启动会失败 -> {e}')
    out.append('decoder 有界重放：' + ('需要关闭，加 --no-enable-decoder-swa-bounded-replay'
                                    if a.rl else '默认开启（对支持的配置）'))
    if a.rl and a.deterministic:
        out.append('注意：dsv4 后端拒绝 --enable-deterministic-inference')
    if a.dspark and a.disagg != 'null':
        out.append('DSpark + PD：传输必须是 Mooncake 或 NIXL，静态验证，CP=1')
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('engram')
    e.add_argument('json')
    for name in ('backend', 'plan'):
        s = sub.add_parser(name)
        s.add_argument('--model', default='deepseek_v41')
        s.add_argument('--device', default='cuda')
        s.add_argument('--sm', type=int, default=100)
        s.add_argument('--kv-dtype', default='fp8_e4m3')
        s.add_argument('--disagg', default='null', choices=['null', 'prefill', 'decode'])
        s.add_argument('--tp', type=int, default=1)
        s.add_argument('--attn-dp', type=int, default=1)
        s.add_argument('--attn-cp', type=int, default=1)
        s.add_argument('--prefill-cp', action='store_true')
        s.add_argument('--layout')
        if name == 'plan':
            s.add_argument('--engram')
            s.add_argument('--rl', action='store_true')
            s.add_argument('--deterministic', action='store_true')
            s.add_argument('--dspark', action='store_true')
    a = p.parse_args(argv)
    if a.cmd == 'engram':
        try:
            print(check_engram(json.loads(a.json)))
        except ValueError as err:
            print(err)
            return 1
        return 0
    if a.cmd == 'backend':
        b, why = resolve_backend(a.model, a.device, a.sm, a.kv_dtype, a.disagg,
                                 a.tp, a.attn_dp, a.attn_cp, a.prefill_cp, a.layout)
        print(b)
        for w in why:
            print('  ' + w)
        return 0
    print('\n'.join(plan(a)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
