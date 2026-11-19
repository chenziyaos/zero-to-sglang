"""SGL-23 7.3 节的冒烟：思考开关、reasoning 分流、工具调用三项。

只用标准库。服务起来以后、压测之前跑一次。--reasoning-parser 没开，思考内容会混进 content；
--tool-call-parser 没开或选错，tool_calls 会是空的，函数调用以文本形式出现在 content 里。
用法：
  python3 smoke.py --url http://127.0.0.1:30000
"""
import argparse
import json
import sys
import urllib.request

WEATHER = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': '查询某个城市当前的天气',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
    },
}
RAW_MARKERS = ('<think>', '</think>', '<tool_call>')
NO_THINK = {'chat_template_kwargs': {'enable_thinking': False}}


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as f:
        return json.load(f)


def chat(url, model, messages, **kw):
    # Qwen 建议思考模式用 0.6/0.95/20，非思考模式用 0.7/0.8/20；冒烟只看格式，用推荐值即可
    body = dict(model=model, messages=messages, max_tokens=256, temperature=0.7, top_p=0.8, top_k=20)
    body.update(kw)
    return call(url, '/v1/chat/completions', body)['choices'][0]['message']


def raw(text):
    return [m for m in RAW_MARKERS if m in (text or '')]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:30000')
    a = p.parse_args(argv)
    model = call(a.url, '/v1/models')['data'][0]['id']
    print('模型：', model)
    ok = True
    q = [{'role': 'user', 'content': '17*19 等于多少？只回答数字。'}]

    # 1. 关思考：答案在 content，reasoning_content 为空
    m = chat(a.url, model, q, **NO_THINK)
    text = m.get('content') or ''
    print('关思考：', text.strip()[:80], '| reasoning_content', '空' if not m.get('reasoning_content') else '非空')
    ok &= '323' in text and not m.get('reasoning_content')

    # 2. 开思考：思考过程进 reasoning_content，content 里不能有 <think>
    m = chat(a.url, model, q, max_tokens=4096, temperature=0.6, top_p=0.95)
    text, rc = m.get('content') or '', m.get('reasoning_content') or ''
    print('开思考：', text.strip()[:80], f'| reasoning_content {len(rc)} 字')
    ok &= '323' in text and len(rc) > 0
    if raw(text):
        print('content 里有', raw(text), '，检查 --reasoning-parser qwen3')
        ok = False

    # 3. 工具调用：关思考，tool_calls 里要有 get_weather
    m = chat(a.url, model, [{'role': 'user', 'content': '北京现在天气怎么样？'}], tools=[WEATHER], **NO_THINK)
    names = [c['function']['name'] for c in (m.get('tool_calls') or [])]
    print('工具调用：', names or '无')
    ok &= 'get_weather' in names
    if raw(m.get('content')):
        print('content 里有', raw(m.get('content')), '，检查 --tool-call-parser qwen25')
        ok = False

    print('通过' if ok else '未通过')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
