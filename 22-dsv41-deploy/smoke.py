"""SGL-22 7.3 节的冒烟：17×19 要答 323，工具调用要解析出 get_weather。

只用标准库。服务起来以后、压测之前跑一次，parser 没开或者模板不对，这里就会露出来。
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


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as f:
        return json.load(f)


def chat(url, model, messages, **kw):
    body = dict(model=model, messages=messages, temperature=0, max_tokens=256, **kw)
    return call(url, '/v1/chat/completions', body)['choices'][0]['message']


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:30000')
    a = p.parse_args(argv)
    model = call(a.url, '/v1/models')['data'][0]['id']
    ok = True

    m = chat(a.url, model, [{'role': 'user', 'content': '17*19 等于多少？只回答数字。'}])
    text = m.get('content') or ''
    print('算术：', text.strip()[:80])
    ok &= '323' in text
    if 'DSML' in text or '<｜' in text:
        print('content 里混着原始标记，检查 --reasoning-parser 和 --tool-call-parser')
        ok = False

    m = chat(a.url, model, [{'role': 'user', 'content': '北京现在天气怎么样？'}], tools=[WEATHER])
    names = [c['function']['name'] for c in (m.get('tool_calls') or [])]
    print('工具调用：', names or '无')
    ok &= 'get_weather' in names

    print('通过' if ok else '未通过')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
