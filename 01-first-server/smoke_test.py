"""SGL-01 冒烟测试：只用标准库，把一个刚起好的 SGLang 服务从健康检查到流式生成过一遍。

    python3 smoke_test.py                         # 默认 http://127.0.0.1:30000
    python3 smoke_test.py --base http://10.0.0.5:30000 --api-key sk-xxx

依次检查：
  1. GET  /ready、/health          两种就绪语义
  2. GET  /v1/models、/model_info  模型名、是否生成模型、parser
  3. POST /v1/chat/completions     非流式，看 reasoning_content 和 usage
  4. POST /v1/chat/completions     流式，测 TTFT 和平均 ITL
  5. POST /generate                原生接口流式，验证默认是“累计文本”
  6. POST /generate                原生接口带 logprob，看 meta_info
任何一步失败就以非零码退出，可以直接放进部署流水线。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

OK, FAIL = '\033[32mOK\033[0m', '\033[31mFAIL\033[0m'


class Client:
    def __init__(self, base, api_key=None, timeout=120):
        self.base, self.timeout = base.rstrip('/'), timeout
        self.h = {'Content-Type': 'application/json'}
        if api_key:
            self.h['Authorization'] = f'Bearer {api_key}'

    def req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.base + path, data=data, headers=self.h, method=method)

    def get(self, path):
        try:
            with urllib.request.urlopen(self.req('GET', path), timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw.strip().startswith(b'{') else raw.decode())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def post(self, path, body):
        with urllib.request.urlopen(self.req('POST', path, body), timeout=self.timeout) as r:
            return json.loads(r.read())

    def sse(self, path, body):
        """逐个产出 (到达时间, 解析后的 JSON)。SGLang 的 SSE 每条是 `data: {...}`，以 `data: [DONE]` 结束。"""
        with urllib.request.urlopen(self.req('POST', path, body), timeout=self.timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    return
                yield time.perf_counter(), json.loads(payload)


def step(name, fn):
    try:
        msg = fn()
        print(f'[{OK}] {name}' + (f'  {msg}' if msg else ''))
    except Exception as e:  # noqa: BLE001
        print(f'[{FAIL}] {name}  {type(e).__name__}: {e}')
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:30000')
    ap.add_argument('--api-key')
    ap.add_argument('--model', help='默认取 /v1/models 返回的第一个')
    a = ap.parse_args()
    c = Client(a.base, a.api_key)
    ctx = {}

    def readiness():
        s1, _ = c.get('/ready')
        s2, _ = c.get('/health')
        assert s1 == 200 and s2 == 200, f'/ready={s1} /health={s2}'
        return '/ready=200 /health=200'

    def models():
        s, j = c.get('/v1/models')
        assert s == 200, j
        ctx['model'] = a.model or j['data'][0]['id']
        s, info = c.get('/model_info')
        assert s == 200, info
        return (f"model={ctx['model']} is_generation={info['is_generation']} "
                f"reasoning_parser={info.get('reasoning_parser')} tool_call_parser={info.get('tool_call_parser')}")

    def chat():
        j = c.post('/v1/chat/completions', {
            'model': ctx['model'], 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': '用一句话解释 KV cache。'}]})
        m, u = j['choices'][0]['message'], j['usage']
        think = (m.get('reasoning_content') or '')[:30].replace('\n', ' ')
        return (f"finish={j['choices'][0]['finish_reason']} prompt={u['prompt_tokens']} "
                f"completion={u['completion_tokens']} reasoning[:30]={think!r} content[:40]={(m['content'] or '')[:40]!r}")

    def chat_stream():
        t0 = time.perf_counter()
        ts, n = [], 0
        for t, j in c.sse('/v1/chat/completions', {
                'model': ctx['model'], 'max_tokens': 128, 'stream': True,
                'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': '从 1 数到 30，用空格分隔。'}]}):
            if j.get('choices') and j['choices'][0]['delta'].get('content'):
                ts.append(t)
            if j.get('usage'):
                n = j['usage']['completion_tokens']
        assert ts, '没有收到任何 content 片段'
        ttft = (ts[0] - t0) * 1e3
        itl = (ts[-1] - ts[0]) / max(len(ts) - 1, 1) * 1e3
        return f'TTFT={ttft:.1f} ms  平均 ITL={itl:.2f} ms  chunks={len(ts)} completion_tokens={n}'

    def native_stream():
        texts = []
        for _, j in c.sse('/generate', {
                'text': 'The capital of France is', 'stream': True,
                'sampling_params': {'max_new_tokens': 16, 'temperature': 0}}):
            texts.append(j['text'])
        cumulative = all(texts[i].startswith(texts[i - 1]) for i in range(1, len(texts)))
        return (f"chunks={len(texts)} 累计文本={cumulative}（默认 incremental_streaming_output=False）"
                f" last={texts[-1]!r}")

    def native_logprob():
        j = c.post('/generate', {
            'text': 'The capital of France is',
            'sampling_params': {'max_new_tokens': 4, 'temperature': 0},
            'return_logprob': True})
        mi = j['meta_info']
        first = mi['output_token_logprobs'][0]
        return (f"finish={mi['finish_reason']} prompt={mi['prompt_tokens']} cached={mi.get('cached_tokens')} "
                f"e2e={mi.get('e2e_latency', 0):.3f}s first_token(logprob,id,text)={first}")

    step('就绪检查', readiness)
    step('模型信息', models)
    step('OpenAI chat 非流式', chat)
    step('OpenAI chat 流式', chat_stream)
    step('原生 /generate 流式', native_stream)
    step('原生 /generate logprob', native_logprob)


if __name__ == '__main__':
    main()
