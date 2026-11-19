#!/usr/bin/env bash
# SGL-23 第 6 章：起好服务后扫并发，每档结果写一个 jsonl。
# 输入输出长度默认 6144 进 2048 出，和 Qwen 官方速度基准的一行对齐；用 IN、OUTLEN 改。
# 用法：PORT=30000 OUT=results ./bench.sh
set -euo pipefail
PORT=${PORT:-30000}
OUT=${OUT:-.}
IN=${IN:-6144}
OUTLEN=${OUTLEN:-2048}
mkdir -p "$OUT"

# 预热，顺带触发 CUDA Graph 和 DeepGEMM 的 JIT
python -m sglang.benchmark.serving --backend sglang --port "$PORT" \
  --dataset-name random --random-input-len 2048 --random-output-len 256 \
  --num-prompts 16 --max-concurrency 4

# 扫并发
for c in 1 4 8 16 32 64; do
  python -m sglang.benchmark.serving --backend sglang --port "$PORT" \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUTLEN" \
    --random-range-ratio 1.0 --num-prompts $((c * 8)) --max-concurrency "$c" \
    --output-file "$OUT/qwen_c${c}.jsonl"
done

# 汇总：每个 jsonl 取最后一行，打印吞吐、TTFT 和 ITL 的中位数
python3 - "$OUT" <<'PY'
import glob, json, os, re, sys
rows = []
for f in glob.glob(os.path.join(sys.argv[1], 'qwen_c*.jsonl')):
    last = json.loads(open(f).read().strip().splitlines()[-1])
    c = int(re.search(r'_c(\d+)', f).group(1))
    rows.append((c, last['output_throughput'], last['median_ttft_ms'], last['median_itl_ms']))
print(f"{'并发':>4} {'输出 tok/s':>10} {'TTFT p50 ms':>12} {'ITL p50 ms':>11}")
for c, t, ttft, itl in sorted(rows):
    print(f'{c:>4} {t:>10.1f} {ttft:>12.0f} {itl:>11.2f}')
PY
