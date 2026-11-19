#!/usr/bin/env bash
# SGL-03：对一个已启动的服务做请求率扫描和并发扫描。验证于 sglang 主干 8fabf5e。
#
# 先用 ../01-first-server/serve.sh 起服务（默认端口 30000），再：
#   bash bench.sh rate                      # 开环：泊松到达，扫请求率
#   bash bench.sh conc                      # 闭环：固定并发，扫并发度
#   bash bench.sh one 8                     # 只跑一个请求率，带完整明细
#   RATES="1 2 4 6 8" IN=2048 OUT=128 bash bench.sh rate
#
# 结果都追加写进 results/ 下的 JSONL，一行一次压测；用 sim_stats.py 读。
set -euo pipefail

MODE=${1:-rate}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MODEL=${MODEL:-Qwen/Qwen3-8B}          # 只用来加载 tokenizer，必须和服务端一致
IN=${IN:-1024}
OUT=${OUT:-256}
RATIO=${RATIO:-1.0}                    # 固定长度；sglang 默认 0.0 会让长度在 1..IN 均匀分布
N=${N:-1000}
RATES=${RATES:-"2 4 6 8 10 12 14 16 18 20"}
CONCS=${CONCS:-"1 4 16 32 64 128 256"}
SLO=${SLO:-"ttft:2000 tpot:50"}
TAG=${TAG:-$(date +%m%d-%H%M)}

mkdir -p results
OUTFILE=results/${MODE}-${TAG}-${IN}-${OUT}.jsonl

run() {
  # $1 = 请求率，$2 = 并发上限（可空）
  local extra=()
  [[ -n "${2:-}" ]] && extra+=(--max-concurrency "$2")
  python -m sglang.benchmark.serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
    --random-range-ratio "$RATIO" --num-prompts "$N" \
    --request-rate "$1" "${extra[@]}" \
    --goodput $SLO --output-details --output-file "$OUTFILE" \
    --warmup-requests 4 --flush-cache --disable-tqdm --tag "$TAG"
}

case "$MODE" in
  rate)
    for r in $RATES; do run "$r"; done ;;
  conc)
    # 闭环时请求率设 inf，并发度决定压力；每档请求数至少是并发度的 8 倍，否则爬坡和收尾占比太大
    for c in $CONCS; do
      n=$(( c * 8 > N ? c * 8 : N ))
      N=$n run inf "$c"
    done ;;
  one)
    run "${2:-8}" ;;
  *)
    echo "用法：bash bench.sh rate|conc|one [rate]"; exit 1 ;;
esac

echo "# 结果在 $OUTFILE"
python3 "$(dirname "$0")/sim_stats.py" "$OUTFILE" --goodput $SLO
