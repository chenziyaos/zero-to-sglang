#!/usr/bin/env bash
# SGL-18：在真机上采一次完整的观测数据，再交给本目录的三个脚本。验证于 sglang 主干 8fabf5e。
#
# 四步各自独立，按顺序跑：
#   bash profile.sh serve        # 带指标和逐请求日志起服务，日志写 logs/server.log
#   bash profile.sh load         # 另一个终端：抓 /metrics -> 压测 -> 再抓 /metrics -> 中途抓 torch trace
#   bash profile.sh analyze      # 三个脚本读上面的产物
#   bash profile.sh nsys         # 可选：在 nsys 下重启服务，按层看 kernel（会关掉 CUDA graph）
#
# 环境变量：HOST PORT MODEL RATE N IN OUT STEPS GRAPH_BS
set -euo pipefail

MODE=${1:-help}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MODEL=${MODEL:-Qwen/Qwen3-8B}
RATE=${RATE:-14}                 # 请求率，压到 decode batch 超过图上限才看得到 eager 回退
N=${N:-1200}
IN=${IN:-2048}
OUT=${OUT:-256}
STEPS=${STEPS:-10}               # torch profiler 抓几步
GRAPH_BS=${GRAPH_BS:-160}        # decode 图的最大 bs，故意设小来复现本章的现象
URL=http://$HOST:$PORT
if [[ -z "${OUTDIR:-}" ]]; then
  # analyze 默认读最近一次 load 的目录，其余模式新建一个
  LAST=$(ls -dt "$(pwd)"/obs-* 2>/dev/null | head -1 || true)
  if [[ "$MODE" == analyze && -n "$LAST" ]]; then OUTDIR=$LAST; else OUTDIR=$(pwd)/obs-$(date +%m%d-%H%M); fi
fi
mkdir -p logs "$OUTDIR"

case "$MODE" in
serve)
  # --enable-metrics：/metrics 暴露 Prometheus 指标
  # --enable-request-time-stats-logging：每条请求结束打一行 ReqTimeStats（排队、前向、entry_time）
  # SGLANG_ENABLE_METRICS_DEVICE_TIMER=1：记 forward_execution_seconds_total，否则 fwd_occupancy 是 NaN
  SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
  SGLANG_TORCH_PROFILER_DIR="$OUTDIR" \
  python3 -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port "$PORT" \
    --enable-metrics --enable-request-time-stats-logging \
    --cuda-graph-max-bs-decode "$GRAPH_BS" \
    2>&1 | tee logs/server.log
  ;;

load)
  curl -sf "$URL/metrics" > "$OUTDIR/metrics_t0.txt"
  python3 -m sglang.benchmark.serving --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 0.5 \
    --num-prompts "$N" --request-rate "$RATE" --disable-tqdm \
    --output-file "$OUTDIR/bench.jsonl" &
  BENCH=$!
  # 等压测进入稳态再抓 trace（sleep 代替 start_step：start_step 是全局前向计数，服务跑过一阵以后写小了等于立即开始）
  # detailed_annotations 让 step[...] 带 g_sk
  sleep 60
  curl -sf -X POST "$URL/start_profile" -H 'Content-Type: application/json' -d "{
      \"output_dir\": \"$OUTDIR\", \"num_steps\": $STEPS, \"activities\": [\"CPU\", \"GPU\"],
      \"detailed_annotations\": true }"
  echo
  wait "$BENCH"
  curl -sf "$URL/metrics" > "$OUTDIR/metrics_t1.txt"
  echo "产物在 $OUTDIR"
  ls -la "$OUTDIR"
  ;;

analyze)
  HERE=$(cd "$(dirname "$0")" && pwd)
  python3 "$HERE/scrape_metrics.py" "$OUTDIR/metrics_t0.txt" "$OUTDIR/metrics_t1.txt"
  python3 "$HERE/reqstats.py" logs/server.log --top 5
  TRACE=$(ls -t "$OUTDIR"/*TP-0*.trace.json.gz 2>/dev/null | head -1 || true)
  if [[ -n "$TRACE" ]]; then
    python3 "$HERE/trace_summary.py" "$TRACE" --top 12
  else
    echo "没找到 *TP-0*.trace.json.gz：trace 可能还在写，或 OUTDIR 不对"
  fi
  ;;

nsys)
  # 层级 NVTX 只在 eager 下有：图回放里捕获的 kernel 不再发 NVTX（官方 benchmark_and_profiling 文档）。
  # 8fabf5e 上 --disable-cuda-graph 已弃用，换成两个 backend=disabled。
  # 起来以后另开终端：bash profile.sh nsys-start，再压一小段负载。
  nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$OUTDIR/layerwise" \
    python3 -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port "$PORT" \
      --enable-layerwise-nvtx-marker \
      --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
  ;;

nsys-start)
  curl -sf -X POST "$URL/start_profile" -H 'Content-Type: application/json' \
    -d '{"num_steps": 10, "activities": ["CUDA_PROFILER"]}'
  echo
  python3 -m sglang.benchmark.serving --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$IN" --random-output-len 64 --num-prompts 64 --disable-tqdm
  echo "nsys 退出后：nsys stats --report cuda_gpu_kern_sum,nvtx_sum $OUTDIR/layerwise.nsys-rep"
  ;;

*)
  sed -n '2,10p' "$0"
  ;;
esac
