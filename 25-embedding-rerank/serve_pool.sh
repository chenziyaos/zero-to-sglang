#!/usr/bin/env bash
# SGL-25 GPU 实验：embedding 和 rerank 服务。单卡即可，0.6B 模型 24 GB 显存足够。本脚本没有在 GPU 上实跑过。
#
# 用法（每个 serve 模式起一个前台服务，另开终端跑 bench / client）：
#   bash serve_pool.sh emb [MODEL]       # Qwen3-Embedding（默认 0.6B）：decoder 结构，要显式加 --is-embedding；带 is_matryoshka 覆盖，允许请求里带 dimensions
#   bash serve_pool.sh emb_nokv [MODEL]  # 同上，再加 --prefill-only-disable-kv-cache：不写 paged KV，要求 FA3/FA4 prefill 后端
#   bash serve_pool.sh m3                # BAAI/bge-m3：XLM-R 编码器，按文档用 triton 注意力后端
#   bash serve_pool.sh rr_bge            # BAAI/bge-reranker-v2-m3：cross-encoder，命令照抄 rerank_models.mdx
#   bash serve_pool.sh rr_qwen           # Qwen/Qwen3-Reranker-0.6B：不能加 --is-embedding，靠模板算 yes / no 的 logprob
#   bash serve_pool.sh bench_emb [LEN]   # sglang.bench_serving --backend sglang-embedding，默认每条 512 token
#   bash serve_pool.sh bench_rr [DOCS]   # 没有现成的 rerank 压测，用 pool_client.py bench --rerank-docs，默认每次 100 篇
#   bash serve_pool.sh client            # pool_client.py：embed、search（bi-encoder 召回）、rerank（cross-encoder 精排）
#
# 环境变量：PORT（默认 30000）、OUT（结果目录，默认 results）、SGLANG_SRC（SGLang 源码目录，rr_qwen 要用里面的 jinja 模板）
#
# 日志里要找的行：
#   "Embedding architecture detected: enabling embedding mode automatically."   编码器结构被自动切成 embedding 模式
#   --prefill-only-disable-kv-cache 的组合不对时，启动直接报 ValueError / RuntimeError，写明缺哪一项
set -Eeuo pipefail

MODE=${1:-}
PORT=${PORT:-30000}
OUT=${OUT:-results}
SGLANG_SRC=${SGLANG_SRC:-$HOME/sglang}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
launch() { python3 -m sglang.launch_server --host 0.0.0.0 --port "$PORT" "$@"; }
served() { curl -s "localhost:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'; }

case "$MODE" in
  emb)
    launch --model-path "${2:-Qwen/Qwen3-Embedding-0.6B}" --is-embedding \
      --json-model-override-args '{"is_matryoshka": true}' 2>&1 | tee "$OUT/emb.log" ;;
  emb_nokv)
    launch --model-path "${2:-Qwen/Qwen3-Embedding-0.6B}" --is-embedding \
      --json-model-override-args '{"is_matryoshka": true}' \
      --chunked-prefill-size -1 --disable-radix-cache --prefill-only-disable-kv-cache 2>&1 | tee "$OUT/emb_nokv.log" ;;
  m3)
    launch --model-path BAAI/bge-m3 --attention-backend triton \
      --disable-radix-cache --chunked-prefill-size -1 2>&1 | tee "$OUT/m3.log" ;;
  rr_bge)
    launch --model-path BAAI/bge-reranker-v2-m3 --disable-radix-cache --chunked-prefill-size -1 \
      --attention-backend triton --is-embedding 2>&1 | tee "$OUT/rr_bge.log" ;;
  rr_qwen)
    launch --model-path Qwen/Qwen3-Reranker-0.6B --trust-remote-code --disable-radix-cache \
      --chat-template "$SGLANG_SRC/examples/chat_template/qwen3_reranker.jinja" 2>&1 | tee "$OUT/rr_qwen.log" ;;
  bench_emb)
    LEN=${2:-512}
    MODEL=$(served)
    for RATE in 50 200 800 inf; do
      python3 -m sglang.bench_serving --backend sglang-embedding --port "$PORT" --model "$MODEL" \
        --dataset-name random --random-input-len "$LEN" --random-output-len 1 --num-prompts 2000 \
        --request-rate "$RATE" --output-file "$OUT/bench_emb_${LEN}.jsonl"
    done ;;
  bench_rr)
    DOCS=${2:-100}
    python3 "$HERE/pool_client.py" bench --url "http://localhost:$PORT" --model "$(served)" --engine sglang \
      --rerank-docs "$DOCS" --n 200 --concurrency 8 --words 200 | tee "$OUT/bench_rr_$DOCS.txt" ;;
  client)
    MODEL=$(served)
    case "$MODEL" in
      *Reranker*|*reranker*)
        python3 "$HERE/pool_client.py" rerank --url "http://localhost:$PORT" --model "$MODEL" --engine sglang \
          --query "显存不够怎么办" ;;
      *)
        python3 "$HERE/pool_client.py" embed --url "http://localhost:$PORT" --model "$MODEL"
        python3 "$HERE/pool_client.py" embed --url "http://localhost:$PORT" --model "$MODEL" --dimensions 256
        python3 "$HERE/pool_client.py" search --url "http://localhost:$PORT" --model "$MODEL" --query "显存不够怎么办"
        python3 "$HERE/pool_client.py" bench --url "http://localhost:$PORT" --model "$MODEL" ;;
    esac ;;
  *)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
