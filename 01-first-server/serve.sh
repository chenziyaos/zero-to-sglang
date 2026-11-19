#!/usr/bin/env bash
# SGL-01：起一个 Qwen3-8B 服务。验证于 sglang v0.5.22 / 主干 8fabf5e。
#
# 用法：
#   bash serve.sh                       # 默认 Qwen/Qwen3-8B，端口 30000
#   MODEL=Qwen/Qwen3-0.6B bash serve.sh # 24 GB 卡上建议先用小模型
#   EXTRA="--mem-fraction-static 0.85" bash serve.sh
#
# 日志同时写到 logs/serve-时间戳.log，给 parse_startup_log.py 用。
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-1}
EXTRA=${EXTRA:-}

mkdir -p logs
LOG=logs/serve-$(date +%Y%m%d-%H%M%S).log

# 采样默认值取模型 generation_config.json（--sampling-defaults model 是默认行为）。
# Qwen3 打开 thinking 解析和工具调用解析。qwen25 在主干已改名 qwen，旧名仍可用但会打弃用告警；
# 两个 parser 也都接受 auto，按 chat template 自动识别。
set -x
sglang serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen \
  --served-model-name qwen3 \
  --enable-metrics \
  $EXTRA 2>&1 | tee "$LOG"
