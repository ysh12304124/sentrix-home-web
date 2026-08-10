#!/usr/bin/env bash
set -euo pipefail

# CompassJudger-1-1.5B-Instruct-int8 on GPU1, ~3G VRAM
export CUDA_VISIBLE_DEVICES=1

MODEL_DIR=/home/realmagic/models/compassjudger/CompassJudger-1-1.5B-Instruct-int8
VLLM_BIN=/home/realmagic/miniconda3/envs/sentrix-judge/bin/vllm
PORT=8110
PID_FILE="$HOME/github/sentrix-home-web/.judge.pid"
LOG_DIR="$HOME/github/sentrix-home-web/logs"
mkdir -p "$LOG_DIR"

if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Judge already running on :${PORT}"
  exit 0
fi

nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name compassjudger-1-1.5b-instruct \
  --dtype auto \
  --max-model-len 1024 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.12 \
  --tensor-parallel-size 1 \
  > "$LOG_DIR/judge.log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started CompassJudger on GPU1 :${PORT} (pid $(cat "$PID_FILE"))"
