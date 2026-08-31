#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -f "$ROOT_DIR/.env.local" ]; then
  set -a
  . "$ROOT_DIR/.env.local"
  set +a
fi
BACKEND_PORT=${BENCH_PORT:-8771}
FRONTEND_PORT=${BENCH_DEV_PORT:-5173}
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
BACKEND_PID_FILE="$ROOT_DIR/logs/orchestrator.pid"
FRONTEND_PID_FILE="$ROOT_DIR/logs/vite.pid"
PHOTOBENCH_QA_CONCURRENCY=${PHOTOBENCH_QA_CONCURRENCY:-12}
PHOTOBENCH_JUDGE_CONCURRENCY=${PHOTOBENCH_JUDGE_CONCURRENCY:-8}
PHOTOBENCH_JUDGE_RETRY_ATTEMPTS=${PHOTOBENCH_JUDGE_RETRY_ATTEMPTS:-6}
PHOTOBENCH_JUDGE_RETRY_BACKOFF_SECONDS=${PHOTOBENCH_JUDGE_RETRY_BACKOFF_SECONDS:-5}
PHOTOBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS=${PHOTOBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS:-60}
PHOTOBENCH_JUDGE_REQUEST_INTERVAL_SECONDS=${PHOTOBENCH_JUDGE_REQUEST_INTERVAL_SECONDS:-0.5}
mkdir -p "$ROOT_DIR/logs"

wait_for_url() {
  URL=$1
  PID=$2
  COUNT=0
  while [ "$COUNT" -lt 300 ]; do
    if curl -fsS "$URL" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
    COUNT=$((COUNT + 1))
  done
  return 1
}

if curl -fsS "http://127.0.0.1:$BACKEND_PORT/api/config" >/dev/null 2>&1; then
  BACKEND_PID=$(lsof -nP -tiTCP:"$BACKEND_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)
  if [ -n "$BACKEND_PID" ]; then
    echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
  fi
  echo "Backend already available: http://127.0.0.1:$BACKEND_PORT/"
else
  nohup env PHOTOBENCH_QA_CONCURRENCY="$PHOTOBENCH_QA_CONCURRENCY" \
    PHOTOBENCH_JUDGE_CONCURRENCY="$PHOTOBENCH_JUDGE_CONCURRENCY" \
    PHOTOBENCH_JUDGE_RETRY_ATTEMPTS="$PHOTOBENCH_JUDGE_RETRY_ATTEMPTS" \
    PHOTOBENCH_JUDGE_RETRY_BACKOFF_SECONDS="$PHOTOBENCH_JUDGE_RETRY_BACKOFF_SECONDS" \
    PHOTOBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS="$PHOTOBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS" \
    PHOTOBENCH_JUDGE_REQUEST_INTERVAL_SECONDS="$PHOTOBENCH_JUDGE_REQUEST_INTERVAL_SECONDS" \
    python3 "$ROOT_DIR/backend/benchmark_orchestrator.py" \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    >> "$ROOT_DIR/logs/orchestrator.log" 2>&1 &
  BACKEND_PID=$!
  echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
  if ! wait_for_url "http://127.0.0.1:$BACKEND_PORT/api/config" "$BACKEND_PID"; then
    echo "Backend failed to start; check $ROOT_DIR/logs/orchestrator.log" >&2
    exit 1
  fi
  echo "Backend started: http://127.0.0.1:$BACKEND_PORT/ (PID $BACKEND_PID)"
fi

if curl -fsS "http://127.0.0.1:$FRONTEND_PORT/" >/dev/null 2>&1; then
  FRONTEND_PID=$(lsof -nP -tiTCP:"$FRONTEND_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)
  if [ -n "$FRONTEND_PID" ]; then
    echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
  fi
  echo "Vite frontend already available: http://127.0.0.1:$FRONTEND_PORT/"
else
  (cd "$ROOT_DIR/frontend" && nohup npm run dev -- --port "$FRONTEND_PORT" \
    >> "$ROOT_DIR/logs/vite.log" 2>&1 & echo $! > "$FRONTEND_PID_FILE")
  FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
  if ! wait_for_url "http://127.0.0.1:$FRONTEND_PORT/" "$FRONTEND_PID"; then
    echo "Vite frontend failed to start; check $ROOT_DIR/logs/vite.log" >&2
    exit 1
  fi
  echo "Vite frontend started: http://127.0.0.1:$FRONTEND_PORT/ (PID $FRONTEND_PID)"
fi

echo "Development mode is ready. Local: http://127.0.0.1:$FRONTEND_PORT/"
if [ -n "$LAN_IP" ]; then
  echo "LAN: http://$LAN_IP:$FRONTEND_PORT/"
fi
