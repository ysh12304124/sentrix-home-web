#!/usr/bin/env bash
# Start this clone's Sentrix Web + API using .env.
#
# Usage:
#   ./start.sh              # start if ports are free
#   ./start.sh -r           # stop existing listeners on these ports, then start
#   ./start.sh --status     # show port / health status only
#
# Requires bash (re-execs if started via `sh ./start.sh`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a
fi

export SENTRIX_DATA_DIR="${SENTRIX_DATA_DIR:-$root/data}"
export SENTRIX_DB_PATH="${SENTRIX_DB_PATH:-$SENTRIX_DATA_DIR/sentrix.db}"
export SENTRIX_ANN_DIR="${SENTRIX_ANN_DIR:-$SENTRIX_DATA_DIR/ann}"
export SENTRIX_API_PORT="${SENTRIX_API_PORT:-11001}"
export PORT="${PORT:-11000}"
export SENTRIX_BACKEND_URL="${SENTRIX_BACKEND_URL:-http://127.0.0.1:${SENTRIX_API_PORT}}"
export PYTHONPATH="${PYTHONPATH:-$root}"

# Feature defaults used by this local clone (overridable via .env).
export SENTRIX_THIN_AGENT_V1="${SENTRIX_THIN_AGENT_V1:-1}"
export SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1="${SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1:-1}"
export SENTRIX_IMAGE_EMBEDDER="${SENTRIX_IMAGE_EMBEDDER:-chinese_clip}"
export SENTRIX_TEXT_EMBEDDER="${SENTRIX_TEXT_EMBEDDER:-clip}"
export SENTRIX_MODEL_SPLIT_V1="${SENTRIX_MODEL_SPLIT_V1:-1}"
export SENTRIX_ANN_INDEX_V1="${SENTRIX_ANN_INDEX_V1:-1}"
export SENTRIX_CORE_MEMORY_V1="${SENTRIX_CORE_MEMORY_V1:-1}"
export SENTRIX_EVIDENCE_RETRIEVAL_V1="${SENTRIX_EVIDENCE_RETRIEVAL_V1:-1}"
export FACE_PROVIDERS="${FACE_PROVIDERS:-CPUExecutionProvider}"
export FACE_EMBEDDING_MODE="${FACE_EMBEDDING_MODE:-legacy}"

web_port="$PORT"
api_port="$SENTRIX_API_PORT"
log_dir="${SENTRIX_LOG_DIR:-$SENTRIX_DATA_DIR/logs}"
api_log="$log_dir/api-${api_port}.log"
web_log="$log_dir/web-${web_port}.log"
restart=0
status_only=0

usage() {
  cat <<'EOF'
Usage: ./start.sh [-r|--restart] [--status] [-h|--help]

Start this clone's Web (PORT) and API (SENTRIX_API_PORT) from .env.

  -r, --restart   Stop listeners on these ports first, then start
  --status        Print port / health status and exit
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--restart) restart=1; shift ;;
    --status) status_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

stop_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN | xargs -r kill >/dev/null 2>&1 || true
  fi
}

show_status() {
  local api_code web_code
  api_code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${api_port}/api/health" 2>/dev/null || echo 000)"
  web_code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${web_port}/" 2>/dev/null || echo 000)"
  echo "Web  :${web_port}  listen=$(port_in_use "$web_port" && echo yes || echo no)  http=${web_code}"
  echo "API  :${api_port}  listen=$(port_in_use "$api_port" && echo yes || echo no)  http=${api_code}"
  echo "DB   ${SENTRIX_DB_PATH}"
  echo "logs ${log_dir}"
}

if [[ "$status_only" -eq 1 ]]; then
  show_status
  exit 0
fi

mkdir -p "$SENTRIX_DATA_DIR/media" "$SENTRIX_ANN_DIR" "$log_dir"

if [[ "$restart" -eq 1 ]]; then
  echo "Stopping existing listeners on :$web_port and :$api_port ..."
  stop_port "$web_port"
  stop_port "$api_port"
  sleep 1
else
  if port_in_use "$api_port"; then
    echo "API port :$api_port already in use. Use ./start.sh -r to restart." >&2
    exit 1
  fi
  if port_in_use "$web_port"; then
    echo "Web port :$web_port already in use. Use ./start.sh -r to restart." >&2
    exit 1
  fi
fi

echo "Starting API on :$api_port ..."
nohup bash "$root/scripts/runtime/start_sentrix_api.sh" >>"$api_log" 2>&1 &
api_pid=$!

echo "Starting Web on :$web_port ..."
(
  cd "$root"
  nohup npm start >>"$web_log" 2>&1 &
)
web_pid=$!

# Wait briefly for health.
ok_api=0
ok_web=0
for _ in $(seq 1 30); do
  if [[ "$ok_api" -eq 0 ]] && curl -sf -m 1 "http://127.0.0.1:${api_port}/api/health" >/dev/null 2>&1; then
    ok_api=1
  fi
  if [[ "$ok_web" -eq 0 ]] && curl -sf -m 1 "http://127.0.0.1:${web_port}/" >/dev/null 2>&1; then
    ok_web=1
  fi
  if [[ "$ok_api" -eq 1 && "$ok_web" -eq 1 ]]; then
    break
  fi
  sleep 0.5
done

echo
show_status
echo
echo "Web:  http://0.0.0.0:${web_port}"
echo "API:  http://0.0.0.0:${api_port}/api/health"
echo "API log: $api_log"
echo "Web log: $web_log"

if [[ "$ok_api" -ne 1 || "$ok_web" -ne 1 ]]; then
  echo "Warning: health check incomplete; inspect logs above." >&2
  exit 1
fi

echo "Ready."
