#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

stop_pid_file() {
  NAME=$1
  PID_FILE=$2
  if [ ! -f "$PID_FILE" ]; then
    echo "$NAME was not started by this script"
    return
  fi
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
  fi
  unlink "$PID_FILE"
  echo "$NAME stopped"
}

stop_pid_file "Vite frontend" "$ROOT_DIR/logs/vite.pid"
stop_pid_file "Backend" "$ROOT_DIR/logs/orchestrator.pid"
