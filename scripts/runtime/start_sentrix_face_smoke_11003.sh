#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1

project_root="/home/asus/Github/Sentrix-Home-Web"
runtime_root="/home/asus/sentrix-benchmarks/photobench-face-smoke-20260807"
port="${SENTRIX_FACE_SMOKE_PORT:-11003}"
pid_file="$runtime_root/sentrix-api.pid"
log_file="$runtime_root/sentrix-api.log"
unit="sentrix-face-smoke-11003.service"

mkdir -p "$runtime_root/data"

if systemctl --user is-active --quiet "$unit"; then
  echo "Face smoke API is already running as $unit" >&2
  exit 1
fi

if ss -ltn | grep -q ":${port} "; then
  echo "Port $port is already in use" >&2
  exit 1
fi

export SENTRIX_API_PORT="$port"
export SENTRIX_DATA_DIR="$runtime_root/data"
export SENTRIX_DB_PATH="$runtime_root/data/sentrix.db"
export FACE_EMBEDDING_MODE="adaface"
export ADAFACE_MODEL_PATH="/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt"
export ADAFACE_REPO_ROOT="/home/asus/models/AdaFace"
export ADAFACE_DEVICE="cpu"
export FACE_PROVIDERS="CPUExecutionProvider"
export SENTRIX_LLM_BACKEND="ollama"
export SENTRIX_PARSE_BACKEND="ollama_local"
export OLLAMA_BASE_URL="http://127.0.0.1:11435"
export OLLAMA_MODEL="gemma4:12b"

systemd-run --user \
  --unit="${unit%.service}" \
  --collect \
  --property="WorkingDirectory=$project_root" \
  --property="StandardOutput=append:$log_file" \
  --property="StandardError=append:$log_file" \
  --setenv="SENTRIX_API_PORT=$SENTRIX_API_PORT" \
  --setenv="SENTRIX_DATA_DIR=$SENTRIX_DATA_DIR" \
  --setenv="SENTRIX_DB_PATH=$SENTRIX_DB_PATH" \
  --setenv="FACE_EMBEDDING_MODE=$FACE_EMBEDDING_MODE" \
  --setenv="ADAFACE_MODEL_PATH=$ADAFACE_MODEL_PATH" \
  --setenv="ADAFACE_REPO_ROOT=$ADAFACE_REPO_ROOT" \
  --setenv="ADAFACE_DEVICE=$ADAFACE_DEVICE" \
  --setenv="FACE_PROVIDERS=$FACE_PROVIDERS" \
  --setenv="PYTHONNOUSERSITE=$PYTHONNOUSERSITE" \
  --setenv="SENTRIX_LLM_BACKEND=$SENTRIX_LLM_BACKEND" \
  --setenv="SENTRIX_PARSE_BACKEND=$SENTRIX_PARSE_BACKEND" \
  --setenv="OLLAMA_BASE_URL=$OLLAMA_BASE_URL" \
  --setenv="OLLAMA_MODEL=$OLLAMA_MODEL" \
  bash scripts/runtime/start_sentrix_api.sh

pid="$(systemctl --user show "$unit" -p MainPID --value)"
echo "$pid" >"$pid_file"

echo "Started face smoke API: pid=$pid port=$port log=$log_file data=$SENTRIX_DATA_DIR"
