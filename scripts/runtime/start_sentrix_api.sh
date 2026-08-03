#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${SENTRIX_PYTHON:-$root/.venv/bin/python}"
port="${SENTRIX_API_PORT:-8090}"

if [[ ! -x "$python_bin" ]]; then
  echo "Sentrix Python runtime is not executable: $python_bin" >&2
  exit 1
fi

site_packages="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
runtime_dirs=()
while IFS= read -r directory; do
  runtime_dirs+=("$directory")
done < <(find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort)

if ((${#runtime_dirs[@]})); then
  runtime_path="$(IFS=:; echo "${runtime_dirs[*]}")"
  export LD_LIBRARY_PATH="${runtime_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export FACE_PROVIDERS="${FACE_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:12b}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
export FACE_EMBEDDING_MODE="${FACE_EMBEDDING_MODE:-adaface}"
export ADAFACE_MODEL_PATH="${ADAFACE_MODEL_PATH:-/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt}"
export ADAFACE_REPO_ROOT="${ADAFACE_REPO_ROOT:-/home/asus/models/AdaFace}"

if [[ "$FACE_EMBEDDING_MODE" == "adaface" ]]; then
  if [[ ! -f "$ADAFACE_MODEL_PATH" ]]; then
    echo "AdaFace checkpoint is unavailable: $ADAFACE_MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$ADAFACE_REPO_ROOT/net.py" ]]; then
    echo "AdaFace repository is unavailable: $ADAFACE_REPO_ROOT/net.py" >&2
    exit 1
  fi
fi

exec "$python_bin" -m uvicorn backend.app:app --host "${SENTRIX_API_HOST:-0.0.0.0}" --port "$port"
