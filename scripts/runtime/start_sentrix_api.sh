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
exec "$python_bin" -m uvicorn backend.app:app --host "${SENTRIX_API_HOST:-0.0.0.0}" --port "$port"
