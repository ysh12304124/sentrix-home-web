#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="$root/.ollama-sentrix"
host="${SENTRIX_OLLAMA_HOST:-127.0.0.1:11435}"
models_dir="${SENTRIX_OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"

mkdir -p "$runtime_dir/logs"

if curl -fsS --max-time 2 "http://${host}/api/version" >/dev/null; then
  echo "Sentrix Ollama is already listening on ${host}"
  exit 0
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/ps | grep -q '"models":\[\]'; then
  echo "Refusing to start: shared Ollama has a resident model on 11434." >&2
  exit 1
fi

nohup env \
  OLLAMA_HOST="$host" \
  OLLAMA_MODELS="$models_dir" \
  OLLAMA_KEEP_ALIVE=0 \
  /usr/local/bin/ollama serve \
  >"$runtime_dir/logs/ollama.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$runtime_dir/ollama.pid"
echo "Started Sentrix Ollama on ${host} (pid ${pid})"
