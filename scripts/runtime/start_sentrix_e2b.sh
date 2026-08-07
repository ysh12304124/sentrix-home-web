#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set +u
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate sentrix-e2b
set -u
export E2B_BASE_MODEL="${E2B_BASE_MODEL:-/home/asus/models/gemma-4-E2B-it}"
export E2B_ADAPTER="${E2B_ADAPTER:-/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47}"
export E2B_HOST="${E2B_HOST:-127.0.0.1}"
export E2B_PORT="${E2B_PORT:-8101}"
export E2B_DTYPE="${E2B_DTYPE:-bf16}"
for f in "$E2B_BASE_MODEL/config.json" "$E2B_BASE_MODEL/processor_config.json"          "$E2B_BASE_MODEL/tokenizer.json" "$E2B_BASE_MODEL/model.safetensors"          "$E2B_ADAPTER/adapter_config.json" "$E2B_ADAPTER/adapter_model.safetensors"; do
  [ -f "$f" ] || { echo "[start_sentrix_e2b] missing: $f" >&2; exit 1; }
done
exec uvicorn services.e2b_server.app:app --host "$E2B_HOST" --port "$E2B_PORT" --workers 1
