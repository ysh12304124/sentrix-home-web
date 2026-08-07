#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

# Activate the E2B conda environment
eval "$(conda shell.bash hook)"
conda activate sentrix-e2b

export E2B_BASE_MODEL="${E2B_BASE_MODEL:-/home/asus/models/gemma-4-E2B-it}"
export E2B_ADAPTER="${E2B_ADAPTER:-/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47}"
export E2B_HOST="${E2B_HOST:-127.0.0.1}"
export E2B_PORT="${E2B_PORT:-8100}"

# Hard gate: all required model files must exist
for f in "$E2B_BASE_MODEL/config.json" \
         "$E2B_BASE_MODEL/processor_config.json" \
         "$E2B_BASE_MODEL/tokenizer.json" \
         "$E2B_ADAPTER/adapter_config.json" \
         "$E2B_ADAPTER/adapter_model.safetensors"; do
  if [ ! -f "$f" ]; then
    echo "[start_sentrix_e2b] missing: $f" >&2
    exit 1
  fi
done

# Also check for model weights (safetensors or pytorch bin)
model_weights_found=0
for pattern in "$E2B_BASE_MODEL/model.safetensors" \
               "$E2B_BASE_MODEL/model*.safetensors" \
               "$E2B_BASE_MODEL/pytorch_model.bin"; do
  if compgen -G "$pattern" > /dev/null 2>&1; then
    model_weights_found=1
    break
  fi
done
if [ "$model_weights_found" -eq 0 ]; then
  echo "[start_sentrix_e2b] no model weights found in $E2B_BASE_MODEL" >&2
  exit 1
fi

exec uvicorn services.e2b_server.app:app \
  --host "$E2B_HOST" \
  --port "$E2B_PORT" \
  --workers 1
