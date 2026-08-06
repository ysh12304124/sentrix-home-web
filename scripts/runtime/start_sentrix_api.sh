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
export SENTRIX_LLM_BACKEND="${SENTRIX_LLM_BACKEND:-vllm}"
export SENTRIX_VLLM_BASE_URL="${SENTRIX_VLLM_BASE_URL:-http://127.0.0.1:8100/v1}"
export SENTRIX_VLLM_MODEL="${SENTRIX_VLLM_MODEL:-gemma4-12b-it}"
# Legacy Ollama settings are kept only for explicit SENTRIX_LLM_BACKEND=ollama fallback.
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:12b}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export E2B_BASE_URL="${E2B_BASE_URL:-http://127.0.0.1:8100}"
# 153 GPU driver/library NVML mismatch breaks the CUDA caching allocator; run
# CLIP embedding on CPU so visual/text recall stays available.
export CLIP_DEVICE="${CLIP_DEVICE:-cpu}"
# R1B proved ViT-B-32 text-to-image is random for Chinese (AUC 0.51); switch the
# visual slot to Chinese-CLIP ViT-L-14 (D3).  Text slot stays CLIP (AUC 0.996).
export SENTRIX_IMAGE_EMBEDDER="${SENTRIX_IMAGE_EMBEDDER:-chinese_clip}"
export SENTRIX_TEXT_EMBEDDER="${SENTRIX_TEXT_EMBEDDER:-clip}"
export FACE_EMBEDDING_MODE="${FACE_EMBEDDING_MODE:-adaface}"
export ADAFACE_MODEL_PATH="${ADAFACE_MODEL_PATH:-/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt}"
export ADAFACE_REPO_ROOT="${ADAFACE_REPO_ROOT:-/home/asus/models/AdaFace}"

# --- Phase 0-8 + 2R feature flags (default on for production) ---
export SENTRIX_THIN_AGENT_V1="${SENTRIX_THIN_AGENT_V1:-1}"
export SENTRIX_SEMANTIC_QUERY_PARSER_V1="${SENTRIX_SEMANTIC_QUERY_PARSER_V1:-1}"
export SENTRIX_EVIDENCE_RETRIEVAL_V1="${SENTRIX_EVIDENCE_RETRIEVAL_V1:-1}"
export SENTRIX_LLM_CLAIM_EXTRACTOR_V1="${SENTRIX_LLM_CLAIM_EXTRACTOR_V1:-1}"
export SENTRIX_CORE_MEMORY_V1="${SENTRIX_CORE_MEMORY_V1:-1}"
export SENTRIX_MEMORY_CORRECTION_V1="${SENTRIX_MEMORY_CORRECTION_V1:-1}"
export SENTRIX_ADVANCED_MEMORY_TOOLS_V1="${SENTRIX_ADVANCED_MEMORY_TOOLS_V1:-1}"
export SENTRIX_ANN_INDEX_V1="${SENTRIX_ANN_INDEX_V1:-1}"
export SENTRIX_EXPLICIT_IMAGE_REINSPECTION="${SENTRIX_EXPLICIT_IMAGE_REINSPECTION:-0}"

# --- Phase R flags ---
export SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1="${SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1:-1}"
export SENTRIX_RETRIEVER_METADATA="${SENTRIX_RETRIEVER_METADATA:-1}"
export SENTRIX_RETRIEVER_ENTITY="${SENTRIX_RETRIEVER_ENTITY:-1}"
export SENTRIX_RETRIEVER_LEXICAL="${SENTRIX_RETRIEVER_LEXICAL:-1}"
export SENTRIX_RETRIEVER_VISUAL_ANN="${SENTRIX_RETRIEVER_VISUAL_ANN:-1}"
export SENTRIX_RETRIEVER_TEXT_ANN="${SENTRIX_RETRIEVER_TEXT_ANN:-1}"
export SENTRIX_RETRIEVER_ADJACENCY="${SENTRIX_RETRIEVER_ADJACENCY:-1}"
export SENTRIX_RETRIEVER_FUSION="${SENTRIX_RETRIEVER_FUSION:-rrf}"
export SENTRIX_GATE_PROBE_V1="${SENTRIX_GATE_PROBE_V1:-1}"
export SENTRIX_MODEL_SPLIT_V1="${SENTRIX_MODEL_SPLIT_V1:-0}"
export SENTRIX_PARSE_BACKEND="${SENTRIX_PARSE_BACKEND:-openai}"
export SENTRIX_PARSE_BASE_URL="${SENTRIX_PARSE_BASE_URL:-$SENTRIX_VLLM_BASE_URL}"
export SENTRIX_PARSE_MODEL="${SENTRIX_PARSE_MODEL:-$SENTRIX_VLLM_MODEL}"

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
