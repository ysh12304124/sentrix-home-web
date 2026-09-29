#!/usr/bin/env bash
set -euo pipefail

# AdaFace checkpoint unpickling needs the user-site transformers package in this
# runtime.  The incompatible torch<->transformers symbol is now handled inside
# AdaFaceAdapter._load_model, so hiding user-site packages is no longer needed.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-0}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a
fi
python_bin="${SENTRIX_PYTHON:-$root/.venv/bin/python}"
port="${SENTRIX_API_PORT:-8090}"
export SENTRIX_DATA_DIR="${SENTRIX_DATA_DIR:-$root/data}"
export SENTRIX_DB_PATH="${SENTRIX_DB_PATH:-$SENTRIX_DATA_DIR/sentrix.db}"
export SENTRIX_ANN_DIR="${SENTRIX_ANN_DIR:-$SENTRIX_DATA_DIR/ann}"
export SENTRIX_VECTOR_BACKEND="${SENTRIX_VECTOR_BACKEND:-sqlite}"
export SENTRIX_QDRANT_PATH="${SENTRIX_QDRANT_PATH:-$SENTRIX_DATA_DIR/qdrant}"
mkdir -p "$SENTRIX_DATA_DIR/media" "$SENTRIX_ANN_DIR"

if [[ ! -x "$python_bin" ]]; then
  echo "Sentrix Python runtime is not executable: $python_bin" >&2
  exit 1
fi

site_packages="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
runtime_dirs=()
while IFS= read -r directory; do
  runtime_dirs+=("$directory")
done < <(find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort)
# GPU face detection (RetinaFace + buffalo_l onnxruntime CUDA) needs cudnn/cublas
# shipped in the stmem conda env; the project .venv does not vendor nvidia libs.
while IFS= read -r directory; do
  runtime_dirs+=("$directory")
done < <(find /home/orin/miniconda3/envs/sentrix/lib/python3.10/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort)

if ((${#runtime_dirs[@]})); then
  runtime_path="$(IFS=:; echo "${runtime_dirs[*]}")"
  export LD_LIBRARY_PATH="${runtime_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export FACE_PROVIDERS="${FACE_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
export SENTRIX_LLM_BACKEND="${SENTRIX_LLM_BACKEND:-vllm}"
export SENTRIX_VLLM_BASE_URL="${SENTRIX_VLLM_BASE_URL:-http://127.0.0.1:8100/v1}"
export SENTRIX_VLLM_MODEL="${SENTRIX_VLLM_MODEL:-gemma4-12b-it}"
export SENTRIX_VLLM_REGISTRY="${SENTRIX_VLLM_REGISTRY:-$root/configs/sentrix_vllm_registry_192_168_0_153.json}"
# Batch image work is configurable, but the backend also caps the effective
# value at the active vLLM profile's max_num_seqs.
export SENTRIX_PIPELINE_MAX_WORKERS="${SENTRIX_PIPELINE_MAX_WORKERS:-2}"
export SENTRIX_EVENT_SUMMARY_MAX_WORKERS="${SENTRIX_EVENT_SUMMARY_MAX_WORKERS:-2}"
# Legacy Ollama settings are kept only for explicit SENTRIX_LLM_BACKEND=ollama fallback.
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:12b}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export E2B_BASE_URL="${E2B_BASE_URL:-http://127.0.0.1:8101}"
# CLIP_DEVICE 用 cuda。原注释理由（"153 GPU driver/library NVML mismatch breaks
# the CUDA caching allocator"）是 **153 的驱动问题**，照搬到 118 会把 chinese-clip
# 摁在 CPU 上。118 的 GPU 是好的：llama-server 跑 --n-gpu-layers 999、人脸走
# CUDAExecutionProvider、8101 bge sidecar 也在 GPU 上。实测同一模型同一 dim：
#   cpu 13.12 s/张  vs  cuda 0.05 s/张  ≈ 260 倍
# 查询侧（视觉路由的文本编码）同样受益。若将来 118 的 CUDA 真的不可用，
# 在 .env 里覆盖 CLIP_DEVICE=cpu 即可回退，无需改本文件。
export CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
export CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/home/orin/lingbot-map-stmem/models/open_clip_pytorch_model.bin}"
export CHINESE_CLIP_CHECKPOINT="${CHINESE_CLIP_CHECKPOINT:-/home/orin/.cache/clip/clip_cn_vit-l-14.pt}"
# R1B proved ViT-B-32 text-to-image is random for Chinese (AUC 0.51); switch the
# visual slot to Chinese-CLIP ViT-L-14 (D3).
#
# TEXT slot 也必须是 bge：data/ann/{episodic,semantic}.manifest.json 记的是
# BAAI/bge-m3 1024 维，8101 sidecar 也跑 bge-m3。此前这里默认 clip，查询侧与
# 索引侧模型不一致——检索不报错，但文本语义通道静默失配。153 生产脚本同样锁定 bge。
export SENTRIX_IMAGE_EMBEDDER="${SENTRIX_IMAGE_EMBEDDER:-chinese_clip}"
export SENTRIX_TEXT_EMBEDDER="${SENTRIX_TEXT_EMBEDDER:-bge}"

# --- Phase 0-8 + 2R feature flags (default on for production) ---
export SENTRIX_THIN_AGENT_V1="${SENTRIX_THIN_AGENT_V1:-1}"
export SENTRIX_SEMANTIC_QUERY_PARSER_V1="${SENTRIX_SEMANTIC_QUERY_PARSER_V1:-1}"
export SENTRIX_EVIDENCE_RETRIEVAL_V1="${SENTRIX_EVIDENCE_RETRIEVAL_V1:-1}"
export SENTRIX_LLM_CLAIM_EXTRACTOR_V1="${SENTRIX_LLM_CLAIM_EXTRACTOR_V1:-1}"
export SENTRIX_CORE_MEMORY_V1="${SENTRIX_CORE_MEMORY_V1:-1}"
export SENTRIX_MEMORY_CORRECTION_V1="${SENTRIX_MEMORY_CORRECTION_V1:-1}"
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

if [[ "$SENTRIX_TEXT_EMBEDDER" == "clip" ]]; then
  if [[ ! -f "$CLIP_CHECKPOINT" ]]; then
    echo "OpenCLIP checkpoint is unavailable: $CLIP_CHECKPOINT" >&2
    exit 1
  fi
  if ! "$python_bin" -c 'import open_clip' >/dev/null 2>&1; then
    echo "OpenCLIP Python dependency is unavailable; install backend/requirements.txt" >&2
    exit 1
  fi
fi

if [[ "$SENTRIX_IMAGE_EMBEDDER" == "chinese_clip" ]]; then
  if [[ ! -f "$CHINESE_CLIP_CHECKPOINT" ]]; then
    echo "Chinese-CLIP checkpoint is unavailable: $CHINESE_CLIP_CHECKPOINT" >&2
    exit 1
  fi
  if ! "$python_bin" -c 'import cn_clip' >/dev/null 2>&1; then
    echo "Chinese-CLIP Python dependency is unavailable; install backend/requirements.txt" >&2
    exit 1
  fi
fi

# Face identity embedding is fixed to buffalo_l (w600k_r50); AdaFace is removed.

exec "$python_bin" -m uvicorn backend.app:app --host "${SENTRIX_API_HOST:-0.0.0.0}" --port "$port"
