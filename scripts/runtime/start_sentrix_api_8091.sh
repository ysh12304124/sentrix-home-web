#!/bin/bash
# Sentrix production API (8091). RX answer pipeline enabled (validated 14/14 on 8092).
# 12B-FC validation flags are intentionally NOT here — those are test-only.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SENTRIX_DATA_DIR="${SENTRIX_DATA_DIR:-$root/data}"
CANDIDATE_STRATEGY="${SENTRIX_CANDIDATE_STRATEGY:-relevance_head_then_event_diversity}"
# bge-m3 text embedder sidecar keepalive（SENTRIX_TEXT_EMBEDDER=bge 依赖）
if ! curl -s -m 2 http://127.0.0.1:8101/health >/dev/null 2>&1; then
  echo starting bge sidecar
  cd "$root"
  PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 SENTRIX_TEXT_EMBEDDER_DEVICE=cpu     setsid nohup .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py > /tmp/bge_sidecar.log 2>&1 < /dev/null &
  sleep 5
fi
cd "$root"
# AdaFace checkpoint unpickles torchmetrics, which needs the user-site
# transformers package in the current Python environment.  Some shells export
# PYTHONNOUSERSITE=1, which hides that package and turns an AdaFace load into a
# fallback even though the checkpoint is present.
unset PYTHONNOUSERSITE
NVIDIA_RUNTIME_ROOT=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia")')
NVIDIA_RUNTIME_LIBS=$(find "$NVIDIA_RUNTIME_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="${NVIDIA_RUNTIME_LIBS}${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}"
# Fail visibly instead of silently serving empty face/visual evidence.
.venv/bin/python - <<'PY' || exit 1
from pathlib import Path
import os
import cn_clip
import onnxruntime
required = [
    # 这段 heredoc 是单引号（<<'PY'），shell **不会**展开 $root —— 写 $root/... 会
    # 原样传给 Python 变成字面量路径。脚本此前已 cd "$root"，这里用 cwd 自己算。
    # 默认值与 backend/face_detector.py 保持一致（$HOME 推导），不在这里另写一份。
    Path(os.getenv("RETINAFACE_MODEL_PATH",
                   str(Path.home() / "benchmarks" / "retinaface" / "retinaface_r50.onnx"))),
    Path(os.getenv("CHINESE_CLIP_CHECKPOINT", str(Path.home() / ".cache/clip/clip_cn_vit-l-14.pt"))),
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing retrieval/face weights: " + ", ".join(missing))
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise SystemExit("ONNX Runtime CUDA provider missing; install onnxruntime-gpu without CPU package shadowing")
PY
# Bound turn workers because each worker can retain visual/embed resources; batch fan-out otherwise causes process-level OOM.
BIG_MODEL_ENV_FILE="${SENTRIX_BIG_MODEL_ENV_FILE:-$HOME/.config/sentrix/big_model.env}"
if [[ -f "$BIG_MODEL_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$BIG_MODEL_ENV_FILE"
  set +a
fi
# Match the 153/Gemma production context standard: 8192 tokens for every model.
# Cloud API calls use their separate provider output budget.
# 平台参数层：三台共用一份代码，机器相关的「人为选择」集中在那里；能探测的
# （CUDA 是否真能用、ffmpeg 有哪些硬解、内存多大、装了 qdrant 没有）交给
# backend/platform_profile.py。放在 exec 之前，让下面 exec env 里的显式值仍能覆盖。
if [[ -f "$root/scripts/deploy/platform-profiles.sh" ]]; then
  # shellcheck disable=SC1091
  source "$root/scripts/deploy/platform-profiles.sh"
  sentrix_apply_platform_profile
fi

exec env CUDA_VISIBLE_DEVICES=0 SENTRIX_VLLM_API_URL=http://127.0.0.1:8500 SENTRIX_VLLM_MANAGER_API=http://127.0.0.1:8500 SENTRIX_VLLM_REGISTRY=$root/configs/sentrix_vllm_registry_local_100.json SENTRIX_VLLM_BASE_URL=http://127.0.0.1:8100/v1 SENTRIX_ASSISTANT_TURN_WORKERS=4 SENTRIX_PIPELINE_MAX_WORKERS=12 SENTRIX_PIPELINE_MAX_RETRIES=1 SENTRIX_EVENT_SUMMARY_MAX_WORKERS=16 SENTRIX_FACE_GPU_SESSION_LIMIT_MIB=1024 SENTRIX_RETINAFACE_GPU_SESSION_LIMIT_MIB=768 SENTRIX_FACE_GPU_MAX_CONCURRENCY=4 SENTRIX_API_PORT=8091 SENTRIX_THIN_AGENT_V1=1 SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1 SENTRIX_IMAGE_EMBEDDER=chinese_clip SENTRIX_TEXT_EMBEDDER=bge SENTRIX_MODEL_SPLIT_V1=1 SENTRIX_AGENT_MODEL_PROFILE=quality_12b SENTRIX_AGENT_STAGE_TRACE=1 SENTRIX_CONVERSATION_STORE_V1=1 SENTRIX_EVIDENCE_ANSWER_12B=1 \
  FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider RETINAFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider FACE_EMBEDDING_MODE=adaface ADAFACE_DEVICE=cpu \
  ADAFACE_MODEL_PATH=$root/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt ADAFACE_REPO_ROOT=$root/models/AdaFace \
  SENTRIX_CANONICAL_SEARCH=1 SENTRIX_RX_V1=1 SENTRIX_AGENT2_ANSWER_CONTEXT=1 SENTRIX_ANSWER_BRIEF_V1=1 SENTRIX_RESPONSE_PLAN_V1=1 SENTRIX_VISIBLE_EVIDENCE_V1=1 \
  SENTRIX_RESPONSE_WRITER_V2=1 SENTRIX_RESPONSE_VALIDATOR_V1=1 SENTRIX_AGENT_PROFILE=goal_driven_candidate SENTRIX_TOOL_LOOP_MAX_TOKENS="${SENTRIX_TOOL_LOOP_MAX_TOKENS:-384}" SENTRIX_BIG_MODEL_MAX_OUTPUT_TOKENS="${SENTRIX_BIG_MODEL_MAX_OUTPUT_TOKENS:-4096}" SENTRIX_CANDIDATE_STRATEGY="$CANDIDATE_STRATEGY" SENTRIX_VIDEO_KEYFRAME_ALGORITHM=hybrid_webp SENTRIX_VECTOR_BACKEND=qdrant SENTRIX_QDRANT_PATH=$root/data/qdrant \
  PYTHONPATH=. .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8091
