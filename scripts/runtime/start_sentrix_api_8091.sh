#!/bin/bash
# Sentrix production API (8091). RX answer pipeline enabled (validated 14/14 on 8092).
# 12B-FC validation flags are intentionally NOT here — those are test-only.
# bge-m3 text embedder sidecar keepalive（SENTRIX_TEXT_EMBEDDER=bge 依赖）
if ! curl -s -m 2 http://127.0.0.1:8101/health >/dev/null 2>&1; then
  echo starting bge sidecar
  cd /home/asus/Github/Sentrix-Home-Web
  PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 SENTRIX_TEXT_EMBEDDER_DEVICE=cpu     setsid nohup .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py > /tmp/bge_sidecar.log 2>&1 < /dev/null &
  sleep 5
fi
cd /home/asus/Github/Sentrix-Home-Web
# AdaFace checkpoint unpickles torchmetrics, which needs the user-site
# transformers package in the current Python environment.  Some shells export
# PYTHONNOUSERSITE=1, which hides that package and turns an AdaFace load into a
# fallback even though the checkpoint is present.
unset PYTHONNOUSERSITE
NVIDIA_RUNTIME_ROOT=/home/asus/Github/Sentrix-Home-Web/.venv/lib/python3.10/site-packages/nvidia
NVIDIA_RUNTIME_LIBS=$(find "$NVIDIA_RUNTIME_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="${NVIDIA_RUNTIME_LIBS}${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}"
exec env CUDA_VISIBLE_DEVICES=0 SENTRIX_PIPELINE_MAX_WORKERS=4 SENTRIX_PIPELINE_MAX_RETRIES=1 SENTRIX_EVENT_SUMMARY_MAX_WORKERS=16 SENTRIX_FACE_GPU_SESSION_LIMIT_MIB=1024 SENTRIX_RETINAFACE_GPU_SESSION_LIMIT_MIB=768 SENTRIX_FACE_GPU_MAX_CONCURRENCY=4 SENTRIX_API_PORT=8091 SENTRIX_THIN_AGENT_V1=1 SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1 SENTRIX_IMAGE_EMBEDDER=chinese_clip SENTRIX_TEXT_EMBEDDER=bge SENTRIX_MODEL_SPLIT_V1=1 SENTRIX_AGENT_MODEL_PROFILE=quality_12b SENTRIX_AGENT_STAGE_TRACE=1 SENTRIX_CONVERSATION_STORE_V1=1 SENTRIX_EVIDENCE_ANSWER_12B=1 CLIP_DEVICE=cpu \
  FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider RETINAFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider FACE_EMBEDDING_MODE=adaface ADAFACE_DEVICE=cpu \
  ADAFACE_MODEL_PATH=/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt ADAFACE_REPO_ROOT=/home/asus/models/AdaFace \
  SENTRIX_CANONICAL_SEARCH=1 SENTRIX_RX_V1=1 SENTRIX_AGENT2_ANSWER_CONTEXT=1 SENTRIX_ANSWER_BRIEF_V1=1 SENTRIX_RESPONSE_PLAN_V1=1 SENTRIX_VISIBLE_EVIDENCE_V1=1 \
  SENTRIX_RESPONSE_WRITER_V2=1 SENTRIX_RESPONSE_VALIDATOR_V1=1 SENTRIX_VIDEO_KEYFRAME_ALGORITHM=hybrid_webp SENTRIX_VECTOR_BACKEND=qdrant SENTRIX_QDRANT_PATH=/home/asus/Github/Sentrix-Home-Web/data/qdrant \
  PYTHONPATH=. .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8091
