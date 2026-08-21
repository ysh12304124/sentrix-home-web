#!/bin/bash
# Sentrix production API (8091). RX answer pipeline enabled (validated 14/14 on 8092).
# 12B-FC validation flags are intentionally NOT here — those are test-only.
cd /home/asus/Github/Sentrix-Home-Web
# AdaFace checkpoint unpickles torchmetrics, which needs the user-site
# transformers package in the current Python environment.  Some shells export
# PYTHONNOUSERSITE=1, which hides that package and turns an AdaFace load into a
# fallback even though the checkpoint is present.
unset PYTHONNOUSERSITE
exec env SENTRIX_API_PORT=8091 SENTRIX_THIN_AGENT_V1=1 SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1 SENTRIX_IMAGE_EMBEDDER=chinese_clip SENTRIX_TEXT_EMBEDDER=clip SENTRIX_MODEL_SPLIT_V1=1 SENTRIX_AGENT_MODEL_PROFILE=quality_12b SENTRIX_AGENT_STAGE_TRACE=1 SENTRIX_CONVERSATION_STORE_V1=1 SENTRIX_EVIDENCE_ANSWER_12B=1 CLIP_DEVICE=cpu \
  FACE_PROVIDERS=CPUExecutionProvider FACE_EMBEDDING_MODE=adaface ADAFACE_DEVICE=cpu \
  ADAFACE_MODEL_PATH=/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt ADAFACE_REPO_ROOT=/home/asus/models/AdaFace \
  SENTRIX_RX_V1=1 SENTRIX_AGENT2_ANSWER_CONTEXT=1 SENTRIX_ANSWER_BRIEF_V1=1 SENTRIX_RESPONSE_PLAN_V1=1 SENTRIX_VISIBLE_EVIDENCE_V1=1 \
  SENTRIX_RESPONSE_WRITER_V2=1 SENTRIX_RESPONSE_VALIDATOR_V1=1 \
  PYTHONPATH=. .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8091
