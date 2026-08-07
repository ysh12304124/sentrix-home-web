#!/bin/bash
cd /home/asus/Github/zhx/sentrix-home-web
NVIDIA_LIBS=$(find /home/asus/Github/Sentrix-Home-Web/.venv/lib/python3.10/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /tmp/backend_9598.log
exec env FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
    OLLAMA_BASE_URL=http://127.0.0.1:11435 \
    OLLAMA_MODEL=gemma4:12b \
    OLLAMA_KEEP_ALIVE=-1 \
    CLIP_DEVICE=cpu \
    FACE_EMBEDDING_MODE=legacy \
    ADAFACE_MODEL_PATH=/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt \
    ADAFACE_REPO_ROOT=/home/asus/models/AdaFace \
    SENTRIX_API_PORT=9598 \
    /home/asus/Github/Sentrix-Home-Web/.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 9598
