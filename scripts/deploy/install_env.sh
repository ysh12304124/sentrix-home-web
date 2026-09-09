#!/usr/bin/env bash
# ============================================================
# Sentrix 环境一键安装（对齐 153）
# 创建三个运行环境：
#   .venv       → 后端/检索/8091（backend/requirements.txt，含 hnswlib/chinese-clip）
#   .venv-text  → 文本嵌入 sidecar（bge，8101）
#   frontend    → npm install + build（photobench 前端 → dist）
# 模型选型（代码锁定 backend/embeddings/scheme.py）：
#   IMAGE=chinese_clip(chinese-clip-ViT-L-14)   TEXT=bge(BAAI/bge-m3)
# 12B vLLM gemma4-12b-it 由外部 sentrix-vllm manager 托管，脚本不装（见 README）。
# ============================================================
set -euo pipefail
SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$SENTRIX_HOME"

PY="${PY:-python3}"
PY_VER_OK=$("$PY" -c 'import sys;print(sys.version_info[:2]>= (3,10))' 2>/dev/null || echo False)
echo "使用 $PY ($($PY --version 2>&1))  python3.10+ 检测=$PY_VER_OK"

# ---------- 1) 后端 .venv ----------
if [ ! -x .venv/bin/python ]; then
  echo "[install] 创建 .venv（后端/检索/8091）"
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -U pip wheel >/dev/null
if [ -f backend/requirements.txt ]; then
  .venv/bin/python -m pip install -r backend/requirements.txt
else
  echo "WARN: 无 backend/requirements.txt，请按仓库实际依赖补齐"
fi

# ---------- 2) 文本嵌入 sidecar .venv-text（bge 8101） ----------
if [ ! -x .venv-text/bin/python ]; then
  echo "[install] 创建 .venv-text（text_embedder_sidecar, BAAI/bge-m3）"
  "$PY" -m venv .venv-text
fi
.venv-text/bin/python -m pip install -U pip wheel >/dev/null
.venv-text/bin/python -m pip install "torch" "sentence-transformers" "fastapi" "uvicorn[standard]" "httpx" "numpy" "transformers"

# ---------- 3) photobench 前端 ----------
echo "[install] 前端 npm install + build"
if [ -d services/photobench/frontend ]; then
  ( cd services/photobench/frontend && npm ci 2>/dev/null || npm install )
  ( cd services/photobench/frontend && npm run build )
fi

echo
echo "安装完成。运行： bash scripts/deploy/start_all.sh"
echo "外部前置：vLLM gemma4-12b-it@8100（sentrix-vllm manager 8500）；judge doubao(ark) key；SQLite 库 data/sentrix.db"
