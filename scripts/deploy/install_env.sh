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
SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
cd "$SENTRIX_HOME" || { echo "no SENTRIX_HOME=$SENTRIX_HOME"; exit 1; }
[ -f backend/app.py ] || { echo "SENTRIX_HOME 不正确（找不到 backend/app.py）: $SENTRIX_HOME"; exit 1; }

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
# chinese-clip 不在 backend/requirements.txt 里（它只在查询侧/向量补齐用），
# 但它是**生产检索的 IMAGE embedder**：不装 → chinese_clip 适配器 unavailable
# → visual_ann 静默 no_candidates（不报错，检索质量悄悄退化）。
# 历史事故：写入侧 ViT-B-32、查询侧 chinese-clip 两套向量并存。
echo "[install] 安装 chinese-clip（IMAGE embedder，检索方案锁定项）"
.venv/bin/python -m pip install "cn-clip==1.6.0"

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

# ---------- 4) chinese-clip 权重检查（1.6GB，不自动下载） ----------
CN_CLIP_CKPT="${CHINESE_CLIP_CHECKPOINT:-$HOME/.cache/clip/clip_cn_vit-l-14.pt}"
if [ -f "$CN_CLIP_CKPT" ]; then
  echo "[check] chinese-clip 权重就绪：$CN_CLIP_CKPT"
else
  cat <<EOF
[check] !! 缺少 chinese-clip 权重：$CN_CLIP_CKPT

  这是生产检索 IMAGE embedder 的必需权重，缺失时 visual_ann 会静默 no_candidates
  （不报错，但图片语义召回全空）。请先下载（约 1.6GB）：

    wget -P "\$(dirname "$CN_CLIP_CKPT")" \\
      https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14.pt

  如放在别处，用环境变量指过来： export CHINESE_CLIP_CHECKPOINT=/path/to/clip_cn_vit-l-14.pt
EOF
fi

echo
echo "安装完成。运行： bash scripts/deploy/start_all.sh"
echo "外部前置：vLLM gemma4-12b-it@8100（sentrix-vllm manager 8500）；judge doubao(ark) key；SQLite 库 data/sentrix.db"
