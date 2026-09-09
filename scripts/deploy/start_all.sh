#!/usr/bin/env bash
# ============================================================
# Sentrix 服务一键启动（与 153 当前运行环境/选型对齐）
#
# 对齐内容（以代码为准）：
#   - 检索方案锁定  backend/embeddings/scheme.py：
#        IMAGE=chinese_clip (chinese-clip-ViT-L-14, 768d)   [进程内]
#        TEXT =bge         (BAAI/bge-m3, 1024d)              [8101 文本嵌入 sidecar]
#   - 12B 生成模型 gemma4-12b-it @8100（由 vllm manager 8500 托管）
#   - 8091 Sentrix API（backend.app，.venv，env 与 153 一致，见 START_8091_ENV）
#   - 8771 PhotoBench orchestrator
#   - Judge：远程 doubao（ark），配置见 services/photobench/config/runtime_connection.json
#
# 用法： bash scripts/deploy/start_all.sh
# 可覆盖变量：SENTRIX_HOME / VLLM_MANAGER_REPO / GEM_START_CMD 等（见下方）
# ============================================================
set -uo pipefail

SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VLLM_MANAGER_REPO="${VLLM_MANAGER_REPO:-/home/asus/sentrix-vllm}"   # 外部 vllm 托管 repo
TEXT_EMBED_PORT=8101
SENTRIX_PORT=8091
ORCH_PORT=8771

log() { echo "[start_all $(date +%H:%M:%S)] $*"; }
up() { ss -ltn 2>/dev/null | grep -q ":$1 " || nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }

cd "$SENTRIX_HOME" || { echo "no SENTRIX_HOME=$SENTRIX_HOME"; exit 1; }

# ---------- 0) 12B vLLM @8100（如未就绪，通过 manager 启动 profile gemma4-12b-it） ----------
if ! curl -sf -m 5 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
  log "启动 12B vLLM gemma4-12b-it @8100（manager $VLLM_MANAGER_REPO）"
  if [ -x "$VLLM_MANAGER_REPO/bin/sentrix_vllm_manager.py" ]; then
    ( cd "$VLLM_MANAGER_REPO" && setsid nohup bash -c '
        source /home/asus/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate sentrix-vllm 2>/dev/null
        python bin/sentrix_vllm_manager.py start gemma4-12b-it ' >/tmp/vllm-gemma.log 2>&1 < /dev/null & )
  else
    log "WARN: 未找到 manager 脚本，请手动确保 8100 已服务 gemma4-12b-it（registry 见 configs/sentrix_vllm_registry_192_168_0_153.json）"
  fi
  # 等待就绪（最多 240s）
  for _ in $(seq 1 48); do curl -sf -m 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && break; sleep 5; done
fi

# ---------- 1) 文本嵌入 sidecar @8101（bge，BAAI/bge-m3） ----------
if ! curl -sf -m 3 http://127.0.0.1:8101/embed >/dev/null 2>&1; then
  log "启动文本嵌入 sidecar（bge @8101）：scripts/maintenance/text_embedder_sidecar.py"
  if [ ! -x .venv-text/bin/python ]; then
    log "缺少 .venv-text（运行 scripts/deploy/install_env.sh 先装）"; exit 1
  fi
  setsid nohup .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py \
      >/tmp/text-embed.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:8101/embed >/dev/null 2>&1 && break; sleep 2; done
fi

# ---------- 2) Sentrix API @8091（env 与 153 start_sentrix_api_8091.sh 一致；
#               TEXT embedder 声明为 bge——查询侧由 embeddings/scheme.py 锁定，
#               不再受 env 切换） ----------
if ! up $SENTRIX_PORT; then
  log "启动 Sentrix API @$SENTRIX_PORT"
  [ -x .venv/bin/python ] || { log "缺少 .venv（运行 install_env.sh）"; exit 1; }
  cat > /tmp/sentrix-8091.env <<'ENV'
SENTRIX_API_PORT=8091
SENTRIX_VLLM_REGISTRY=configs/sentrix_vllm_registry_192_168_0_153.json
SENTRIX_VLLM_BASE_URL=http://127.0.0.1:8100/v1
E2B_BASE_URL=http://127.0.0.1:8101
SENTRIX_THIN_AGENT_V1=1
SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1
SENTRIX_IMAGE_EMBEDDER=chinese_clip
SENTRIX_TEXT_EMBEDDER=bge
SENTRIX_MODEL_SPLIT_V1=1
SENTRIX_AGENT_MODEL_PROFILE=quality_12b
SENTRIX_AGENT_STAGE_TRACE=1
SENTRIX_EVIDENCE_ANSWER_12B=1
CLIP_DEVICE=cpu
SENTRIX_RX_V1=1
SENTRIX_CONVERSATION_STORE_V1=1
SENTRIX_ANSWER_BRIEF_V1=1
SENTRIX_RESPONSE_PLAN_V1=1
SENTRIX_VISIBLE_EVIDENCE_V1=1
SENTRIX_RESPONSE_WRITER_V2=1
SENTRIX_RESPONSE_VALIDATOR_V1=1
ENV
  setsid nohup env $(grep -v '^#' /tmp/sentrix-8091.env | xargs) \
      .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port $SENTRIX_PORT \
      >/tmp/sentrix-web.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do up $SENTRIX_PORT && break; sleep 2; done
fi

# ---------- 3) PhotoBench orchestrator @8771 ----------
if ! up $ORCH_PORT; then
  log "启动 orchestrator @$ORCH_PORT"
  setsid nohup python3 services/photobench/backend/benchmark_orchestrator.py \
      --host 0.0.0.0 --port $ORCH_PORT >/tmp/orch8771.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do up $ORCH_PORT && break; sleep 2; done
fi

echo
log "完成：8100 vLLM(12B) / 8101 文本嵌入(bge) / $SENTRIX_PORT Sentrix API / $ORCH_PORT orchestrator"
ss -ltnp 2>/dev/null | grep -E ":8100 |:8101 |:$SENTRIX_PORT |:$ORCH_PORT " || true
