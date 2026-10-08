#!/usr/bin/env bash
# ============================================================
# Sentrix 服务一键启动（与 153 生产运行方式一致）
#
# 启动顺序（8091 之前必须完成向量补齐，之后 Qdrant 目录锁被占）：
#   8100 vLLM(12B) → 8101 bge sidecar → [preflight + 视觉向量一致性]
#   → 8091 Sentrix API → 8771 orchestrator → health / 检索探测
#
# 关键设计——不重复维护第二份 env：
#   8091 直接调用生产启动脚本 scripts/runtime/start_sentrix_api_8091.sh。
#   实测教训：在部署脚本里另抄一份 env，漏掉 SENTRIX_VECTOR_BACKEND/SENTRIX_QDRANT_PATH
#   会让向量层静默退化成 SQLite 全表扫（检索"没报错但明显变差"）；漏掉 HF_HUB_OFFLINE
#   会让 bge sidecar 反复连 huggingface 超时起不来。
#
# 用法： bash scripts/deploy/start_all.sh
# 可覆盖：SENTRIX_HOME / VLLM_MANAGER_REPO / TEXT_EMBED_PORT / ORCH_PORT /
#         SENTRIX_ENSURE_SCOPES（指定则自动补齐这些 scope 的视觉向量；留空只检测告警）
# ============================================================
set -uo pipefail

SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
VLLM_MANAGER_REPO="${VLLM_MANAGER_REPO:-/home/asus/sentrix-vllm}"   # 外部 vllm 托管 repo
TEXT_EMBED_PORT="${TEXT_EMBED_PORT:-8101}"
SENTRIX_PORT="${SENTRIX_PORT:-8091}"
ORCH_PORT="${ORCH_PORT:-8771}"

log() { echo "[start_all $(date +%H:%M:%S)] $*"; }
up() { ss -ltn 2>/dev/null | grep -q ":$1 " || nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }

cd "$SENTRIX_HOME" || { echo "no SENTRIX_HOME=$SENTRIX_HOME"; exit 1; }
# 自检：路径解析错一级会 cd 到仓库父目录，后续相对路径全部失效（实测踩过）。
[ -f backend/app.py ] || { echo "SENTRIX_HOME 不正确（找不到 backend/app.py）: $SENTRIX_HOME"; exit 1; }

# ---------- 0) 12B vLLM @8100（如未就绪，通过 manager 启动 profile gemma4-12b-it） ----------
if ! curl -sf -m 5 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
  log "启动 12B vLLM gemma4-12b-it @8100（manager $VLLM_MANAGER_REPO）"
  if [ -x "$VLLM_MANAGER_REPO/bin/sentrix_vllm_manager.py" ]; then
    ( cd "$VLLM_MANAGER_REPO" && setsid nohup bash -c '
        source /home/asus/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate sentrix-vllm 2>/dev/null
        python bin/sentrix_vllm_manager.py start gemma4-12b-it ' >/tmp/vllm-gemma.log 2>&1 < /dev/null & )
  else
    log "WARN: 未找到 manager 脚本（$VLLM_MANAGER_REPO/bin/sentrix_vllm_manager.py）"
  fi
  # 等待就绪（最多 240s）
  for _ in $(seq 1 48); do curl -sf -m 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf -m 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    log "WARN: 8100 未在 240s 内就绪 —— Agent 问答/评测会失败（检索链路本身不受影响）"
    log "      排查：nvidia-smi；tail -20 $VLLM_MANAGER_REPO/log/gemma4-12b-it.log"
    log "      已知坑：vLLM 的平台探测强依赖 NVML（pynvml.nvmlInit 失败即判定无 CUDA 平台）。"
    log "      若报 NVMLError_LibRmVersionMismatch，是内核模块与用户态库版本不一致（驱动升级后未重启），需重启机器。"
  fi
fi

# ---------- 1) 文本嵌入 sidecar @8101（bge，BAAI/bge-m3） ----------
if ! curl -sf -m 3 "http://127.0.0.1:$TEXT_EMBED_PORT/embed" >/dev/null 2>&1; then
  log "启动文本嵌入 sidecar（bge @$TEXT_EMBED_PORT）：scripts/maintenance/text_embedder_sidecar.py"
  if [ ! -x .venv-text/bin/python ]; then
    log "缺少 .venv-text（运行 scripts/deploy/install_env.sh 先装）"; exit 1
  fi
  # HF_HUB_OFFLINE=1 必带：生产机常无外网，联网重试会让 sidecar 一直起不来
  # （实测：不设时反复连 huggingface.co 超时，8101 始终不监听）。
  setsid nohup env PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 \
      SENTRIX_TEXT_EMBEDDER_DEVICE=cpu \
      .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py \
      >/tmp/text-embed.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do curl -sf -m 3 "http://127.0.0.1:$TEXT_EMBED_PORT/embed" >/dev/null 2>&1 && break; sleep 2; done
fi

# ---------- 1.5) 检索 embedder preflight + 视觉向量一致性 ----------
# 必须排在 8091 启动之前：8091 起来后会持有 Qdrant 目录锁，此后再补齐向量会撞锁
# （153 重启 SOP 同样把补齐放在停机窗口，见 scripts/restart_sentrix_8091_153.sh 步骤 3.5）。
#
# 为什么必须做：写入侧的图片向量历史来自 ViT-B-32（通用 CLIP），而查询侧锁定
# chinese-clip；visual_ann 检索前会要求库里存在查询模型的向量，缺了就**静默**
# no_candidates——不报错，只是图片语义召回全空。
if ! up $SENTRIX_PORT; then
  [ -x .venv/bin/python ] || { log "缺少 .venv（运行 install_env.sh）"; exit 1; }

  if ! .venv/bin/python -c "import cn_clip" >/dev/null 2>&1; then
    log "!! .venv 缺少 cn_clip —— 查询侧 chinese-clip 不可用，视觉检索会静默 no_candidates"
    log "   修复： .venv/bin/python -m pip install 'cn-clip==1.6.0'（或重跑 install_env.sh）"
    exit 1
  fi
  CN_CLIP_CKPT="${CHINESE_CLIP_CHECKPOINT:-$HOME/.cache/clip/clip_cn_vit-l-14.pt}"
  if [ ! -f "$CN_CLIP_CKPT" ]; then
    log "!! 缺少 chinese-clip 权重：$CN_CLIP_CKPT（视觉检索会静默失效；下载方式见 install_env.sh 输出）"
    exit 1
  fi
  log "IMAGE embedder 就绪：chinese-clip（$CN_CLIP_CKPT）"

  if [ -f scripts/maintenance/ensure_visual_vectors.py ]; then
    ENSURE_SCOPES="${SENTRIX_ENSURE_SCOPES-}"
    if [ -n "$ENSURE_SCOPES" ]; then
      # 指定 scope（逗号分隔）：只对缺 chinese-clip 向量的重嵌，已补齐的零成本跳过。
      log "补齐视觉向量（chinese-clip）：scope=$ENSURE_SCOPES"
      .venv/bin/python scripts/maintenance/ensure_visual_vectors.py --apply --scope "$ENSURE_SCOPES" \
        || log "WARN: 视觉向量补齐失败（非致命；对应 scope 的图片语义召回可能退化）"
    else
      # 未指定 scope 时只做只读检测并告警：全量重嵌在 CPU 上可能数千张/数小时，
      # 不能堵在启动路径上。新库无 scope，此步秒过。
      log "检测视觉向量一致性（chinese-clip，只读）…"
      _ensure_out=$(.venv/bin/python scripts/maintenance/ensure_visual_vectors.py --db data/sentrix.db 2>/dev/null || true)
      _missing=$(printf '%s' "$_ensure_out" | sed -n 's/.*"missing_scope_count": *\([0-9][0-9]*\).*/\1/p' | head -1)
      if [ -n "$_missing" ] && [ "$_missing" != "0" ]; then
        log "WARN: $_missing 个 scope 缺 chinese-clip 视觉向量 —— 这些 scope 的图片语义召回会静默为空"
        log "      补齐： .venv/bin/python scripts/maintenance/ensure_visual_vectors.py --apply --scope <scope_id,...>"
        log "      或启动前设 SENTRIX_ENSURE_SCOPES=<scope_id,...> 自动补齐"
      else
        log "视觉向量一致性 OK（无缺失 scope）"
      fi
    fi
  fi
fi

# ---------- 2) Sentrix API @8091（复用生产启动脚本，不另抄 env） ----------
if ! up $SENTRIX_PORT; then
  RUNTIME_START="scripts/runtime/start_sentrix_api_8091.sh"
  if [ -f "$RUNTIME_START" ]; then
    log "启动 Sentrix API @$SENTRIX_PORT（复用 $RUNTIME_START，含生产完整 env）"
    mkdir -p logs
    setsid nohup bash "$RUNTIME_START" >> logs/sentrix-api-8091.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do up $SENTRIX_PORT && break; sleep 2; done
  else
    log "!! 未找到 $RUNTIME_START —— 拒绝用简化 env 启动 8091"
    log "   该脚本含生产完整 env（qdrant 向量后端 / Face provider / worker 配额 / AdaFace 等）。"
    log "   手写第二份 env 极易漏项，导致向量层静默降级为 SQLite 全表扫（检索"没报错但明显变差"）。"
    exit 1
  fi
fi

# ---------- 3) PhotoBench orchestrator @8771 ----------
if ! up $ORCH_PORT; then
  log "启动 orchestrator @$ORCH_PORT"
  setsid nohup python3 services/photobench/backend/benchmark_orchestrator.py \
      --host 0.0.0.0 --port $ORCH_PORT >/tmp/orch8771.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do up $ORCH_PORT && break; sleep 2; done
fi

echo
# ---------- 4) 就绪校验：health + 向量后端 + 检索探测 ----------
if curl -s -m 5 "http://127.0.0.1:$SENTRIX_PORT/api/health" >/dev/null 2>&1; then
  log "8091 health 就绪"
  _vi=$(curl -s -m 5 "http://127.0.0.1:$SENTRIX_PORT/api/health" 2>/dev/null \
        | python3 -c "import sys,json;m=(json.load(sys.stdin).get('memory') or {}).get('vectorIndex') or {};print('%s|%s|%s' % (m.get('backend'),m.get('qdrant_enabled'),m.get('qdrant_available')))" 2>/dev/null || echo "?|?|?")
  log "向量后端：$_vi（期望 qdrant|True|True；出现 sqlite 说明启动 env 不完整）"
  if [ -f scripts/probe_sentrix_retrieval.py ]; then
    .venv/bin/python scripts/probe_sentrix_retrieval.py --host "127.0.0.1:$SENTRIX_PORT" \
      && log "检索层探测通过" \
      || log "WARN: 检索探测失败——查 /api/health 的 memory.vectorIndex（常见原因：Qdrant 锁被占）"
  fi
else
  log "WARN: 8091 health 未就绪，查 logs/sentrix-api-8091.log"
fi

log "完成：8100 vLLM(12B) / $TEXT_EMBED_PORT 文本嵌入(bge) / $SENTRIX_PORT Sentrix API / $ORCH_PORT orchestrator"
ss -ltnp 2>/dev/null | grep -E ":8100 |:$TEXT_EMBED_PORT |:$SENTRIX_PORT |:$ORCH_PORT " || true
