#!/usr/bin/env bash
# ============================================================
# Sentrix 在 macOS (Apple Silicon) 上的一键启动
#
# 与 Linux 版 scripts/deploy/start_all.sh 的差异，以及为什么：
#
#   1. 推理后端 8100：Linux 走 vLLM（需要 CUDA，Mac 上没有）。这里改用
#      llama.cpp 的 llama-server —— 它提供 OpenAI 兼容的 /v1/chat/completions，
#      正好满足 SENTRIX_LLM_BACKEND=vllm 的协议要求（46/118 也是这么走的）。
#      选 qwen35-q4km + mmproj，与 46 上验证过的那套权重一致。
#
#   2. 8091 用 scripts/runtime/start_sentrix_api.sh（通用版），不是 _8091.sh。
#      后者是 153 专属：里面写死了 CUDA_VISIBLE_DEVICES=0、vllm manager 8500、
#      12B profile。通用版读 .env + platform-profiles.sh，正是为「一份代码多机跑」
#      设计的。注意 start_all.sh 反复警告的「不要手写第二份 env」依然成立 ——
#      本脚本同样不重写 env，只是换了一个生产启动脚本。
#
#   3. setsid / ss / readlink -f 都是 Linux-only：
#        setsid   → macOS 没有，用 nohup + & 即可（本来就在脚本里后台化）
#        ss       → 用 lsof -tiTCP:PORT -sTCP:LISTEN
#        readlink → 用 cd/dirname/pwd
#
# 用法： bash scripts/deploy/start_all_mac.sh
# 可覆盖：SENTRIX_HOME / SENTRIX_PORT / ORCH_PORT / WEB_PORT / TEXT_EMBED_PORT / VLM_PORT
# ============================================================
set -uo pipefail

SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 系统 python3 是 3.9.6（CommandLineTools），编排器要 3.10+ 的语法就得用 brew 的 3.12。
# 注意 /opt/homebrew/bin/python3 这个链接**不存在**，brew 只给了 python3.12。
PY312="$(command -v python3.12 || echo /opt/homebrew/bin/python3.12)"
SENTRIX_PORT="${SENTRIX_PORT:-8091}"
ORCH_PORT="${ORCH_PORT:-8771}"
WEB_PORT="${WEB_PORT:-4174}"
TEXT_EMBED_PORT="${TEXT_EMBED_PORT:-8101}"
VLM_PORT="${VLM_PORT:-8100}"

cd "$SENTRIX_HOME" || { echo "no SENTRIX_HOME=$SENTRIX_HOME"; exit 1; }
# llama-server 与 macmon 都由 Homebrew 安装。orchestrator 用 macmon pipe 读温度和功耗。
export PATH="/opt/homebrew/bin:${PATH}"
# 自检：路径解析错一级会 cd 到仓库父目录，后续相对路径全部失效。
[ -f backend/app.py ] || { echo "SENTRIX_HOME 不正确（找不到 backend/app.py）: $SENTRIX_HOME"; exit 1; }

log() { echo "[start_mac $(date +%H:%M:%S)] $*"; }

# macOS 没有 ss；用 lsof 探监听口。
up() { lsof -tiTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

mkdir -p logs data/media data/ann

# ---------- 0) VLM @8100（llama.cpp + Metal）----------
if ! up "$VLM_PORT"; then
  LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || echo /opt/homebrew/bin/llama-server)}"
  if [ ! -x "$LLAMA_SERVER" ]; then
    log "!! 找不到 llama-server（brew install llama.cpp）。8100 起不来，Agent 问答会失败。"
  else
    log "启动 VLM @${VLM_PORT}（llama.cpp/Metal，qwen35-q4km + mmproj）"
    # 参数以 **46 上跑 qwen35-q4km 那次 487qa** 的实际启动命令为准：
    #   --ctx-size 24576 --parallel 4 → 每槽 6144
    #     不要抄 153 历史里 gemma4 的 16384（每槽 4096）——那会让 agent 的提示词
    #     被截断在 4096：2026-09-22 实测 Mac 用 16384 时 llm_prompt_tokens_max=4086
    #     正好顶到每槽上限，而 46 同题同模型能到 5899。截断导致 agent 丢证据、
    #     多绕弯：prompt 总量 +23%（5.24M vs 4.28M），QA 耗时 +41%（22203s vs
    #     15716s），准确率还更低（0.698 vs 0.781）。注意单请求生成速度其实
    #     Mac 7.7 t/s > 46 7.3 t/s，慢的是任务结构不是硬件。
    #   --reasoning off                 → **必须**。qwen3.5 是推理模型，不关思维链时
    #                                     输出全进 reasoning_content，content 为空，
    #                                     agent 拿到空回复（实测 60 token 全被吃掉）。
    #   --metrics                       → PhotoBench 遥测读 /metrics，缺了资源曲线是空的
    #   --cache-ram 512 --ctx-checkpoints 1
    #       **必须显式限制**。默认 --cache-ram 是 8192（8GB）—— 在 16GB 统一内存的机器上
    #       这是致命的：建相册阶段缓存只长到 9K token 看不出问题，但 QA 阶段 agent
    #       反复发带长工具定义的不同提示词，会把缓存喂到上限。2026-09-22 实测：
    #       llama-server 吃到 11GB，整机 swap 11.6GB、换页流量 ~4GB/分钟，
    #       生成速度从 20 t/s 崩到 4.3 t/s（QA 速率掉一半）。
    #       收紧后 llama-server 11GB → 2GB，swap 11.6GB → 3.7GB，空闲 15% → 46%。
    nohup "$LLAMA_SERVER" \
      -m "$SENTRIX_HOME/models/llama/qwen35-q4km.gguf" \
      --mmproj "$SENTRIX_HOME/models/llama/qwen35-mmproj-f16.gguf" \
      --alias qwen35-q4km \
      --host 0.0.0.0 --port "$VLM_PORT" \
      --n-gpu-layers 999 \
      --ctx-size 24576 --parallel 4 \
      --metrics --reasoning off \
      --cache-ram 512 --ctx-checkpoints 1 \
      --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      > logs/vlm-$VLM_PORT.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do up "$VLM_PORT" && break; sleep 2; done
    up "$VLM_PORT" && log "VLM 就绪" || log "WARN: VLM 未在 120s 内就绪，查 logs/vlm-$VLM_PORT.log"
  fi
fi

# ---------- 1) 文本嵌入 sidecar @8101（bge-m3）----------
if ! up "$TEXT_EMBED_PORT"; then
  if [ ! -x .venv-text/bin/python ]; then
    log "缺少 .venv-text —— 先跑 scripts/deploy/install_env.sh 或手动建"
  else
    log "启动文本嵌入 sidecar（bge @${TEXT_EMBED_PORT}）"
    # HF_HUB_OFFLINE=1 必带：权重已预置，联网重试会让 sidecar 一直起不来。
    nohup env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
        SENTRIX_TEXT_EMBEDDER_DEVICE=cpu \
        .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py \
        --host 127.0.0.1 --port "$TEXT_EMBED_PORT" \
        > logs/text-embed.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do up "$TEXT_EMBED_PORT" && break; sleep 2; done
    up "$TEXT_EMBED_PORT" && log "bge sidecar 就绪" || log "WARN: 8101 未就绪，查 logs/text-embed.log"
  fi
fi

# ---------- 2) 视觉向量一致性（必须在 8091 之前：之后 Qdrant 目录锁被占）----------
if [ -x .venv/bin/python ]; then
  log "检测视觉向量一致性（chinese-clip，只读）…"
  .venv/bin/python scripts/maintenance/ensure_visual_vectors.py --db data/sentrix.db 2>/dev/null \
    | sed -n 's/.*"missing_scope_count": *\([0-9]*\).*/[start_mac] 缺失 scope 数: \1/p' || true
fi

# ---------- 3) Sentrix API @8091 ----------
if ! up "$SENTRIX_PORT"; then
  log "启动 Sentrix API @${SENTRIX_PORT}（scripts/runtime/start_sentrix_api.sh，读 .env）"
  nohup bash scripts/runtime/start_sentrix_api.sh \
      > logs/sentrix-api-$SENTRIX_PORT.log 2>&1 < /dev/null &
  for _ in $(seq 1 90); do up "$SENTRIX_PORT" && break; sleep 2; done
  up "$SENTRIX_PORT" && log "API 就绪" || log "WARN: 8091 未就绪，查 logs/sentrix-api-$SENTRIX_PORT.log"
fi

# ---------- 4) PhotoBench orchestrator @8771 ----------
if ! up "$ORCH_PORT"; then
  log "启动 orchestrator @$ORCH_PORT"
  # 必须用项目 .venv：orchestrator 会 import backend.runtime_providers，而它 import httpx。
  # 153 上 `python3` 恰好是 anaconda 的 3.10（自带 httpx），所以那边裸 python3 能跑；
  # 这里裸 python3.12 没有 httpx，会 ModuleNotFoundError。用 .venv 才是与生产等价的环境。
  #
  # PHOTOBENCH_QA_CONCURRENCY 必须显式给：默认值取服务模型快照里的 max_num_seqs，
  # 而本机没有 vllm manager（拿不到快照）→ 回落到 1 → 487 题串行，要跑半天。
  # 这里对齐 llama-server 的 --parallel 4。
  # judge 走云端 API，不占本地推理槽，按代码上限给 8。
  nohup env PHOTOBENCH_QA_CONCURRENCY="${PHOTOBENCH_QA_CONCURRENCY:-4}" \
      PHOTOBENCH_JUDGE_CONCURRENCY="${PHOTOBENCH_JUDGE_CONCURRENCY:-8}" \
      .venv/bin/python services/photobench/backend/benchmark_orchestrator.py \
      --host 0.0.0.0 --port "$ORCH_PORT" \
      > logs/orch-$ORCH_PORT.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do up "$ORCH_PORT" && break; sleep 2; done
  up "$ORCH_PORT" && log "orchestrator 就绪" || log "WARN: 8771 未就绪，查 logs/orch-$ORCH_PORT.log"
fi

# ---------- 5) Web @4174 ----------
if ! up "$WEB_PORT"; then
  log "启动 Web @$WEB_PORT"
  PORT="$WEB_PORT" SENTRIX_BACKEND_URL="http://127.0.0.1:$SENTRIX_PORT" \
    nohup node server.js > logs/web-$WEB_PORT.log 2>&1 < /dev/null &
  for _ in $(seq 1 20); do up "$WEB_PORT" && break; sleep 1; done
  up "$WEB_PORT" && log "Web 就绪" || log "WARN: $WEB_PORT 未就绪，查 logs/web-$WEB_PORT.log"
fi

# ---------- 6) 就绪校验 ----------
echo
log "===== 端口状态 ====="
for p in "$VLM_PORT:VLM" "$TEXT_EMBED_PORT:bge" "$SENTRIX_PORT:API" "$ORCH_PORT:orchestrator" "$WEB_PORT:Web"; do
  port="${p%%:*}"; name="${p##*:}"
  if up "$port"; then echo "  ✅ $name  :$port"; else echo "  ❌ $name  :$port"; fi
done

echo
if curl -s -m 5 "http://127.0.0.1:$SENTRIX_PORT/api/health" >/dev/null 2>&1; then
  _vi=$(curl -s -m 5 "http://127.0.0.1:$SENTRIX_PORT/api/health" 2>/dev/null \
        | "$PY312" -c "import sys,json;m=(json.load(sys.stdin).get('memory') or {}).get('vectorIndex') or {};print('%s|%s|%s' % (m.get('backend'),m.get('qdrant_enabled'),m.get('qdrant_available')))" 2>/dev/null || echo "?|?|?")
  log "向量后端：${_vi}（期望 qdrant|True|True；出现 sqlite 说明启动 env 不完整）"
fi
if curl -s -m 5 "http://127.0.0.1:$VLM_PORT/v1/models" >/dev/null 2>&1; then
  log "VLM /v1/models: $(curl -s -m 5 http://127.0.0.1:$VLM_PORT/v1/models | head -c 200)"
fi

log "完成"
