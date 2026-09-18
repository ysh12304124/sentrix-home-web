#!/usr/bin/env bash
# SHW后端启动脚本(持久化)
# 006.0.2固化于2026-08-24,替代/tmp/be_start.sh
# 用法: bash start_shw_backend.sh
set -euo pipefail

cd /home/sscy/lingbot-map/004SHW

# 加载仓库.env基础配置
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# 加载固化覆盖(ollama本地模型)
ENV_OVERRIDE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/shw_backend_env.sh"
if [[ -f "$ENV_OVERRIDE" ]]; then
  set -a
  source "$ENV_OVERRIDE"
  set +a
fi

# 固化端口(防止被仓库.env里的12001覆盖)
export SENTRIX_API_PORT=9598

# ollama预热(防冷加载崩):确认服务存活+模型常驻
echo -n "Checking ollama..."
for i in $(seq 1 20); do
  if curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo " alive"
    echo -n "Warming ${OLLAMA_MODEL:-gemma4:e2b} (keep_alive=-1)..."
    curl -s -m 120 http://127.0.0.1:11434/api/generate -d "{\"model\":\"${OLLAMA_MODEL:-gemma4:e2b}\",\"prompt\":\"warm\",\"keep_alive\":-1}" >/dev/null 2>&1
    sleep 2
    if ollama ps 2>/dev/null | grep -q "Forever"; then
      echo " ready (Forever)"
    else
      echo " loaded (check 'ollama ps' to verify)"
    fi
    break
  fi
  if [ $i -eq 20 ]; then echo " UNREACHABLE, starting backend anyway"; fi
  sleep 1
done

# 目录准备
export SENTRIX_DATA_DIR="${SENTRIX_DATA_DIR:-$PWD/data}"
export SENTRIX_DB_PATH="${SENTRIX_DB_PATH:-$SENTRIX_DATA_DIR/sentrix.db}"
export SENTRIX_ANN_DIR="${SENTRIX_ANN_DIR:-$SENTRIX_DATA_DIR/ann}"
mkdir -p "$SENTRIX_DATA_DIR/media" "$SENTRIX_ANN_DIR" 2>/dev/null || true

echo "Starting SHW backend on :${SENTRIX_API_PORT:-9598}"
echo "LLM: $SENTRIX_LLM_BACKEND / $OLLAMA_MODEL"
exec /home/sscy/conda_envs/lingbot-map/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port "${SENTRIX_API_PORT:-9598}"
