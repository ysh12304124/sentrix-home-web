#!/bin/bash
# 153 本机版：8091 Sentrix API 安全重启 SOP（2026-08-22 检索降级事故后固化）。
# 本机直接运行（无 ssh 层），由 benchmark 侧同名脚本改编部署。
#
# 背景：03:25 的一次重启只按端口杀了监听进程，进程树中持 qdrant 目录锁的
# 进程存活，新 8091 的 QdrantClient 拿不到锁后向量层静默降级（现已改为
# health 暴露 + ERROR 日志），检索全部退化为 SQLite 全表扫（恒定 20-25s、
# 无报错），数日后才定位。本脚本按正确姿势重启：杀干净进程树 → 复核全灭
# → 正式启动脚本拉起 → health + 探测。
#
# 已知坑（均实际踩过）：
#   1. pkill/pgrep -f 的模式若与自身命令行相同会自杀（ssh 场景断连）。
#      本脚本用 [b]ackend 字符类规避自匹配。
#   2. lsof -ti :8091 只杀监听者，可能留下持锁的子进程——禁止用于重启。
#   3. 153 上 9598/11001/11011/8099 等其他 backend.app 实例分属不同项目
#      与库，必须按完整命令行精确匹配，不能宽匹配 "backend.app"。
#
# 用法：bash scripts/restart_sentrix_8091.sh [--force]
#   --force  跳过活跃 run 检查（仍会提示当前 run 状态）

set -euo pipefail

PROJECT_DIR="/home/asus/Github/Sentrix-Home-Web"
START_SCRIPT="scripts/runtime/start_sentrix_api_8091.sh"
LOG_FILE="logs/sentrix-api-8091.log"
PROBE_SCRIPT="scripts/probe_sentrix_retrieval.py"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

echo "== 步骤 1/5: 检查活跃 run =="
ACTIVE=$(curl -s -m 5 http://127.0.0.1:8771/api/runs | python3 -c 'import json,sys; runs=json.load(sys.stdin); items=runs if isinstance(runs,list) else runs.get("runs",[]); print(len([r for r in items if r.get("status") in ("running","pending","queued")]))' 2>/dev/null || echo -1)
if [[ "$ACTIVE" != "0" && "$FORCE" != "1" ]]; then
  echo "!! 8771 有 $ACTIVE 个活跃 run，拒绝重启（确认无评测在跑后用 --force）"
  exit 2
fi
[[ "$FORCE" == "1" ]] && echo "(--force：跳过活跃 run 拦截，检测值=$ACTIVE)"

echo "== 步骤 2/5: 停止旧 8091（按完整命令行精确匹配，不误伤其他端口实例）=="
pkill -f 'uvicorn [b]ackend.app:app --host 0.0.0.0 --port 8091' || true
sleep 3

echo "== 步骤 3/5: 复核旧进程全灭 =="
REMAIN=$(pgrep -af 'uvicorn [b]ackend.app:app --host 0.0.0.0 --port 8091' || true)
if [[ -n "$REMAIN" ]]; then
  echo "!! 残留进程，强杀:"
  echo "$REMAIN"
  pkill -9 -f 'uvicorn [b]ackend.app:app --host 0.0.0.0 --port 8091' || true
  sleep 1
  pgrep -af 'uvicorn [b]ackend.app:app --host 0.0.0.0 --port 8091' && { echo "!! 强杀后仍有残留，人工排查 qdrant 锁持有者"; exit 9; } || true
  echo "已强杀"
else
  echo "旧进程已全灭（无残留）"
fi
OTHERS=$(pgrep -af 'uvicorn [b]ackend.app' | grep -v -- '--port 8091' | wc -l)
echo "其他 backend.app 实例保留: $OTHERS 个（8099/9598/11001/11011 等，分属不同项目）"

echo "== 步骤 3.5/5: 确保常用测试集视觉向量一致（chinese_clip 补齐 + 同步 Qdrant）=="
# 停机窗口 qdrant 锁已释放，是唯一安全的补齐时机。只保证最近最常用的测试集
# scope 按当前生产 embedder（chinese-clip）编码；其余历史一次性评测 scope 不补
# （避免每次重启全量重嵌 3057 张）。缺向量时 visual_ann 会静默 no_candidates。
ACTIVE_SCOPES="album_ca0cc0ddda3a,album_cba01be9502b,album3"
if "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/maintenance/ensure_visual_vectors.py" --apply --scope "$ACTIVE_SCOPES"; then
  echo "视觉向量一致性检查通过（常用测试集无缺失或已补齐）"
else
  echo "!! 视觉向量补齐失败（非致命，但常用测试集检索可能退化）——查 chinese_clip 模型/磁盘；health.memory.vectorIndex 会暴露状态"
fi

echo "== 步骤 4/5: 启动脚本拉起 =="
cd "$PROJECT_DIR"
nohup bash "$START_SCRIPT" >> "$LOG_FILE" 2>&1 < /dev/null &
disown
sleep 2

echo "== 步骤 5/5: 等待 health 就绪 =="
READY=0
for i in $(seq 1 24); do
  if curl -s -m 3 http://127.0.0.1:8091/api/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
if [[ "$READY" != "1" ]]; then
  echo "!! 8091 启动后 2 分钟内 health 未就绪，请查 $PROJECT_DIR/$LOG_FILE"
  exit 3
fi
echo "8091 已就绪（第 $i 次探测通过）"

echo "== 追加: Level 1 检索健康探测 =="
if [[ -f "$PROBE_SCRIPT" ]]; then
  python3 "$PROBE_SCRIPT" --host 127.0.0.1:8091 \
    && echo "重启完成，检索层健康" \
    || { echo "!! 重启成功但检索探测失败——查 health.memory.vectorIndex，常见原因: qdrant 锁被占"; exit 4; }
else
  echo "(探测脚本 $PROBE_SCRIPT 不存在，跳过；可从 benchmark 项目 scripts/ 同步)"
fi
