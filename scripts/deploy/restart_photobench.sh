#!/usr/bin/env bash
# ============================================================
# PhotoBench orchestrator (8771) 重启
#
# 为什么单独一个脚本——两个实测踩过的坑：
#
# 1) 就绪要按进程判断，不能按端口。8771 启动时会扫 results/ 下 200 多个 run，
#    实测约 55 秒才绑定端口。这期间 `ss | grep 8771` 和 curl 都会失败，很容易
#    误判成"没起来"然后反复重启，反而把端口拖进 TIME_WAIT。
#
# 2) 必须等旧进程真正释放端口再起新的。旧进程收到 SIGTERM 后要几秒才关监听，
#    不等就起新进程会 "Address already in use" 直接退出——而此时旧进程也没了，
#    服务彻底空窗。所以这里显式轮询到端口空闲为止。
#
# 解释器默认沿用**当前正在跑的那个进程**，而不是写死一个路径：换解释器可能
# 改变依赖解析（8771 需要 httpx），重启不该顺带改运行时。
#
# 用法： bash scripts/deploy/restart_photobench.sh [--force]
#   --force  有进行中的 run 时也强制重启（默认拒绝）
# 可覆盖：SENTRIX_HOME / ORCH_PORT / ORCH_PYTHON / ORCH_LOG / ORCH_PID_FILE
# ============================================================
set -uo pipefail

SENTRIX_HOME="${SENTRIX_HOME:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
ORCH_PORT="${ORCH_PORT:-8771}"
ORCH_LOG="${ORCH_LOG:-/tmp/photobench-${ORCH_PORT}.log}"
ORCH_PID_FILE="${ORCH_PID_FILE:-/tmp/photobench-${ORCH_PORT}.pid}"
START_TIMEOUT="${START_TIMEOUT:-240}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

log() { echo "[restart_photobench $(date +%H:%M:%S)] $*"; }
die() { echo "[restart_photobench] 错误：$*" >&2; exit 1; }

cd "$SENTRIX_HOME" || die "no SENTRIX_HOME=$SENTRIX_HOME"
[ -f services/photobench/backend/benchmark_orchestrator.py ] \
  || die "SENTRIX_HOME 不正确（找不到 orchestrator）: $SENTRIX_HOME"

# 按 /proc/*/cmdline 精确匹配，避免 pgrep -f 把本脚本自己算进去
# （脚本正文里就有 benchmark_orchestrator.py 这个字符串）。
orchestrator_pids() {
  local pid exe arg
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ -r "/proc/$pid/cmdline" ] || continue
    exe=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1)
    case "$exe" in *python*) ;; *) continue ;; esac
    while IFS= read -r arg; do
      case "$arg" in *benchmark_orchestrator.py) echo "$pid"; break ;; esac
    done < <(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null)
  done
}

listening_pid() {
  ss -ltnp 2>/dev/null | grep -F ":$ORCH_PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

port_free() { [ -z "$(ss -ltn 2>/dev/null | grep -F ":$ORCH_PORT ")" ]; }

# ---------- 1) 有进行中的 run 就拒绝（除非 --force） ----------
RUNNING_PIDS=$(orchestrator_pids | tr '\n' ' ')
if [ -n "${RUNNING_PIDS// /}" ]; then
  log "当前 orchestrator PID: $RUNNING_PIDS"
  active=$(curl -fsS -m 5 "http://127.0.0.1:$ORCH_PORT/api/runs?page=1&page_size=1" 2>/dev/null \
    | grep -o '"active_count":[0-9]*' | head -1 | cut -d: -f2)
  active="${active:-unknown}"
  log "进行中的 run 数：$active"
  if [ "$active" != "0" ] && [ "$FORCE" != "1" ]; then
    die "有进行中的 run，拒绝重启（确认要重启请加 --force）"
  fi
else
  log "未发现运行中的 orchestrator"
fi

# ---------- 2) 记下当前解释器，重启沿用它 ----------
ORCH_PYTHON="${ORCH_PYTHON:-}"
if [ -z "$ORCH_PYTHON" ]; then
  first_pid=$(orchestrator_pids | head -1)
  if [ -n "$first_pid" ]; then
    ORCH_PYTHON=$(tr '\0' '\n' < "/proc/$first_pid/cmdline" 2>/dev/null | head -1)
    log "沿用运行中的解释器：$ORCH_PYTHON"
  else
    ORCH_PYTHON="$SENTRIX_HOME/.venv/bin/python"
    log "服务未运行，使用默认解释器：$ORCH_PYTHON"
  fi
fi
[ -x "$ORCH_PYTHON" ] || die "解释器不可执行：$ORCH_PYTHON"

# ---------- 3) 停旧进程 ----------
for pid in $(orchestrator_pids); do
  log "SIGTERM $pid"
  kill "$pid" 2>/dev/null
done
for _ in $(seq 1 40); do
  [ -z "$(orchestrator_pids)" ] && break
  sleep 0.5
done
for pid in $(orchestrator_pids); do
  log "进程未退出，SIGKILL $pid"
  kill -9 "$pid" 2>/dev/null
done

# 等端口真正空闲——不等就起新进程会 bind 失败，且旧进程已死，服务空窗
for _ in $(seq 1 40); do port_free && break; sleep 0.5; done
port_free || die "$ORCH_PORT 端口未释放，中止（避免新进程 bind 失败）"
log "端口 $ORCH_PORT 已释放"

# ---------- 4) 启动 ----------
log "启动：$ORCH_PYTHON backend/benchmark_orchestrator.py --host 0.0.0.0 --port $ORCH_PORT"
( cd "$SENTRIX_HOME/services/photobench" && \
  setsid nohup "$ORCH_PYTHON" backend/benchmark_orchestrator.py \
    --host 0.0.0.0 --port "$ORCH_PORT" >> "$ORCH_LOG" 2>&1 < /dev/null & )
new_pid=""
for _ in $(seq 1 20); do
  new_pid=$(orchestrator_pids | head -1)
  [ -n "$new_pid" ] && break
  sleep 0.5
done
[ -n "$new_pid" ] || die "新进程没起来，看 $ORCH_LOG"
echo "$new_pid" > "$ORCH_PID_FILE"
log "新 PID $new_pid（已写入 $ORCH_PID_FILE）"

# ---------- 5) 按进程/接口就绪，而不是按端口 ----------
ready=0
for i in $(seq 1 "$START_TIMEOUT"); do
  if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "--- $ORCH_LOG 末尾 ---"; tail -20 "$ORCH_LOG"
    die "新进程已退出"
  fi
  if curl -fsS -m 2 "http://127.0.0.1:$ORCH_PORT/api/config" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" = "1" ] || { echo "--- $ORCH_LOG 末尾 ---"; tail -20 "$ORCH_LOG"; die "启动超时（${START_TIMEOUT}s）"; }

# ---------- 6) 确认监听的就是新进程 ----------
bound=$(listening_pid)
log "就绪（约 ${i}s）· 监听 PID ${bound:-未知} · 新 PID $new_pid"
[ "$bound" = "$new_pid" ] || log "警告：监听 PID 与新进程不一致，请人工确认"
log "完成：http://0.0.0.0:$ORCH_PORT/"
