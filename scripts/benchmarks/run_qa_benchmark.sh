#!/usr/bin/env bash
# 一键启动 Sentrix QA 自动测评（对齐 4174 真实生产链路）
# 用法: ./run_qa_benchmark.sh [--limit N] [--base URL] [--scope album3] [--no-judge]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/scripts/benchmarks/run_qa_benchmark.py"
QA="${SENTRIX_QA_FILE:-/Users/rm001/Downloads/album3/qa/full-album3.jsonl}"
BASE="${SENTRIX_QA_BASE:-http://192.168.0.153:4174}"
SCOPE="${SENTRIX_QA_SCOPE:-album3}"
OUT="${SENTRIX_QA_OUT:-$HOME/Downloads/sentrix_qa_report}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) EXTRA+=(--limit "$2"); shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --no-judge) EXTRA+=(--no-judge); shift ;;
    --concurrency) EXTRA+=(--concurrency "$2"); shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

echo "▶ Sentrix QA 一键测评"
echo "  QA   : $QA"
echo "  4174 : $BASE (scope=$SCOPE)"
echo "  输出 : $OUT"
echo

python3 "$PY" --qa "$QA" --base "$BASE" --scope "$SCOPE" --out "$OUT" "${EXTRA[@]}"

HTML="$OUT/qa_report.html"
if [[ -f "$HTML" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then open "$HTML"; else xdg-open "$HTML" >/dev/null 2>&1 || true; fi
  echo
  echo "已打开报告: $HTML"
fi
