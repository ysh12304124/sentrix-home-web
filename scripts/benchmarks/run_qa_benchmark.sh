#!/usr/bin/env bash
# 一键启动 Sentrix QA 自动测评（对齐 4174 真实生产链路）
# 结果归档到 <out>/runs/<ts>_<tag>/ 并上传 QA Dashboard（4174/qa）
# 用法: ./run_qa_benchmark.sh [--tag phasee] [--limit N] [--base URL] [--scope album3] [--no-judge]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/scripts/benchmarks/run_qa_benchmark.py"
QA="${SENTRIX_QA_FILE:-/Users/rm001/Downloads/album3/qa/full-album3.jsonl}"
BASE="${SENTRIX_QA_BASE:-http://192.168.0.153:4174}"
SCOPE="${SENTRIX_QA_SCOPE:-album3}"
OUT="${SENTRIX_QA_OUT:-$HOME/Downloads/sentrix_qa_report}"
TAG="${SENTRIX_QA_TAG:-qa}"
NOTE=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) EXTRA+=(--limit "$2"); shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --note) NOTE="$2"; shift 2 ;;
    --no-judge) EXTRA+=(--no-judge); shift ;;
    --concurrency) EXTRA+=(--concurrency "$2"); shift 2 ;;
    --no-upload) EXTRA+=(--no-upload); shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

echo "▶ Sentrix QA 一键测评"
echo "  QA   : $QA"
echo "  4174 : $BASE (scope=$SCOPE)"
echo "  输出 : $OUT  标签: $TAG"
echo

python3 "$PY" --qa "$QA" --base "$BASE" --scope "$SCOPE" --out "$OUT" \
  --tag "$TAG" --note "$NOTE" ${EXTRA[@]+"${EXTRA[@]}"}
