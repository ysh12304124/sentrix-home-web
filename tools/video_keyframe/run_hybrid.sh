#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$PACKAGE_DIR/.venv/bin/python}"
MODEL_DIR="$PACKAGE_DIR/models/keyframe"

if [[ "$PYTHON" != */* ]]; then
  PYTHON="$(command -v "$PYTHON" || true)"
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: environment not found: $PYTHON" >&2
  echo "Run: bash $PACKAGE_DIR/install.sh" >&2
  exit 2
fi

exec "$PYTHON" "$PACKAGE_DIR/katna/run_yolo_prefilter_event_webp.py" \
  --katna-root "$PACKAGE_DIR/katna" \
  --pipeline-root "$PACKAGE_DIR" \
  --yolo-model "$MODEL_DIR/yolo11n.pt" \
  --pose-model "$MODEL_DIR/yolo11n-pose.pt" \
  "$@"
