#!/usr/bin/env bash
# Layered backup for the Sentrix production data (Phase R8-8 deployment safety).
#
# - source : git bundle of the psh branch
# - database : SQLite backup API (never `cp` of a live WAL-mode database)
# - media / ANN / weights : direct copy (these are not being written mid-copy
#   the way the SQLite WAL is; still listed for the manifest)
#
# Usage: bash scripts/maintenance/backup_sentrix.sh [dest_dir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="${1:-${SENTRIX_BACKUP_DIR:-/home/asus/sentrix-backups}/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$dest/db" "$dest/media" "$dest/ann" "$dest/weights" "$dest/source"
echo "backup -> $dest"

python_bin="${SENTRIX_PYTHON:-$root/.venv/bin/python}"
db_path="${SENTRIX_DB_PATH:-$root/data/sentrix.db}"

# 1) Source via git bundle (authoritative history, not a working-tree copy).
(cd "$root" && git bundle create "$dest/source/psh.bundle" --all 2>/dev/null || true)

# 2) Database via SQLite online backup API — safe against a live WAL writer.
"$python_bin" - "$db_path" "$dest/db/sentrix.db" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
src_conn = sqlite3.connect(src)
dst_conn = sqlite3.connect(dst)
src_conn.backup(dst_conn)
dst_conn.close()
src_conn.close()
print("db backed up via SQLite backup API")
PY

# 3) Media, ANN, model weights, logs.
cp -r "$root/data/household-benchmark-source" "$dest/media/" 2>/dev/null || true
cp -r "$root/data/media" "$dest/media/uploads" 2>/dev/null || true
cp -r "$root/data/ann" "$dest/ann/" 2>/dev/null || true
cp -r "$root/data/models" "$dest/weights/" 2>/dev/null || true

# 4) Manifest + checksums.
{
  echo "generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "db: $(sha256sum "$dest/db/sentrix.db" 2>/dev/null | awk '{print $1}')"
  find "$dest" -type f -exec sha256sum {} \; | sort
} > "$dest/MANIFEST.sha256"

echo "done. manifest: $dest/MANIFEST.sha256"
du -sh "$dest"
