#!/usr/bin/env bash
# Clear imported album data and derived intermediates for this clone.
#
# Removes:
#   - SQLite database (+ WAL/SHM)
#   - ANN indexes, HEIC previews, media copies, local DB backups
#
# Keeps by default:
#   - data/album   (source photos)
#   - data/models  (model weights / checkpoints)
#
# Stops this clone's Web/API ports before clearing (required: a live API can
# keep serving from a deleted SQLite inode). Does not touch other ports
# (e.g. production 8091).
#
# Usage:
#   ./clear.sh              # interactive confirm
#   ./clear.sh -y           # no prompt
#   ./clear.sh --include-album   # also delete source photos under data/album
#   ./clear.sh --keep-running    # do not stop Web/API (not recommended)
#
# Requires bash (re-execs if started via `sh ./clear.sh`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a
fi

data_dir="${SENTRIX_DATA_DIR:-$root/data}"
db_path="${SENTRIX_DB_PATH:-$data_dir/sentrix.db}"
ann_dir="${SENTRIX_ANN_DIR:-$data_dir/ann}"
web_port="${PORT:-11000}"
api_port="${SENTRIX_API_PORT:-11001}"
yes=0
include_album=0
keep_running=0

usage() {
  cat <<'EOF'
Usage: ./clear.sh [-y|--yes] [--include-album] [--keep-running] [-h|--help]

Clear imported album data (database + derived intermediates).
Keeps source photos (data/album) and model weights (data/models) unless
--include-album is set.

By default stops this clone's Web/API ports from .env (PORT / SENTRIX_API_PORT)
before deleting files, then leaves services stopped. Restart them yourself.

  -y, --yes           Skip confirmation prompt
  --include-album     Also delete source photos under data/album
  --keep-running      Do not stop Web/API (not recommended)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) yes=1; shift ;;
    --include-album) include_album=1; shift ;;
    --keep-running) keep_running=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$data_dir" ]]; then
  echo "Data directory not found: $data_dir" >&2
  exit 1
fi

echo "Will clear imported data under: $data_dir"
echo "  DB:       $db_path (+ -wal/-shm if present)"
echo "  ANN:      $ann_dir"
echo "  previews: $data_dir/previews"
echo "  media:    $data_dir/media"
echo "  backups:  $data_dir/backups"
if [[ "$include_album" -eq 1 ]]; then
  echo "  album:    $data_dir/album  (source photos)"
else
  echo "  keep:     $data_dir/album, $data_dir/models"
fi
if [[ "$keep_running" -eq 1 ]]; then
  echo "  ports:    keep running (Web $web_port / API $api_port)"
else
  echo "  stop:     Web :$web_port and API :$api_port before clear"
fi

if [[ "$yes" -ne 1 ]]; then
  read -r -p "Continue? [y/N] " answer
  case "${answer:-}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

stop_local_ports() {
  local port
  for port in "$@"; do
    [[ -n "$port" ]] || continue
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
      lsof -tiTCP:"$port" -sTCP:LISTEN | xargs -r kill >/dev/null 2>&1 || true
    else
      echo "Warning: neither fuser nor lsof found; cannot stop port $port" >&2
    fi
  done
  # Give listeners a moment to release the DB file.
  sleep 1
}

if [[ "$keep_running" -ne 1 ]]; then
  echo "Stopping local services on :$web_port and :$api_port ..."
  stop_local_ports "$web_port" "$api_port"
else
  echo "Warning: services left running; clear may not take effect until restart." >&2
fi

rm_dir_contents() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  else
    mkdir -p "$dir"
  fi
}

# Database (+ WAL companions).
rm -f "$db_path" "${db_path}-wal" "${db_path}-shm"

# Derived / intermediate directories.
rm_dir_contents "$ann_dir"
rm_dir_contents "$data_dir/previews"
rm_dir_contents "$data_dir/media"
rm_dir_contents "$data_dir/backups"

if [[ "$include_album" -eq 1 ]]; then
  rm_dir_contents "$data_dir/album"
fi

# Ensure expected empty dirs exist for next import/start.
mkdir -p "$data_dir/media" "$ann_dir" "$data_dir/previews" "$data_dir/backups" "$data_dir/album" "$data_dir/models"

echo "Done. Imported data cleared."
echo "Source album photos kept at: $data_dir/album"
echo "Services on :$web_port / :$api_port were stopped; restart them when ready."
echo "API will recreate an empty schema on next start."
