#!/bin/bash
# Sentrix production web portal (4174).
# Static frontend + API proxy to 8091.
# PhotoBench evaluation service (8771) is lazy-started on first QA click
# via POST /api/photobench/ensure handled by server.js.

root=/home/asus/Github/Sentrix-Home-Web
cd "$root" || exit 1

exec env PORT=4174 \
  SENTRIX_BACKEND_URL=http://127.0.0.1:8091 \
  PHOTOBENCH_PYTHON="$root/.venv/bin/python" \
  node server.js
