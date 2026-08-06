#!/usr/bin/env python3
"""bge-m3 embedding sidecar (Phase R9-4).

Runs in the isolated ``.venv-text`` environment (requirements-text.txt) and
exposes a tiny HTTP API so the main API process never imports torch /
sentence-transformers:

  GET  /health   -> {"status": "ok", "model", "dimension"}
  POST /embed    -> {"vector": [...1024 floats...], "model"}

Env:
  SENTRIX_TEXT_EMBED_MODEL       model id (default BAAI/bge-m3)
  SENTRIX_TEXT_EMBEDDER_DEVICE   cpu | cuda (default cpu)
  SENTRIX_TEXT_EMBEDDER_PORT     listen port (default 8101)
  SENTRIX_TEXT_EMBEDDER_HOST     bind host (default 127.0.0.1)

Start (in .venv-text):
  PYTHONNOUSERSITE=1 .venv-text/bin/python scripts/maintenance/text_embedder_sidecar.py
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer

MODEL = os.getenv("SENTRIX_TEXT_EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("SENTRIX_TEXT_EMBEDDER_DEVICE", "cpu")
DIMENSION = 1024

_model = None
_lock = threading.Lock()


def _load():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL, device=DEVICE)
    return _model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok" if _model is not None else "loading",
                             "model": MODEL, "dimension": DIMENSION})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            text = str(body.get("text") or "")
            model = _load()
            with _lock:
                vector = [float(value) for value in model.encode(text).tolist()]
            self._send(200, {"vector": vector, "model": MODEL})
        except Exception as exc:
            self._send(500, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SENTRIX_TEXT_EMBEDDER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SENTRIX_TEXT_EMBEDDER_PORT", "8101")))
    args = parser.parse_args()
    _load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"bge-m3 sidecar ready on {args.host}:{args.port} model={MODEL} device={DEVICE}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
