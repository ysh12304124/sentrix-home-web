#!/usr/bin/env python3
"""Local Sentence-Transformers embedding sidecar.

Runs in the isolated ``.venv-text`` environment (requirements-text.txt) and
exposes a tiny HTTP API so the main API process never imports torch /
sentence-transformers:

  GET  /health   -> {"status": "ok", "model", "dimension"}
  POST /embed    -> {"vector": [...1024 floats...], "model"}

Env:
  SENTRIX_TEXT_EMBED_MODEL       local model path or Hugging Face id
  SENTRIX_TEXT_EMBED_MODEL_ID    stable id persisted with vectors
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
MODEL_ID = os.getenv("SENTRIX_TEXT_EMBED_MODEL_ID", MODEL)
DEVICE = os.getenv("SENTRIX_TEXT_EMBEDDER_DEVICE", "cpu")
MAX_LENGTH = max(32, int(os.getenv("SENTRIX_TEXT_EMBEDDER_MAX_LENGTH", "512")))
QUERY_INSTRUCTION = os.getenv("SENTRIX_TEXT_EMBEDDING_INSTRUCTION", "").strip()
DIMENSION = 1024

_model = None
_lock = threading.Lock()


def _load():
    global _model, DIMENSION
    if _model is None:
        _model = SentenceTransformer(MODEL, device=DEVICE, local_files_only=True)
        _model.max_seq_length = MAX_LENGTH
        dimension_fn = getattr(
            _model, "get_embedding_dimension", _model.get_sentence_embedding_dimension,
        )
        dimension = dimension_fn()
        if dimension:
            DIMENSION = int(dimension)
    return _model


def _encode(model, text: str, input_type: str) -> list[float]:
    """Encode a retrieval query or document with model-native prompting."""
    kwargs = {"normalize_embeddings": True}
    prompts = getattr(model, "prompts", {}) or {}
    prompt_name = "query" if input_type == "query" else "document"
    if input_type == "query" and QUERY_INSTRUCTION:
        text = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"
    elif prompt_name in prompts:
        kwargs["prompt_name"] = prompt_name
    vector = model.encode(text, **kwargs)
    return [float(value) for value in vector.tolist()]


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
                             "model": MODEL_ID, "model_path": MODEL,
                             "dimension": DIMENSION, "max_length": MAX_LENGTH})
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
            input_type = str(body.get("input_type") or "query").strip().lower()
            if input_type not in {"query", "document"}:
                raise ValueError("input_type must be query or document")
            model = _load()
            with _lock:
                vector = _encode(model, text, input_type)
            self._send(200, {"vector": vector, "model": MODEL_ID,
                             "dimension": len(vector), "input_type": input_type})
        except Exception as exc:
            self._send(500, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SENTRIX_TEXT_EMBEDDER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SENTRIX_TEXT_EMBEDDER_PORT", "8101")))
    args = parser.parse_args()
    _load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"text embedding sidecar ready on {args.host}:{args.port} "
          f"model={MODEL_ID} path={MODEL} device={DEVICE}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
