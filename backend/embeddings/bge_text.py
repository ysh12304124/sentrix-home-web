"""bge-m3 text query embedder — HTTP SidecarClient (Phase R9-4).

The bge-m3 model lives in an isolated venv (``.venv-text``, requirements-text.txt)
behind a small HTTP sidecar (scripts/maintenance/text_embedder_sidecar.py).  The
main API process never imports ``torch`` / ``sentence-transformers``; this client
only talks HTTP and keeps a lightweight circuit breaker so a dead sidecar degrades
to ``text_available=False`` instead of hanging every query.

Swapping this embedder does NOT touch the visual space, QuerySpec, Gate or
Kernel (P0-2/P0-3) — it is an independent ``TextQueryEmbedder``.
"""

from __future__ import annotations

import os
import time

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None

_DEFAULT_URL = os.getenv("SENTRIX_TEXT_EMBEDDER_URL", "http://127.0.0.1:8101")


class BgeM3TextQueryEmbedder:
    def __init__(self, model_id: str | None = None, base_url: str | None = None):
        self._model_id = model_id or os.getenv("SENTRIX_TEXT_EMBED_MODEL", "BAAI/bge-m3")
        self._base_url = (base_url or _DEFAULT_URL).rstrip("/")
        self._timeout = float(os.getenv("SENTRIX_TEXT_EMBEDDER_TIMEOUT", "10"))
        self._failures = 0
        self._tripped_at = 0.0

    @property
    def model_id(self):
        return self._model_id

    @property
    def dimension(self):
        return 1024

    @property
    def available(self):
        if httpx is None:
            return False
        if self._tripped_at and time.monotonic() - self._tripped_at < 30.0:
            return False
        try:
            response = httpx.get(f"{self._base_url}/health", timeout=1.5)
            return response.status_code == 200
        except Exception:
            return False

    def _record_failure(self):
        self._failures += 1
        if self._failures >= 3:
            self._tripped_at = time.monotonic()
            self._failures = 0

    def _record_success(self):
        self._failures = 0
        self._tripped_at = 0.0

    def embed_query(self, text: str) -> list[float]:
        if httpx is None:
            return []
        if self._tripped_at and time.monotonic() - self._tripped_at < 30.0:
            return []
        try:
            response = httpx.post(f"{self._base_url}/embed",
                                  json={"text": str(text or "")},
                                  timeout=self._timeout)
            response.raise_for_status()
            vector = response.json().get("vector") or []
            self._record_success()
            return [float(value) for value in vector]
        except Exception:
            self._record_failure()
            return []
