"""bge-m3 text query embedder (Phase R D3 backup, inactive by default).

Activated only when the R1B text-retrieval evaluation fails and the user picks
bge-m3.  Kept behind an import gate so the repository works without the
``sentence-transformers`` dependency installed.

Architecture note: this is an independent ``TextQueryEmbedder`` — swapping it
does NOT touch the visual space, QuerySpec, Gate or Kernel.  P0-2/P0-3.
"""

from __future__ import annotations

import os


class BgeM3TextQueryEmbedder:
    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or os.getenv("SENTRIX_TEXT_EMBED_MODEL", "BAAI/bge-m3")
        self._model = None

    @property
    def model_id(self):
        return self._model_id

    @property
    def dimension(self):
        return 1024

    @property
    def available(self):
        try:
            return self._load() is not None
        except Exception:
            return False

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_id)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        if model is None:
            return []
        try:
            return [float(value) for value in model.encode(str(text or "")).tolist()]
        except Exception:
            return []
