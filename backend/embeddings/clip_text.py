"""CLIP text query embedder.

Wraps ``ClipAdapter.embed_text`` as a ``TextQueryEmbedder`` for the
Observation / Event text ANN index.  The current index was built with the same
CLIP text encoder, so this is the matching query encoder today.

If R1B text retrieval fails for Chinese, swap to ``bge_text.py`` under the
same protocol — no Kernel / QuerySpec change required.
"""

from __future__ import annotations


class ClipTextQueryEmbedder:
    def __init__(self, clip, model_id: str | None = None):
        self._clip = clip
        self._model_id = model_id or getattr(clip, "model_name", "ViT-B-32")

    @property
    def model_id(self):
        return self._model_id

    @property
    def dimension(self):
        return 512

    @property
    def available(self):
        return bool(getattr(self._clip, "evidence_ready", False))

    def embed_query(self, text: str) -> list[float]:
        if not self.available:
            return []
        try:
            vector = self._clip.embed_text(text)
        except Exception:
            return []
        return vector or []
