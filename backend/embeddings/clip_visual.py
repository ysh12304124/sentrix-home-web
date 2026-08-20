"""CLIP visual query embedder.

Wraps ``ClipAdapter.embed_text`` as a ``VisualQueryEmbedder``: the query is
embedded by the same text encoder that produced the Asset image embeddings in
the visual ANN index, so text-to-image cosine is meaningful.

Intended for the *current* checkpoint (ViT-B-32); if R1B shows the Chinese
text capability is insufficient, replace this adapter with a Chinese-CLIP
backed one under the same protocol (Phase R D3/D8).
"""

from __future__ import annotations


class ClipVisualQueryEmbedder:
    def __init__(self, clip, model_id: str | None = None):
        self._clip = clip
        self._model_id = model_id or getattr(clip, "model_name", "ViT-B-32")

    @property
    def model_id(self):
        return self._model_id

    @property
    def dimension(self):
        return int(getattr(self._clip, "embedding_dimension", 512))

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
