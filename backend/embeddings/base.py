"""Embedding-router contracts (Phase R P0-3).

The Thin Agent, Gate and QuerySpec never depend on a concrete embedding model.
Two independent query embedder protocols keep visual cross-modal and text
retrieval in separate adapter slots so a failing Chinese encoder can be
replaced without touching the Kernel or Agent:

- ``VisualQueryEmbedder``: text query -> vector in the SAME space as Asset
  image embeddings (a CLIP-family text encoder aligned to the image encoder).
- ``TextQueryEmbedder``: text query -> vector in the space of Observation /
  Event text embeddings.

Both are deliberately narrow: no model metadata, no device, no batching —
just ``model_id``, ``dimension`` and ``embed_query``.
"""

from __future__ import annotations

from typing import Protocol


class VisualQueryEmbedder(Protocol):
    model_id: str
    dimension: int

    def embed_query(self, text: str) -> list[float]: ...


class TextQueryEmbedder(Protocol):
    model_id: str
    dimension: int

    def embed_query(self, text: str) -> list[float]: ...
