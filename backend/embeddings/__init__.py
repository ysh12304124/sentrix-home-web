"""Embedding router package (Phase R P0-3).

Exports the router so composition roots can ``from .embeddings import
EmbeddingRouter`` without reaching into internal adapter modules.
"""

from .base import TextQueryEmbedder, VisualQueryEmbedder
from .router import EmbeddingRouter

__all__ = ["EmbeddingRouter", "VisualQueryEmbedder", "TextQueryEmbedder"]
