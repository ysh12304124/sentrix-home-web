"""Embedding router package (Phase R P0-3).

Exports the router so composition roots can ``from .embeddings import
EmbeddingRouter`` without reaching into internal adapter modules.
"""

from .base import TextQueryEmbedder, VisualQueryEmbedder
from .chinese_clip_visual import ChineseClipVisualEmbedder
from .router import EmbeddingRouter

__all__ = ["EmbeddingRouter", "ChineseClipVisualEmbedder", "VisualQueryEmbedder", "TextQueryEmbedder"]
