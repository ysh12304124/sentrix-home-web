"""Composition-root EmbeddingRouter.

Owns the two independent embedder slots and exposes a single ``embed`` entry
point for the retrieval layer.  Construction happens at the Application
Composition Root (``app.py``); the Kernel and Agent only ever see this object.

Selection is env-driven:
  SENTRIX_IMAGE_EMBEDDER = clip | chinese_clip        (visual slot)
  SENTRIX_TEXT_EMBEDDER  = clip | bge                  (text slot)

A slot whose embedder is unavailable reports ``available=False`` and the
corresponding ANN retriever skips with a trace reason — never a silent empty.
"""

from __future__ import annotations

import os
import time


class EmbeddingRouter:
    def __init__(self, visual=None, text=None):
        self.visual = visual
        self.text = text
        self._timing_events = []

    @classmethod
    def from_clip(cls, clip):
        """Build with the LOCKED scheme (see embeddings/scheme.py).

        历史上 IMAGE/TEXT 可被 env 切成 clip/chinese_clip/bge，导致向量/索引多套并存、
        检索时好时坏。现钉死单一方案，不再读 env 选择：IMAGE=chinese_clip、TEXT=bge。
        """
        from . import scheme
        image_kind = scheme.IMAGE_EMBEDDER
        text_kind = scheme.TEXT_EMBEDDER
        visual = None
        if image_kind == "chinese_clip":
            from .chinese_clip_visual import ChineseClipVisualEmbedder
            visual = ChineseClipVisualEmbedder.shared()
        elif clip is not None:
            from .clip_visual import ClipVisualQueryEmbedder
            visual = ClipVisualQueryEmbedder(clip)
        else:
            raise ValueError(f"unknown SENTRIX_IMAGE_EMBEDDER: {image_kind}")
        text = None
        if text_kind == "bge":
            from .bge_text import BgeM3TextQueryEmbedder
            text = BgeM3TextQueryEmbedder()
        elif clip is not None:
            from .clip_text import ClipTextQueryEmbedder
            text = ClipTextQueryEmbedder(clip)
        else:
            text = None
        return cls(visual=visual, text=text)

    @property
    def visual_available(self):
        return bool(self.visual and getattr(self.visual, "available", False))

    @property
    def text_available(self):
        return bool(self.text and getattr(self.text, "available", False))

    def embed_visual(self, text: str) -> list[float]:
        started = time.monotonic()
        try:
            if not self.visual_available:
                return []
            return self.visual.embed_query(text) or []
        finally:
            self._timing_events.append({
                "slot": "visual",
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            })

    def embed_text(self, text: str) -> list[float]:
        started = time.monotonic()
        try:
            if not self.text_available:
                return []
            return self.text.embed_query(text) or []
        finally:
            self._timing_events.append({
                "slot": "text",
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            })

    def get_and_clear_timing_events(self) -> list[dict]:
        events = list(self._timing_events)
        self._timing_events.clear()
        return events

    @staticmethod
    def _slot_status(slot):
        if slot is None:
            return {"configured": False, "available": False}
        status_fn = getattr(slot, "status", None)
        if callable(status_fn):
            try:
                value = status_fn()
                if isinstance(value, dict):
                    return dict(value)
            except Exception as exc:
                return {"configured": True, "available": False,
                        "status_error": f"{type(exc).__name__}: {exc}"}
        try:
            available = bool(getattr(slot, "available", False))
        except Exception as exc:
            return {"configured": True, "available": False,
                    "status_error": f"{type(exc).__name__}: {exc}"}
        return {
            "configured": True,
            "available": available,
            "model_id": getattr(slot, "model_id", None),
            "dimension": getattr(slot, "dimension", None),
        }

    def status(self) -> dict:
        """Return observable slot health without exposing model internals."""
        return {
            "visual": self._slot_status(self.visual),
            "text": self._slot_status(self.text),
        }
