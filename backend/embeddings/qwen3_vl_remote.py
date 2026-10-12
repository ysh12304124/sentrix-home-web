"""HTTP adapter for the local Qwen3-VL-Embedding sidecar."""

from __future__ import annotations

import os
import threading

import httpx


class Qwen3VLVisualAdapter:
    """Drop-in replacement for ``ClipAdapter`` in the visual vector slot."""

    _shared = None
    _shared_lock = threading.Lock()

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv(
            "SENTRIX_QWEN3_VL_EMBEDDER_URL", "http://127.0.0.1:8102",
        )).rstrip("/")
        self.enabled = True
        self.model_name = os.getenv(
            "SENTRIX_QWEN3_VL_EMBED_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B",
        )
        self.device = os.getenv("SENTRIX_QWEN3_VL_EMBEDDER_DEVICE", "cuda")
        self.error = None
        self.weights_ready = True
        self._timeout = float(os.getenv("SENTRIX_QWEN3_VL_EMBEDDER_TIMEOUT", "180"))

    @classmethod
    def shared(cls):
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def embedding_dimension(self):
        return 2048

    # EmbeddingRouter's visual-query slot uses the generic model_id/dimension/
    # available/embed_query protocol.  The same adapter is also used by the
    # ingestion pipeline, whose older protocol names these model_name,
    # embedding_dimension, evidence_ready and embed_text respectively.
    @property
    def model_id(self):
        return self.model_name

    @property
    def dimension(self):
        return self.embedding_dimension

    @property
    def available(self):
        return self.evidence_ready

    @property
    def evidence_ready(self):
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=3)
            self.error = None if response.status_code == 200 else response.text[:300]
            return response.status_code == 200
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def _embed(self, payload: dict) -> list[float]:
        try:
            response = httpx.post(f"{self.base_url}/embed", json=payload, timeout=self._timeout)
            response.raise_for_status()
            vector = response.json().get("vector") or []
            self.error = None
            return [float(value) for value in vector]
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []

    def embed_images(self, paths, *, batch_size: int = 1) -> list[list[float]]:
        values = [str(path) for path in paths]
        vectors: list[list[float]] = []
        try:
            for offset in range(0, len(values), max(1, int(batch_size))):
                batch = values[offset:offset + max(1, int(batch_size))]
                response = httpx.post(
                    f"{self.base_url}/embed",
                    json={"inputs": [{"image": path} for path in batch]},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                rows = response.json().get("vectors") or []
                if len(rows) != len(batch):
                    raise RuntimeError("Qwen3-VL sidecar returned an unexpected vector count")
                vectors.extend([[float(value) for value in row] for row in rows])
            self.error = None
            return vectors
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []

    def embed_image(self, path):
        return self._embed({"image": str(path)})

    def embed_text(self, text):
        if not str(text or "").strip():
            return []
        instruction = os.getenv(
            "SENTRIX_QWEN3_VL_QUERY_INSTRUCTION",
            "Retrieve images or video keyframes relevant to the user's personal memory question.",
        )
        return self._embed({"text": str(text), "instruction": instruction})

    def embed_query(self, text):
        return self.embed_text(text)


def create_visual_media_embedder():
    kind = os.getenv("SENTRIX_IMAGE_EMBEDDER", "clip").strip().lower()
    if kind in {"qwen3_vl", "qwen3-vl", "qwen_vl"}:
        return Qwen3VLVisualAdapter.shared()
    from ..model_clients import ClipAdapter

    return ClipAdapter()
