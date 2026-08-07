"""Chinese-CLIP visual query embedder (Phase R D3, backup for ViT-B-32).

R1B proved the standard ViT-B-32 CLIP text-to-image alignment is effectively
random for Chinese (AUC ~0.51).  Chinese-CLIP (ViT-L/14) is the D3-selected
visual replacement: the text encoder is a Chinese BERT so ``毛绒睡衣`` aligns
with a hoodie image that formation labelled as ``连帽衫``.

Implements ``VisualQueryEmbedder`` and adds ``embed_image`` so the visual ANN
index can be rebuilt in the same space.  Requires ``cn_clip`` and a
Chinese-CLIP checkpoint (``~/.cache/clip/clip_cn_vit-l-14.pt`` on 153).
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_CHECKPOINT = os.getenv(
    "CHINESE_CLIP_CHECKPOINT",
    str(Path.home() / ".cache" / "clip" / "clip_cn_vit-l-14.pt"),
)


class ChineseClipVisualEmbedder:
    def __init__(self, checkpoint: str | None = None, model_name: str = "ViT-L-14", device: str | None = None):
        self.checkpoint = checkpoint or _DEFAULT_CHECKPOINT
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._device = device or os.getenv("CLIP_DEVICE", "cpu")
        self._error = None

    @property
    def model_id(self):
        return f"chinese-clip-{self.model_name}"

    @property
    def dimension(self):
        return 768

    @property
    def available(self):
        try:
            return self._load() is not None
        except Exception:
            return False

    def _load(self):
        if self._model is not None:
            return self._model
        if not Path(self.checkpoint).is_file():
            self._error = f"checkpoint missing: {self.checkpoint}"
            return None
        try:
            from cn_clip.clip import load_from_name
            model, preprocess = load_from_name(self.model_name, device=self._device)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            return model
        except Exception as error:
            self._error = str(error)
            return None

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        if model is None or not str(text or "").strip():
            return []
        try:
            import torch
            from cn_clip.clip import tokenize
            with torch.no_grad():
                vector = model.encode_text(tokenize([str(text)]).to(self._device))
            return [float(value) for value in vector[0].cpu().tolist()]
        except Exception:
            return []

    def embed_image(self, path: str) -> list[float]:
        model = self._load()
        if model is None or not Path(path).is_file():
            return []
        try:
            import torch
            from PIL import Image
            from backend.image_io import ensure_heif_support

            ensure_heif_support()
            image = self._preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self._device)
            with torch.no_grad():
                vector = model.encode_image(image)
            return [float(value) for value in vector[0].cpu().tolist()]
        except Exception:
            return []
