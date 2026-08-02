"""Face embedding contracts and quality signals owned by Sentrix."""

import math
import os
from pathlib import Path
from dataclasses import dataclass


class FaceEmbeddingUnavailable(RuntimeError):
    """Raised when a configured identity model cannot be loaded."""


def normalize_embedding(values):
    numbers = [float(value) for value in (values or [])]
    norm = math.sqrt(sum(value * value for value in numbers))
    if norm == 0:
        return []
    return [value / norm for value in numbers]


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float]
    model_name: str
    model_version: str
    quality_signal: float


class FaceEmbeddingAdapter:
    """Small adapter boundary that can be backed by AdaFace, MagFace, or tests.

    The backend receives an aligned face crop and may return either an embedding
    or ``(embedding, quality_signal)``. No fallback to another model is allowed
    inside this boundary because that would mislabel stored vectors.
    """

    def __init__(self, model_name, model_version="unconfigured", backend=None):
        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self.backend = backend

    @property
    def available(self):
        return callable(self.backend)

    def embed(self, aligned_face):
        if not self.available:
            raise FaceEmbeddingUnavailable(
                f"{self.model_name} embedding backend is unavailable"
            )
        raw = self.backend(aligned_face)
        quality_signal = None
        if isinstance(raw, tuple) and len(raw) == 2:
            raw, quality_signal = raw
        values = [float(value) for value in (raw or [])]
        normalized = normalize_embedding(values)
        if not normalized:
            raise FaceEmbeddingUnavailable(
                f"{self.model_name} returned an empty embedding"
            )
        if quality_signal is None:
            quality_signal = math.sqrt(sum(value * value for value in values))
        return EmbeddingResult(
            embedding=normalized,
            model_name=self.model_name,
            model_version=self.model_version,
            quality_signal=float(quality_signal),
        )


class AdaFaceAdapter(FaceEmbeddingAdapter):
    """Named production boundary for the configured AdaFace model.

    Loading a concrete TorchScript/ONNX checkpoint is deliberately kept in the
    runtime integration layer. Until a backend is supplied, ``available`` is
    false and callers must report that AdaFace is unavailable instead of storing
    a vector produced by another model under the AdaFace name.
    """

    def __init__(self, model_path=None, architecture="ir_50", device=None, model_version=None, backend=None):
        self.model_path = Path(model_path or os.getenv("ADAFACE_MODEL_PATH", ""))
        self.architecture = architecture or os.getenv("ADAFACE_ARCHITECTURE", "ir_50")
        self.device = device or os.getenv("ADAFACE_DEVICE", "auto")
        self._model = None
        self._torch = None
        self._using_external_backend = backend is not None
        version = model_version or os.getenv("ADAFACE_MODEL_VERSION") or (
            self.model_path.name if str(self.model_path) else "unconfigured"
        )
        super().__init__("adaface", version, backend or self._infer)

    def _device(self, torch):
        requested = str(self.device or "auto").strip().lower()
        if requested == "auto":
            cuda = getattr(torch, "cuda", None)
            return "cuda:0" if cuda and cuda.is_available() else "cpu"
        return requested

    @property
    def available(self):
        if not callable(self.backend):
            return False
        if self._using_external_backend:
            return True
        return self.model_path.is_file()

    def _repository_root(self):
        """Find the official AdaFace source shipped beside a configured checkpoint."""
        configured = os.getenv("ADAFACE_REPO_ROOT", "").strip()
        candidates = [Path(configured)] if configured else []
        candidates.extend(getattr(self.model_path, "parents", ()))
        for candidate in candidates:
            if (candidate / "net.py").is_file():
                return str(candidate)
        return None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FaceEmbeddingUnavailable(
                f"AdaFace checkpoint does not exist: {self.model_path}"
            )
        try:
            import sys
            import torch

            repo_root = self._repository_root()
            if repo_root and repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            import net

            model = net.build_model(self.architecture)
            # The official AdaFace file is a trusted Lightning checkpoint. It
            # contains a ModelCheckpoint object, so PyTorch 2.6's restricted
            # weights-only loader cannot deserialize it.
            self.device = self._device(torch)
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {
                key[6:]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            } or state_dict
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._torch = torch
            self._model = model
            return model
        except FaceEmbeddingUnavailable:
            raise
        except Exception as error:
            raise FaceEmbeddingUnavailable(f"unable to load AdaFace: {error}") from error

    def _infer(self, aligned_face):
        try:
            import numpy as np

            torch = self._torch or __import__("torch")
            model = self._load_model()
            if hasattr(aligned_face, "convert"):
                image = np.asarray(aligned_face.convert("RGB"))
            else:
                image = np.asarray(aligned_face)
            if image.ndim != 3 or image.shape[2] != 3:
                raise FaceEmbeddingUnavailable("AdaFace expects an RGB 3-channel aligned crop")
            if image.shape[0] != 112 or image.shape[1] != 112:
                from PIL import Image

                image = np.asarray(Image.fromarray(image.astype("uint8")).resize((112, 112)))
            bgr = ((image[:, :, ::-1].astype("float32") / 255.0) - 0.5) / 0.5
            tensor = torch.tensor(bgr.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                feature, norm = model(tensor)
            vector = feature[0].detach().cpu().tolist()
            quality = float(norm[0].detach().cpu().reshape(-1)[0]) if norm is not None else 0.0
            return vector, quality
        except FaceEmbeddingUnavailable:
            raise
        except Exception as error:
            raise FaceEmbeddingUnavailable(f"unable to run AdaFace: {error}") from error


class MagFaceAdapter(FaceEmbeddingAdapter):
    """Named comparison boundary for a configured MagFace model."""

    def __init__(self, model_version="unconfigured", backend=None):
        super().__init__("magface", model_version, backend)


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, float(value)))


def pose_bucket(pose):
    """Return a stable view bucket; InsightFace pose is pitch, yaw, roll."""
    values = [float(value) for value in (pose if pose is not None else [])]
    yaw = values[1] if len(values) > 1 else 0.0
    if abs(yaw) <= 20.0:
        return "frontal"
    return "profile_right" if yaw > 0 else "profile_left"


def compute_face_quality(detection_confidence, area_ratio, sharpness, pose):
    """Compute a bounded quality score without treating profile as no-face."""
    values = [float(value) for value in (pose if pose is not None else [])]
    yaw = abs(values[1]) if len(values) > 1 else 0.0
    pose_score = max(0.15, 1.0 - min(90.0, yaw) / 90.0)
    area_score = _clamp(float(area_ratio) / 0.08)
    sharpness_score = _clamp(sharpness)
    return _clamp(
        0.40 * _clamp(detection_confidence)
        + 0.20 * area_score
        + 0.20 * sharpness_score
        + 0.20 * pose_score
    )
