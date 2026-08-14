#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldMM Keyframe / Event Memory Pipeline
========================================

A single-file reproduction of the user's keyframe method:

1. Candidate scheduling at a configurable scan FPS.
2. Blur / unstable-view gating using Laplacian sharpness, frame difference,
   motion coverage, and phase-correlation camera shift.
3. Information-gain keyframe selection using HSV novelty, frame change, and
   temporal constraints.
4. Optional semantic analysis using YOLO detection, YOLO-World, YOLO Pose,
   MediaPipe hands, and a local Transformers face-expression model.
5. Camera-motion compensated multi-object tracking.
6. Scene, object, action, and expression event aggregation.
7. DMD-based content-boundary analysis.
8. Deterministic 15% research-summary selection and WorldMM-friendly exports.

The core extraction path only requires NumPy and OpenCV. Semantic modules are
optional and fail closed: if a model or package is unavailable, extraction,
DMD analysis, and export still complete.

Example:
    uv run python tools/keyframe_pipeline/worldmm_keyframe_pipeline.py \
      --video data/example.mp4 \
      --output output/keyframe_memory/example \
      --width 640 --sample-fps 10 --analysis-fps 5 \
      --device 0 --yolo-model yolo11n.pt --pose-model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np


LOGGER = logging.getLogger("worldmm_keyframe")


# ---------------------------------------------------------------------------
# Configuration and data structures
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    video: str
    output: str
    width: int = 640
    sample_fps: float = 10.0
    analysis_fps: float = 5.0
    device: str = "0"
    yolo_model: str = "yolo11n.pt"
    world_model: str = ""
    world_classes: list[str] = field(default_factory=list)
    pose_model: str = "yolo11n-pose.pt"
    emotion_model: str = ""
    enable_hands: bool = False
    disable_semantics: bool = False
    allow_model_download: bool = False
    speech_json: str = ""
    subtitle_srt: str = ""
    video_id: str = ""
    max_scan_frames: int = 0
    save_rejected_images: bool = False

    # Quality gate, matching the documented method.
    min_sharpness: float = 28.0
    stable_sharpness_ratio: float = 0.38
    rapid_sharpness_ratio: float = 0.70
    unstable_sharpness_ratio: float = 1.10
    unstable_change: float = 5.0
    rapid_change: float = 18.0
    unstable_shift: float = 0.5
    rapid_shift: float = 4.0
    unstable_motion_coverage: float = 0.10

    # Information-gain selection.
    min_time_gap: float = 0.35
    max_time_gap: float = 2.0
    information_gain_threshold: float = 0.16

    # Scene and summary.
    min_scene_gap: float = 2.5
    min_scene_keyframes: int = 3
    summary_ratio: float = 0.15
    summary_frame_duration: float = 0.5
    summary_min_gap: float = 0.45


@dataclass
class VideoMeta:
    path: str
    video_id: str
    source_fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    codec: str
    file_size_bytes: int
    sha256: str


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float]  # normalized xyxy
    source: str
    track_id: Optional[int] = None


@dataclass
class FrameRecord:
    keyframe_code: str
    frame_index: int
    timestamp: float
    source_image: str
    labeled_image: str
    selection_reason: str
    sharpness: float
    sharpness_threshold: float
    candidate_change: float
    motion_coverage: float
    camera_shift: float
    visual_novelty: float
    information_gain: float
    scene_score: float
    boundary_score: float = 0.0
    speech_activity: float = 0.0
    quality: float = 0.0
    semantic_density: float = 0.0
    summary_score: float = 0.0
    objects: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    expressions: list[dict[str, Any]] = field(default_factory=list)
    hands: list[dict[str, Any]] = field(default_factory=list)
    descriptor: list[float] = field(default_factory=list)
    semantic_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
    temp.replace(path)


def relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def robust_normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    replacement = float(np.nanmedian(array[finite]))
    array = np.where(finite, array, replacement)
    low, high = np.percentile(array, [5.0, 95.0])
    if high - low < 1e-9:
        low, high = float(array.min()), float(array.max())
    if high - low < 1e-9:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def mad(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    height = max(2, int(round(frame.shape[0] * scale)))
    width_even = width - (width % 2)
    height_even = height - (height % 2)
    return cv2.resize(frame, (max(2, width_even), max(2, height_even)), interpolation=cv2.INTER_AREA)


def to_gray_small(frame: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def sharpness_laplacian(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 320:
        scale = 320.0 / gray.shape[1]
        gray = cv2.resize(gray, (320, max(2, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def phase_shift(previous: Optional[np.ndarray], current: np.ndarray) -> tuple[float, tuple[float, float]]:
    if previous is None or previous.shape != current.shape:
        return 0.0, (0.0, 0.0)
    try:
        (dx, dy), response = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
        if not math.isfinite(dx) or not math.isfinite(dy) or response < 0.01:
            return 0.0, (0.0, 0.0)
        return float(math.hypot(dx, dy)), (float(dx), float(dy))
    except cv2.error:
        return 0.0, (0.0, 0.0)


def hsv_histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.astype(np.float32)


def histogram_novelty(previous: Optional[np.ndarray], current: np.ndarray) -> float:
    if previous is None:
        return 1.0
    return clamp(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return clamp(1.0 - float(np.dot(a, b) / denom), 0.0, 2.0)


def frame_descriptor(frame: np.ndarray) -> np.ndarray:
    """Compact deterministic descriptor used for diversity/representativeness."""
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    channels = []
    for channel, bins, maximum in [(0, 16, 180), (1, 8, 256), (2, 8, 256)]:
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, maximum]).reshape(-1)
        hist = hist / max(float(hist.sum()), 1e-9)
        channels.append(hist)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    pooled = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255.0
    vector = np.concatenate([*channels, pooled]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def normalized_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [clamp(x1 / width), clamp(y1 / height), clamp(x2 / width), clamp(y2 / height)]


def denormalized_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(clamp(float(x1)) * width),
        int(clamp(float(y1)) * height),
        int(clamp(float(x2)) * width),
        int(clamp(float(y2)) * height),
    )


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_distance(a: Sequence[float], b: Sequence[float], shift_norm: tuple[float, float] = (0.0, 0.0)) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    bx -= shift_norm[0]
    by -= shift_norm[1]
    return float(math.hypot(ax - bx, ay - by) / math.sqrt(2.0))


def point_in_expanded_box(point: tuple[float, float], box: Sequence[float], expansion: float = 0.08) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - expansion <= x <= x2 + expansion and y1 - expansion <= y <= y2 + expansion


def parse_world_classes(raw: str) -> list[str]:
    if not raw.strip():
        return []
    candidate = Path(raw)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
        if candidate.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------


def probe_video(video_path: Path, video_id: str = "") -> VideoMeta:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = safe_float(capture.get(cv2.CAP_PROP_FPS), 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    capture.release()
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
    return VideoMeta(
        path=str(video_path.resolve()),
        video_id=video_id or video_path.stem,
        source_fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        width=width,
        height=height,
        codec=codec,
        file_size_bytes=video_path.stat().st_size,
        sha256=sha256_file(video_path),
    )


# ---------------------------------------------------------------------------
# Optional semantic analyzers
# ---------------------------------------------------------------------------


class SemanticAnalyzer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = None
        self.world_detector = None
        self.pose_detector = None
        self.hand_detector = None
        self.emotion_processor = None
        self.emotion_model = None
        self.emotion_id2label: dict[int, str] = {}
        self.torch = None
        self.loaded_components: dict[str, Any] = {}
        if config.disable_semantics:
            LOGGER.info("Semantic analysis disabled by command-line option.")
            return
        self._load_ultralytics()
        self._load_hands()
        self._load_emotion()

    def _load_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO
            try:
                from ultralytics import YOLOWorld  # type: ignore
            except Exception:
                YOLOWorld = None
        except Exception as exc:
            LOGGER.warning("Ultralytics unavailable; object/pose models disabled: %s", exc)
            return

        def safe_load(name: str, kind: str, world: bool = False):
            if not name:
                return None
            path = Path(name)
            if not path.exists() and not self.config.allow_model_download:
                LOGGER.warning(
                    "%s model %r is not a local file and downloads are disabled; skip it. "
                    "Use --allow-model-download or provide a local model path.",
                    kind,
                    name,
                )
                return None
            try:
                if world and YOLOWorld is not None:
                    model = YOLOWorld(name)
                else:
                    model = YOLO(name)
                LOGGER.info("Loaded %s model: %s", kind, name)
                return model
            except Exception as exc:
                LOGGER.warning("Failed to load %s model %r: %s", kind, name, exc)
                return None

        self.detector = safe_load(self.config.yolo_model, "YOLO detector")
        self.pose_detector = safe_load(self.config.pose_model, "YOLO pose")
        self.world_detector = safe_load(self.config.world_model, "YOLO-World", world=True)
        if self.world_detector is not None and self.config.world_classes:
            try:
                self.world_detector.set_classes(self.config.world_classes)
            except Exception as exc:
                LOGGER.warning("Unable to set YOLO-World classes: %s", exc)
        self.loaded_components.update(
            {
                "yolo": bool(self.detector),
                "yolo_world": bool(self.world_detector),
                "pose": bool(self.pose_detector),
            }
        )

    def _load_hands(self) -> None:
        if not self.config.enable_hands:
            self.loaded_components["mediapipe_hands"] = False
            return
        try:
            import mediapipe as mp  # type: ignore

            self.hand_detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=4,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self.loaded_components["mediapipe_hands"] = True
            LOGGER.info("Loaded MediaPipe Hands.")
        except Exception as exc:
            self.loaded_components["mediapipe_hands"] = False
            LOGGER.warning("MediaPipe Hands unavailable: %s", exc)

    def _load_emotion(self) -> None:
        if not self.config.emotion_model:
            self.loaded_components["emotion"] = False
            return
        model_path = Path(self.config.emotion_model)
        if not model_path.exists() and not self.config.allow_model_download:
            LOGGER.warning("Emotion model is not local and downloads are disabled: %s", model_path)
            self.loaded_components["emotion"] = False
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.torch = torch
            local_only = not self.config.allow_model_download
            self.emotion_processor = AutoImageProcessor.from_pretrained(
                self.config.emotion_model, local_files_only=local_only
            )
            self.emotion_model = AutoModelForImageClassification.from_pretrained(
                self.config.emotion_model, local_files_only=local_only
            )
            device = "cuda:0" if str(self.config.device) != "cpu" and torch.cuda.is_available() else "cpu"
            self.emotion_model.to(device).eval()
            self.emotion_device = device
            self.emotion_id2label = {
                int(key): str(value)
                for key, value in getattr(self.emotion_model.config, "id2label", {}).items()
            }
            self.loaded_components["emotion"] = True
            LOGGER.info("Loaded face-expression model: %s on %s", self.config.emotion_model, device)
        except Exception as exc:
            self.loaded_components["emotion"] = False
            LOGGER.warning("Face-expression model unavailable: %s", exc)

    def _predict_ultralytics(self, model: Any, frame: np.ndarray, confidence: float) -> Any:
        if model is None:
            return None
        try:
            results = model.predict(
                source=frame,
                conf=confidence,
                iou=0.70,
                imgsz=max(frame.shape[:2]),
                device=self.config.device,
                verbose=False,
            )
            return results[0] if results else None
        except Exception as exc:
            LOGGER.warning("Ultralytics inference failed and this model will be disabled: %s", exc)
            if model is self.detector:
                self.detector = None
            elif model is self.world_detector:
                self.world_detector = None
            elif model is self.pose_detector:
                self.pose_detector = None
            return None

    @staticmethod
    def _extract_detections(result: Any, source: str, width: int, height: int) -> list[Detection]:
        detections: list[Detection] = []
        if result is None or getattr(result, "boxes", None) is None:
            return detections
        names = getattr(result, "names", {}) or {}
        boxes = result.boxes
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            return detections
        ambiguous = {"person", "chair", "cup", "bottle", "book", "cell phone", "remote"}
        for box, confidence, class_id in zip(xyxy, confs, classes):
            label = str(names.get(int(class_id), class_id)).strip().lower().replace("_", " ")
            threshold = 0.48 if source == "yolo_world" and label in ambiguous else (
                0.42 if source == "yolo" and label in ambiguous else (0.30 if source == "yolo_world" else 0.34)
            )
            if float(confidence) < threshold:
                continue
            normalized = normalized_box(box, width, height)
            area = max(0.0, normalized[2] - normalized[0]) * max(0.0, normalized[3] - normalized[1])
            if area < 0.0002 or area > 0.98:
                continue
            detections.append(
                Detection(label=label, confidence=float(confidence), bbox=normalized, source=source)
            )
        return detections

    @staticmethod
    def _merge_detections(detections: list[Detection]) -> list[Detection]:
        merged: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for existing in merged:
                same_or_alias = detection.label == existing.label or {
                    detection.label,
                    existing.label,
                } <= {"cell phone", "phone", "mobile phone"}
                if same_or_alias and box_iou(detection.bbox, existing.bbox) >= 0.55:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(detection)
        return merged

    @staticmethod
    def _pose_actions(result: Any, detections: list[Detection]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        actions: list[dict[str, Any]] = []
        poses: list[dict[str, Any]] = []
        if result is None or getattr(result, "keypoints", None) is None:
            return actions, poses
        try:
            keypoints = result.keypoints.xyn.detach().cpu().numpy()
            confidences = result.keypoints.conf.detach().cpu().numpy()
        except Exception:
            return actions, poses
        person_detections = [item for item in detections if item.label == "person"]
        interactive_labels = {"cup", "bottle", "book", "cell phone", "phone", "laptop", "fork", "spoon"}
        for pose_index, (points, scores) in enumerate(zip(keypoints, confidences)):
            valid = scores >= 0.25
            if int(valid.sum()) < 6:
                continue
            valid_points = points[valid]
            min_x, min_y = valid_points.min(axis=0)
            max_x, max_y = valid_points.max(axis=0)
            pose_box = [float(min_x), float(min_y), float(max_x), float(max_y)]
            track_id = None
            if person_detections:
                nearest = min(person_detections, key=lambda item: center_distance(item.bbox, pose_box))
                track_id = nearest.track_id
            pose_record = {
                "pose_index": pose_index,
                "track_id": track_id,
                "bbox": pose_box,
                "keypoints": points.tolist(),
                "confidence": scores.tolist(),
            }
            poses.append(pose_record)

            labels: set[str] = set()
            # COCO: shoulders 5/6, wrists 9/10, hips 11/12, knees 13/14, ankles 15/16.
            for shoulder, wrist in [(5, 9), (6, 10)]:
                if scores[shoulder] >= 0.25 and scores[wrist] >= 0.25 and points[wrist][1] < points[shoulder][1] - 0.04:
                    labels.add("raising hand")
            torso_height = abs(float(points[12][1] - points[6][1])) if scores[12] > 0.25 and scores[6] > 0.25 else 0.0
            leg_height = abs(float(points[16][1] - points[12][1])) if scores[16] > 0.25 and scores[12] > 0.25 else 0.0
            if torso_height > 0.08 and leg_height > 0.18:
                labels.add("standing")
            elif torso_height > 0.06 and 0.05 < leg_height <= 0.18:
                labels.add("sitting")

            wrist_points = []
            for wrist in (9, 10):
                if scores[wrist] >= 0.25:
                    wrist_points.append((float(points[wrist][0]), float(points[wrist][1])))
            for obj in detections:
                if obj.label not in interactive_labels:
                    continue
                if any(point_in_expanded_box(point, obj.bbox, expansion=0.10) for point in wrist_points):
                    labels.add(f"interacting with {obj.label}")
            for label in sorted(labels):
                actions.append(
                    {
                        "label": label,
                        "track_id": track_id,
                        "confidence": float(np.mean(scores[valid])),
                        "source": "pose_rule",
                    }
                )
        return actions, poses

    def _detect_hands(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.hand_detector is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hand_detector.process(rgb)
        except Exception as exc:
            LOGGER.warning("MediaPipe hand inference failed: %s", exc)
            self.hand_detector = None
            return []
        hands: list[dict[str, Any]] = []
        if not result.multi_hand_landmarks:
            return hands
        handedness = result.multi_handedness or []
        for index, landmarks in enumerate(result.multi_hand_landmarks):
            points = np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
            hand_label = "unknown"
            score = 0.0
            if index < len(handedness) and handedness[index].classification:
                hand_label = handedness[index].classification[0].label.lower()
                score = float(handedness[index].classification[0].score)
            tips = [4, 8, 12, 16, 20]
            bases = [2, 5, 9, 13, 17]
            extended = [points[tip, 1] < points[base, 1] for tip, base in zip(tips[1:], bases[1:])]
            thumb_index_distance = float(np.linalg.norm(points[4, :2] - points[8, :2]))
            if thumb_index_distance < 0.055:
                gesture = "pinch"
            elif all(extended):
                gesture = "open hand"
            elif not any(extended):
                gesture = "fist"
            elif extended[0] and not any(extended[1:]):
                gesture = "pointing"
            else:
                gesture = "hand motion"
            hands.append(
                {
                    "hand_id": index,
                    "handedness": hand_label,
                    "confidence": score,
                    "gesture": gesture,
                    "wrist": points[0, :2].tolist(),
                    "landmarks": points.tolist(),
                }
            )
        return hands

    def _classify_expression(self, frame: np.ndarray, person_boxes: list[Detection]) -> list[dict[str, Any]]:
        if self.emotion_model is None or self.emotion_processor is None or not person_boxes:
            return []
        # Haar is used as an offline fallback face detector. YuNet can be supplied by
        # replacing this method without changing the export schema.
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))
        expressions: list[dict[str, Any]] = []
        height, width = frame.shape[:2]
        for x, y, w, h in faces:
            face_box = normalized_box([x, y, x + w, y + h], width, height)
            containing_person = None
            cx, cy = box_center(face_box)
            for person in person_boxes:
                px1, py1, px2, py2 = person.bbox
                if px1 <= cx <= px2 and py1 <= cy <= py1 + 0.58 * (py2 - py1):
                    containing_person = person
                    break
            if containing_person is None:
                continue
            crop = frame[max(0, y): min(height, y + h), max(0, x): min(width, x + w)]
            if crop.size == 0:
                continue
            try:
                from PIL import Image

                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inputs = self.emotion_processor(images=image, return_tensors="pt")
                inputs = {key: value.to(self.emotion_device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    logits = self.emotion_model(**inputs).logits[0]
                    probabilities = self.torch.softmax(logits, dim=-1)
                confidence, class_id = probabilities.max(dim=-1)
                confidence_value = float(confidence.detach().cpu())
                if confidence_value < 0.65:
                    continue
                label = self.emotion_id2label.get(int(class_id), str(int(class_id))).lower()
                expressions.append(
                    {
                        "label": label,
                        "confidence": confidence_value,
                        "track_id": containing_person.track_id,
                        "bbox": face_box,
                        "source": "face_expression_transformer",
                    }
                )
            except Exception as exc:
                LOGGER.warning("Emotion inference failed: %s", exc)
                self.emotion_model = None
                break
        return expressions

    def analyze(self, frame: np.ndarray) -> dict[str, Any]:
        height, width = frame.shape[:2]
        yolo_result = self._predict_ultralytics(self.detector, frame, 0.30)
        world_result = self._predict_ultralytics(self.world_detector, frame, 0.28)
        pose_result = self._predict_ultralytics(self.pose_detector, frame, 0.25)
        detections = self._extract_detections(yolo_result, "yolo", width, height)
        detections.extend(self._extract_detections(world_result, "yolo_world", width, height))
        detections = self._merge_detections(detections)
        actions, poses = self._pose_actions(pose_result, detections)
        hands = self._detect_hands(frame)
        expressions = self._classify_expression(frame, [item for item in detections if item.label == "person"])
        return {
            "detections": detections,
            "actions": actions,
            "poses": poses,
            "hands": hands,
            "expressions": expressions,
        }


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class TrackManager:
    def __init__(self, max_gap_sec: float = 3.0):
        self.max_gap_sec = max_gap_sec
        self.next_track_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        camera_shift_pixels: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> list[Detection]:
        width, height = frame_size
        shift_norm = (
            camera_shift_pixels[0] / max(width, 1),
            camera_shift_pixels[1] / max(height, 1),
        )
        active_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - float(track["last_timestamp"]) <= self.max_gap_sec
        ]
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id in active_ids:
                track = self.tracks[track_id]
                previous_box = track["last_bbox"]
                iou_cost = 1.0 - box_iou(previous_box, detection.bbox)
                distance_cost = center_distance(previous_box, detection.bbox, shift_norm)
                class_penalty = 0.0 if track["label"] == detection.label else 1.5
                cost = 0.7 * iou_cost + 1.4 * distance_cost + class_penalty
                if cost <= 1.85:
                    candidates.append((cost, detection_index, track_id))
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(candidates):
            if detection_index in used_detections or track_id in used_tracks:
                continue
            self._assign(track_id, detections[detection_index], timestamp, shift_norm)
            used_detections.add(detection_index)
            used_tracks.add(track_id)
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            detection.track_id = track_id
            center = box_center(detection.bbox)
            self.tracks[track_id] = {
                "track_id": track_id,
                "label": detection.label,
                "first_timestamp": timestamp,
                "last_timestamp": timestamp,
                "last_bbox": detection.bbox,
                "observations": [
                    {
                        "timestamp": timestamp,
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                        "source": detection.source,
                        "center": list(center),
                        "relative_center": list(center),
                    }
                ],
            }
        return detections

    def _assign(
        self,
        track_id: int,
        detection: Detection,
        timestamp: float,
        shift_norm: tuple[float, float],
    ) -> None:
        track = self.tracks[track_id]
        detection.track_id = track_id
        center = box_center(detection.bbox)
        relative_center = [center[0] - shift_norm[0], center[1] - shift_norm[1]]
        track["last_timestamp"] = timestamp
        track["last_bbox"] = detection.bbox
        track["observations"].append(
            {
                "timestamp": timestamp,
                "bbox": detection.bbox,
                "confidence": detection.confidence,
                "source": detection.source,
                "center": list(center),
                "relative_center": relative_center,
            }
        )

    def export(self) -> list[dict[str, Any]]:
        exported = []
        for track_id in sorted(self.tracks):
            track = dict(self.tracks[track_id])
            observations = track["observations"]
            centers = np.asarray([item["relative_center"] for item in observations], dtype=np.float64)
            if len(centers) >= 2:
                path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
                net_displacement = float(np.linalg.norm(centers[-1] - centers[0]))
            else:
                path_length = 0.0
                net_displacement = 0.0
            track["compensated_path_length"] = path_length
            track["net_displacement"] = net_displacement
            track["moving"] = path_length >= 0.14 and net_displacement >= 0.035
            track["observation_count"] = len(observations)
            exported.append(track)
        return exported


# ---------------------------------------------------------------------------
# Speech activity adapters
# ---------------------------------------------------------------------------


def load_speech_intervals(config: PipelineConfig) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    if config.subtitle_srt:
        try:
            import pysrt

            subtitles = pysrt.open(config.subtitle_srt, encoding="utf-8")
            for item in subtitles:
                start = item.start.ordinal / 1000.0
                end = item.end.ordinal / 1000.0
                intervals.append((start, end))
        except Exception as exc:
            LOGGER.warning("Unable to read SRT speech intervals: %s", exc)
    if config.speech_json:
        try:
            payload = json.loads(Path(config.speech_json).read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("segments", payload.get("items", []))
            for item in records:
                if not isinstance(item, dict):
                    continue
                start = safe_float(item.get("start", item.get("start_time", item.get("timestamp", 0.0))))
                end = safe_float(item.get("end", item.get("end_time", start + safe_float(item.get("duration", 0.0)))))
                if end > start:
                    intervals.append((start, end))
        except Exception as exc:
            LOGGER.warning("Unable to read speech JSON: %s", exc)
    intervals.sort()
    return intervals


def speech_activity_at(timestamp: float, intervals: Sequence[tuple[float, float]], radius: float = 0.25) -> float:
    for start, end in intervals:
        if start - radius <= timestamp <= end + radius:
            return 1.0
        if start > timestamp + radius:
            break
    return 0.0


# ---------------------------------------------------------------------------
# DMD boundary analysis
# ---------------------------------------------------------------------------


def dmd_predict(previous_descriptors: Sequence[np.ndarray], rank: int = 4) -> Optional[np.ndarray]:
    """One-step low-rank DMD prediction from previous descriptor observations."""
    if len(previous_descriptors) < 4:
        return None
    matrix = np.stack(previous_descriptors, axis=1).astype(np.float64)
    x = matrix[:, :-1]
    y = matrix[:, 1:]
    try:
        u, singular, vt = np.linalg.svd(x, full_matrices=False)
        usable = min(rank, int(np.sum(singular > 1e-8)))
        if usable <= 0:
            return None
        u_r = u[:, :usable]
        s_r = singular[:usable]
        v_r = vt[:usable, :].T
        a_tilde = u_r.T @ y @ v_r @ np.diag(1.0 / s_r)
        eigenvalues, w = np.linalg.eig(a_tilde)
        phi = y @ v_r @ np.diag(1.0 / s_r) @ w
        b = np.linalg.pinv(phi) @ matrix[:, -1]
        prediction = phi @ (eigenvalues * b)
        return np.real(prediction).astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def analyze_dmd_boundaries(
    video_path: Path,
    meta: VideoMeta,
    analysis_fps: float,
    max_frames: int = 0,
) -> list[dict[str, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen video for DMD analysis: {video_path}")
    stride = max(1, int(round(meta.source_fps / max(analysis_fps, 0.01))))
    records: list[dict[str, float]] = []
    history: deque[np.ndarray] = deque(maxlen=10)
    previous_gray: Optional[np.ndarray] = None
    previous_hist: Optional[np.ndarray] = None
    source_index = -1
    sampled = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_index += 1
        if source_index % stride != 0:
            continue
        sampled += 1
        if max_frames and sampled > max_frames:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA)
        descriptor = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255.0
        prediction = dmd_predict(list(history), rank=4)
        if prediction is None:
            residual = 0.0
        else:
            residual = float(np.linalg.norm(descriptor - prediction) / max(np.linalg.norm(descriptor), 1e-8))
        current_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        hist_change = histogram_novelty(previous_hist, current_hist)
        if previous_gray is None:
            pixel_change = 0.0
            camera_shift = 0.0
        else:
            pixel_change = float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
            camera_shift, _ = phase_shift(previous_gray, gray)
        records.append(
            {
                "frame_index": source_index,
                "timestamp": source_index / meta.source_fps,
                "dmd_residual": residual,
                "histogram_change": hist_change,
                "pixel_change": pixel_change,
                "camera_shift": camera_shift,
            }
        )
        history.append(descriptor)
        previous_gray = gray
        previous_hist = current_hist
    capture.release()
    if not records:
        return records
    dmd_norm = robust_normalize([record["dmd_residual"] for record in records])
    hist_norm = robust_normalize([record["histogram_change"] for record in records])
    pixel_norm = robust_normalize([record["pixel_change"] for record in records])
    shift_norm = robust_normalize([record["camera_shift"] for record in records])
    scores = 0.42 * dmd_norm + 0.25 * hist_norm + 0.18 * pixel_norm + 0.15 * shift_norm
    threshold = max(float(np.percentile(scores, 88.0)), float(np.median(scores) + 2.8 * mad(scores)))
    min_gap = 1.5
    last_boundary = -1e9
    for index, (record, score) in enumerate(zip(records, scores)):
        local_peak = (
            score >= (scores[index - 1] if index > 0 else -1.0)
            and score >= (scores[index + 1] if index + 1 < len(scores) else -1.0)
        )
        boundary = bool(score >= threshold and local_peak and record["timestamp"] - last_boundary >= min_gap)
        if boundary:
            last_boundary = record["timestamp"]
        record["boundary_score"] = float(score)
        record["boundary_threshold"] = threshold
        record["is_boundary"] = boundary
    return records


# ---------------------------------------------------------------------------
# Scene and event construction
# ---------------------------------------------------------------------------


def choose_scene_boundaries(frames: list[FrameRecord], config: PipelineConfig) -> list[int]:
    if len(frames) < 2:
        return [0]
    scores = np.asarray([frame.scene_score for frame in frames], dtype=np.float64)
    threshold = max(0.24, float(np.median(scores) + 2.4 * mad(scores)))
    candidate_indices = sorted(range(1, len(frames)), key=lambda index: scores[index], reverse=True)
    selected = [0]
    for index in candidate_indices:
        if scores[index] < threshold:
            break
        if any(abs(frames[index].timestamp - frames[existing].timestamp) < config.min_scene_gap for existing in selected):
            continue
        if all(abs(index - existing) >= config.min_scene_keyframes for existing in selected):
            selected.append(index)
    selected.sort()
    return selected


def merge_similar_short_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(scenes) <= 1:
        return scenes
    merged = [scenes[0]]
    for current in scenes[1:]:
        previous = merged[-1]
        previous_labels = set(previous.get("semantic_labels", []))
        current_labels = set(current.get("semantic_labels", []))
        union = previous_labels | current_labels
        similarity = len(previous_labels & current_labels) / len(union) if union else 0.0
        short = (previous["end_time"] - previous["start_time"] < 3.0) or (
            current["end_time"] - current["start_time"] < 3.0
        )
        if short and similarity >= 0.60:
            previous["end_time"] = current["end_time"]
            previous["end_frame_index"] = current["end_frame_index"]
            previous["keyframe_codes"].extend(current["keyframe_codes"])
            previous["semantic_labels"] = sorted(union)
        else:
            merged.append(current)
    for index, scene in enumerate(merged, start=1):
        scene["scene_id"] = f"scene_{index:04d}"
    return merged


def build_scenes(frames: list[FrameRecord], meta: VideoMeta, config: PipelineConfig) -> list[dict[str, Any]]:
    if not frames:
        return []
    boundaries = choose_scene_boundaries(frames, config)
    scenes: list[dict[str, Any]] = []
    for scene_index, start_index in enumerate(boundaries):
        end_index = boundaries[scene_index + 1] - 1 if scene_index + 1 < len(boundaries) else len(frames) - 1
        members = frames[start_index : end_index + 1]
        labels = sorted({label for frame in members for label in frame.semantic_labels})
        scenes.append(
            {
                "scene_id": f"scene_{scene_index + 1:04d}",
                "start_time": members[0].timestamp,
                "end_time": members[-1].timestamp if end_index < len(frames) - 1 else max(members[-1].timestamp, meta.duration_sec),
                "start_frame_index": members[0].frame_index,
                "end_frame_index": members[-1].frame_index,
                "keyframe_codes": [frame.keyframe_code for frame in members],
                "semantic_labels": labels,
            }
        )
    return merge_similar_short_scenes(scenes)


def aggregate_labeled_observations(
    observations: list[dict[str, Any]],
    key_fields: Sequence[str],
    max_gap: float,
    min_observations: int,
    event_type: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[tuple(item.get(field) for field in key_fields)].append(item)
    events: list[dict[str, Any]] = []
    event_counter = 0
    for key, items in grouped.items():
        items.sort(key=lambda item: item["timestamp"])
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for item in items:
            if current and item["timestamp"] - current[-1]["timestamp"] > max_gap:
                chunks.append(current)
                current = []
            current.append(item)
        if current:
            chunks.append(current)
        for chunk in chunks:
            if len(chunk) < min_observations:
                continue
            event_counter += 1
            sampled = chunk if len(chunk) <= 5 else [chunk[index] for index in np.linspace(0, len(chunk) - 1, 5).astype(int)]
            events.append(
                {
                    "event_id": f"{event_type.lower()}_{event_counter:05d}",
                    "type": event_type,
                    "label": chunk[0].get("label", event_type.lower()),
                    "track_id": chunk[0].get("track_id"),
                    "start_time": chunk[0]["timestamp"],
                    "end_time": chunk[-1]["timestamp"],
                    "observation_count": len(chunk),
                    "keyframe_codes": [item["keyframe_code"] for item in sampled],
                    "confidence": float(np.mean([safe_float(item.get("confidence"), 0.0) for item in chunk])),
                    "evidence": chunk,
                }
            )
    return events


def build_events(
    frames: list[FrameRecord],
    tracks: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat_observations: list[dict[str, Any]] = []
    action_observations: list[dict[str, Any]] = []
    expression_observations: list[dict[str, Any]] = []
    for frame in frames:
        for item in frame.objects:
            flat_observations.append(
                {
                    **item,
                    "type": "ObjectObservation",
                    "timestamp": frame.timestamp,
                    "keyframe_code": frame.keyframe_code,
                }
            )
        for item in frame.actions:
            record = {
                **item,
                "type": "ActionObservation",
                "timestamp": frame.timestamp,
                "keyframe_code": frame.keyframe_code,
            }
            flat_observations.append(record)
            action_observations.append(record)
        for item in frame.expressions:
            record = {
                **item,
                "type": "ExpressionObservation",
                "timestamp": frame.timestamp,
                "keyframe_code": frame.keyframe_code,
            }
            flat_observations.append(record)
            if str(item.get("label", "")).lower() not in {"neutral", "calm"}:
                expression_observations.append(record)

    object_events: list[dict[str, Any]] = []
    for track in tracks:
        if track.get("observation_count", 0) < 2:
            continue
        observations = track["observations"]
        selected = observations if len(observations) <= 5 else [
            observations[index] for index in np.linspace(0, len(observations) - 1, 5).astype(int)
        ]
        nearest_codes = []
        for observation in selected:
            nearest = min(frames, key=lambda frame: abs(frame.timestamp - observation["timestamp"]))
            nearest_codes.append(nearest.keyframe_code)
        object_events.append(
            {
                "event_id": f"object_{track['track_id']:05d}",
                "type": "Object",
                "label": track["label"],
                "track_id": track["track_id"],
                "start_time": track["first_timestamp"],
                "end_time": track["last_timestamp"],
                "observation_count": track["observation_count"],
                "moving": track["moving"],
                "keyframe_codes": nearest_codes,
                "confidence": float(np.mean([item["confidence"] for item in observations])),
                "evidence": observations,
            }
        )
    action_events = aggregate_labeled_observations(
        action_observations, ("track_id", "label"), 1.6, 2, "Action"
    )
    expression_events = aggregate_labeled_observations(
        expression_observations, ("track_id", "label"), 2.1, 2, "Expression"
    )
    all_events = object_events + action_events + expression_events

    hierarchy: list[dict[str, Any]] = []
    for scene in scenes:
        members = [
            event
            for event in all_events
            if event["end_time"] >= scene["start_time"] and event["start_time"] <= scene["end_time"]
        ]
        action_members = [event for event in members if event["type"] == "Action"]
        anchors = action_members or [
            {
                "event_id": f"{scene['scene_id']}_observation",
                "type": "Observation",
                "label": "observation",
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "keyframe_codes": scene["keyframe_codes"][:5],
            }
        ]
        scene_events = []
        for anchor in anchors:
            overlapping = [
                event
                for event in members
                if event["event_id"] != anchor["event_id"]
                and event["end_time"] >= anchor["start_time"]
                and event["start_time"] <= anchor["end_time"]
            ]
            scene_events.append(
                {
                    "anchor": anchor,
                    "actions": [event for event in overlapping if event["type"] == "Action"],
                    "objects": [event for event in overlapping if event["type"] == "Object"],
                    "expressions": [event for event in overlapping if event["type"] == "Expression"],
                }
            )
        hierarchy.append({**scene, "events": scene_events})
    return all_events, hierarchy


# ---------------------------------------------------------------------------
# Summary selection and metrics
# ---------------------------------------------------------------------------


def map_boundary_scores(frames: list[FrameRecord], dmd_records: list[dict[str, Any]]) -> None:
    if not frames or not dmd_records:
        return
    timestamps = np.asarray([record["timestamp"] for record in dmd_records], dtype=np.float64)
    for frame in frames:
        index = int(np.argmin(np.abs(timestamps - frame.timestamp)))
        frame.boundary_score = float(dmd_records[index]["boundary_score"])


def score_summary_frames(frames: list[FrameRecord], speech_intervals: Sequence[tuple[float, float]]) -> None:
    if not frames:
        return
    semantic_raw = []
    quality_raw = []
    for frame in frames:
        semantic = len(frame.objects) + 0.8 * len(frame.actions) + 1.2 * len(frame.expressions)
        semantic_raw.append(semantic)
        quality_raw.append(frame.sharpness / max(frame.sharpness_threshold, 1e-6))
    semantic_norm = robust_normalize(semantic_raw)
    quality_norm = robust_normalize(quality_raw)
    information_norm = robust_normalize([frame.information_gain for frame in frames])
    boundary_norm = robust_normalize([frame.boundary_score for frame in frames])
    for index, frame in enumerate(frames):
        frame.semantic_density = float(semantic_norm[index])
        frame.quality = float(quality_norm[index])
        frame.speech_activity = speech_activity_at(frame.timestamp, speech_intervals)
        frame.summary_score = float(
            0.28 * information_norm[index]
            + 0.22 * boundary_norm[index]
            + 0.22 * semantic_norm[index]
            + 0.13 * frame.speech_activity
            + 0.15 * quality_norm[index]
        )


def select_summary_frames(
    frames: list[FrameRecord],
    duration_sec: float,
    config: PipelineConfig,
) -> list[FrameRecord]:
    if not frames:
        return []
    budget = max(1, int(math.ceil(duration_sec * config.summary_ratio / config.summary_frame_duration)))
    budget = min(budget, len(frames))
    selected: list[FrameRecord] = []
    for frame in sorted(frames, key=lambda item: (item.summary_score, item.information_gain), reverse=True):
        if any(abs(frame.timestamp - existing.timestamp) < config.summary_min_gap for existing in selected):
            continue
        selected.append(frame)
        if len(selected) >= budget:
            break
    if len(selected) < budget:
        for frame in sorted(frames, key=lambda item: item.timestamp):
            if frame not in selected:
                selected.append(frame)
            if len(selected) >= budget:
                break
    return sorted(selected, key=lambda item: item.timestamp)


def summary_metrics(frames: list[FrameRecord], selected: list[FrameRecord]) -> dict[str, Any]:
    if not frames or not selected:
        return {
            "diversity_reward": 0.0,
            "representativeness_reward": 0.0,
            "selected_count": len(selected),
        }
    selected_desc = [np.asarray(frame.descriptor, dtype=np.float32) for frame in selected]
    pairwise = []
    for i in range(len(selected_desc)):
        for j in range(i + 1, len(selected_desc)):
            pairwise.append(cosine_distance(selected_desc[i], selected_desc[j]))
    diversity = float(np.mean(pairwise)) if pairwise else 0.0
    all_desc = [np.asarray(frame.descriptor, dtype=np.float32) for frame in frames]
    nearest = [min(cosine_distance(desc, selected_item) for selected_item in selected_desc) for desc in all_desc]
    representativeness = float(np.mean(np.exp(-np.asarray(nearest, dtype=np.float64))))
    return {
        "diversity_reward": diversity,
        "representativeness_reward": representativeness,
        "selected_count": len(selected),
        "candidate_keyframe_count": len(frames),
    }


# ---------------------------------------------------------------------------
# Visualization and export
# ---------------------------------------------------------------------------


def draw_semantics(frame: np.ndarray, semantic: dict[str, Any]) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    for detection in semantic.get("detections", []):
        x1, y1, x2, y2 = denormalized_box(detection.bbox, width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        text = f"{detection.label} {detection.confidence:.2f} T{detection.track_id or '-'}"
        cv2.putText(canvas, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    for action_index, action in enumerate(semantic.get("actions", [])):
        cv2.putText(
            canvas,
            str(action.get("label", "action")),
            (10, 25 + action_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return canvas


def write_technical_curves(path: Path, dmd_records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "timestamp",
        "dmd_residual",
        "histogram_change",
        "pixel_change",
        "camera_shift",
        "boundary_score",
        "boundary_threshold",
        "is_boundary",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in dmd_records:
            writer.writerow({field: record.get(field, "") for field in fields})


def export_worldmm_jsonl(path: Path, meta: VideoMeta, frames: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            object_labels = [item["label"] for item in frame.objects]
            action_labels = [item["label"] for item in frame.actions]
            expression_labels = [item["label"] for item in frame.expressions]
            textual_evidence = "; ".join(
                filter(
                    None,
                    [
                        f"objects: {', '.join(object_labels)}" if object_labels else "",
                        f"actions: {', '.join(action_labels)}" if action_labels else "",
                        f"expressions: {', '.join(expression_labels)}" if expression_labels else "",
                    ],
                )
            )
            record = {
                "video_id": meta.video_id,
                "frame_id": frame.keyframe_code,
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "image_path": frame.source_image,
                "labeled_image_path": frame.labeled_image,
                "text": textual_evidence,
                "objects": frame.objects,
                "actions": frame.actions,
                "expressions": frame.expressions,
                "selection_reason": frame.selection_reason,
                "information_gain": frame.information_gain,
                "boundary_score": frame.boundary_score,
                "summary_score": frame.summary_score,
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class KeyframePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.video_path = Path(config.video).expanduser().resolve()
        self.output_root = Path(config.output).expanduser().resolve()
        self.semantic_dir = self.output_root / "semantic"
        self.research_dir = self.output_root / "research"
        self.package_dir = self.output_root / "keyframe_package"
        self.original_dir = self.package_dir / "original"
        self.labeled_dir = self.package_dir / "labeled_tracks"
        self.rejected_dir = self.semantic_dir / "rejected_images"
        for path in [
            self.semantic_dir,
            self.research_dir,
            self.original_dir,
            self.labeled_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        if config.save_rejected_images:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.meta = probe_video(self.video_path, config.video_id)
        self.semantic_analyzer = SemanticAnalyzer(config)
        self.track_manager = TrackManager()
        self.rejected_frames: list[dict[str, Any]] = []
        self.view_transitions: list[dict[str, Any]] = []
        self.frames: list[FrameRecord] = []
        self.scan_count = 0
        self.redundant_count = 0

    def run(self) -> dict[str, Any]:
        start = time.perf_counter()
        LOGGER.info("Processing %s", self.video_path)
        self._scan_and_select()
        dmd_records = analyze_dmd_boundaries(
            self.video_path,
            self.meta,
            self.config.analysis_fps,
            max_frames=self.config.max_scan_frames,
        )
        map_boundary_scores(self.frames, dmd_records)
        speech_intervals = load_speech_intervals(self.config)
        score_summary_frames(self.frames, speech_intervals)
        summary = select_summary_frames(self.frames, self.meta.duration_sec, self.config)
        tracks = self.track_manager.export()
        scenes = build_scenes(self.frames, self.meta, self.config)
        events, hierarchy = build_events(self.frames, tracks, scenes)
        metrics = summary_metrics(self.frames, summary)
        elapsed = time.perf_counter() - start
        return self._export(dmd_records, summary, tracks, scenes, events, hierarchy, metrics, elapsed)

    def _scan_and_select(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        stride = max(1, int(round(self.meta.source_fps / max(self.config.sample_fps, 0.01))))
        previous_candidate_gray: Optional[np.ndarray] = None
        previous_selected_gray: Optional[np.ndarray] = None
        previous_selected_hist: Optional[np.ndarray] = None
        previous_selected_time = -1e9
        rolling_sharpness: deque[float] = deque(maxlen=120)
        source_index = -1
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            source_index += 1
            if source_index % stride != 0:
                continue
            self.scan_count += 1
            if self.config.max_scan_frames and self.scan_count > self.config.max_scan_frames:
                break
            timestamp = source_index / self.meta.source_fps
            frame = resize_keep_aspect(source_frame, self.config.width)
            gray = to_gray_small(frame)
            sharpness = sharpness_laplacian(frame)
            reference_sharpness = (
                float(np.percentile(np.asarray(rolling_sharpness), 70.0)) if rolling_sharpness else sharpness
            )
            rolling_sharpness.append(sharpness)
            if previous_candidate_gray is None:
                candidate_change = 0.0
                motion_coverage = 0.0
                camera_shift = 0.0
                shift_vector = (0.0, 0.0)
            else:
                difference = cv2.absdiff(gray, previous_candidate_gray)
                candidate_change = float(difference.mean())
                motion_coverage = float(np.mean(difference > 10))
                camera_shift, shift_vector = phase_shift(previous_candidate_gray, gray)
            unstable = (
                candidate_change >= self.config.unstable_change
                and camera_shift >= self.config.unstable_shift
                and motion_coverage >= self.config.unstable_motion_coverage
            )
            rapid = unstable or candidate_change >= self.config.rapid_change or camera_shift >= self.config.rapid_shift
            if unstable:
                sharpness_threshold = max(
                    self.config.min_sharpness,
                    reference_sharpness * self.config.unstable_sharpness_ratio,
                )
            elif rapid:
                sharpness_threshold = max(
                    self.config.min_sharpness,
                    reference_sharpness * self.config.rapid_sharpness_ratio,
                )
            else:
                sharpness_threshold = max(
                    self.config.min_sharpness,
                    reference_sharpness * self.config.stable_sharpness_ratio,
                )
            eligible = sharpness >= sharpness_threshold
            if not eligible:
                reason = "view_transition_motion_blur" if rapid or unstable else "low_information_blur"
                record = {
                    "frame_index": source_index,
                    "timestamp": timestamp,
                    "reason": reason,
                    "sharpness": sharpness,
                    "sharpness_threshold": sharpness_threshold,
                    "candidate_change": candidate_change,
                    "motion_coverage": motion_coverage,
                    "camera_shift": camera_shift,
                }
                if self.config.save_rejected_images:
                    image_path = self.rejected_dir / f"frame_{source_index:08d}_t{timestamp:010.3f}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    record["image"] = relative_to_output(image_path, self.output_root)
                self.rejected_frames.append(record)
                if reason == "view_transition_motion_blur":
                    self.view_transitions.append(record)
                previous_candidate_gray = gray
                continue

            histogram = hsv_histogram(frame)
            novelty = histogram_novelty(previous_selected_hist, histogram)
            information_gain = 0.72 * novelty + 0.28 * clamp(candidate_change / 15.0)
            time_gap = timestamp - previous_selected_time
            stable_view = not unstable and not rapid
            selection_reason = ""
            if not self.frames:
                selection_reason = "first_valid_frame"
            elif stable_view and time_gap >= self.config.max_time_gap:
                selection_reason = "maximum_time_gap"
            elif stable_view and time_gap >= self.config.min_time_gap and information_gain >= self.config.information_gain_threshold:
                selection_reason = "visual_information_gain"
            if not selection_reason:
                self.redundant_count += 1
                previous_candidate_gray = gray
                continue

            semantic = self.semantic_analyzer.analyze(frame)
            detections: list[Detection] = semantic["detections"]
            selected_shift, selected_shift_vector = phase_shift(previous_selected_gray, gray)
            detections = self.track_manager.update(
                detections,
                timestamp,
                selected_shift_vector,
                (frame.shape[1], frame.shape[0]),
            )
            semantic["detections"] = detections
            # Pose actions were computed before track assignment. Attach nearest person track now.
            people = [item for item in detections if item.label == "person"]
            for action in semantic["actions"]:
                if action.get("track_id") is None and len(people) == 1:
                    action["track_id"] = people[0].track_id
            keyframe_code = f"{len(self.frames) + 1:05d}_frame_{source_index:08d}_t{timestamp:010.3f}"
            original_path = self.original_dir / f"{keyframe_code}.jpg"
            labeled_path = self.labeled_dir / f"{keyframe_code}.jpg"
            cv2.imwrite(str(original_path), source_frame)
            cv2.imwrite(str(labeled_path), draw_semantics(frame, semantic))

            objects = [asdict(item) for item in detections]
            actions = semantic["actions"]
            expressions = semantic["expressions"]
            hands = semantic["hands"]
            labels = sorted(
                {item["label"] for item in objects}
                | {item["label"] for item in actions}
                | {item["label"] for item in expressions}
            )
            flow_score = selected_shift
            scene_score = 0.64 * novelty + 0.36 * clamp(flow_score / 7.0)
            record = FrameRecord(
                keyframe_code=keyframe_code,
                frame_index=source_index,
                timestamp=timestamp,
                source_image=relative_to_output(original_path, self.output_root),
                labeled_image=relative_to_output(labeled_path, self.output_root),
                selection_reason=selection_reason,
                sharpness=sharpness,
                sharpness_threshold=sharpness_threshold,
                candidate_change=candidate_change,
                motion_coverage=motion_coverage,
                camera_shift=camera_shift,
                visual_novelty=novelty,
                information_gain=information_gain,
                scene_score=scene_score,
                objects=objects,
                actions=actions,
                expressions=expressions,
                hands=hands,
                descriptor=frame_descriptor(frame).tolist(),
                semantic_labels=labels,
            )
            self.frames.append(record)
            previous_selected_gray = gray
            previous_selected_hist = histogram
            previous_selected_time = timestamp
            previous_candidate_gray = gray
            if len(self.frames) % 25 == 0:
                LOGGER.info(
                    "Scanned %d candidates, selected %d keyframes, rejected %d",
                    self.scan_count,
                    len(self.frames),
                    len(self.rejected_frames),
                )
        capture.release()

    def _export(
        self,
        dmd_records: list[dict[str, Any]],
        summary: list[FrameRecord],
        tracks: list[dict[str, Any]],
        scenes: list[dict[str, Any]],
        events: list[dict[str, Any]],
        hierarchy: list[dict[str, Any]],
        metrics: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        frame_payload = [asdict(frame) for frame in self.frames]
        summary_payload = [
            {
                "keyframe_code": frame.keyframe_code,
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "source_image": frame.source_image,
                "labeled_image": frame.labeled_image,
                "summary_score": frame.summary_score,
                "information_gain": frame.information_gain,
                "boundary_score": frame.boundary_score,
                "semantic_density": frame.semantic_density,
                "speech_activity": frame.speech_activity,
                "quality": frame.quality,
            }
            for frame in summary
        ]
        runtime = {
            "total_runtime_sec": elapsed,
            "processing_fps_source_frames": self.meta.frame_count / elapsed if elapsed > 0 else 0.0,
            "processing_fps_scanned_candidates": self.scan_count / elapsed if elapsed > 0 else 0.0,
        }
        manifest = {
            "schema_version": "worldmm-keyframe-memory-v1",
            "created_at_unix": time.time(),
            "video": asdict(self.meta),
            "config": asdict(self.config),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "semantic_components": self.semantic_analyzer.loaded_components,
            },
            "counts": {
                "scanned_candidate_count": self.scan_count,
                "memory_keyframe_count": len(self.frames),
                "research_summary_keyframe_count": len(summary),
                "rejected_frame_count": len(self.rejected_frames),
                "view_transition_count": len(self.view_transitions),
                "skipped_redundant_frame_count": self.redundant_count,
                "track_count": len(tracks),
                "scene_count": len(scenes),
                "event_count": len(events),
            },
            "compression": {
                "source_to_memory_ratio": self.meta.frame_count / max(len(self.frames), 1),
                "source_to_summary_ratio": self.meta.frame_count / max(len(summary), 1),
                "memory_reduction_percent": 100.0 * (1.0 - len(self.frames) / max(self.meta.frame_count, 1)),
                "summary_reduction_percent": 100.0 * (1.0 - len(summary) / max(self.meta.frame_count, 1)),
            },
            "runtime": runtime,
        }
        write_json(self.semantic_dir / "manifest.json", manifest)
        write_json(self.semantic_dir / "frames.json", frame_payload)
        write_json(self.semantic_dir / "rejected_frames.json", self.rejected_frames)
        write_json(self.semantic_dir / "view_transitions.json", self.view_transitions)
        write_json(self.semantic_dir / "tracks.json", tracks)
        write_json(self.semantic_dir / "events_flat.json", events)
        write_json(self.semantic_dir / "events.json", hierarchy)
        write_json(self.semantic_dir / "event_hierarchy.json", hierarchy)
        write_json(self.package_dir / "keyframes.json", frame_payload)
        write_json(self.research_dir / "summary_keyframes.json", summary_payload)
        write_json(
            self.research_dir / "research_metrics.json",
            {
                **metrics,
                "summary_ratio": self.config.summary_ratio,
                "summary_frame_duration": self.config.summary_frame_duration,
                "boundary_count": sum(bool(record.get("is_boundary")) for record in dmd_records),
            },
        )
        write_technical_curves(self.research_dir / "technical_curves.csv", dmd_records)
        export_worldmm_jsonl(self.package_dir / "worldmm_visual_records.jsonl", self.meta, self.frames)
        memory = {
            "schema_version": "worldmm-keyframe-memory-v1",
            "video": asdict(self.meta),
            "manifest": "semantic/manifest.json",
            "memory_keyframes": frame_payload,
            "research_summary_keyframes": summary_payload,
            "scenes": scenes,
            "events": hierarchy,
            "tracks": tracks,
            "worldmm_adapter": {
                "visual_records_jsonl": "keyframe_package/worldmm_visual_records.jsonl",
                "recommended_input": "research/summary_keyframes.json",
                "full_evidence_input": "semantic/frames.json",
            },
        }
        write_json(self.output_root / "memory.json", memory)
        result = {
            "status": "complete",
            "output": str(self.output_root),
            "memory": str(self.output_root / "memory.json"),
            "manifest": str(self.semantic_dir / "manifest.json"),
            "memory_keyframes": len(self.frames),
            "summary_keyframes": len(summary),
            "runtime_sec": elapsed,
        }
        write_json(self.output_root / "job.json", result)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file WorldMM-compatible keyframe, event, and DMD summary pipeline."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output task directory.")
    parser.add_argument("--video-id", default="", help="Optional stable WorldMM video ID.")
    parser.add_argument("--width", default=640, type=int, help="Processing width; <=0 keeps original.")
    parser.add_argument("--sample-fps", default=10.0, type=float, help="Candidate scan FPS.")
    parser.add_argument("--analysis-fps", default=5.0, type=float, help="DMD analysis FPS.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, 1, or cpu.")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Local YOLO detector path or model name.")
    parser.add_argument("--world-model", default="", help="Optional local YOLO-World path or model name.")
    parser.add_argument(
        "--world-classes",
        default="",
        help="Comma-separated classes or a text/JSON file for YOLO-World.",
    )
    parser.add_argument("--pose-model", default="yolo11n-pose.pt", help="Local YOLO pose path or model name.")
    parser.add_argument("--emotion-model", default="", help="Optional local Transformers expression model.")
    parser.add_argument("--enable-hands", action="store_true", help="Enable optional MediaPipe hand analysis.")
    parser.add_argument("--disable-semantics", action="store_true", help="Run only quality, selection, DMD, scenes, export.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Ultralytics/Transformers to download missing models.",
    )
    parser.add_argument("--speech-json", default="", help="Optional transcript/ASR segment JSON.")
    parser.add_argument("--subtitle-srt", default="", help="Optional SRT for speech-activity score.")
    parser.add_argument("--max-scan-frames", default=0, type=int, help="Limit sampled frames for smoke tests; 0 means all.")
    parser.add_argument("--save-rejected-images", action="store_true", help="Save rejected frame JPEGs.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    config = PipelineConfig(
        video=args.video,
        output=args.output,
        video_id=args.video_id,
        width=args.width,
        sample_fps=args.sample_fps,
        analysis_fps=args.analysis_fps,
        device=args.device,
        yolo_model=args.yolo_model,
        world_model=args.world_model,
        world_classes=parse_world_classes(args.world_classes),
        pose_model=args.pose_model,
        emotion_model=args.emotion_model,
        enable_hands=args.enable_hands,
        disable_semantics=args.disable_semantics,
        allow_model_download=args.allow_model_download,
        speech_json=args.speech_json,
        subtitle_srt=args.subtitle_srt,
        max_scan_frames=args.max_scan_frames,
        save_rejected_images=args.save_rejected_images,
    )
    try:
        result = KeyframePipeline(config).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        output = Path(args.output).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "job.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
