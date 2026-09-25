"""SVD-folded keyframe extraction with sparse YOLO semantic events.

Two SVD folds bracket a sparse semantic pass: the first identifies duplicate
frames and limits YOLO to representative frames, while the second folds each
generated event independently. Production runs require the semantic detector;
the visual-only fallback can be enabled explicitly for diagnostics.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import WorldMMKeyframe, WorldMMResult, WorldMMScene


DEFAULT_CONFIG: dict[str, Any] = {
    "sample_fps": 10.0,
    "resize": [160, 90],
    "window_seconds": 8.0,
    "stride_seconds": 2.0,
    "energy_threshold": 0.90,
    "fold_resize": [64, 36],
    "fold_similarity_threshold": 0.985,
    "fold_residual_floor": 0.18,
    "min_rank": 1,
    "max_rank": 64,
    "min_event_seconds": 2.0,
    "score_weights": {"reconstruction": 0.40, "subspace": 0.30, "spectrum": 0.20, "coverage": 0.10},
    "boundary_quantile": 0.78,
    "dedup_similarity": 0.985,
    "yolo_events": {
        "enabled": True,
        "required": True,
        "scan_fps": 1.0,
        "width": 1280,
        "batch_size": 4,
        "timeout_seconds": 7200,
    },
    # Candidate generation should only remove near-identical frames. Semantic
    # duplication is decided later from the temporally covered VLM evidence.
    "candidate_dedup_similarity": 0.998,
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_svd_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML when PyYAML is present, with a safe built-in fallback."""
    candidate = Path(path) if path else Path(__file__).resolve().parents[2] / "svd_config.yaml"
    if not candidate.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        return _merge(DEFAULT_CONFIG, payload if isinstance(payload, dict) else {})
    except Exception:
        # The checked-in config is intentionally simple, so this parser is
        # sufficient when an installation does not include PyYAML.
        values: dict[str, Any] = {}
        section: str | None = None
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            without_comment = raw_line.split("#", 1)[0].rstrip()
            if not without_comment.strip() or ":" not in without_comment:
                continue
            indent = len(without_comment) - len(without_comment.lstrip())
            key, raw = [part.strip() for part in without_comment.split(":", 1)]
            if indent and section:
                if not isinstance(values.get(section), dict):
                    values[section] = {}
                values[section][key] = _parse_simple_yaml_value(raw)
                continue
            if not raw:
                section = key
                values[key] = {}
                continue
            section = None
            values[key] = _parse_simple_yaml_value(raw)
        return _merge(DEFAULT_CONFIG, values)


def _parse_simple_yaml_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in {"true", "yes", "on"}:
        return True
    if raw.lower() in {"false", "no", "off"}:
        return False
    if raw.startswith("[") and raw.endswith("]"):
        return [_parse_simple_yaml_value(item) for item in raw[1:-1].split(",") if item.strip()]
    try:
        return float(raw) if any(char in raw for char in ".eE") else int(raw)
    except ValueError:
        return raw.strip("\"'")


def _normalise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    low, high = float(values.min()), float(values.max())
    if high - low > 1e-6:
        values = (values - low) / (high - low)
    return values


class SVDLowRankKeyframeExtractor:
    """Extract representative frames using local low-rank temporal structure."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path | None = None):
        self.config = _merge(load_svd_config(config_path), config or {})

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        width, height = [int(v) for v in self.config.get("resize", [160, 90])]
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fold_width, fold_height = [int(v) for v in self.config.get("fold_resize", [64, 36])]
        # Keep substantially more spatial detail than a tiny thumbnail while
        # retaining a compact feature matrix for long-video decomposition.
        compact = cv2.resize(gray, (fold_width, fold_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        colour = small.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
        return np.concatenate([compact.reshape(-1), colour])

    def sample_video(self, video_path: str | Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        sample_fps = max(0.1, float(self.config.get("sample_fps", 10.0)))
        sample_interval = 1.0 / sample_fps
        next_sample_time = 0.0
        last_timestamp = -1.0
        width, height = [int(v) for v in self.config.get("resize", [160, 90])]
        samples: list[tuple[float, int, np.ndarray]] = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if timestamp <= last_timestamp and fps > 0:
                timestamp = index / fps
            last_timestamp = timestamp
            if timestamp + 1e-6 >= next_sample_time:
                thumbnail = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                samples.append((timestamp, index, thumbnail))
                while next_sample_time <= timestamp + 1e-6:
                    next_sample_time += sample_interval
            index += 1
        cap.release()
        if not samples:
            raise RuntimeError("video contains no decodable frames")
        sampled_duration = float(samples[-1][0] - samples[0][0]) if len(samples) > 1 else 0.0
        actual_sample_fps = (len(samples) - 1) / sampled_duration if sampled_duration > 0 else 0.0
        return samples, {
            "fps": fps, "frame_count": frame_count or index, "duration_sec": duration or samples[-1][0],
            "sample_fps_target": sample_fps, "sample_fps_actual": actual_sample_fps,
        }

    def build_feature_matrix(self, samples) -> np.ndarray:
        return np.vstack([self.preprocess(frame) for _, _, frame in samples]).astype(np.float32)

    def analyze_fold_groups(self, samples) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Flatten uniformly resized frames, run SVD, and group low-residual repeats."""
        started = time.perf_counter()
        width, height = [int(v) for v in self.config.get("fold_resize", [64, 36])]
        vectors = []
        for _, _, frame in samples:
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            vectors.append(gray.reshape(-1).astype(np.float32) / 255.0)
        matrix = np.vstack(vectors).astype(np.float32)
        mean = matrix.mean(axis=0, keepdims=True)
        centered = matrix - mean

        decomposition = "exact_svd"
        decomposition_components = min(centered.shape) if centered.ndim == 2 else 0
        if centered.shape[0] and centered.shape[1]:
            total_energy = float(np.square(centered, dtype=np.float64).sum())
            if centered.shape[1] > max(1024, 4 * int(self.config.get("max_rank", 64))):
                u, singular, vt, decomposition_components = self.randomized_svd(
                    centered, int(self.config.get("max_rank", 64)),
                )
                decomposition = "randomized_svd"
            else:
                u, singular, vt = np.linalg.svd(centered, full_matrices=False)
            rank, energy = self.estimate_rank(singular, total_energy=total_energy)
            low_rank = (u[:, :rank] * singular[:rank]) @ vt[:rank, :]
        else:
            singular = np.empty(0, dtype=np.float32)
            rank, energy = 1, 1.0
            low_rank = centered

        denominator = np.maximum(np.sqrt(np.mean(np.square(centered), axis=1)), 0.05)
        residual = np.sqrt(np.mean(np.square(centered - low_rank), axis=1)) / denominator
        median = float(np.median(residual)) if len(residual) else 0.0
        mad = float(np.median(np.abs(residual - median))) if len(residual) else 0.0
        residual_limit = max(
            float(self.config.get("fold_residual_floor", 0.18)),
            median + 3.0 * 1.4826 * mad,
        )
        is_outlier = residual > residual_limit

        # Compare demeaned low-rank reconstructions so exposure changes do not
        # split otherwise repeated views into different fold groups.
        signatures = low_rank - low_rank.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(signatures, axis=1)
        signatures = np.divide(
            signatures,
            norms[:, None],
            out=np.zeros_like(signatures),
            where=norms[:, None] > 1e-8,
        )
        similarity_threshold = float(self.config.get("fold_similarity_threshold", 0.985))
        group_representatives = np.zeros((len(samples), signatures.shape[1]), dtype=np.float32)
        group_ids: list[str] = []
        assignments: list[dict[str, Any]] = []
        outlier_index = 0
        for index, ((timestamp, frame_index, _), signature) in enumerate(zip(samples, signatures)):
            if is_outlier[index]:
                outlier_index += 1
                group_id = None
                status = "outlier"
                similarity = None
            else:
                group_id = None
                similarity = None
                is_new_group = False
                if group_ids:
                    similarities = group_representatives[:len(group_ids)] @ signature
                    if norms[index] <= 1e-8:
                        similarities = np.where(
                            np.linalg.norm(group_representatives[:len(group_ids)], axis=1) <= 1e-8,
                            1.0,
                            similarities,
                        )
                    match = int(np.argmax(similarities))
                    similarity = float(similarities[match])
                    if similarity >= similarity_threshold:
                        group_id = group_ids[match]
                if group_id is None:
                    group_id = f"fold_{len(group_ids) + 1:05d}"
                    group_ids.append(group_id)
                    group_representatives[len(group_ids) - 1] = signature
                    is_new_group = True
                status = "representative" if is_new_group else "folded"
            assignments.append({
                "sample_index": index,
                "timestamp_sec": float(timestamp),
                "frame_index": int(frame_index),
                "group_id": group_id,
                "status": status,
                "reconstruction_residual": float(residual[index]),
                "similarity_to_group": similarity,
                "outlier_id": f"outlier_{outlier_index:05d}" if is_outlier[index] else None,
            })

        group_sizes: dict[str, int] = {}
        for item in assignments:
            if item["group_id"]:
                group_sizes[item["group_id"]] = group_sizes.get(item["group_id"], 0) + 1
        folded_count = sum(max(0, count - 1) for count in group_sizes.values())
        metrics = {
            "method": "svd_frame_folding_v1",
            "matrix_shape": [int(value) for value in matrix.shape],
            "frame_vector_dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "decomposition": decomposition,
            "decomposition_components": int(decomposition_components),
            "rank": int(rank),
            "energy_retained": float(energy),
            "energy_threshold": float(self.config.get("energy_threshold", 0.90)),
            "energy_threshold_reached": bool(energy >= float(self.config.get("energy_threshold", 0.90))),
            "similarity_threshold": similarity_threshold,
            "reconstruction_residual_limit": float(residual_limit),
            "fold_group_count": len(group_sizes),
            "folded_frame_count": int(folded_count),
            "outlier_count": int(np.count_nonzero(is_outlier)),
            "mean_reconstruction_residual": float(np.mean(residual)) if len(residual) else 0.0,
            "p95_reconstruction_residual": float(np.percentile(residual, 95)) if len(residual) else 0.0,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "group_sizes": group_sizes,
        }
        return assignments, metrics

    def estimate_rank(self, singular_values: np.ndarray, total_energy: float | None = None) -> tuple[int, float]:
        if singular_values.size == 0:
            return 1, 1.0
        energy = np.square(singular_values.astype(np.float64))
        denominator = max(float(total_energy if total_energy is not None else energy.sum()), 1e-12)
        ratio = np.cumsum(energy) / denominator
        threshold = float(self.config.get("energy_threshold", 0.90))
        rank = int(np.searchsorted(ratio, threshold) + 1)
        rank = max(int(self.config.get("min_rank", 1)), min(rank, int(self.config.get("max_rank", 16)), len(singular_values)))
        return rank, float(ratio[rank - 1])

    def randomized_svd(self, matrix: np.ndarray, max_rank: int, oversample: int = 16,
                       power_iterations: int = 2):
        """Compute a deterministic truncated SVD without allocating a full U basis."""
        rows, columns = matrix.shape
        target = min(rows, columns, max(1, int(max_rank)) + max(0, int(oversample)))
        rng = np.random.default_rng(2026)
        projection = rng.standard_normal((columns, target), dtype=np.float32)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            basis, _ = np.linalg.qr(matrix @ projection, mode="reduced")
            for _ in range(max(0, int(power_iterations))):
                basis, _ = np.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
            compressed = basis.T @ matrix
            small_u, singular, vt = np.linalg.svd(compressed, full_matrices=False)
            u = basis @ small_u
        if not (np.isfinite(u).all() and np.isfinite(singular).all() and np.isfinite(vt).all()):
            raise RuntimeError("randomized SVD produced non-finite components")
        return u, singular, vt, target

    def low_rank_decomposition(self, matrix: np.ndarray):
        if matrix.size == 0:
            return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), 1, 1.0
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        rank, energy = self.estimate_rank(singular)
        return u[:, :rank], singular, vt[:rank, :], rank, energy

    @staticmethod
    def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
        size = max(len(a), len(b))
        pa = np.pad(a, (0, size - len(a))) + 1e-12
        pb = np.pad(b, (0, size - len(b))) + 1e-12
        pa, pb = pa / pa.sum(), pb / pb.sum()
        mid = (pa + pb) / 2.0
        return float(0.5 * np.sum(pa * np.log(pa / mid)) + 0.5 * np.sum(pb * np.log(pb / mid)))

    def reconstruction_novelty(self, matrix: np.ndarray, u: np.ndarray, vt: np.ndarray) -> np.ndarray:
        if not len(matrix):
            return np.empty(0)
        reconstructed = (u @ (u.T @ matrix)) if u.size else np.zeros_like(matrix)
        return np.mean(np.square(matrix - reconstructed), axis=0)

    def subspace_change(self, previous_u: np.ndarray | None, current_u: np.ndarray) -> float:
        if previous_u is None or not previous_u.size or not current_u.size:
            return 0.0
        rank = min(previous_u.shape[1], current_u.shape[1])
        a, b = previous_u[:, :rank].astype(np.float64), current_u[:, :rank].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > 1e4:
            return 0.0
        # For orthonormal SVD bases, ||AAᵀ-BBᵀ||²_F = 2r-2||AᵀB||²_F.
        # This is numerically safer and avoids constructing a large D×D matrix.
        overlap = a.T @ b
        if not np.isfinite(overlap).all():
            return 0.0
        squared_distance = max(0.0, 2.0 * rank - 2.0 * float(np.square(overlap).sum()))
        return float(math.sqrt(squared_distance / max(1, a.shape[0])))

    def spectrum_change(self, previous_s: np.ndarray | None, current_s: np.ndarray) -> float:
        if previous_s is None or not len(current_s):
            return 0.0
        return self._js_divergence(np.square(previous_s), np.square(current_s))

    def detect_event_boundaries(self, scores: np.ndarray, timestamps: np.ndarray) -> list[int]:
        if len(scores) <= 1:
            return [0]
        finite = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        threshold = float(np.quantile(finite, float(self.config.get("boundary_quantile", 0.78))))
        min_gap = max(float(self.config.get("min_event_seconds", 2.0)), 0.1)
        candidates = [i for i in range(1, len(finite) - 1) if finite[i] >= threshold and finite[i] >= finite[i - 1] and finite[i] >= finite[i + 1]]
        selected: list[int] = []
        for index in sorted(candidates, key=lambda i: finite[i], reverse=True):
            if all(abs(float(timestamps[index] - timestamps[other])) >= min_gap for other in selected):
                selected.append(index)
        return [0] + sorted(selected)

    def select_event_keyframes(self, start: int, end: int, scores: np.ndarray, features: np.ndarray, max_count: int,
                               fold_group_ids: list[str | None] | None = None) -> list[int]:
        if end <= start:
            return [start]
        count = max(1, min(max_count, end - start))
        selected: list[int] = []
        selected_groups: set[str] = set()
        threshold = float(self.config.get("candidate_dedup_similarity", 0.998))
        # Cover the entire event instead of taking the globally highest SVD
        # scores, which can all cluster within a few hundred milliseconds.
        # eSVD groups still suppress exact repeats inside each temporal zone.
        edges = np.linspace(start, end, count + 1)
        for bucket in range(count):
            left = max(start, int(math.floor(edges[bucket])))
            right = min(end, max(left + 1, int(math.ceil(edges[bucket + 1]))))
            centre = (float(edges[bucket]) + float(edges[bucket + 1]) - 1.0) / 2.0
            width = max(1.0, float(edges[bucket + 1] - edges[bucket]))
            ranked = sorted(
                range(left, right),
                key=lambda index: (
                    float(scores[index]) - 0.15 * abs(float(index) - centre) / width,
                    -abs(float(index) - centre),
                ),
                reverse=True,
            )
            chosen = None
            for index in ranked:
                group_id = fold_group_ids[index] if fold_group_ids else None
                if group_id and group_id in selected_groups:
                    continue
                if all(
                    float(np.dot(features[index], features[other]) /
                          (np.linalg.norm(features[index]) * np.linalg.norm(features[other]) + 1e-8)) < threshold
                    for other in selected
                ):
                    chosen = index
                    break
            if chosen is None:
                continue
            selected.append(chosen)
            group_id = fold_group_ids[chosen] if fold_group_ids else None
            if group_id:
                selected_groups.add(group_id)
        return sorted(selected or [start])

    def deduplicate(self, indices: list[int], features: np.ndarray) -> list[int]:
        result: list[int] = []
        for index in indices:
            if not result:
                result.append(index)
                continue
            similarity = float(np.dot(features[index], features[result[-1]]) / (np.linalg.norm(features[index]) * np.linalg.norm(features[result[-1]]) + 1e-8))
            if similarity < float(self.config.get("candidate_dedup_similarity", 0.998)):
                result.append(index)
        return result

    def select_yolo_candidates(self, samples, fold_assignments, scores) -> list[int]:
        """Pick at most one first-pass SVD representative per configured time bucket."""
        scan_fps = max(0.1, float(self.config.get("yolo_events", {}).get("scan_fps", 1.0)))
        eligible = [
            index for index, item in enumerate(fold_assignments)
            if item.get("status") in {"representative", "outlier"}
        ]
        buckets: dict[int, int] = {}
        for index in eligible:
            timestamp = float(samples[index][0])
            bucket = int(math.floor(timestamp * scan_fps + 1e-8))
            previous = buckets.get(bucket)
            if previous is None or float(scores[index]) > float(scores[previous]):
                buckets[bucket] = index
        return sorted(buckets.values(), key=lambda index: float(samples[index][0]))

    def _run_yolo_scan(self, video_path: str | Path, candidates: list[int], samples, output: Path) -> dict[str, Any]:
        """Run the sparse detector in the configured video-tool Python environment."""
        if not candidates:
            raise RuntimeError("the first SVD fold selected no YOLO candidates")
        video_tools = Path(__file__).resolve().parents[2] / "tools" / "video_keyframe"
        worker = Path(__file__).with_name("yolo_event_worker.py")
        candidates_path = output / "yolo_scan_candidates.json"
        result_path = output / "yolo_scan.json"
        log_path = output / "yolo_scan.log"
        request = [{
            "sample_index": int(index), "frame_index": int(samples[index][1]),
            "timestamp_sec": float(samples[index][0]),
        } for index in candidates]
        candidates_path.write_text(json.dumps(request), encoding="utf-8")
        model = os.getenv("SENTRIX_VIDEO_YOLO_MODEL", str(video_tools / "models" / "keyframe" / "yolo11n.pt"))
        device = os.getenv("SENTRIX_VIDEO_DEVICE", "cpu")
        python = os.getenv("SENTRIX_VIDEO_PYTHON", sys.executable)
        config = self.config.get("yolo_events", {})
        command = [
            python, str(worker), "--video", str(Path(video_path).resolve()),
            "--candidates", str(candidates_path), "--output", str(result_path),
            "--model", model, "--device", device,
            "--width", str(max(64, int(config.get("width", 640)))),
            "--batch-size", str(max(1, int(config.get("batch_size", 16)))),
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, check=False,
                timeout=max(1, int(config.get("timeout_seconds", 7200))),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            log_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            raise RuntimeError(f"could not complete sparse YOLO scan: {type(error).__name__}: {error}") from error
        log_path.write_text(
            process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""), encoding="utf-8",
        )
        if process.returncode != 0 or not result_path.is_file():
            detail = (process.stderr or process.stdout or "worker produced no result").strip()[-1600:]
            raise RuntimeError(f"sparse YOLO worker failed ({process.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if len(payload.get("samples") or []) != len(candidates):
            raise RuntimeError("sparse YOLO worker returned an incomplete scan")
        return payload

    @staticmethod
    def _observation_signature(record: dict[str, Any], person_only: bool = False) -> tuple[dict[str, float], np.ndarray]:
        """Represent detected subjects by confidence, area and coarse layout."""
        labels: dict[str, float] = {}
        layout = np.zeros(9, dtype=np.float64)
        for detection in record.get("detections") or []:
            label = str(detection.get("label") or "").strip().lower()
            if not label or (person_only and label != "person"):
                continue
            bbox = detection.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [min(1.0, max(0.0, float(value))) for value in bbox]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            confidence = min(1.0, max(0.0, float(detection.get("confidence") or 0.0)))
            weight = confidence * math.sqrt(max(area, 1e-8))
            labels[label] = labels.get(label, 0.0) + weight
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cell = min(2, int(cy * 3)) * 3 + min(2, int(cx * 3))
            layout[cell] += weight
        return labels, layout

    @staticmethod
    def _signature_distance(left: tuple[dict[str, float], np.ndarray],
                            right: tuple[dict[str, float], np.ndarray]) -> float:
        left_labels, left_layout = left
        right_labels, right_layout = right
        keys = set(left_labels) | set(right_labels)
        denominator = sum(max(left_labels.get(key, 0.0), right_labels.get(key, 0.0)) for key in keys)
        overlap = sum(min(left_labels.get(key, 0.0), right_labels.get(key, 0.0)) for key in keys)
        label_distance = 0.0 if denominator <= 1e-9 else 1.0 - overlap / denominator
        layout_denominator = float(np.maximum(left_layout, right_layout).sum())
        layout_overlap = float(np.minimum(left_layout, right_layout).sum())
        layout_distance = 0.0 if layout_denominator <= 1e-9 else 1.0 - layout_overlap / layout_denominator
        return float((label_distance + layout_distance) / 2.0)

    @staticmethod
    def _standardize_block(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        median = np.median(values, axis=0, keepdims=True)
        mad = np.median(np.abs(values - median), axis=0, keepdims=True) * 1.4826
        std = np.std(values, axis=0, keepdims=True)
        scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
        normalized = (values - median) / scale
        active = np.std(normalized, axis=0) > 1e-8
        return normalized[:, active] if np.any(active) else np.zeros((len(values), 0), dtype=np.float64)

    def _compress_state_block(self, values: np.ndarray) -> np.ndarray:
        normalized = self._standardize_block(values)
        if not normalized.shape[1] or len(normalized) <= 1:
            return normalized
        u, singular, _vt = np.linalg.svd(normalized, full_matrices=False)
        energy = np.square(singular)
        ratio = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
        rank = max(1, int(np.searchsorted(ratio, float(self.config.get("energy_threshold", 0.90))) + 1))
        rank = min(rank, len(singular))
        return u[:, :rank] * singular[:rank]

    @staticmethod
    def _segment_sse(values: np.ndarray, start: int, end: int) -> float:
        segment = values[start:end]
        if not len(segment) or not segment.shape[1]:
            return 0.0
        centered = segment - segment.mean(axis=0, keepdims=True)
        return float(np.square(centered).sum())

    @staticmethod
    def _best_split_gain(values: np.ndarray) -> tuple[float, int | None, float]:
        count = len(values)
        if count < 4 or not values.shape[1]:
            return 0.0, None, 0.0
        cumulative = np.cumsum(values, axis=0)
        cumulative_sq = np.cumsum(np.square(values), axis=0)

        def sse(left: int, right: int) -> float:
            size = right - left
            total = cumulative[right - 1] - (cumulative[left - 1] if left else 0.0)
            total_sq = cumulative_sq[right - 1] - (cumulative_sq[left - 1] if left else 0.0)
            return max(0.0, float(total_sq.sum() - np.square(total).sum() / max(1, size)))

        unsplit = sse(0, count)
        best_gain, best_split = 0.0, None
        for split in range(2, count - 1):
            gain = unsplit - sse(0, split) - sse(split, count)
            if gain > best_gain:
                best_gain, best_split = gain, split
        return best_gain, best_split, unsplit

    def _adaptive_bic_splits(self, values: np.ndarray) -> list[int]:
        """Choose sequential state changes by BIC, with no target count or duration."""
        values = self._compress_state_block(values)
        if len(values) < 4 or not values.shape[1]:
            return []
        splits: list[int] = []

        def recurse(start: int, end: int) -> None:
            if end - start < 4:
                return
            segment = values[start:end]
            count, dimensions = segment.shape
            best_gain, relative_split, unsplit_sse = self._best_split_gain(segment)
            if relative_split is None:
                return
            split_sse = max(0.0, unsplit_sse - best_gain)
            unsplit = (
                count * dimensions * math.log(max(unsplit_sse / max(1, count * dimensions), 1e-8))
                + (dimensions + 1) * math.log(max(2, count))
            )
            split_cost = (
                count * dimensions * math.log(max(split_sse / max(1, count * dimensions), 1e-8))
                + (2 * (dimensions + 1) + 1) * math.log(max(2, count))
            )
            if split_cost >= unsplit:
                return
            # A deterministic permutation test rejects apparent change points
            # that are equally likely after destroying temporal order. This
            # makes the number of states data-driven instead of configured.
            rng = np.random.default_rng(2026 + start * 131 + count * 17 + dimensions)
            null_gains = [
                self._best_split_gain(segment[rng.permutation(count)])[0]
                for _ in range(128)
            ]
            if best_gain <= float(np.quantile(null_gains, 0.99)):
                return
            best_split = start + relative_split
            splits.append(best_split)
            recurse(start, best_split)
            recurse(best_split, end)

        recurse(0, len(values))
        return sorted(set(splits))

    def _adaptive_ensemble_splits(self, blocks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Find changes supported by one decisive or two independent modalities."""
        prepared = {
            name: self._compress_state_block(values)
            for name, values in blocks.items()
        }
        prepared = {name: values for name, values in prepared.items() if values.shape[1]}
        if not prepared:
            return []
        count = len(next(iter(prepared.values())))
        changes: list[dict[str, Any]] = []

        def proposal(name: str, values: np.ndarray, start: int, end: int):
            segment = values[start:end]
            observations, dimensions = segment.shape
            gain, relative_split, unsplit_sse = self._best_split_gain(segment)
            if relative_split is None:
                return None
            split_sse = max(0.0, unsplit_sse - gain)
            unsplit_bic = (
                observations * dimensions * math.log(max(unsplit_sse / max(1, observations * dimensions), 1e-8))
                + (dimensions + 1) * math.log(max(2, observations))
            )
            split_bic = (
                observations * dimensions * math.log(max(split_sse / max(1, observations * dimensions), 1e-8))
                + (2 * (dimensions + 1) + 1) * math.log(max(2, observations))
            )
            if split_bic >= unsplit_bic:
                return None
            rng = np.random.default_rng(
                2026 + start * 131 + observations * 17 + dimensions * 7 + sum(ord(char) for char in name)
            )
            null_gains = np.asarray([
                self._best_split_gain(segment[rng.permutation(observations)])[0]
                for _ in range(256)
            ], dtype=np.float64)
            confidence = float((1 + np.count_nonzero(null_gains < gain)) / (len(null_gains) + 1))
            return {
                "modality": name, "position": start + int(relative_split),
                "confidence": confidence, "gain": float(gain),
            }

        def recurse(start: int, end: int) -> None:
            if end - start < 4:
                return
            # Different sensors react at slightly different points during a
            # gradual transition (for example, visual appearance before the
            # detector has a stable box).  Scale their alignment window with
            # statistical sequence resolution instead of a fixed time span.
            alignment_radius = max(1, int(math.ceil(math.log2(max(2, end - start)))))
            proposals = [
                item for name, values in prepared.items()
                if (item := proposal(name, values, start, end)) is not None
            ]
            decisive = [item for item in proposals if item["confidence"] >= 0.99]
            corroborated = []
            candidates = [item for item in proposals if item["confidence"] >= 0.90]
            for item in candidates:
                supporters = [
                    other for other in candidates
                    if other["modality"] != item["modality"]
                    and abs(int(other["position"]) - int(item["position"])) <= alignment_radius
                ]
                if supporters:
                    corroborated.append((item, *supporters))
            if decisive:
                anchor = max(decisive, key=lambda item: item["confidence"])
                nearby = [
                    item for item in candidates
                    if abs(int(item["position"]) - int(anchor["position"])) <= alignment_radius
                ]
            elif corroborated:
                nearby = list(max(corroborated, key=lambda group: sum(item["confidence"] for item in group)))
                anchor = max(nearby, key=lambda item: item["confidence"])
            else:
                return
            position = int(round(np.median([int(item["position"]) for item in nearby])))
            if position <= start or position >= end:
                return
            changes.append({
                "position": position,
                "reason": str(anchor["modality"]) + "_change",
                "confidence": round(float(max(item["confidence"] for item in nearby)), 6),
                "person_change_significant": any(
                    item["modality"] == "person" and item["confidence"] >= 0.90 for item in nearby
                ),
                "semantic_change_significant": any(
                    item["modality"] in {"semantic", "person"} and item["confidence"] >= 0.90
                    for item in nearby
                ),
                "multimodal_change_significant": len({
                    item["modality"] for item in nearby if item["confidence"] >= 0.90
                }) >= 2,
                "modalities": {
                    item["modality"]: round(float(item["confidence"]), 6) for item in nearby
                },
            })
            recurse(start, position)
            recurse(position, end)

        recurse(0, count)
        return sorted(changes, key=lambda item: int(item["position"]))

    def build_yolo_event_spans(self, observations, timestamps, features, scores) -> list[dict[str, Any]]:
        """Build content-adaptive semantic states from detector, visual and eSVD changes."""
        if not observations:
            return []
        ordered = sorted(observations, key=lambda item: int(item["sample_index"]))
        semantic_signatures = [self._observation_signature(item) for item in ordered]
        person_signatures = [self._observation_signature(item, person_only=True) for item in ordered]
        semantic_changes, person_changes, visual_changes, esvd_changes = [], [], [], []
        for position in range(1, len(ordered)):
            previous_index = max(0, min(len(features) - 1, int(ordered[position - 1]["sample_index"])))
            current_index = max(0, min(len(features) - 1, int(ordered[position]["sample_index"])))
            semantic_changes.append(self._signature_distance(
                semantic_signatures[position - 1], semantic_signatures[position],
            ))
            person_changes.append(self._signature_distance(
                person_signatures[position - 1], person_signatures[position],
            ))
            visual_changes.append(float(1.0 - np.clip(
                np.dot(features[previous_index], features[current_index]) /
                (np.linalg.norm(features[previous_index]) * np.linalg.norm(features[current_index]) + 1e-8),
                -1.0, 1.0,
            )))
            esvd_changes.append(abs(float(scores[current_index]) - float(scores[previous_index])))

        vocabulary = sorted({key for labels, _layout in semantic_signatures for key in labels})
        semantic_matrix = np.asarray([
            [labels.get(key, 0.0) for key in vocabulary] + layout.tolist()
            for labels, layout in semantic_signatures
        ], dtype=np.float64)
        person_matrix = np.asarray([
            [labels.get("person", 0.0), sum(
                1 for detection in record.get("detections") or []
                if str(detection.get("label") or "").lower() == "person"
            )] + layout.tolist()
            for record, (labels, layout) in zip(ordered, person_signatures)
        ], dtype=np.float64)
        observation_indices = [max(0, min(len(features) - 1, int(item["sample_index"]))) for item in ordered]
        visual_matrix = features[observation_indices]
        esvd_matrix = np.asarray([[float(scores[index])] for index in observation_indices], dtype=np.float64)
        adaptive_changes = self._adaptive_ensemble_splits({
            "semantic": semantic_matrix, "person": person_matrix,
            "visual": visual_matrix, "esvd": esvd_matrix,
        })

        boundaries = [0]
        boundary_details: dict[int, dict[str, Any]] = {}
        for adaptive_change in adaptive_changes:
            record_position = int(adaptive_change["position"])
            change_index = record_position - 1
            sample_index = max(0, min(len(timestamps) - 1, int(ordered[record_position]["sample_index"])))
            if sample_index <= boundaries[-1]:
                continue
            person_split = bool(adaptive_change.get("person_change_significant"))
            components = {
                "semantic": semantic_changes[change_index], "person": person_changes[change_index],
                "visual": visual_changes[change_index], "esvd": esvd_changes[change_index],
            }
            reason = "person_composition_change" if person_split else str(adaptive_change.get("reason") or "adaptive_change")
            boundaries.append(sample_index)
            boundary_details[sample_index] = {
                "boundary_reason": reason,
                "person_change_significant": bool(person_split),
                "semantic_change_significant": bool(
                    adaptive_change.get("semantic_change_significant")
                ),
                "multimodal_change_significant": bool(
                    adaptive_change.get("multimodal_change_significant")
                ),
                "adaptive_change_components": {key: round(float(value), 6) for key, value in components.items()},
                "adaptive_change_score": adaptive_change.get("confidence"),
                "adaptive_change_modalities": adaptive_change.get("modalities") or {},
            }
        boundaries.append(len(timestamps))

        spans = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            member_observations = [
                row for row in ordered if start <= int(row["sample_index"]) < end
            ]
            counts: dict[str, int] = {}
            confidence: dict[str, list[float]] = {}
            for row in member_observations:
                for detection in row.get("detections") or []:
                    label = str(detection.get("label") or "").strip()
                    if not label:
                        continue
                    counts[label] = counts.get(label, 0) + 1
                    confidence.setdefault(label, []).append(float(detection.get("confidence") or 0.0))
            # Keep labels seen in at least 15% of the event's scan observations;
            # if all detections are sparse, retain the most frequent one.
            minimum_hits = max(1, math.ceil(len(member_observations) * 0.15))
            labels = [label for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                      if count >= minimum_hits][:12]
            end_index = min(end - 1, len(timestamps) - 1)
            person_rows = []
            for row in member_observations:
                people = [
                    item for item in row.get("detections") or []
                    if str(item.get("label") or "").lower() == "person"
                ]
                person_rows.append((len(people), sum(
                    max(0.0, float((item.get("bbox") or [0, 0, 0, 0])[2]) - float((item.get("bbox") or [0, 0, 0, 0])[0]))
                    * max(0.0, float((item.get("bbox") or [0, 0, 0, 0])[3]) - float((item.get("bbox") or [0, 0, 0, 0])[1]))
                    for item in people if len(item.get("bbox") or []) == 4
                ), int(row["sample_index"])))
            person_peak = max(person_rows, default=(0, 0.0, int(start)))
            subject_rows = []
            for row in member_observations:
                signature, _layout = self._observation_signature(row)
                subject_rows.append((sum(signature.values()), int(row["sample_index"])))
            subject_peak = max(subject_rows, default=(0.0, int(start)))
            boundary_detail = boundary_details.get(start, {
                "boundary_reason": "video_start", "person_change_significant": False,
                "semantic_change_significant": False,
                "multimodal_change_significant": False,
                "adaptive_change_components": {}, "adaptive_change_score": 0.0,
            })
            spans.append({
                "start_index": int(start), "end_index": int(end),
                "start_sec": float(timestamps[start]), "end_sec": float(timestamps[end_index]),
                "semantic_labels": ["yolo_event", *labels] if labels else ["yolo_event", "no_objects_detected"],
                "yolo_labels": labels,
                "mean_label_confidence": {
                    label: round(float(np.mean(confidence[label])), 4) for label in labels if confidence.get(label)
                },
                "yolo_observation_count": len(member_observations),
                "max_person_count": int(person_peak[0]),
                "person_count_peak_index": int(person_peak[2]),
                "subject_coverage_peak_index": int(subject_peak[1]),
                **boundary_detail,
            })
        for index, span in enumerate(spans, 1):
            span["event_id"] = f"yolo_event_{index:05d}"
        return spans

    @staticmethod
    def _continuous_camera_transition(left: np.ndarray, right: np.ndarray) -> bool:
        """Recognize one camera take across a detector-only event boundary."""
        first = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        second = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(first, maxCorners=240, qualityLevel=0.01, minDistance=4)
        if points is None or len(points) < 12:
            return False
        tracked, status, errors = cv2.calcOpticalFlowPyrLK(
            first, second, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if tracked is None or status is None:
            return False
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < 12:
            return False
        source = points.reshape(-1, 2)[valid]
        target = tracked.reshape(-1, 2)[valid]
        transform, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if transform is None or inliers is None:
            return False
        inlier_ratio = float(inliers.reshape(-1).mean())
        median_error = float(np.median(np.asarray(errors).reshape(-1)[valid])) if errors is not None else 999.0
        return inlier_ratio >= 0.45 and median_error <= 25.0

    def merge_continuous_camera_spans(self, spans, samples) -> list[dict[str, Any]]:
        """Merge camera-continuous false boundaries while preserving subject changes."""
        merged: list[dict[str, Any]] = []
        for source in spans:
            item = dict(source)
            if (
                merged
                and not item.get("person_change_significant")
                and not item.get("semantic_change_significant")
            ):
                previous = merged[-1]
                left_index = max(0, min(len(samples) - 1, int(previous["end_index"]) - 1))
                right_index = max(0, min(len(samples) - 1, int(item["start_index"])))
                if self._continuous_camera_transition(samples[left_index][2], samples[right_index][2]):
                    previous["end_index"] = item["end_index"]
                    previous["end_sec"] = item["end_sec"]
                    previous["semantic_labels"] = list(dict.fromkeys(
                        list(previous.get("semantic_labels") or []) + list(item.get("semantic_labels") or [])
                    ))
                    previous["yolo_labels"] = list(dict.fromkeys(
                        list(previous.get("yolo_labels") or []) + list(item.get("yolo_labels") or [])
                    ))
                    previous["yolo_observation_count"] = int(previous.get("yolo_observation_count") or 0) + int(item.get("yolo_observation_count") or 0)
                    previous["camera_motion_merged"] = True
                    if int(item.get("max_person_count") or 0) > int(previous.get("max_person_count") or 0):
                        previous["max_person_count"] = item.get("max_person_count")
                        previous["person_count_peak_index"] = item.get("person_count_peak_index")
                    continue
            merged.append(item)
        for index, span in enumerate(merged, 1):
            span["event_id"] = f"yolo_event_{index:05d}"
        return merged

    def select_event_representative(self, start: int, end: int, scores: np.ndarray,
                                    features: np.ndarray, samples, event_span) -> int:
        """Choose a stable, clear and information-complete medoid for one adaptive state."""
        indices = np.arange(start, end, dtype=np.int64)
        if len(indices) <= 1:
            return int(indices[0]) if len(indices) else int(start)
        local = features[indices].astype(np.float64)
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        unit = np.divide(local, norms, out=np.zeros_like(local), where=norms > 1e-8)
        centrality = np.mean(unit @ unit.T, axis=1)
        sharpness = np.asarray([
            float(cv2.Laplacian(cv2.cvtColor(samples[int(index)][2], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            for index in indices
        ], dtype=np.float64)
        novelty = np.asarray([float(scores[int(index)]) for index in indices], dtype=np.float64)
        structure = np.asarray([
            self._structural_subject_coverage(samples[int(index)][2]) for index in indices
        ], dtype=np.float64)

        def percentile(values: np.ndarray) -> np.ndarray:
            if len(values) <= 1 or float(np.max(values) - np.min(values)) <= 1e-12:
                return np.ones(len(values), dtype=np.float64) * 0.5
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(values), dtype=np.float64)
            ranks[order] = np.linspace(0.0, 1.0, len(values))
            return ranks

        metrics = [percentile(centrality), percentile(sharpness), percentile(novelty)]
        if int(event_span.get("max_person_count") or 0) > 0:
            peaks = [
                int(event_span.get("person_count_peak_index") or start),
                int(event_span.get("subject_coverage_peak_index") or start),
            ]
            scale = max(1.0, float(end - start))
            coverage = np.asarray([
                max(math.exp(-abs(float(index - peak)) / scale) for peak in peaks)
                for index in indices
            ], dtype=np.float64)
            metrics.append(percentile(coverage))
        else:
            metrics.append(percentile(structure))
        aggregate = np.mean(np.vstack(metrics), axis=0)
        return int(indices[int(np.argmax(aggregate))])

    @staticmethod
    def _structural_subject_coverage(frame: np.ndarray) -> float:
        """Estimate how much of the frame is occupied by one coherent structure."""
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        connected = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(connected)
        if count <= 1:
            return 0.0
        return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(connected.size)

    def _visual_event_spans(self, boundaries, timestamps) -> list[dict[str, Any]]:
        spans = []
        for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
            if end <= start:
                continue
            spans.append({
                "event_id": f"svd_visual_event_{index:05d}",
                "start_index": int(start), "end_index": int(end),
                "start_sec": float(timestamps[start]),
                "end_sec": float(timestamps[min(end - 1, len(timestamps) - 1)]),
                "semantic_labels": ["svd_lowrank"], "yolo_labels": [],
                "yolo_observation_count": 0,
            })
        return spans

    def fold_within_events(self, samples, event_spans) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Second SVD fold, scoped to semantic event boundaries from the first pass."""
        assignments: list[dict[str, Any] | None] = [None] * len(samples)
        event_metrics = []
        group_sizes: dict[str, int] = {}
        for event_number, span in enumerate(event_spans, 1):
            start, end = int(span["start_index"]), int(span["end_index"])
            local, metrics = self.analyze_fold_groups(samples[start:end])
            event_metrics.append({"event_id": span["event_id"], **metrics})
            for item in local:
                global_index = start + int(item["sample_index"])
                row = dict(item)
                row["sample_index"] = global_index
                row["event_id"] = span["event_id"]
                row["first_pass_group_id"] = None
                if row.get("group_id"):
                    local_group = row["group_id"]
                    row["group_id"] = f"event_{event_number:05d}:{local_group}"
                    row["event_fold_group_id"] = local_group
                    group_sizes[row["group_id"]] = group_sizes.get(row["group_id"], 0) + 1
                else:
                    row["event_fold_group_id"] = None
                assignments[global_index] = row
        # Defensive fill for malformed or empty event input, preserving a valid
        # contract rather than silently dropping any sampled frame.
        for index, item in enumerate(assignments):
            if item is None:
                assignments[index] = {
                    "sample_index": index, "timestamp_sec": float(samples[index][0]),
                    "frame_index": int(samples[index][1]), "group_id": None,
                    "status": "unassigned", "reconstruction_residual": None,
                    "similarity_to_group": None, "outlier_id": None,
                    "event_id": None, "event_fold_group_id": None,
                }
        concrete = [item for item in assignments if item is not None]
        folded_count = sum(max(0, count - 1) for count in group_sizes.values())
        metrics = {
            "method": "svd_frame_folding_within_yolo_events_v1",
            "fold_group_count": len(group_sizes), "folded_frame_count": int(folded_count),
            "outlier_count": sum(1 for item in concrete if item["status"] == "outlier"),
            "runtime_seconds": round(sum(float(item.get("runtime_seconds") or 0.0) for item in event_metrics), 4),
            "event_count": len(event_spans), "event_metrics": event_metrics,
            "group_sizes": group_sizes,
        }
        return concrete, metrics

    def extract(self, video_path: str | Path, video_id: str, output_dir: str | Path) -> WorldMMResult:
        started = time.perf_counter()
        output = Path(output_dir).resolve()
        frames_dir = output / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        samples, video = self.sample_video(video_path)
        features = _normalise(self.build_feature_matrix(samples))
        fold_assignments, fold_metrics = self.analyze_fold_groups(samples)
        timestamps = np.asarray([item[0] for item in samples], dtype=np.float64)
        scores = np.zeros(len(samples), dtype=np.float64)
        reconstruction = np.zeros(len(samples), dtype=np.float64)
        subspace = np.zeros(len(samples), dtype=np.float64)
        spectrum = np.zeros(len(samples), dtype=np.float64)
        ranks = np.ones(len(samples), dtype=np.int32)
        energies = np.ones(len(samples), dtype=np.float64)
        window = max(2, int(round(float(self.config.get("window_seconds", 8.0)) * float(self.config.get("sample_fps", 2.0)))))
        stride = max(1, int(round(float(self.config.get("stride_seconds", 2.0)) * float(self.config.get("sample_fps", 2.0)))))
        previous_u = previous_s = None
        for left in range(0, len(samples), stride):
            right = min(len(samples), left + window)
            matrix = features[left:right].T
            u, singular, vt, rank, energy = self.low_rank_decomposition(matrix)
            local_rec = self.reconstruction_novelty(matrix, u, vt)
            centre = left + max(0, (right - left - 1) // 2)
            reconstruction[left:right] = np.maximum(reconstruction[left:right], local_rec)
            change = self.subspace_change(previous_u, u)
            spec = self.spectrum_change(previous_s, singular)
            subspace[centre] = max(subspace[centre], change)
            spectrum[centre] = max(spectrum[centre], spec)
            ranks[left:right] = rank
            energies[left:right] = energy
            previous_u, previous_s = u, singular
        for values in (reconstruction, subspace, spectrum):
            peak = float(np.percentile(values, 95))
            if peak > 1e-9:
                values /= peak
        weights = self.config.get("score_weights", {})
        coverage = np.linspace(0.0, 1.0, len(samples), dtype=np.float64)
        scores[:] = (float(weights.get("reconstruction", 0.40)) * reconstruction
                     + float(weights.get("subspace", 0.30)) * subspace
                     + float(weights.get("spectrum", 0.20)) * spectrum
                     + float(weights.get("coverage", 0.10)) * coverage)
        yolo_options = self.config.get("yolo_events", {})
        yolo_payload = None
        yolo_observations = []
        yolo_status: dict[str, Any] = {
            "enabled": bool(yolo_options.get("enabled", True)),
            "method": "yolo_object_change_v1",
            "scan_fps_target": float(yolo_options.get("scan_fps", 1.0)),
            "input_sample_count": 0,
        }
        event_spans = []
        if yolo_status["enabled"]:
            candidates = self.select_yolo_candidates(samples, fold_assignments, scores)
            yolo_status["input_sample_count"] = len(candidates)
            yolo_status["input_reduction_ratio"] = round(1.0 - len(candidates) / max(1, len(samples)), 4)
            try:
                yolo_payload = self._run_yolo_scan(video_path, candidates, samples, output)
                yolo_observations = list(yolo_payload.get("samples") or [])
                event_spans = self.build_yolo_event_spans(yolo_observations, timestamps, features, scores)
                event_spans = self.merge_continuous_camera_spans(event_spans, samples)
                if not event_spans:
                    raise RuntimeError("YOLO scan completed but produced no event spans")
                yolo_status.update({
                    "status": "complete", "model": yolo_payload.get("model"),
                    "device": yolo_payload.get("device"),
                    "input_sample_count": len(yolo_observations),
                    "event_count": len(event_spans),
                    "unique_labels": sorted({label for row in yolo_observations for label in row.get("labels", [])}),
                })
            except Exception as error:
                yolo_status.update({"status": "fallback", "fallback": "svd_visual_boundaries",
                                    "error": f"{type(error).__name__}: {error}"})
                if bool(yolo_options.get("required", True)):
                    raise RuntimeError(
                        "required YOLO semantic event scan failed; refusing visual-only fallback: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                event_spans = []
        else:
            yolo_status.update({"status": "disabled", "fallback": "svd_visual_boundaries"})
        if not event_spans:
            boundaries = self.detect_event_boundaries(scores, timestamps)
            boundaries.append(len(samples))
            event_spans = self._visual_event_spans(boundaries, timestamps)

        second_pass_assignments, second_pass_metrics = self.fold_within_events(samples, event_spans)
        for first, second in zip(fold_assignments, second_pass_assignments):
            second["first_pass_group_id"] = first.get("group_id")
            second["first_pass_status"] = first.get("status")
        fold_group_ids = [item.get("group_id") for item in second_pass_assignments]
        scenes: list[WorldMMScene] = []
        all_selected: list[int] = []
        source_cap = cv2.VideoCapture(str(video_path))
        if not source_cap.isOpened():
            raise RuntimeError(f"cannot reopen video for selected keyframes: {video_path}")
        for scene_index, event_span in enumerate(event_spans):
            start, end = int(event_span["start_index"]), int(event_span["end_index"])
            if end <= start:
                continue
            if int(event_span.get("max_person_count") or 0) == 0:
                picks = [max(
                    range(start, end),
                    key=lambda index: self._structural_subject_coverage(samples[index][2]),
                )]
                selection_reason = "largest_complete_structural_subject"
            else:
                picks = [self.select_event_representative(
                    start, end, scores, features, samples, event_span,
                )]
                selection_reason = "adaptive_state_medoid_quality_coverage"
            keyframes: list[WorldMMKeyframe] = []
            for ordinal, index in enumerate(picks, 1):
                timestamp, frame_index, thumbnail = samples[index]
                target = frames_dir / f"scene_{scene_index:04d}_kf_{ordinal:04d}.jpg"
                source_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, source_frame = source_cap.read()
                if not ok:
                    source_cap.release()
                    raise RuntimeError(f"cannot reread selected source frame {frame_index}")
                if not cv2.imwrite(str(target), source_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                    source_cap.release()
                    raise RuntimeError(f"cannot write selected keyframe: {target}")
                fold = second_pass_assignments[index]
                first_fold = fold_assignments[index]
                raw = {
                    "reconstruction_error": float(reconstruction[index]),
                    "subspace_change": float(subspace[index]), "spectrum_change": float(spectrum[index]),
                    "svd_rank": int(ranks[index]), "energy_ratio": float(energies[index]),
                    "fold_group_id": fold["group_id"], "fold_status": fold["status"],
                    "fold_residual": fold["reconstruction_residual"],
                    "first_pass_fold_group_id": first_fold["group_id"],
                    "first_pass_fold_status": first_fold["status"],
                    "second_pass_fold_group_id": fold.get("event_fold_group_id"),
                    "yolo_event_id": event_span["event_id"],
                    "yolo_labels": event_span.get("yolo_labels", []),
                }
                keyframes.append(WorldMMKeyframe(code=f"{video_id}_svd_{scene_index:04d}_{ordinal:04d}", path=str(target), timestamp_sec=float(timestamp), frame_index=int(frame_index), score=float(scores[index]), selection_reason=selection_reason, raw=raw))
                all_selected.append(index)
            scene_raw = {key: value for key, value in event_span.items() if key not in {"start_index", "end_index"}}
            scene_raw["boundary_score"] = float(scores[start])
            scenes.append(WorldMMScene(
                scene_id=f"{video_id}_{event_span['event_id']}", index=scene_index,
                start_sec=float(timestamps[start]),
                end_sec=float(timestamps[min(end - 1, len(timestamps) - 1)]),
                keyframes=keyframes, semantic_labels=list(event_span.get("semantic_labels") or ["svd_lowrank"]),
                raw=scene_raw,
            ))
        source_cap.release()
        yolo_status["events"] = event_spans
        yolo_status["scan_sample_count"] = len(yolo_observations)
        second_pass_metrics["first_pass_folded_frame_count"] = fold_metrics["folded_frame_count"]
        second_pass_metrics["first_pass_group_count"] = fold_metrics["fold_group_count"]
        metrics = {
            "timestamps": timestamps.tolist(), "score": scores.tolist(),
            "reconstruction_error": reconstruction.tolist(),
            "subspace_change": subspace.tolist(), "spectrum_change": spectrum.tolist(),
            "svd_rank": ranks.tolist(), "energy_ratio": energies.tolist(),
            "folding": fold_metrics,
            "folding_passes": {"first_pass_global": fold_metrics, "second_pass_per_event": second_pass_metrics},
            "event_generation": yolo_status,
            "runtime_seconds": round(time.perf_counter() - started, 4), "config": self.config,
        }
        algorithm_name = "svd_lowrank_yolo_two_pass" if yolo_status.get("status") == "complete" else "svd_lowrank_two_pass_visual_fallback"
        (output / "analysis.json").write_text(json.dumps({
            "video": video, "algorithm": algorithm_name,
            "metrics": metrics, "fold_assignments": fold_assignments,
            "second_pass_fold_assignments": second_pass_assignments,
            "yolo_observations": yolo_observations, "selected_indices": all_selected,
        }, ensure_ascii=False), encoding="utf-8")
        return WorldMMResult(
            video=video, scenes=scenes, output_dir=str(output),
            manifest={"method": algorithm_name, "algorithm": "svd_lowrank", "metrics": metrics},
            full_keyframe_count=len(samples), summary_keyframe_count=len(all_selected),
            selected_keyframe_count=len(all_selected),
        )

    run = extract
