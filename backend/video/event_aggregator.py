"""Temporal Event Aggregation for the frozen hybrid keyframe package.

This module is deliberately independent from the v2.0 scene importer.  It
keeps every keyframe as evidence and only creates a second, event-level index.
The first implementation is local/adjacent temporal grouping; it is not a
global KMeans pass and it never edits the baseline video_scene rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EVENTAGG_METHOD_VERSION = "keyframe-hybrid-v2.1-eventagg"
EVENTAGG_CACHE_VERSION = "eventagg-dino-cache-v1"
DEFAULT_CONFIG = {
    "enabled": True,
    "visual_weight": 0.60,
    "object_weight": 0.20,
    "time_weight": 0.10,
    "pose_weight": 0.10,
    "merge_threshold": 0.68,
    "strong_visual_threshold": 0.84,
    "soft_time_gap_sec": 60.0,
    "hard_time_gap_sec": 180.0,
    "min_event_frames": 2,
    "max_event_duration_sec": 300.0,
    "singleton_merge_threshold": 0.80,
    "representative_frames": 1,
    "evidence_preview_frames": 3,
    "preserve_all_frames": True,
    "model_name": "dinov2_vits14",
    "model_version": "dinov2_vits14_pretrain",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(item):
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("label") or item.get("name") or item.get("primary") or "").strip()
    return ""


def _confidence(item):
    if isinstance(item, dict):
        return max(0.01, min(1.0, _as_float(item.get("confidence"), 1.0)))
    return 1.0


def _object_weights(items):
    values = {}
    for item in items or []:
        name = _label(item)
        if name:
            values[name] = max(values.get(name, 0.0), _confidence(item))
    return values


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _weighted_jaccard(left, right):
    keys = set(left) | set(right)
    if not keys:
        return None
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return float(numerator / denominator) if denominator else 0.0


def _fallback_image_embedding(path):
    """Small deterministic CPU fallback used only when DINO cannot load."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    histograms = [np.histogram(array[:, :, channel], bins=16, range=(0.0, 1.0), density=True)[0] for channel in range(3)]
    gray = array.mean(axis=2).reshape(-1)
    vector = np.concatenate([*histograms, gray[:: max(1, len(gray) // 128)][:128]])
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32).tolist() if norm else vector.tolist()


class DINOEmbedder:
    """Batch DINOv2 inference with hash/model/version-bound JSON cache."""

    def __init__(self, cache_dir, model_name=None, model_version=None, device=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or DEFAULT_CONFIG["model_name"]
        self.model_version = model_version or DEFAULT_CONFIG["model_version"]
        self.device = device or ("cuda:0" if self._cuda_available() else "cpu")
        self.model = None
        self.transform = None
        self.load_error = None

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @property
    def identity(self):
        return {"name": self.model_name, "version": self.model_version, "device": self.device}

    def _load(self):
        if self.model is not None or self.load_error:
            return self.model
        try:
            import torch
            import torchvision.transforms as transforms

            torch_home = os.getenv("TORCH_HOME", str(self.cache_dir.parent / "torch"))
            os.environ.setdefault("TORCH_HOME", torch_home)
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
        return self.model

    def _cache_path(self, video_id):
        return self.cache_dir / f"{video_id}.json"

    def _read_cache(self, video_id):
        path = self._cache_path(video_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("cache_version") != EVENTAGG_CACHE_VERSION:
                return {}
            if value.get("model") != self.identity:
                return {}
            return value.get("frames") or {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_cache(self, video_id, frames):
        path = self._cache_path(video_id)
        temp = path.with_suffix(".tmp")
        payload = {
            "cache_version": EVENTAGG_CACHE_VERSION, "model": self.identity,
            "created_at": _now(), "frames": frames,
        }
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    def embed(self, video_id, records, batch_size=16):
        started = time.perf_counter()
        cache = self._read_cache(video_id)
        missing = []
        embeddings = {}
        for record in records:
            frame_id = str(record.frame_id)
            frame_hash = str(record.frame_hash or "")
            cached = cache.get(frame_id) or {}
            if cached.get("frame_hash") == frame_hash and cached.get("embedding"):
                embeddings[frame_id] = cached["embedding"]
            else:
                missing.append(record)

        model = self._load()
        used_fallback = model is None
        if model is None:
            for record in missing:
                embeddings[str(record.frame_id)] = _fallback_image_embedding(record.image_path)
        else:
            import torch
            from PIL import Image

            with torch.no_grad():
                for offset in range(0, len(missing), max(1, int(batch_size))):
                    batch_records = missing[offset:offset + max(1, int(batch_size))]
                    batch = torch.stack([self.transform(Image.open(item.image_path).convert("RGB")) for item in batch_records]).to(self.device)
                    output = model(batch)
                    output = torch.nn.functional.normalize(output, dim=1).detach().cpu().tolist()
                    for item, vector in zip(batch_records, output):
                        embeddings[str(item.frame_id)] = vector

        for record in records:
            frame_id = str(record.frame_id)
            cache[frame_id] = {
                "frame_id": frame_id, "frame_hash": record.frame_hash,
                "model_name": self.model_name, "model_version": self.model_version,
                "embedding": embeddings[frame_id], "created_at": _now(),
            }
        self._write_cache(video_id, cache)
        return embeddings, {
            "embedding_seconds": round(time.perf_counter() - started, 4),
            "embedding_cache_hits": len(records) - len(missing),
            "embedding_cache_misses": len(missing),
            "embedding_model": self.model_name if not used_fallback else "histogram_fallback",
            "embedding_model_version": self.model_version if not used_fallback else "fallback-v1",
            "embedding_device": self.device if not used_fallback else "cpu",
            "embedding_error": self.load_error,
        }


@dataclass
class EventFrame:
    frame_id: str
    timestamp: float
    image_path: str
    frame_hash: str
    objects: list = field(default_factory=list)
    object_weights: dict = field(default_factory=dict)
    person_count: int = 0
    pose_summary: str = ""
    visual_embedding: list = field(default_factory=list)
    sharpness: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class EventGroup:
    event_id: str
    members: list[EventFrame]
    merge_score: float = 1.0
    merge_breakdown: dict = field(default_factory=dict)
    representative: EventFrame | None = None
    preview: list[EventFrame] = field(default_factory=list)

    @property
    def start(self):
        return self.members[0].timestamp

    @property
    def end(self):
        return self.members[-1].timestamp

    @property
    def objects(self):
        values = {}
        for member in self.members:
            for label, weight in member.object_weights.items():
                values[label] = max(values.get(label, 0.0), weight)
        return sorted(values, key=lambda label: (-values[label], label))

    @property
    def person_count_range(self):
        values = [member.person_count for member in self.members]
        return [min(values or [0]), max(values or [0])]


class EventAggregator:
    def __init__(self, config=None, cache_dir=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.cache_dir = Path(cache_dir or os.getenv("SENTRIX_EVENTAGG_CACHE", "cache/eventagg"))

    def _pair_score(self, current, previous):
        config = self.config
        delta = max(0.0, current.timestamp - previous.timestamp)
        visual = _cosine(current.visual_embedding, previous.visual_embedding)
        object_score = _weighted_jaccard(current.object_weights, previous.object_weights)
        time_score = math.exp(-delta / max(0.001, float(config["soft_time_gap_sec"])))
        pose_available = bool(current.pose_summary and previous.pose_summary)
        pose_score = (1.0 if current.pose_summary == previous.pose_summary else 0.0) if pose_available else None
        weighted = [(visual, float(config["visual_weight"])), (object_score, float(config["object_weight"])), (time_score, float(config["time_weight"]))]
        if pose_score is not None:
            weighted.append((pose_score, float(config["pose_weight"])))
        else:
            # Missing pose never makes a pair fail; redistribute its weight.
            weighted = [(score, weight / sum(item[1] for item in weighted)) for score, weight in weighted]
        score = sum(score * weight for score, weight in weighted)
        object_change = 1.0 - (object_score if object_score is not None else 0.0)
        person_change = current.person_count != previous.person_count
        pose_change = pose_available and current.pose_summary != previous.pose_summary
        boundary_penalty = 0.0
        if person_change:
            boundary_penalty += 0.10
        if object_change >= 0.55:
            boundary_penalty += 0.08
        if pose_change:
            boundary_penalty += 0.06
        score = max(0.0, score - boundary_penalty)
        return score, {
            "visual": round(visual, 5), "object": round(object_score if object_score is not None else 0.0, 5),
            "time": round(time_score, 5), "pose": round(pose_score if pose_score is not None else 0.0, 5),
            "pose_available": pose_available, "delta_t": round(delta, 4),
            "object_change": round(object_change, 5), "person_change": person_change,
            "pose_change": pose_change, "boundary_penalty": round(boundary_penalty, 5),
            "score": round(score, 5),
        }

    def _choose_representative(self, group):
        members = group.members
        if len(members) == 1:
            return members[0]
        sharpness = np.asarray([max(0.0, item.sharpness) for item in members], dtype=np.float32)
        richness = np.asarray([sum(item.object_weights.values()) for item in members], dtype=np.float32)
        center = (group.start + group.end) / 2.0
        temporal = np.asarray([math.exp(-abs(item.timestamp - center) / max(group.end - group.start, 1.0)) for item in members], dtype=np.float32)

        def normalize(values):
            if len(values) == 0 or float(values.max() - values.min()) <= 1e-6:
                return np.ones(len(values), dtype=np.float32)
            return (values - values.min()) / (values.max() - values.min())

        scores = 0.45 * normalize(sharpness) + 0.35 * normalize(richness) + 0.20 * normalize(temporal)
        return members[int(np.argmax(scores))]

    def aggregate(self, records, video_id, threshold=None):
        started = time.perf_counter()
        config = {**self.config}
        if threshold is not None:
            config["merge_threshold"] = float(threshold)
        ordered = sorted(records, key=lambda item: item.timestamp)
        groups = []
        boundaries = []
        if ordered:
            current = [ordered[0]]
            last_score, last_breakdown = 1.0, {"score": 1.0}
            for previous, item in zip(ordered, ordered[1:]):
                delta = item.timestamp - previous.timestamp
                score, breakdown = self._pair_score(item, previous)
                duration = item.timestamp - current[0].timestamp
                strong_visual = breakdown["visual"] >= float(config["strong_visual_threshold"])
                boundary = delta > float(config["hard_time_gap_sec"])
                if not boundary and duration <= float(config["max_event_duration_sec"]):
                    boundary = score < float(config["merge_threshold"])
                    if strong_visual and breakdown["object_change"] < 0.70 and not breakdown["person_change"]:
                        boundary = False
                if boundary:
                    groups.append((current, last_score, last_breakdown))
                    boundaries.append({"timestamp": item.timestamp, "reason": "hard_gap" if delta > float(config["hard_time_gap_sec"]) else "score", "score": breakdown})
                    current = [item]
                else:
                    current.append(item)
                last_score, last_breakdown = score, breakdown
            groups.append((current, last_score, last_breakdown))

        # Merge a singleton only with an adjacent event; this is deliberately
        # conservative and cannot bridge a hard gap or an unrelated region.
        changed = True
        while changed:
            changed = False
            for index, (members, score, breakdown) in enumerate(list(groups)):
                if len(members) != 1 or len(groups) <= 1:
                    continue
                candidates = []
                for neighbor in (index - 1, index + 1):
                    if 0 <= neighbor < len(groups):
                        other = groups[neighbor][0]
                        pair, detail = self._pair_score(members[0], other[-1] if neighbor < index else other[0])
                        if pair >= float(config["singleton_merge_threshold"]) and other[-1].timestamp - other[0].timestamp <= float(config["max_event_duration_sec"]):
                            candidates.append((pair, neighbor, detail))
                if candidates:
                    _, neighbor, detail = max(candidates, key=lambda item: item[0])
                    if neighbor < index:
                        groups[neighbor] = (groups[neighbor][0] + members, detail["score"], detail)
                    else:
                        groups[neighbor] = (members + groups[neighbor][0], detail["score"], detail)
                    groups.pop(index)
                    changed = True
                    break

        result = []
        for index, (members, score, breakdown) in enumerate(groups):
            group = EventGroup(f"eventagg_{video_id}_{index + 1:04d}", members, score, breakdown)
            group.representative = self._choose_representative(group)
            by_center = sorted(members, key=lambda item: abs(item.timestamp - (group.start + group.end) / 2.0))
            group.preview = sorted(by_center[:max(1, int(config["evidence_preview_frames"]))], key=lambda item: item.timestamp)
            result.append(group)

        baseline_objects = set(item for record in ordered for item in record.object_weights)
        event_objects = set(item for group in result for item in group.objects)
        evidence_count = sum(len(group.members) for group in result)
        metrics = {
            "baseline_frame_count": len(ordered), "event_count": len(result),
            "timeline_compression": round(1.0 - len(result) / len(ordered), 6) if ordered else 0.0,
            "singleton_event_count": sum(len(group.members) == 1 for group in result),
            "singleton_rate": round(sum(len(group.members) == 1 for group in result) / len(result), 6) if result else 0.0,
            "mean_frames_per_event": round(float(np.mean([len(group.members) for group in result])) if result else 0.0, 6),
            "median_frames_per_event": round(float(np.median([len(group.members) for group in result])) if result else 0.0, 6),
            "p90_frames_per_event": round(float(np.percentile([len(group.members) for group in result], 90)) if result else 0.0, 6),
            "max_frames_per_event": max([len(group.members) for group in result] or [0]),
            "evidence_retention": round(evidence_count / len(ordered), 6) if ordered else 0.0,
            "object_coverage": round(len(event_objects & baseline_objects) / len(baseline_objects), 6) if baseline_objects else 1.0,
            "aggregation_seconds": round(time.perf_counter() - started, 4),
            "boundary_count": len(boundaries), "boundaries": boundaries,
            "config": config,
        }
        return result, metrics

    @staticmethod
    def records_from_package(package, frame_assets=None):
        """Build records from hybrid package JSON without another detector pass."""
        frame_assets = {str(item.get("source_frame_index")): item for item in (frame_assets or [])}
        semantic_frames = {int(item.get("encoded_frame_index", 0)): item for item in package.get("frames", []) if isinstance(item, dict)}
        records = []
        for index, frame in semantic_frames.items():
            asset = frame_assets.get(str(frame.get("source_frame_index"))) or {}
            raw_objects = frame.get("objects") or asset.get("objects") or []
            actions = frame.get("actions") or asset.get("actions") or []
            image_path = frame.get("webp_path") or asset.get("path")
            if not image_path:
                continue
            path = Path(image_path)
            if not path.is_file():
                continue
            frame_hash = str(asset.get("content_sha256") or "")
            if not frame_hash:
                frame_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            labels = {_label(item) for item in raw_objects + actions if _label(item)}
            records.append(EventFrame(
                frame_id=str(asset.get("id") or f"frame_{index:05d}"),
                timestamp=_as_float(frame.get("source_timestamp_sec"), 0.0),
                image_path=str(path), frame_hash=frame_hash,
                objects=raw_objects, object_weights=_object_weights(raw_objects + actions),
                person_count=sum(1 for item in labels if item.lower() == "person"),
                pose_summary=str(frame.get("pose_summary") or ""), raw={**frame, "actions": actions},
            ))
        return records

    @staticmethod
    def group_to_dict(group):
        return {
            "event_id": group.event_id, "start_sec": group.start, "end_sec": group.end,
            "duration_sec": max(0.0, group.end - group.start), "frame_count": len(group.members),
            "representative_frame_id": group.representative.frame_id if group.representative else None,
            "member_frame_ids": [item.frame_id for item in group.members],
            "preview_frame_ids": [item.frame_id for item in group.preview],
            "object_summary": group.objects, "person_count_range": group.person_count_range,
            "merge_score": group.merge_score, "merge_breakdown": group.merge_breakdown,
            "method_version": EVENTAGG_METHOD_VERSION, "place_id": None,
        }
