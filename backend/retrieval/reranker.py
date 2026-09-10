"""Temporal-clustered BGE reranking for the final RRF candidate head.

The retrieval kernel remains responsible for recall and RRF fusion.  This
module only reorders the bounded candidate head and removes near-duplicate
video frames.  It never reads benchmark ground truth.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SPACE_RE = re.compile(r"\s+")
_NOISE_VALUES = {
    "uniform_sample",
    "uniform sample",
    "uniform-sample",
    "sample",
    "keyframe",
    "frame",
}


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _clean_text(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float)):
        return ""
    text = _SPACE_RE.sub(" ", str(value)).strip(" \t\r\n,;|：:，；")
    if not text or text.casefold() in _NOISE_VALUES:
        return ""
    return text


def _flatten_text(value: Any, *, include_keys: bool = False) -> list[str]:
    """Flatten structured semantic values without serialising JSON noise."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_text(item, include_keys=include_keys))
        return out
    if isinstance(value, dict):
        out = []
        # Prefer the actual semantic values and ignore bookkeeping fields.
        priority = (
            "label", "name", "description", "object_description", "details",
            "attributes", "subject", "predicate", "relation", "object",
            "action", "event", "text", "value", "primary",
        )
        seen_keys = set()
        for key in priority:
            if key in value:
                seen_keys.add(key)
                out.extend(_flatten_text(value[key], include_keys=include_keys))
        if not out:
            for key, item in value.items():
                if key in seen_keys or key.casefold() in {
                    "id", "asset_id", "observation_id", "source", "source_type",
                    "sample_type", "confidence", "score", "bbox", "box",
                    "timestamp", "timestamp_sec", "frame_index", "model",
                }:
                    continue
                values = _flatten_text(item, include_keys=include_keys)
                if include_keys and values:
                    cleaned_key = _clean_text(key.replace("_", " "))
                    if cleaned_key:
                        out.append(cleaned_key)
                out.extend(values)
        return out
    return []


def _unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _flatten_text(value):
            key = _SPACE_RE.sub(" ", text).casefold()
            if key and key not in seen:
                seen.add(key)
                out.append(text)
    return out


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


@dataclass(frozen=True)
class TemporalCluster:
    center_id: str
    member_ids: tuple[str, ...]
    source_video_id: str | None = None
    center_timestamp_sec: float | None = None


@dataclass
class RerankResult:
    asset_ids: list[str]
    score_by_asset: dict[str, float]
    clusters: list[TemporalCluster]
    telemetry: dict[str, Any]


def _timeline_key(asset: dict) -> tuple[str | None, float | None]:
    metadata = _as_dict(asset.get("metadata_json"))
    parent = _first_present(
        asset.get("parent_asset_id"),
        metadata.get("parent_asset_id"),
        metadata.get("source_video_asset_id"),
        metadata.get("video_asset_id"),
        metadata.get("parent_video_id"),
    )
    timestamp = _first_present(
        asset.get("source_timestamp_sec"),
        metadata.get("source_timestamp_sec"),
        metadata.get("timestamp_sec"),
        metadata.get("video_timestamp_sec"),
        metadata.get("event_start_sec"),
    )
    try:
        timestamp = float(timestamp) if timestamp is not None else None
    except (TypeError, ValueError):
        timestamp = None
    if not parent or timestamp is None or not math.isfinite(timestamp):
        return None, None
    return str(parent), timestamp


def cluster_candidates(
    candidate_ids: list[str],
    store,
    *,
    window_seconds: float = 1.0,
) -> list[TemporalCluster]:
    """Greedily cluster candidates around RRF-ranked centres.

    Each unassigned candidate becomes a centre in the incoming rank order.
    Lower-ranked candidates from the same source video within +/- ``window``
    of that centre join it.  This is deliberately centre-based rather than
    transitive, so frames at 0.0, 0.8 and 1.6 seconds do not collapse into one
    1.6-second-wide cluster when the configured radius is one second.
    """
    window = max(0.0, float(window_seconds))
    assets = {asset_id: (store.get_asset(asset_id) or {}) for asset_id in candidate_ids}
    timeline = {asset_id: _timeline_key(asset) for asset_id, asset in assets.items()}
    assigned: set[str] = set()
    clusters: list[TemporalCluster] = []
    for index, center_id in enumerate(candidate_ids):
        if center_id in assigned:
            continue
        assigned.add(center_id)
        video_id, center_ts = timeline.get(center_id, (None, None))
        members = [center_id]
        if video_id is not None and center_ts is not None:
            for other_id in candidate_ids[index + 1:]:
                if other_id in assigned:
                    continue
                other_video, other_ts = timeline.get(other_id, (None, None))
                if (other_video == video_id and other_ts is not None
                        and abs(other_ts - center_ts) <= window):
                    assigned.add(other_id)
                    members.append(other_id)
        clusters.append(TemporalCluster(
            center_id=center_id,
            member_ids=tuple(members),
            source_video_id=video_id,
            center_timestamp_sec=center_ts,
        ))
    return clusters


def _nested_values(container: Any, keys: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    if not isinstance(container, dict):
        return values
    for key in keys:
        value = container.get(key)
        if value not in (None, "", [], {}):
            values.append(value)
    gamma = container.get("gamma")
    if isinstance(gamma, dict):
        for key in keys:
            value = gamma.get(key)
            if value not in (None, "", [], {}):
                values.append(value)
    return values


def build_cluster_text(cluster: TemporalCluster, store, *, max_chars: int = 6000) -> str:
    """Build one de-duplicated reranker passage for a temporal cluster."""
    context_buckets: dict[str, list[Any]] = {
        "Time": [],
        "Place": [],
        "People": [],
    }
    buckets: dict[str, list[Any]] = {
        "Caption": [],
        "OCR": [],
        "Object": [],
        "Object Description": [],
        "Relation": [],
        "Action/Event": [],
    }
    visual_fallback: list[Any] = []

    for asset_id in cluster.member_ids:
        asset = store.get_asset(asset_id) or {}
        asset_meta = _as_dict(asset.get("metadata_json"))
        geocode = _as_dict(asset_meta.get("reverse_geocode"))
        context_buckets["Time"].extend([
            asset.get("captured_at"), asset_meta.get("captured_at"),
        ])
        context_buckets["Place"].extend([
            asset.get("captured_location"), asset_meta.get("captured_location"),
            geocode.get("label"), geocode.get("name"), geocode.get("city"),
            geocode.get("district"), geocode.get("province"),
        ])
        observations = store.list_observations(asset_id=asset_id, limit=20) or []
        for observation in observations:
            detail = _as_dict(observation.get("detail") or observation.get("detail_json"))
            canonical = _as_dict(observation.get("canonical") or observation.get("canonical_json"))
            raw = _as_dict(observation.get("raw") or observation.get("raw_json"))
            context_buckets["Time"].append(observation.get("captured_at"))
            context_buckets["Place"].extend([
                observation.get("place"),
                *_nested_values(detail, ("place",)),
                *_nested_values(canonical, ("place",)),
                *_nested_values(raw, ("place",)),
            ])
            context_buckets["People"].extend([
                observation.get("people"), observation.get("people_json"),
                *_nested_values(detail, ("people",)),
                *_nested_values(canonical, ("people",)),
                *_nested_values(raw, ("people",)),
            ])

            buckets["Caption"].extend([
                observation.get("caption"),
                *_nested_values(detail, ("caption", "florence_caption")),
                *_nested_values(canonical, ("caption", "florence_caption")),
                *_nested_values(raw, ("caption", "florence_caption")),
            ])
            buckets["OCR"].extend([
                observation.get("ocr_text"),
                *_nested_values(detail, ("ocr", "ocr_text")),
                *_nested_values(canonical, ("ocr", "ocr_text")),
                *_nested_values(raw, ("ocr", "ocr_text")),
            ])

            object_values = [
                observation.get("objects"), observation.get("objects_json"),
                *_nested_values(detail, ("objects",)),
                *_nested_values(canonical, ("objects",)),
                *_nested_values(raw, ("objects",)),
            ]
            object_labels: list[Any] = []
            for object_value in object_values:
                if isinstance(object_value, str):
                    try:
                        object_value = json.loads(object_value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        object_labels.append(object_value)
                        continue
                items = object_value if isinstance(object_value, list) else [object_value]
                for item in items:
                    if isinstance(item, dict):
                        object_labels.extend([item.get("label"), item.get("name"), item.get("primary")])
                    else:
                        object_labels.append(item)
            buckets["Object"].extend(object_labels)
            buckets["Object Description"].extend([
                *_nested_values(detail, ("object_description", "object_descriptions")),
                *_nested_values(canonical, ("object_description", "object_descriptions")),
                *_nested_values(raw, ("object_description", "object_descriptions")),
            ])
            # Object dictionaries often keep their descriptions under details.
            for object_value in object_values:
                if isinstance(object_value, str):
                    try:
                        object_value = json.loads(object_value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                if isinstance(object_value, list):
                    for item in object_value:
                        if isinstance(item, dict):
                            buckets["Object Description"].extend([
                                item.get("description"), item.get("object_description"),
                                item.get("details"), item.get("attributes"),
                            ])

            buckets["Relation"].extend([
                observation.get("spatial_relations"),
                observation.get("spatial_relations_json"),
                *_nested_values(detail, ("relation", "relations", "spatial_relations")),
                *_nested_values(canonical, ("relation", "relations", "spatial_relations")),
                *_nested_values(raw, ("relation", "relations", "spatial_relations")),
            ])
            buckets["Action/Event"].extend([
                observation.get("activity"), observation.get("event_type"),
                observation.get("transcript"),
                *_nested_values(detail, ("action", "actions", "event", "event_type")),
                *_nested_values(canonical, ("action", "actions", "event", "event_type")),
                *_nested_values(raw, ("action", "actions", "event", "event_type")),
            ])
            visual_fallback.extend([
                observation.get("visual_text"),
                *_nested_values(detail, ("visual_text",)),
                *_nested_values(canonical, ("visual_text",)),
                *_nested_values(raw, ("visual_text",)),
            ])
        visual_fallback.extend(_nested_values(asset_meta, ("visual_text",)))

    lines: list[str] = []
    globally_seen: set[str] = set()
    for label, raw_values in context_buckets.items():
        values = []
        for value in _unique(raw_values):
            key = _SPACE_RE.sub(" ", value).casefold()
            if key not in globally_seen:
                globally_seen.add(key)
                values.append(value)
        if values:
            lines.append(f"{label}: {'; '.join(values)}")
    semantic_line_count = len(lines)
    for label, raw_values in buckets.items():
        values = []
        for value in _unique(raw_values):
            key = _SPACE_RE.sub(" ", value).casefold()
            if key not in globally_seen:
                globally_seen.add(key)
                values.append(value)
        if values:
            lines.append(f"{label}: {'; '.join(values)}")

    # visual_text already contains caption/OCR/object in legacy exports.  It
    # must only be used when every structured semantic field is empty.
    if len(lines) == semantic_line_count:
        fallback = _unique(visual_fallback)
        if fallback:
            lines.append(f"Visual: {'; '.join(fallback)}")
    return "\n".join(lines)[:max(1, int(max_chars))]


class LocalBgeReranker:
    """Lazy local cross-encoder with optional idle GPU/CPU unload."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        idle_unload_seconds: float = 60.0,
    ):
        self.model_path = str(model_path)
        self.requested_device = str(device or "auto")
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.idle_unload_seconds = max(0.0, float(idle_unload_seconds))
        self._model = None
        self._tokenizer = None
        self._device = "unloaded"
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._timer: threading.Timer | None = None
        self.last_load_ms = 0.0

    @property
    def device(self) -> str:
        return self._device

    def _resolved_device(self, torch) -> str:
        requested = self.requested_device.casefold()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.requested_device

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"BGE reranker model directory not found: {model_dir}")
        started = time.monotonic()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._resolved_device(torch)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        self._tokenizer = tokenizer
        self._model = model
        self._device = str(device)
        self.last_load_ms = round((time.monotonic() - started) * 1000, 1)

    def _schedule_idle_unload(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.idle_unload_seconds <= 0:
            return
        expected_last_used = self._last_used

        def _idle_unload():
            with self._lock:
                if self._last_used != expected_last_used:
                    return
                if time.monotonic() - self._last_used + 0.05 < self.idle_unload_seconds:
                    return
                self.unload()

        self._timer = threading.Timer(self.idle_unload_seconds, _idle_unload)
        self._timer.daemon = True
        self._timer.start()

    def unload(self):
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._device = "unloaded"
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            self._load()
            import torch

            scores: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(passages), self.batch_size):
                    batch = passages[start:start + self.batch_size]
                    encoded = self._tokenizer(
                        [[query, passage] for passage in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self._device) for key, value in encoded.items()}
                    logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
                    scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            self._last_used = time.monotonic()
            self._schedule_idle_unload()
            return [float(score) for score in scores]


_RERANKER: LocalBgeReranker | None = None
_RERANKER_CONFIG: tuple | None = None
_RERANKER_LOCK = threading.Lock()


def get_local_reranker() -> LocalBgeReranker:
    global _RERANKER, _RERANKER_CONFIG
    model_path = os.getenv("SENTRIX_RERANKER_MODEL_PATH", "").strip()
    config = (
        model_path,
        os.getenv("SENTRIX_RERANKER_DEVICE", "auto"),
        int(os.getenv("SENTRIX_RERANKER_BATCH_SIZE", "8")),
        int(os.getenv("SENTRIX_RERANKER_MAX_LENGTH", "512")),
        float(os.getenv("SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS", "60")),
    )
    with _RERANKER_LOCK:
        if _RERANKER is None or _RERANKER_CONFIG != config:
            if _RERANKER is not None:
                _RERANKER.unload()
            _RERANKER = LocalBgeReranker(
                config[0], device=config[1], batch_size=config[2],
                max_length=config[3], idle_unload_seconds=config[4],
            )
            _RERANKER_CONFIG = config
        return _RERANKER


def rerank_candidates(
    query: str,
    candidate_ids: list[str],
    store,
    *,
    scorer=None,
    window_seconds: float = 1.0,
    max_passage_chars: int = 6000,
    coarse_rank_weight: float | None = None,
    bge_rank_weight: float | None = None,
    fusion_rrf_k: float | None = None,
    skip_whole_video_only: bool | None = None,
) -> RerankResult:
    """Cluster and BGE-rerank a coarse RRF candidate list."""
    started = time.monotonic()
    cluster_started = time.monotonic()
    clusters = cluster_candidates(candidate_ids, store, window_seconds=window_seconds)
    cluster_ms = round((time.monotonic() - cluster_started) * 1000, 1)

    skip_whole_video = (
        str(os.getenv("SENTRIX_RERANKER_SKIP_WHOLE_VIDEO_ONLY", "1")).strip().lower()
        in {"1", "true", "yes", "on"}
        if skip_whole_video_only is None else bool(skip_whole_video_only)
    )
    whole_video_only = bool(candidate_ids) and all(
        str((store.get_asset(asset_id) or {}).get("media_type") or "") == "video"
        and _timeline_key(store.get_asset(asset_id) or {}) == (None, None)
        for asset_id in candidate_ids
    )
    if skip_whole_video and whole_video_only:
        return RerankResult(
            asset_ids=[cluster.center_id for cluster in clusters],
            score_by_asset={},
            clusters=clusters,
            telemetry={
                "status": "skipped_whole_video_only",
                "reason": "no persisted keyframe timestamps; preserve coarse RRF order",
                "coarse_candidates": len(candidate_ids),
                "temporal_clusters": len(clusters),
                "cluster_window_seconds": float(window_seconds),
                "cluster_ms": cluster_ms,
                "text_build_ms": 0.0,
                "model_ms": 0.0,
                "total_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )

    text_started = time.monotonic()
    passages = [build_cluster_text(cluster, store, max_chars=max_passage_chars)
                for cluster in clusters]
    text_ms = round((time.monotonic() - text_started) * 1000, 1)
    nonempty = [index for index, passage in enumerate(passages) if passage]
    if not nonempty:
        return RerankResult(
            asset_ids=[cluster.center_id for cluster in clusters],
            score_by_asset={},
            clusters=clusters,
            telemetry={
                "status": "no_text", "coarse_candidates": len(candidate_ids),
                "temporal_clusters": len(clusters), "cluster_ms": cluster_ms,
                "text_build_ms": text_ms,
                "model_ms": 0.0,
                "total_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )

    scorer = scorer or get_local_reranker()
    model_started = time.monotonic()
    nonempty_scores = scorer.score(query, [passages[index] for index in nonempty])
    model_ms = round((time.monotonic() - model_started) * 1000, 1)
    if len(nonempty_scores) != len(nonempty):
        raise RuntimeError("BGE reranker returned a different number of scores")
    all_scores = [-1.0] * len(clusters)
    for index, score in zip(nonempty, nonempty_scores):
        all_scores[index] = float(score)
    coarse_weight = (float(os.getenv("SENTRIX_RERANKER_COARSE_RANK_WEIGHT", "1"))
                     if coarse_rank_weight is None else float(coarse_rank_weight))
    bge_weight = (float(os.getenv("SENTRIX_RERANKER_BGE_RANK_WEIGHT", "0.25"))
                  if bge_rank_weight is None else float(bge_rank_weight))
    rrf_k = (float(os.getenv("SENTRIX_RERANKER_FUSION_RRF_K", "60"))
             if fusion_rrf_k is None else float(fusion_rrf_k))
    rrf_k = max(0.0, rrf_k)
    bge_order = sorted(range(len(clusters)), key=lambda index: (-all_scores[index], index))
    bge_ranks = {index: rank for rank, index in enumerate(bge_order, 1)}
    fused_scores = {
        index: (coarse_weight / (rrf_k + index + 1)
                + bge_weight / (rrf_k + bge_ranks[index]))
        for index in range(len(clusters))
    }
    order = sorted(range(len(clusters)), key=lambda index: (-fused_scores[index], index))
    ranked_clusters = [clusters[index] for index in order]
    score_by_asset = {
        clusters[index].center_id: round(all_scores[index], 8) for index in range(len(clusters))
    }
    telemetry = {
        "status": "ok",
        "model": Path(getattr(scorer, "model_path", "bge-reranker")).name,
        "device": getattr(scorer, "device", "unknown"),
        "coarse_candidates": len(candidate_ids),
        "temporal_clusters": len(clusters),
        "merged_candidates": len(candidate_ids) - len(clusters),
        "cluster_window_seconds": float(window_seconds),
        "ranking_policy": "coarse_bge_rank_fusion",
        "coarse_rank_weight": coarse_weight,
        "bge_rank_weight": bge_weight,
        "fusion_rrf_k": rrf_k,
        "cluster_ms": cluster_ms,
        "text_build_ms": text_ms,
        "model_load_ms": float(getattr(scorer, "last_load_ms", 0.0)),
        "model_ms": model_ms,
        "total_ms": round((time.monotonic() - started) * 1000, 1),
    }
    return RerankResult(
        asset_ids=[cluster.center_id for cluster in ranked_clusters],
        score_by_asset=score_by_asset,
        clusters=ranked_clusters,
        telemetry=telemetry,
    )
