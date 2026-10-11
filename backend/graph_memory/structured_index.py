# -*- coding: utf-8 -*-
"""
Structured Index & Filtering

Uses the structured `objects` (score + bbox) and `relations` (subject /
predicate / object / score) fields that were previously stored but never read.

Two capabilities:
  1. Score filtering: drop low-confidence detections (< MIN_SCORE) from
     object_labels / relation_labels before they enter the graph.
  2. Structured index: predicate -> {frame_id: [triplets]} and
     spatial index (frame_id -> [(label, cx, cy, w, h, score)]) for
     predicate-precise matching and bbox-based spatial queries at retrieval.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

MIN_OBJECT_SCORE = 0.15
MIN_RELATION_SCORE = 0.05


def filter_objects(raw_objects: list, min_score=MIN_OBJECT_SCORE) -> Tuple[list, list]:
    """Return (filtered_labels, filtered_objects) dropping low-confidence detections."""
    kept_objs = [o for o in raw_objects if (o.get("score") or 0) >= min_score]
    # de-dup labels (same label may appear in multiple bboxes)
    labels = list(dict.fromkeys(o.get("label", "") for o in kept_objs if o.get("label")))
    return labels, kept_objs


def filter_relations(raw_relations: list, min_score=MIN_RELATION_SCORE) -> Tuple[list, list]:
    """Return (filtered_labels, filtered_relations)."""
    kept_rels = [r for r in raw_relations if (r.get("score") or 0) >= min_score]
    labels = []
    seen = set()
    for r in kept_rels:
        s = "%s %s %s" % (r.get("subject",""), r.get("predicate",""), r.get("object",""))
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            labels.append(s)
    return labels, kept_rels


def _bbox_is_normalized(bbox: list) -> bool:
    """WorldMM emits both pixel boxes and already-normalized [0,1] boxes."""
    return all(0.0 <= float(v) <= 1.5 for v in bbox[:4])


def _bbox_extent(bbox: list) -> Tuple[float, float]:
    if not bbox or len(bbox) < 4:
        return 0.0, 0.0
    if _bbox_is_normalized(bbox):
        return abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1]))
    # Ego4D QA frames in this collection are 1440x1080. This matters for
    # already-normalized YOLO boxes mixed with pixel-space SGG proposals.
    return ((float(bbox[2]) - float(bbox[0])) / 1440.0,
            (float(bbox[3]) - float(bbox[1])) / 1080.0)


def bbox_center(bbox: list) -> Tuple[float, float]:
    """Return (cx, cy) normalized to [0,1] for pixel or normalized boxes."""
    if not bbox or len(bbox) < 4:
        return 0.5, 0.5
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if _bbox_is_normalized(bbox):
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return ((x1 + x2) / 2.0 / 1440.0), ((y1 + y2) / 2.0 / 1080.0)


def spatial_region(cx: float, cy: float) -> str:
    """Map normalized center to a region string."""
    horiz = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    vert = "top" if cy < 0.33 else ("bottom" if cy > 0.67 else "middle")
    if horiz == "center" and vert == "middle":
        return "center"
    if horiz == "center":
        return vert
    if vert == "middle":
        return horiz
    return "%s-%s" % (vert, horiz)


class StructuredIndex:
    """Indexes predicate and spatial info for precise retrieval."""

    def __init__(self):
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
        self.spatial_index: Dict[str, list] = {}
        self.relation_store: Dict[str, list] = {}

    def add_frame(self, frame_id: str, objects: list, relations: list):
        """Index a frame's structured objects/relations."""
        self.relation_store[frame_id] = relations
        for r in relations:
            pred = r.get("predicate", "")
            if pred:
                self.predicate_index[pred].add(frame_id)
        spatial = []
        for o in objects:
            label = o.get("label", "")
            score = o.get("score", 0)
            bbox = o.get("bbox") or []
            cx, cy = bbox_center(bbox)
            w, h = _bbox_extent(bbox)
            spatial.append({
                "label": label,
                "score": score,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "region": spatial_region(cx, cy),
            })
        self.spatial_index[frame_id] = spatial

    def query_predicate(self, predicates: list, subject=None, obj=None) -> Set[str]:
        """Return frame_ids matching ANY of the given predicates."""
        result = set()
        for pred in predicates:
            for fid in self.predicate_index.get(pred, set()):
                if subject or obj:
                    rels = self.relation_store.get(fid, [])
                    for r in rels:
                        if r.get("predicate") not in predicates:
                            continue
                        if subject and subject not in r.get("subject","").lower():
                            continue
                        if obj and obj not in r.get("object","").lower():
                            continue
                        result.add(fid)
                        break
                else:
                    result.add(fid)
        return result

    def query_spatial(self, label: str, region: str) -> Set[str]:
        """Return frame_ids where label appears in the given region."""
        result = set()
        for fid, items in self.spatial_index.items():
            for it in items:
                if label.lower() in it["label"].lower() and it["region"] == region:
                    result.add(fid)
                    break
        return result

    def get_object_regions(self, frame_id: str, label: str) -> list:
        """Return regions where a label appears in a frame."""
        regions = []
        for it in self.spatial_index.get(frame_id, []):
            if label.lower() in it["label"].lower():
                regions.append(it["region"])
        return regions

    def to_dict(self) -> dict:
        """Serialize the in-memory index for SQLite persistence."""
        return {
            "predicate_index": {k: sorted(v)
                                for k, v in self.predicate_index.items()},
            "spatial_index": self.spatial_index,
            "relation_store": self.relation_store,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredIndex":
        """Restore the index from a serialized dict."""
        idx = cls()
        idx.predicate_index = defaultdict(
            set, {k: set(v) for k, v in data.get("predicate_index", {}).items()})
        idx.spatial_index = data.get("spatial_index", {})
        idx.relation_store = data.get("relation_store", {})
        return idx


# --- query intent parsing ---

_ACTION_PREDICATES = {
    "hold": ["holding", "carrying"],
    "hold of": ["holding", "carrying"],
    "pick": ["holding", "carrying"],
    "use": ["holding", "using", "touching"],
    "carry": ["carrying", "holding"],
    "wear": ["wearing", "on back of"],
    "sit": ["sitting on"],
    "stand": ["standing on"],
    "lean": ["leaning on"],
    "drive": ["driving", "driving on", "riding"],
    "ride": ["riding", "driving on"],
    "walk": ["walking on"],
    "look": ["looking at"],
    "hang": ["hanging from"],
    "cut": ["holding"],
}

_SPATIAL_PATTERNS = [
    (re.compile(r"\b(left|right|center|top|bottom)\b"), None),
    (re.compile(r"\b(upper-left|upper-right|lower-left|lower-right)\b"), None),
]

_ZH_SPATIAL = {
    "\u5de6\u8fb9": "left", "\u5de6\u4fa7": "left", "\u5de6\u9762": "left",
    "\u53f3\u8fb9": "right", "\u53f3\u4fa7": "right", "\u53f3\u9762": "right",
    "\u4e2d\u95f4": "center", "\u4e2d\u592e": "center",
    "\u4e0a\u9762": "top", "\u4e0a\u65b9": "top",
    "\u4e0b\u9762": "bottom", "\u4e0b\u65b9": "bottom",
}


def extract_query_intent(question_en: str) -> dict:
    """Parse an English query for action-predicate and spatial intent.

    Returns dict with optional keys:
        predicates: list of predicates to match
        spatial_region: region string
        spatial_object: object label for spatial query
    """
    q = question_en.lower()
    result = {}

    for kw, preds in _ACTION_PREDICATES.items():
        if kw in q:
            result["predicates"] = preds
            break

    region = None
    for pat, _ in _SPATIAL_PATTERNS:
        m = pat.search(q)
        if m:
            region = m.group(1)
            break
    if region:
        result["spatial_region"] = region

    return result
