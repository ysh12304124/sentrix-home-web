"""V2.3 Multi-Timescale Sliding Window state analysis.

The module is intentionally detector/state driven and VLM-free.  It accepts
either the existing hybrid WebP records (fair event-only ablation) or a new
low-rate scan produced by ``run_mtsw.py``.  Images are evidence; state changes
decide which images deserve a new keyframe.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


MTSW_METHOD_VERSION = "keyframe-hybrid-v2.3.2-merge"
MTSW_CACHE_VERSION = "mtsw-state-v1"
MTSW_CONFIG = {
    "scan_fps": 2.0,
    "micro_window_sec": 2.0,
    "event_short_window_sec": 3.0,
    "event_reference_window_sec": 10.0,
    "context_max_events": 5,
    "context_max_time_sec": 180.0,
    "history_size": 30,
    "z_threshold": 1.7,
    "strong_z_threshold": 3.0,
    "persistence_frames": 2,
    "dense_before_sec": 1.5,
    "dense_after_sec": 1.5,
    "phash_max_distance": 4,
    "visual_similarity": 0.94,
    "pose_min_persistence_sec": 0.4,
    "event_min_frames": 1,
    "event_soft_max_frames": 6,
    "bridge_enabled": True,
    "bridge_max_interrupt_events": 1,
    "bridge_max_time_gap_sec": 180.0,
    "bridge_threshold": 0.78,
    "visual_weight": 0.15,
    "interaction_weight": 0.25,
    "person_weight": 0.20,
    "object_weight": 0.15,
    "pose_weight": 0.15,
    "appearance_weight": 0.10,
    "event_merge_threshold": 0.20,
    "event_merge_gap_sec": 20.0,
    "event_max_duration_sec": 600.0,
    "person_duplicate_phash_distance": 16,
    "person_duplicate_visual_similarity": 0.88,
    "black_mean_threshold": 24.0,
    "black_p95_threshold": 55.0,
}


def _label(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("label") or value.get("name") or "").strip()
    return ""


def _box(value):
    if isinstance(value, dict):
        value = value.get("bbox") or value.get("box") or []
    try:
        values = [float(item) for item in value]
        return values[:4] if len(values) >= 4 else []
    except (TypeError, ValueError):
        return []


def _confidence(value):
    try:
        return float(value.get("confidence", value.get("conf", 1.0))) if isinstance(value, dict) else 1.0
    except (TypeError, ValueError):
        return 1.0


def _cosine(left, right):
    if left is None or right is None or len(left) != len(right) or not len(left):
        return 0.0
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _jaccard(left, right):
    a, b = set(left or []), set(right or [])
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _phash(image):
    if image is None:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8]
    median = float(np.median(dct[1:, 1:]))
    bits = (dct >= median).astype(np.uint8).reshape(-1)
    return "".join("1" if bit else "0" for bit in bits)


def _phash_distance(left, right):
    if not left or not right or len(left) != len(right):
        return 999
    return sum(a != b for a, b in zip(left, right))


def _sharpness(image):
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _appearance(image):
    if image is None:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    norm = float(np.linalg.norm(hist))
    return (hist / norm).astype(np.float32).tolist() if norm else hist.tolist()


def _color_name(image, bbox):
    if image is None or len(bbox) < 4:
        return "unknown"
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    crop = image[y1 + (y2 - y1) // 5:y1 + (y2 - y1) * 3 // 5, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    hue, saturation, value = hsv
    if value < 45:
        return "black"
    if saturation < 35 and value > 185:
        return "white"
    if saturation < 40:
        return "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "brown"


@dataclass
class FrameState:
    frame_index: int
    timestamp: float
    image_path: str = ""
    image: object | None = None
    objects: list = field(default_factory=list)
    people: list = field(default_factory=list)
    pose_state: str = "unknown"
    interactions: list = field(default_factory=list)
    scene_embedding: list = field(default_factory=list)
    appearance_embedding: list = field(default_factory=list)
    sharpness: float = 0.0
    black_frame: bool = False
    phash: str = ""
    change_score: float = 0.0
    z_score: float = 0.0
    boundary: bool = False
    strong_boundary: bool = False
    selection_reason: list = field(default_factory=list)
    selected: bool = False
    duplicate_of: str | None = None

    @property
    def object_labels(self):
        return [_label(item) for item in self.objects if _label(item)]

    @property
    def person_count(self):
        return len(self.people) if self.people else sum(1 for item in self.objects if _label(item) == "person")

    @property
    def state_signature(self):
        interactions = tuple(sorted(str(item.get("relation") or item) for item in self.interactions))
        return (tuple(sorted(self.object_labels)), self.person_count, self.pose_state, interactions)


def build_interactions(people, objects, previous=None):
    relations = []
    people = list(people or [])
    objects = [item for item in objects or [] if _label(item) != "person"]
    for person in people:
        pb = _box(person)
        if len(pb) < 4:
            continue
        pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        pscale = max(1.0, math.hypot(pb[2] - pb[0], pb[3] - pb[1]))
        for obj in objects:
            ob = _box(obj)
            if len(ob) < 4:
                continue
            ocx, ocy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
            distance = math.hypot(pcx - ocx, pcy - ocy) / pscale
            relation = "far" if distance > 2.0 else "near" if distance > 0.9 else "touching_candidate"
            relations.append({"object": _label(obj), "relation": relation, "distance": round(distance, 4), "confidence": round(_confidence(obj), 4)})
    return relations


def state_from_record(record, index=0):
    raw = getattr(record, "raw", {}) or {}
    objects = list(getattr(record, "objects", []) or raw.get("objects") or [])
    people = [item for item in objects if _label(item) == "person"]
    return FrameState(
        frame_index=int(raw.get("source_frame_index") or raw.get("frame_index") or index),
        timestamp=float(getattr(record, "timestamp", raw.get("timestamp_sec") or 0)),
        image_path=str(getattr(record, "image_path", "") or ""), objects=objects, people=people,
        pose_state=str(getattr(record, "pose_summary", "") or raw.get("pose_state") or "unknown"),
        scene_embedding=list(getattr(record, "visual_embedding", []) or []),
        appearance_embedding=list(raw.get("appearance_embedding") or []),
        sharpness=float(getattr(record, "sharpness", 0.0) or 0),
    )


class MTSWEngine:
    def __init__(self, config=None):
        self.config = {**MTSW_CONFIG, **(config or {})}

    def _prepare(self, states):
        for index, state in enumerate(states):
            if not state.interactions:
                state.interactions = build_interactions(state.people, state.objects, states[index - 1] if index else None)
            if state.image is not None:
                if not state.appearance_embedding:
                    state.appearance_embedding = _appearance(state.image)
                if not state.phash:
                    state.phash = _phash(state.image)
                if not state.sharpness:
                    state.sharpness = _sharpness(state.image)
                gray = cv2.cvtColor(state.image, cv2.COLOR_BGR2GRAY)
                state.black_frame = float(gray.mean()) < float(self.config["black_mean_threshold"]) and float(np.percentile(gray, 95)) < float(self.config["black_p95_threshold"])

    def _difference(self, previous, current):
        previous_interactions = {(item.get("object"), item.get("relation")) for item in previous.interactions}
        current_interactions = {(item.get("object"), item.get("relation")) for item in current.interactions}
        interaction = 1.0 - _jaccard(previous_interactions, current_interactions)
        person = min(1.0, abs(previous.person_count - current.person_count) / max(1, previous.person_count, current.person_count))
        object_change = 1.0 - _jaccard(previous.object_labels, current.object_labels)
        pose = 0.0 if previous.pose_state == current.pose_state else 1.0
        scene = 1.0 - _cosine(previous.scene_embedding, current.scene_embedding) if previous.scene_embedding and current.scene_embedding else 0.0
        appearance = 1.0 - _cosine(previous.appearance_embedding, current.appearance_embedding) if previous.appearance_embedding and current.appearance_embedding else 0.0
        values = {"interaction": interaction, "person": person, "object": object_change, "pose": pose, "scene": scene, "appearance": appearance}
        weights = {"scene": "visual_weight", "appearance": "appearance_weight", "interaction": "interaction_weight", "person": "person_weight", "object": "object_weight", "pose": "pose_weight"}
        score = sum(values[key] * self.config[weights[key]] for key in values)
        return round(float(score), 6), values

    def analyze(self, states):
        started = time.perf_counter()
        states = list(states)
        self._prepare(states)
        history = deque(maxlen=max(3, int(self.config["history_size"])))
        persistence = 0
        all_transitions = []
        for index, state in enumerate(states):
            if index == 0:
                history.append(0.0)
                continue
            score, breakdown = self._difference(states[index - 1], state)
            state.change_score = score
            state.change_breakdown = breakdown
            values = np.asarray(history, dtype=np.float32)
            mean, std = float(values.mean()) if len(values) else 0.0, float(values.std()) if len(values) else 0.0
            state.z_score = round((score - mean) / (std + 1e-6), 6)
            is_strong = state.z_score >= float(self.config["strong_z_threshold"])
            if state.z_score >= float(self.config["z_threshold"]):
                persistence += 1
            else:
                persistence = 0
            state.strong_boundary = is_strong
            state.boundary = is_strong or persistence >= int(self.config["persistence_frames"])
            if state.boundary:
                reasons = [key + "_change" for key, value in breakdown.items() if value >= 0.35]
                state.selection_reason = reasons or ["adaptive_state_change"]
                all_transitions.append({"frame_index": state.frame_index, "timestamp": state.timestamp, "score": score, "z_score": state.z_score, "reasons": state.selection_reason, "breakdown": breakdown})
                persistence = 0
            history.append(score)

        selected = self.select_state_frames(states, all_transitions)
        selected, dedup_cases = self.deduplicate(selected)
        events, bridge_cases = self.build_events(selected)
        metrics = self.metrics(states, selected, events, all_transitions, dedup_cases, bridge_cases)
        metrics["analysis_wall_seconds"] = round(time.perf_counter() - started, 4)
        return {"states": states, "transitions": all_transitions, "selected": selected, "events": events, "dedup_cases": dedup_cases, "bridge_cases": bridge_cases, "metrics": metrics}

    def select_state_frames(self, states, transitions):
        if not states:
            return []
        by_index = {state.frame_index: state for state in states}
        selected = []
        for transition in transitions:
            state = by_index.get(transition["frame_index"])
            if not state:
                continue
            for candidate in (state if not state.black_frame else None, self._nearest_usable(states, state.timestamp - self.config["dense_before_sec"]), self._nearest_usable(states, state.timestamp + self.config["dense_after_sec"]), self._nearest_usable(states, state.timestamp)):
                if candidate and candidate not in selected:
                    candidate.selected = True
                    if candidate is not state and not candidate.selection_reason:
                        candidate.selection_reason = ["pre_state" if candidate.timestamp < state.timestamp else "post_state"]
                    selected.append(candidate)
        if not selected:
            anchor = max([item for item in states if not item.black_frame] or states, key=lambda item: item.sharpness)
            anchor.selected, anchor.selection_reason = True, ["initial_state"]
            selected = [anchor]
        return sorted(selected, key=lambda item: item.timestamp)

    @staticmethod
    def _nearest(states, timestamp):
        return min(states, key=lambda item: abs(item.timestamp - timestamp)) if states else None

    @staticmethod
    def _nearest_usable(states, timestamp):
        usable = [item for item in states if not item.black_frame and abs(item.timestamp - timestamp) <= 3.0]
        return min(usable, key=lambda item: abs(item.timestamp - timestamp)) if usable else None

    def deduplicate(self, selected):
        kept, cases = [], []
        for state in selected:
            duplicate = next((item for item in reversed(kept) if abs(item.timestamp - state.timestamp) <= 8.0 and ((_phash_distance(item.phash, state.phash) <= int(self.config["phash_max_distance"]) or _cosine(item.scene_embedding, state.scene_embedding) >= float(self.config["visual_similarity"])) and item.state_signature == state.state_signature or (item.person_count > 0 and item.person_count == state.person_count and item.pose_state == state.pose_state and item.interactions == state.interactions and _phash_distance(item.phash, state.phash) <= int(self.config["person_duplicate_phash_distance"]) and _cosine(item.appearance_embedding, state.appearance_embedding) >= float(self.config["person_duplicate_visual_similarity"])))), None)
            if duplicate:
                if state.sharpness > duplicate.sharpness:
                    duplicate.duplicate_of = state.duplicate_of or duplicate.frame_index
                    kept.remove(duplicate); kept.append(state)
                    cases.append({"kept": state.frame_index, "removed": duplicate.frame_index, "reason": "person_duplicate_replaced_by_sharper" if state.person_count else "state_duplicate_replaced_by_sharper"})
                else:
                    state.selected = False; state.duplicate_of = str(duplicate.frame_index)
                    cases.append({"kept": duplicate.frame_index, "removed": state.frame_index, "reason": "person_duplicate" if state.person_count else "state_duplicate"})
                continue
            kept.append(state)
        return sorted(kept, key=lambda item: item.timestamp), cases

    def build_events(self, selected):
        if not selected:
            return [], []
        events = [{"event_index": 0, "members": [selected[0]], "states": [selected[0].state_signature]}]
        for state in selected[1:]:
            current = events[-1]
            previous = current["members"][-1]
            gap = state.timestamp - previous.timestamp
            continuity = 0.25 * _jaccard(previous.object_labels, state.object_labels) + 0.40 * (1.0 if previous.person_count == state.person_count else 0.0) + 0.35 * _cosine(previous.appearance_embedding, state.appearance_embedding)
            if gap <= float(self.config["event_merge_gap_sec"]) and continuity >= float(self.config["event_merge_threshold"]) and (state.timestamp - current["members"][0].timestamp) <= float(self.config["event_max_duration_sec"]):
                current["members"].append(state); current["states"].append(state.state_signature)
            else:
                events.append({"event_index": len(events), "members": [state], "states": [state.state_signature]})
        bridge_cases = []
        if self.config.get("bridge_enabled") and len(events) >= 3:
            for left, middle, right in zip(events, events[1:], events[2:]):
                left_last, right_first = left["members"][-1], right["members"][0]
                score = 0.55 * _cosine(left_last.appearance_embedding, right_first.appearance_embedding) + 0.25 * (1.0 if left_last.person_count == right_first.person_count else 0.0) + 0.20 * _jaccard(left_last.object_labels, right_first.object_labels)
                accepted = score >= float(self.config["bridge_threshold"]) and (right_first.timestamp - left_last.timestamp) <= float(self.config["bridge_max_time_gap_sec"])
                bridge_cases.append({"left": left["event_index"], "interruption": middle["event_index"], "right": right["event_index"], "score": round(score, 6), "accepted": accepted})
                if accepted:
                    left.setdefault("interruptions", []).append(middle["event_index"])
                    left["members"].extend(right["members"]); left["states"].extend(right["states"])
                    right["merged_into"] = left["event_index"]
        return [event for event in events if "merged_into" not in event], bridge_cases

    def metrics(self, states, selected, events, transitions, dedup_cases, bridge_cases):
        reasons = {key: 0 for key in ["state", "interaction", "scene", "person", "object", "pose"]}
        for item in selected:
            for reason in item.selection_reason:
                key = reason.split("_")[0]
                if key in reasons:
                    reasons[key] += 1
        transition_indices = {item["frame_index"] for item in transitions}
        selected_indices = {item.frame_index for item in selected}
        return {
            "scan_frames": len(states), "micro_states": len(states), "state_transitions": len(transitions),
            "candidate_frames": len(transitions) * 3, "final_keyframes": len(selected), "events": len(events),
            "duplicate_removed": len(dedup_cases), "state_change_keyframes": reasons["state"],
            "interaction_change_keyframes": reasons["interaction"], "scene_change_keyframes": reasons["scene"],
            "person_change_keyframes": reasons["person"], "object_change_keyframes": reasons["object"], "pose_change_keyframes": reasons["pose"],
            "state_coverage": round(len(transition_indices & selected_indices) / max(1, len(transition_indices)), 6),
            "interaction_coverage": round(sum(1 for item in transitions if "interaction_change" in item["reasons"] and item["frame_index"] in selected_indices) / max(1, sum(1 for item in transitions if "interaction_change" in item["reasons"])), 6),
            "duplicate_rate": round(len(dedup_cases) / max(1, len(dedup_cases) + len(selected)), 6),
            "singleton_rate": round(sum(1 for event in events if len(event["members"]) == 1) / max(1, len(events)), 6),
            "mean_states_per_event": round(sum(len(event["members"]) for event in events) / max(1, len(events)), 6),
            "bridge_candidates": len(bridge_cases), "bridge_accepted": sum(1 for item in bridge_cases if item["accepted"]),
            "evidence_retention": 1.0,
            "information_density": round(sum(1 for item in selected if item.selection_reason) / max(1, len(selected)), 6),
            "black_frames_in_scan": sum(1 for item in states if item.black_frame),
            "black_frames_selected": sum(1 for item in selected if item.black_frame),
            "person_duplicates_removed": sum(1 for item in dedup_cases if "person_duplicate" in item["reason"]),
        }


def records_from_images(image_records):
    """Build states from lightweight scan/package records."""
    states = []
    for index, item in enumerate(image_records):
        if isinstance(item, FrameState):
            states.append(item)
            continue
        states.append(state_from_record(item, index))
    return states
