"""
Keyframe Memory Builder

Replaces the conversation-based MemoryBuilder logic: instead of building
EVENT nodes from dialogue turns, it builds FRAME nodes from Qdrant
keyframes (ego4d_nlq_frames) and constructs the same four edge families
(temporal / semantic / entity / causal) plus hierarchy edges
(FRAME -> CLIP -> VIDEO).

The original MAGMA node types are reused:
    EVENT  node  <- one keyframe            (was: one dialogue turn)
    SESSION node <- one video clip          (was: one conversation session)
    EPISODE node <- one full video          (was: optional episode grouping)
    ENTITY  node <- one high-frequency object (new: hub to avoid edge blow-up)
"""

import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .graph_db import (
    NetworkXGraphDB,
    EventNode,
    EpisodeNode,
    SessionNode,
    Link,
    NodeType,
    LinkType,
)
from .keyword_enrichment import tokenize_text, contains_cjk
from .structured_index import filter_objects, filter_relations, StructuredIndex

logger = logging.getLogger(__name__)


class KeyframeMemoryBuilder:
    """Build a TRG-style graph from Qdrant keyframe data."""

    IMAGE_VECTOR_NAME = "image"

    # Frames-per-object threshold: above this an object becomes a hub node
    HUB_OBJECT_THRESHOLD = max(2, int(os.getenv("OBJECT_HUB_THRESHOLD", "30")))
    # Frames-per-relation threshold: above this a relation gets its own hub
    # node (EVENT -> hub), avoiding all-pairs SHARES_RELATION blow-up while
    # keeping every pair of frames sharing the relation reachable in 2 hops.
    RELATION_HUB_THRESHOLD = max(2, int(os.getenv("RELATION_HUB_THRESHOLD", "30")))
    # Only the most salient object detections get entity-hub edges.  All
    # labels remain on the EVENT node and keyword/structured indexes; this cap
    # only removes low-information graph edges.
    OBJECT_HUB_MAX_PER_FRAME = max(1, int(os.getenv("OBJECT_HUB_MAX_PER_FRAME", "4")))
    # Ubiquitous/background labels carry almost no retrieval signal but create
    # hundreds of thousands of hub edges in full-scale builds.
    NON_INFORMATIVE_OBJECT_HUB_LABELS = frozenset({
        "person", "people", "human", "hand", "body", "camera",
        "floor", "wall", "ceiling", "sky", "ground", "pavement", "road",
        "floor-wood", "wall-tile", "ceiling-tile", "door", "window",
    })
    VISUAL_TOP_K = max(1, int(os.getenv("VISUAL_SIMILAR_TOP_K", "3")))
    VISUAL_MIN_SCORE = float(os.getenv("VISUAL_SIMILAR_MIN_SCORE", "0.80"))
    VISUAL_MUTUAL = os.getenv("VISUAL_SIMILAR_MUTUAL", "1").lower() not in {
        "0", "false", "no", "off"
    }
    VISUAL_MATRIX_CHUNK = max(64, int(os.getenv("VISUAL_MATRIX_CHUNK", "256")))

    def __init__(self, qdrant_adapter=None, graph_db=None, enable_causal_edges=None):
        self.qa = qdrant_adapter
        self.graph_db = graph_db or NetworkXGraphDB()
        if enable_causal_edges is None:
            enable_causal_edges = os.getenv("GRAPH_CAUSAL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.enable_causal_edges = bool(enable_causal_edges)
        self.node_index = {}
        # track ids for batch link creation
        self.frame_ids = []
        self.clip_nodes = {}
        self.video_nodes = {}
        self.clip_to_video = {}
        self.object_hubs = {}
        self.relation_hubs = {}
        # frame -> set(object_label) and frame -> set(relation_label) for edges
        self.frame_objects = {}
        self.frame_relations = {}
        self.frame_vectors = {}
        self.structured_index = StructuredIndex()
        self.label_matcher = None

    # ------------------------------------------------------------------ #
    #  Dataset compatibility
    # ------------------------------------------------------------------ #

    WORLDMM_FRAME_WIDTH = 1440
    WORLDMM_FRAME_HEIGHT = 1080
    TASK_PRIORITY = {"moments+nlq": 0, "nlq": 1, "moments": 2, "vq": 3}

    @classmethod
    def _normalise_bbox(cls, bbox):
        if not bbox or len(bbox) < 4:
            return bbox
        values = [float(v) for v in bbox[:4]]
        if all(0.0 <= v <= 1.5 for v in values):
            return values
        return [
            max(0.0, values[0] / cls.WORLDMM_FRAME_WIDTH),
            max(0.0, values[1] / cls.WORLDMM_FRAME_HEIGHT),
            min(1.0, values[2] / cls.WORLDMM_FRAME_WIDTH),
            min(1.0, values[3] / cls.WORLDMM_FRAME_HEIGHT),
        ]

    @classmethod
    def _normalise_objects(cls, raw_objects):
        """Unify WorldMM pixel boxes and YOLO normalized boxes."""
        out = []
        for obj in raw_objects or []:
            item = dict(obj)
            if item.get("bbox"):
                item["bbox"] = cls._normalise_bbox(item["bbox"])
            out.append(item)
        return out

    @staticmethod
    def _remap_dataset_path(value, kind):
        """Map paths embedded in the shared Qdrant payload to this machine."""
        if not value:
            return value
        value = str(value)
        if kind == "clip":
            match = re.search(r"(?:^|[\\/])v2[\\/]clips[\\/](.+)$", value, flags=re.I)
            root = os.getenv("EGO4D_CLIPS_ROOT", r"D:\RuiMeng\ego4d_QA\v2\clips")
            relative = match.group(1) if match else Path(value).name
        else:
            match = re.search(
                r"(?:^|[\\/])data[\\/](frames|worldmm_output|cache)[\\/](.+)$",
                value, flags=re.I)
            root = os.getenv("EGO4D_DATA_ROOT", r"D:\RuiMeng\memory_rag\new_memory\ego4d\data")
            relative = (match.group(1) + "/" + match.group(2)) if match else Path(value).name
        return str(Path(root).joinpath(*re.split(r"[\\/]+", relative)))

    @staticmethod
    def _label_counts(rows, key="label", limit=8):
        counts = defaultdict(int)
        for row in rows or []:
            label = str((row or {}).get(key) or "").strip()
            if label:
                counts[label] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]

    @classmethod
    def _deduplicate_frame_points(cls, frame_points):
        """Keep one graph EVENT per physical frame_uid.

        The collection currently has unique frame_uid values, but this guard
        also supports task-partitioned copies without duplicating graph nodes.
        """
        groups = {}
        for point in frame_points:
            payload = point.get("payload") or {}
            key = payload.get("frame_uid") or str(point.get("id"))
            groups.setdefault(key, []).append(point)

        out, duplicate_points = [], 0
        for points in groups.values():
            if len(points) > 1:
                duplicate_points += len(points) - 1
                points.sort(key=lambda p: (
                    cls.TASK_PRIORITY.get((p.get("payload") or {}).get("task"), 99),
                    str(p.get("id"))))
            canonical = dict(points[0])
            if len(points) > 1:
                payload = dict(canonical.get("payload") or {})
                tasks, query_counts = set(payload.get("tasks") or []), dict(payload.get("query_counts_by_task") or {})
                for other in points[1:]:
                    op = other.get("payload") or {}
                    tasks.update(op.get("tasks") or [])
                    for task, count in (op.get("query_counts_by_task") or {}).items():
                        query_counts[task] = max(query_counts.get(task, 0), count)
                payload["tasks"] = sorted(tasks)
                payload["query_counts_by_task"] = query_counts
                canonical["payload"] = payload
            out.append(canonical)
        return out, len(frame_points), duplicate_points

    # ------------------------------------------------------------------ #
    #  Node creation
    # ------------------------------------------------------------------ #

    def _frame_content_narrative(self, payload: dict) -> str:
        """Compose the text that gets indexed/searched for a frame."""
        parts = []
        narr = (payload.get("narration_text") or "").strip()
        speech = (payload.get("speech_text") or "").strip()
        action = (payload.get("action_text") or "").strip()
        if narr:
            parts.append(narr)
        if speech:
            parts.append(speech)
        if action:
            parts.append("Action: " + action)

        rels = payload.get("relation_labels") or []
        if rels:
            parts.append("Scene: " + "; ".join(rels[:6]))

        obj_counts = payload.get("object_counts") or {}
        if obj_counts:
            obj_str = ", ".join("%s x%d" % (k, v) for k, v in list(obj_counts.items())[:8])
            parts.append("Objects: " + obj_str)

        event_title = str(payload.get("event_title") or "").strip()
        event_summary = str(payload.get("event_summary") or "").strip()
        if event_title:
            parts.append("Event: " + event_title)
        if event_summary:
            parts.append("Event summary: " + event_summary)

        clip_t = payload.get("clip_time_sec")
        vid_t = payload.get("video_time_sec")
        clip = (payload.get("clip_uid") or "")[:8]
        loc = "Location: clip %s @ %ss / video @ %ss" % (clip, clip_t, vid_t)
        parts.append(loc)
        # Wall-clock capture time and human-readable place are separate from
        # in-video timestamps.  Put both on the searchable narrative so graph
        # anchors can answer date/location questions instead of relying only on
        # the generic clip location string.
        captured = str(payload.get("captured_at") or "").strip()
        place = str(payload.get("place") or "").strip()
        if captured:
            parts.append("Captured: " + captured)
        if place:
            parts.append("Place: " + place)
        return "\n".join(parts)

    def create_frame_node(self, point: dict) -> EventNode:
        """One deduplicated physical keyframe -> one EVENT node."""
        payload = point.get("payload") or {}
        point_id = point.get("id")
        vectors = point.get("vector") or {}
        image_vec = vectors.get(self.IMAGE_VECTOR_NAME) or vectors.get("visual")

        clip_t = payload.get("clip_time_sec") or 0.0
        video_t = payload.get("video_time_sec")
        timeline_t = video_t if video_t is not None else clip_t
        ts = datetime(2000, 1, 1) + timedelta(seconds=float(timeline_t or 0.0))

        raw_objects = self._normalise_objects(payload.get("objects"))
        raw_relations = payload.get("relations") or []
        object_labels, filtered_objects = filter_objects(raw_objects)
        relation_labels, filtered_relations = filter_relations(raw_relations)

        frame_path = self._remap_dataset_path(payload.get("frame_path"), "frame")
        clip_path = self._remap_dataset_path(payload.get("clip_path"), "clip")
        worldmm_dir = self._remap_dataset_path(payload.get("worldmm_output_dir"), "worldmm_output")
        track_keys = [
            "%s:%s:%s" % (payload.get("clip_uid"), obj.get("track_id"), obj.get("label"))
            for obj in filtered_objects if obj.get("track_id") is not None
        ]
        action_labels = [str(a.get("label") or "").strip()
                         for a in payload.get("worldmm_actions") or [] if a.get("label")]
        expression_labels = [str(a.get("label") or "").strip()
                             for a in payload.get("worldmm_expressions") or [] if a.get("label")]

        node = EventNode(
            node_id=str(point_id),
            node_type=NodeType.EVENT,
            timestamp=ts,
            content_narrative=self._frame_content_narrative(payload),
            attributes={
                "qdrant_id": str(point_id),
                "clip_uid": payload.get("clip_uid"),
                "video_uid": payload.get("video_uid"),
                "frame_uid": payload.get("frame_uid"),
                "event_id": payload.get("event_id"),
                "event_summary": payload.get("event_summary") or "",
                "event_title": payload.get("event_title") or "",
                "source_asset_id": payload.get("source_asset_id") or str(point_id),
                "source_scene_index": payload.get("source_scene_index"),
                "frame_path": frame_path,
                "source_frame_path": payload.get("frame_path"),
                "clip_path": clip_path,
                "source_clip_path": payload.get("clip_path"),
                "source_video_file_name": payload.get("source_video_file_name") or "",
                "point_id": payload.get("point_id") or str(point_id),
                "frame_idx": payload.get("frame_idx"),
                "frame_seq": payload.get("frame_seq"),
                "clip_time_sec": clip_t,
                "video_time_sec": payload.get("video_time_sec"),
                "captured_at": payload.get("captured_at") or "",
                "place": payload.get("place") or "",
                "captured_location": payload.get("captured_location") or "",
                "person_ids": list(payload.get("person_ids") or []),
                "person_labels": list(payload.get("person_labels") or []),
                "entity_refs": list(payload.get("entity_refs") or []),
                "sample_fps": payload.get("sample_fps"),
                "clip_video_start_sec": payload.get("clip_video_start_sec"),
                "clip_video_end_sec": payload.get("clip_video_end_sec"),
                "clip_start_sec": payload.get("clip_start_sec"),
                "clip_end_sec": payload.get("clip_end_sec"),
                "clip_fps": payload.get("clip_fps"),
                "clip_num_frames": payload.get("clip_num_frames"),
                "level": payload.get("level"),
                "vector_names": payload.get("vector_names") or [],
                "entities": object_labels,
                "object_labels": object_labels,
                "object_counts": payload.get("object_counts") or {},
                "relation_labels": relation_labels,
                "relation_counts": payload.get("relation_counts") or {},
                "objects": filtered_objects,
                "object_track_keys": list(dict.fromkeys(track_keys)),
                "relations": filtered_relations,
                "narration_text": payload.get("narration_text") or "",
                "action_text": payload.get("action_text") or "",
                "speech_text": payload.get("speech_text") or "",
                "emotion_text": payload.get("emotion_text") or "",
                "event_text": payload.get("event_text") or "",
                "face_text": payload.get("face_text") or "",
                "object_text": payload.get("object_text") or "",
                "relation_text": payload.get("relation_text") or "",
                "visual_text": payload.get("visual_text") or "",
                "keyframe_source": payload.get("keyframe_source"),
                "worldmm_keyframe_code": payload.get("worldmm_keyframe_code"),
                "worldmm_selection_reason": payload.get("worldmm_selection_reason"),
                "worldmm_information_gain": payload.get("worldmm_information_gain"),
                "worldmm_boundary_score": payload.get("worldmm_boundary_score"),
                "worldmm_summary_score": payload.get("worldmm_summary_score"),
                "worldmm_actions": payload.get("worldmm_actions") or [],
                "worldmm_action_labels": list(dict.fromkeys(action_labels)),
                "worldmm_expressions": payload.get("worldmm_expressions") or [],
                "worldmm_expression_labels": list(dict.fromkeys(expression_labels)),
                "worldmm_output_dir": worldmm_dir,
                "source_worldmm_output_dir": payload.get("worldmm_output_dir"),
                "sgg_proposal_source": payload.get("sgg_proposal_source"),
                "sgg_proposal_stats": payload.get("sgg_proposal_stats") or {},
                "dataset": payload.get("dataset"),
                "version": payload.get("version"),
                "task": payload.get("task"),
                "tasks": payload.get("tasks") or ([] if not payload.get("task") else [payload.get("task")]),
                "query_counts_by_task": payload.get("query_counts_by_task") or {},
                "split": payload.get("split"),
                "snapshot_name": payload.get("snapshot_name"),
                "scope_id": payload.get("scope_id") or "home-default",
            },
            embedding_vector=image_vec,
        )
        self.frame_vectors[node.node_id] = image_vec
        self.structured_index.add_frame(str(point_id), filtered_objects, filtered_relations)
        return node
    def create_clip_node(self, clip_uid: str, frame_points: list) -> SessionNode:
        """Aggregate the physical frames of a clip into a SESSION node."""
        payloads = [fp["payload"] for fp in frame_points]
        obj_counter, rel_counter, action_counter = defaultdict(int), defaultdict(int), defaultdict(int)
        expression_counter, reason_counter = defaultdict(int), defaultdict(int)
        info_gains, tasks, query_counts = [], set(), {}
        for p in payloads:
            for o in p.get("object_labels") or []:
                obj_counter[o] += 1
            for r in p.get("relation_labels") or []:
                rel_counter[r] += 1
            for action in p.get("worldmm_actions") or []:
                label = str((action or {}).get("label") or "").strip()
                if label:
                    action_counter[label] += 1
            for expression in p.get("worldmm_expressions") or []:
                label = str((expression or {}).get("label") or "").strip()
                if label:
                    expression_counter[label] += 1
            reason = str(p.get("worldmm_selection_reason") or "").strip()
            if reason:
                reason_counter[reason] += 1
            if p.get("worldmm_information_gain") is not None:
                try:
                    info_gains.append(float(p["worldmm_information_gain"]))
                except (TypeError, ValueError):
                    pass
            tasks.update(p.get("tasks") or ([] if not p.get("task") else [p.get("task")]))
            for task, count in (p.get("query_counts_by_task") or {}).items():
                query_counts[task] = max(query_counts.get(task, 0), count)

        top_objects = sorted(obj_counter.items(), key=lambda x: -x[1])[:8]
        top_relations = sorted(rel_counter.items(), key=lambda x: -x[1])[:8]
        top_actions = sorted(action_counter.items(), key=lambda x: -x[1])[:6]
        top_expressions = sorted(expression_counter.items(), key=lambda x: -x[1])[:6]
        top_reasons = sorted(reason_counter.items(), key=lambda x: -x[1])[:4]
        video_uid = payloads[0].get("video_uid") if payloads else None
        times = [p.get("clip_time_sec") or 0 for p in payloads]
        avg_gain = sum(info_gains) / len(info_gains) if info_gains else None

        summary = "Clip %s (%d frames). Top objects: %s. Top relations: %s.%s" % (
            clip_uid[:8], len(payloads),
            ", ".join("%s(%d)" % x for x in top_objects),
            ", ".join("%s(%d)" % x for x in top_relations),
            (" Top actions: %s." % ", ".join("%s(%d)" % x for x in top_actions)) if top_actions else "",
        )

        node = SessionNode(
            node_id="clip_" + clip_uid,
            session_id=0,
            summary=summary,
            date_time="t=%.0f-%.0f" % (min(times), max(times)),
            attributes={
                "clip_uid": clip_uid,
                "video_uid": video_uid,
                "frame_count": len(payloads),
                "top_objects": top_objects,
                "top_relations": top_relations,
                "top_actions": top_actions,
                "top_expressions": top_expressions,
                "selection_reasons": top_reasons,
                "avg_worldmm_information_gain": avg_gain,
                "tasks": sorted(tasks),
                "query_counts_by_task": query_counts,
                "time_range": (min(times), max(times)),
                "clip_path": self._remap_dataset_path(payloads[0].get("clip_path"), "clip") if payloads else None,
            },
            embedding_vector=None,
        )
        return node

    def create_video_node(self, video_uid: str, clip_ids: list, frame_count: int,
                          payloads: list | None = None) -> EpisodeNode:
        payloads = payloads or []
        action_counter, reason_counter = defaultdict(int), defaultdict(int)
        info_gains, tasks = [], set()
        for p in payloads:
            for action in p.get("worldmm_actions") or []:
                label = str((action or {}).get("label") or "").strip()
                if label:
                    action_counter[label] += 1
            reason = str(p.get("worldmm_selection_reason") or "").strip()
            if reason:
                reason_counter[reason] += 1
            if p.get("worldmm_information_gain") is not None:
                try:
                    info_gains.append(float(p["worldmm_information_gain"]))
                except (TypeError, ValueError):
                    pass
            tasks.update(p.get("tasks") or ([] if not p.get("task") else [p.get("task")]))
        top_actions = sorted(action_counter.items(), key=lambda x: -x[1])[:8]
        top_reasons = sorted(reason_counter.items(), key=lambda x: -x[1])[:5]
        avg_gain = sum(info_gains) / len(info_gains) if info_gains else None
        summary = "Video %s: %d clips, %d frames.%s" % (
            video_uid[:8], len(clip_ids), frame_count,
            (" Top actions: %s." % ", ".join("%s(%d)" % x for x in top_actions)) if top_actions else "",
        )
        return EpisodeNode(
            node_id="video_" + video_uid,
            title=video_uid[:8],
            summary=summary,
            start_timestamp=None,
            end_timestamp=None,
            event_count=frame_count,
            attributes={
                "video_uid": video_uid,
                "clip_ids": clip_ids,
                "tasks": sorted(tasks),
                "top_actions": top_actions,
                "selection_reasons": top_reasons,
                "avg_worldmm_information_gain": avg_gain,
            },
            embedding_vector=None,
        )
    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #

    def index_frame(self, event_id, content, attributes):
        """Build a keyword index (narrative + identity/context fields).

        ``tokenize_text`` caps long narratives at 128 tokens; dates/places are
        appended near the end of the narrative and were therefore frequently
        truncated.  Index the compact context fields separately so a location
        or capture-date anchor is never lost to the token budget.
        """
        for text in [content, attributes.get("place"), attributes.get("captured_at"),
                     attributes.get("event_title"), attributes.get("event_summary")]:
            if not text:
                continue
            for tok in tokenize_text(text):
                tok = tok.strip(".,!?;:\"\x27")
                if len(tok) >= 2 or contains_cjk(tok):
                    self.node_index.setdefault(tok, set()).add(event_id)
        for o in attributes.get("object_labels") or []:
            for tok in tokenize_text(o):
                tok = tok.strip(".,!?;:\"\x27")
                if tok:
                    self.node_index.setdefault(tok, set()).add(event_id)
        for r in attributes.get("relation_labels") or []:
            for tok in tokenize_text(r):
                tok = tok.strip(".,!?;:\"\x27")
                if tok:
                    self.node_index.setdefault(tok, set()).add(event_id)

    @classmethod
    def _salient_object_labels(cls, objects: list, labels: list,
                               max_labels: int) -> list:
        """Select graph-worthy object labels using detection salience.

        EVENT attributes keep every label.  This selection only limits
        redundant ENTITY hub edges, so rare/weak detections can still be found
        by keyword and structured retrieval without bloating graph traversal.
        """
        label_set = {str(x).lower() for x in labels or []}
        label_set -= cls.NON_INFORMATIVE_OBJECT_HUB_LABELS
        if not label_set:
            return []

        max_scores = defaultdict(float)
        counts = defaultdict(int)
        for obj in objects or []:
            label = str((obj or {}).get("label") or "").lower().strip()
            if label not in label_set:
                continue
            counts[label] += 1
            try:
                max_scores[label] = max(max_scores[label], float(obj.get("score") or 0.0))
            except (TypeError, ValueError):
                pass
        ranked = sorted(
            label_set,
            key=lambda label: (-max_scores.get(label, 0.0), -counts.get(label, 0), label),
        )
        return ranked[:max_labels]

    # ------------------------------------------------------------------ #
    #  Edge creation
    # ------------------------------------------------------------------ #

    def create_temporal_links(self, frame_ids, bidirectional=False):
        """Adjacent frames within a clip: TIME_PRECEDES.

        Only the forward edge (A -> B) is created by default. The reverse
        direction is derivable at query time from the incoming PRECEDES
        edge, so TIME_SUCCEEDS (a duplicate reverse copy) is skipped to
        halve the temporal edge count.
        """
        created = 0
        for i in range(len(frame_ids) - 1):
            src, tgt = frame_ids[i], frame_ids[i + 1]
            src_node = self.graph_db.get_node(src)
            tgt_node = self.graph_db.get_node(tgt)
            same_clip = (
                src_node and tgt_node
                and src_node.attributes.get("clip_uid") == tgt_node.attributes.get("clip_uid")
            )
            if not same_clip:
                continue
            self.graph_db.add_link(Link(
                source_node_id=src, target_node_id=tgt,
                link_type=LinkType.TEMPORAL,
                properties={"sub_type": "TIME_PRECEDES", "sequence_index": i,
                            "confidence": 1.0, "evidence_tier": "confirmed"},
            ))
            created += 1
            if bidirectional:
                self.graph_db.add_link(Link(
                    source_node_id=tgt, target_node_id=src,
                    link_type=LinkType.TEMPORAL,
                    properties={"sub_type": "TIME_SUCCEEDS", "sequence_index": i,
                                "confidence": 1.0, "evidence_tier": "confirmed"},
                ))
                created += 1
        return created

    def create_video_temporal_links(self, frame_ids):
        """Connect adjacent events across clip boundaries within one video.

        TIME_PRECEDES used to stop at ``clip_uid`` boundaries, which left a
        long video as a set of disconnected scene islands.  We add only the
        boundary links here; same-clip adjacency is still handled by
        :meth:`create_temporal_links`.
        """
        created = 0
        ordered = []
        for fid in frame_ids:
            node = self.graph_db.get_node(fid)
            if not node:
                continue
            attrs = node.attributes or {}
            try:
                t = float(attrs.get("video_time_sec") if attrs.get("video_time_sec") is not None
                          else attrs.get("clip_time_sec") or 0.0)
            except (TypeError, ValueError):
                t = 0.0
            ordered.append((str(attrs.get("video_uid") or ""), t,
                            str(attrs.get("clip_uid") or ""), fid))
        by_video = defaultdict(list)
        for row in ordered:
            if row[0]:
                by_video[row[0]].append(row)
        for rows in by_video.values():
            rows.sort(key=lambda row: (row[1], row[2], row[3]))
            for left, right in zip(rows, rows[1:]):
                if left[2] == right[2]:
                    continue
                self.graph_db.add_link(Link(
                    source_node_id=left[3], target_node_id=right[3],
                    link_type=LinkType.TEMPORAL,
                    properties={"sub_type": "SCENE_NEXT", "confidence": 1.0,
                                "confidence_score": 1.0, "evidence_tier": "confirmed",
                                "video_uid": left[0]},
                ))
                created += 1
        return created

    def create_clip_hierarchy_links(self, clip_node, frame_ids, bidirectional=False):
        """CLIP -> FRAME containment.

        Only CLIP_CONTAINS is created by default. PART_OF_CLIP is a
        duplicate reverse copy and is derivable at query time from the
        incoming CLIP_CONTAINS edge, so it is skipped to halve hierarchy
        edges.
        """
        created = 0
        for fid in frame_ids:
            self.graph_db.add_link(Link(
                source_node_id=clip_node.node_id, target_node_id=fid,
                link_type=LinkType.SEMANTIC,
                properties={"sub_type": "CLIP_CONTAINS", "confidence": 1.0,
                            "confidence_score": 1.0, "evidence_tier": "confirmed"},
            ))
            created += 1
            if bidirectional:
                self.graph_db.add_link(Link(
                    source_node_id=fid, target_node_id=clip_node.node_id,
                    link_type=LinkType.SEMANTIC,
                    properties={"sub_type": "PART_OF_CLIP", "confidence": 1.0,
                                "confidence_score": 1.0, "evidence_tier": "confirmed"},
                ))
                created += 1
        return created

    def create_video_hierarchy_links(self):
        """VIDEO -> CLIP containment (parent node points to child node)."""
        created = 0
        for video_uid, video_node_id in self.video_nodes.items():
            video_node = self.graph_db.get_node(video_node_id)
            if not video_node:
                continue
            for clip_uid in video_node.attributes.get("clip_ids", []):
                clip_node_id = "clip_" + clip_uid
                if clip_node_id in self.graph_db.nodes:
                    self.graph_db.add_link(Link(
                        source_node_id=video_node.node_id, target_node_id=clip_node_id,
                        link_type=LinkType.SEMANTIC,
                        properties={"sub_type": "VIDEO_CONTAINS", "confidence": 1.0,
                                    "confidence_score": 1.0, "evidence_tier": "confirmed"},
                    ))
                    created += 1
        return created

    def create_object_links(self, object_hubs, bidirectional=False):
        """Entity edges: SAME_ENTITY for shared objects, hub nodes for high-freq.

        Mirrors MAGMA's original design:
        - Low-freq objects: pairwise SAME_ENTITY edges between frames sharing
          the entity (window-limited to avoid O(N^2) explosion).
        - High-freq objects (>= HUB_OBJECT_THRESHOLD frames): ENTITY hub node
          with MENTIONS_OBJECT edges for multi-hop traversal. MENTIONED_IN
          (the reverse copy) is skipped by default; query traversal is
          undirected so the hub is still reachable from every frame.
        """
        created = 0
        obj_to_frames = defaultdict(list)
        for fid, objs in self.frame_objects.items():
            for o in objs:
                obj_to_frames[o].append(fid)

        for obj, fids in obj_to_frames.items():
            if not fids:
                continue
            if obj in object_hubs:
                # high-frequency: route through hub node
                hub_id = object_hubs[obj]
                for fid in fids:
                    self.graph_db.add_link(Link(
                        source_node_id=fid, target_node_id=hub_id,
                        link_type=LinkType.ENTITY,
                        properties={"sub_type": "MENTIONS_OBJECT", "entity": obj,
                                    "confidence": 0.9, "confidence_score": 0.9,
                                    "evidence_tier": "supported"},
                    ))
                    created += 1
                    if bidirectional:
                        self.graph_db.add_link(Link(
                            source_node_id=hub_id, target_node_id=fid,
                            link_type=LinkType.ENTITY,
                            properties={"sub_type": "MENTIONED_IN", "entity": obj,
                                        "confidence": 0.9, "confidence_score": 0.9,
                                        "evidence_tier": "supported"},
                        ))
                        created += 1
            else:
                # low-frequency: pairwise SAME_ENTITY edges (window-limited)
                for i in range(len(fids)):
                    for j in range(i + 1, min(i + 8, len(fids))):
                        self.graph_db.add_link(Link(
                            source_node_id=fids[i], target_node_id=fids[j],
                            link_type=LinkType.ENTITY,
                            properties={"sub_type": "SAME_ENTITY", "entity": obj, "confidence": 0.9,
                                        "confidence_score": 0.9, "evidence_tier": "supported"},
                        ))
                        created += 1
        return created

    def create_relation_links(self, relation_hubs=None, max_time_diff=60.0,
                              max_frames_window=8):
        """Semantic edges for frames sharing a relation_label.

        High-frequency relations route through an ENTITY hub node via
        SHARES_RELATION edges (one edge per frame), so any pair of frames
        sharing the relation is reachable in 2 hops and the O(N^2)
        all-pairs explosion is avoided. Low-frequency relations keep
        window-limited frame-to-frame SHARES_RELATION edges.
        """
        created = 0
        relation_hubs = relation_hubs or {}
        rel_to_frames = defaultdict(list)
        for fid, rels in self.frame_relations.items():
            for r in rels:
                rel_to_frames[r].append(fid)

        for rel, fids in rel_to_frames.items():
            if len(fids) < 2:
                continue
            if rel in relation_hubs:
                hub_id = relation_hubs[rel]
                for fid in fids:
                    self.graph_db.add_link(Link(
                        source_node_id=fid, target_node_id=hub_id,
                        link_type=LinkType.SEMANTIC,
                        properties={"sub_type": "SHARES_RELATION",
                                    "relation": rel, "confidence": 0.7,
                                    "confidence_score": 0.7, "evidence_tier": "supported"},
                    ))
                    created += 1
                continue
            timed = []
            for fid in fids:
                node = self.graph_db.get_node(fid)
                if node:
                    timed.append((node.attributes.get("video_time_sec") or 0, fid))
            timed.sort()
            for i in range(len(timed)):
                ti, fi = timed[i]
                for j in range(i + 1, min(i + 1 + max_frames_window, len(timed))):
                    tj, fj = timed[j]
                    if tj - ti > max_time_diff:
                        break  # sorted by time; later frames are only farther
                    self.graph_db.add_link(Link(
                        source_node_id=fi, target_node_id=fj,
                        link_type=LinkType.SEMANTIC,
                        properties={"sub_type": "SHARES_RELATION", "relation": rel, "confidence": 0.7,
                                    "confidence_score": 0.7, "evidence_tier": "supported"},
                    ))
                    created += 1
        return created

    def create_visual_similar_links(self, top_k=None, min_score=None):
        """Create visual-neighbour edges without per-frame Qdrant HTTP calls.

        The previous implementation issued one Qdrant search per frame for any
        graph over 300 frames.  At full scale that is hundreds of thousands of
        sequential HTTP round trips and was the dominant 28-hour bottleneck.
        All frame vectors are already present in memory, so we use chunked
        matrix multiplication and add only mutual/top-ranked neighbours.
        """
        top_k = self.VISUAL_TOP_K if top_k is None else top_k
        min_score = self.VISUAL_MIN_SCORE if min_score is None else min_score

        vectors = {nid: v for nid, v in self.frame_vectors.items() if v is not None}
        if len(vectors) < 2:
            return 0
        return self._create_visual_similar_links_local(
            vectors, top_k=top_k, min_score=min_score,
            mutual=self.VISUAL_MUTUAL,
            chunk_size=self.VISUAL_MATRIX_CHUNK,
        )

    def _create_visual_similar_links_local(self, vectors: Dict, top_k=5,
                                            min_score=0.75, mutual=True,
                                            chunk_size=256) -> int:
        """Exact top-k cosine links using bounded-memory matrix chunks."""
        ids = list(vectors)
        dim = len(np.asarray(vectors[ids[0]], dtype=np.float32))
        mat = np.empty((len(ids), dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            mat[i] = np.asarray(vectors[nid], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

        top_neighbors = [set() for _ in ids]
        kth = min(top_k, len(ids) - 1)
        if kth <= 0:
            return 0

        for start in range(0, len(ids), chunk_size):
            stop = min(start + chunk_size, len(ids))
            sims = mat[start:stop] @ mat.T
            rows = np.arange(start, stop)
            sims[np.arange(stop - start), rows] = -np.inf
            candidates = np.argpartition(-sims, kth=kth, axis=1)[:, :kth]
            for local_i, global_i in enumerate(rows):
                for global_j in candidates[local_i]:
                    if sims[local_i, global_j] >= min_score:
                        top_neighbors[global_i].add(int(global_j))

        created = 0
        seen_pairs = set()
        for i, nid in enumerate(ids):
            for j in top_neighbors[i]:
                if i == j:
                    continue
                if mutual and i not in top_neighbors[j]:
                    continue
                nb_id = ids[j]
                pair = frozenset((nid, nb_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                # Preserve deterministic direction: lower node id -> higher.
                src, tgt = sorted((nid, nb_id))
                score = float(np.dot(mat[i], mat[j]))
                self.graph_db.add_link(Link(
                    source_node_id=src, target_node_id=tgt,
                    link_type=LinkType.SEMANTIC,
                    properties={"sub_type": "VISUAL_SIMILAR",
                                "similarity": score, "confidence": score,
                                "confidence_score": score, "evidence_tier": "inferred"},
                ))
                created += 1
        return created

    # ------------------------------------------------------------------ #
    #  Main build entry
    # ------------------------------------------------------------------ #

    def build_from_frames(self, frame_points: list) -> dict:
        """Build the full graph from raw Qdrant WorldMM frame points."""
        if not frame_points:
            return {"error": "no frames"}

        total_started = time.perf_counter()
        frame_points, input_points, duplicate_points = self._deduplicate_frame_points(frame_points)
        clip_groups = defaultdict(list)
        for fp in frame_points:
            clip_groups[fp["payload"].get("clip_uid")].append(fp)

        all_frame_ids = []
        temporal_links = clip_hierarchy_links = 0
        stats_frames = stats_clips = 0
        node_started = time.perf_counter()

        for clip_uid, clip_frames in clip_groups.items():
            clip_frames.sort(key=lambda fp: (fp["payload"].get("frame_seq") is None,
                                             fp["payload"].get("frame_seq") or 0))
            frame_ids_in_clip = []
            for fp in clip_frames:
                node = self.create_frame_node(fp)
                self.graph_db.add_node(node)
                all_frame_ids.append(node.node_id)
                frame_ids_in_clip.append(node.node_id)
                self.frame_objects[node.node_id] = set(
                    self._salient_object_labels(
                        node.attributes.get("objects") or [],
                        node.attributes.get("object_labels") or [],
                        self.OBJECT_HUB_MAX_PER_FRAME,
                    )
                )
                self.frame_relations[node.node_id] = set(node.attributes.get("relation_labels") or [])
                self.index_frame(node.node_id, node.content_narrative, node.attributes)
                stats_frames += 1

            clip_node = self.create_clip_node(clip_uid, clip_frames)
            self.graph_db.add_node(clip_node)
            self.clip_nodes[clip_uid] = clip_node.node_id
            stats_clips += 1
            temporal_links += self.create_temporal_links(frame_ids_in_clip)
            clip_hierarchy_links += self.create_clip_hierarchy_links(
                clip_node, frame_ids_in_clip)

        node_build_seconds = round(time.perf_counter() - node_started, 3)

        # Entity hubs are derived after frame nodes so frequency is based on
        # salient graph-worthy labels, not every payload detection.
        obj_freq = defaultdict(int)
        for labels in self.frame_objects.values():
            for obj in labels:
                obj_freq[obj] += 1
        object_hubs = {}
        for obj, cnt in sorted(obj_freq.items()):
            if cnt >= self.HUB_OBJECT_THRESHOLD:
                hub = EventNode(
                    node_id="obj_" + uuid.uuid4().hex[:8],
                    node_type=NodeType.ENTITY,
                    timestamp=datetime.now(),
                    content_narrative="Object hub: %s (in %d frames)" % (obj, cnt),
                    attributes={"entity_type": "object", "label": obj, "frame_count": cnt},
                    embedding_vector=None,
                )
                self.graph_db.add_node(hub)
                object_hubs[obj] = hub.node_id
        self.object_hubs = object_hubs

        rel_freq = defaultdict(int)
        for relations in self.frame_relations.values():
            for rel in relations:
                rel_freq[rel] += 1
        relation_hubs = {}
        for rel, cnt in sorted(rel_freq.items()):
            if cnt >= self.RELATION_HUB_THRESHOLD:
                hub = EventNode(
                    node_id="rel_" + uuid.uuid4().hex[:8],
                    node_type=NodeType.ENTITY,
                    timestamp=datetime.now(),
                    content_narrative="Relation hub: %s (in %d frames)" % (rel, cnt),
                    attributes={"entity_type": "relation", "label": rel,
                                "frame_count": cnt},
                    embedding_vector=None,
                )
                self.graph_db.add_node(hub)
                relation_hubs[rel] = hub.node_id
        self.relation_hubs = relation_hubs

        stats = {
            "input_points": input_points,
            "deduplicated_points": duplicate_points,
            "frames": stats_frames,
            "clips": stats_clips,
            "videos": 0,
            "objects_hubs": len(object_hubs),
            "relation_hubs": len(relation_hubs),
        }

        vid_clips = defaultdict(list)
        vid_frame_count = defaultdict(int)
        vid_payloads = defaultdict(list)
        for fp in frame_points:
            vu = fp["payload"].get("video_uid")
            cu = fp["payload"].get("clip_uid")
            vid_clips[vu].append(cu)
            vid_frame_count[vu] += 1
            vid_payloads[vu].append(fp["payload"])
        for video_uid, clip_uids in vid_clips.items():
            vnode = self.create_video_node(
                video_uid, sorted(set(clip_uids)), vid_frame_count[video_uid],
                payloads=vid_payloads[video_uid])
            self.graph_db.add_node(vnode)
            self.video_nodes[video_uid] = vnode.node_id
            stats["videos"] += 1

        self.frame_ids = all_frame_ids
        link_started = time.perf_counter()
        video_hierarchy_links = self.create_video_hierarchy_links()
        link_stats = {
            "temporal": temporal_links,
            "scene_temporal": self.create_video_temporal_links(all_frame_ids),
            "clip_hierarchy": clip_hierarchy_links,
            "video_hierarchy": video_hierarchy_links,
            "object_links": self.create_object_links(object_hubs),
            "relation_links": self.create_relation_links(relation_hubs),
        }
        link_build_seconds = round(time.perf_counter() - link_started, 3)
        visual_started = time.perf_counter()
        link_stats["visual_similar"] = self.create_visual_similar_links()
        visual_seconds = round(time.perf_counter() - visual_started, 3)

        causal_pair_seconds = 0.0
        causal_stats = {"edges_added": 0, "skipped": not self.enable_causal_edges}
        if self.enable_causal_edges:
            from .causal_edge_builder import CausalEdgeBuilder, extract_causal_pairs
            causal_pair_started = time.perf_counter()
            causal_pairs = extract_causal_pairs(self.graph_db)
            causal_pair_seconds = round(time.perf_counter() - causal_pair_started, 3)
            causal_started = time.perf_counter()
            causal_stats = CausalEdgeBuilder().build_causal_edges(
                causal_pairs, self.graph_db)
            causal_stats["pair_extraction_seconds"] = causal_pair_seconds
            causal_stats["build_seconds"] = round(time.perf_counter() - causal_started, 3)
        link_stats["causal"] = causal_stats.get("edges_added", 0)
        stats["causal"] = causal_stats

        stats["links"] = link_stats
        stats["timing_seconds"] = {
            "total": round(time.perf_counter() - total_started, 3),
            "nodes": node_build_seconds,
            "links": link_build_seconds,
            "visual_similar": visual_seconds,
            "causal_pairs": causal_pair_seconds,
            "causal": causal_stats.get("build_seconds", 0),
            "causal_llm": causal_stats.get("llm_seconds", 0),
        }
        stats["total_links"] = len(self.graph_db.links)
        stats["total_nodes"] = len(self.graph_db.nodes)
        return stats
    def save_to_sqlite(self, db_path):
        """Persist graph to SQLite via the existing SQLiteGraphDB."""
        from .sqlite_graph_db import SQLiteGraphDB
        sqdb = SQLiteGraphDB(db_path)
        sqdb.graph.clear()
        sqdb.nodes.clear()
        sqdb.links.clear()
        sqdb.node_to_links.clear()
        for nid, node in self.graph_db.nodes.items():
            sqdb.add_node(node)
        for lid, link in self.graph_db.links.items():
            sqdb.graph.add_edge(link.source_node_id, link.target_node_id, key=lid, **link.to_dict())
            sqdb.links[lid] = link
            sqdb.node_to_links.setdefault(link.source_node_id, set()).add(lid)
            sqdb.node_to_links.setdefault(link.target_node_id, set()).add(lid)
        sqdb.save()
        sqdb.save_keyword_index(self.node_index)
        sqdb.save_json_metadata("structured_index", self.structured_index.to_dict())
        logger.info("Saved keyframe graph to %s (%d nodes, %d links)", db_path, len(sqdb.nodes), len(sqdb.links))
        return sqdb

    @classmethod
    def load_from_sqlite(cls, db_path, qdrant_adapter=None):
        """Rebuild a builder (graph + keyword index) from a SQLite file.

        Returns a KeyframeMemoryBuilder with graph_db and node_index populated.
        Does NOT rebuild edges or fetch vectors; those are already in SQLite.
        The qdrant_adapter is attached for optional runtime vector recall.
        """
        from .sqlite_graph_db import SQLiteGraphDB
        sqdb = SQLiteGraphDB(db_path)
        sqdb.load()          # graph_nodes/graph_edges -> nodes/links/graph
        node_index = sqdb.load_keyword_index()

        builder = cls.__new__(cls)
        builder.qa = qdrant_adapter
        builder.graph_db = sqdb   # SQLiteGraphDB is a NetworkXGraphDB subclass
        builder.node_index = node_index
        builder.frame_ids = [nid for nid, n in sqdb.nodes.items() if n.node_type.value == "EVENT"]
        builder.clip_nodes = {n.attributes.get("clip_uid"): nid
                              for nid, n in sqdb.nodes.items()
                              if n.node_type.value == "SESSION" and hasattr(n, "attributes")}
        builder.video_nodes = {n.attributes.get("video_uid"): nid
                               for nid, n in sqdb.nodes.items()
                               if n.node_type.value == "EPISODE" and hasattr(n, "attributes")}
        builder.clip_to_video = {}
        builder.object_hubs = {n.attributes.get("label"): nid
                               for nid, n in sqdb.nodes.items()
                               if n.node_type.value == "ENTITY" and hasattr(n, "attributes")}
        builder.frame_objects = {}
        builder.frame_relations = {}

        # prefer the persisted structured index; rebuild only for old DBs
        from .structured_index import StructuredIndex
        structured_data = sqdb.load_json_metadata("structured_index")
        if structured_data:
            builder.structured_index = StructuredIndex.from_dict(structured_data)
        else:
            builder.structured_index = StructuredIndex()
            for nid, n in sqdb.nodes.items():
                if n.node_type.value == "EVENT" and hasattr(n, "attributes"):
                    objs = n.attributes.get("objects") or []
                    rels = n.attributes.get("relations") or []
                    builder.structured_index.add_frame(nid, objs, rels)

        # build semantic label matcher from graph labels
        from .semantic_label_matcher import SemanticLabelMatcher
        try:
            builder.label_matcher = SemanticLabelMatcher()
            builder.label_matcher.build_from_graph(sqdb, structured_index=builder.structured_index)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to build label matcher: %s", e)
            builder.label_matcher = None

        logger.info("Loaded keyframe graph from %s (%d nodes, %d links, %d index terms)",
                    db_path, len(sqdb.nodes), len(sqdb.links), len(node_index))
        return builder
