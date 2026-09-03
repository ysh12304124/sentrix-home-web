"""Build and query the MAGMA-style keyframe graph from Sentrix data."""
from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .graph_db import EventNode, Link, LinkType, NodeType
from .keyframe_memory_builder import KeyframeMemoryBuilder
from .keyframe_query_engine import KeyframeQueryEngine
from .sentrix_frame_provider import SentrixFrameProvider


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class GraphMemoryService:
    """Facade used by scripts and FastAPI routes."""

    def __init__(self, db_path: str | Path | None = None, graph_path: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("SENTRIX_DATA_DIR", root / "data"))
        self.db_path = str(Path(db_path or os.getenv("SENTRIX_DB_PATH", data_dir / "sentrix.db")).resolve())
        configured_graph = graph_path or os.getenv("SENTRIX_GRAPH_DB_PATH")
        self.graph_path = str(Path(configured_graph or (data_dir / "graph" / "graph_memory.db")).resolve())
        self._builder: KeyframeMemoryBuilder | None = None
        self._lock = threading.RLock()

    @property
    def graph_exists(self) -> bool:
        return Path(self.graph_path).is_file()

    def build(
        self,
        scope_id: str | None = None,
        include_images: bool = False,
        enable_causal_edges: bool | None = None,
    ) -> dict:
        # Ordinary uploaded photos are first-class memories too.  The previous
        # default only projected video keyframes, so the graph expander could
        # never recover an image that lexical/ANN recall missed.  Keep an
        # explicit opt-out for large deployments/tests while enabling the
        # complete memory projection by default.
        include_images = bool(include_images or _truthy(
            os.getenv("SENTRIX_GRAPH_INCLUDE_IMAGES"), True))
        provider = SentrixFrameProvider(self.db_path)
        frames = provider.fetch_frames(scope_id=scope_id, include_images=include_images)
        if not frames:
            return {
                "ok": False,
                "error": "no keyframes found",
                "db_path": self.db_path,
                "graph_path": self.graph_path,
                "hint": "upload and process a video first, or pass include_images=true for existing image memories",
            }

        if enable_causal_edges is None:
            enable_causal_edges = _truthy(os.getenv("GRAPH_CAUSAL_ENABLED"), True)
        elif not enable_causal_edges and _truthy(os.getenv("GRAPH_CAUSAL_ENABLED"), True):
            # The API/benchmark payload historically sent ``causal: false``
            # because causal construction was an optional enhancement.  That
            # left every graph snapshot without causal edges.  Treat the
            # default as enabled while retaining an explicit environment opt-
            # out (GRAPH_CAUSAL_ENABLED=0) for constrained deployments.
            enable_causal_edges = True

        builder = KeyframeMemoryBuilder(enable_causal_edges=enable_causal_edges)
        stats = builder.build_from_frames(frames)
        if stats.get("error"):
            return {"ok": False, "error": stats["error"], "stats": stats}

        entity_stats = self._attach_sentrix_entities(builder, frames)
        context_stats = self._attach_context_nodes(builder, frames)
        self._attach_event_summaries(builder, frames)
        stats["sentrix_entities"] = entity_stats
        stats["context_nodes"] = context_stats
        stats["total_nodes"] = len(builder.graph_db.nodes)
        stats["total_links"] = len(builder.graph_db.links)

        Path(self.graph_path).parent.mkdir(parents=True, exist_ok=True)
        graph_db = builder.save_to_sqlite(self.graph_path)
        try:
            graph_db.save_json_metadata("sentrix_build", {
                "built_at": datetime.now().isoformat(),
                "db_path": self.db_path,
                "scope_id": scope_id or "all",
                "include_images": bool(include_images),
                "causal_enabled": bool(enable_causal_edges),
                "stats": stats,
            })
        finally:
            graph_db.close()

        with self._lock:
            self._builder = None

        return {
            "ok": True,
            "db_path": self.db_path,
            "graph_path": self.graph_path,
            "stats": stats,
        }

    def status(self) -> dict:
        path = Path(self.graph_path)
        result: dict[str, Any] = {
            "exists": path.is_file(),
            "path": str(path),
            "db_path": self.db_path,
        }
        if not path.is_file():
            return result

        conn = sqlite3.connect(self.graph_path)
        try:
            conn.row_factory = sqlite3.Row
            node_count = int(conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0])
            edge_count = int(conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0])
            node_types = Counter()
            for row in conn.execute("SELECT data FROM graph_nodes"):
                try:
                    node_types[json.loads(row["data"]).get("node_type", "UNKNOWN")] += 1
                except Exception:
                    node_types["INVALID"] += 1

            edge_types = Counter()
            edge_subtypes = Counter()
            for row in conn.execute("SELECT data FROM graph_edges"):
                try:
                    data = json.loads(row["data"])
                    edge_types[data.get("link_type", "UNKNOWN")] += 1
                    edge_subtypes[(data.get("properties") or {}).get("sub_type", "UNKNOWN")] += 1
                except Exception:
                    edge_types["INVALID"] += 1

            metadata_row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'sentrix_build'"
            ).fetchone()
        finally:
            conn.close()

        result.update({
            "nodes": node_count,
            "edges": edge_count,
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "edge_subtypes": dict(edge_subtypes),
            "built_at": None,
        })
        if metadata_row:
            try:
                metadata = json.loads(metadata_row["value"])
                result["built_at"] = metadata.get("built_at")
                result["last_build"] = metadata
            except Exception:
                pass
        return result

    def quality(self, scope_id: str | None = None) -> dict:
        """Compute read-only, full-graph structural quality metrics.

        This endpoint deliberately does not mutate the graph or participate in
        QA execution.  It audits every stored node and edge in the current
        graph snapshot; ``scope_id`` optionally limits the audit to nodes from
        one memory space and their incident edges.
        """
        path = Path(self.graph_path)
        if not path.is_file():
            return {"available": False, "reason": "graph memory has not been built", "scope_id": scope_id}
        conn = sqlite3.connect(self.graph_path)
        try:
            node_rows = conn.execute("SELECT id, data FROM graph_nodes").fetchall()
            edge_rows = conn.execute("SELECT id, source_id, target_id, data FROM graph_edges").fetchall()
        finally:
            conn.close()
        nodes = {}
        for node_id, raw in node_rows:
            try:
                nodes[str(node_id)] = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                nodes[str(node_id)] = {}
        if scope_id:
            scoped = {
                node_id for node_id, data in nodes.items()
                if (data.get("attributes") or {}).get("scope_id") == scope_id
            }
            selected_edges = [row for row in edge_rows if row[1] in scoped or row[2] in scoped]
            selected_ids = scoped | {row[1] for row in selected_edges} | {row[2] for row in selected_edges}
            nodes = {node_id: data for node_id, data in nodes.items() if node_id in selected_ids}
        else:
            selected_edges = edge_rows

        def timestamp(data):
            attrs = data.get("attributes") or {}
            value = attrs.get("captured_at") or data.get("timestamp")
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
            except (TypeError, ValueError):
                return None

        def confidence(properties):
            raw = properties.get("confidence_score", properties.get("confidence"))
            try:
                return float(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None

        deterministic_subtypes = {
            "TIME_PRECEDES", "SCENE_NEXT", "CLIP_CONTAINS", "VIDEO_CONTAINS",
            "OCCURRED_AT", "CAPTURED_ON", "MENTIONS_PERSON", "MENTIONS_OBJECT",
            "SAME_ENTITY", "BELONGS_TO_SESSION", "PART_OF_CLIP",
        }
        valid_edges = supported_edges = consistent_edges = 0
        duplicate_keys = set()
        duplicate_count = 0
        type_counts = {}
        invalid_by_type = {}
        degree = {node_id: 0 for node_id in nodes}
        for edge_id, source, target, raw in selected_edges:
            try:
                data = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                data = {}
            link_type = str(data.get("link_type") or "UNKNOWN")
            props = data.get("properties") or {}
            subtype = str(props.get("sub_type") or "")
            type_counts[link_type] = type_counts.get(link_type, 0) + 1
            # A pair may legitimately share more than one object/relation;
            # include that discriminator so those edges are not misreported as
            # duplicates merely because their subtype is the same.
            discriminator = props.get("entity") or props.get("relation") or props.get("object") or ""
            key = (str(source), str(target), link_type, subtype, str(discriminator))
            if key in duplicate_keys:
                duplicate_count += 1
            duplicate_keys.add(key)
            endpoints_ok = source in nodes and target in nodes and source != target
            if endpoints_ok:
                valid_edges += 1
                degree[source] = degree.get(source, 0) + 1
                degree[target] = degree.get(target, 0) + 1
            confidence_value = confidence(props)
            evidence_ok = subtype in deterministic_subtypes or (confidence_value is not None and confidence_value >= 0.5)
            if link_type == "SEMANTIC" and subtype == "VISUAL_SIMILAR":
                try:
                    evidence_ok = float(props.get("similarity")) >= 0.75
                except (TypeError, ValueError):
                    evidence_ok = False
            if endpoints_ok and evidence_ok:
                supported_edges += 1
            structurally_ok = endpoints_ok
            if structurally_ok and link_type == "TEMPORAL":
                left, right = timestamp(nodes[source]), timestamp(nodes[target])
                if subtype == "TIME_PRECEDES" and props.get("sequence_index") is not None:
                    # The builder creates this edge from ordered frame
                    # sequence; EXIF capture times are not reliable for video
                    # frames and may be reordered by the source device.
                    structurally_ok = True
                elif left and right and subtype in {"TIME_PRECEDES", "SCENE_NEXT"}:
                    structurally_ok = left <= right
            if structurally_ok and link_type == "CAUSAL":
                left, right = timestamp(nodes[source]), timestamp(nodes[target])
                if left and right:
                    structurally_ok = left <= right
                structurally_ok = structurally_ok and evidence_ok
            if structurally_ok:
                consistent_edges += 1
            else:
                invalid_by_type[link_type] = invalid_by_type.get(link_type, 0) + 1

        total_nodes = len(nodes)
        total_edges = len(selected_edges)
        connected_nodes = sum(1 for value in degree.values() if value > 0)
        return {
            "available": True,
            "scope_id": scope_id,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_connected_rate": connected_nodes / total_nodes if total_nodes else None,
            "edge_evidence_support_rate": supported_edges / total_edges if total_edges else None,
            "edge_consistency_rate": consistent_edges / total_edges if total_edges else None,
            "duplicate_edge_rate": duplicate_count / total_edges if total_edges else None,
            "valid_edge_rate": valid_edges / total_edges if total_edges else None,
            "connected_nodes": connected_nodes,
            "supported_edges": supported_edges,
            "consistent_edges": consistent_edges,
            "duplicate_edges": duplicate_count,
            "edge_type_counts": type_counts,
            "invalid_edges_by_type": invalid_by_type,
            "computed_at": datetime.now().isoformat(),
        }

    def search(self, query: str, top_k: int = 10, scope_id: str | None = None) -> dict:
        if not self.graph_exists:
            return {"ok": False, "error": "graph memory has not been built", "path": self.graph_path}
        if not str(query or "").strip():
            return {"ok": False, "error": "query is required"}

        # The graph is a derived index.  Rebuild it lazily when new assets or
        # events were written after the last build, so normal retrieval never
        # serves an old two-frame graph after a video upload.
        self._refresh_if_stale(scope_id=scope_id)

        with self._lock:
            builder = self._load_builder()
            try:
                engine = KeyframeQueryEngine(
                    builder.graph_db,
                    builder.node_index,
                    structured_index=builder.structured_index,
                    label_matcher=builder.label_matcher,
                    enable_full_scan=True,
                )
                result = engine.query(
                    str(query),
                    top_k=max(1, min(int(top_k or 10), 50)),
                    scope={"scope_id": scope_id} if scope_id else None,
                    graph_enabled=True,
                )
            finally:
                builder.graph_db.close()
                self._builder = None

        nodes = [self._node_payload(node) for node in result.get("nodes") or []]
        return {
            "ok": True,
            "query": query,
            "query_type": result.get("query_type"),
            "context": result.get("context"),
            "nodes": nodes,
            "graph_paths": result.get("graph_paths") or [],
            "stats": self._serializable(result.get("stats") or {}),
            "needs_visual": result.get("needs_visual", False),
        }

    def expand_from_assets(self, query: str, seed_asset_ids: list[str],
                           top_k: int = 30, scope_id: str | None = None) -> dict:
        """Expand ordinary retrieval seeds through the derived graph.

        This is deliberately separate from ``search``: the normal evidence
        retriever remains the recall backbone, while graph traversal only
        contributes candidates reachable from those already-recalled assets.
        """
        if not str(query or "").strip() or not seed_asset_ids:
            return {"ok": True, "nodes": [], "paths": []}
        self._refresh_if_stale(scope_id=scope_id)
        if not self.graph_exists:
            return {"ok": False, "error": "graph memory has not been built", "nodes": [], "paths": []}
        with self._lock:
            builder = self._load_builder()
            try:
                engine = KeyframeQueryEngine(
                    builder.graph_db, builder.node_index,
                    structured_index=builder.structured_index,
                    label_matcher=builder.label_matcher,
                    enable_full_scan=True,
                )
                matched = engine.label_matcher.match(query) if engine.label_matcher else {"objects": [], "predicates": [], "all": []}
                keywords = matched.get("objects", []) + matched.get("predicates", [])
                if not keywords:
                    keywords = engine._extract_object_terms(query)
                query_type = engine.detect_query_type(query, matched)
                params = engine.get_adaptive_params(query_type)
                seeds = set(str(value) for value in seed_asset_ids if value)
                anchors = []
                for node in builder.graph_db.nodes.values():
                    if node.node_type.value != "EVENT":
                        continue
                    attrs = node.attributes or {}
                    source_id = str(attrs.get("source_asset_id") or attrs.get("frame_uid") or node.node_id)
                    if source_id in seeds or node.node_id in seeds:
                        if not scope_id or str(attrs.get("scope_id") or "home-default") == str(scope_id):
                            node.similarity_score = 1.0
                            anchors.append(node)
                traversed, paths = engine._graph_traversal(
                    anchors, keywords, params, question=query,
                    max_nodes=max(20, min(int(top_k or 30) * 8, 240)),
                    scope={"scope_id": scope_id} if scope_id else None,
                    bidirectional=query_type == "multi_hop",
                )
                nodes = [node for node, score in traversed if node.node_type.value == "EVENT"]
                return {"ok": True, "nodes": [
                    {"asset_id": str((node.attributes or {}).get("source_asset_id") or node.node_id),
                     "score": float(score), "event_id": (node.attributes or {}).get("event_id"),
                     "node_id": node.node_id}
                    for node, score in traversed if node.node_type.value == "EVENT"
                ], "paths": paths, "query_type": query_type}
            finally:
                builder.graph_db.close()
                self._builder = None

    def _refresh_if_stale(self, scope_id: str | None = None) -> None:
        include_images = _truthy(os.getenv("SENTRIX_GRAPH_INCLUDE_IMAGES"), True)
        causal_enabled = _truthy(os.getenv("GRAPH_CAUSAL_ENABLED"), True)
        if not self.graph_exists:
            self.build(scope_id=scope_id, include_images=include_images)
            return
        try:
            graph_mtime = Path(self.graph_path).stat().st_mtime
            conn = sqlite3.connect(self.db_path)
            try:
                params = []
                where = "WHERE media_type = 'image' AND (derived_kind IN ('video_keyframe', 'video_keyframe_webp')"
                if include_images:
                    where += " OR derived_kind IS NULL"
                where += ")"
                row = conn.execute(f"SELECT COUNT(*), MAX(COALESCE(updated_at, created_at)) FROM assets {where}", params).fetchone()
            finally:
                conn.close()
            current_count = int(row[0] or 0) if row else 0
            latest = row[1] if row else None
            built_count = 0
            try:
                gconn = sqlite3.connect(self.graph_path)
                metadata_row = gconn.execute("SELECT value FROM metadata WHERE key = 'sentrix_build'").fetchone()
                gconn.close()
                if metadata_row:
                    metadata = json.loads(metadata_row[0])
                    built_count = int(((metadata.get("stats") or {}).get("frames") or 0))
            except Exception:
                built_count = 0
            built_include_images = False
            built_causal = False
            try:
                gconn = sqlite3.connect(self.graph_path)
                metadata_row = gconn.execute("SELECT value FROM metadata WHERE key = 'sentrix_build'").fetchone()
                gconn.close()
                if metadata_row:
                    metadata = json.loads(metadata_row[0])
                    built_include_images = bool(metadata.get("include_images"))
                    built_causal = bool(metadata.get("causal_enabled"))
            except Exception:
                built_include_images = False
            if current_count and (current_count != built_count
                                  or (include_images and not built_include_images)
                                  or (causal_enabled and not built_causal)):
                # Keep one complete derived graph and apply scope filtering at
                # traversal time; rebuilding a scope-specific graph here
                # would overwrite memories from other albums.
                self.build(scope_id=None, include_images=include_images)
                return
            if latest:
                try:
                    latest_ts = datetime.fromisoformat(str(latest).replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError, OSError):
                    latest_ts = 0.0
                if latest_ts > graph_mtime + 1.0:
                    self.build(scope_id=None, include_images=include_images)
        except Exception:
            # A stale-check failure must never take down retrieval; the
            # existing graph remains a safe fallback.
            return

    def _load_builder(self) -> KeyframeMemoryBuilder:
        if self._builder is None:
            self._builder = KeyframeMemoryBuilder.load_from_sqlite(self.graph_path)
        return self._builder

    @staticmethod
    def _node_payload(node: Any) -> dict:
        attrs = dict(getattr(node, "attributes", {}) or {})
        return {
            "id": node.node_id,
            "node_type": node.node_type.value,
            "timestamp": node.timestamp.isoformat() if getattr(node, "timestamp", None) else None,
            "score": getattr(node, "similarity_score", None),
            "caption": getattr(node, "content_narrative", ""),
            "frame_path": attrs.get("frame_path") or attrs.get("source_frame_path"),
            "video_uid": attrs.get("video_uid"),
            "clip_uid": attrs.get("clip_uid"),
            "time_sec": attrs.get("video_time_sec", attrs.get("clip_time_sec")),
            "event_id": attrs.get("sentrix_event_id") or attrs.get("event_id"),
            "person_ids": attrs.get("person_ids") or [],
            "objects": attrs.get("object_labels") or [],
            "relations": attrs.get("relation_labels") or [],
            "causal_context": getattr(node, "causal_context", None) or [],
            "attributes": attrs,
        }

    @staticmethod
    def _serializable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): GraphMemoryService._serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [GraphMemoryService._serializable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _attach_sentrix_entities(self, builder: KeyframeMemoryBuilder, frames: list[dict]) -> dict:
        """Add stable Sentrix entity/person/place nodes to the MAGMA graph."""
        entities_by_id: dict[str, dict] = {}
        frames_by_entity: dict[str, list[dict]] = defaultdict(list)

        for frame in frames:
            payload = frame.get("payload") or {}
            for entity in payload.get("entity_refs") or []:
                entity_id = str(entity.get("id") or "").strip()
                if not entity_id:
                    continue
                entities_by_id[entity_id] = entity
                frames_by_entity[entity_id].append(frame)

        entity_nodes: dict[str, str] = {}
        added_nodes = 0
        added_edges = 0
        same_person_edges = 0
        for entity_id, entity in entities_by_id.items():
            entity_type = str(entity.get("entity_type") or "object").lower()
            label = str(entity.get("canonical_name") or entity.get("family_role") or entity_id).strip()
            node_id = f"sentrix_entity_{entity_id}"
            node = EventNode(
                node_id=node_id,
                node_type=NodeType.ENTITY,
                timestamp=datetime.now(),
                content_narrative=f"{entity_type}: {label}",
                attributes={
                    "entity_type": entity_type,
                    "label": label,
                    "sentrix_entity_id": entity_id,
                    "status": entity.get("status"),
                    "family_role": entity.get("family_role"),
                },
                embedding_vector=None,
            )
            builder.graph_db.add_node(node)
            entity_nodes[entity_id] = node_id
            added_nodes += 1

            for frame in frames_by_entity[entity_id]:
                frame_id = str(frame.get("id"))
                subtype = "MENTIONS_PERSON" if entity_type == "person" else (
                    "OCCURRED_AT" if entity_type == "place" else "MENTIONS_OBJECT"
                )
                builder.graph_db.add_link(Link(
                    source_node_id=frame_id,
                    target_node_id=node_id,
                    link_type=LinkType.ENTITY,
                    properties={"sub_type": subtype, "confidence": float(entity.get("confidence") or 1.0)},
                ))
                added_edges += 1

        # Person consistency edges connect chronological appearances of the
        # same confirmed entity. They are intentionally limited to adjacent
        # appearances to avoid an O(n²) graph for common people.
        for entity_id, frame_list in frames_by_entity.items():
            entity = entities_by_id[entity_id]
            if str(entity.get("entity_type") or "").lower() != "person":
                continue
            ordered = sorted(
                frame_list,
                key=lambda item: (
                    str((item.get("payload") or {}).get("video_uid") or ""),
                    float((item.get("payload") or {}).get("video_time_sec") or 0.0),
                    str(item.get("id")),
                ),
            )
            for left, right in zip(ordered, ordered[1:]):
                builder.graph_db.add_link(Link(
                    source_node_id=str(left["id"]),
                    target_node_id=str(right["id"]),
                    link_type=LinkType.ENTITY,
                    properties={
                        "sub_type": "SAME_ENTITY",
                        "sentrix_entity_id": entity_id,
                        "confidence": min(1.0, float(entity.get("confidence") or 1.0)),
                    },
                ))
                same_person_edges += 1

        return {
            "entities": len(entity_nodes),
            "entity_edges": added_edges,
            "same_person_edges": same_person_edges,
            "person_entities": sum(1 for item in entities_by_id.values() if str(item.get("entity_type") or "").lower() == "person"),
        }

    @staticmethod
    def _stable_context_id(kind: str, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"sentrix_{kind}_{digest}"

    @staticmethod
    def _normalise_place(value: Any) -> str:
        text = re.sub(r"\s+", "", str(value or "")).strip()
        if not text or text in {"其他或不确定", "未知", "不确定"}:
            return ""
        return text

    @staticmethod
    def _date_key(value: Any) -> str:
        text = str(value or "").strip()
        match = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", text)
        if not match:
            return ""
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    def _attach_context_nodes(self, builder: KeyframeMemoryBuilder, frames: list[dict]) -> dict:
        """Attach stable place/date hubs for cross-event long-term recall."""
        contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
        for frame in frames:
            payload = frame.get("payload") or {}
            frame_id = str(frame.get("id") or "")
            if not frame_id or not builder.graph_db.get_node(frame_id):
                continue
            place = self._normalise_place(payload.get("place"))
            date_key = self._date_key(payload.get("captured_at"))
            for kind, value in (("place", place), ("date", date_key)):
                if value:
                    contexts[(kind, value)].append(frame_id)

        created_nodes = 0
        created_edges = 0
        for (kind, value), frame_ids in contexts.items():
            node_id = self._stable_context_id(kind, value)
            if not builder.graph_db.get_node(node_id):
                node = EventNode(
                    node_id=node_id,
                    node_type=NodeType.ENTITY,
                    timestamp=datetime.now(),
                    content_narrative=(f"地点: {value}" if kind == "place" else f"日期: {value}"),
                    attributes={"entity_type": kind, "label": value,
                                "canonical_name": value, "long_term_context": True},
                    embedding_vector=None,
                )
                builder.graph_db.add_node(node)
                created_nodes += 1
            subtype = "OCCURRED_AT" if kind == "place" else "CAPTURED_ON"
            seen_frames = set()
            for frame_id in frame_ids:
                if frame_id in seen_frames:
                    continue
                seen_frames.add(frame_id)
                builder.graph_db.add_link(Link(
                    source_node_id=frame_id, target_node_id=node_id,
                    link_type=LinkType.ENTITY,
                    properties={"sub_type": subtype, "context": value,
                                "confidence": 1.0, "source": "sentrix_metadata"},
                ))
                created_edges += 1
        return {"nodes": created_nodes, "edges": created_edges,
                "places": sum(1 for kind, _ in contexts if kind == "place"),
                "dates": sum(1 for kind, _ in contexts if kind == "date")}

    def _attach_event_summaries(self, builder: KeyframeMemoryBuilder, frames: list[dict]) -> None:
        event_ids = list(dict.fromkeys(
            str((frame.get("payload") or {}).get("event_id") or "")
            for frame in frames
            if (frame.get("payload") or {}).get("event_id")
        ))
        if not event_ids:
            return

        placeholders = ",".join("?" for _ in event_ids)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT id, title, summary, place, time_start, time_end FROM events WHERE id IN ({placeholders})",
                event_ids,
            ).fetchall()
        finally:
            conn.close()
        events = {str(row["id"]): dict(row) for row in rows}

        for node in list(builder.graph_db.nodes.values()):
            if node.node_type.value != "SESSION":
                continue
            event_id = str((node.attributes or {}).get("clip_uid") or "")
            event = events.get(event_id)
            if not event:
                continue
            title = str(event.get("title") or "").strip()
            summary = str(event.get("summary") or "").strip()
            if title:
                node.summary = title if not summary else f"{title}\n{summary}"
            if summary:
                node.attributes["sentrix_event_summary"] = summary
            node.attributes["sentrix_event_id"] = event_id
            node.attributes["sentrix_event_title"] = title
            node.attributes["sentrix_event_place"] = event.get("place")
            node.attributes["sentrix_event_time_start"] = event.get("time_start")
            node.attributes["sentrix_event_time_end"] = event.get("time_end")
