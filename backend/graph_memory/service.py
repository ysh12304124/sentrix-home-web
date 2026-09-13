"""Build and query the MAGMA-style keyframe graph from Sentrix data."""
from __future__ import annotations

import copy
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

    # FastAPI creates a short-lived facade for each request.  Keeping these
    # guards on the instance did not serialize concurrent quality polls, so
    # two requests could scan the 100k+ edge SQLite file at the same time.
    _quality_lock_global = threading.Lock()
    _quality_cache_global: dict[tuple[str, str, int, int], dict] = {}

    # Bump when the persisted projection fields or event-to-frame links
    # change.  The next query then rebuilds the derived graph automatically
    # instead of silently serving an older snapshot.
    GRAPH_SCHEMA_VERSION = 2

    @staticmethod
    def _face_cluster_quality(scope_id: str | None) -> dict:
        """Return only production-observable clustering diagnostics.

        Benchmark identity sidecars are evaluator data, not application data.
        The API must never scan ``services/photobench/data`` (or any other
        reference-answer artifact) while serving a user request.  Pairwise
        precision/recall is computed by the PhotoBench evaluator, which may
        explicitly load its dataset annotations after the run; production
        quality therefore reports this metric as unavailable here.
        """
        return {
            "available": False,
            "scope_id": scope_id,
            "ground_truth_source": None,
            "reason": "identity GT is evaluator-only; runtime does not load benchmark annotations",
        }

    @staticmethod
    def _load_independent_edge_ground_truth(scope_id: str | None) -> dict:
        """Load an explicitly independent graph-edge GT file, if configured.

        The graph itself cannot be its own ground truth.  To keep the API
        honest, a GT file is accepted only when it declares
        ``independent_ground_truth: true`` (or ``metadata.source`` is
        ``human``/``manual``/``external``).  Edges use persisted graph node
        ids and the same tuple fields as the evaluator.  With no such file,
        callers receive ``available=False`` instead of a fabricated score.
        """
        root = Path(__file__).resolve().parents[2]
        configured = str(os.getenv("SENTRIX_GRAPH_EDGE_GT_PATH") or "").strip()
        candidates = [Path(configured)] if configured else [
            root / "data" / "graph" / "graph_edge_ground_truth.jsonl",
            root / "data" / "graph" / "graph_edge_ground_truth.json",
        ]
        path = next((item for item in candidates if item and item.is_file()), None)
        result = {
            "available": False,
            "path": str(path) if path else None,
            "independent": False,
            "edges": set(),
            "reason": "independent graph-edge GT file not configured",
        }
        if path is None:
            return result
        try:
            if path.suffix.lower() == ".jsonl":
                records = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
                payload = {"edges": records}
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result["reason"] = f"GT file could not be read: {exc}"
            return result
        if isinstance(payload, list):
            metadata = {}
            records = payload
        elif isinstance(payload, dict):
            metadata = payload.get("metadata") or {}
            records = payload.get("edges") or payload.get("relations") or []
        else:
            result["reason"] = "GT file must contain an edges array"
            return result
        source_marker = str(metadata.get("source") or payload.get("source") or "").strip().lower() \
            if isinstance(payload, dict) else ""
        independent = bool(
            (payload.get("independent_ground_truth") if isinstance(payload, dict) else False)
            or source_marker in {"human", "manual", "external", "annotated"}
        )
        if not independent:
            result["reason"] = "GT file is not marked independent_ground_truth"
            return result
        edges = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            record_scope = str(record.get("scope_id") or "").strip()
            if scope_id and record_scope and record_scope != scope_id:
                continue
            source = record.get("source") or record.get("source_id")
            target = record.get("target") or record.get("target_id")
            subtype = record.get("subtype") or record.get("sub_type") or ""
            link_type = record.get("link_type") or record.get("type") or ""
            if not source or not target or not subtype or not link_type:
                continue
            discriminator = (
                record.get("discriminator")
                or record.get("entity")
                or record.get("relation")
                or record.get("object")
                or ""
            )
            edges.add((str(source), str(target), str(link_type), str(subtype), str(discriminator)))
        result.update({
            "available": bool(edges),
            "independent": True,
            "edges": edges,
            "reason": None if edges else "GT file contains no scoped edges",
        })
        return result

    def __init__(self, db_path: str | Path | None = None, graph_path: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("SENTRIX_DATA_DIR", root / "data"))
        self.db_path = str(Path(db_path or os.getenv("SENTRIX_DB_PATH", data_dir / "sentrix.db")).resolve())
        configured_graph = graph_path or os.getenv("SENTRIX_GRAPH_DB_PATH")
        self.graph_path = str(Path(configured_graph or (data_dir / "graph" / "graph_memory.db")).resolve())
        # The long-lived application service uses the default constructor and
        # keeps its read-only builder hot between requests.  Explicit graph
        # paths are commonly used by tests/one-shot maintenance commands;
        # preserve their historical close-after-query behaviour so temporary
        # SQLite files can be deleted immediately on Windows.
        self._cache_builder = graph_path is None
        self._builder: KeyframeMemoryBuilder | None = None
        self._lock = threading.RLock()
        # Graph quality is a read-heavy endpoint polled by the benchmark UI.
        # Serialise it and cache by the graph file fingerprint: concurrent
        # 100+ MB SQLite scans were able to exhaust the Windows Python process
        # while QA workers were using the same API process.
        self._quality_lock = self._quality_lock_global
        self._quality_cache = self._quality_cache_global

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
        # Do not replace/delete graph rows while the quality endpoint is
        # scanning them.  The lock only covers the SQLite persistence window;
        # expensive graph construction above can still run independently.
        # Close any reader before replacing the SQLite file.  On Windows an
        # open connection can keep the old file locked and, more importantly,
        # a cached builder would otherwise continue serving stale nodes.
        with self._lock:
            self._dispose_builder()

        with self._quality_lock:
            graph_db = builder.save_to_sqlite(self.graph_path)
            try:
                graph_db.save_json_metadata("sentrix_build", {
                    "built_at": datetime.now().isoformat(),
                    "db_path": self.db_path,
                    "scope_id": scope_id or "all",
                    "include_images": bool(include_images),
                    "causal_enabled": bool(enable_causal_edges),
                    "graph_schema_version": self.GRAPH_SCHEMA_VERSION,
                    "stats": stats,
                })
            finally:
                graph_db.close()
            self._quality_cache.clear()

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
        """Return a cached, serialised graph-quality snapshot.

        The graph is large enough that repeatedly materialising every edge for
        each UI poll is unsafe on Windows.  A file fingerprint invalidates the
        cache after a graph rebuild, while the lock prevents overlapping scans.
        """
        path = Path(self.graph_path)
        if not path.is_file():
            return {"available": False, "reason": "graph memory has not been built", "scope_id": scope_id}
        try:
            stat = path.stat()
            cache_key = (self.graph_path, str(scope_id or ""),
                         int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            cache_key = None
        with self._quality_lock:
            if cache_key is not None:
                cached = self._quality_cache.get(cache_key)
                if cached is not None:
                    return copy.deepcopy(cached)
            result = self._quality_uncached(scope_id)
            if cache_key is not None and result.get("available"):
                # Keep only the most recent snapshot for a scope.  This bounds
                # memory when the UI changes scopes during a long run.
                scope_key = cache_key[1]
                for key in list(self._quality_cache):
                    if key[0] == self.graph_path and key[1] == scope_key:
                        self._quality_cache.pop(key, None)
                self._quality_cache[cache_key] = copy.deepcopy(result)
            return result

    def _quality_uncached(self, scope_id: str | None = None) -> dict:
        """Compute read-only, full-graph structural quality metrics.

        This endpoint deliberately does not mutate the graph or participate in
        QA execution.  It audits every stored node and edge in the current
        graph snapshot; ``scope_id`` optionally limits the audit to nodes from
        one memory space and their incident edges.
        """
        path = Path(self.graph_path)
        if not path.is_file():
            return {"available": False, "reason": "graph memory has not been built", "scope_id": scope_id}
        # Read-only/query-only mode prevents an audit request from taking part
        # in SQLite writes and gives the connection a bounded lock wait.
        conn = sqlite3.connect(self.graph_path, timeout=30)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            node_rows = conn.execute("SELECT id, data FROM graph_nodes").fetchall()
            nodes = {}
            for node_id, raw in node_rows:
                try:
                    nodes[str(node_id)] = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    nodes[str(node_id)] = {}
            # A scope owns EVENT nodes.  The old implementation treated every
            # node reached by an edge from one scoped event as part of the
            # scope.  That pulled in neighbouring events from other albums
            # through SAME_ENTITY/SHARES_RELATION chains and then used those
            # neighbours to build the reference denominator.  The result was
            # an artificially low recall (the screenshot showed ~17%) even
            # though the builder had constructed the scoped edges correctly.
            # Keep the owned event set separate from the small set of shared
            # hubs/containers needed to evaluate those events.
            owned_events = None
            if scope_id:
                owned_events = {
                    node_id for node_id, data in nodes.items()
                    if str(data.get("node_type") or "") == "EVENT"
                    and (data.get("attributes") or {}).get("scope_id") == scope_id
                }
            # Stream edges instead of fetchall(): this avoids holding the
            # cursor's native result buffer and all 100k+ JSON rows twice.
            edge_rows = []
            for row in conn.execute("SELECT id, source_id, target_id, data FROM graph_edges"):
                edge_rows.append(row)
        finally:
            conn.close()
        if scope_id:
            owned_events = owned_events or set()
            # Include hubs and hierarchy containers directly attached to an
            # owned event, but never import a non-owned EVENT neighbour.  A
            # video container is one extra hop above its scoped SESSION.
            support_ids = set(owned_events)
            for row in edge_rows:
                source, target = str(row[1]), str(row[2])
                if source not in owned_events and target not in owned_events:
                    continue
                for endpoint in (source, target):
                    data = nodes.get(endpoint) or {}
                    if str(data.get("node_type") or "") != "EVENT":
                        support_ids.add(endpoint)
            session_ids = {
                node_id for node_id in support_ids
                if str((nodes.get(node_id) or {}).get("node_type") or "") == "SESSION"
            }
            for row in edge_rows:
                source, target = str(row[1]), str(row[2])
                if source not in session_ids and target not in session_ids:
                    continue
                for endpoint in (source, target):
                    data = nodes.get(endpoint) or {}
                    if str(data.get("node_type") or "") == "EPISODE":
                        support_ids.add(endpoint)
            selected_edges = [
                row for row in edge_rows
                if str(row[1]) in support_ids and str(row[2]) in support_ids
            ]
            nodes = {
                node_id: data for node_id, data in nodes.items()
                if node_id in support_ids
            }
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
        # Build a deterministic *source-consistency* relation set from source
        # fields carried by the graph projection.  This is independent of the
        # persisted edge rows, but not independent of the upstream detector
        # or metadata quality.  Semantic similarity and causal edges are
        # reported as unverifiable instead of being counted as ground truth.
        reference = set()
        events_by_clip = {}
        sessions_by_clip = {}
        episodes = []
        entity_nodes_by_id = {}
        # Object labels and Sentrix object entities are two different source
        # contracts.  Keep detector hubs separate from stable entity nodes so
        # the reference does not require both representations for every
        # label.
        object_hubs_by_label = defaultdict(set)
        relation_hubs = {}
        events_by_relation = defaultdict(list)
        for node_id, data in nodes.items():
            attrs = data.get("attributes") or {}
            node_type = str(data.get("node_type") or "")
            if (node_type == "EVENT"
                    and (owned_events is None or node_id in owned_events)
                    and attrs.get("clip_uid")):
                events_by_clip.setdefault(str(attrs["clip_uid"]), []).append((node_id, attrs))
                for relation in attrs.get("relation_labels") or []:
                    relation = str(relation).strip()
                    if relation:
                        events_by_relation[relation].append((node_id, attrs))
            elif node_type == "SESSION" and attrs.get("clip_uid"):
                sessions_by_clip[str(attrs["clip_uid"])] = node_id
            elif node_type == "EPISODE" and attrs.get("clip_ids"):
                episodes.append((node_id, attrs))
            elif node_type == "ENTITY":
                if attrs.get("sentrix_entity_id"):
                    entity_nodes_by_id[str(attrs["sentrix_entity_id"])] = node_id
                if (attrs.get("entity_type") == "object"
                        and attrs.get("label")
                        and not attrs.get("sentrix_entity_id")):
                    object_hubs_by_label[str(attrs["label"])].add(node_id)
                if attrs.get("entity_type") == "relation" and attrs.get("label"):
                    relation_hubs[str(attrs["label"])] = node_id
        for clip_uid, rows in events_by_clip.items():
            rows.sort(key=lambda item: (
                item[1].get("frame_seq") is None,
                item[1].get("frame_seq") if item[1].get("frame_seq") is not None else item[1].get("video_time_sec") or 0,
                item[0]))
            for (source, _), (target, _) in zip(rows, rows[1:]):
                reference.add((source, target, "TEMPORAL", "TIME_PRECEDES", ""))
            session = sessions_by_clip.get(clip_uid)
            if session:
                for event, _ in rows:
                    reference.add((session, event, "SEMANTIC", "CLIP_CONTAINS", ""))
        for episode, attrs in episodes:
            for clip_uid in attrs.get("clip_ids") or []:
                session = sessions_by_clip.get(str(clip_uid))
                if session:
                    reference.add((episode, session, "SEMANTIC", "VIDEO_CONTAINS", ""))
        for event, data in nodes.items():
            if (str(data.get("node_type") or "") != "EVENT"
                    or (owned_events is not None and event not in owned_events)):
                continue
            attrs = data.get("attributes") or {}
            for entity in attrs.get("entity_refs") or []:
                if not isinstance(entity, dict) or not entity.get("id"):
                    continue
                target = entity_nodes_by_id.get(str(entity["id"]))
                kind = str(entity.get("entity_type") or "")
                subtype = (
                    "MENTIONS_PERSON" if kind == "person" else
                    "OCCURRED_AT" if kind == "place" else
                    "CAPTURED_ON" if kind in {"time", "date"} else
                    "MENTIONS_OBJECT" if kind == "object" else ""
                )
                if target and subtype:
                    reference.add((event, target, "ENTITY", subtype, ""))
            for label in attrs.get("object_labels") or []:
                for target in object_hubs_by_label.get(str(label), set()):
                    # Object-hub edges carry the label as their discriminator;
                    # preserve it so hub edges are matched one-to-one rather
                    # than being reported as false positives merely because
                    # the reference key used an empty discriminator.
                    reference.add((event, target, "ENTITY", "MENTIONS_OBJECT", str(label)))
            # Metadata context hubs are independently generated by
            # _attach_context_nodes.  They are deterministic reference edges
            # too; omitting them made every date edge (and the second place
            # representation) look like an unverified false positive.
            place_value = self._normalise_place(
                attrs.get("place") or attrs.get("captured_location")
            )
            if place_value:
                target = self._stable_context_id("place", place_value)
                if target in nodes:
                    reference.add((event, target, "ENTITY", "OCCURRED_AT", ""))
            date_value = self._date_key(attrs.get("captured_at"))
            if date_value:
                target = self._stable_context_id("date", date_value)
                if target in nodes:
                    reference.add((event, target, "ENTITY", "CAPTURED_ON", ""))

        # SHARES_RELATION is deterministic at the graph-topology level: the
        # builder derives it from the frame relation labels, then either links
        # each frame to a relation hub or links the time-neighbouring frames.
        # Recreate that rule from the persisted source fields instead of
        # counting the builder's own edges as their reference.  This measures
        # graph construction consistency; it is not an accuracy score for the
        # upstream relation detector (that requires human relation labels).
        for relation, rows in events_by_relation.items():
            unique_rows = {str(node_id): (str(node_id), attrs) for node_id, attrs in rows}
            if relation in relation_hubs:
                hub_id = str(relation_hubs[relation])
                for event in sorted(unique_rows):
                    reference.add((event, hub_id, "SEMANTIC", "SHARES_RELATION", relation))
                continue
            timed = []
            for event, attrs in unique_rows.values():
                try:
                    video_time = float(attrs.get("video_time_sec") or 0.0)
                except (TypeError, ValueError):
                    video_time = 0.0
                timed.append((video_time, event))
            timed.sort(key=lambda item: (item[0], item[1]))
            for index, (left_time, left_event) in enumerate(timed):
                for other_index in range(index + 1, min(index + 1 + 8, len(timed))):
                    right_time, right_event = timed[other_index]
                    if right_time - left_time > 60.0:
                        break
                    reference.add((left_event, right_event, "SEMANTIC", "SHARES_RELATION", relation))

        evaluable_types = {"TEMPORAL", "SEMANTIC", "ENTITY"}
        evaluable_subtypes = {"TIME_PRECEDES", "CLIP_CONTAINS", "VIDEO_CONTAINS",
                              "MENTIONS_PERSON", "OCCURRED_AT", "CAPTURED_ON",
                              "MENTIONS_OBJECT", "SHARES_RELATION"}
        predicted_reference = set()
        evaluable_edge_subtype_counts = Counter()
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
            if link_type in evaluable_types and subtype in evaluable_subtypes:
                predicted_reference.add(key)
                evaluable_edge_subtype_counts[subtype] += 1
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
        # The rule-reference metric is the product's deterministic graph
        # construction metric.  It is useful for regression and coverage,
        # but it is not independent human truth because both sides originate
        # from Sentrix source fields and projection rules.
        source_consistency_true_positive = len(predicted_reference & reference)
        source_consistency_precision = (
            source_consistency_true_positive / len(predicted_reference)
            if predicted_reference else None
        )
        source_consistency_recall = (
            source_consistency_true_positive / len(reference)
            if reference else None
        )
        source_consistency_f1 = (
            2 * source_consistency_precision * source_consistency_recall
            / (source_consistency_precision + source_consistency_recall)
            if source_consistency_precision is not None
            and source_consistency_recall is not None
            and source_consistency_precision + source_consistency_recall else None
        )
        independent_gt = self._load_independent_edge_ground_truth(scope_id)
        independent_reference = independent_gt["edges"]
        independent_true_positive = len(predicted_reference & independent_reference)
        independent_precision = (
            independent_true_positive / len(predicted_reference)
            if independent_gt["available"] and predicted_reference else None
        )
        independent_recall = (
            independent_true_positive / len(independent_reference)
            if independent_gt["available"] and independent_reference else None
        )
        independent_f1 = (
            2 * independent_precision * independent_recall
            / (independent_precision + independent_recall)
            if independent_precision is not None and independent_recall is not None
            and independent_precision + independent_recall else None
        )
        independent_available = bool(independent_gt["available"])
        # The product's "可验证边" metric is intentionally rule-reference
        # based: hard constraints define the expected relation set.  Use an
        # independent GT when one is supplied, otherwise keep this engineering
        # metric visible instead of returning an empty dashboard.  The source
        # consistency values remain duplicated under explicit diagnostic keys.
        reference_metric_available = independent_available or bool(reference)
        reference_metric_true_positive = (
            independent_true_positive if independent_available else source_consistency_true_positive
        )
        reference_metric_precision = (
            independent_precision if independent_available else source_consistency_precision
        )
        reference_metric_recall = (
            independent_recall if independent_available else source_consistency_recall
        )
        reference_metric_f1 = independent_f1 if independent_available else source_consistency_f1
        source_reference_subtype_counts = Counter(key[3] for key in reference)
        predicted_subtype_counts = Counter(key[3] for key in predicted_reference)
        source_true_positive_subtype_counts = Counter(
            key[3] for key in (predicted_reference & reference)
        )
        source_recall_by_subtype = {
            subtype: (
                source_true_positive_subtype_counts.get(subtype, 0) / count
                if count else None
            )
            for subtype, count in source_reference_subtype_counts.items()
        }
        source_precision_by_subtype = {
            subtype: (
                source_true_positive_subtype_counts.get(subtype, 0) / count
                if count else None
            )
            for subtype, count in predicted_subtype_counts.items()
        }
        active_reference = independent_reference if independent_available else reference
        active_true_positive = (
            predicted_reference & active_reference
        )
        active_reference_subtype_counts = Counter(key[3] for key in active_reference)
        active_true_positive_subtype_counts = Counter(key[3] for key in active_true_positive)
        active_recall_by_subtype = {
            subtype: active_true_positive_subtype_counts.get(subtype, 0) / count
            if count else None
            for subtype, count in active_reference_subtype_counts.items()
        }
        active_precision_by_subtype = {
            subtype: active_true_positive_subtype_counts.get(subtype, 0) / count
            if count else None
            for subtype, count in predicted_subtype_counts.items()
        }
        independent_reference_subtype_counts = Counter(key[3] for key in independent_reference)
        independent_true_positive_subtype_counts = Counter(
            key[3] for key in (predicted_reference & independent_reference)
        )
        independent_recall_by_subtype = {
            subtype: independent_true_positive_subtype_counts.get(subtype, 0) / count
            if count else None
            for subtype, count in independent_reference_subtype_counts.items()
        }
        independent_precision_by_subtype = {
            subtype: independent_true_positive_subtype_counts.get(subtype, 0) / count
            if count else None
            for subtype, count in predicted_subtype_counts.items()
        }
        traceable_nodes = 0
        for data in nodes.values():
            attrs = data.get("attributes") or {}
            node_type = str(data.get("node_type") or "")
            is_traceable = (
                (node_type == "EVENT" and attrs.get("source_asset_id") and attrs.get("scope_id"))
                or (node_type == "SESSION" and attrs.get("clip_uid"))
                or (node_type == "EPISODE" and attrs.get("video_uid"))
                or (node_type == "ENTITY" and (attrs.get("label") or attrs.get("sentrix_entity_id")))
            )
            traceable_nodes += int(bool(is_traceable))
        connected_nodes = sum(1 for value in degree.values() if value > 0)
        face_quality = self._face_cluster_quality(scope_id)
        return {
            "available": True,
            "scope_id": scope_id,
            # This is intentionally not called ground-truth accuracy: the
            # reference set is reconstructed from Sentrix source fields and
            # deterministic projection rules.  It is independent of the
            # persisted edge rows, but it is not independent of the upstream
            # detector/metadata inputs.  A human-labelled relation dataset is
            # required before these P/R/F1 values can be called real-world
            # graph accuracy.
            "reference_edge_evaluation_mode": (
                "independent_ground_truth" if independent_available else "rule_reference"
            ),
            "reference_edge_ground_truth_available": independent_available,
            "reference_edge_truth_source": (
                independent_gt["path"] if independent_available
                else "sentrix_source_fields_and_projection_rules"
            ),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_connected_rate": connected_nodes / total_nodes if total_nodes else None,
            "node_source_traceability_rate": traceable_nodes / total_nodes if total_nodes else None,
            "edge_evidence_support_rate": supported_edges / total_edges if total_edges else None,
            "edge_consistency_rate": consistent_edges / total_edges if total_edges else None,
            "duplicate_edge_rate": duplicate_count / total_edges if total_edges else None,
            "valid_edge_rate": valid_edges / total_edges if total_edges else None,
            "connected_nodes": connected_nodes,
            "valid_edges": valid_edges,
            "supported_edges": supported_edges,
            "consistent_edges": consistent_edges,
            "duplicate_edges": duplicate_count,
            # Public reference-edge metrics use deterministic rule-reference
            # data by default, and switch to independent GT when configured.
            "reference_edge_count": len(independent_reference) if independent_available else len(reference),
            "evaluable_predicted_edge_count": len(predicted_reference),
            "reference_edge_true_positive_count": reference_metric_true_positive,
            "reference_edge_precision": reference_metric_precision,
            "reference_edge_recall": reference_metric_recall,
            "reference_edge_f1": reference_metric_f1,
            "source_consistency_reference_edge_count": len(reference),
            "source_consistency_true_positive_count": source_consistency_true_positive,
            "source_consistency_precision": source_consistency_precision,
            "source_consistency_recall": source_consistency_recall,
            "source_consistency_f1": source_consistency_f1,
            "independent_ground_truth_edge_count": len(independent_reference) if independent_available else 0,
            "independent_ground_truth_reason": independent_gt["reason"],
            "reference_edge_subtype_counts": dict(active_reference_subtype_counts),
            "reference_edge_true_positive_subtype_counts": dict(active_true_positive_subtype_counts),
            "reference_edge_recall_by_subtype": active_recall_by_subtype,
            "reference_edge_precision_by_subtype": active_precision_by_subtype,
            "source_consistency_reference_edge_subtype_counts": dict(source_reference_subtype_counts),
            "source_consistency_true_positive_subtype_counts": dict(source_true_positive_subtype_counts),
            "source_consistency_recall_by_subtype": source_recall_by_subtype,
            "source_consistency_precision_by_subtype": source_precision_by_subtype,
            "evaluable_edge_subtype_counts": dict(evaluable_edge_subtype_counts),
            "unverifiable_edge_count": max(0, total_edges - len(predicted_reference)),
            "edge_type_counts": type_counts,
            "invalid_edges_by_type": invalid_by_type,
            "face_clustering": face_quality,
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
                    # The service is the graph-retrieval entry point.  Keep
                    # multi-hop path recovery enabled here; the query engine
                    # still gates the extra expansion on query_type, so
                    # ordinary/temporal queries keep their existing path.
                    # Previously the default False silently disabled the
                    # explicit path-cover stage for global graph queries,
                    # leaving multi-hop routed QA with zero graph candidates.
                    enable_multi_hop_expansion=True,
                )
                result = engine.query(
                    str(query),
                    top_k=max(1, min(int(top_k or 10), 50)),
                    scope={"scope_id": scope_id} if scope_id else None,
                    graph_enabled=True,
                )
                # The structured/semantic channels are intentionally strict,
                # but a non-empty result is not proof that the graph head is
                # complete.  In practice they often return only the five
                # metadata-matched EVENT nodes for a paraphrased question;
                # stopping there made graph retrieval a reranker of the same
                # ANN candidates and it could never recover a missed event.
                # Merge the deterministic persisted-text fallback whenever
                # the strict head is sparse.  This uses only graph EVENT
                # fields (never benchmark answers) and keeps the hard scope
                # boundary in the fallback itself.
                strict_nodes = list(result.get("nodes") or [])
                target_size = max(1, min(int(top_k or 10), 50))
                # Do not make lexical graph evidence conditional on the strict
                # head being short.  A relation/multi-hop query can fill all
                # 24 strict positions with generic visual neighbours while
                # still missing the two EVENT nodes whose persisted relation
                # text exactly matches the question.  The previous
                # ``len(strict) < top_k`` guard discarded those matches and
                # made real SHARES_RELATION paths unretrievable.
                fallback_nodes = self._text_fallback_nodes(
                    builder, engine, str(query), scope_id, top_k)
                if fallback_nodes:
                    strict_sources = {
                        str((getattr(node, "attributes", {}) or {}).get("source_asset_id")
                            or node.node_id)
                        for node in strict_nodes
                    }
                    unique_fallback = []
                    fallback_sources = set()
                    for node in fallback_nodes:
                        source = str(
                            (getattr(node, "attributes", {}) or {}).get("source_asset_id")
                            or node.node_id
                        )
                        if source in strict_sources or source in fallback_sources:
                            continue
                        unique_fallback.append(node)
                        fallback_sources.add(source)
                    if unique_fallback:
                        if len(strict_nodes) < target_size:
                            merged_nodes = (strict_nodes + unique_fallback)[:target_size]
                        else:
                            # Reserve a small, bounded part of a saturated
                            # graph head for direct persisted-text matches.
                            # This affects only graph results; normal ANN/FTS
                            # candidates remain in the fused baseline.
                            quota = min(len(unique_fallback), max(1, min(4, target_size // 6)))
                            merged_nodes = unique_fallback[:quota] + [
                                node for node in strict_nodes
                                if str((getattr(node, "attributes", {}) or {}).get("source_asset_id")
                                       or node.node_id) not in fallback_sources
                            ][:target_size - quota]
                        result["nodes"] = merged_nodes
                        result["stats"] = dict(result.get("stats") or {})
                        result["stats"]["text_fallback"] = max(
                            0, len(merged_nodes) - len(strict_nodes))
                        result["stats"]["text_fallback_forced"] = min(
                            len(unique_fallback), len(merged_nodes))
            finally:
                if not self._cache_builder:
                    self._dispose_builder()

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
                    enable_multi_hop_expansion=True,
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
                    # A path answer often asks for the event *after* an anchor
                    # (e.g. "看完小火车之后去了哪里"). The next event naturally
                    # does not repeat the anchor keyword, so keyword-gating every
                    # neighbour would erase the path.
                    allow_unmatched_events=query_type in {"multi_hop", "temporal"},
                )
                return {"ok": True, "nodes": [
                    {"asset_id": str((node.attributes or {}).get("source_asset_id") or node.node_id),
                     "score": float(score), "event_id": (node.attributes or {}).get("event_id"),
                     "node_id": node.node_id}
                    for node, score in traversed if node.node_type.value == "EVENT"
                ], "paths": paths, "query_type": query_type}
            finally:
                if not self._cache_builder:
                    self._dispose_builder()

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
                # A scoped graph snapshot is a valid derived index for that
                # scope.  Comparing it with the full database count makes the
                # first query after a scoped rebuild look stale (e.g. 352
                # frames vs 6,791 globally) and silently starts an expensive
                # full rebuild.  Compare like-for-like instead.
                built_scope = None
                try:
                    gconn = sqlite3.connect(self.graph_path)
                    metadata_row = gconn.execute(
                        "SELECT value FROM metadata WHERE key = 'sentrix_build'"
                    ).fetchone()
                    gconn.close()
                    if metadata_row:
                        built_scope = str((json.loads(metadata_row[0]) or {}).get("scope_id") or "")
                except Exception:
                    built_scope = None
                if built_scope and built_scope not in {"all", "None"}:
                    where += " AND scope_id = ?"
                    params.append(built_scope)
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
            built_schema_version = 0
            try:
                gconn = sqlite3.connect(self.graph_path)
                metadata_row = gconn.execute("SELECT value FROM metadata WHERE key = 'sentrix_build'").fetchone()
                gconn.close()
                if metadata_row:
                    metadata = json.loads(metadata_row[0])
                    built_include_images = bool(metadata.get("include_images"))
                    built_causal = bool(metadata.get("causal_enabled"))
                    built_schema_version = int(metadata.get("graph_schema_version") or 0)
            except Exception:
                built_include_images = False
            # Do not invalidate a scoped snapshot merely because the caller
            # asks for the same scope.  A different scope is intentionally
            # left to the caller's explicit rebuild request; retrieval must
            # never launch a full rebuild in the request path.
            scope_matches_snapshot = (
                not built_scope or built_scope in {"all", "None"}
                or not scope_id or built_scope == str(scope_id)
            )
            if current_count and scope_matches_snapshot and (current_count != built_count
                                  or (include_images and not built_include_images)
                                  or (causal_enabled and not built_causal)
                                  or built_schema_version < self.GRAPH_SCHEMA_VERSION):
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

    def _dispose_builder(self) -> None:
        """Release the cached SQLite reader before a graph rebuild."""
        builder = self._builder
        self._builder = None
        if builder is not None:
            try:
                builder.graph_db.close()
            except Exception:
                pass

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
            "captured_at": attrs.get("captured_at"),
            "place": attrs.get("place") or attrs.get("sentrix_event_place"),
            "event_title": attrs.get("event_title") or attrs.get("sentrix_event_title"),
            "event_summary": attrs.get("event_summary") or attrs.get("sentrix_event_summary"),
            "source_asset_id": attrs.get("source_asset_id") or attrs.get("frame_uid"),
            "source_video_file_name": attrs.get("source_video_file_name"),
            "event_id": attrs.get("sentrix_event_id") or attrs.get("event_id"),
            "person_ids": attrs.get("person_ids") or [],
            "objects": attrs.get("object_labels") or [],
            "relations": attrs.get("relation_labels") or [],
            "causal_context": getattr(node, "causal_context", None) or [],
            "attributes": attrs,
        }

    @staticmethod
    def _text_fallback_nodes(builder: KeyframeMemoryBuilder,
                             engine: KeyframeQueryEngine, question: str,
                             scope_id: str | None, top_k: int) -> list[Any]:
        """Recover EVENT nodes when strict graph matching has no hit.

        Candidates still come only from persisted graph EVENT nodes and their
        stored text fields. This bridges paraphrased multi-hop wording
        without consulting benchmark answers or issuing another model call.
        """
        text = re.sub(r"\s+", "", str(question or "")).lower()
        if not text:
            return []
        terms: list[str] = []
        try:
            matched = engine.label_matcher.match(question) if engine.label_matcher else {}
            terms.extend(str(item).strip().lower() for key in ("objects", "predicates", "all")
                          for item in (matched.get(key) or []))
            terms.extend(str(item).strip().lower() for item in engine._extract_object_terms(question))
        except Exception:
            pass
        if len(text) <= 48:
            for width in (4, 3, 2):
                terms.extend(text[index:index + width]
                             for index in range(0, max(0, len(text) - width + 1)))
        terms = list(dict.fromkeys(item for item in terms if len(item) >= 2))
        if not terms:
            return []
        scored: list[tuple[float, Any]] = []
        for node in builder.graph_db.nodes.values():
            if node.node_type.value != "EVENT":
                continue
            attrs = node.attributes or {}
            if scope_id and str(attrs.get("scope_id") or "home-default") != str(scope_id):
                continue
            haystack = " ".join(str(value or "") for value in (
                getattr(node, "content_narrative", ""), attrs.get("caption"),
                attrs.get("event_title"), attrs.get("event_summary"),
                attrs.get("place"), attrs.get("sentrix_event_place"),
                attrs.get("object_labels"), attrs.get("relation_labels"),
            )).lower()
            matched_terms = sum(1 for term in terms if term in haystack)
            if matched_terms <= 0:
                continue
            node.similarity_score = float(matched_terms) / max(1, len(terms))
            scored.append((float(matched_terms), node))
        scored.sort(key=lambda item: (-item[0], str(item[1].node_id)))
        return [node for _, node in scored[:max(1, min(int(top_k or 10), 50))]]

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
                # Keep the edge subtype aligned with the entity semantics.
                # Previously every non-person/non-place entity (including
                # dates and atmosphere labels) was written as
                # MENTIONS_OBJECT, which polluted object precision and made
                # CAPTURED_ON recall look like zero.
                subtype_by_entity_type = {
                    "person": "MENTIONS_PERSON",
                    "place": "OCCURRED_AT",
                    "time": "CAPTURED_ON",
                    "date": "CAPTURED_ON",
                    "object": "MENTIONS_OBJECT",
                }
                subtype = subtype_by_entity_type.get(entity_type)
                if not subtype:
                    # Unknown semantic entity types stay available in the
                    # EVENT attributes but do not become a false object edge.
                    continue
                builder.graph_db.add_link(Link(
                    source_node_id=frame_id,
                    target_node_id=node_id,
                    link_type=LinkType.ENTITY,
                    properties={"sub_type": subtype,
                                "confidence": float(entity.get("confidence") or 1.0),
                                "confidence_score": float(entity.get("confidence") or 1.0),
                                "evidence_tier": "supported"},
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
                        "confidence_score": min(1.0, float(entity.get("confidence") or 1.0)),
                        "evidence_tier": "confirmed",
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
            attrs = node.attributes or {}
            if node.node_type.value == "EVENT":
                event_id = str(attrs.get("event_id") or "")
            elif node.node_type.value == "SESSION":
                event_id = str(attrs.get("clip_uid") or "")
            else:
                continue
            event = events.get(event_id)
            if not event:
                continue
            title = str(event.get("title") or "").strip()
            summary = str(event.get("summary") or "").strip()
            if title and node.node_type.value == "SESSION":
                node.summary = title if not summary else f"{title}\n{summary}"
            # EVENT nodes are what the query engine returns.  Project the
            # session's event evidence onto them as well, so the graph and
            # ordinary retrieval expose one consistent memory record.
            if summary and not attrs.get("event_summary"):
                attrs["event_summary"] = summary
            if title and not attrs.get("event_title"):
                attrs["event_title"] = title
            attrs["sentrix_event_summary"] = summary
            attrs["sentrix_event_id"] = event_id
            attrs["sentrix_event_title"] = title
            attrs["sentrix_event_place"] = event.get("place")
            attrs["sentrix_event_time_start"] = event.get("time_start")
            attrs["sentrix_event_time_end"] = event.get("time_end")
            node.attributes = attrs
