"""Adapters that project Sentrix memory rows into MAGMA frame points.

The graph builder in this package is intentionally independent from the
FastAPI request layer.  It accepts the ``frame_points`` structure used by
MAGMA-V3, so this module is the only place that knows how to read the
Sentrix SQLite schema.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


KEYFRAME_KINDS = ("video_keyframe", "video_keyframe_webp")


def _loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


class SentrixFrameProvider:
    """Read keyframes from ``sentrix.db`` and build MAGMA-compatible points."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).resolve())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_frames(
        self,
        scope_id: str | None = None,
        video_asset_id: str | None = None,
        include_images: bool = False,
    ) -> list[dict]:
        with self._connect() as conn:
            rows = self._asset_rows(conn, scope_id, video_asset_id, include_images)
            return [self._frame_point(conn, row) for row in rows]

    def _asset_rows(
        self,
        conn: sqlite3.Connection,
        scope_id: str | None,
        video_asset_id: str | None,
        include_images: bool,
    ) -> list[sqlite3.Row]:
        clauses = ["a.media_type = 'image'"]
        params: list[Any] = []
        if include_images:
            clauses.append("(a.derived_kind IN (?, ?) OR a.derived_kind IS NULL)")
            params.extend(KEYFRAME_KINDS)
        else:
            placeholders = ",".join("?" for _ in KEYFRAME_KINDS)
            clauses.append(f"a.derived_kind IN ({placeholders})")
            params.extend(KEYFRAME_KINDS)
        if scope_id:
            clauses.append("a.scope_id = ?")
            params.append(scope_id)
        if video_asset_id:
            clauses.append("COALESCE(a.parent_asset_id, a.id) = ?")
            params.append(video_asset_id)

        return list(conn.execute(f"""
            SELECT a.id, a.scope_id, a.path, a.file_name, a.captured_at,
                   a.captured_location, a.parent_asset_id, a.derived_kind,
                   a.source_timestamp_sec, a.source_frame_index,
                   a.source_scene_index,
                   p.captured_at AS parent_captured_at,
                   p.captured_location AS parent_captured_location,
                   p.file_name AS parent_file_name
            FROM assets a
            LEFT JOIN assets p ON p.id = a.parent_asset_id
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(a.parent_asset_id, a.id), a.source_scene_index,
                     a.source_timestamp_sec, a.source_frame_index, a.created_at
        """, params))

    def _frame_point(self, conn: sqlite3.Connection, asset: sqlite3.Row) -> dict:
        asset_id = str(asset["id"])
        observation = self._observation(conn, asset_id)
        event = self._event_for_observation(conn, observation.get("id"))
        vector = self._visual_vector(conn, asset_id)
        entities = self._entities(conn, asset_id, event.get("id") if event else None)

        canonical = dict(observation.get("canonical") or {})
        raw = dict(observation.get("raw") or {})
        objects = self._objects(observation, canonical)
        relations = self._relations(observation, canonical, raw)
        object_labels = list(dict.fromkeys(item["label"] for item in objects if item.get("label")))
        relation_labels = [
            f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}".strip()
            for item in relations
        ]

        parent_id = asset["parent_asset_id"] or asset_id
        scene_index = int(asset["source_scene_index"] or 0)
        event_id = str(event["id"]) if event else ""
        clip_uid = event_id or f"{parent_id}:scene:{scene_index}"
        timestamp = float(asset["source_timestamp_sec"] or 0.0)

        person_entities = [item for item in entities if item.get("entity_type") == "person"]
        place_entity = next((item for item in entities if item.get("entity_type") == "place"), None)
        place = place_entity.get("canonical_name") if place_entity else (
            observation.get("place") or (event or {}).get("place") or ""
        )
        # GPS is often retained while reverse-geocoding is unavailable at
        # ingest time.  Resolve it lazily and append it to (rather than
        # replacing) the model's scene label: both "室内空间" and the real
        # administrative/landmark location remain searchable.
        location = str(asset["captured_location"] or asset["parent_captured_location"] or "")
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)", location)
        if match:
            try:
                from ..geocoding import default_reverse_geocoder
                geo = default_reverse_geocoder().lookup(
                    {"latitude": match.group(1), "longitude": match.group(2)},
                    filename=asset["file_name"],
                )
                geo_place = str(geo.get("label") or geo.get("name") or "").strip()
                if geo_place and geo_place not in str(place):
                    place = (str(place).strip() + " " + geo_place).strip()
            except Exception:
                pass

        payload = {
            "video_uid": str(parent_id),
            "clip_uid": clip_uid,
            "frame_uid": asset_id,
            "frame_idx": int(asset["source_frame_index"] or 0),
            "frame_seq": int(asset["source_frame_index"] or scene_index or 0),
            "source_scene_index": scene_index,
            "clip_time_sec": timestamp,
            "video_time_sec": timestamp,
            "frame_path": str(asset["path"] or ""),
            "clip_path": str(asset["parent_file_name"] or ""),
            "narration_text": str(observation.get("caption") or (event or {}).get("title") or ""),
            "speech_text": str(observation.get("transcript") or ""),
            "action_text": str(observation.get("activity") or ""),
            "objects": objects,
            "relations": relations,
            "object_labels": object_labels,
            "relation_labels": list(dict.fromkeys(relation_labels)),
            "object_counts": dict(Counter(object_labels)),
            "relation_counts": dict(Counter(relation_labels)),
            "person_ids": [str(item["id"]) for item in person_entities],
            "person_labels": [str(item.get("canonical_name") or "") for item in person_entities],
            "entity_refs": entities,
            "place": str(place or ""),
            "captured_at": str(asset["captured_at"] or asset["parent_captured_at"] or
                                  (event or {}).get("time_start") or ""),
            "scope_id": str(asset["scope_id"] or "home-default"),
            "event_id": event_id,
            "source_asset_id": asset_id,
            "source_type": "sentrix_keyframe" if asset["derived_kind"] else "sentrix_image",
            "event_summary": str(event.get("summary") or "") if event else "",
            "event_title": str(event.get("title") or "") if event else "",
        }
        return {"id": asset_id, "vector": {"image": vector}, "payload": payload}

    def _observation(self, conn: sqlite3.Connection, asset_id: str) -> dict:
        row = conn.execute(
            "SELECT * FROM observations WHERE asset_id = ? ORDER BY created_at DESC, revision DESC LIMIT 1",
            (asset_id,),
        ).fetchone()
        if not row:
            return {"id": "", "canonical": {}, "raw": {}, "caption": "", "activity": "",
                    "place": "", "transcript": "", "objects": [], "spatial_relations": []}
        result = dict(row)
        canonical = _loads(result.get("canonical_json"), {})
        if isinstance(canonical, dict):
            inner = canonical.get("canonical")
            if isinstance(inner, dict):
                merged = dict(canonical)
                # Inner canonical values win only when the outer key is absent.
                for key, value in inner.items():
                    merged.setdefault(key, value)
                canonical = merged
        result["canonical"] = canonical if isinstance(canonical, dict) else {}
        raw = _loads(result.get("raw_json"), {})
        result["raw"] = raw if isinstance(raw, dict) else {}
        result["objects"] = _loads(result.get("objects_json"), [])
        result["spatial_relations"] = _loads(result.get("spatial_relations_json"), [])
        return result

    def _event_for_observation(self, conn: sqlite3.Connection, observation_id: str) -> dict | None:
        if not observation_id:
            return None
        row = conn.execute("""
            SELECT e.* FROM events e
            JOIN event_observations eo ON eo.event_id = e.id
            WHERE eo.observation_id = ?
            ORDER BY e.updated_at DESC LIMIT 1
        """, (observation_id,)).fetchone()
        return dict(row) if row else None

    def _visual_vector(self, conn: sqlite3.Connection, asset_id: str) -> list[float]:
        row = conn.execute("""
            SELECT vector_json FROM memory_vectors
            WHERE space = 'visual' AND source_type = 'asset' AND source_id = ?
            ORDER BY updated_at DESC LIMIT 1
        """, (asset_id,)).fetchone()
        if not row:
            return []
        vector = _loads(row["vector_json"], [])
        return [float(value) for value in vector] if isinstance(vector, list) else []

    def _entities(
        self, conn: sqlite3.Connection, asset_id: str, event_id: str | None
    ) -> list[dict]:
        rows = conn.execute("""
            SELECT DISTINCT e.id, e.entity_type, e.canonical_name, e.status,
                   e.family_role, eo.confidence
            FROM entity_observations eo
            JOIN entities e ON e.id = eo.entity_id
            WHERE eo.observation_id = (
                SELECT id FROM observations WHERE asset_id = ?
                ORDER BY created_at DESC, revision DESC LIMIT 1
            )
        """, (asset_id,)).fetchall()

        entity_ids = {str(row["id"]) for row in rows}
        if event_id:
            for row in conn.execute("""
                SELECT DISTINCT e.id, e.entity_type, e.canonical_name, e.status,
                       e.family_role, ep.confidence
                FROM event_participants ep
                JOIN entities e ON e.id = ep.person_id
                WHERE ep.event_id = ?
            """, (event_id,)):
                if str(row["id"]) not in entity_ids:
                    rows.append(row)
                    entity_ids.add(str(row["id"]))

        for row in conn.execute("""
            SELECT DISTINCT e.id, e.entity_type, e.canonical_name, e.status,
                   e.family_role, fi.detection_confidence AS confidence
            FROM face_instances fi
            JOIN face_clusters fc ON fc.id = fi.cluster_id
            JOIN entities e ON e.id = fc.entity_id
            WHERE fi.asset_id = ? AND fc.entity_id IS NOT NULL
        """, (asset_id,)):
            if str(row["id"]) not in entity_ids:
                rows.append(row)
                entity_ids.add(str(row["id"]))

        result = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["entity_type"] = str(item.get("entity_type") or "object")
            item["canonical_name"] = str(item.get("canonical_name") or item.get("family_role") or "未知实体")
            result.append(item)
        return result

    @staticmethod
    def _objects(observation: dict, canonical: dict) -> list[dict]:
        semantic = canonical.get("semantic") if isinstance(canonical.get("semantic"), dict) else {}
        raw_objects: Iterable[Any] = []
        raw_objects = list(observation.get("objects") or [])
        raw_objects.extend((semantic or {}).get("objects") or [])
        raw_objects.extend(canonical.get("objects") or [])

        objects: list[dict] = []
        seen = set()
        for item in raw_objects:
            if isinstance(item, str):
                item = {"label": item}
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("primary") or "").strip()
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            obj = dict(item)
            obj["label"] = label
            obj.setdefault("score", 1.0)
            objects.append(obj)
        return objects

    @staticmethod
    def _relations(observation: dict, canonical: dict, raw: dict) -> list[dict]:
        facts: list[Any] = []
        gamma = raw.get("gamma") if isinstance(raw.get("gamma"), dict) else {}
        facts.extend(gamma.get("facts") or [])
        facts.extend(canonical.get("facts") or [])
        facts.extend(raw.get("facts") or [])

        relations: list[dict] = []
        seen = set()
        for item in facts:
            if isinstance(item, dict):
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                obj = str(item.get("object") or "").strip()
                if subject and predicate and obj:
                    key = (subject, predicate, obj)
                    if key not in seen:
                        seen.add(key)
                        relation = dict(item)
                        relation.update({"subject": subject, "predicate": predicate, "object": obj})
                        relation.setdefault("score", 1.0)
                        relations.append(relation)

        # Sentrix stores free-form spatial phrases when no SGG triple is
        # available.  Keep them as a scene-level relation so keyword recall can
        # still find them, without pretending we know the exact subject/object.
        for phrase in observation.get("spatial_relations") or []:
            text = str(phrase or "").strip()
            if not text:
                continue
            key = ("画面", "包含", text)
            if key not in seen:
                seen.add(key)
                relations.append({"subject": "画面", "predicate": "包含", "object": text, "score": 1.0})
        return relations
