import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone


def make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def json_value(value, fallback):
    return json.dumps(value if value is not None else fallback, ensure_ascii=False)


class MemoryStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self):
        self.connection.close()

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                path TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'queued',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES assets(id),
                captured_at TEXT,
                source_type TEXT NOT NULL,
                caption TEXT,
                activity TEXT,
                place TEXT,
                people_json TEXT NOT NULL DEFAULT '[]',
                objects_json TEXT NOT NULL DEFAULT '[]',
                ocr_text TEXT NOT NULL DEFAULT '',
                event_type TEXT,
                transcript TEXT,
                confidence REAL NOT NULL DEFAULT 0,
                raw_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                event_type TEXT,
                time_start TEXT,
                time_end TEXT,
                place TEXT,
                activity TEXT,
                summary TEXT,
                participants_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_observations (
                event_id TEXT NOT NULL REFERENCES events(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                PRIMARY KEY(event_id, observation_id)
            );
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                source_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                family_role TEXT,
                summary TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_revisions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                field_name TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                source TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_clusters (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                entity_id TEXT REFERENCES entities(id),
                representative_embedding_json TEXT NOT NULL DEFAULT '[]',
                member_count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_instances (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES assets(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                cluster_id TEXT NOT NULL REFERENCES face_clusters(id),
                bbox_json TEXT NOT NULL DEFAULT '[]',
                embedding_json TEXT NOT NULL DEFAULT '[]',
                detection_confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                face_instance_id TEXT REFERENCES face_instances(id),
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(entity_id, observation_id, face_instance_id)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                subject_entity_id TEXT NOT NULL REFERENCES entities(id),
                predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL REFERENCES entities(id),
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_relationship_id TEXT REFERENCES relationships(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_vectors (
                id TEXT PRIMARY KEY,
                space TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                model_name TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(space, source_type, source_id, model_name)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                valid_from TEXT,
                valid_to TEXT,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_fact_id TEXT REFERENCES facts(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                outline_json TEXT NOT NULL DEFAULT '[]',
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS invites (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                expires_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_observations_asset ON observations(asset_id);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(time_start);
            CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate ON facts(subject, predicate);
            CREATE INDEX IF NOT EXISTS idx_face_instances_cluster ON face_instances(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_observation ON entity_mentions(observation_id);
            CREATE INDEX IF NOT EXISTS idx_relationship_subject_object ON relationships(subject_entity_id, object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_space ON memory_vectors(space);
            """
        )
        self.connection.commit()
        self._migrate_legacy_persons()

    def _migrate_legacy_persons(self):
        """Keep the first prototype's person candidates visible in the native entity view."""
        rows = self._rows("SELECT * FROM persons")
        timestamp = now_iso()
        for person in rows:
            entity_id = f"entity_{person['id']}"
            self.connection.execute(
                """INSERT OR IGNORE INTO entities(
                    id, entity_type, canonical_name, status, family_role, summary, confidence, created_at, updated_at
                ) VALUES (?, 'person', ?, ?, NULL, ?, ?, ?, ?)""",
                (entity_id, person["name"], person["status"], "历史人物候选", float(person["confidence"] or 0), person["created_at"] or timestamp, person["updated_at"] or timestamp),
            )
        self.connection.commit()

    def count(self, table):
        if table not in {"assets", "observations", "events", "event_observations", "persons", "entities", "entity_revisions", "face_clusters", "face_instances", "entity_mentions", "relationships", "memory_vectors", "facts", "stories", "invites"}:
            raise ValueError("unsupported table")
        return self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def _row(self, query, params=()):
        row = self.connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def _rows(self, query, params=()):
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def _decode(self, row, fields):
        if not row:
            return row
        result = dict(row)
        for field in fields:
            try:
                result[field] = json.loads(result[field] or "[]")
            except (TypeError, json.JSONDecodeError):
                result[field] = []
        return result

    def create_asset(self, asset_id, file_name, media_type, path, mime_type=None, size_bytes=0, metadata=None):
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO assets(id, file_name, media_type, path, mime_type, size_bytes, metadata_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (asset_id, file_name, media_type, path, mime_type, size_bytes, json_value(metadata, {}), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_asset(asset_id)

    def get_asset(self, asset_id):
        return self._decode(self._row("SELECT * FROM assets WHERE id = ?", (asset_id,)), ["metadata_json"])

    def list_assets(self, media_type=None, status=None, limit=200):
        clauses = []
        params = []
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._rows(f"SELECT * FROM assets {where} ORDER BY created_at DESC LIMIT ?", params)
        return [self._decode(row, ["metadata_json"]) for row in rows]

    def update_asset(self, asset_id, status, metadata=None):
        self.connection.execute(
            "UPDATE assets SET status = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
            (status, json_value(metadata, {}), now_iso(), asset_id),
        )
        self.connection.commit()
        return self.get_asset(asset_id)

    def add_observation(self, asset_id, data):
        observation_id = data.get("id") or make_id("obs")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO observations(
                id, asset_id, captured_at, source_type, caption, activity, place,
                people_json, objects_json, ocr_text, event_type, transcript, confidence, raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                observation_id,
                asset_id,
                data.get("captured_at"),
                data.get("source_type", "image"),
                data.get("caption", ""),
                data.get("activity", ""),
                data.get("place", ""),
                json_value(data.get("people"), []),
                json_value(data.get("objects"), []),
                data.get("ocr_text", ""),
                data.get("event_type", ""),
                data.get("transcript", ""),
                float(data.get("confidence", 0) or 0),
                json_value(data.get("raw"), {}),
                timestamp,
            ),
        )
        self.connection.commit()
        return self.get_observation(observation_id)

    def get_observation(self, observation_id):
        result = self._decode(self._row("SELECT * FROM observations WHERE id = ?", (observation_id,)), ["people_json", "objects_json", "raw_json"])
        if result:
            result["people"] = result.get("people_json", [])
            result["objects"] = result.get("objects_json", [])
            result["raw"] = result.get("raw_json", {})
        return result

    def list_observations(self, limit=100):
        rows = self._rows("SELECT * FROM observations ORDER BY created_at DESC LIMIT ?", (limit,))
        values = []
        for row in rows:
            value = self._decode(row, ["people_json", "objects_json", "raw_json"])
            value["people"] = value.get("people_json", [])
            value["objects"] = value.get("objects_json", [])
            value["raw"] = value.get("raw_json", {})
            values.append(value)
        return values

    def _event_candidates(self, observation):
        rows = self._rows("SELECT * FROM events WHERE status = 'active' ORDER BY time_start DESC")
        captured = parse_time(observation.get("captured_at"))
        place = (observation.get("place") or "").strip().lower()
        event_type = (observation.get("event_type") or "").strip().lower()
        candidates = []
        for row in rows:
            event_time = parse_time(row.get("time_start"))
            if captured and event_time and abs((captured - event_time).total_seconds()) > 6 * 3600:
                continue
            row_place = (row.get("place") or "").strip().lower()
            row_type = (row.get("event_type") or "").strip().lower()
            if place and row_place and place != row_place:
                continue
            if event_type and row_type and event_type != row_type:
                continue
            candidates.append(row)
        return candidates

    def merge_observation_into_event(self, observation):
        candidates = self._event_candidates(observation)
        event = candidates[0] if candidates else None
        people = list(dict.fromkeys((json.loads(event["participants_json"]) if event else []) + (observation.get("people") or [])))
        captured_at = observation.get("captured_at")
        if event:
            start = min(filter(None, [event.get("time_start"), captured_at])) if any([event.get("time_start"), captured_at]) else None
            end = max(filter(None, [event.get("time_end"), captured_at])) if any([event.get("time_end"), captured_at]) else None
            summary = event.get("summary") or observation.get("caption") or observation.get("activity") or "家庭事件"
            if observation.get("caption") and observation["caption"] not in summary:
                summary = f"{summary}；{observation['caption']}"
            self.connection.execute(
                """UPDATE events SET time_start = ?, time_end = ?, place = ?, activity = ?, summary = ?,
                participants_json = ?, confidence = MAX(confidence, ?), revision = revision + 1, updated_at = ? WHERE id = ?""",
                (start, end, event.get("place") or observation.get("place"), event.get("activity") or observation.get("activity"), summary, json_value(people, []), float(observation.get("confidence", 0) or 0), now_iso(), event["id"]),
            )
            event_id = event["id"]
        else:
            event_id = make_id("evt")
            title = observation.get("activity") or observation.get("event_type") or "未命名家庭事件"
            self.connection.execute(
                """INSERT INTO events(id, title, event_type, time_start, time_end, place, activity, summary,
                participants_json, confidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, title, observation.get("event_type"), captured_at, captured_at, observation.get("place"), observation.get("activity"), observation.get("caption") or title, json_value(people, []), float(observation.get("confidence", 0) or 0), now_iso(), now_iso()),
            )
        self.connection.execute("INSERT OR IGNORE INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event_id, observation["id"]))
        self.connection.commit()
        return self.get_event(event_id)

    def get_event(self, event_id):
        row = self._decode(self._row("SELECT * FROM events WHERE id = ?", (event_id,)), ["participants_json"])
        if row:
            row["participants"] = row.get("participants_json", [])
            row["observation_ids"] = [item["observation_id"] for item in self._rows("SELECT observation_id FROM event_observations WHERE event_id = ?", (event_id,))]
        return row

    def list_events(self, limit=100):
        rows = self._rows("SELECT * FROM events WHERE status = 'active' ORDER BY time_start DESC LIMIT ?", (limit,))
        return [self.get_event(row["id"]) for row in rows]

    def get_event_detail(self, event_id):
        event = self.get_event(event_id)
        if not event:
            return None
        observations = []
        for observation_id in event["observation_ids"]:
            observation = self.get_observation(observation_id)
            if observation:
                observation["asset"] = self.get_asset(observation["asset_id"])
                observations.append(observation)
        facts = [fact for fact in self.list_facts(500) if any(item["id"] in fact["evidence_ids_json"] for item in observations)]
        return {"event": event, "observations": observations, "facts": facts}

    def create_event(self, data):
        event_id = data.get("id") or make_id("evt")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO events(id, title, event_type, time_start, time_end, place, activity, summary,
            participants_json, confidence, status, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?)""",
            (event_id, data.get("title") or "未命名事件", data.get("event_type", "人工事件"), data.get("time_start"), data.get("time_end"), data.get("place", ""), data.get("activity", ""), data.get("summary", ""), json_value(data.get("participants"), []), float(data.get("confidence", 1) or 1), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_event(event_id)

    def update_event(self, event_id, fields):
        event = self.get_event(event_id)
        if not event:
            return None
        allowed = {"title", "event_type", "time_start", "time_end", "place", "activity", "summary", "status"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return event
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values()) + [now_iso(), event_id]
        self.connection.execute(f"UPDATE events SET {assignments}, revision = revision + 1, updated_at = ? WHERE id = ?", params)
        self.connection.commit()
        return self.get_event(event_id)

    def _fact_row(self, row):
        return self._decode(row, ["evidence_ids_json"])

    def get_fact(self, fact_id):
        return self._fact_row(self._row("SELECT * FROM facts WHERE id = ?", (fact_id,)))

    def list_facts(self, limit=200):
        rows = self._rows("SELECT * FROM facts ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [self._fact_row(row) for row in rows]

    def maintain_fact(self, subject, predicate, object_value, evidence_ids, confidence=0.5):
        active = self._row("SELECT * FROM facts WHERE subject = ? AND predicate = ? AND status = 'active' ORDER BY revision DESC LIMIT 1", (subject, predicate))
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        if active and active["object"] == object_value:
            existing = json.loads(active["evidence_ids_json"] or "[]")
            merged = list(dict.fromkeys(existing + evidence_ids))
            self.connection.execute("UPDATE facts SET evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ? WHERE id = ?", (json_value(merged, []), float(confidence), now_iso(), active["id"]))
            self.connection.commit()
            return self.get_fact(active["id"])
        status = "pending" if active else "active"
        fact_id = make_id("fact")
        revision = (active["revision"] + 1) if active else 1
        self.connection.execute(
            """INSERT INTO facts(id, subject, predicate, object, status, confidence, evidence_ids_json,
            supersedes_fact_id, revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact_id, subject, predicate, object_value, status, float(confidence), json_value(evidence_ids, []), active["id"] if active else None, revision, now_iso(), now_iso()),
        )
        self.connection.commit()
        return self.get_fact(fact_id)

    def confirm_fact(self, fact_id):
        fact = self.get_fact(fact_id)
        if not fact:
            raise KeyError(fact_id)
        self.connection.execute("UPDATE facts SET status = 'superseded', updated_at = ? WHERE subject = ? AND predicate = ? AND status = 'active' AND id != ?", (now_iso(), fact["subject"], fact["predicate"], fact_id))
        self.connection.execute("UPDATE facts SET status = 'active', updated_at = ? WHERE id = ?", (now_iso(), fact_id))
        self.connection.commit()
        return self.get_fact(fact_id)

    def reject_fact(self, fact_id):
        self.connection.execute("UPDATE facts SET status = 'retracted', updated_at = ? WHERE id = ?", (now_iso(), fact_id))
        self.connection.commit()
        return self.get_fact(fact_id)

    def list_persons(self):
        rows = self._rows("SELECT * FROM persons ORDER BY updated_at DESC")
        return [self._decode(row, ["source_json"]) for row in rows]

    def get_person(self, person_id):
        return self._decode(self._row("SELECT * FROM persons WHERE id = ?", (person_id,)), ["source_json"])

    def update_person(self, person_id, name=None, status=None):
        person = self.get_person(person_id)
        if not person:
            return None
        self.connection.execute("UPDATE persons SET name = ?, status = ?, updated_at = ? WHERE id = ?", (name or person["name"], status or person["status"], now_iso(), person_id))
        self.connection.commit()
        return self.get_person(person_id)

    def upsert_person(self, name, confidence=0, status="pending", source=None):
        existing = self._row("SELECT * FROM persons WHERE name = ?", (name,))
        if existing:
            self.connection.execute("UPDATE persons SET confidence = MAX(confidence, ?), status = ?, source_json = ?, updated_at = ? WHERE id = ?", (float(confidence), status if status == "confirmed" else existing["status"], json_value(source, {}), now_iso(), existing["id"]))
            self.connection.commit()
            return self._row("SELECT * FROM persons WHERE id = ?", (existing["id"],))
        person_id = make_id("person")
        self.connection.execute("INSERT INTO persons(id, name, status, confidence, source_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (person_id, name, status, float(confidence), json_value(source, {}), now_iso(), now_iso()))
        self.connection.commit()
        return self._row("SELECT * FROM persons WHERE id = ?", (person_id,))

    @staticmethod
    def _normalise_vector(vector):
        values = [float(value) for value in (vector or [])]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else []

    @staticmethod
    def _cosine(left, right):
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def upsert_vector(self, space, source_type, source_id, vector, model_name, metadata=None):
        values = self._normalise_vector(vector)
        if not values:
            return None
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO memory_vectors(id, space, source_type, source_id, vector_json, model_name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(space, source_type, source_id, model_name) DO UPDATE SET
                vector_json = excluded.vector_json, metadata_json = excluded.metadata_json, updated_at = excluded.updated_at""",
            (make_id("vec"), space, source_type, source_id, json_value(values, []), model_name, json_value(metadata, {}), timestamp, timestamp),
        )
        self.connection.commit()
        return self._row("SELECT * FROM memory_vectors WHERE space = ? AND source_type = ? AND source_id = ? AND model_name = ?", (space, source_type, source_id, model_name))

    def search_vectors(self, space, vector, limit=10):
        query = self._normalise_vector(vector)
        if not query:
            return []
        results = []
        for row in self._rows("SELECT * FROM memory_vectors WHERE space = ?", (space,)):
            try:
                candidate = json.loads(row["vector_json"] or "[]")
                score = self._cosine(query, candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            result = self._decode(row, ["metadata_json"])
            result["score"] = score
            results.append(result)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:max(1, limit)]

    def create_entity(self, name, entity_type="person", status="pending", family_role=None, confidence=0.0, summary=""):
        entity_id = make_id("entity")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO entities(id, entity_type, canonical_name, status, family_role, summary, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity_id, entity_type, name, status, family_role, summary, float(confidence or 0), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_entity(entity_id)

    def list_entities(self, status=None):
        params = [status] if status else []
        where = "WHERE status = ?" if status else ""
        entities = self._rows(f"SELECT * FROM entities {where} ORDER BY updated_at DESC", params)
        for entity in entities:
            entity["cluster_count"] = self.connection.execute("SELECT COUNT(*) FROM face_clusters WHERE entity_id = ?", (entity["id"],)).fetchone()[0]
            entity["mention_count"] = self.connection.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = ?", (entity["id"],)).fetchone()[0]
            entity["relationship_count"] = self.connection.execute("SELECT COUNT(*) FROM relationships WHERE (subject_entity_id = ? OR object_entity_id = ?) AND status != 'retracted'", (entity["id"], entity["id"])).fetchone()[0]
        return entities

    def get_entity(self, entity_id):
        return self._row("SELECT * FROM entities WHERE id = ?", (entity_id,))

    def get_entity_detail(self, entity_id):
        entity = self.get_entity(entity_id)
        if not entity:
            return None
        clusters = self._rows("SELECT * FROM face_clusters WHERE entity_id = ? ORDER BY updated_at DESC", (entity_id,))
        for cluster in clusters:
            cluster["samples"] = self._rows(
                """SELECT fi.id, fi.asset_id, fi.observation_id, fi.bbox_json, fi.detection_confidence, a.file_name, a.media_type
                FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE fi.cluster_id = ? ORDER BY fi.created_at DESC LIMIT 12""",
                (cluster["id"],),
            )
        relationships = self.list_relationships(entity_id)
        facts = [fact for fact in self.list_facts(1000) if fact["subject"] == entity["canonical_name"] or fact["object"] == entity["canonical_name"]]
        return {"entity": entity, "clusters": clusters, "relationships": relationships, "facts": facts}

    def create_face_cluster(self, embedding, confidence=0.0):
        cluster_id = make_id("cluster")
        timestamp = now_iso()
        values = self._normalise_vector(embedding)
        self.connection.execute(
            """INSERT INTO face_clusters(id, representative_embedding_json, member_count, confidence, created_at, updated_at)
            VALUES (?, ?, 0, ?, ?, ?)""",
            (cluster_id, json_value(values, []), float(confidence or 0), timestamp, timestamp),
        )
        self.connection.commit()
        entity = self.create_entity(f"待确认人物簇 · {cluster_id}", "person", "pending", None, float(confidence or 0), "由 buffalo_l embedding 生成，等待用户确认")
        self.connection.execute("UPDATE face_clusters SET entity_id = ? WHERE id = ?", (entity["id"], cluster_id))
        self.connection.commit()
        return self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))

    def add_face_instance(self, asset_id, observation_id, face, threshold=0.45, model_name="buffalo_l"):
        embedding = self._normalise_vector(face.get("embedding"))
        clusters = self._rows("SELECT * FROM face_clusters WHERE status IN ('pending', 'confirmed')")
        best = None
        best_score = 0.0
        for cluster in clusters:
            representative = json.loads(cluster["representative_embedding_json"] or "[]")
            score = self._cosine(embedding, representative)
            if score > best_score:
                best, best_score = cluster, score
        if not best or best_score < threshold:
            best = self.create_face_cluster(embedding, face.get("confidence", 0))
        elif embedding:
            members = self._rows("SELECT embedding_json FROM face_instances WHERE cluster_id = ?", (best["id"],))
            vectors = [json.loads(item["embedding_json"] or "[]") for item in members if item["embedding_json"]]
            vectors.append(embedding)
            mean = [sum(values) / len(values) for values in zip(*vectors)] if vectors else embedding
            self.connection.execute("UPDATE face_clusters SET representative_embedding_json = ?, member_count = ?, confidence = MAX(confidence, ?), revision = revision + 1, updated_at = ? WHERE id = ?", (json_value(self._normalise_vector(mean), []), len(vectors), float(face.get("confidence", 0) or 0), now_iso(), best["id"]))
        instance_id = make_id("face")
        self.connection.execute(
            """INSERT INTO face_instances(id, asset_id, observation_id, cluster_id, bbox_json, embedding_json, detection_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (instance_id, asset_id, observation_id, best["id"], json_value(face.get("bbox"), []), json_value(embedding, []), float(face.get("confidence", 0) or 0), now_iso()),
        )
        self.connection.execute("UPDATE face_clusters SET member_count = (SELECT COUNT(*) FROM face_instances WHERE cluster_id = ?), updated_at = ? WHERE id = ?", (best["id"], now_iso(), best["id"]))
        self.connection.commit()
        self.upsert_vector("visual", "face_instance", instance_id, embedding, model_name, {"cluster_id": best["id"], "asset_id": asset_id, "observation_id": observation_id})
        return {"id": instance_id, "cluster_id": best["id"], "score": best_score, "embedding": embedding}

    def list_face_clusters(self, status=None):
        params = [status] if status else []
        where = "WHERE fc.status = ?" if status else ""
        rows = self._rows(f"""SELECT fc.*, e.canonical_name, e.family_role, e.status AS entity_status
            FROM face_clusters fc LEFT JOIN entities e ON e.id = fc.entity_id {where} ORDER BY fc.updated_at DESC""", params)
        for row in rows:
            row["samples"] = self._rows("""SELECT fi.id, fi.asset_id, fi.observation_id, fi.bbox_json, fi.detection_confidence, a.file_name, a.media_type
                FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE fi.cluster_id = ? ORDER BY fi.created_at DESC LIMIT 12""", (row["id"],))
        return rows

    def confirm_face_cluster(self, cluster_id, name, family_role=None):
        cluster = self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))
        if not cluster:
            return None
        entity = self.get_entity(cluster["entity_id"]) if cluster["entity_id"] else None
        if not entity:
            entity = self.create_entity(name, "person", "confirmed", family_role, 1.0, "用户确认的人物实体")
        else:
            self.connection.execute("UPDATE entities SET canonical_name = ?, status = 'confirmed', family_role = ?, confidence = MAX(confidence, 1), updated_at = ? WHERE id = ?", (name, family_role, now_iso(), entity["id"]))
            entity = self.get_entity(entity["id"])
        self.connection.execute("UPDATE face_clusters SET status = 'confirmed', entity_id = ?, updated_at = ?, revision = revision + 1 WHERE id = ?", (entity["id"], now_iso(), cluster_id))
        instances = self._rows("SELECT * FROM face_instances WHERE cluster_id = ?", (cluster_id,))
        observation_ids = []
        for instance in instances:
            observation_ids.append(instance["observation_id"])
            self.connection.execute("INSERT OR IGNORE INTO entity_mentions(id, entity_id, observation_id, face_instance_id, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)", (make_id("mention"), entity["id"], instance["observation_id"], instance["id"], float(instance["detection_confidence"] or 0), now_iso()))
            self._add_entity_to_observation(instance["observation_id"], entity)
        if family_role:
            self.maintain_fact(name, "家庭角色", family_role, list(dict.fromkeys(observation_ids)), confidence=1.0)
        self.connection.execute("UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?", (f"已确认人物，出现在 {len(set(observation_ids))} 条观察中", now_iso(), entity["id"]))
        self.connection.commit()
        self._refresh_event_participants(observation_ids)
        self._propose_cooccurrence_relationships(observation_ids)
        return self.get_entity_detail(entity["id"])

    def _add_entity_to_observation(self, observation_id, entity):
        observation = self.get_observation(observation_id)
        if not observation:
            return
        people = observation.get("people") or []
        values = []
        found = False
        for person in people:
            if isinstance(person, dict) and person.get("entity_id") == entity["id"]:
                found = True
            values.append(person)
        if not found:
            values.append({"entity_id": entity["id"], "name": entity["canonical_name"], "status": entity["status"]})
        self.connection.execute("UPDATE observations SET people_json = ? WHERE id = ?", (json_value(values, []), observation_id))

    def _refresh_event_participants(self, observation_ids):
        for observation_id in set(observation_ids):
            rows = self._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (observation_id,))
            for row in rows:
                event = self.get_event(row["event_id"])
                participants = event.get("participants") or []
                observation = self.get_observation(observation_id) or {}
                merged = participants[:]
                for person in observation.get("people") or []:
                    key = person.get("entity_id") if isinstance(person, dict) else person
                    if key and not any((item.get("entity_id") if isinstance(item, dict) else item) == key for item in merged):
                        merged.append(person)
                self.connection.execute("UPDATE events SET participants_json = ?, revision = revision + 1, updated_at = ? WHERE id = ?", (json_value(merged, []), now_iso(), event["id"]))

    def _propose_cooccurrence_relationships(self, observation_ids):
        for observation_id in set(observation_ids):
            entities = self._rows("""SELECT DISTINCT em.entity_id FROM entity_mentions em
                WHERE em.observation_id = ? ORDER BY em.entity_id""", (observation_id,))
            for index, left in enumerate(entities):
                for right in entities[index + 1:]:
                    self.create_relationship(left["entity_id"], "共同出现在同一观察中", right["entity_id"], [observation_id], 0.5, "pending")

    def reject_face_cluster(self, cluster_id):
        self.connection.execute("UPDATE face_clusters SET status = 'rejected', updated_at = ?, revision = revision + 1 WHERE id = ?", (now_iso(), cluster_id))
        self.connection.commit()
        return self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))

    def list_relationships(self, entity_id=None):
        if entity_id:
            rows = self._rows("""SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
                FROM relationships r JOIN entities s ON s.id = r.subject_entity_id JOIN entities o ON o.id = r.object_entity_id
                WHERE r.subject_entity_id = ? OR r.object_entity_id = ? ORDER BY r.updated_at DESC""", (entity_id, entity_id))
        else:
            rows = self._rows("""SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
                FROM relationships r JOIN entities s ON s.id = r.subject_entity_id JOIN entities o ON o.id = r.object_entity_id ORDER BY r.updated_at DESC""")
        return [self._decode(row, ["evidence_ids_json"]) for row in rows]

    def create_relationship(self, subject_entity_id, predicate, object_entity_id, evidence_ids=None, confidence=0.5, status="pending"):
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        existing = self._row("SELECT * FROM relationships WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ? AND status IN ('active', 'pending') ORDER BY revision DESC LIMIT 1", (subject_entity_id, predicate, object_entity_id))
        if existing:
            old_evidence = json.loads(existing["evidence_ids_json"] or "[]")
            merged = list(dict.fromkeys(old_evidence + evidence_ids))
            next_status = "active" if status == "active" else existing["status"]
            self.connection.execute("UPDATE relationships SET status = ?, evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ?, revision = revision + 1 WHERE id = ?", (next_status, json_value(merged, []), float(confidence or 0), now_iso(), existing["id"]))
            self.connection.commit()
            return next((item for item in self.list_relationships() if item["id"] == existing["id"]), None)
        relationship_id = make_id("rel")
        revision = 1
        self.connection.execute("""INSERT INTO relationships(id, subject_entity_id, predicate, object_entity_id, status, confidence, evidence_ids_json, supersedes_relationship_id, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""", (relationship_id, subject_entity_id, predicate, object_entity_id, status, float(confidence or 0), json_value(evidence_ids, []), revision, now_iso(), now_iso()))
        self.connection.commit()
        return self.list_relationships()[0]

    def confirm_relationship(self, relationship_id):
        relationship = self._row("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
        if not relationship:
            return None
        self.connection.execute("UPDATE relationships SET status = 'superseded', updated_at = ? WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ? AND status = 'active' AND id != ?", (now_iso(), relationship["subject_entity_id"], relationship["predicate"], relationship["object_entity_id"], relationship_id))
        self.connection.execute("UPDATE relationships SET status = 'active', updated_at = ? WHERE id = ?", (now_iso(), relationship_id))
        self.connection.commit()
        return next((item for item in self.list_relationships() if item["id"] == relationship_id), None)

    def create_story(self, data):
        story_id = data.get("id") or make_id("story")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO stories(id, title, status, outline_json, event_ids_json, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (story_id, data.get("title") or "未命名故事", data.get("status", "draft"), json_value(data.get("outline"), []), json_value(data.get("event_ids"), []), data.get("content", ""), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_story(story_id)

    def get_story(self, story_id):
        story = self._decode(self._row("SELECT * FROM stories WHERE id = ?", (story_id,)), ["outline_json", "event_ids_json"])
        if story:
            story["outline"] = story.pop("outline_json")
            story["event_ids"] = story.pop("event_ids_json")
        return story

    def list_stories(self):
        rows = self._rows("SELECT * FROM stories ORDER BY updated_at DESC")
        return [self.get_story(row["id"]) for row in rows]

    def update_story(self, story_id, fields):
        story = self.get_story(story_id)
        if not story:
            return None
        values = {}
        for key in ("title", "status", "content"):
            if key in fields:
                values[key] = fields[key]
        if "outline" in fields:
            values["outline_json"] = json_value(fields["outline"], [])
        if "event_ids" in fields:
            values["event_ids_json"] = json_value(fields["event_ids"], [])
        if not values:
            return story
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.connection.execute(f"UPDATE stories SET {assignments}, updated_at = ? WHERE id = ?", (*values.values(), now_iso(), story_id))
        self.connection.commit()
        return self.get_story(story_id)

    def delete_story(self, story_id):
        story = self.get_story(story_id)
        if not story:
            return None
        self.connection.execute("DELETE FROM stories WHERE id = ?", (story_id,))
        self.connection.commit()
        return {"id": story_id, "deleted": True}

    def create_invite(self, label):
        invite_id = make_id("invite")
        token = uuid.uuid4().hex
        self.connection.execute("INSERT INTO invites(id, label, token, created_at) VALUES (?, ?, ?, ?)", (invite_id, label or "家庭成员", token, now_iso()))
        self.connection.commit()
        return self._row("SELECT * FROM invites WHERE id = ?", (invite_id,))
