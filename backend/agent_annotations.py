"""Agent-owned, auditable annotations stored separately from canonical memory."""

from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import threading
import uuid


SCHEMA_VERSION = 1
_MIGRATION_LOCK = threading.Lock()

MIGRATION_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS agent_schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL,
        checksum TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_user_assertions (
        id TEXT PRIMARY KEY,
        scope_id TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        conversation_id TEXT,
        subject_entity_id TEXT,
        event_id TEXT,
        observation_id TEXT,
        asset_id TEXT,
        assertion_text TEXT NOT NULL,
        normalized_value TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        asserted_by TEXT NOT NULL DEFAULT 'user',
        visibility_mode TEXT NOT NULL DEFAULT 'private',
        created_at TEXT NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        supersedes_id TEXT,
        request_id TEXT,
        idempotency_key TEXT,
        UNIQUE(scope_id, actor_id, idempotency_key),
        UNIQUE(scope_id, request_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_impressions (
        id TEXT PRIMARY KEY,
        scope_id TEXT NOT NULL,
        subject_entity_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        impression_text TEXT NOT NULL,
        epistemic_type TEXT NOT NULL DEFAULT 'agent_impression',
        support_event_ids_json TEXT NOT NULL DEFAULT '[]',
        support_observation_ids_json TEXT NOT NULL DEFAULT '[]',
        confidence REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        created_by TEXT NOT NULL DEFAULT 'impression_proposer',
        visibility_mode TEXT NOT NULL DEFAULT 'private',
        created_at TEXT NOT NULL,
        last_reviewed_at TEXT,
        supersedes_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_proactivity_preferences (
        id TEXT PRIMARY KEY,
        scope_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        level INTEGER NOT NULL DEFAULT 2,
        ignore_streak INTEGER NOT NULL DEFAULT 0,
        acceptance_count INTEGER NOT NULL DEFAULT 0,
        dismissal_count INTEGER NOT NULL DEFAULT 0,
        last_outcome TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE(scope_id, viewer_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_scene_cooldowns (
        id TEXT PRIMARY KEY,
        scope_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        scene_key TEXT NOT NULL,
        offered_at TEXT NOT NULL,
        cooldown_until TEXT NOT NULL,
        outcome TEXT NOT NULL,
        repetition_count INTEGER NOT NULL DEFAULT 1,
        UNIQUE(scope_id, viewer_id, scene_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_claim_conflicts (
        id TEXT PRIMARY KEY,
        scope_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        subject_ref TEXT NOT NULL,
        claim_ref TEXT NOT NULL,
        conflicting_ref TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        resolved_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_annotation_visibility (
        annotation_id TEXT NOT NULL,
        viewer_id TEXT NOT NULL,
        granted_at TEXT NOT NULL,
        revoked_at TEXT,
        PRIMARY KEY(annotation_id, viewer_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_assertions_scope_viewer ON agent_user_assertions(scope_id, viewer_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_agent_impressions_subject ON agent_impressions(scope_id, viewer_id, subject_entity_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_agent_cooldowns_viewer ON agent_scene_cooldowns(scope_id, viewer_id, cooldown_until)",
    "CREATE INDEX IF NOT EXISTS idx_agent_conflicts_subject ON agent_claim_conflicts(scope_id, viewer_id, subject_ref, status)",
)

MIGRATION_CHECKSUM = hashlib.sha256("\n".join(MIGRATION_STATEMENTS).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class AnnotationStore:
    """Own Agent annotations without changing canonical memory tables."""

    def __init__(self, connection, enabled=None):
        self.connection = connection
        self.available = False
        self.error = None
        if enabled is None:
            enabled = os.getenv("SENTRIX_ANNOTATION_STORE", "1").lower() not in {"0", "false", "off"}
        if not enabled:
            return
        self._migrate()

    def _migrate(self):
        try:
            with _MIGRATION_LOCK:
                if self.connection.in_transaction:
                    raise RuntimeError("annotation migration requires an idle connection")
                self.connection.execute("PRAGMA busy_timeout = 30000")
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute(MIGRATION_STATEMENTS[0])
                rows = self.connection.execute(
                    "SELECT version, checksum FROM agent_schema_migrations ORDER BY version"
                ).fetchall()
                current = {int(row[0]): str(row[1]) for row in rows}
                for version, checksum in current.items():
                    if version > SCHEMA_VERSION or (version == SCHEMA_VERSION and checksum != MIGRATION_CHECKSUM):
                        raise RuntimeError(f"unsupported agent schema version/checksum: {version}")
                if SCHEMA_VERSION not in current:
                    for statement in MIGRATION_STATEMENTS[1:]:
                        self.connection.execute(statement)
                    self.connection.execute(
                        "INSERT INTO agent_schema_migrations(version, applied_at, checksum) VALUES (?, ?, ?)",
                        (SCHEMA_VERSION, _now(), MIGRATION_CHECKSUM),
                    )
                self.connection.commit()
                self.available = True
        except Exception as error:
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass
            self.error = str(error)
            self.available = False

    def _row(self, query, params=()):
        row = self.connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def _require_available(self):
        return self.available

    def _reference_exists(self, table, value):
        if not value:
            return True
        try:
            return self.connection.execute(f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1", (value,)).fetchone() is not None
        except sqlite3.OperationalError:
            return False

    def _assertion_status(self, subject_entity_id=None, event_id=None, observation_id=None, asset_id=None):
        references = (
            ("entities", subject_entity_id),
            ("events", event_id),
            ("observations", observation_id),
            ("assets", asset_id),
        )
        return "orphaned" if any(value and not self._reference_exists(table, value) for table, value in references) else "pending"

    def record_user_assertion(
        self, *, scope_id, actor_id, viewer_id, conversation_id, assertion_text,
        subject_entity_id=None, event_id=None, observation_id=None, asset_id=None,
        normalized_value=None, supersedes_id=None, request_id=None, idempotency_key=None,
    ):
        if not self._require_available():
            return None
        if not idempotency_key:
            raise ValueError("user assertion requires an idempotency key")
        existing = self._row(
            """SELECT * FROM agent_user_assertions
            WHERE scope_id = ? AND actor_id = ? AND idempotency_key = ?""",
            (scope_id, actor_id, idempotency_key),
        )
        if existing:
            return existing
        values = (
            _id("assertion"), scope_id, actor_id, viewer_id, conversation_id,
            subject_entity_id, event_id, observation_id, asset_id, assertion_text,
            normalized_value, self._assertion_status(subject_entity_id, event_id, observation_id, asset_id),
            "user", "private", _now(), None, None, supersedes_id, request_id, idempotency_key,
        )
        try:
            self.connection.execute(
                """INSERT INTO agent_user_assertions(
                    id, scope_id, actor_id, viewer_id, conversation_id,
                    subject_entity_id, event_id, observation_id, asset_id,
                    assertion_text, normalized_value, status, asserted_by,
                    visibility_mode, created_at, valid_from, valid_to,
                    supersedes_id, request_id, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            self.connection.rollback()
            existing = self._row(
                """SELECT * FROM agent_user_assertions
                WHERE scope_id = ? AND actor_id = ? AND idempotency_key = ?""",
                (scope_id, actor_id, idempotency_key),
            )
            if existing:
                return existing
            raise
        return self._row("SELECT * FROM agent_user_assertions WHERE id = ?", (values[0],))

    def record_impression(self, *, scope_id, subject_entity_id, viewer_id, impression_text,
                          support_event_ids, support_observation_ids=(), confidence=0,
                          status="pending"):
        if not self._require_available():
            return None
        impression_id = _id("impression")
        self.connection.execute(
            """INSERT INTO agent_impressions(
                id, scope_id, subject_entity_id, viewer_id, impression_text,
                support_event_ids_json, support_observation_ids_json, confidence,
                status, created_at, last_reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (impression_id, scope_id, subject_entity_id, viewer_id, impression_text,
             json.dumps(list(support_event_ids or []), ensure_ascii=False),
             json.dumps(list(support_observation_ids or []), ensure_ascii=False),
             float(confidence), status, _now(), _now()),
        )
        self.connection.commit()
        return self._row("SELECT * FROM agent_impressions WHERE id = ?", (impression_id,))

    def upsert_preference(self, scope_id, viewer_id, *, enabled=True, level=2,
                          ignore_streak=0, acceptance_count=0, dismissal_count=0,
                          last_outcome=None):
        if not self._require_available():
            return None
        self.connection.execute(
            """INSERT INTO agent_proactivity_preferences(
                id, scope_id, viewer_id, enabled, level, ignore_streak,
                acceptance_count, dismissal_count, last_outcome, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id, viewer_id) DO UPDATE SET
                enabled = excluded.enabled, level = excluded.level,
                ignore_streak = excluded.ignore_streak,
                acceptance_count = excluded.acceptance_count,
                dismissal_count = excluded.dismissal_count,
                last_outcome = excluded.last_outcome,
                updated_at = excluded.updated_at""",
            (_id("preference"), scope_id, viewer_id, int(bool(enabled)), int(level), int(ignore_streak),
             int(acceptance_count), int(dismissal_count), last_outcome, _now()),
        )
        self.connection.commit()
        return self._row(
            "SELECT * FROM agent_proactivity_preferences WHERE scope_id = ? AND viewer_id = ?",
            (scope_id, viewer_id),
        )

    def get_preference(self, scope_id, viewer_id):
        if not self._require_available():
            return None
        return self._row(
            "SELECT * FROM agent_proactivity_preferences WHERE scope_id = ? AND viewer_id = ?",
            (scope_id, viewer_id),
        )

    def get_scene_cooldown(self, scope_id, viewer_id, scene_key):
        if not self._require_available():
            return None
        return self._row(
            """SELECT * FROM agent_scene_cooldowns
            WHERE scope_id = ? AND viewer_id = ? AND scene_key = ?""",
            (scope_id, viewer_id, scene_key),
        )

    def record_proactivity_outcome(self, scope_id, viewer_id, scene_key, outcome,
                                   *, cooldown_until, enabled=None):
        """Persist viewer-level control state without touching canonical memory."""
        if not self._require_available():
            return None
        current = self.get_preference(scope_id, viewer_id) or {
            "enabled": 1, "level": 2, "ignore_streak": 0,
            "acceptance_count": 0, "dismissal_count": 0,
        }
        accepted = outcome == "accepted"
        ignored = outcome in {"ignored", "dismissed", "repeated"}
        ignore_streak = 0 if accepted else int(current.get("ignore_streak", 0) or 0) + (1 if ignored else 0)
        level = int(current.get("level", 2) or 2)
        if ignore_streak >= 2:
            level = 0
        if outcome == "enabled":
            enabled = True
            level = max(level, 1)
            ignore_streak = 0
        if outcome == "disabled":
            enabled = False
            level = 0
        self.upsert_preference(
            scope_id, viewer_id,
            enabled=bool(current.get("enabled", 1)) if enabled is None else enabled,
            level=level,
            ignore_streak=ignore_streak,
            acceptance_count=int(current.get("acceptance_count", 0) or 0) + (1 if accepted else 0),
            dismissal_count=int(current.get("dismissal_count", 0) or 0) + (1 if ignored else 0),
            last_outcome=outcome,
        )
        return self.upsert_scene_cooldown(
            scope_id, viewer_id, scene_key, _now(), cooldown_until, outcome,
        )

    def upsert_scene_cooldown(self, scope_id, viewer_id, scene_key, offered_at, cooldown_until, outcome):
        if not self._require_available():
            return None
        self.connection.execute(
            """INSERT INTO agent_scene_cooldowns(
                id, scope_id, viewer_id, scene_key, offered_at,
                cooldown_until, outcome, repetition_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(scope_id, viewer_id, scene_key) DO UPDATE SET
                offered_at = excluded.offered_at,
                cooldown_until = excluded.cooldown_until,
                outcome = excluded.outcome,
                repetition_count = agent_scene_cooldowns.repetition_count + 1""",
            (_id("cooldown"), scope_id, viewer_id, scene_key, offered_at, cooldown_until, outcome),
        )
        self.connection.commit()
        return self._row(
            "SELECT * FROM agent_scene_cooldowns WHERE scope_id = ? AND viewer_id = ? AND scene_key = ?",
            (scope_id, viewer_id, scene_key),
        )
