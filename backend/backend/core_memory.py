"""Phase 5 — Core Memory Card storage and lifecycle.

Agent-owned tables only.  Canonical facts continue to live in
``entities/semantic_profiles/semantic_claims/relationships``; a Core Memory
Item is a cached, epistemic-typed view that references the canonical source
by ``(source_type, source_id, source_revision)``.  When the underlying
Revision changes, :meth:`CoreMemoryStore.invalidate_by_source_revision`
removes only affected items — the whole card stays.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta


EPISTEMIC_TYPES = ("confirmed_fact", "user_assertion", "observed_pattern", "agent_inference", "unknown", "contradicted")


def _make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _now():
    return datetime.utcnow().isoformat(timespec="seconds")


class CoreMemoryStore:
    """CRUD + lifecycle for Core Memory Cards."""

    def __init__(self, store):
        self.store = store
        self.connection = getattr(store, "connection", None)
        self._ensure_schema()

    def _ensure_schema(self):
        if self.connection is None:
            return
        self.connection.executescript(
            """CREATE TABLE IF NOT EXISTS agent_core_memory_cards (
                card_id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                card_revision INTEGER NOT NULL DEFAULT 1,
                priority REAL NOT NULL DEFAULT 0.0,
                last_accessed_at TEXT,
                query_count INTEGER NOT NULL DEFAULT 0,
                distinct_conversation_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope_id, subject_type, subject_id)
            );

            CREATE TABLE IF NOT EXISTS agent_core_memory_items (
                item_id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL REFERENCES agent_core_memory_cards(card_id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                epistemic_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ids_json TEXT NOT NULL DEFAULT '[]',
                source_revisions_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_validated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_query_accesses (
                access_id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL REFERENCES agent_core_memory_cards(card_id) ON DELETE CASCADE,
                conversation_id TEXT,
                viewer_id TEXT,
                accessed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_core_cards_scope_subject
              ON agent_core_memory_cards(scope_id, subject_type, subject_id);
            CREATE INDEX IF NOT EXISTS idx_core_items_card
              ON agent_core_memory_items(card_id);
            CREATE INDEX IF NOT EXISTS idx_core_items_source
              ON agent_core_memory_items(source_type, item_id);
            CREATE INDEX IF NOT EXISTS idx_query_accesses_card
              ON agent_query_accesses(card_id, accessed_at);
            """
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert_card(self, *, scope_id, subject_type, subject_id, display_name):
        now = _now()
        row = self.connection.execute(
            "SELECT card_id FROM agent_core_memory_cards WHERE scope_id = ? AND subject_type = ? AND subject_id = ?",
            (scope_id, subject_type, subject_id),
        ).fetchone()
        if row:
            self.connection.execute(
                "UPDATE agent_core_memory_cards SET display_name = ?, updated_at = ? WHERE card_id = ?",
                (display_name, now, row["card_id"]),
            )
            card_id = row["card_id"]
        else:
            card_id = _make_id("core")
            self.connection.execute(
                """INSERT INTO agent_core_memory_cards(card_id, scope_id, subject_type, subject_id,
                    display_name, card_revision, priority, query_count, distinct_conversation_count,
                    created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, 0.0, 0, 0, ?, ?)""",
                (card_id, scope_id, subject_type, subject_id, display_name, now, now),
            )
        self.connection.commit()
        return card_id

    def upsert_item(self, *, card_id, text, epistemic_type, source_type, source_ids, source_revisions):
        if epistemic_type not in EPISTEMIC_TYPES:
            raise ValueError(f"invalid epistemic_type: {epistemic_type}")
        item_id = _make_id("core_item")
        now = _now()
        self.connection.execute(
            """INSERT INTO agent_core_memory_items(item_id, card_id, text, epistemic_type,
                source_type, source_ids_json, source_revisions_json, status, created_at, last_validated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (item_id, card_id, text, epistemic_type, source_type,
             json.dumps(list(source_ids or []), ensure_ascii=False),
             json.dumps(dict(source_revisions or {}), ensure_ascii=False),
             now, now),
        )
        self.connection.commit()
        return item_id

    def list_cards(self, scope_id=None, subject_ids=None, limit=5):
        clauses = []
        params = []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if subject_ids:
            marks = ",".join("?" * len(subject_ids))
            clauses.append(f"subject_id IN ({marks})")
            params.extend(subject_ids)
        clauses.append("EXISTS (SELECT 1 FROM agent_core_memory_items i WHERE i.card_id = agent_core_memory_cards.card_id AND i.status = 'active')")
        params.append(limit)
        rows = self.connection.execute(
            f"""SELECT * FROM agent_core_memory_cards WHERE {' AND '.join(clauses)}
               ORDER BY priority DESC, updated_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._card_with_items(dict(row)) for row in rows]

    def _card_with_items(self, card):
        items = self.connection.execute(
            "SELECT * FROM agent_core_memory_items WHERE card_id = ? AND status = 'active' ORDER BY created_at",
            (card["card_id"],),
        ).fetchall()
        card["items"] = [
            {
                **dict(row),
                "source_ids": json.loads(row["source_ids_json"] or "[]"),
                "source_revisions": json.loads(row["source_revisions_json"] or "{}"),
            }
            for row in items
        ]
        return card

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def record_access(self, *, card_id, conversation_id=None, viewer_id=None):
        now = _now()
        self.connection.execute(
            "INSERT INTO agent_query_accesses(access_id, card_id, conversation_id, viewer_id, accessed_at) VALUES (?, ?, ?, ?, ?)",
            (_make_id("access"), card_id, conversation_id, viewer_id, now),
        )
        counts = self.connection.execute(
            """SELECT COUNT(*) AS n, COUNT(DISTINCT COALESCE(conversation_id, access_id)) AS distinct_conv
               FROM agent_query_accesses WHERE card_id = ? AND accessed_at >= ?""",
            (card_id, (datetime.utcnow() - timedelta(days=30)).isoformat(timespec="seconds")),
        ).fetchone()
        priority = 1.0 if counts["n"] >= 3 and counts["distinct_conv"] >= 2 else 0.0
        self.connection.execute(
            """UPDATE agent_core_memory_cards
               SET query_count = ?, distinct_conversation_count = ?, priority = ?, last_accessed_at = ?
               WHERE card_id = ?""",
            (counts["n"], counts["distinct_conv"], priority, now, card_id),
        )
        self.connection.commit()

    def invalidate_by_source_revision(self, source_type, source_id, revision):
        """Deactivate only items whose stored revision precedes the new one."""
        rows = self.connection.execute(
            "SELECT item_id, source_ids_json, source_revisions_json FROM agent_core_memory_items WHERE source_type = ? AND status = 'active'",
            (source_type,),
        ).fetchall()
        for row in rows:
            ids = json.loads(row["source_ids_json"] or "[]")
            if source_id not in ids:
                continue
            revs = json.loads(row["source_revisions_json"] or "{}")
            stored = int(revs.get(source_id, 0) or 0)
            if stored < revision:
                self.connection.execute(
                    "UPDATE agent_core_memory_items SET status = 'invalidated', last_validated_at = ? WHERE item_id = ?",
                    (_now(), row["item_id"]),
                )
        self.connection.commit()

    def demote_stale_cards(self, *, days_threshold=90):
        cutoff = (datetime.utcnow() - timedelta(days=days_threshold)).isoformat(timespec="seconds")
        self.connection.execute(
            "UPDATE agent_core_memory_cards SET priority = 0.0 WHERE last_accessed_at IS NOT NULL AND last_accessed_at < ?",
            (cutoff,),
        )
        self.connection.commit()
