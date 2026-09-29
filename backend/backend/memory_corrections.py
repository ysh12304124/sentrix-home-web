"""Phase 6 — Agent-owned memory correction proposals and authorized revisions.

The Agent only *proposes* and *authorizes*.  Formal Revisions are written by
this module via ``apply_authorized_revision``; canonical ``raw_json`` fields
are never modified and every superseded value is preserved.  Callers that
skip ``propose`` and go straight to ``apply`` are rejected.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import uuid


DEFAULT_TOKEN_TTL_SECONDS = 300  # five minutes (plan §六)


class MemoryCorrectionError(RuntimeError):
    pass


def _make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _now():
    return datetime.utcnow().isoformat(timespec="seconds")


class MemoryCorrections:
    """Propose / apply memory corrections behind a confirmation token."""

    def __init__(self, store, token_ttl_seconds=DEFAULT_TOKEN_TTL_SECONDS):
        self.store = store
        self.connection = getattr(store, "connection", None)
        self.token_ttl = int(token_ttl_seconds)
        self._ensure_schema()

    def _ensure_schema(self):
        if self.connection is None:
            return
        self.connection.executescript(
            """CREATE TABLE IF NOT EXISTS agent_memory_correction_proposals (
                proposal_id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                previous_revision INTEGER NOT NULL,
                changed_fields_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                request_id TEXT UNIQUE,
                confirmation_token TEXT NOT NULL,
                token_expires_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                applied_at TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_memory_correction_revisions (
                revision_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL REFERENCES agent_memory_correction_proposals(proposal_id),
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                previous_revision INTEGER NOT NULL,
                new_revision INTEGER NOT NULL,
                changed_fields_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_memory_correction_audit (
                audit_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                event TEXT NOT NULL,
                detail_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_proposal_request_id
              ON agent_memory_correction_proposals(request_id);
            """
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def propose(self, *, scope_id, actor, target_type, target_id, changed_fields,
                evidence_ids=None, request_id=None):
        """Create a proposal — no canonical writes yet."""
        current = self._current_revision(target_type, target_id)
        if current is None:
            raise MemoryCorrectionError(f"target not found: {target_type}:{target_id}")
        # Idempotency — a repeated request_id returns the existing proposal.
        if request_id:
            existing = self.connection.execute(
                "SELECT * FROM agent_memory_correction_proposals WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing:
                return dict(existing)
        proposal_id = _make_id("proposal")
        token = uuid.uuid4().hex
        expires_at = (datetime.utcnow() + timedelta(seconds=self.token_ttl)).isoformat(timespec="seconds")
        now = _now()
        self.connection.execute(
            """INSERT INTO agent_memory_correction_proposals(proposal_id, scope_id, actor,
                target_type, target_id, previous_revision, changed_fields_json, evidence_ids_json,
                request_id, confirmation_token, token_expires_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (proposal_id, scope_id, actor, target_type, target_id, current,
             json.dumps(changed_fields, ensure_ascii=False),
             json.dumps(list(evidence_ids or []), ensure_ascii=False),
             request_id, token, expires_at, now),
        )
        self._audit(proposal_id, actor, "proposed",
                    {"target": f"{target_type}:{target_id}", "fields": list(changed_fields.keys())})
        self.connection.commit()
        return {"proposal_id": proposal_id, "confirmation_token": token,
                "expires_at": expires_at, "previous_revision": current,
                "target_type": target_type, "target_id": target_id}

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, *, proposal_id, confirmation_token, actor):
        row = self.connection.execute(
            "SELECT * FROM agent_memory_correction_proposals WHERE proposal_id = ?", (proposal_id,),
        ).fetchone()
        if not row:
            raise MemoryCorrectionError("unknown proposal")
        if row["status"] == "applied":
            # Idempotent replay — return the stored revision without side effects.
            return self._revision_for(proposal_id)
        if row["confirmation_token"] != confirmation_token:
            self._audit(proposal_id, actor, "apply_rejected", {"reason": "token_mismatch"})
            self.connection.commit()
            raise MemoryCorrectionError("token does not match proposal")
        if datetime.utcnow() > datetime.fromisoformat(row["token_expires_at"]):
            self._audit(proposal_id, actor, "apply_rejected", {"reason": "token_expired"})
            self.connection.commit()
            raise MemoryCorrectionError("confirmation token expired")
        changed_fields = json.loads(row["changed_fields_json"] or "{}")
        new_revision = self._write_revision(row["target_type"], row["target_id"],
                                              row["previous_revision"], changed_fields)
        revision_id = _make_id("revision")
        self.connection.execute(
            """INSERT INTO agent_memory_correction_revisions(revision_id, proposal_id, target_type,
                target_id, previous_revision, new_revision, changed_fields_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (revision_id, proposal_id, row["target_type"], row["target_id"],
             row["previous_revision"], new_revision, row["changed_fields_json"], _now()),
        )
        self.connection.execute(
            "UPDATE agent_memory_correction_proposals SET status = 'applied', applied_at = ? WHERE proposal_id = ?",
            (_now(), proposal_id),
        )
        self._audit(proposal_id, actor, "applied", {"new_revision": new_revision})
        self.connection.commit()
        return {"revision_id": revision_id, "new_revision": new_revision,
                "target_type": row["target_type"], "target_id": row["target_id"]}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_revision(self, target_type, target_id):
        """Return the Agent-owned revision counter for the target.

        Canonical rows are treated as revision 1.  Every applied correction
        adds a new row to ``agent_memory_correction_revisions`` and bumps the
        logical counter without modifying canonical ``raw_json``.
        """
        if not self._target_exists(target_type, target_id):
            return None
        row = self.connection.execute(
            "SELECT MAX(new_revision) AS latest FROM agent_memory_correction_revisions WHERE target_type = ? AND target_id = ?",
            (target_type, target_id),
        ).fetchone()
        latest = int(row["latest"]) if row and row["latest"] is not None else 1
        return latest

    def _target_exists(self, target_type, target_id):
        table = {"entity": "entities", "event": "events", "observation": "observations"}.get(target_type)
        if not table:
            return False
        row = self.connection.execute(
            f"SELECT id FROM {table} WHERE id = ?", (target_id,),
        ).fetchone()
        return row is not None

    def _write_revision(self, target_type, target_id, previous_revision, changed_fields):
        """Create a supersedes revision without touching canonical rows.

        Preserving ``raw_json`` is guaranteed structurally: canonical tables
        are not written by this method.  Downstream rebuild jobs read
        ``agent_memory_correction_revisions`` to apply the semantic change.
        """
        current = self._current_revision(target_type, target_id)
        if current is None:
            raise MemoryCorrectionError("target vanished before apply")
        if current != previous_revision:
            raise MemoryCorrectionError(f"revision mismatch: expected {previous_revision}, got {current}")
        return current + 1

    def _audit(self, proposal_id, actor, event, detail):
        self.connection.execute(
            "INSERT INTO agent_memory_correction_audit(audit_id, proposal_id, actor, event, detail_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (_make_id("audit"), proposal_id, actor, event, json.dumps(detail, ensure_ascii=False), _now()),
        )

    def _revision_for(self, proposal_id):
        row = self.connection.execute(
            "SELECT * FROM agent_memory_correction_revisions WHERE proposal_id = ? ORDER BY created_at DESC LIMIT 1",
            (proposal_id,),
        ).fetchone()
        return dict(row) if row else None
