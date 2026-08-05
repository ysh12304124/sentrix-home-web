"""Retrieval-side derived projections (Phase 3).

``observation_search_terms`` is a rebuildable index over canonical Observation
fields.  It does not own facts — every row must trace back to a specific
Observation revision, and the whole table can be dropped and rebuilt without
losing data.

Phase 3.5 ANN indices attach to the same normalized rows so scope and revision
metadata stay consistent between structured search and vector recall.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import uuid
from typing import Iterable


_FIELD_TYPES = ("place", "activity", "object", "clothing", "ocr", "person_bridge", "caption")


@dataclass(frozen=True)
class SearchTerm:
    id: str
    observation_id: str
    asset_id: str
    scope_id: str
    field_type: str
    normalized_value: str
    confidence: float
    source_type: str
    source_revision: int


def _make_id():
    return f"term_{uuid.uuid4().hex[:16]}"


def _normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _terms_from_field(observation, field_type, source_key):
    """Yield ``(field_type, normalized_value)`` pairs for one Observation."""
    raw = observation.get(source_key)
    if raw is None:
        return
    if isinstance(raw, list):
        for item in raw:
            value = _normalize(item)
            if value:
                yield field_type, value
    else:
        value = _normalize(raw)
        if value:
            yield field_type, value


class RetrievalIndex:
    """Manage the ``observation_search_terms`` derived table.

    Callers use :meth:`refresh_from_observation` when an Observation is added
    or its revision changes.  :meth:`rebuild_all` regenerates every row from
    canonical Observations (used by the maintenance script).
    """

    def __init__(self, store):
        self.store = store
        self.connection = getattr(store, "connection", None)
        self._ensure_schema()

    def _ensure_schema(self):
        if self.connection is None:
            return
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS observation_search_terms (
                id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                field_type TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                source_type TEXT NOT NULL DEFAULT 'observation',
                source_revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_search_terms_scope_field ON observation_search_terms(scope_id, field_type)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_search_terms_observation ON observation_search_terms(observation_id)"
        )
        self.connection.commit()

    def refresh_from_observation(self, observation):
        """Delete then reinsert rows for the given Observation."""
        if self.connection is None:
            return
        observation_id = observation.get("id")
        asset_id = observation.get("asset_id")
        if not observation_id or not asset_id:
            return
        scope_id = observation.get("scope_id") or "home-default"
        revision = int(observation.get("revision", 1) or 1)
        confidence = float(observation.get("confidence", 0) or 0)
        self.connection.execute(
            "DELETE FROM observation_search_terms WHERE observation_id = ?", (observation_id,)
        )
        rows = []
        term_sources = (
            ("place", "place"), ("activity", "activity"), ("caption", "caption"),
            ("object", "objects"), ("clothing", "clothing"), ("ocr", "ocr_text"),
            ("person_bridge", "people"),
        )
        for field_type, source_key in term_sources:
            for _, value in _terms_from_field(observation, field_type, source_key):
                rows.append((
                    _make_id(), observation_id, asset_id, scope_id, field_type, value,
                    confidence, "observation", revision,
                ))
        if rows:
            now = observation.get("updated_at") or observation.get("created_at") or ""
            for row in rows:
                self.connection.execute(
                    """INSERT INTO observation_search_terms(id, observation_id, asset_id, scope_id,
                        field_type, normalized_value, confidence, source_type, source_revision,
                        created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row + (now, now),
                )
        self.connection.commit()

    def rebuild_all(self, scope_id: str | None = None) -> int:
        """Recompute every row from canonical Observations."""
        observations = self.store.list_observations(scope_id=scope_id, limit=10_000)
        self.connection.execute(
            "DELETE FROM observation_search_terms" if scope_id is None
            else "DELETE FROM observation_search_terms WHERE scope_id = ?",
            () if scope_id is None else (scope_id,),
        )
        self.connection.commit()
        for observation in observations:
            self.refresh_from_observation(observation)
        return len(observations)

    def search(self, scope_id: str | None, field_type: str, value: str) -> Iterable[dict]:
        """Return rows matching a normalized substring for the field type."""
        if self.connection is None:
            return []
        term = _normalize(value)
        if not term:
            return []
        clauses = ["field_type = ?", "normalized_value LIKE ?"]
        params = [field_type, f"%{term}%"]
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        rows = self.connection.execute(
            f"SELECT * FROM observation_search_terms WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        return [dict(row) for row in rows]
