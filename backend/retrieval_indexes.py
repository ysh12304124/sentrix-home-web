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


# Phase R P0-1: Chinese lexical retrieval is explicit pre-tokenization, not a
# hope that unicode61 produces bigrams.  Every token below is a general-purpose
# unit — never a benchmark-specific synonym.
def pre_tokenize(value):
    """Whole normalized value + latin words + CJK bigrams, space-joined.

    Example: 浅黄色毛绒睡衣 -> ["浅黄色毛绒睡衣", "浅黄", "黄色", "色毛",
    "毛绒", "绒睡", "睡衣"].  Single CJK characters are intentionally absent so
    a lone "色"/"毛" can never count as support.
    """
    value = _normalize(value)
    if not value:
        return []
    tokens = set()
    # A single CJK character is never a token — it can't carry meaning alone.
    if len(value) > 1:
        tokens.add(value)
    for word in re.findall(r"[a-zA-Z0-9]+", value):
        tokens.add(word)
    chars = [char for char in value if not char.isspace()]
    for index in range(len(chars) - 1):
        tokens.add(chars[index] + chars[index + 1])
    return sorted(tokens)


def _fts_content(value):
    return " ".join(pre_tokenize(value))


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
        # Phase R P0-1: pre-tokenized FTS5 projection.  Only the tokens column is
        # indexed; scope/field/asset/observation are UNINDEXED metadata.  If the
        # SQLite build lacks FTS5 the virtual table creation is skipped and the
        # LIKE-based ``search`` remains the fallback.
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS observation_search_fts USING fts5(
                    tokens,
                    scope_id UNINDEXED,
                    field_type UNINDEXED,
                    asset_id UNINDEXED,
                    observation_id UNINDEXED
                )"""
            )
        except Exception:
            pass
        self.connection.commit()

    def refresh_from_observation(self, observation):
        """Delete then reinsert rows for the given Observation (terms + FTS)."""
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
        self.connection.execute(
            "DELETE FROM observation_search_fts WHERE observation_id = ?", (observation_id,)
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
                try:
                    self.connection.execute(
                        """INSERT INTO observation_search_fts(tokens, scope_id, field_type,
                            asset_id, observation_id) VALUES (?, ?, ?, ?, ?)""",
                        (_fts_content(row[5]), scope_id, field_type, asset_id, observation_id),
                    )
                except Exception:
                    pass
        self.connection.commit()

    def rebuild_all(self, scope_id: str | None = None) -> int:
        """Recompute every row from canonical Observations (terms + FTS)."""
        observations = self.store.list_observations(scope_id=scope_id, limit=10_000)
        self.connection.execute(
            "DELETE FROM observation_search_terms" if scope_id is None
            else "DELETE FROM observation_search_terms WHERE scope_id = ?",
            () if scope_id is None else (scope_id,),
        )
        try:
            self.connection.execute("DELETE FROM observation_search_fts")
        except Exception:
            pass
        self.connection.commit()
        for observation in observations:
            self.refresh_from_observation(observation)
        return len(observations)

    def search_fts(self, query: str, scope_id: str | None = None, limit: int = 20) -> Iterable[dict]:
        """Pre-tokenized FTS candidate recall.

        Every query token is matched independently and assets are ranked by how
        many distinct tokens hit, with an exact whole-query hit boosted.  A
        single-character query never yields a token (pre_tokenize drops single
        CJK chars), so one-char "colour" scans cannot be treated as support.
        """
        if self.connection is None:
            return []
        tokens = pre_tokenize(query)
        if not tokens:
            return []
        whole = tokens[0]
        token_hits: dict[str, int] = {}
        for token in tokens:
            try:
                rows = self.connection.execute(
                    "SELECT asset_id FROM observation_search_fts "
                    "WHERE observation_search_fts MATCH ?"
                    + (" AND scope_id = ?" if scope_id else ""),
                    (f'"{token}"', scope_id) if scope_id else (f'"{token}"',),
                ).fetchall()
            except Exception:
                continue
            for row in rows:
                asset_id = row[0]
                token_hits[asset_id] = token_hits.get(asset_id, 0) + 1
        if not token_hits:
            return []
        scored = []
        for asset_id, count in token_hits.items():
            exact_boost = 2.0 if count == 1 and whole and _whole_is_exact(asset_id, whole, scope_id) else 0.0
            scored.append({"asset_id": asset_id, "score": count + exact_boost, "token_hits": count})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def _whole_is_exact(self, asset_id, whole, scope_id):
        try:
            row = self.connection.execute(
                "SELECT 1 FROM observation_search_terms WHERE asset_id = ? AND normalized_value = ?"
                + (" AND scope_id = ?" if scope_id else ""),
                (asset_id, whole, scope_id) if scope_id else (asset_id, whole),
            ).fetchone()
            return bool(row)
        except Exception:
            return False

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
