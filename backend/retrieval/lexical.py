"""LexicalRetriever — pre-tokenized FTS5 recall (Phase R P0-1).

Replaces the old single-char ``_contains`` semantics.  Only full-token and
CJK-bigram matches count; a single character can never be a matched fact.
Whole query and each facet are matched independently, then merged by Asset.

FTS is a *recall* channel.  Whether a condition is actually supported is the
Kernel's condition pass — never copied from a matched FTS row directly
(P1-2).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..retrieval_indexes import RetrievalIndex
from .base import CandidateHit, HardFilterContext, RetrievalQuery


@dataclass
class LexicalRetriever:
    name: str = "lexical"
    kind: str = "primary"

    def __init__(self, store):
        self.store = store
        self.index = RetrievalIndex(store)
        # A benchmark creates a fresh scope for every run.  Keep health state
        # per scope instead of globally, otherwise the first scope searched in
        # a process suppresses the self-heal for every later album.
        self._populated_scopes: set[str | None] = set()

    def _scope(self, filters: HardFilterContext) -> str | None:
        if filters.all_authorized or not filters.scope_ids:
            return None
        return filters.scope_ids[0]

    def _ensure_populated(self, scope_id=None):
        """Self-heal an empty *or partial* lexical projection.

        Checking only ``COUNT(*) == 0`` left a partially written FTS table
        (five rows after an interrupted/old scoped rebuild) permanently
        enabled.  That made lexical recall silently disappear for almost the
        whole album.  Compare distinct indexed observations with the current
        scope and rebuild that scope when the projection is materially behind.
        """
        scope_key = str(scope_id or "") or None
        if scope_key in self._populated_scopes:
            return
        try:
            conn = self.store.connection
            where = "WHERE a.scope_id = ?" if scope_id else ""
            params = (scope_id,) if scope_id else ()
            expected = int(conn.execute(
                "SELECT COUNT(DISTINCT o.id) FROM observations o "
                "JOIN assets a ON a.id = o.asset_id " + where, params
            ).fetchone()[0] or 0)
            indexed = int(conn.execute(
                "SELECT COUNT(DISTINCT observation_id) FROM observation_search_fts "
                + ("WHERE scope_id = ?" if scope_id else ""), params
            ).fetchone()[0] or 0)
            # Some observations legitimately have no textual fields.  A 50%
            # floor still catches truncated projections while avoiding a
            # rebuild for tiny/mostly-empty fixtures.
            if expected and indexed < max(1, expected // 2):
                self.index.rebuild_all(scope_id=scope_id)
        except Exception:
            pass
        self._populated_scopes.add(scope_key)

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        scope = self._scope(filters)
        self._ensure_populated(scope)
        queries = [query.whole_query] if query.whole_query else []
        queries.extend(facet.surface_text for facet in query.facets if facet.surface_text)
        queries = list(dict.fromkeys(item for item in queries if item))
        scores: dict[str, float] = {}
        matched: dict[str, str] = {}
        for surface in queries:
            rows = list(self.index.search_fts(surface, scope_id=scope, limit=limit * 4))
            for row in rows:
                asset_id = row["asset_id"]
                scores[asset_id] = scores.get(asset_id, 0.0) + row["score"]
                if asset_id not in matched:
                    matched[asset_id] = surface
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits = []
        for index, (asset_id, score) in enumerate(ranked[:limit]):
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=score,
                score_kind="token_hits",
                higher_is_better=True,
                rank=index + 1,
                source_id=asset_id,
                matched_text=matched.get(asset_id),
                metadata={"token_hits": score},
            ))
        return hits
