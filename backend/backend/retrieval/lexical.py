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
        self._populated = False

    def _scope(self, filters: HardFilterContext) -> str | None:
        if filters.all_authorized or not filters.scope_ids:
            return None
        return filters.scope_ids[0]

    def _ensure_populated(self):
        """Self-heal the derived projection when the maintenance script has not
        run yet (e.g. a fresh local fixture or an old DB).  Built once per
        process; the maintenance script remains the canonical bulk builder."""
        if self._populated:
            return
        try:
            count = self.store.connection.execute(
                "SELECT COUNT(*) FROM observation_search_fts"
            ).fetchone()[0]
            if count == 0:
                self.index.rebuild_all()
        except Exception:
            pass
        self._populated = True

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        self._ensure_populated()
        scope = self._scope(filters)
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
