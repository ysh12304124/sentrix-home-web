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
import logging
import threading
import time
from typing import ClassVar

from ..retrieval_indexes import RetrievalIndex
from .base import CandidateHit, HardFilterContext, RetrievalQuery


@dataclass
class LexicalRetriever:
    name: str = "lexical"
    kind: str = "primary"

    # Retriever objects are request-local, but index health is process-wide.
    # A per-instance lock allowed concurrent QA requests to launch overlapping
    # scoped FTS rebuilds.  Share both the lock and healthy-scope memo across
    # instances.
    _shared_populate_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_populated_scopes: ClassVar[set[tuple[int, str | None]]] = set()
    _shared_population_retry_after: ClassVar[dict[tuple[int, str | None], float]] = {}

    def __init__(self, store):
        self.store = store
        self.index = RetrievalIndex(store)
        # A benchmark creates a fresh scope for every run.  Keep health state
        # per scope instead of globally, otherwise the first scope searched in
        # a process suppresses the self-heal for every later album.
        self._populated_scopes = self._shared_populated_scopes
        # A fresh benchmark scope can be queried by many QA workers at once.
        # Serialize the one-time FTS self-heal; concurrent rebuild_all calls
        # can corrupt SQLite/FTS on Windows and crash the API process.
        self._populate_lock = self._shared_populate_lock

    def _scope(self, filters: HardFilterContext) -> str | None:
        if filters.all_authorized or not filters.scope_ids:
            return None
        return filters.scope_ids[0]

    def _ensure_populated(self, scope_id=None):
        with self._populate_lock:
            return self._ensure_populated_unlocked(scope_id)

    def _ensure_populated_unlocked(self, scope_id=None):
        """Self-heal an empty *or partial* lexical projection.

        Checking only ``COUNT(*) == 0`` left a partially written FTS table
        (five rows after an interrupted/old scoped rebuild) permanently
        enabled.  That made lexical recall silently disappear for almost the
        whole album.  Compare distinct indexed observations with the current
        scope and rebuild that scope when the projection is materially behind.
        """
        # Include the store identity: test fixtures and maintenance jobs may
        # open separate SQLite connections containing the same scope name.
        # Sharing a bare scope string across those stores would skip required
        # health checks in the second database.
        scope_key = (id(self.store), str(scope_id or "") or None)
        if scope_key in self._populated_scopes:
            return
        # A freshly-created benchmark scope can briefly race the final SQLite
        # commit from graph/pipeline processing.  Do not turn one transient
        # lock/connection failure into a permanently healthy-but-empty index.
        # Retry on a later query instead of poisoning this process-wide memo.
        retry_after = self._shared_population_retry_after.get(scope_key, 0.0)
        if retry_after > time.monotonic():
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
            # Do not require an event_summary term here: RetrievalIndex indexes
            # canonical Observation fields, while event text is joined by the
            # metadata/event ranking path.  The old impossible condition made
            # every request rebuild the same FTS projection (22-60s each).
            if expected and indexed < max(1, expected // 2):
                self.index.rebuild_all(scope_id=scope_id)
                # A rebuild that returns without enough rows is not healthy;
                # leave the scope un-memoized so the next request can retry.
                indexed_after = int(conn.execute(
                    "SELECT COUNT(DISTINCT observation_id) FROM observation_search_fts "
                    + ("WHERE scope_id = ?" if scope_id else ""), params
                ).fetchone()[0] or 0)
                if indexed_after < max(1, expected // 2):
                    self._shared_population_retry_after[scope_key] = time.monotonic() + 2.0
                    return
        except Exception as exc:
            self._shared_population_retry_after[scope_key] = time.monotonic() + 2.0
            logging.getLogger(__name__).warning(
                "lexical index self-heal failed for scope %s: %s",
                scope_id or "all", exc,
            )
            return
        self._shared_population_retry_after.pop(scope_key, None)
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
