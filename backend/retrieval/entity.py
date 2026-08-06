"""EntityRetriever — confirmed-entity and person-bridge recall.

Person conditions are hard constraints; this retriever turns confirmed
entities into candidate assets via the ``person_bridge`` rows in
``observation_search_terms`` (or a direct people match on Observations when
the derived table is absent).

A person hit is a candidate only — the Kernel decides certainty through the
condition pass; this retriever never declares identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import CandidateHit, HardFilterContext, RetrievalQuery


@dataclass
class EntityRetriever:
    name: str = "entity"
    kind: str = "primary"

    def __init__(self, store):
        self.store = store

    def _person_names(self, query: RetrievalQuery) -> list[str]:
        return [constraint.value for constraint in query.constraints if constraint.dimension == "person"]

    def _resolve_asset_ids_for_person(self, person: str, scope_id: str | None) -> set[str]:
        # Prefer the derived person_bridge projection when present.
        try:
            table = self.store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='observation_search_terms'"
            ).fetchone()
            if table:
                rows = self.store.connection.execute(
                    "SELECT asset_id FROM observation_search_terms "
                    "WHERE field_type = 'person_bridge' AND normalized_value = ?"
                    + (" AND scope_id = ?" if scope_id else ""),
                    (person.lower(), scope_id) if scope_id else (person.lower(),),
                ).fetchall()
                return {row[0] for row in rows}
        except Exception:
            pass
        assets = set()
        for observation in self.store.list_observations(limit=100_000):
            if scope_id and (observation.get("scope_id") or "home-default") != scope_id:
                continue
            people = observation.get("people") or []
            if person in people:
                assets.add(observation.get("asset_id"))
        return assets

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        names = self._person_names(query)
        if not names:
            return []
        candidates = set()
        for name in names:
            scope = filters.scope_ids[0] if filters.scope_ids and not filters.all_authorized else None
            candidates |= self._resolve_asset_ids_for_person(name, scope)
        hits = []
        for asset_id in sorted(candidates):
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=1.0,
                score_kind="discrete",
                higher_is_better=True,
                rank=len(hits) + 1,
                source_id=asset_id,
                matched_text=names[0] if names else None,
                metadata={"person_names": names},
            ))
            if len(hits) >= limit:
                break
        return hits
