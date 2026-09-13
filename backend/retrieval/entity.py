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

    def _rows(self, query: str, params=()):
        """Use MemoryStore's locked projection, with a tiny test-store fallback."""
        reader = getattr(self.store, "_rows", None)
        if callable(reader):
            return reader(query, params)
        # Lightweight test doubles predate MemoryStore._rows.  They are not
        # shared by the API worker, so direct access is safe in that case.
        rows = self.store.connection.execute(query, params).fetchall()
        return [dict(row) if hasattr(row, "keys") else row for row in rows]

    def _row(self, query: str, params=()):
        reader = getattr(self.store, "_row", None)
        if callable(reader):
            return reader(query, params)
        row = self.store.connection.execute(query, params).fetchone()
        return dict(row) if row is not None and hasattr(row, "keys") else row

    @staticmethod
    def _value(row, key: str, index: int = 0):
        return row.get(key) if isinstance(row, dict) else (row[index] if row else None)

    def _resolve_asset_ids_for_person(self, person: str, scope_id: str | None) -> set[str]:
        # Face-cluster membership is the canonical identity index. The
        # observation people_json projection is intentionally sparse, so
        # relying on it made person/relationship questions miss even when a
        # face was already assigned to a confirmed cluster.
        cluster_assets = set()
        try:
            rows = self._rows(
                "SELECT DISTINCT fi.asset_id FROM face_instances fi "
                "JOIN face_clusters fc ON fc.id = fi.cluster_id "
                "JOIN entities e ON e.id = fc.entity_id "
                "WHERE fc.status = 'confirmed' AND e.status = 'confirmed' "
                "AND e.entity_type = 'person' AND e.canonical_name = ?"
                + (" AND fi.asset_id IN (SELECT id FROM assets WHERE scope_id = ?)" if scope_id else ""),
                (person, scope_id) if scope_id else (person,),
            )
            cluster_assets = {str(self._value(row, "asset_id")) for row in rows
                              if self._value(row, "asset_id")}
        except Exception:
            cluster_assets = set()
        # Prefer the derived person_bridge projection when present.
        try:
            table = self._row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='observation_search_terms'"
            )
            if table:
                rows = self._rows(
                    "SELECT asset_id FROM observation_search_terms "
                    "WHERE field_type = 'person_bridge' AND normalized_value = ?"
                    + (" AND scope_id = ?" if scope_id else ""),
                    (person.lower(), scope_id) if scope_id else (person.lower(),),
                )
                asset_ids = {self._value(row, "asset_id") for row in rows
                             if self._value(row, "asset_id")}
                if asset_ids:
                    return asset_ids | cluster_assets
        except Exception:
            pass
        assets = set(cluster_assets)
        for observation in self.store.list_observations(limit=100_000):
            if scope_id and (observation.get("scope_id") or "home-default") != scope_id:
                continue
            people = observation.get("people") or []
            names = {
                str(item.get("name") or item.get("canonical_name") or "")
                if isinstance(item, dict) else str(item)
                for item in people
            }
            if person in names:
                assets.add(observation.get("asset_id"))
        return assets

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        names = self._person_names(query)
        if not names:
            return []
        # Keep the union for recall, but retain how many requested identities
        # co-occur on each asset.  The old set-only implementation discarded
        # that signal and sorted relation candidates by asset id, so a photo
        # containing both people could lose to an unrelated photo containing
        # only one of them.
        asset_match_counts: dict[str, int] = {}
        for name in names:
            scope = filters.scope_ids[0] if filters.scope_ids and not filters.all_authorized else None
            for asset_id in self._resolve_asset_ids_for_person(name, scope):
                if asset_id:
                    asset_match_counts[str(asset_id)] = asset_match_counts.get(str(asset_id), 0) + 1
        hits = []
        for asset_id in sorted(asset_match_counts,
                               key=lambda value: (-asset_match_counts[value], value)):
            matched_count = asset_match_counts[asset_id]
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=matched_count / max(1, len(names)),
                score_kind="discrete",
                higher_is_better=True,
                rank=len(hits) + 1,
                source_id=asset_id,
                matched_text=", ".join(names),
                metadata={"person_names": names, "matched_person_count": matched_count},
            ))
            if len(hits) >= limit:
                break
        return hits
