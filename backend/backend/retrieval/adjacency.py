"""AdjacencyRetriever — seed-based expansion (Phase R P0-9, R3B).

Not a first-round parallel retriever.  The Kernel runs primary recall, merges,
evaluates conditions, and only then feeds reliable seeds (exact/strong level)
into :meth:`expand`.  Expansion follows three edges, each with its own budget:

  - shared Event            (``event_observations`` join)
  - time window             (captured_at within +/- window of a seed)
  - batch / device          (same source_album_id or source_device_id)

Every expanded candidate must re-pass the Kernel's hard filters and condition
pass; adjacency alone never promotes an asset to ``matched``.  Expanders
inherit their seed's evidence class and never receive an anchor boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base import CandidateHit, HardFilterContext, RetrievalQuery


def _parse_datetime(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


@dataclass
class AdjacencyRetriever:
    name: str = "adjacency"
    kind: str = "expander"

    def __init__(self, store, *, config=None):
        self.store = store
        self.config = config
        self._budgets = {
            "max_seeds": 8, "per_seed_budget": 6, "event": 4,
            "batch": 2, "time_window_minutes": 120,
        }
        if config is not None:
            for key in self._budgets:
                self._budgets[key] = config.adjacency_budget(key) or self._budgets[key]

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        # The Kernel drives expansion via ``expand`` with reliable seeds; a bare
        # retrieve has no seeds and therefore nothing to expand.
        return []

    def expand(self, seed_asset_ids: list[str], filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        if not seed_asset_ids:
            return []
        seeds = list(dict.fromkeys(seed_asset_ids))[: self._budgets["max_seeds"]]
        seed_set = set(seeds)
        expanded = self._event_expansion(seeds, filters) | \
                   self._time_window_expansion(seeds, filters) | \
                   self._batch_expansion(seeds, filters)
        expanded -= seed_set
        hits = []
        for rank, asset_id in enumerate(sorted(expanded)):
            if rank >= limit:
                break
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=0.0,
                score_kind="adjacency",
                higher_is_better=True,
                rank=rank + 1,
                source_id=asset_id,
                metadata={"edge": "event_or_time_or_batch", "seeded": True},
            ))
        return hits

    def _seed_observation_ids(self, seeds):
        seed_obs = [obs for obs in self.store.list_observations(limit=100_000)
                    if obs.get("asset_id") in set(seeds)]
        return [obs.get("id") for obs in seed_obs if obs.get("id")]

    def _event_expansion(self, seeds, filters):
        obs_ids = self._seed_observation_ids(seeds)
        if not obs_ids:
            return set()
        placeholders = ",".join("?" for _ in obs_ids)
        try:
            rows = self.store.connection.execute(
                f"SELECT event_id, observation_id FROM event_observations "
                f"WHERE observation_id IN ({placeholders})", obs_ids
            ).fetchall()
        except Exception:
            return set()
        event_budget = self._budgets["event"]
        sibling_obs = set()
        per_event = {}
        for event_id, observation_id in rows:
            per_event.setdefault(event_id, []).append(observation_id)
        for event_id, members in per_event.items():
            sibling_obs.update(members[:event_budget])
        sibling_obs -= set(obs_ids)
        if not sibling_obs:
            return set()
        obs_rows = [obs for obs in self.store.list_observations(limit=100_000)
                    if obs.get("id") in sibling_obs]
        return {obs.get("asset_id") for obs in obs_rows if self._passes(obs, filters)}

    def _time_window_expansion(self, seeds, filters):
        minutes = self._budgets["time_window_minutes"]
        window = timedelta(minutes=minutes)
        assets = [asset for asset in self.store.list_assets(limit=100_000)
                  if asset.get("id") in set(seeds) or True]
        seed_times = {asset["id"]: _parse_datetime(asset.get("captured_at"))
                      for asset in self.store.list_assets(limit=100_000) if asset.get("id") in set(seeds)}
        seed_times = {asset_id: t for asset_id, t in seed_times.items() if t is not None}
        if not seed_times:
            return set()
        expanded = set()
        for asset in assets:
            if asset.get("id") in set(seeds):
                continue
            captured = _parse_datetime(asset.get("captured_at"))
            if captured is None:
                continue
            if any(abs((captured - seed_time).total_seconds()) <= window.total_seconds()
                   for seed_time in seed_times.values()):
                if self._passes(asset, filters):
                    expanded.add(asset["id"])
        return expanded

    def _batch_expansion(self, seeds, filters):
        seed_assets = [asset for asset in self.store.list_assets(limit=100_000)
                       if asset.get("id") in set(seeds)]
        keys = set()
        for asset in seed_assets:
            album = asset.get("source_album_id")
            device = asset.get("source_device_id")
            if album:
                keys.add(("album", album))
            if device:
                keys.add(("device", device))
        if not keys:
            return set()
        expanded = set()
        batch_budget = self._budgets["batch"]
        for key_kind, key_value in keys:
            count = 0
            for asset in self.store.list_assets(limit=100_000):
                if asset.get("id") in set(seeds):
                    continue
                if key_kind == "album" and asset.get("source_album_id") == key_value:
                    if self._passes(asset, filters):
                        expanded.add(asset["id"])
                        count += 1
                elif key_kind == "device" and asset.get("source_device_id") == key_value:
                    if self._passes(asset, filters):
                        expanded.add(asset["id"])
                        count += 1
                if count >= batch_budget:
                    break
        return expanded

    def _passes(self, asset_or_observation, filters):
        scope_id = asset_or_observation.get("scope_id") or "home-default"
        if not filters.all_authorized and filters.scope_ids and scope_id not in filters.scope_ids:
            return False
        media_type = asset_or_observation.get("media_type")
        if filters.media_types and media_type not in filters.media_types:
            return False
        if media_type in filters.negated_media:
            return False
        if filters.time_bounds:
            captured = _parse_datetime(asset_or_observation.get("captured_at"))
            if captured is not None and not (filters.time_bounds[0] <= captured < filters.time_bounds[1]):
                return False
        return True
