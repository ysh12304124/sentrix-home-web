"""MetadataRetriever — structured recall over asset metadata.

Scope, media type and time bounds are the only things this retriever filters
on.  It is the "structured only" channel in the ablation matrix and doubles as
the hard-prefilter candidate universe before the Kernel's own hard pass.

Pre-R2 the Kernel walked every asset anyway; this retriever makes that walk
explicit and cheap by applying the same scope/time/media constraints up front.
"""

from __future__ import annotations
import os

from dataclasses import dataclass, field
from typing import Any

from .base import CandidateHit, HardFilterContext, RetrievalQuery


@dataclass
class MetadataRetriever:
    name: str = "metadata"
    kind: str = "primary"

    def __init__(self, store):
        self.store = store

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        # Metadata only produces a recall signal when there is a positive
        # structured condition (time window / media type) to match.  Without
        # one it returns nothing, so a pure-semantic query does not get polluted
        # by "every asset in scope" becoming an anchor.
        if not filters.time_bounds and not filters.media_types and not filters.place:
            return []
        from ..geocoding import place_text_matches
        import json as _json
        internal_limit = max(int(limit or 20),
                             int(os.environ.get("SENTRIX_METADATA_RECALL_LIMIT", "200")))
        hits = []
        assets = self.store.list_assets(limit=100_000)
        for asset in assets:
            asset_scope = asset.get("scope_id") or "home-default"
            if not filters.all_authorized and filters.scope_ids and asset_scope not in filters.scope_ids:
                continue
            media_type = asset.get("media_type")
            if filters.media_types and media_type not in filters.media_types:
                continue
            if media_type in filters.negated_media:
                continue
            if filters.time_bounds:
                captured = _parse_datetime(asset.get("captured_at"))
                if captured is not None and not (filters.time_bounds[0] <= captured < filters.time_bounds[1]):
                    continue
            # place 预筛（镜像 kernel 判定：geocode 匹配或缺失保留，不匹配剔除）
            if filters.place:
                # ``MemoryStore`` returns metadata_json as a JSON string.  The
                # old code only handled a dict, so every asset with GPS was
                # silently treated as having no geocode and passed the place
                # prefilter.  That polluted the metadata channel with the
                # whole scope and let insertion order displace exact places.
                metadata = asset.get("metadata_json") or {}
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except (TypeError, ValueError):
                        metadata = {}
                geo = metadata.get("reverse_geocode") if isinstance(metadata, dict) else None
                if geo and not place_text_matches(filters.place, geo):
                    continue
            hits.append(CandidateHit(
                asset_id=asset["id"],
                retriever=self.name,
                raw_score=0.0,
                score_kind="discrete",
                higher_is_better=True,
                rank=len(hits) + 1,
                source_id=asset["id"],
                source_revision=asset.get("revision"),
                metadata={"scope_id": asset_scope, "media_type": media_type,
                          "captured_at": asset.get("captured_at")},
            ))
            if len(hits) >= internal_limit:
                break
        return hits


def _parse_datetime(value):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
