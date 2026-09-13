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
import re

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
        if (not filters.time_bounds and not filters.annual_time_window
                and not filters.media_types and not filters.place):
            return []
        from ..geocoding import place_text_matches
        import json as _json
        internal_limit = max(int(limit or 20),
                             int(os.environ.get("SENTRIX_METADATA_RECALL_LIMIT", "200")))
        hits = []
        query_text = str(getattr(query, "whole_query", "") or "").lower()
        query_terms = self._query_terms(query_text)
        scope = (filters.scope_ids[0]
                 if not filters.all_authorized and len(filters.scope_ids) == 1
                 else None)
        event_context = self._event_contexts(scope_id=scope)
        # Apply the authoritative scope in SQL.  Loading every album and then
        # discarding foreign rows cost ~8.5s on the production 500MB database.
        assets = self.store.list_assets(scope_id=scope, limit=100_000)
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
            if filters.annual_time_window:
                captured = _parse_datetime(asset.get("captured_at"))
                if captured is not None and not _in_annual_window(captured, filters.annual_time_window):
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
            # Event summaries/titles are authoritative memory evidence for
            # many video questions, but are not duplicated onto the asset
            # metadata row.  Use them as a deterministic ranking signal (and
            # keep the open-world behaviour when no geocode is available).
            event_text, event_place = event_context.get(str(asset["id"]), ("", ""))
            event_score = 0.0
            if filters.place:
                requested = str(filters.place).lower().strip()
                if requested and requested in event_text:
                    event_score += 80.0
                if event_place and requested and requested in event_place.lower():
                    event_score += 40.0
            if query_terms and event_text:
                event_score += min(40.0, 6.0 * sum(term in event_text for term in query_terms))
            if filters.time_bounds and event_text:
                # Date text in an event summary is useful when the asset's
                # capture timestamp is missing or represented in another zone.
                start, end = filters.time_bounds
                for stamp in re.findall(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", event_text):
                    try:
                        from datetime import datetime
                        parsed = datetime.strptime(stamp.replace("/", "-"), "%Y-%m-%d")
                        if start.date() <= parsed.date() < end.date():
                            event_score += 35.0
                            break
                    except (TypeError, ValueError):
                        continue
            hits.append(CandidateHit(
                asset_id=asset["id"],
                retriever=self.name,
                raw_score=event_score,
                score_kind="discrete",
                higher_is_better=True,
                rank=len(hits) + 1,
                source_id=asset["id"],
                source_revision=asset.get("revision"),
                metadata={"scope_id": asset_scope, "media_type": media_type,
                          "captured_at": asset.get("captured_at")},
            ))
        # Metadata is normally a scope enumerator.  Sorting by event evidence
        # turns it into a useful first-class recall channel without changing
        # hard filtering or the downstream condition pass.
        hits.sort(key=lambda hit: (-float(hit.raw_score or 0.0), hit.asset_id))
        ranked_hits = []
        for rank, hit in enumerate(hits[:internal_limit], 1):
            ranked_hits.append(CandidateHit(
                asset_id=hit.asset_id, retriever=hit.retriever,
                raw_score=hit.raw_score, score_kind=hit.score_kind,
                higher_is_better=hit.higher_is_better, rank=rank,
                source_id=hit.source_id, source_revision=hit.source_revision,
                matched_text=hit.matched_text, metadata=hit.metadata,
            ))
        return ranked_hits

    @staticmethod
    def _query_terms(text: str) -> list[str]:
        terms = set(re.findall(r"[a-z0-9]{3,}", text))
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        terms.update(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        stop = {"什么", "哪个", "哪里", "时间", "地点", "照片", "视频", "记录", "请问", "我在"}
        return [term for term in terms if term not in stop]

    def _event_contexts(self, scope_id: str | None = None) -> dict[str, tuple[str, str]]:
        """Load linked event evidence once per retrieval, avoiding N+1 SQL."""
        contexts: dict[str, tuple[str, str]] = {}
        try:
            row = self.store.connection.execute(
                "SELECT o.asset_id, e.title, e.summary, e.place, e.time_start, e.time_end "
                "FROM events e JOIN event_observations eo ON eo.event_id=e.id "
                "JOIN observations o ON o.id=eo.observation_id "
                + ("WHERE e.scope_id=? " if scope_id else "")
                + "ORDER BY e.updated_at DESC",
                (scope_id,) if scope_id else (),
            )
            for item in row.fetchall():
                asset_id = str(item[0] or "")
                if not asset_id:
                    continue
                old_text, old_place = contexts.get(asset_id, ("", ""))
                values = [str(item[index] or "").lower() for index in range(1, 6)]
                text = " ".join(value for value in values if value)
                place = str(item[3] or "")
                # The query is newest-first; retaining the first place keeps
                # the latest event label while concatenating all summaries
                # preserves recall across duplicate event projections.
                contexts[asset_id] = ((old_text + " " + text).strip(),
                                      (old_place or place).strip())
            return contexts
        except Exception:
            return contexts


def _in_annual_window(captured, window):
    start_month, start_day, end_month, end_day = window
    current = (captured.month, captured.day)
    start = (start_month, start_day)
    end = (end_month, end_day)
    if start < end:
        return start <= current < end
    return current >= start or current < end


def _parse_datetime(value):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
