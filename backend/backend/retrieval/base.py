"""Retriever contracts and shared value objects (Phase R §6).

P0-19: scores are never a bare float.  ``CandidateHit`` carries ``raw_score``,
``score_kind``, ``higher_is_better`` and ``rank`` so RRF can use the rank and
the probe layer can only consume ``calibrated_score`` (space-calibrated).

``HardFilterContext`` documents which filters may run inside an ANN index
(scope / media are encodeable) versus only after recall (time ranges,
must_not).  The Kernel is responsible for the final hard pass regardless of
what any retriever returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..query_contracts import Constraint, QueryFacet


@dataclass(frozen=True)
class HardFilterContext:
    scope_ids: tuple[str, ...]
    viewer_id: str = "owner"
    media_types: tuple[str, ...] | None = None      # hard media filter (e.g. image only)
    time_bounds: tuple[datetime, datetime] | None = None
    negated_media: frozenset[str] = frozenset()      # must_not media (video etc.)
    negated_dimensions: frozenset[str] = frozenset()
    all_authorized: bool = False
    place: str | None = None          # semantic place prefilter (recall backbone)

    @classmethod
    def from_spec(cls, spec) -> "HardFilterContext":
        import re
        from ..query_contracts import HARD
        media_types, negated_media = [], set()
        time_bounds = None
        negated_dimensions = set()
        place = None
        for constraint in spec.constraints:
            if constraint.dimension == "place" and not constraint.negated:
                place = str(constraint.value or "").strip() or None
            if constraint.strictness != HARD:
                continue
            if constraint.dimension == "media":
                value = constraint.value
                if constraint.negated:
                    negated_media.add(value)
                else:
                    media_types.append(value)
            elif constraint.dimension == "time" and not constraint.negated:
                bounds = _parse_time_bounds(constraint.value)
                if bounds:
                    time_bounds = bounds
            elif constraint.negated:
                negated_dimensions.add(constraint.dimension)
        return cls(
            scope_ids=tuple(spec.scope_ids),
            viewer_id=spec.viewer_id,
            media_types=tuple(media_types) or None,
            time_bounds=time_bounds,
            negated_media=frozenset(negated_media),
            negated_dimensions=frozenset(negated_dimensions),
            all_authorized=spec.scope_mode == "all_authorized",
            place=place,
        )


def _parse_time_bounds(value: str):
    from ..query_contracts import parse_time_expression
    try:
        return parse_time_expression(value)
    except Exception:
        return None


@dataclass
class RetrievalQuery:
    whole_query: str
    facets: list[QueryFacet] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    result_requirement: dict[str, Any] = field(default_factory=dict)
    # Composition-root owned embedders; ANN retrievers use them if available.
    embedding_router: Any = None

    @classmethod
    def from_spec(cls, spec, *, embedding_router=None) -> "RetrievalQuery":
        whole = " ".join(c.source_text or c.value for c in spec.constraints)
        facets = list(spec.facets)
        if not facets and not whole:
            facets = [QueryFacet(dimension="semantic", surface_text=whole)]
        return cls(
            whole_query=whole.strip(),
            facets=facets,
            constraints=list(spec.constraints),
            result_requirement=dict(spec.result_requirement),
            embedding_router=embedding_router,
        )


@dataclass(frozen=True)
class CandidateHit:
    asset_id: str
    retriever: str
    raw_score: float
    score_kind: str
    higher_is_better: bool
    rank: int
    calibrated_score: float | None = None
    source_id: str | None = None
    source_revision: int | None = None
    matched_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    name: str
    kind: str  # "primary" or "expander"

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]: ...
