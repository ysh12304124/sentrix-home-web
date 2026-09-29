"""RetrievalStrategy — decide where the answer must come from (TFPE2-2, first slice).

The 12B parser already judged answer_type + strategy_hint (open-vocabulary, not
keyword rules).  This deterministic layer merges that judgment with a capability
gate so the zero-tolerance holds: a query that is exactly answerable from
structured columns is never answered by ANN estimation.

Gate rules (schema-capability, not query patterns):
- ``answer_type`` in the structured set AND no picture-only condition -> run the
  Structured Executor (structured_fact / aggregation / entity_fact).
- model picked a picture strategy but the query has no picture-only dimension
  (e.g. "去年拍了多少张照片") -> override to structured and trace the override.
- model picked structured but a picture-only dimension is present (衣着/物体/颜色
  ...) -> downgrade to hybrid; the normal retrieval path owns that turn.
- model silent on strategy -> conservative default derived from answer_type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .query_contracts import QuerySpec, STRUCTURED_ANSWER_TYPES

STRUCTURED_STRATEGIES = {"structured_fact", "aggregation", "entity_fact"}

# Dimensions that require looking at the picture (or free-text semantics the
# structured executor cannot guarantee).  Place is excluded: it is matched
# against structured place text (D3) for this slice.
_PICTURE_BLOCKING_DIMS = {
    "clothing", "object", "visual", "color", "ocr", "activity",
    "relationship", "semantic", "other",
}

# Channels the Structured Executor never needs.
_STRUCTURED_SKIPPED_CHANNELS = ["visual_ann", "text_ann", "adjacency", "lexical"]


@dataclass
class RetrievalStrategy:
    strategy: str
    reason: str = ""
    model_hint: str = ""
    visual_required: bool = False
    text_semantic_required: bool = False
    aggregation: dict | None = None
    skipped_channels: list[str] = field(default_factory=list)
    required_sources: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "chosen_strategy": self.strategy, "reason": self.reason,
            "model_hint": self.model_hint, "visual_required": self.visual_required,
            "text_semantic_required": self.text_semantic_required,
            "aggregation": self.aggregation,
            "skipped_channels": self.skipped_channels,
            "required_sources": self.required_sources,
        }


def _has_picture_blocking_condition(spec: QuerySpec) -> bool:
    return any(c.dimension in _PICTURE_BLOCKING_DIMS for c in spec.constraints)


def _aggregation_for(draft) -> dict | None:
    agg = (draft.structured or {}).get("aggregation")
    if isinstance(agg, dict) and agg.get("op"):
        return {"op": agg["op"], "group_by": agg.get("group_by")}
    op = {
        "count": "count", "exists": "exists", "boolean": "exists",
        "first_occurrence": "first", "last_occurrence": "last",
        "grouped_list": "group_by", "list": "list", "date": "first",
        "date_range": "group_by",
    }.get(draft.answer_type)
    return {"op": op, "group_by": None} if op else None


def plan_retrieval_strategy(draft, spec: QuerySpec) -> RetrievalStrategy:
    model_hint = draft.strategy_hint
    answer_type = draft.answer_type
    picture_blocked = _has_picture_blocking_condition(spec)
    structured_capable = answer_type in STRUCTURED_ANSWER_TYPES and not picture_blocked

    if structured_capable:
        if answer_type in {"first_occurrence", "last_occurrence"} and spec.entity_ids:
            strategy = "entity_fact"
        elif answer_type == "grouped_list" and (draft.structured or {}).get("aggregation", {}).get("op") == "group_by":
            strategy = "aggregation"
        elif answer_type in {"grouped_list", "date_range"}:
            strategy = "aggregation"
        else:
            strategy = "structured_fact"
        reason = f"answer_type={answer_type}; structured columns answer exactly; no picture-only dimension"
        if model_hint in {"visual_semantic", "hybrid"}:
            reason += " (model hint overridden: no picture-only dimension present)"
        return RetrievalStrategy(
            strategy=strategy,
            reason=reason,
            model_hint=model_hint or "unset",
            visual_required=False,
            text_semantic_required=False,
            aggregation=_aggregation_for(draft),
            skipped_channels=list(_STRUCTURED_SKIPPED_CHANNELS),
            required_sources=_sources_for(strategy, answer_type),
        )

    if model_hint in STRUCTURED_STRATEGIES and picture_blocked:
        return RetrievalStrategy(
            strategy="hybrid",
            reason="model chose structured but a picture-only dimension is present; hybrid owns this turn",
            model_hint=model_hint,
            visual_required=True,
            text_semantic_required=True,
            aggregation=None,
            skipped_channels=[],
            required_sources=["retrieval_kernel"],
        )

    strategy = model_hint if model_hint in {"visual_semantic", "hybrid", "semantic_text", "asset_delivery"} else "hybrid"
    return RetrievalStrategy(
        strategy=strategy,
        reason=f"answer_type={answer_type}; visual or free-text semantics required -> {strategy}",
        model_hint=model_hint or "unset",
        visual_required=strategy != "semantic_text",
        text_semantic_required=strategy == "hybrid",
        aggregation=None,
        skipped_channels=[],
        required_sources=["retrieval_kernel"],
    )


def _sources_for(strategy: str, answer_type: str) -> list[str]:
    if strategy == "entity_fact":
        return ["entity_mentions", "observations.captured_at", "assets.captured_at"]
    if strategy == "aggregation":
        return ["assets.captured_at", "assets.media_type", "observations.place", "events.place"]
    return ["assets.captured_at", "assets.media_type", "assets.captured_location", "observations.place"]
