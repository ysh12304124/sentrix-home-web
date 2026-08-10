"""Thin Agent query contracts and the boundary between model text and runtime state."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any, Callable


HARD = "deterministic_hard"
SEMANTIC = "semantic_required"
PREFERENCE = "ranking_preference"

# TFPE v2: model-judged answer shape and retrieval strategy.  These are
# produced by the 12B parser (open-vocabulary judgment), never by keyword
# rules.  Code only validates them against the whitelist and executes.
ANSWER_TYPES = {
    "boolean", "count", "date", "date_range", "first_occurrence",
    "last_occurrence", "exists", "list", "grouped_list", "asset_set",
    "summary", "person_summary",
}
STRATEGY_HINTS = {
    "structured_fact", "aggregation", "entity_fact", "semantic_text",
    "visual_semantic", "hybrid", "asset_delivery",
}
STRUCTURED_ANSWER_TYPES = {
    "boolean", "count", "date", "date_range", "first_occurrence",
    "last_occurrence", "exists", "list", "grouped_list",
}


@dataclass(frozen=True)
class Constraint:
    dimension: str
    value: str
    strictness: str
    proof_policy: str
    negated: bool = False
    source_text: str = ""

    @property
    def key(self):
        return f"{self.dimension}:{self.value}"


@dataclass(frozen=True)
class QueryAction:
    """A single user goal produced by the semantic parser.

    Multiple actions may coexist (e.g. answer_question + return_assets); the
    supplementary plan forbids compressing them into one intent.
    """

    type: str
    target: str = "general"
    coverage: str = "best"


@dataclass(frozen=True)
class QueryFacet:
    """A single semantic dimension the user mentioned.

    Facets preserve the model's read of a user turn without being upgraded to a
    hard constraint.  They stay orthogonal to Constraint so retrieval can still
    consult surface language when strict conditions do not fire.
    """

    dimension: str
    surface_text: str


@dataclass
class QueryParseDraft:
    intent: str = "answer"
    answer_target: str = "general"
    entity_names: list[str] = field(default_factory=list)
    time_expression: str | None = None
    media_expressions: list[str] = field(default_factory=list)
    semantic_conditions: list[dict[str, Any]] = field(default_factory=list)
    negative_conditions: list[dict[str, Any]] = field(default_factory=list)
    result_requirement: dict[str, Any] = field(default_factory=dict)
    # TFPE v2: the model judges the answer shape and the retrieval strategy.
    # answer_type defaults to asset_set (find images); strategy_hint is empty
    # when the model made no call — code then falls back conservatively.
    answer_type: str = "asset_set"
    strategy_hint: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    # Debug: the raw model JSON after identity-field stripping (admin layer
    # only, never a writer input). Runtime identity fields the model echoes
    # back are removed before storage so they never leak into admin output.
    raw_json: Any = None
    # R9: proposed_mode is the ONLY writable mode field; it is advisory — the
    # Router decides the final route.  ``mode`` is a derived compatibility
    # property for legacy callers and the serialization layer.
    proposed_mode: str = "none"
    actions: list[QueryAction] = field(default_factory=list)
    facets: list[QueryFacet] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # R9-6: set when the parser produced NO model output (timeout / failure /
    # safe fallback).  The Router must then not trust a "none" proposal for
    # intro verbs — a parser-down family query must not fall into normal chat.
    parser_failed: bool = False

    @property
    def mode(self) -> str:
        return self.proposed_mode


@dataclass
class QuerySpec:
    query_id: str
    scope_mode: str
    scope_ids: list[str]
    viewer_id: str
    conversation_id: str
    intent: str
    answer_target: str
    constraints: list[Constraint]
    entity_ids: list[str] = field(default_factory=list)
    result_requirement: dict[str, Any] = field(default_factory=dict)
    # Phase 2R-3 composite carry-through — every downstream tool may read these,
    # but the legacy ``intent``/``answer_target`` remain the primary switch.
    actions: list[QueryAction] = field(default_factory=list)
    facets: list[QueryFacet] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    @property
    def scope_id(self):
        return self.scope_ids[0] if len(self.scope_ids) == 1 else None

    def constraints_for(self, dimension: str):
        return [item for item in self.constraints if item.dimension == dimension]


def _as_list(value):
    return value if isinstance(value, list) else []


def _clean_text(value):
    return str(value or "").strip()


def _valid_iso_date(value):
    value = _clean_text(value)
    if not value:
        return False
    try:
        datetime.fromisoformat(value[:10])
        return True
    except (TypeError, ValueError):
        return False


def _sanitize_structured(raw) -> dict[str, Any]:
    """Whitelist the model's structured slot — never invent fields."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    time_range = raw.get("time_range")
    if isinstance(time_range, dict):
        start = _clean_text(time_range.get("start")) or None
        end = _clean_text(time_range.get("end")) or None
        if start and not _valid_iso_date(start):
            start = None
        if end and not _valid_iso_date(end):
            end = None
        if start or end:
            out["time_range"] = {"start": start, "end": end}
    media_type = _clean_text(raw.get("media_type")).lower()
    if media_type in {"image", "video", "audio", "text"}:
        out["media_type"] = media_type
    place = _clean_text(raw.get("place"))
    if place:
        out["place"] = place
    aggregation = raw.get("aggregation")
    if isinstance(aggregation, dict):
        op = _clean_text(aggregation.get("op"))
        if op in {"count", "group_by", "first", "last", "exists", "list"}:
            group_by = _clean_text(aggregation.get("group_by"))
            out["aggregation"] = {
                "op": op,
                "group_by": group_by if group_by in {"month", "year", "date", "place", "media"} else None,
            }
    return out


def _coerce_actions(raw):
    actions = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            action_type = _clean_text(item.get("type"))
            if not action_type:
                continue
            actions.append(QueryAction(
                type=action_type,
                target=_clean_text(item.get("target")) or "general",
                coverage=_clean_text(item.get("coverage")) or "best",
            ))
        elif isinstance(item, str) and item.strip():
            actions.append(QueryAction(type=item.strip()))
    return actions


def _coerce_facets(raw):
    facets = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        dimension = _clean_text(item.get("dimension"))
        surface = _clean_text(item.get("surface_text")) or _clean_text(item.get("value"))
        if dimension and surface:
            facets.append(QueryFacet(dimension=dimension, surface_text=surface))
    return facets


def sanitize_query_parse(raw: Any, message: str = "") -> QueryParseDraft:
    """Accept only semantic fields produced by a model.

    Runtime identity, scope, conversation and canonical IDs are deliberately
    ignored here.  They are supplied by ``build_query_spec`` from the request.
    """
    raw = raw if isinstance(raw, dict) else {}
    semantic_conditions = []
    for item in _as_list(raw.get("semantic_conditions")):
        if not isinstance(item, dict):
            continue
        dimension = _clean_text(item.get("dimension"))
        value = _clean_text(item.get("value"))
        if dimension and value:
            semantic_conditions.append({
                "dimension": dimension,
                "value": value,
                "strictness": _clean_text(item.get("strictness")) or SEMANTIC,
                "source_text": _clean_text(item.get("source_text")) or value,
            })
    negative_conditions = []
    for item in _as_list(raw.get("negative_conditions")):
        if isinstance(item, dict) and _clean_text(item.get("value")):
            negative_conditions.append({
                "dimension": _clean_text(item.get("dimension")) or "other",
                "value": _clean_text(item.get("value")),
                "source_text": _clean_text(item.get("source_text")) or _clean_text(item.get("value")),
            })
    requirement = raw.get("result_requirement") if isinstance(raw.get("result_requirement"), dict) else {}
    answer_type = _clean_text(raw.get("answer_type"))
    if answer_type not in ANSWER_TYPES:
        answer_type = "asset_set"
    strategy_hint = _clean_text(raw.get("strategy_hint"))
    if strategy_hint not in STRATEGY_HINTS:
        strategy_hint = ""
    structured = _sanitize_structured(raw.get("structured"))
    actions = _coerce_actions(raw.get("actions"))
    facets = _coerce_facets(raw.get("facets"))
    ambiguities = [_clean_text(item) for item in _as_list(raw.get("ambiguities")) if _clean_text(item)]
    mode = _clean_text(raw.get("mode")).lower()
    if mode not in {"none", "contextual", "evidence"}:
        mode = ""
    intent = _clean_text(raw.get("intent"))
    answer_target = _clean_text(raw.get("answer_target"))
    if not intent and actions:
        primary_type = actions[0].type
        intent = {
            "answer_question": "answer",
            "return_assets": "find_assets",
            "summarize_person": "answer",
            "summarize_event": "answer",
            "timeline": "timeline",
            "compare": "compare",
            "propose_correction": "correction",
        }.get(primary_type, primary_type)
    if not answer_target:
        # Derive the answer target from ANY action that carries a concrete
        # target (summarize_person -> person, return_assets -> general, ...).
        # Previously only answer_question was consulted, so a summarize_person
        # intent kept answer_target="general" and the person summary never ran.
        for action in actions:
            if action.target and action.target != "general":
                answer_target = action.target
                break
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return QueryParseDraft(
        intent=intent or "answer",
        answer_target=answer_target or "general",
        entity_names=list(dict.fromkeys(_clean_text(item) for item in _as_list(raw.get("entity_names")) if _clean_text(item))),
        time_expression=_clean_text(raw.get("time_expression")) or None,
        media_expressions=list(dict.fromkeys(_clean_text(item) for item in _as_list(raw.get("media_expressions")) if _clean_text(item))),
        semantic_conditions=semantic_conditions,
        negative_conditions=negative_conditions,
        result_requirement={
            "mode": _clean_text(requirement.get("mode")) or "best",
            "top_k": max(1, min(100, int(requirement.get("top_k", 10) or 10))),
            "return_original_assets": bool(requirement.get("return_original_assets", False)) or any(action.type == "return_assets" for action in actions),
        },
        answer_type=answer_type,
        strategy_hint=strategy_hint,
        structured=structured,
        # R9: proposed_mode is advisory only.  No action-derived forcing — the
        # Router decides whether the message is household.
        proposed_mode=mode or "none",
        actions=actions,
        facets=facets,
        ambiguities=ambiguities,
        confidence=confidence,
    )


def parse_time_expression(value: str | None):
    value = _clean_text(value)
    # 1) 月/日优先：2023年10月 / 2023-10 / 2023年10月5日
    match = re.search(r"((?:19|20)\d{2})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])?(?:(\d{1,2})\s*日?)?", value)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), match.group(3)
        if day:
            start = datetime(year, month, int(day))
            return start, start + timedelta(days=1)
        start = datetime(year, month, 1)
        end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
        return start, end
    # 2) 纯年份：2023 / 2023年 -> 整年范围
    year_match = re.fullmatch(r"((?:19|20)\d{2})\s*年?", _clean_text(value) or "")
    if year_match:
        year = int(year_match.group(1))
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)
    return None


def _media_type(value: str):
    value = _clean_text(value).lower()
    if any(token in value for token in ("视频", "video", "录像")):
        return "video"
    if any(token in value for token in ("音频", "audio", "录音")):
        return "audio"
    if any(token in value for token in ("文本", "text", "文字")):
        return "text"
    if any(token in value for token in ("照片", "图片", "原图", "image")):
        return "image"
    return None


def build_query_spec(
    parsed: QueryParseDraft,
    *,
    scope_id: str | None,
    viewer_id: str,
    conversation_id: str,
    entity_resolver: Callable[[str], str | None] | None = None,
    query_id: str = "query_runtime",
):
    """Fill request-owned fields and derive the three constraint classes."""
    raw_scope = _clean_text(scope_id)
    all_authorized = scope_id == ""
    scope = raw_scope or "home-default"
    constraints: list[Constraint] = []
    if parsed.time_expression and parse_time_expression(parsed.time_expression):
        constraints.append(Constraint("time", parsed.time_expression, HARD, "asset_metadata", source_text=parsed.time_expression))
    for expression in parsed.media_expressions:
        media_type = _media_type(expression)
        if media_type:
            constraints.append(Constraint("media", media_type, HARD, "asset_metadata", source_text=expression))
    entity_ids = []
    for name in parsed.entity_names:
        entity_id = entity_resolver(name) if entity_resolver else None
        if entity_id:
            entity_ids.append(entity_id)
            constraints.append(Constraint("person", name, HARD, "confirmed_bridge", source_text=name))
    semantic_dimensions = {"place", "activity", "object", "clothing", "visual", "ocr", "person", "relationship", "time", "other", "semantic"}
    for item in parsed.semantic_conditions:
        dimension = item["dimension"] if item["dimension"] in semantic_dimensions else "semantic"
        constraints.append(Constraint(dimension, item["value"], SEMANTIC, "direct_or_possible", source_text=item.get("source_text", item["value"])))
    for item in parsed.negative_conditions:
        constraints.append(Constraint(item["dimension"], item["value"], HARD, "asset_metadata", negated=True, source_text=item["source_text"]))
    return QuerySpec(
        query_id=query_id,
        scope_mode="all_authorized" if all_authorized else "single",
        scope_ids=[] if all_authorized else [scope],
        viewer_id=_clean_text(viewer_id) or "owner",
        conversation_id=_clean_text(conversation_id) or "conversation_runtime",
        intent=parsed.intent,
        answer_target=parsed.answer_target,
        constraints=constraints,
        entity_ids=list(dict.fromkeys(entity_ids)),
        result_requirement=parsed.result_requirement,
        actions=list(parsed.actions),
        facets=list(parsed.facets),
        ambiguities=list(parsed.ambiguities),
    )
