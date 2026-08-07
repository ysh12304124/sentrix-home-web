"""VisibleEvidence selection (RX-3).

Retrieval can internally recall many candidates; the user only ever sees a
carefully chosen subset.  ``select_visible_assets`` picks at most a few
exact/strong images plus at most a few approximate ones that carry an
explainable supported aspect, collapses near-duplicate groups to a
representative, and returns ``[]`` when every candidate is unknown (unless the
user explicitly asked for all relevant results).
"""

from __future__ import annotations

from .answer_brief import VisibleAsset, condition_aspects

_LEVEL_ORDER = {"exact": 0, "strong": 1, "approximate": 2}
_DEFAULT_MAX_EXACT = 3
_DEFAULT_MAX_APPROXIMATE = 3
_ALL_RELEVANT_CAP = 60


def _recall_strength(item):
    attributions = item.get("attributions") or []
    if not attributions:
        return None
    best = 0.0
    for attr in attributions:
        score = attr.get("score") or 0.0
        kind = attr.get("score_kind") or ""
        if kind == "cosine_similarity":
            best = max(best, float(score))
        elif kind == "token_hits":
            best = max(best, min(1.0, float(score) / 4.0))
        elif kind == "discrete":
            best = max(best, 1.0)
        elif kind == "adjacency":
            best = max(best, 0.1)
    return round(best, 4)


def _all_unknown(item) -> bool:
    """True when the item has conditions and every one of them is unknown."""
    conds = (item.get("condition_results") or {}).values()
    if not conds:
        return False
    return all(cond.get("status") == "unknown" for cond in conds)


def _has_matched(item) -> bool:
    return any(cond.get("status") == "matched"
               for cond in (item.get("condition_results") or {}).values())


def _display_reason(level: str) -> str:
    if level == "exact":
        return "完全符合你描述的条件"
    if level == "strong":
        return "高度接近你描述的内容"
    return "最接近你描述，但部分维度还不能确认"


def _dedupe_near_duplicates(items, keep_all: bool):
    """Collapse near-duplicate groups to a representative unless keep_all."""
    by_group: dict[str, list] = {}
    ungrouped: list = []
    for item in items:
        group = item.get("near_duplicate_group")
        if group:
            by_group.setdefault(group, []).append(item)
        else:
            ungrouped.append(item)
    out = list(ungrouped)
    for group, members in by_group.items():
        members.sort(key=lambda i: _recall_strength(i) or 0.0, reverse=True)
        if keep_all:
            out.extend(members)
            continue
        representative = members[0]
        representative = dict(representative)
        representative["near_duplicate_size"] = len(members)
        out.append(representative)
    return out


def select_visible_assets(packet, *, all_relevant: bool = False,
                          max_exact: int = _DEFAULT_MAX_EXACT,
                          max_approximate: int = _DEFAULT_MAX_APPROXIMATE) -> list[VisibleAsset]:
    """Pick the user-visible image subset (display handles only)."""
    items = list(packet.assets or [])

    def level_rank(item):
        return _LEVEL_ORDER.get(item.get("level"), 3)

    items.sort(key=lambda i: (level_rank(i), -(_recall_strength(i) or 0.0)))
    exact_strong = [i for i in items if i.get("level") in {"exact", "strong"}]
    approximate = [i for i in items if i.get("level") == "approximate"]

    if all_relevant:
        chosen = exact_strong + approximate
        chosen = _dedupe_near_duplicates(chosen, keep_all=True)
        chosen = chosen[:_ALL_RELEVANT_CAP]
    else:
        explainable = [i for i in approximate if not _all_unknown(i)]
        chosen = exact_strong[:max_exact] + explainable[:max_approximate]
        chosen = _dedupe_near_duplicates(chosen, keep_all=False)

    visible: list[VisibleAsset] = []
    for index, item in enumerate(chosen, 1):
        supported, uncertain = condition_aspects(item)
        visible.append(VisibleAsset(
            asset_id=item["asset_id"],
            display_handle=f"照片{index}",
            captured_at=item.get("captured_at"),
            result_level=item.get("level") or "approximate",
            supported_aspects=supported,
            uncertain_aspects=uncertain,
            display_reason=_display_reason(item.get("level") or "approximate"),
            file_name=item.get("file_name"),
            media_url=f"/api/assets/{item['asset_id']}/file" if item.get("media_type") == "image" else None,
            near_duplicate_size=item.get("near_duplicate_size", 1),
        ))
    return visible
