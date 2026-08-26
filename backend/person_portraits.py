"""Evidence-bound living portrait compilation and validation.

The portrait writer only consumes a bounded evidence pack. Confirmed facts,
objective moments, model explanations and unconfirmed hypotheses stay in
separate layers; a portrait is not saved until it passes deterministic checks.
"""

import re
from collections import Counter

from .person_graph import ROLE_OPTIONS
from .person_moments import SENSITIVE_TERMS

PERSON_PORTRAIT_PROMPT = (
    "你是家庭相册的人物画像作者。输入是一个结构化的证据包，包含确认事实、客观人物瞬间、"
    "模型解释和未确认假设。\n"
    "以具体反复出现的瞬间和互动为主体，写一段 80 到 500 个汉字的鲜活人物画像。\n"
    "允许‘爱热闹、有仪式感、常照顾孩子’等低风险印象；不要以次数开头；\n"
    "未确认的解释必须使用‘似乎、可能、从照片看’等不确定措辞；禁止敏感推断和虚构。\n"
    "严格返回 JSON：\n"
    '{"portrait_text":"...","themes":[{"title":"...","summary":"...",'
    '"evidence_refs":[{"kind":"person_moment","id":"..."}]}]}'
)

_HEDGE_TERMS = ("似乎", "可能", "从照片看", "看起来", "像是")
_UNDECIDED = ("本人", "无法判断")
_HANZI = re.compile(r"[一-鿿]")


def _contains_sensitive(text):
    return any(term in text for term in SENSITIVE_TERMS)


def _mentions_role(text):
    for role in ROLE_OPTIONS:
        if role in _UNDECIDED:
            continue
        if role in text:
            return True
    return False


def compile_portrait_evidence(store, person_id, max_moments=12, per_event=2):
    person = store.get_entity(person_id) or {}
    scope_id = person.get("scope_id") or "home-default"
    confirmed_name = person.get("canonical_name") if person.get("status") == "confirmed" else None
    confirmed_role = person.get("family_role") if person.get("role_state") == "confirmed" else None

    moments = store.list_person_moments(person_id=person_id, scope_id=scope_id, status="active")
    by_event = {}
    for moment in moments:
        by_event.setdefault(moment.get("event_id"), []).append(moment)
    sampled = []
    for items in by_event.values():
        ordered = sorted(items, key=lambda m: -float(m.get("confidence") or 0))
        sampled.extend(ordered[:per_event])
    seen = {}
    for moment in sampled:
        key = (moment.get("action_text") or "", tuple(moment.get("interaction_target_ids") or []))
        current = seen.get(key)
        if current is None or float(moment.get("confidence") or 0) > float(current.get("confidence") or 0):
            seen[key] = moment
    sampled = sorted(seen.values(), key=lambda m: str(m.get("created_at") or ""))[:max_moments]
    moment_evidence = [{
        "kind": "person_moment",
        "id": moment["id"],
        "asset_id": moment["asset_id"],
        "observation_id": moment["observation_id"],
        "event_id": moment["event_id"],
        "action_text": moment.get("action_text") or "",
        "interaction_text": moment.get("interaction_text") or "",
        "participation_style": moment.get("participation_style") or "",
        "visible_affect": moment.get("visible_affect") or "",
        "confidence": float(moment.get("confidence") or 0),
    } for moment in sampled]

    confirmed_relationships = [
        relationship for relationship in store.list_person_relationships(scope_id)
        if relationship["subject_entity_id"] == person_id or relationship["object_entity_id"] == person_id
    ]
    suggested_relationships = [
        hypothesis for hypothesis in store.list_relationship_hypotheses(scope_id, status="suggested")
        if hypothesis["subject_person_id"] == person_id or hypothesis["object_person_id"] == person_id
    ]

    recurring = Counter(
        moment.get("action_text") or "" for moment in moment_evidence if moment.get("action_text")
    ).most_common(3)
    appearance_themes = []
    for action, count in recurring:
        appearance_themes.append({
            "title": f"反复{action}",
            "summary": f"从照片看，他常{action}。",
            "evidence_refs": [
                {"kind": "person_moment", "id": moment["id"]}
                for moment in moment_evidence if moment.get("action_text") == action
            ][:2],
        })

    return {
        "person": {
            "id": person_id,
            "display_name": person.get("canonical_name") or "未命名成员",
            "confirmed_name": confirmed_name,
            "confirmed_role": confirmed_role,
            "identity_state": person.get("identity_state") or "clustered",
        },
        "confirmed_relationships": confirmed_relationships,
        "suggested_relationships": suggested_relationships,
        "moments": moment_evidence,
        "recurring_patterns": appearance_themes,
        "appearance_themes": appearance_themes,
        "time_changes": [],
        "contradictions": [],
        "unknowns": [],
    }


def _evidence_ref_valid(pack, ref):
    kind = str(ref.get("kind") or "")
    ref_id = str(ref.get("id") or "")
    if kind == "person_moment":
        return any(moment["id"] == ref_id for moment in pack.get("moments") or [])
    if kind in ("relationship", "semantic_claim", "appearance"):
        return True
    return False


def validate_portrait(pack, portrait):
    errors = []
    text = str(portrait.get("portrait_text") or "")
    hanzi = len(_HANZI.findall(text))
    if hanzi < 80:
        errors.append(f"portrait too short: {hanzi} hanzi < 80")
    if hanzi > 500:
        errors.append(f"portrait too long: {hanzi} hanzi > 500")
    themes = portrait.get("themes") or []
    if not isinstance(themes, list) or not (2 <= len(themes) <= 6):
        errors.append(f"themes must be 2-6, got {len(themes) if isinstance(themes, list) else 'n/a'}")
    for theme in themes:
        refs = theme.get("evidence_refs") or []
        if not refs:
            errors.append("theme missing evidence_refs")
        for ref in refs:
            if not _evidence_ref_valid(pack, ref):
                errors.append(f"invalid evidence ref: {ref}")
    if _contains_sensitive(text):
        errors.append("portrait contains sensitive attribute language")
    person = pack.get("person") or {}
    if not person.get("confirmed_role") and _mentions_role(text):
        if not any(hedge in text for hedge in _HEDGE_TERMS):
            errors.append("unconfirmed role without hedging")
    return (not errors, errors)


def deterministic_portrait(pack):
    moments = pack.get("moments") or []
    actions = [
        moment.get("action_text") or "" for moment in moments if moment.get("action_text")
    ]
    top = Counter(actions).most_common(3)
    if not top:
        return {
            "portrait_text": "从照片看，这位家庭成员在相册中反复出现，但具体瞬间证据不足，无法形成完整画像。",
            "themes": [],
        }
    body = "，".join(f"他常{action}" for action, _count in top)
    text = f"从照片看，{body}。他从照片看常出现在这些瞬间里，让人感到熟悉和安心。"
    themes = []
    for action, _count in top:
        refs = [
            {"kind": "person_moment", "id": moment["id"]}
            for moment in moments if moment.get("action_text") == action
        ][:2]
        themes.append({
            "title": f"反复{action}",
            "summary": f"从照片看，他常{action}。",
            "evidence_refs": refs,
        })
    return {"portrait_text": text, "themes": themes}


def normalize_writer_output(parsed):
    themes = []
    raw_themes = parsed.get("themes") if isinstance(parsed, dict) else []
    for theme in raw_themes or []:
        title = str(theme.get("title") or "").strip()
        summary = str(theme.get("summary") or "").strip()
        refs = []
        for ref in theme.get("evidence_refs") or []:
            if isinstance(ref, dict) and ref.get("id"):
                refs.append({"kind": str(ref.get("kind") or "person_moment"), "id": str(ref["id"])})
        themes.append({"title": title, "summary": summary, "evidence_refs": refs})
    return {
        "portrait_text": str(parsed.get("portrait_text") or "").strip(),
        "themes": themes,
    }
