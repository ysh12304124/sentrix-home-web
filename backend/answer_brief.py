"""AnswerBrief — the formal boundary between retrieval and natural language (RX-1).

The AnswerBrief is built by deterministic code from the EvidencePacket, the
QuerySpec and session state.  It is the ONLY thing the 12B Response Writer may
consume: user goal, controllable facts, uncertainties, visible images (display
handles only), and the presentation contract.  ANN scores, condition keys and
internal IDs never enter the brief's serialized form that reaches the writer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .query_contracts import QuerySpec

# response_mode values (D2 / §5.1).
RESPONSE_MODES = {
    "chat", "exact_result", "approximate_result", "no_result",
    "asset_delivery", "person_summary", "clarify",
}


def _human_text(key: str, status: str) -> str:
    """User-facing text for a condition — never exposes the internal key."""
    value = key.split(":", 1)[1] if ":" in key else key
    if status == "matched":
        return f"记录中有「{value}」"
    if status == "possible":
        return f"记录中可能有「{value}」，但无法完全确认"
    return "目前无法确认其中的关键活动或视觉细节。"


def condition_aspects(item: dict) -> tuple[list[str], list[str]]:
    """Derive supported / uncertain aspect labels from a packet asset.

    Returned labels are human-facing (no condition_key, no score).  A matched
    condition is a supported aspect; possible/unknown become uncertain aspects.
    """
    supported: list[str] = []
    uncertain: list[str] = []
    for key, cond in (item.get("condition_results") or {}).items():
        value = key.split(":", 1)[1] if ":" in key else key
        if not value:
            continue
        status = cond.get("status")
        if status == "matched":
            supported.append(value)
        elif status == "possible":
            uncertain.append(f"{value}（可能）")
        elif status == "unknown":
            uncertain.append(value)
    return supported, uncertain


@dataclass
class Fact:
    fact_id: str
    text: str
    certainty: str            # confirmed | possible
    evidence_ids: list[str] = field(default_factory=list)
    allowed_paraphrases: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"fact_id": self.fact_id, "text": self.text,
                "certainty": self.certainty, "evidence_ids": self.evidence_ids}


@dataclass
class Uncertainty:
    topic: str
    status: str               # unknown | possible
    reason: str = ""

    def as_dict(self):
        return {"topic": self.topic, "status": self.status, "reason": self.reason}


@dataclass
class VisibleAsset:
    asset_id: str
    display_handle: str
    captured_at: str | None = None
    result_level: str = "approximate"
    supported_aspects: list[str] = field(default_factory=list)
    uncertain_aspects: list[str] = field(default_factory=list)
    display_reason: str = ""
    file_name: str | None = None
    media_url: str | None = None
    near_duplicate_size: int = 1

    def as_dict(self):
        # internal id stays for the admin layer; the writer sees only the handle.
        return {"asset_id": self.asset_id, "display_handle": self.display_handle,
                "captured_at": self.captured_at, "result_level": self.result_level,
                "supported_aspects": self.supported_aspects,
                "uncertain_aspects": self.uncertain_aspects,
                "display_reason": self.display_reason,
                "file_name": self.file_name, "media_url": self.media_url,
                "near_duplicate_size": self.near_duplicate_size}


@dataclass
class Presentation:
    show_images: bool = False
    auto_expand_images: bool = False
    show_evidence_entry: bool = False
    show_debug: bool = False

    def as_dict(self):
        return {"show_images": self.show_images, "auto_expand_images": self.auto_expand_images,
                "show_evidence_entry": self.show_evidence_entry, "show_debug": self.show_debug}


@dataclass
class AnswerBrief:
    brief_id: str
    user_goal: str
    response_mode: str
    direct_answer: str = ""
    facts: list[Fact] = field(default_factory=list)
    uncertainties: list[Uncertainty] = field(default_factory=list)
    visible_assets: list[VisibleAsset] = field(default_factory=list)
    hidden_assets_count: int = 0
    must_not_say: list[str] = field(default_factory=list)
    presentation: Presentation = field(default_factory=Presentation)
    follow_up_options: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "brief_id": self.brief_id, "user_goal": self.user_goal,
            "response_mode": self.response_mode, "direct_answer": self.direct_answer,
            "facts": [f.as_dict() for f in self.facts],
            "uncertainties": [u.as_dict() for u in self.uncertainties],
            "visible_assets": [v.as_dict() for v in self.visible_assets],
            "hidden_assets_count": self.hidden_assets_count,
            "must_not_say": self.must_not_say,
            "presentation": self.presentation.as_dict(),
            "follow_up_options": self.follow_up_options,
        }

    def writer_payload(self):
        """The slice the writer may see — display handles, no internal keys."""
        return {
            "user_goal": self.user_goal, "response_mode": self.response_mode,
            "facts": [{"fact_id": f.fact_id, "text": f.text, "certainty": f.certainty}
                      for f in self.facts],
            "uncertainties": [{"topic": u.topic, "status": u.status, "reason": u.reason}
                              for u in self.uncertainties],
            "visible_assets": [{"display_handle": v.display_handle,
                                "captured_at": v.captured_at,
                                "supported_aspects": v.supported_aspects,
                                "uncertain_aspects": v.uncertain_aspects,
                                "near_duplicate_size": v.near_duplicate_size}
                               for v in self.visible_assets],
            "hidden_assets_count": self.hidden_assets_count,
            "must_not_say": self.must_not_say,
            "presentation": self.presentation.as_dict(),
        }


def user_goal(spec: QuerySpec) -> str:
    """Map the user's parsed actions/target to one high-level goal."""
    if spec.result_requirement.get("return_original_assets") or any(
            a.type == "return_assets" for a in (spec.actions or [])):
        return "deliver_images"
    if spec.answer_target == "person":
        return "person_summary"
    if spec.answer_target == "clothing":
        return "clothing_check"
    return "find_and_explain_images"


def derive_response_mode(user_goal: str, packet) -> str:
    """Pick the response form from the goal + what the packet actually holds.

    Person summaries always use the person_summary form; its no-evidence state
    is expressed as an empty facts list (the writer then produces a gap answer).
    """
    if user_goal == "person_summary":
        return "person_summary"
    if user_goal == "deliver_images":
        return "asset_delivery" if (packet.assets or packet.exact_results or packet.approximate_results) else "no_result"
    if packet.exact_results or packet.strong_results:
        return "exact_result"
    if packet.approximate_results:
        return "approximate_result"
    return "no_result"


def build_facts_and_uncertainties(packet) -> tuple[list[Fact], list[Uncertainty]]:
    """One fact per condition_key with union evidence_ids (mirrors _allowed_facts).

    Does not fabricate: no evidence -> empty facts.
    """
    facts_by_key: dict[str, dict] = {}
    possible_by_key: dict[str, dict] = {}
    for item in packet.assets:
        for key, condition in (item.get("condition_results") or {}).items():
            status = condition.get("status")
            bucket = facts_by_key if status == "matched" else possible_by_key if status == "possible" else None
            if bucket is None:
                continue
            entry = bucket.get(key)
            if entry is None:
                entry = {"text": _human_text(key, status), "certainty": "confirmed" if status == "matched" else "possible",
                         "evidence_ids": [], "condition_key": key}
                bucket[key] = entry
            for evidence_id in item.get("evidence_ids", []):
                if evidence_id not in entry["evidence_ids"]:
                    entry["evidence_ids"].append(evidence_id)
    facts: list[Fact] = []
    for index, entry in enumerate([*facts_by_key.values(), *possible_by_key.values()], 1):
        facts.append(Fact(fact_id=f"fact_{index}", text=entry["text"],
                          certainty=entry["certainty"], evidence_ids=entry["evidence_ids"]))
    uncertainties: list[Uncertainty] = []
    for gap in packet.gaps or []:
        topic = gap.get("condition") or gap.get("dimension") or "细节"
        uncertainties.append(Uncertainty(topic=str(topic).split(":", 1)[-1], status="unknown",
                                         reason=str(gap.get("reason") or "没有直接证据支持")))
    for item in packet.assets:
        for key, condition in (item.get("condition_results") or {}).items():
            if condition.get("status") == "unknown":
                value = key.split(":", 1)[1] if ":" in key else key
                if value and not any(u.topic == value for u in uncertainties):
                    uncertainties.append(Uncertainty(topic=value, status="unknown",
                                                     reason="没有直接视觉或人物绑定证据"))
    return facts, uncertainties


def build_must_not_say(user_goal: str, packet, facts: list[Fact],
                       uncertainties: list[Uncertainty]) -> list[str]:
    """Deterministic prohibitions the writer must not emit."""
    banned: list[str] = []
    for u in uncertainties:
        if u.status == "unknown" and u.topic and u.topic not in {"细节"}:
            banned.append(f"确定{u.topic}")
            banned.append(f"肯定是{u.topic}")
    if user_goal == "person_summary" and not facts:
        banned += ["多次出现", "常常", "经常", "喜欢", "性格"]
    return list(dict.fromkeys(banned))


def build_answer_brief(message: str, spec: QuerySpec, packet, *,
                       visible_assets: list[VisibleAsset] | None = None,
                       decision=None) -> AnswerBrief:
    """Deterministic AnswerBrief for one turn.  Never calls a model."""
    goal = user_goal(spec)
    mode = derive_response_mode(goal, packet)
    facts, uncertainties = build_facts_and_uncertainties(packet)
    visible = list(visible_assets or [])
    total_assets = len(packet.assets)
    if mode == "asset_delivery" and not visible:
        mode = "no_result"
    hidden = max(0, total_assets - len(visible))
    presentation = Presentation(
        show_images=bool(visible),
        auto_expand_images=mode == "asset_delivery",
        show_evidence_entry=bool(visible or facts or uncertainties or packet.gaps),
        show_debug=False,
    )
    must_not_say = build_must_not_say(goal, packet, facts, uncertainties)
    return AnswerBrief(
        brief_id=f"brief_{uuid.uuid4().hex[:12]}",
        user_goal=goal,
        response_mode=mode,
        direct_answer=_direct_answer(mode, len(visible), len(facts)),
        facts=facts,
        uncertainties=uncertainties,
        visible_assets=visible,
        hidden_assets_count=hidden,
        must_not_say=must_not_say,
        presentation=presentation,
    )


def _direct_answer(mode: str, visible_count: int, fact_count: int) -> str:
    """Internal controlled conclusion used to orient the writer, not user text."""
    if mode == "asset_delivery":
        return f"已找到 {visible_count} 张授权图片并展示。"
    if mode == "exact_result":
        return f"存在 {fact_count} 条可确认事实，展示 {visible_count} 张最相关图片。"
    if mode == "approximate_result":
        return f"没有完全匹配；展示 {visible_count} 张最接近的图片，并说明不能确认的维度。"
    if mode == "no_result":
        return "没有找到足够可靠的证据。"
    if mode == "person_summary":
        return f"人物总结：{fact_count} 条可陈述事实，其余保留未知。" if fact_count else "人物证据不足，只给 gap。"
    return "正常聊天。"
