"""12B Response Writer (RX-4).

Turns an AnswerBrief + ResponsePlan into a natural answer.  The model has full
expression freedom within the fact boundary: it sees only display handles,
facts, uncertainties and prohibitions — never asset ids, condition keys or
scores.  When the model is unavailable or unusable, a deterministic per-mode
safe fallback is returned (never a database report).
"""

from __future__ import annotations

import json

from .answer_brief import AnswerBrief
from .model_clients import parse_json_response
from .response_plan import ResponsePlan

_PROHIBITED = [
    "数据库", "检索", "工具调用", "asset_", "obs_", "event_", "entity_",
    "matched", "possible", "unknown", "condition", "fusion_score",
    "retrieval_trace", "根据本地事件记忆检索到",
]

_MODE_INSTRUCTIONS = {
    "asset_delivery": (
        "用户要的是原图。简短确认已经找到并展示照片，列出将展示的照片代号。"
        "不要做匹配度分析，不要说不确定的话，不要提内部字段。1-2 句即可。"
    ),
    "exact_result": (
        "先给结论（找到了什么、几张最相关），再简要说明可以确认的事实，最后自然提到图片。"
        "不确定的维度用自然语言说明，不要说成确定。"
    ),
    "approximate_result": (
        "先说没有完全匹配的照片；说明接近在哪里（哪些维度对上）；说明还不能确认什么；"
        "提示将展示最接近的几张照片。"
    ),
    "no_result": (
        "明确说目前没有找到足够可靠的证据；给出一个具体的补充方向（例如补充人物、地点或日期）；"
        "不要罗列内部字段，不要编造。"
    ),
    "person_summary": (
        "基于事实分层总结人物：可直接陈述的、记录中出现的模式、无法确认的。"
        "如果没有任何可确认事实，只说目前证据不足并提示补充照片，不要编造性格、偏好或关系。"
    ),
    "clarify": (
        "只问一个高价值的问题来弄清用户要什么，不要展示候选内容。"
    ),
    "chat": (
        "正常自然对话，不要提家庭记忆。"
    ),
}


class _FactsIndex:
    def __init__(self, brief):
        self._ids = {f.fact_id for f in brief.facts}

    def resolve(self, fact_id):
        return fact_id if fact_id in self._ids else None


def build_prompt(brief: AnswerBrief, plan: ResponsePlan, message: str) -> str:
    payload = brief.writer_payload()
    instruction = _MODE_INSTRUCTIONS.get(brief.response_mode, _MODE_INSTRUCTIONS["chat"])
    banned = list(dict.fromkeys([*_PROHIBITED, *brief.must_not_say]))
    return (
        "你是 Sentrix，一个自然、克制的家庭记忆助手。\n"
        "下面给出用户目标、受控事实、可展示图片和禁止事项。\n"
        "请写一段自然回答：组织顺序、语气、长短由你决定，但不得超出这些事实。\n\n"
        f"回答要求：{instruction}\n"
        f"禁止事项（不得出现在回答里）：{json.dumps(banned, ensure_ascii=False)}\n\n"
        f"受控信息：{json.dumps(payload, ensure_ascii=False)}\n\n"
        f"用户问题：{message}\n\n"
        "只输出 JSON：\n"
        '{"text": "自然回答", "statements": [{"text": "句子原文", "fact_id": "fact_id或null", "certainty": "confirmed|possible|uncertainty"}]}'
    )


def _normalize_statements(raw, brief: AnswerBrief):
    facts = _FactsIndex(brief)
    statements = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            statements.append({
                "text": text,
                "fact_id": facts.resolve(item.get("fact_id")),
                "certainty": item.get("certainty") or _certainty_for(item.get("fact_id"), brief),
            })
    return statements


def _certainty_for(fact_id, brief):
    for fact in brief.facts:
        if fact.fact_id == fact_id:
            return fact.certainty
    return "uncertainty"


def write_response(brief: AnswerBrief, plan: ResponsePlan, message: str, gamma=None, router=None):
    """Return (answer, statements) or (None, []) when the model is unavailable."""
    prompt = build_prompt(brief, plan, message)
    try:
        if router is not None:
            raw = router.chat("answer", prompt, json_mode=True)
        elif gamma is not None and hasattr(gamma, "chat"):
            raw = gamma.chat(prompt, json_mode=True, role="answer")
        else:
            return None, []
    except Exception:
        return None, []
    parsed = parse_json_response(raw)
    if isinstance(parsed, dict):
        text = str(parsed.get("text") or "").strip()
        if text:
            return text, _normalize_statements(parsed.get("statements"), brief)
    text = str(raw or "").strip()
    if text:
        return text, []
    return None, []


def safe_fallback(brief: AnswerBrief, plan: ResponsePlan) -> tuple[str, list]:
    """Deterministic per-mode fallback — never a database report."""
    mode = brief.response_mode
    if mode == "asset_delivery":
        n = len(brief.visible_assets)
        return (f"已找到并展示以下{n}张照片。" if n else "目前没有找到可以展示的照片。"), []
    if mode == "no_result":
        return "目前没有找到足够可靠的依据。可以补充人物、地点或日期再试试。", []
    if mode == "approximate_result":
        return "没有完全匹配的照片；下面这几张比较接近，但部分细节还不能确认。", []
    if mode == "person_summary":
        if brief.facts:
            return "根据现有记录整理了下面这个人物的介绍。", []
        return "目前还没有足够的照片或记录来介绍这个人，只确认了这个人。可以补充照片后再介绍。", []
    if mode == "exact_result":
        return "我找到了一些符合你描述的照片，下面是其中最相关的几张。", []
    if mode == "clarify":
        return "你是想让我在你存下的照片或记忆里找这个，还是想聊点别的？", []
    return "我在听。", []
