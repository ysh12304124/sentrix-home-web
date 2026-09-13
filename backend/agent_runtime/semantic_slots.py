"""Semantic slot parsing for retrieval (model-driven).

Splits a search question into structured retrieval slots so the code can run
each deterministic component (time bounds, place, person, event) through its
own precise channel instead of one whole-sentence embedding.  The model owns
semantic understanding; the code owns the hard constraints.  Any parse failure
or empty-slot result returns ``None`` so the caller falls back to the existing
whole-sentence retrieval path (a slot parse must never shrink recall).
"""

from __future__ import annotations

import inspect
import json
import re

_SLOT_PROMPT = """你是家庭照片检索的语义解析器。把用户问题拆成结构化检索槽位，供代码做精确检索。
只输出 JSON，不要解释：
{
  "time": {"expr": "2017年秋天", "year": 2017, "months": [], "days": [],
           "kind": "absolute|relative|none", "certainty": "high|low"},
  "place": {"name": "邯郸馆陶", "hint": "邯郸市馆陶县", "certainty": "high|low"},
  "person": [{"name": "父母", "certainty": "high"}],
  "event": {"name": "婚礼", "certainty": "high"},
  "object": ["展架", "小汽车"],
  "query_core": "在新人的展架旁拍的照片"
}
规则：
- time：只提取用户明确表达的时间信息，**绝不推测年份**（会筛错时间）。
  · year：用户说出年份才填（"2017年秋天"→2017）；问题是"哪一年/几几年"或没给年份 → year 留空不要猜。
  · months：明确的月份列表（"9月"→[9]；"秋天/国庆节"这类可交给代码，不必填）。
  · days：明确的日列表（"1号到7号"→[1,2,3,4,5,6,7]）；不确定的节日日期可不填。
  · expr：原样保留时间词（"2017年秋天"/"国庆节"/"去年10月"）；季节/节日/相对时间的补全由代码做，
    模型不要在 year/months/days 里推算（例如"去年10月"不要写成固定的 year，只写 expr）。
  确实没有时间信息 → kind:"none"。
- place：给标准地名 hint（省市区/区县）；识别不出给 certainty=low 或留空。
- person：从已提及的家庭成员/人名提取，未提及则为空数组。
- event：事件/活动类型（婚礼、出游、聚餐等）；无则 certainty=low 或留空。
- object：照片里的可见对象/物品。
- query_core：剔除时间/地点/人物/事件后的语义核心；若剩余内容不足以表达问题，
  则保留原问题原文。
拆不出任何槽位时输出：
{"time":{},"place":{},"person":[],"event":{},"object":[],"query_core":"<原问题>"}"""


def parse_semantic_slots(question: str, chat_fn) -> dict | None:
    """模型拆槽。成功返回 slots dict；失败或无确定性槽位返回 None（调用方回退整句）。

    ``chat_fn`` 是 gamma.chat 风格的调用：第一个参数为完整 prompt 字符串（系统指令
    与用户问题拼在一起，因为 gamma.chat 只支持单条 user 消息、不接受 messages 列表——
    若传列表会把系统提示词当字符串丢给模型，拆槽必然失败）。
    """
    if not (question or "").strip():
        return None
    prompt = _SLOT_PROMPT + "\n\n用户问题：" + question
    try:
        signature = inspect.signature(chat_fn)
        if "call_type" in signature.parameters:
            raw = chat_fn(prompt, call_type="slot_extract") or ""
        else:
            raw = chat_fn(prompt) or ""
    except Exception:
        return None
    payload = _parse_json(raw)
    if payload is None:
        rewritten = _rewrite_json(raw)
        payload = _parse_json(rewritten) if rewritten else None
    if not isinstance(payload, dict):
        return None
    return _normalize_slots(payload, question)


def _parse_json(raw):
    text = str(raw or "").strip()
    text = re.sub(r"```(?:json|JSON)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _rewrite_json(raw):
    """保守格式重写：仅当词法结构确认只缺最外层一个 } 时补一个，否则放弃。"""
    text = str(raw or "").strip()
    try:
        if text.count("{") - text.count("}") == 1:
            candidate = text + "}"
            json.loads(candidate)
            return candidate
    except (TypeError, ValueError):
        pass
    return ""


# 代码确定性补全：季节/节日的月、日由这里映射，不让模型猜年份/算日期。
_SEASON_MONTHS = (
    ("春天", {3, 4, 5}), ("春季", {3, 4, 5}),
    ("夏天", {6, 7, 8}), ("夏季", {6, 7, 8}),
    ("秋天", {9, 10, 11}), ("秋季", {9, 10, 11}),
    ("冬天", {12, 1, 2}), ("冬季", {12, 1, 2}),
)
_CN_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "两": 2}


def _cn_month_to_int(text: str) -> int | None:
    """把中文月份（九月/十月/十一月/十二月）转阿拉伯数字；阿拉伯数字原样。"""
    text = (text or "").strip()
    if text.isdigit():
        return int(text)
    if not text or any(ch not in _CN_DIGIT for ch in text):
        return None
    if "十" not in text:
        return _CN_DIGIT.get(text)
    tens, _, ones = text.partition("十")
    return (_CN_DIGIT.get(tens, 1) if tens else 1) * 10 + _CN_DIGIT.get(ones, 0)


_RELATIVE_TIME_MARKERS = (
    "去年", "今年", "前年", "上个月", "上上个月", "这两年", "近两年",
    "近一年", "最近一年", "最近两年",
)


def _expand_time_components(year, months, days, expr: str):
    """用 expr 里的季节/节日/显式月日做确定性补全（绝不补年份）。

    相对时间（去年/今年/上个月…）年份由代码在调用方换算成绝对表达式，这里不做
    任何跨年分量补全——否则"去年10月"会被展开成"任意年份的10月"而筛错年份。
    """
    months = {int(item) for item in (months or ()) if str(item).isdigit()}
    days = {int(item) for item in (days or ()) if str(item).isdigit()}
    low = str(expr or "")
    if any(marker in low for marker in _RELATIVE_TIME_MARKERS):
        return year, months, days
    for keyword, ms in _SEASON_MONTHS:
        if keyword in low:
            months |= ms
    if "国庆" in low:
        months.add(10)
        days |= set(range(1, 8))
    for match in re.findall(r"(?<!\d)(\d{1,2})\s*月", low):
        value = int(match)
        if 1 <= value <= 12:
            months.add(value)
    for match in re.findall(r"[一二三四五六七八九十两]{1,3}\s*月", low):
        value = _cn_month_to_int(match[:-1])
        if value:
            months.add(value)
    if year is None:
        year_match = re.search(r"20\d{2}", low)
        if year_match:
            year = int(year_match.group())
    return year, months, days


def _normalize_slots(payload: dict, question: str) -> dict | None:
    t = payload.get("time") or {}
    p = payload.get("place") or {}
    ev = payload.get("event") or {}
    obj = payload.get("object") or []
    if not isinstance(obj, list):
        obj = [str(obj)] if obj else []
    persons = payload.get("person") or []
    if isinstance(persons, str):
        persons = [{"name": persons}]
    persons = [item for item in persons
               if isinstance(item, dict) and item.get("name")]

    expr = str(t.get("expr") or "") if isinstance(t, dict) else ""
    try:
        model_year = int(t.get("year")) if isinstance(t, dict) and t.get("year") not in (None, "") else None
    except (TypeError, ValueError):
        model_year = None
    raw_months = t.get("months") if isinstance(t, dict) else []
    raw_days = t.get("days") if isinstance(t, dict) else []
    if not isinstance(raw_months, list):
        raw_months = [raw_months] if raw_months else []
    if not isinstance(raw_days, list):
        raw_days = [raw_days] if raw_days else []
    year, months, days = _expand_time_components(model_year, raw_months, raw_days, expr)

    slots = {
        "time": {
            "expr": expr,
            "year": year,
            "months": sorted(months),
            "days": sorted(days),
            "kind": str(t.get("kind") or "none") if isinstance(t, dict) else "none",
            "certainty": str(t.get("certainty") or "low") if isinstance(t, dict) else "low",
        },
        "place": {
            "name": str(p.get("name") or "") if isinstance(p, dict) else "",
            "hint": str(p.get("hint") or "") if isinstance(p, dict) else "",
            "certainty": str(p.get("certainty") or "low") if isinstance(p, dict) else "low",
        },
        "person": persons,
        "event": {
            "name": str(ev.get("name") or "") if isinstance(ev, dict) else "",
            "certainty": str(ev.get("certainty") or "low") if isinstance(ev, dict) else "low",
        },
        "object": [str(item) for item in obj],
        "query_core": str(payload.get("query_core") or "").strip() or question,
    }
    # expr 非空也视为有槽：相对时间分量由代码换算（否则"去年10月"会因无分量而丢时间约束）
    time_has = bool(year or months or days or expr)
    # 无任何确定性槽位 → None（回退整句检索）
    has_slot = bool(time_has or slots["place"]["name"]
                    or slots["person"] or slots["event"].get("name"))
    return slots if has_slot else None
