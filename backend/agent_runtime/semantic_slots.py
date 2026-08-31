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
  "time": {"expr": "2017年秋天", "bounds": ["2017-09-01", "2017-11-30"],
           "kind": "absolute|relative|none", "certainty": "high|low"},
  "place": {"name": "邯郸馆陶", "hint": "邯郸市馆陶县", "certainty": "high|low"},
  "person": [{"name": "父母", "certainty": "high"}],
  "event": {"name": "婚礼", "certainty": "high"},
  "object": ["展架", "小汽车"],
  "query_core": "在新人的展架旁拍的照片"
}
规则：
- time：**只要问题里出现任何时间信息（年份/月份/日期/季节/节日/哪一年），必须给 bounds**
  （[开始,结束]，格式 YYYY-MM-DD）。"2017年""2017年9月""2017年秋天""国庆节"都必须转成
  bounds（季节/节日：春天03-01~05-31、夏天06-01~08-31、秋天09-01~11-30、冬天12-01~02-28、
  国庆10-01~10-07）。确实没有时间信息才 kind=none、bounds 为空数组。
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

    slots = {
        "time": {
            "expr": str(t.get("expr") or "") if isinstance(t, dict) else "",
            "bounds": [],
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
    bounds = t.get("bounds") if isinstance(t, dict) else []
    if (isinstance(bounds, list) and len(bounds) == 2
            and bounds[0] and bounds[1]):
        slots["time"]["bounds"] = [str(bounds[0]), str(bounds[1])]
    # 无任何确定性槽位 → None（回退整句检索）
    has_slot = bool(slots["time"]["bounds"] or slots["place"]["name"]
                    or slots["person"] or slots["event"].get("name"))
    return slots if has_slot else None
