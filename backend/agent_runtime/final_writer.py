"""Phase F F1 — Final Answer Writer（受控事实 → 直接回答）。

tool-loop 的 final 由模型生成；当草稿违反 Answer Policy（retrieval jargon
泄漏 / 以检索过程开头 / 过度 hedge / 该说"不知道"却绕圈）时，用 Final
Response Context 做一次 text-only 重写，禁止编造、禁止改写硬值。

Writer 只做风格与结构修正：事实边界完全来自 TaskState 的确定性字段，
不重新调用昂贵工具，不引入新事实。
"""

from __future__ import annotations

import json
import re

# 与 Answer Style Benchmark 同源的内部词汇清单（检测用，重写时一并禁止）
JARGON_TERMS = [
    "候选照片", "候选", "partial_support", "candidate_only", "full_support",
    "no_match", "匹配程度", "检索结果", "相似候选", "query_satisfaction",
    "条件已确认", "部分条件已确认", "未查看", "相似匹配", "基于关键词", "部分确认",
]

_PREFIX_RE = re.compile(
    r"^(?:[^，。；]{0,20}，)?(?:我|我为您|为您|已经)?(?:已经)?(?:找到|检索到|查询到|搜索到)"
    r"\s*\d+\s*张(?:候选)?照片")
_HEDGE_WHEN_CONFIRMED_RE = re.compile(r"还不能完全确认|无法完全确认|还不能确定|目前还不能|还不能完全确定")
_PROMISE_RE = re.compile(r"我(?:可以|能|会)?(?:继续)?(?:帮你|为您)?(?:再)?(?:核对|确认|查看|找)")
_DENIES_FOUND_RE = re.compile(r"没有找到|没找到|未找到|没有获取到|找不到|不存在相关|没有相关")

# Phase G G7：空壳收尾/套话（自然化时删除，不引入新事实）
_BOILERPLATE_TAIL_RE = re.compile(
    r"[。；]?(?:以上是我目前能确认的部分信息|以上是目前能确认的部分|以上是我能确认的内容|"
    r"以上是能确认的内容|以上是部分确认信息)[。！!]?\s*$")
_TAIL_JUNK_RE = re.compile(
    r"[；。](?:找到\s*\d+\s*张接近的照片|部分信息能对上，还有细节不能完全确认|"
    r"我可以继续帮你核对|目前看起来有\s*\d+\s*张相关照片)[。！!]?\s*$")

_CERTAINTY_LABEL = {
    "full_support": "confirmed",
    "partial_support": "likely",
    "candidate_only": "uncertain",
    "no_match": "none",
}


def _certainty(satisfaction: str | None) -> str:
    return _CERTAINTY_LABEL.get(satisfaction or "", "uncertain")


def build_final_context(message: str, task: dict) -> dict:
    """从 TaskState 构造 Final Response Context（受控事实 + 缺口 + 状态）。"""
    facts: list[dict] = []
    unknowns: list[str] = []
    resolution_state = "none"
    ctx_has_results = False
    total = task.get("result_total")
    satisfaction = task.get("search_satisfaction")
    if total is not None:
        if total > 0:
            facts.append({
                "value": f"找到 {total} 张相关照片。",
                "certainty": _certainty(satisfaction),
                "source": "search",
            })
            ctx_has_results = True
        else:
            unknowns.append("没有找到符合条件的照片。")
    for tr in task.get("tool_results") or []:
        tool = tr.get("tool")
        if tool == "search_memories":
            for item in (tr.get("preview") or []):
                place = (item or {}).get("place") or ""
                handle = (item or {}).get("handle") or ""
                if place:
                    facts.append({
                        "value": f"{handle} 拍摄于 {place}。",
                        "certainty": "confirmed", "source": "search_place",
                    })
        elif tool == "inspect_photo":
            text = (tr.get("inspect_text") or "").strip()
            if text:
                handle = tr.get("inspect_handle") or ""
                facts.append({
                    "value": f"{handle} 的观察：{text[:400]}",
                    "certainty": "confirmed", "source": "inspect",
                })
                resolution_state = "visual_done"
        elif tool == "read_photo_text":
            ocr = (tr.get("ocr_text") or "").strip()
            if ocr:
                facts.append({
                    "value": f"照片里的文字：{ocr[:600]}",
                    "certainty": "confirmed", "source": "ocr",
                })
                resolution_state = "ocr_done"
            else:
                resolution_state = "unresolved"
                unknowns.append("照片里没有识别到可用的文字。")
    # query_memory_facts 确定性事实
    op = task.get("fact_operation")
    value = task.get("fact_value")
    if op in {"count", "media"} and isinstance(value, int):
        facts.append({"value": f"符合条件的结果数量为 {value}。", "certainty": "confirmed", "source": "facts"})
    elif op in {"first", "last", "date"} and value:
        label = {"first": "最早一次", "last": "最近一次", "date": "相关时间"}.get(op, "时间")
        facts.append({"value": f"{label}是 {value}。", "certainty": "confirmed", "source": "facts"})
    elif op == "exists":
        facts.append({"value": "存在相关记录。", "certainty": "confirmed", "source": "facts"}
                     if value is True else {"value": "不存在相关记录。", "certainty": "confirmed", "source": "facts"})
    if op == "meal":
        foods = value or []
        if foods:
            facts.append({
                "value": "明确食物（按事件去重）：" + "、".join(
                    f"{f.get('food')}({f.get('events')}次)" for f in foods[:10]),
                "certainty": "confirmed", "source": "facts",
            })
    rows = task.get("fact_rows") or []
    if rows:
        facts.append({
            "value": "分组：" + "、".join(f"{r.get('group')}({r.get('count')}条)" for r in rows[:6]),
            "certainty": "confirmed", "source": "facts",
        })
    return {
        "user_question": message,
        "facts": facts,
        "unknowns": unknowns,
        "requested_images": bool(task.get("result_preview")),
        "resolution_state": resolution_state,
        "facts_confirmed": any(f.get("certainty") == "confirmed" for f in facts),
        "has_results": ctx_has_results,
    }


def needs_rewrite(answer: str, context: dict) -> bool:
    """草稿是否违反 Answer Policy，需要 writer 重写。"""
    if not answer or not answer.strip():
        return True
    for term in JARGON_TERMS:
        if term in answer:
            return True
    if _PREFIX_RE.search(answer):
        return True
    if context.get("facts_confirmed") and _HEDGE_WHEN_CONFIRMED_RE.search(answer):
        return True
    if context.get("has_results") and _DENIES_FOUND_RE.search(answer):
        return True
    if context.get("resolution_state") == "unresolved" and _PROMISE_RE.search(answer):
        return True
    if _BOILERPLATE_TAIL_RE.search(answer) or _TAIL_JUNK_RE.search(answer):
        return True
    return False


def naturalize_answer(answer: str) -> str:
    """G7：确定性自然化——删除空壳收尾套话，保留全部实质事实。

    只做删减与标点整理，绝不改写数字/价格/人名/地点等硬值，也不新增内容。
    """
    text = (answer or "").strip()
    if not text:
        return text
    # 迭代删除：空壳收尾可能是嵌套的（"…找到 N 张接近的照片。；部分信息能对上。以上是…"）
    for _ in range(3):
        new = _BOILERPLATE_TAIL_RE.sub("", text)
        new = _TAIL_JUNK_RE.sub("", new)
        new = re.sub(r"(?:^|。)(?:以上是我|以上是|以上是我目前)(?:能确认的|目前能确认的)(?:部分|内容)。?", "", new)
        if new == text:
            break
        text = new
    text = text.strip().strip("； ")
    if text and text[-1] not in "。！？!?；":
        text += "。"
    return text


_WRITER_SYSTEM = (
    "你是 Sentrix，一个自然、克制的家庭记忆助手。你的任务：根据「受控事实」直接回答用户问题。\n"
    "规则：\n"
    "1. 先直接回答用户问题本身；\n"
    "2. 必要的不确定性用自然语言（“看起来是…”；“可能是…，但我还不能完全确定。”）；\n"
    "3. 最多一句有帮助的补充。\n"
    "4. 禁止以“我找到 N 张候选照片/检索到…”开头；禁止出现这些内部词汇：候选照片、候选、相似候选、"
    "匹配程度、条件已确认、未查看、partial_support、candidate_only、full_support、no_match、"
    "query_satisfaction、检索结果。\n"
    "5. 数字/价格/电话/年份/日期/人名必须与受控事实完全一致，禁止改写或增减。\n"
    "6. 如果受控事实为空、与问题无关，或照片里没有识别到可用内容，直接说“现有照片里看不出来/不知道”，"
    "不要编造，不要承诺“可以继续核对”。\n"
    "6b. 如果受控事实明确“找到 N 张相关照片”，绝对不要说“没有找到/未找到”，只如实说明能确认到什么程度。\n"
    "7. 不要复述检索或工具过程。\n"
)


def rewrite_final(chat_fn, context: dict, draft: str) -> str | None:
    """一次 text-only 重写；失败返回 None（调用方保留草稿）。"""
    payload = {
        "facts": context.get("facts") or [],
        "unknowns": context.get("unknowns") or [],
    }
    user = (
        "受控事实：\n" + json.dumps(payload["facts"], ensure_ascii=False) +
        "\n\n不确定/缺失：\n" + json.dumps(payload["unknowns"], ensure_ascii=False) +
        "\n\n用户问题：" + str(context.get("user_question") or "") +
        "\n\n（你上一版草稿风格不达标，仅供参考，不要照抄结构：）\n" + str(draft or "")[:800] +
        "\n\n请直接输出最终回答文本。"
    )
    try:
        raw = chat_fn([
            {"role": "system", "content": _WRITER_SYSTEM},
            {"role": "user", "content": user},
        ])
    except Exception:
        return None
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip('"').strip()
    if not text:
        return None
    return text[:1600]
