"""Phase G G3 — Completion State / Gate（最小动态 requirements，不重建 Planner）。

需求由 Tool 返回的确定性信号动态形成：
  retrieve_evidence / resolve_visual / resolve_ocr / deliver_media / answer

CompletionGate 只做两件事：
  1) 模型想提前 final 时，给出“任务还没完成”的自然提示（模型仍自己决定下一步 Tool）；
  2) 把每个需求的满足状态暴露给 runtime / telemetry / dashboard。

代码只告诉模型“你的任务还没完成”，不替模型规划 workflow。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

RETRIEVE_EVIDENCE = "retrieve_evidence"
RESOLVE_VISUAL = "resolve_visual"
RESOLVE_OCR = "resolve_ocr"
DELIVER_MEDIA = "deliver_media"
ANSWER = "answer"

_RETRIEVAL_TOOLS = {
    "search_memories", "query_memory_facts", "search_conversation_history",
    "get_core_memory", "get_person_memory",
}

# 纯聊天/通用知识：不需要检索家庭记忆
_CHAT_ONLY_RE = re.compile(
    r"你好|谢谢|在吗|再见|哈哈|好的|嗯|哦|你是谁|你叫什么|你会什么|帮我写|"
    r"写一|写个|改一|翻译|解释一下什么是|什么是|怎么用|步骤|教程", re.I)
# 涉及家庭记忆/照片证据的问题信号
_EVIDENCE_INTENT_RE = re.compile(
    r"照片|图片|记录|回忆|拍|合影|去了|在哪里|哪里|什么|谁|多少|几|"
    r"时间|日期|年份|月份|几号|活动|旅游|旅行|度假|菜单|价格|招牌|店名|"
    r"电话|穿着|衣服|颜色|天气|沙子|沙雕|人物|家人|地点|城市|哪个|哪张")
# 视觉细节意图
_VISUAL_RE = re.compile(
    r"桌上|桌面|颜色|几个|多少人|招牌|文字|天气|外套|衣服|猫|雪|小孩|穿着|穿|"
    r"在做什么|有没有|是什么|放着|写了|内容|细节|长什么样|什么颜色|什么动物")
# OCR 文字读取意图
_OCR_RE = re.compile(
    r"菜单|价格|多少钱|售价|招牌|店名|电话|写了什么|什么字|文字|创始于|"
    r"价位|几块钱|面单|多少钱一份|报警电话|多少钱|价格是|上面写了")
# 明确要求查看/交付照片
_IMAGE_RE = re.compile(
    r"给我看看|给我看|发我|发给我|发来|原图|都给我|全部给我|"
    r"展示|显示(?:一下|给我)?|让我看看|看看(?:这些|照片|图)?|"
    r"打开(?:照片|图片)|看图|给我图|把.{0,6}(?:照片|图片|图)", re.I)


@dataclass
class Requirement:
    code: str
    label: str
    tool: str | None
    reason: str
    satisfied: bool = False
    pending: bool = True

    def as_dict(self) -> dict:
        return {
            "code": self.code, "label": self.label, "tool": self.tool,
            "reason": self.reason, "satisfied": self.satisfied,
            "pending": self.pending and not self.satisfied,
        }


# 需求优先级：OCR > 视觉复核 > 交付照片 > 检索证据
_PRIORITY = {RESOLVE_OCR: 0, RESOLVE_VISUAL: 1, DELIVER_MEDIA: 2, RETRIEVE_EVIDENCE: 3}


class CompletionState:
    """随 Tool-Loop 动态形成的完成需求集合（幂等 update）。"""

    def __init__(self, message: str = ""):
        self.message = message or ""
        self.requirements: dict[str, Requirement] = {}

    # ---- 需求增补 ----
    def _add(self, code: str, label: str, tool: str | None, reason: str, *, satisfied: bool):
        if code in self.requirements:
            self.requirements[code].satisfied = satisfied
            return
        self.requirements[code] = Requirement(
            code=code, label=label, tool=tool, reason=reason, satisfied=satisfied)

    def update(self, task_state: dict | None) -> None:
        """根据当前 TaskState 动态增补/刷新需求（在每次工具结果后调用，幂等）。"""
        task_state = task_state or {}
        tool_results = task_state.get("tool_results") or []
        tools_called = {str(tr.get("tool") or "") for tr in tool_results}

        # 1) retrieve_evidence：涉及家庭记忆的问题必须先完成一次检索
        if self._wants_evidence(self.message):
            self._add(RETRIEVE_EVIDENCE, "检索家庭记忆", None,
                      "回答这个问题需要先查看家庭记忆/照片记录，请先调用检索工具。",
                      satisfied=bool(tools_called & _RETRIEVAL_TOOLS))

        # 2) resolve_visual：检索返回候选 + 视觉意图 + 未 inspect
        has_preview = any(
            tr.get("tool") == "search_memories" and (tr.get("preview") or [])
            for tr in tool_results)
        if has_preview and _VISUAL_RE.search(self.message):
            self._add(RESOLVE_VISUAL, "复核照片细节", "inspect_photo",
                      "需要调用 inspect_photo 复核预览照片后才能回答视觉细节。",
                      satisfied="inspect_photo" in tools_called)

        # 3) resolve_ocr：OCR 意图 或 search 明确推荐 read_photo_text，且尚未调用
        ocr_intent = bool(_OCR_RE.search(self.message))
        recommended_ocr = self._search_recommends(tool_results, "read_photo_text")
        if ocr_intent or recommended_ocr:
            self._add(RESOLVE_OCR, "读取照片文字", "read_photo_text",
                      "需要调用 read_photo_text 读取照片中的文字后，才能回答文字/数字类问题。",
                      satisfied="read_photo_text" in tools_called)

        # 4) deliver_media：用户明确要求查看照片，且尚未交付原图
        if _IMAGE_RE.search(self.message):
            delivered = bool(tools_called & {"get_original_photos", "get_result_page"})
            self._add(DELIVER_MEDIA, "交付照片", "get_original_photos",
                      "用户要求查看照片，需要交付原图/可查看的照片。",
                      satisfied=delivered)

    @staticmethod
    def _wants_evidence(message: str) -> bool:
        if not message:
            return False
        if _CHAT_ONLY_RE.search(message):
            return False
        return bool(_EVIDENCE_INTENT_RE.search(message))

    @staticmethod
    def _search_recommends(tool_results: list, tool: str) -> bool:
        for tr in tool_results or []:
            if tr.get("tool") != "search_memories":
                continue
            rec = (tr.get("observation") or {}).get("recommended_resolution") or {}
            if rec.get("needed") and rec.get("tool") == tool:
                return True
        return False

    # ---- 查询 ----
    def blocking(self) -> list[Requirement]:
        """未完成且仍 pending 的需求，按优先级排序（OCR > 视觉 > 交付 > 检索）。"""
        blocked = [r for r in self.requirements.values() if not r.satisfied and r.pending]
        return sorted(blocked, key=lambda r: _PRIORITY.get(r.code, 9))

    def is_blocked(self) -> bool:
        return bool(self.blocking())

    def as_dict(self) -> dict:
        return {
            "message": self.message,
            "requirements": [r.as_dict() for r in sorted(
                self.requirements.values(), key=lambda r: _PRIORITY.get(r.code, 9))],
            "blocking": [r.code for r in self.blocking()],
        }
