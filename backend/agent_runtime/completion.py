"""Phase G G3 — Completion State / Gate（最小动态 requirements，不重建 Planner）。

需求由 Tool 返回的确定性信号动态形成：
  retrieve_evidence / resolve_visual / resolve_ocr / deliver_media / answer

CompletionGate 只做两件事：
  1) 模型想提前 final 时，给出“任务还没完成”的自然提示（模型仍自己决定下一步 Tool）；
  2) 把每个需求的满足状态暴露给 runtime / telemetry / dashboard。

代码只告诉模型“你的任务还没完成”，不替模型规划 workflow。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .intent import (chat_only, evidence_intent, image_delivery_intent,
                     ocr_intent, visual_intent)

RETRIEVE_EVIDENCE = "retrieve_evidence"
RESOLVE_VISUAL = "resolve_visual"
RESOLVE_OCR = "resolve_ocr"
DELIVER_MEDIA = "deliver_media"
ANSWER = "answer"

_RETRIEVAL_TOOLS = {
    "search_memories", "query_memory_facts", "search_conversation_history",
    "get_core_memory", "get_person_memory",
}

# Agent 2 planner requirements are authoritative when they contain a type that
# this legacy completion gate can map.  Unknown types deliberately fall back
# to the existing intent heuristics so a planner extension cannot disable the
# legacy safety net.
_AGENT2_ACTIVE_STATUSES = frozenset({"open", "running", "partially_supported"})
_AGENT2_EVIDENCE_CODES = {
    RETRIEVE_EVIDENCE: frozenset({
        "memory_asset", "structured_fact",
        "location_metadata", "temporal_metadata", "confirmed_identity",
    }),
    RESOLVE_VISUAL: frozenset({"visual_observation"}),
    RESOLVE_OCR: frozenset({"visible_text"}),
}

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

    def update(self, task_state: dict | None, agent2_task_state=None) -> None:
        """根据当前 TaskState 动态增补/刷新需求（在每次工具结果后调用，幂等）。"""
        task_state = task_state or {}
        tool_results = task_state.get("tool_results") or []
        tools_called = {str(tr.get("tool") or "") for tr in tool_results}

        # 1) retrieve_evidence：涉及家庭记忆的问题必须先完成一次检索
        model_retrieve = self._agent2_pending(agent2_task_state, RETRIEVE_EVIDENCE)
        wants_retrieve = (model_retrieve if model_retrieve is not None
                          else self._wants_evidence(self.message))
        if model_retrieve is False and RETRIEVE_EVIDENCE in self.requirements:
            self.requirements[RETRIEVE_EVIDENCE].satisfied = True
        if wants_retrieve:
            self._add(RETRIEVE_EVIDENCE, "检索家庭记忆", None,
                      "回答这个问题需要先查看家庭记忆/照片记录，请先调用检索工具。",
                      satisfied=(not model_retrieve) if model_retrieve is not None
                      else bool(tools_called & _RETRIEVAL_TOOLS))

        # 2) resolve_visual：检索返回候选 + 视觉意图 + 未 inspect
        has_preview = any(
            tr.get("tool") == "search_memories" and (tr.get("preview") or [])
            for tr in tool_results)
        model_visual = self._agent2_pending(agent2_task_state, RESOLVE_VISUAL)
        wants_visual = (model_visual if model_visual is not None
                        else visual_intent(self.message))
        if model_visual is False and RESOLVE_VISUAL in self.requirements:
            self.requirements[RESOLVE_VISUAL].satisfied = True
        if has_preview and wants_visual:
            self._add(RESOLVE_VISUAL, "复核照片细节", "inspect_photo",
                      "需要调用 inspect_photo 复核预览照片后才能回答视觉细节。",
                      satisfied=(not model_visual) if model_visual is not None
                      else self._tool_succeeded(tool_results, "inspect_photo"))

        # 3) resolve_ocr：OCR 意图 或 search 明确推荐 read_photo_text，且尚未调用
        model_ocr = self._agent2_pending(agent2_task_state, RESOLVE_OCR)
        wants_ocr = (model_ocr if model_ocr is not None else ocr_intent(self.message))
        if model_ocr is False and RESOLVE_OCR in self.requirements:
            self.requirements[RESOLVE_OCR].satisfied = True
        recommended_ocr = self._search_recommends(tool_results, "read_photo_text")
        if wants_ocr or recommended_ocr:
            self._add(RESOLVE_OCR, "读取照片文字", "read_photo_text",
                      "需要调用 read_photo_text 读取照片中的文字后，才能回答文字/数字类问题。",
                      satisfied=(not model_ocr) if model_ocr is not None
                      else self._tool_succeeded(tool_results, "read_photo_text"))

        # 4) deliver_media：用户明确要求查看照片，且尚未交付原图
        if image_delivery_intent(self.message):
            delivered = bool(tools_called & {"get_original_photos", "get_result_page"})
            self._add(DELIVER_MEDIA, "交付照片", "get_original_photos",
                      "用户要求查看照片，需要交付原图/可查看的照片。",
                      satisfied=delivered)

    @staticmethod
    def _wants_evidence(message: str) -> bool:
        if not message:
            return False
        if chat_only(message):
            return False
        return evidence_intent(message)

    @staticmethod
    def _search_recommends(tool_results: list, tool: str) -> bool:
        for tr in tool_results or []:
            if tr.get("tool") != "search_memories":
                continue
            # New result contracts expose this at the tool-result top level;
            # retain the nested form for older traces/replays.
            rec = (tr.get("recommended_resolution") or
                   (tr.get("observation") or {}).get("recommended_resolution") or {})
            if rec.get("needed") and rec.get("tool") == tool:
                return True
        return False

    @staticmethod
    def _tool_succeeded(tool_results: list, tool: str) -> bool:
        """Return true only when the tool produced usable evidence.

        A tool invocation is not evidence by itself: OCR can return a natural
        ``partial`` result and visual inspection can return a failure summary.
        ``record_tool_result`` keeps the original observation fields so this
        check remains valid for both flattened task state and replay traces.
        """
        failure_statuses = {"partial", "failed", "error", "unavailable"}

        def has_value(value) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, dict, set)):
                return bool(value)
            return True

        for result in tool_results or []:
            if result.get("tool") != tool:
                continue
            observation = result.get("observation")
            if not isinstance(observation, dict):
                observation = {}
            status = str(result.get("status") or observation.get("status") or "").lower()
            reason = str(result.get("reason") or observation.get("reason") or "").lower()
            if status in failure_statuses or reason in {"ocr_failed", "model_unavailable"}:
                continue
            blocked = result.get("blocked") or observation.get("blocked")
            if blocked:
                continue
            if tool == "read_photo_text":
                text = (result.get("ocr_text") or result.get("full_text") or
                        observation.get("full_text") or "")
                exact = (result.get("exact_values") or
                         observation.get("exact_values") or [])
                regions = (result.get("text_regions") or
                           observation.get("text_regions") or [])
                if has_value(text) or has_value(exact) or has_value(regions):
                    return True
                continue
            # Prefer the actual model observation.  ``inspect_text`` is the
            # flattened compatibility field used by existing replay traces.
            text = (result.get("inspect_observation") or
                    result.get("inspect_text") or
                    observation.get("observation") or "")
            if (has_value(text) and str(result.get("certainty") or
                                       observation.get("certainty") or "supported").lower()
                    != "uncertain"):
                return True
        return False

    @staticmethod
    def _agent2_pending(agent2_task_state, code: str) -> bool | None:
        """Return model-derived pending state, or ``None`` for regex fallback.

        A non-empty mapped planner requirement suppresses the corresponding
        regex signal.  Terminal planner states are therefore treated as no
        longer pending, while unknown/unmapped evidence types preserve legacy
        behavior.
        """
        requirements = getattr(agent2_task_state, "requirements", None)
        if not requirements:
            return None
        evidence_types = _AGENT2_EVIDENCE_CODES.get(code, frozenset())
        relevant = [state for state in requirements.values()
                    if getattr(getattr(state, "requirement", None), "evidence_type", None)
                    in evidence_types]
        if not relevant:
            return None
        return any(getattr(state, "status", "open") in _AGENT2_ACTIVE_STATUSES
                   for state in relevant)

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
