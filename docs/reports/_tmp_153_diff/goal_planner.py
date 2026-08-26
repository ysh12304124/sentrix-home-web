"""Goal declaration planner for Agent 2.

The model declares evidence needs in a clean, lightweight format; this adapter validates
the declaration, maps simplified needs to strict EvidenceRequirements if needed, and returns
the structured TaskDeclaration.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass

from .planner_contracts import parse_planner_action
from .task_state import TaskDeclaration, EvidenceRequirement


# 极简 Planning 提示词：小于 120 tokens，不包含庞杂工具描述与冗余参数
_DECLARATION_PROMPT = """你正在规划家庭记忆任务。只返回一个精简的 JSON 对象，格式如下：
{{"action":"declare","declaration":{{"goal":"<用户目标>","scope_id":"{scope_id}","requirements":[{{"id":"req_1","evidence_type":"<证据类型>","description":"<描述>"}}]}}}}
证据类型（evidence_type）只能是以下之一：
- memory_asset（查找照片/视频）
- location_metadata（地点/城市/度假村）
- temporal_metadata（时间/日期）
- confirmed_identity（人物/家庭成员）
- visual_observation（视觉细节/颜色/物品/人数）
- visible_text（文字/招牌/价格/数字）
- structured_fact（统计/数量/最早/最近）
- user_statement（历史对话）

不要调用工具，不要输出 SQL，不要直接回答用户。"""

_TYPE_MAP = {
    "photo": "memory_asset",
    "image": "memory_asset",
    "video": "memory_asset",
    "asset": "memory_asset",
    "location": "location_metadata",
    "place": "location_metadata",
    "time": "temporal_metadata",
    "date": "temporal_metadata",
    "person": "confirmed_identity",
    "identity": "confirmed_identity",
    "visual": "visual_observation",
    "detail": "visual_observation",
    "text": "visible_text",
    "ocr": "visible_text",
    "price": "visible_text",
    "fact": "structured_fact",
    "count": "structured_fact",
}


@dataclass(frozen=True)
class PlannerDeclarationResult:
    declaration: TaskDeclaration | None = None
    fallback_reason: str = ""
    raw: str = ""
    prompt: list | None = None

    @property
    def ok(self) -> bool:
        return self.declaration is not None


class GoalPlanner:
    def __init__(self, *, chat_fn):
        self.chat_fn = chat_fn

    def declare(self, message: str, *, scope_id: str, history: str = "",
                include_debug: bool = False, step_id: str = "planner_step_0") -> PlannerDeclarationResult:
        messages = [
            {"role": "system", "content": _DECLARATION_PROMPT.format(scope_id=scope_id)},
            {"role": "user", "content": message},
        ]
        if history:
            messages.insert(1, {"role": "system", "content": "历史对话背景：\n" + history})
        prompt_copy = copy.deepcopy(messages) if include_debug else None
        try:
            sig = inspect.signature(self.chat_fn)
            if "call_type" in sig.parameters:
                raw = self.chat_fn(messages, call_type="planner", step_id=step_id) or ""
            else:
                raw = self.chat_fn(messages) or ""
        except Exception:
            return PlannerDeclarationResult(fallback_reason="planner_call_error", prompt=prompt_copy)
        try:
            payload = self._parse_json(raw)
            # 兼容小模型可能返回的扁平或近义词结构
            payload = self._normalize_payload(payload, scope_id=scope_id, default_goal=message)
            action = parse_planner_action(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return PlannerDeclarationResult(fallback_reason="invalid_planner_action", raw=raw, prompt=prompt_copy)
        if action.kind != "declare" or action.declaration is None:
            return PlannerDeclarationResult(fallback_reason="invalid_planner_action", raw=raw, prompt=prompt_copy)
        if action.declaration.scope_id != scope_id:
            return PlannerDeclarationResult(fallback_reason="scope_mismatch", raw=raw, prompt=prompt_copy)
        return PlannerDeclarationResult(declaration=action.declaration, raw=raw, prompt=prompt_copy)

    @classmethod
    def _normalize_payload(cls, payload: dict, *, scope_id: str, default_goal: str) -> dict:
        """把小模型可能简化的 declaration 结构归一为严格 TaskDeclaration schema。"""
        if not isinstance(payload, dict):
            return payload
        decl = payload.get("declaration")
        if not isinstance(decl, dict):
            if payload.get("action") == "declare" and ("requirements" in payload or "needs" in payload):
                decl = payload
                payload = {"action": "declare", "declaration": decl}
            else:
                return payload
        
        if not decl.get("scope_id"):
            decl["scope_id"] = scope_id
        if not decl.get("goal"):
            decl["goal"] = default_goal
            
        reqs = decl.get("requirements") or decl.get("needs") or []
        normalized_reqs = []
        for idx, item in enumerate(reqs):
            if isinstance(item, str):
                etype = _TYPE_MAP.get(item.lower(), "memory_asset")
                normalized_reqs.append({
                    "id": f"req_{idx+1}",
                    "evidence_type": etype,
                    "description": item,
                })
            elif isinstance(item, dict):
                etype = str(item.get("evidence_type") or item.get("type") or "memory_asset").strip()
                etype = _TYPE_MAP.get(etype.lower(), etype)
                normalized_reqs.append({
                    "id": str(item.get("id") or f"req_{idx+1}"),
                    "evidence_type": etype,
                    "description": str(item.get("description") or item.get("desc") or ""),
                })
        if not normalized_reqs:
            normalized_reqs.append({
                "id": "req_1",
                "evidence_type": "memory_asset",
                "description": "查找相关记忆照片",
            })
        decl["requirements"] = normalized_reqs
        return payload

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("planner did not return JSON")
        payload = json.loads(text[start:end + 1])
        if not isinstance(payload, dict):
            raise ValueError("planner action must be an object")
        return payload
