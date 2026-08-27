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
from .evidence_contract import planner_evidence_types


# 极简 Planning 提示词：小于 120 tokens，不包含庞杂工具描述与冗余参数
_DECLARATION_PROMPT = """你正在规划家庭记忆任务。只返回一个精简的 JSON 对象，格式如下：
{{"action":"declare","declaration":{{"goal":"<用户目标>","scope_id":"{scope_id}","requirements":[{{"id":"req_1","evidence_type":"<证据类型>","description":"<描述>"}}]}}}}
证据类型（evidence_type）只能来自系统注册表，不得创造新类型。
{evidence_types}

只声明回答问题所必需的最小证据集合，不要把检索步骤本身当成答案证据重复声明。规则：
- 数量、是否存在、分组等结构化问题声明 structured_fact；拍摄时间/日期/年份声明 temporal_metadata；只有用户明确要求找出照片时才增加 memory_asset。
- 用户没有明确要求“历史对话/之前说过什么”时，不要声明 user_statement。
- 地点问题声明 location_metadata；照片内容/颜色/动作声明 visual_observation；照片文字/数字声明 visible_text。
- 如果问题问视频/事件“做了什么、展示了什么、发生了什么、先后做了什么”，声明 structured_fact，优先读取事件摘要；只有问题明确问单张画面中可见的颜色、物体细节或人数时才声明 visual_observation。
- 身份问题只有在需要确认照片中的人名时才声明 photo_identity；不要用 visual_observation 代替身份。
- 同一种 evidence_type 只声明一次；不要为了同一个答案同时声明多个等价需求。
每个 requirement 都必须能由注册表中的工具直接或通过 prerequisite 获得。

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
        evidence_lines = "\n".join(f"- {item}" for item in planner_evidence_types())
        prompt = _DECLARATION_PROMPT.format(
            scope_id=scope_id,
            evidence_types=evidence_lines,
        )
        messages = [
            {"role": "system", "content": prompt},
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
        if action.kind not in {"declare", "revise"} or action.declaration is None:
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
        seen_types = set()
        for idx, item in enumerate(reqs):
            if isinstance(item, str):
                etype = _TYPE_MAP.get(item.lower(), "memory_asset")
                if etype in seen_types:
                    continue
                seen_types.add(etype)
                normalized_reqs.append({
                    "id": f"req_{idx+1}",
                    "evidence_type": etype,
                    "description": item,
                    "required": True,
                })
            elif isinstance(item, dict):
                etype = str(item.get("evidence_type") or item.get("type") or "memory_asset").strip()
                etype = _TYPE_MAP.get(etype.lower(), etype)
                if etype in seen_types:
                    continue
                seen_types.add(etype)
                normalized_reqs.append({
                    "id": str(item.get("id") or f"req_{idx+1}"),
                    "evidence_type": etype,
                    "description": str(item.get("description") or item.get("desc") or ""),
                    "required": bool(item.get("required", True)),
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
