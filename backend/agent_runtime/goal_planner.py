"""Shadow-only declaration planner for Agent 2.

The model declares evidence needs; this adapter validates the declaration and
returns an observable fallback instead of granting it authority to execute.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass

from .planner_contracts import parse_planner_action
from .task_state import TaskDeclaration


_DECLARATION_PROMPT = """你正在规划一项家庭记忆任务。只返回一个 JSON 对象，格式如下：
{{"action":"declare","declaration":{{"goal":"<用户目标>","scope_id":"{scope_id}","requirements":[{{"id":"req_1","evidence_type":"<证据类型>","description":"<证据描述>"}}]}}}}
仅声明回答用户请求所必需的最小证据需求。
可选的证据类型（evidence_type）包括：
- memory_asset：相册照片/视频等媒体实体资产
- visible_text：照片/招牌/菜单/账单中直接可见的文字或数字
- visual_observation：照片中的视觉细节（如衣服颜色、物品、人物动作等）
- temporal_metadata：拍摄时间或日期元数据
- location_metadata：拍摄地点、城市或地标元数据
- confirmed_identity：确认人物身份或家庭成员关系
- structured_fact：数据库结构化统计事实（如计数、分组、最早/最近等）
- memory_reference：记忆引用或历史记录
- user_statement：用户显式声明的信息
- transcript：音频或视频转录文本

不要调用工具，不要编造不存在的资产，不要输出 SQL，不要直接回答用户。"""


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
            messages.insert(1, {"role": "system", "content": "Conversation context:\n" + history})
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
            action = parse_planner_action(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return PlannerDeclarationResult(fallback_reason="invalid_planner_action", raw=raw, prompt=prompt_copy)
        if action.kind != "declare" or action.declaration is None:
            return PlannerDeclarationResult(fallback_reason="invalid_planner_action", raw=raw, prompt=prompt_copy)
        if action.declaration.scope_id != scope_id:
            return PlannerDeclarationResult(fallback_reason="scope_mismatch", raw=raw, prompt=prompt_copy)
        return PlannerDeclarationResult(declaration=action.declaration, raw=raw, prompt=prompt_copy)

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
