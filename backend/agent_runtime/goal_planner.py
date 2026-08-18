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


_DECLARATION_PROMPT = """You are planning a family-memory task. Return exactly one JSON object:
{{"action":"declare","declaration":{{"goal":"...","scope_id":"{scope_id}","requirements":[{{"id":"...","evidence_type":"...","description":"..."}}]}}}}
Declare only evidence needed to answer the request. Evidence types are: structured_fact, memory_asset, memory_reference, visual_observation, visible_text, temporal_metadata, location_metadata, confirmed_identity, user_statement, transcript. Do not call tools, invent assets, use SQL, or answer the user."""


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
