"""AgentRuntime 薄循环（v2 §4.1/§32.3）。

model -> tool -> observation -> model -> ... -> final
模型选择 Tool；代码通过 ToolPolicy 提供边界；BudgetManager 限制循环。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .budget_manager import BudgetState
from .final_guard import FinalGuard
from .profile import get_profile
from .result_set import TaskState
from .tool_policy import ToolPolicy
from .tool_registry import get_tool

SYSTEM_TEMPLATE = """你是 Sentrix 家庭记忆助手。你通过与工具协作完成用户请求。

可用工具（JSON 动作）：
{tools}

规则：
- 需要家庭记忆事实时调用工具；不需要时直接 final。
- 每次只输出一个 JSON 对象（不要 markdown、不要解释、不要多余文字）：
  {{"action":"tool_call","tool":"...","arguments":{{...}},"public_status":"..."}}
  或 {{"action":"final","answer":"..."}}
- 只使用工具返回的事实回答，不编造数字或细节；工具没有返回的内容不要编造。
- rows/value 是工具的真实结果：只能报告其中实际出现的月份、地点、数字；
  不要补充 rows 中没有的项目，也不要自行概括出 rows 不支持的维度。
- public_status 是给用户看的简短进度说明。
"""


@dataclass
class ToolResult:
    tool: str
    status: str
    observation: dict | None = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class RuntimeTurn:
    profile: str
    budget: BudgetState
    steps: list = field(default_factory=list)
    public_progress: list = field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"   # complete | partial | timeout | error
    reason: str = ""


class AgentRuntime:
    """Thin tool-loop runtime. 模型调用通过传入的 chat_fn 注入。"""

    def __init__(self, *, chat_fn, profile_name: str | None = None,
                 scope_id="home-default", viewer_id="owner"):
        self.chat_fn = chat_fn
        self.profile = get_profile(profile_name)
        self.scope_id = scope_id
        self.viewer_id = viewer_id

    def _tool_descriptions(self) -> str:
        lines = []
        from .tool_registry import list_tools
        for spec in list_tools():
            if spec.readiness == "blocked":
                continue
            lines.append(f"- {spec.name}: {spec.description} 输入schema={json.dumps(spec.input_schema, ensure_ascii=False)}")
        return "\n".join(lines) or "(无工具)"

    def _parse_action(self, text: str) -> dict | None:
        """把模型输出解析为 action JSON；对常见畸形输出做有限修复。"""
        text = (text or "").strip()
        if not text:
            return None
        for candidate in self._action_candidates(text):
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("action") in ("tool_call", "final"):
                return parsed
        return None

    @staticmethod
    def _action_candidates(text: str) -> list[str]:
        """按可靠性从高到低生成可尝试解析的候选文本。"""
        import re as _re
        text = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.I)
        text = _re.sub(r"\s*```\s*$", "", text)
        cands = [text]
        # 提取最外层平衡花括号；未闭合时保留到末尾
        start = text.find("{")
        if start >= 0:
            depth = 0
            end = None
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            cands.append(text[start:end] if end else text[start:])
        base = list(cands)
        # 截断补全：补闭合括号
        for s in base:
            opens, closes = s.count("{"), s.count("}")
            if opens > closes:
                cands.append(s + "}" * (opens - closes))
        # 截断补全：字符串未闭合时补引号与括号
        for s in base:
            if s.replace("\\\"", "").count('"') % 2 == 1:
                opens, closes = s.count("{"), s.count("}")
                cands.append(s + '"' + "}" * max(0, opens - closes))
        # 畸形 :{"} -> :{}（如 "filters":{"}）
        for s in base:
            cands.append(_re.sub(r":\s*\{\s*\"\s*\}", ":{}", s))
        # 对象键缺值移除（如 "person":} -> }）
        for s in base:
            cands.append(_re.sub(r",?\s*\"[^\"]*\":\s*([}\]])", r"\1", s))
        # 尾随逗号
        for s in base:
            cands.append(_re.sub(r",\s*([}\]])", r"\1", s))
        seen, out = set(), []
        for s in cands:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def run(self, message: str, *, history: str = "", task_state: dict | None = None) -> RuntimeTurn:
        turn = RuntimeTurn(profile=self.profile.name, budget=BudgetState(
            max_model_steps=self.profile.max_model_steps,
            max_tool_calls=self.profile.max_tool_calls,
            max_inspections=self.profile.max_inspections,
            wall_time_s=self.profile.wall_time_s,
            final_reserve_s=self.profile.final_reserve_s,
        ))
        turn.budget.start()
        policy = ToolPolicy(scope_id=self.scope_id, viewer_id=self.viewer_id, budget=turn.budget)
        task = TaskState(user_goal=message)
        guard = FinalGuard(scope_id=self.scope_id, viewer_id=self.viewer_id)
        system = SYSTEM_TEMPLATE.format(tools=self._tool_descriptions())
        messages = [{"role": "system", "content": system}]
        if history:
            messages.append({"role": "system", "content": f"最近对话：\n{history}"})
        messages.append({"role": "user", "content": message})

        parse_retries = 0
        max_parse_retries = 1
        while True:
            if not turn.budget.can_model_step():
                turn.status = "partial" if turn.steps else "timeout"
                turn.reason = "model step budget exhausted"
                break
            turn.budget.record_model_step()
            try:
                raw = self.chat_fn(messages)
            except Exception as exc:
                turn.status = "error"
                turn.reason = f"model_call_error: {exc}"
                break
            turn.steps.append({"type": "model", "raw": (raw or "")[:500]})
            action = self._parse_action(raw)
            if action is None:
                if parse_retries < max_parse_retries and turn.budget.can_model_step():
                    parse_retries += 1
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": (
                        "你的上一条输出不是合法的 JSON 对象，无法解析。"
                        "请严格只输出一个 JSON 对象（action 只能是 tool_call 或 final），"
                        "不要 markdown、不要多余文字、不要省略结尾的引号或括号。"
                    )})
                    continue
                turn.status = "error"
                turn.reason = "unparseable_action"
                break
            if action.get("action") == "final":
                turn.final_answer = str(action.get("answer") or "")
                problems = guard.check(
                    turn.final_answer,
                    task_state={
                        "result_mode": task.result_mode,
                        "has_more": task.has_more,
                        "delivery_state": task.delivery_state,
                        "fulfillment": task.fulfillment,
                        "fact_total": task.fact_total,
                        "fact_value": task.fact_value,
                        "fact_operation": task.fact_operation,
                        "fact_rows": task.fact_rows,
                        "fact_group_by": task.fact_group_by,
                        "last_tool": task.last_tool,
                    },
                    delivered_count=task.delivered_count,
                )
                if problems:
                    turn.status = "blocked_by_guard"
                    turn.reason = ";".join(problems)
                    break
                turn.status = "complete"
                break
            if action.get("action") != "tool_call":
                turn.status = "error"
                turn.reason = f"unknown_action:{action.get('action')}"
                break
            tool_name = action.get("tool") or ""
            arguments = action.get("arguments") or {}
            public_status = action.get("public_status") or "正在处理。"
            spec = get_tool(tool_name)
            if spec is None:
                turn.steps.append({"type": "tool", "tool": tool_name, "status": "error",
                                   "reason": "unknown_tool"})
                turn.reason = "unknown_tool:" + tool_name
                break
            t0 = time.monotonic()
            decision = policy.execute(spec, arguments, context={
                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                "task_state": task.as_dict(), "history": history,
            })
            latency = round(time.monotonic() - t0, 2)
            result = ToolResult(tool=tool_name,
                                status="ok" if decision.allowed else "denied",
                                observation=decision.observation,
                                error=decision.error, latency_s=latency)
            turn.steps.append({
                "type": "tool", "tool": tool_name, "arguments": arguments,
                "status": result.status, "observation": result.observation,
                "error": result.error, "latency_s": latency,
            })
            turn.public_progress.append({"text": public_status, "status": result.status})
            if not decision.allowed:
                turn.reason = f"tool_denied:{tool_name}:{decision.reason}"
                break
            task.update_from_tool(tool_name, arguments, result.observation or {})
            # Observation 进入下一步模型上下文
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "tool_call_id": tool_name, "content": json.dumps(
                result.observation or {}, ensure_ascii=False)})

        return turn
