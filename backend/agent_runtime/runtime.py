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
from .emergency import render_emergency_summary
from .final_guard import FinalGuard
from .judge import judge_faithfulness
from .time_context import current_time_line
from .profile import get_profile
from .result_set import TaskState
from .tool_policy import ToolPolicy
from .tool_registry import get_tool

SYSTEM_TEMPLATE = """你是 Sentrix 家庭记忆助手。你通过与工具协作完成用户请求。

可用工具（JSON 动作）：
{tools}

{current_time}

规则：
- 需要家庭记忆事实时调用工具；不需要时直接 final。
- 每次只输出一个 JSON 对象，直接输出，不要用 markdown 代码块（不要 ```）、不要解释、不要多余文字：
  {{"action":"tool_call","tool":"...","arguments":{{...}},"public_status":"..."}}
  或 {{"action":"final","answer":"...","evidence_refs":["tool_call_1", ...]}}
- 不要重复调用相同的工具和参数：同一轮里相同 tool+arguments 只允许一次，重复会被拒绝。
- inspect_photo 的 asset_handle 必须原样使用 search_memories 返回 preview 里的 handle（如 photo_1、photo_2）。
  不要编造 preview 之外的 handle。
- 当用户询问照片里的视觉细节（桌上物品、衣服颜色、人数、文字/招牌、天气、穿什么、有没有某物）时，
  如果 search_memories 返回了 preview 候选（有 photo_1 等 handle），你必须调用 inspect_photo 复核 preview 里的照片，
  不能只 search 后就回答“无法确认”，也不要反问用户上传/选择照片。
- 只有当 search_memories 返回 total=0 或 preview 为空时，才不能 inspect_photo；
  此时绝对不要调用 inspect_photo、不要编造 handle、不要声称找到候选照片，直接如实说没有找到符合条件的照片。
- final 时必须用 evidence_refs 列出你实际引用的工具调用编号（本轮工具调用会按顺序编号 tool_call_1、tool_call_2 …；纯聊天不引用）。
- 只使用工具返回的事实回答，不编造数字或细节；工具没有返回的内容不要编造。
- rows/value 是工具的真实结果：只能报告其中实际出现的月份、地点、数字；
  不要补充 rows 中没有的项目，也不要自行概括出 rows 不支持的维度。
- search_memories 的 preview 只显示前几张；用户要求更多/下一页/还有吗 时，用 get_result_page（result_set_id 用 search_memories 返回的，page 从 1 开始）。
- 时间、数量、首末存在性、日期、分组等确定性事实一律用 query_memory_facts，并把用户问题里的时间写进 filters.time（如 '2023年'、'2025-05'）。用户问任何年份/月份都必须如实填进 filters.time，不要省略；不要用 search_memories 代替，也不要用模型估算。
- 按月份/地点统计分布用 query_memory_facts 的 operation=group，并填 group_by（month 或 place）。
- operation=group 且 group_by=place 时，工具会返回 known_location_assets/unknown_location_assets 覆盖信息：
  只要 unknown_location_assets>0，回答必须如实说明还有多少张照片没有可靠地点信息，不能把地点说成完整清单。
- operation=meal 回答'吃过什么/吃饭/火锅'类问题：工具会返回 explicit_foods（明确食物，按事件去重）、
  meal_scene_events（只能确认在吃饭）、possible_events；回答时优先列 explicit_foods，
  有 meal_scene_events 时要说明其中一部分只能确认在用餐、不能确认具体菜品。
- search_memories 返回的 query_satisfaction 决定怎么说：
  full_support=可以确认；partial_support=部分条件确认，必须说出哪些还没确认；
  candidate_only=只是相似候选，**不能说"找到了/确认是"**，要说"找到几张接近的候选，还不能完全确认"；
  no_match=没有候选，**不能说找到**。
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
    task_state: dict = field(default_factory=dict)


def _trusted_facts(task_state: dict) -> list[str]:
    """从 TaskState 提取用户可见、可用于恢复的确定性事实（C2 recovery 用）。"""
    facts: list[str] = []
    op = task_state.get("fact_operation")
    value = task_state.get("fact_value")
    if op in {"count", "media"} and isinstance(value, int):
        facts.append(f"工具确认符合条件的结果数量为 {value}。")
    elif op in {"first", "last", "date"} and value:
        label = {"first": "最早一次", "last": "最近一次", "date": "相关时间"}.get(op, "时间")
        facts.append(f"工具确认{label}是 {value}。")
    elif op == "exists":
        facts.append("工具确认存在相关记录。" if value is True else "工具确认不存在相关记录。")
    rows = task_state.get("fact_rows") or []
    if rows:
        sample = rows[:6]
        facts.append("工具返回的分组为：" + "、".join(
            f"{r.get('group')}({r.get('count')}条)" for r in sample))
    result_total = task_state.get("result_total")
    if result_total is not None:
        satisfaction = task_state.get("search_satisfaction")
        label = {"full_support": "已确认", "partial_support": "部分确认",
                 "candidate_only": "只是相似候选", "no_match": "无结果"}.get(
            satisfaction, "")
        facts.append(f"检索到 {result_total} 张候选照片{('，' + label) if label else ''}。")
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            facts.append(f"照片复核观察：{tr['inspect_text']}")
    return facts


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

    @staticmethod
    def _emit_progress(turn, callback, *, stage: str, text: str, status: str) -> None:
        """记录一条公开进度事件（C13 数据合同：stage/step_index/timestamp 增量推送）。"""
        from datetime import datetime
        event = {
            "text": text,
            "status": status,
            "stage": stage,
            "step_index": len(turn.public_progress) + 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        turn.public_progress.append(event)
        if callback is not None:
            try:
                callback(event)
            except Exception:
                pass

    def run(self, message: str, *, history: str = "", task_state: dict | None = None,
            progress_callback=None, selected_handle: str | None = None,
            selected_result_set_id: str | None = None) -> RuntimeTurn:
        """progress_callback(event: dict) 在每次新增公开进度事件后调用（C13 数据合同：stage/step_index/timestamp 增量推送）。"""
        turn = RuntimeTurn(profile=self.profile.name, budget=BudgetState(
            max_model_steps=self.profile.max_model_steps,
            max_tool_calls=self.profile.max_tool_calls,
            max_inspections=self.profile.max_inspections,
            wall_time_s=self.profile.wall_time_s,
            final_reserve_s=self.profile.final_reserve_s,
        ))
        turn.budget.start()
        policy = ToolPolicy(scope_id=self.scope_id, viewer_id=self.viewer_id, budget=turn.budget)
        task = TaskState.from_dict(task_state, user_goal=message)
        guard = FinalGuard(scope_id=self.scope_id, viewer_id=self.viewer_id)
        system = SYSTEM_TEMPLATE.format(tools=self._tool_descriptions(),
                                        current_time=current_time_line())
        messages = [{"role": "system", "content": system}]
        if task.current_result_set:
            # B3.1：跨 turn 续接同一结果集，让模型知道当前可分页的结果集
            from .tools import result_set_context
            ctx = result_set_context(task.current_result_set, self.scope_id)
            if ctx:
                messages.append({"role": "system", "content": ctx})
        if selected_handle and selected_result_set_id:
            # Phase C C15：用户点选了结果集里的照片，模型可直接用该 handle 复核/交付原图
            messages.append({"role": "system", "content": (
                f"用户当前选中了结果集 {selected_result_set_id} 里的照片 "
                f"（handle={selected_handle}）。问'这张/原图/里面有几个人'时，"
                f"直接用 get_original_photos(handle={selected_handle}) 或 "
                f"inspect_photo(asset_handle={selected_handle})，不要重新全库搜索。"
            )})
        if history:
            messages.append({"role": "system", "content": f"最近对话：\n{history}"})
        messages.append({"role": "user", "content": message})
        self._emit_progress(turn, progress_callback, stage="thinking", status="running",
                            text="正在理解你的问题…")

        parse_retries = 0
        max_parse_retries = 1
        guard_retries = 0
        max_guard_retries = 1
        seen_tool_calls = set()
        dedup_retries = 0
        max_dedup_retries = 2
        search_has_preview = False
        inspect_called = False
        tool_call_seq = 0
        visual_retries = 0
        max_visual_retries = 1
        visual_intent = bool(__import__("re").search(
            r"桌上|桌面|颜色|几个|多少人|招牌|文字|天气|外套|衣服|猫|雪|小孩|穿着|穿|在做什么|"
            r"有没有|是什么|放着|写了|内容|细节", message))
        while True:
            if not turn.budget.can_model_step():
                turn.status = "partial" if turn.steps else "timeout"
                turn.reason = "model step budget exhausted"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="预算用尽")
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
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="输出无法解析")
                break
            if action.get("action") == "final":
                # 视觉细节意图 + 有 preview 候选 + 未 inspect → 确定性纠正一步（不依赖 12B 随机自觉）
                if search_has_preview and not inspect_called and visual_intent \
                        and visual_retries < max_visual_retries and turn.budget.can_model_step():
                    visual_retries += 1
                    denies_found = bool(__import__("re").search(
                        r"没(?:有|找到)|未找到|没有获取到|找不到|还没有",
                        str(action.get("answer") or "")))
                    messages.append({"role": "assistant", "content": raw})
                    if denies_found:
                        messages.append({"role": "user", "content": (
                            "你的回答说“没有找到”，但 search_memories 实际返回了候选照片（preview 里有 photo_1 等 handle），"
                            "这与工具结果矛盾。请调用 inspect_photo 复核 preview 里的 photo_1（asset_handle=photo_1），"
                            "根据观察如实回答；如果观察与用户假设不符，以观察为准。"
                        )})
                    else:
                        messages.append({"role": "user", "content": (
                            "用户询问的是照片里的视觉细节，而你还没有调用 inspect_photo。"
                            "search_memories 的 preview 里有可复核的照片（photo_1 等 handle）。"
                            "请先调用 inspect_photo（asset_handle 用 preview 里的 handle），得到观察后再输出 final。"
                        )})
                    continue
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
                        "search_satisfaction": task.search_satisfaction,
                        "condition_summary": task.search_condition_summary,
                        "tool_results": task.tool_results,
                        "evidence_refs": action.get("evidence_refs") or [],
                    },
                    delivered_count=task.delivered_count,
                )
                # L2：L1 确定性规则通过后，有工具结果时用 12B 评审语义级真实性
                if not problems and task.tool_results and turn.budget.can_model_step():
                    turn.budget.record_model_step()
                    trusted = _trusted_facts(task.as_dict())
                    faithful, judge_problems = judge_faithfulness(
                        self.chat_fn, query=message, tool_results=task.tool_results,
                        answer=turn.final_answer, trusted_facts=trusted)
                    turn.steps.append({"type": "judge", "faithful": faithful,
                                       "problems": list(judge_problems)})
                    if not faithful:
                        problems = judge_problems
                if problems:
                    if guard_retries < max_guard_retries and turn.budget.can_model_step():
                        guard_retries += 1
                        self._emit_progress(
                            turn, progress_callback,
                            stage="recovering", status="running",
                            text="结果里有一处信息对不上，我正在重新核对。")
                        messages.append({"role": "assistant", "content": raw})
                        inspect_obs = [
                            tr.get("inspect_text") for tr in task.tool_results
                            if tr.get("tool") == "inspect_photo" and tr.get("inspect_text")
                        ]
                        trusted = _trusted_facts(task.as_dict())
                        issue_lines = problems.natural_messages if hasattr(problems, "natural_messages") \
                            else [str(p) for p in problems]
                        recovery = (
                            "你的最终回答与工具结果有冲突，需要修正后重新输出 final：\n- "
                            + "\n- ".join(issue_lines) +
                            "\n\n可信事实（只能基于这些，不要重新调用昂贵工具）：\n- "
                            + "\n- ".join(trusted or ["(无工具结果)"]) +
                            ("\n注意：检索或复核没有产生可用照片时，不要声称找到候选照片，"
                             "也不要引用被拒/空的 inspect 调用；直接如实说没有找到或无法确认。"
                             if any("fabrication_from_empty" in p or "inspection_fabrication" in p
                                    for p in problems) else "") +
                            ("\ninspect_photo 的实际观察是：" + "；".join(inspect_obs)
                             + "\n如果观察与用户假设矛盾，以观察为准回答，不要迎合用户假设。"
                             if inspect_obs else "") +
                            "\n请只输出一个 JSON final（保留 evidence_refs 引用你实际使用的工具结果，"
                            "并在 evidence_refs 中列出你引用过的 inspect_photo 调用编号），"
                            "并按 query_satisfaction 如实表述（candidate_only 不能声称确认）。"
                        )
                        messages.append({"role": "user", "content": recovery})
                        continue
                    turn.status = "blocked_by_guard"
                    turn.reason = ";".join(problems)
                    if task.tool_results:
                        turn.final_answer = render_emergency_summary(
                            task.as_dict(), reason="回答未通过事实校验")
                    break
                self._emit_progress(
                    turn, progress_callback,
                    stage="finalizing", status="complete",
                    text="正在整理回答…")
                turn.status = "complete"
                break
            if action.get("action") != "tool_call":
                turn.status = "error"
                turn.reason = f"unknown_action:{action.get('action')}"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="动作无法识别")
                break
            tool_name = action.get("tool") or ""
            arguments = action.get("arguments") or {}
            # 模型有时把参数包在 arguments.schema 里，统一展开（工具契约兼容层）
            if isinstance(arguments.get("schema"), dict):
                arguments = {**arguments, **arguments["schema"]}
            public_status = action.get("public_status") or "正在处理。"
            tool_call_seq += 1
            tool_call_id = f"tool_call_{tool_call_seq}"
            call_signature = json.dumps({"tool": tool_name, "arguments": arguments},
                                        ensure_ascii=False, sort_keys=True)
            if call_signature in seen_tool_calls:
                if dedup_retries < max_dedup_retries and turn.budget.can_model_step():
                    dedup_retries += 1
                    messages.append({"role": "assistant", "content": raw})
                    if dedup_retries >= max_dedup_retries:
                        messages.append({"role": "user", "content": (
                            "你再次重复调用相同的工具和参数，被拒绝。"
                            "你不能再重复调用该工具。请立即调用 inspect_photo（asset_handle=photo_1）复核预览照片，"
                            "或直接输出 final 回答。"
                        )})
                    else:
                        messages.append({"role": "user", "content": (
                            "你刚用相同的工具和参数调用过，重复调用会被拒绝。"
                            "请换一个动作：如果需要看照片细节请调用 inspect_photo（使用预览里的 handle），"
                            "否则直接输出 final。"
                        )})
                    continue
                turn.status = "partial" if turn.steps else "error"
                turn.reason = f"tool_denied:{tool_name}:duplicate_tool_call"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="重复调用被拒绝")
                break
            seen_tool_calls.add(call_signature)
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
            self._emit_progress(
                turn, progress_callback,
                stage="tool_result" if result.status == "ok" else "tool_error",
                status=result.status, text=public_status)
            if not decision.allowed:
                turn.status = "partial" if turn.steps else "error"
                turn.reason = f"tool_denied:{tool_name}:{decision.reason}"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="工具调用被拒绝")
                break
            task.update_from_tool(tool_name, arguments, result.observation or {})
            task.record_tool_result(tool_call_id, tool_name, result.observation or {})
            if tool_name == "search_memories" and (result.observation or {}).get("can_inspect"):
                search_has_preview = True
            if tool_name == "inspect_photo":
                inspect_called = True
            # Observation 进入下一步模型上下文
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "tool_call_id": tool_name, "content": json.dumps(
                result.observation or {}, ensure_ascii=False)})

        turn.task_state = task.as_dict()
        return turn
