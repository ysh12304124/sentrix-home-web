"""Goal declaration planner for Agent 2.

The model declares evidence needs in a clean, lightweight format; this adapter validates
the declaration, maps simplified needs to strict EvidenceRequirements if needed, and returns
the structured TaskDeclaration.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
from dataclasses import dataclass

from .planner_contracts import parse_planner_action
from .task_state import TaskDeclaration, EvidenceRequirement
from .evidence_contract import planner_evidence_types


# Planning 提示词。
#
# 输出示例里的每个字段都用 <...> 占位符，**不要**放任何真实值 —— 早期版本在示例里
# 直接写死了真实的 scope_id，结果小模型会整段照抄示例：占位符原样吐出、而那个
# 16 位十六进制的 scope_id 抄错一位（...2344... → ...2345...），触发 scope_mismatch，
# 长问题里约 70% 因此被判失败。scope 是系统注入的参数，模型不产出它。
_DECLARATION_PROMPT = """你正在规划家庭记忆任务。只返回一个 JSON 对象，不要解释，不要 Markdown 代码块围栏。

输出格式（<尖括号> 里是占位符，必须替换成这道题的实际内容；不要把占位符原样输出）：
{{"action":"declare","declaration":{{"goal":"<用简体中文完整描述用户想要什么>","requirements":[{{"id":"req_1","evidence_type":"<从下方注册表里选>","description":"<这条需求具体要找什么>"}}]}}}}

不要输出 scope_id 字段 —— 相册范围由系统注入，你不需要也不应该填写。

evidence_type 只能取自下方注册表，不得创造新类型：
{evidence_types}

规则：
- goal 用简体中文完整描述用户目标，保留完整语义，不要压缩成几个词。
- 只声明回答问题所必需的最小证据集合，不要把检索步骤本身当成答案证据重复声明。
- 答案要从相册照片里得到的（地点/时间/人物/内容/合影张数/有没有某物/照片里的文字），声明 memory_asset。
- 拍摄时间、日期、年份声明 temporal_metadata；照片里的文字、数字、价格声明 visible_text。
- 地点问题声明 location_metadata；照片内容、颜色、动作声明 visual_observation。
- 只有需要确认照片中的人名时才声明 photo_identity，不要用 visual_observation 代替。
- structured_fact 只用于能从照片或文字直接确认的确定事实；金额类若没有照片或文字依据就不要声明，检索不到就如实说无法确认。
- 用户没有明确提到历史对话时，不要声明 user_statement。
- 同一种 evidence_type 只声明一次。

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
    def __init__(self, *, chat_fn, enable_format_rewrite: bool = True):
        self.chat_fn = chat_fn
        self.enable_format_rewrite = enable_format_rewrite
        # 规划最多尝试几次（含首次）。失败会把「上一轮输出 + 失败原因」回灌后重试。
        # 默认 3：实测第 2 次能救回绝大多数格式/scope 类失败，再往上收益很小。
        self.max_plan_attempts = max(1, int(os.getenv("SENTRIX_PLANNER_MAX_ATTEMPTS", "3")))

    def declare(self, message: str, *, scope_id: str, history: str = "",
                include_debug: bool = False, step_id: str = "planner_step_0") -> PlannerDeclarationResult:
        evidence_lines = "\n".join(f"- {item}" for item in planner_evidence_types())
        prompt = _DECLARATION_PROMPT.format(evidence_types=evidence_lines)
        # 历史拼进 system 文本，**不要**再插一条 system 消息：llama.cpp 的 Jinja
        # 聊天模板要求 system 必须在开头，插入第二条会直接 500
        # （"Jinja Exception: System message must be at the beginning"）。
        # 实测 24 题（4.9%）因此失败。此修法同步自 153。
        if history:
            prompt += "\n\n历史对话背景：\n" + history
        base_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ]
        prompt_copy = copy.deepcopy(base_messages) if include_debug else None

        # 规划失败后**带着错误原因重试**：模型看不到自己的输出和具体错误就无法自我纠正。
        # 实测长问题里约 70% 是因为照抄提示词示例、把 scope_id/占位符抄错而失败；
        # 单纯重发同样的提示词没用（模型会犯同样的错），必须把上一轮的原文和失败
        # 原因一起回灌，它才能针对性修正。
        messages = list(base_messages)
        last_raw = ""
        last_reason = "invalid_planner_action"
        for attempt in range(self.max_plan_attempts):
            try:
                sig = inspect.signature(self.chat_fn)
                if "call_type" in sig.parameters:
                    raw = self.chat_fn(messages, call_type="planner", step_id=step_id) or ""
                else:
                    raw = self.chat_fn(messages) or ""
            except Exception:
                # 裸 except 会让 planner 失败原因彻底消失（外层只看到 reason=planner_call_error，
                # 分不清是超时、连接被拒还是 400）。这里必须留下真实异常，否则只能靠猜。
                import logging
                logging.getLogger("sentrix.goal_planner").exception(
                    "planner call failed: scope=%s step=%s attempt=%s", scope_id, step_id, attempt + 1)
                return PlannerDeclarationResult(fallback_reason="planner_call_error", prompt=prompt_copy)

            last_raw = raw
            declaration, reason = self._parse_declaration(raw, scope_id=scope_id, message=message)
            if declaration is not None:
                return PlannerDeclarationResult(declaration=declaration, raw=raw, prompt=prompt_copy)
            last_reason = reason
            if attempt + 1 >= self.max_plan_attempts:
                break
            messages = list(base_messages) + [
                {"role": "assistant", "content": str(raw or "")},
                {"role": "user", "content": self._correction_hint(reason, scope_id)},
            ]
        return PlannerDeclarationResult(fallback_reason=last_reason, raw=last_raw, prompt=prompt_copy)

    @staticmethod
    def _correction_hint(reason: str, scope_id: str) -> str:
        """把上一轮的失败原因翻译成给模型的修正指令（重规划用）。"""
        by_reason = {
            "scope_mismatch": (
                "不要输出 scope_id 字段。相册范围由系统注入，你填任何值都是错的。"),
            "invalid_planner_action": (
                "输出的 JSON 结构必须是 "
                '{"action":"declare","declaration":{"goal":"...","requirements":'
                '[{"id":"req_1","evidence_type":"...","description":"..."}]}}，'
                "并且必须完整闭合（每个 { 都要有配对的 }）。"),
        }
        return (
            "你上一次的输出无法被解析（原因：%s）。%s "
            "另外：只输出一个 JSON 对象，不要解释、不要 Markdown 代码块围栏（```）；"
            "尖括号 <...> 是占位符，必须替换成这道题的实际内容，不要把占位符原样写出来。"
            "请重新输出修正后的 JSON。" % (reason, by_reason.get(reason, ""))
        )

    def _parse_declaration(self, raw: str, *, scope_id: str, message: str):
        """解析并校验一次规划输出，返回 (TaskDeclaration | None, 失败原因)。"""
        try:
            payload = self._parse_json(raw)
            # 兼容小模型可能返回的扁平或近义词结构
            payload = self._normalize_payload(payload, scope_id=scope_id, default_goal=message)
            action = parse_planner_action(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "invalid_planner_action"
        if action.kind not in {"declare", "revise"} or action.declaration is None:
            return None, "invalid_planner_action"
        if action.declaration.scope_id != scope_id:
            return None, "scope_mismatch"
        return action.declaration, ""

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
            # 允许零需求：无照片证据可确认的全库统计/拒答题，或纯聊天，不强行凑一个
            # memory_asset 需求。模型在回答阶段决定是否需要检索。
            pass
        decl["requirements"] = normalized_reqs
        return payload

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        # 结尾围栏必须独立剥离：模型常给出「只有结尾 ``` 、没有开头 ```」的输出，
        # 旧写法把这两步嵌在一起，导致结尾的 ``` 残留 → json.loads 在非末尾位置报错
        # → 下面「补一个收尾 }」的修复路径被跳过 → planner 直接判 invalid_planner_action。
        # 实测 5 题里 4 题因此失败（模型只是漏了根对象的 }，本可自动补上）。
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start = text.find("{")
        if start < 0:
            raise ValueError("planner did not return JSON")
        candidate = text[start:]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if (exc.pos < len(candidate) - 1
                    or GoalPlanner._root_brace_balance(candidate) != 1):
                raise
            payload = json.loads(candidate + "}")
        if not isinstance(payload, dict):
            raise ValueError("planner action must be an object")
        return payload

    @staticmethod
    def _root_brace_balance(text: str) -> int:
        """Return unmatched object braces without treating quoted braces as syntax."""
        balance = 0
        quoted = False
        escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
        return balance

