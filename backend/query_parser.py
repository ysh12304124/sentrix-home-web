"""Model-driven query parser with sanitize, one-shot repair and safe fallback.

Model responsibility: open-vocabulary semantics — intent, actions, target,
facets, semantic conditions, natural negation, contextual reference.

Code responsibility: schema, sanitizer, explicit date/media/negation recovery
from the raw user message, and safe fallback when the model is unavailable.
Never a keyword-table classifier.
"""

from __future__ import annotations

from datetime import datetime
import json
import re

from .model_clients import parse_json_response
from .query_contracts import QueryParseDraft, sanitize_query_parse


_PARSER_MARKER = "查询解析器"
_REPAIR_MARKER = "QuerySpec 修复器"


_QUERY_PARSE_DRAFT_SCHEMA = {
    "mode": "none|contextual|evidence",
    "actions": [
        {
            "type": "answer_question|return_assets|summarize_person|summarize_event|timeline|compare|propose_correction",
            "target": "person|event|activity|place|object|clothing|relationship|time|general",
            "coverage": "best|top_k|all_relevant",
        }
    ],
    "facets": [{"dimension": "person|time|place|activity|clothing|object|visual|relationship|semantic", "surface_text": "用户原文片段"}],
    "entity_names": ["..."],
    "time_expression": "如 2024 年 5 月 或 去年春节",
    "media_expressions": ["照片|图片|视频|原图"],
    "semantic_conditions": [{"dimension": "place|activity|clothing|object|visual|ocr", "value": "开放语义值", "strictness": "semantic_required"}],
    "negative_conditions": [{"dimension": "person|media|other", "value": "被排除的值"}],
    "result_requirement": {"mode": "best|top_k|all_relevant", "top_k": 10},
    "ambiguities": ["需要用户澄清的问题"],
    "confidence": 0.0,
}


_PARSER_PROMPT = """你是 Sentrix 的查询解析器，不负责回答用户问题，也不能读取数据库。
你的任务是把用户消息和最近对话转换为严格 JSON QueryParseDraft，不输出运行时身份和数据库 ID。

规则：
1. 普通聊天、写作、建议、情绪支持返回 mode=none，不能要求家庭证据。
2. 具体人物、时间、地点、照片、衣着、活动、关系、原图、比较和时间线属于家庭记忆请求，mode=evidence。
3. 自然人物提及但没有问历史事实的可以是 mode=contextual。
4. 日期、明确人物、媒体类型和"不要/不是/排除"是候选条件；由后端确定性代码决定其是否属于 deterministic_hard。
5. 做饭、晚饭、自拍、颜色、材质等视觉或语义描述属于 semantic_conditions。
6. "都、所有、全部、还有哪些"使用 result_requirement.mode=all_relevant。
7. "介绍一下某人"是 person 目标。
8. 不能创建实体 ID，不能猜测人物身份，不能调用工具，不能补充家庭事实。
9. 一句话可以包含多个 action（例如 answer_question + return_assets），不要压缩成单一目标。
10. facets 保留用户提到的所有维度，surface_text 用原文片段。
11. 只输出 JSON，不要输出 scope_id/scope_mode/viewer_id/conversation_id/entity_ids，不要 Markdown。

当前时间：{{now}}
最近对话：{{conversation}}
用户消息：{{message}}

输出 schema：
{{query_parse_draft_json_schema}}"""


_REPAIR_PROMPT = """你是 Sentrix QuerySpec 修复器。
只修复 JSON 结构、枚举值和字段类型，不改变用户原文已经明确表达的硬条件。
不得添加人物、日期、地点、媒体或证据。
如果无法确定，将字段置空或放入 semantic_conditions。

用户原文：{{message}}
模型原始 JSON：{{raw_json}}
代码发现的问题：{{validation_errors}}
请只输出修复后的 QueryParseDraft JSON。"""


_DATE_RE = re.compile(r"20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])?(?:\s*\d{1,2}\s*日?)?")


class QueryParser:
    """Emit a validated ``QueryParseDraft`` from a user turn.

    ``PydanticAIPlanner`` is tried first when available.  On any failure the
    parser falls back to raw ``gamma.chat(prompt, json_mode=True)`` and, if the
    JSON is malformed, one repair attempt using the plan §7.2 prompt.  A final
    safe fallback returns ``mode="none"`` with no keyword contamination.
    """

    def __init__(self, gamma=None, framework_planner=None):
        self.gamma = gamma
        self.framework_planner = framework_planner

    def parse(self, message, recent_turns="", now=None):
        raw = self._call_parser(message, recent_turns, now)
        draft, errors = self._draft_and_validate(raw)
        if errors and raw:
            repaired = self._call_repair(message, raw, errors)
            if repaired:
                draft, errors = self._draft_and_validate(repaired)
        if errors:
            draft = self._safe_fallback()
        return self._apply_deterministic_overlay(draft, message)

    def _call_parser(self, message, recent_turns, now):
        prompt = self._render_parser_prompt(message, recent_turns, now)
        if self.framework_planner is not None and getattr(self.framework_planner, "available", False):
            try:
                result = self.framework_planner.plan(prompt)
            except Exception:
                result = None
            if isinstance(result, dict) and result:
                return result
        return self._invoke_gamma(prompt)

    def _call_repair(self, message, raw_json, errors):
        prompt = _REPAIR_PROMPT.replace("{{message}}", str(message or "")).replace(
            "{{raw_json}}", json.dumps(raw_json, ensure_ascii=False, default=str)
        ).replace("{{validation_errors}}", "; ".join(errors))
        return self._invoke_gamma(prompt)

    def _invoke_gamma(self, prompt):
        if not self.gamma or not hasattr(self.gamma, "chat"):
            return None
        try:
            text = self.gamma.chat(prompt, json_mode=True)
        except Exception:
            return None
        if not text:
            return None
        result = parse_json_response(text)
        return result if isinstance(result, dict) and result else None

    @staticmethod
    def _render_parser_prompt(message, recent_turns, now):
        now_iso = (now or datetime.now()).isoformat(timespec="seconds")
        return (
            _PARSER_PROMPT
            .replace("{{now}}", now_iso)
            .replace("{{conversation}}", str(recent_turns or "")[-1200:])
            .replace("{{message}}", str(message or ""))
            .replace("{{query_parse_draft_json_schema}}", json.dumps(_QUERY_PARSE_DRAFT_SCHEMA, ensure_ascii=False))
        )

    @staticmethod
    def _draft_and_validate(raw):
        draft = sanitize_query_parse(raw or {}, message="")
        errors = []
        if draft.mode not in {"none", "contextual", "evidence"}:
            errors.append("mode missing or invalid")
        return draft, errors

    @staticmethod
    def _safe_fallback():
        return QueryParseDraft(intent="answer", answer_target="general", mode="none")

    @staticmethod
    def _apply_deterministic_overlay(draft, message):
        value = str(message or "")
        if not draft.time_expression:
            match = _DATE_RE.search(value)
            if match:
                draft.time_expression = match.group(0)
        for token in ("不要", "排除", "不是"):
            idx = value.find(token)
            if idx < 0:
                continue
            window = value[idx : idx + 20]
            if "视频" in window and not any(item.get("value") == "video" and item.get("dimension") == "media" for item in draft.negative_conditions):
                draft.negative_conditions.append({"dimension": "media", "value": "video", "source_text": window})
            if "照片" in window and not any(item.get("value") == "image" and item.get("dimension") == "media" for item in draft.negative_conditions):
                # Rare but harmless — user says "不要照片" excludes images.
                draft.negative_conditions.append({"dimension": "media", "value": "image", "source_text": window})
        return draft
