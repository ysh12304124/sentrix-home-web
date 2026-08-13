"""AgentRuntime 薄循环（v2 §4.1/§32.3）。

model -> tool -> observation -> model -> ... -> final
模型选择 Tool；代码通过 ToolPolicy 提供边界；BudgetManager 限制循环。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from .budget_manager import BudgetState
from .answer_nucleus import (build_nucleus, classify_deterministic,
                             render_simple)
from .completion import (CompletionState, DELIVER_MEDIA, RETRIEVE_EVIDENCE,
                         RESOLVE_OCR, RESOLVE_VISUAL)
from .emergency import render_emergency_summary
from .final_guard import FinalGuard
from .intent import multi_image_intent, ocr_intent, visual_intent
from .judge import judge_faithfulness
from .time_context import current_time_line
from .profile import get_profile
from .result_set import TaskState
from .tool_policy import ToolPolicy
from .tool_registry import get_tool


def _natural_partial(task_state: dict, problems=None) -> str:
    """G6 Recovery v3 第三层：自然 partial——展示已确认部分与缺口，不猜、不暴露工程错误。"""
    problems = problems or []
    issues_text = [str(p) for p in problems]
    ocr_failed = any("ocr" in p for p in issues_text)
    if ocr_failed:
        base = "我找到相关照片了，但这次没能可靠读出里面的文字/数字。原图在下方证据里，你可以直接打开看看。"
    else:
        base = "我找到了一些相关记录，但这次没能把你要的信息完全确认出来。"
    extras: list[str] = []
    places: list[str] = []
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "search_memories":
            for p in (tr.get("preview") or []) or []:
                place = str(p.get("place") or "").strip()
                if len(place) >= 2 and place not in places:
                    places.append(place)
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            extras.append(f"照片复核能看到：{tr['inspect_text'][:80]}")
        if tr.get("tool") == "read_photo_text" and (tr.get("ocr_text") or "").strip():
            extras.append(f"照片里读到的文字：{tr['ocr_text'][:80]}")
    if places:
        extras.append("能确认的地点：" + "、".join(places[:3]))
    if extras:
        base += " " + "；".join(extras[:2]) + "。"
    return base + "你可以让我继续核对，或换个问法再试。"


def _model_visible_observation(observation: dict | None) -> dict:
    """Keep tool evidence for planning while excluding telemetry from LLM context."""
    if not isinstance(observation, dict):
        return {}
    hidden = {"retrieval_timing", "debug", "telemetry", "trace", "_model_call_metrics"}
    compact = {key: value for key, value in observation.items() if key not in hidden}
    preview = compact.get("preview")
    if isinstance(preview, list):
        compact["preview"] = preview[:5]
    asset_ids = compact.get("asset_ids")
    if isinstance(asset_ids, list):
        compact["asset_ids"] = asset_ids[:20]
    return compact


def _model_visible_action(action: dict) -> str:
    """Feed the parsed action back without model reasoning or prose."""
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _merge_system_constraint(messages: list[dict], constraint: str) -> None:
    """Keep all system instructions at the beginning for strict chat templates."""
    if not constraint:
        return
    if not messages or messages[0].get("role") != "system":
        raise ValueError("agent messages must start with a system message")
    first = messages[0]
    existing = str(first.get("content") or "").rstrip()
    messages[0] = {
        **first,
        "content": f"{existing}\n\n{constraint}" if existing else constraint,
    }


SYSTEM_TEMPLATE = """你是 Sentrix 家庭记忆助手。你通过与工具协作完成用户请求。

可用工具（JSON 动作）：
{tools}

{current_time}

规则：
- 需要家庭记忆事实时调用工具；不需要时直接 final。
- 声称"没有找到/找不到/不存在相关记录"之前，必须先至少调用一次检索工具
  （search_memories / query_memory_facts / search_conversation_history / get_core_memory / get_person_memory）。
  未检索就断言"没有找到"会被纠正并要求重新检索。
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
- search_memories 的 preview 只显示前几张，每张带 place 字段（照片所在地，来自 GPS 反地理编码，
  如'秦皇岛市昌黎县'/'Chiang Mai'）；用户要求更多/下一页/还有吗 时，用 get_result_page（result_set_id 用 search_memories 返回的，page 从 1 开始）。
- 问'在哪里/哪个城市/什么地点/哪举办的'时，用 search_memories 检索并在回答中引用 preview 的 place 字段；
  query_memory_facts 只返回时间/数量/分组，不能回答照片地点。
- 时间、数量、首末存在性、日期、分组等确定性事实一律用 query_memory_facts，并把用户问题里的时间写进 filters.time（如 '2023年'、'2025-05'）。用户问任何年份/月份都必须如实填进 filters.time，不要省略；不要用 search_memories 代替，也不要用模型估算。
- 按月份/地点统计分布用 query_memory_facts 的 operation=group，并填 group_by（month 或 place）。
- operation=group 且 group_by=place 时，工具会返回 known_location_assets/unknown_location_assets 覆盖信息：
  只要 unknown_location_assets>0，回答必须如实说明还有多少张照片没有可靠地点信息，不能把地点说成完整清单。
- operation=meal 回答'吃过什么/吃饭/火锅'类问题：工具会返回 explicit_foods（明确食物，按事件去重）、
  meal_scene_events（只能确认在吃饭）、possible_events；回答必须逐项列出 explicit_foods 里的食物
  （如'火锅、蛋糕…'）并说明各出现几次，有 meal_scene_events 时还要说明其中一部分只能确认在用餐、
  不能确认具体菜品；没有 explicit_foods 时才只说用餐场景。
- final 回答直接给答案，先回答用户问题本身；需要说明不确定时用自然语言，不要复述检索过程。
- 回答结构：1) 直接答案 2) 必要的 uncertainty 3) 可选一句补充。不要以"我为您找到 N 张候选照片/检索到…"开头。
- 内部检索词汇（query_satisfaction、candidate_only、partial_support、full_support、no_match、候选照片、
  匹配程度、检索结果、相似候选）不得原样出现在 final 回答里；需要用用户能懂的话转译。
- 不确定性用自然语言四级：
  确定 → 直接给答案（"是在秦皇岛如是海度假村。"）；
  较可能 → "看起来是在…"；
  不确定 → "可能是在…，但我还不能完全确定。"；
  无依据 → "现有记录里看不出来。"。
- 内部工具状态只用于决定怎么说（这些词本身不能出现在回答里）：
  full_support=可以确认；partial_support=部分条件确认，用自然语言说出哪些还没确认（如"地点可以确认，时间还不能完全确定"）；
  candidate_only=只是相似候选，不能声称确认，要说"找到几张接近的，还不能完全确认"；
  no_match=如实说没有找到。
- 检索满足度与照片复核是两层，分开表述：检索层描述用户语义条件（活动/地点/时间）是否确认；
  复核层（inspect_photo/read_photo_text）确认照片里直接可见的细节（雪、人、物品、文字、颜色、价格）。
  即使照片里看到了山/雪，也不能把 candidate_only 的"爬山"说成已确认；
  示例："找到 3 张接近的，'爬山'还不能完全确认；最接近的一张照片里没有看到明显积雪。"
- 地点问题：search_memories preview 每张带 place 字段（GPS 反地理编码），回答时直接引用该地点，
  不要因为没有 inspect 就回答"无法确认地点"。
- 检索条件已确认（condition_summary 标记 matched/confirmed，或 query_satisfaction=full_support）时，直接当作确定事实回答，
  不要画蛇添足加"还不能完全确认"；只有 candidate_only 或关键条件 unknown 时才说"找到几张接近的，还不能完全确认"。
- 部分条件确认（partial_support）时：先直接回答已经确认的部分（如地点、时间、数量、价格），
  再用一句自然语言带过未确认的条件；绝对不要因为部分条件未知就整体说"无法确认/还不能确定在哪里/没有直接给你结论"。
- 照片里的文字/数字问题（菜单价格、招牌、店名、电话、年份、写了什么）：当 search_memories 的
  recommended_resolution 提示用 read_photo_text 时，调用 read_photo_text 读取文字后再回答，
  不要只 search 后就说"无法确认"或反问用户。
- 如果已经调用 read_photo_text / inspect_photo，但照片里仍读不到可靠内容（没有文字、看不清、
  与问题无关），直接如实回答"现有照片里看不出来/不知道"，不要继续绕圈子，不要承诺"可以继续核对"。
- filters.place 填结构化地点名（城市/区县/景区/地标）。系统会按行政区匹配照片的 GPS 反地理编码：
  例如"秦皇岛如是海度假村"也能匹配"河北省秦皇岛市昌黎县"的照片，"清迈"能匹配英文"Chiang Mai"。
  不要把要找的目标/活动/主题当作 place（"沙雕"是主题不是地点）。地点不确定时留空，只按时间和人物过滤。
- public_status 是给用户看的简短进度说明。
"""


_IMAGE_REQUEST_RE = __import__("re").compile(
    r"给我看看|给我看|发我|发给我|发来|原图|都给我|全部给我|"
    r"展示|显示(?:一下|给我)?|让我看看|看看(?:这些|照片|图)?|"
    r"把.{0,6}(?:照片|图片|图)|第二张|第三张|第\d张|那张|哪张|"
    r"打开(?:照片|图片)|看图|给我图", __import__("re").I)
_INLINE_QUESTION_RE = __import__("re").compile(
    r"这张|这张照片|这张图|这图|图里|照片里|画面里|里面|放大|细节|"
    r"有几个人|几个人|穿.{0,4}(?:什么|颜色)|桌上|桌子上|写的?什么|什么字|"
    r"招牌|文字|天气|在哪拍的|哪里拍的", __import__("re").I)
_CHAT_ONLY_RE = __import__("re").compile(
    r"你好|谢谢|在吗|再见|哈哈|好的|嗯|哦|你是谁|你叫什么|你会什么|帮我写|"
    r"写一|写个|改一|翻译|解释一下什么是|什么是|怎么用|步骤|教程", __import__("re").I)


@dataclass
class ToolResult:
    tool: str
    status: str
    observation: dict | None = None
    error: str | None = None
    latency_s: float = 0.0


def _pending_resolution(task) -> dict | None:
    """Phase E §14：Premature Final Guard —— 检索明确要求视觉/OCR 复核但尚未执行时，阻止提前 final。

    依据 search_memories observation 里的 recommended_resolution（确定性字段），
    不依赖模型自觉。对应工具已调用过则不再要求。
    """
    for tr in reversed(task.tool_results or []):
        if tr.get("tool") == "search_memories":
            obs = tr.get("observation") or {}
            rec = obs.get("recommended_resolution") or {}
            if rec.get("needed") and rec.get("tool"):
                called = any((x.get("tool") or "") == rec["tool"]
                             for x in (task.tool_results or []))
                if not called:
                    return rec
    return None


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
    answer_grounding: dict = field(default_factory=dict)
    termination_reason: str = ""
    ocr_partial: bool = False
    ocr_partial_reason: str = ""
    nucleus_injected: bool = False


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
    if op == "meal":
        foods = task_state.get("fact_value") or []
        if foods:
            facts.append("工具确认的明确食物（按事件去重）为：" + "、".join(
                f"{f.get('food')}({f.get('events')}次)" for f in foods[:10]))
    rows = task_state.get("fact_rows") or []
    if rows:
        sample = rows[:6]
        facts.append("工具返回的分组为：" + "、".join(
            f"{r.get('group')}({r.get('count')}条)" for r in sample))
    result_total = task_state.get("result_total")
    if result_total is not None:
        satisfaction = task_state.get("search_satisfaction")
        label = {"full_support": "可以确认", "partial_support": "部分信息已确认",
                 "candidate_only": "还不能完全确认", "no_match": "无结果"}.get(
            satisfaction, "")
        facts.append(f"找到 {result_total} 张相关照片{('，' + label) if label else ''}。")
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            handle = tr.get("inspect_handle") or ""
            facts.append(f"照片{(' ' + handle) if handle else ''}复核观察：{tr['inspect_text']}")
    return facts


def _confirmed_facts(task_state: dict) -> list[str]:
    """Phase F 回归修复：提取工具已确认、可直接引用的实质事实（地点/OCR/观察），
    recovery 与 L2 评审时注入，避免模型只能复述“找到 N 张/部分确认”套话。"""
    facts: list[str] = []
    places: list[str] = []
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "search_memories":
            for p in (tr.get("preview") or []) or []:
                place = str(p.get("place") or "").strip()
                if len(place) >= 2 and place not in places:
                    places.append(place)
        if tr.get("tool") == "read_photo_text" and (tr.get("ocr_text") or "").strip():
            facts.append(f"照片文字读到的内容：{tr['ocr_text']}")
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            handle = tr.get("inspect_handle") or ""
            facts.append(f"照片{(' ' + handle) if handle else ''}复核观察：{tr['inspect_text']}")
    if places:
        facts.append("照片 GPS 反编码确认的地点：" + "、".join(places[:5]) + "。")
    return facts


def _build_answer_grounding(*, message: str, task: TaskState,
                            selected_handle: str | None = None) -> dict:
    """D1：Evidence-by-Default 契约。

    display_mode：
      - result_grid：用户明确要求图片 → 图片直接可见。
      - inline_images：用户针对当前选中照片提问 → 当前照片直接可见。
      - collapsed：回答依赖证据但未要求图片 → 原始证据默认折叠。
      - none：闲聊/一般知识/纯写作，无家庭证据。
    """
    tool_results = task.tool_results or []
    evidence_handles: list[str] = []
    evidence_assets: list[str] = []
    rep_assets: list[dict] = []
    for tr in reversed(tool_results):
        if tr.get("tool") == "inspect_photo" and tr.get("inspect_handle"):
            handle = str(tr["inspect_handle"])
            if handle not in evidence_handles:
                evidence_handles.append(handle)
                rep_assets.append({"handle": handle, "kind": "inspection",
                                   "observation": (tr.get("inspect_text") or "")[:80]})
        for s in (tr.get("samples") or []) or []:
            if isinstance(s, dict) and s.get("asset_id"):
                aid = s["asset_id"]
                if aid not in evidence_assets:
                    evidence_assets.append(aid)
                    if len(rep_assets) < 6:
                        rep_assets.append({"asset_id": aid,
                                           "captured_at": s.get("captured_at") or "",
                                           "caption": (s.get("caption") or s.get("transcript") or "")[:80]})
    for handle in (task.result_preview or [])[:12]:
        if handle and handle not in evidence_handles:
            evidence_handles.append(handle)
            if len(rep_assets) < 6:
                rep_assets.append({"handle": handle, "kind": "result_preview"})
    evidence_count = len(evidence_handles) + len(evidence_assets)
    used_evidence = evidence_count > 0
    explicit_image = bool(_IMAGE_REQUEST_RE.search(message or ""))
    inline_question = bool(selected_handle) and bool(_INLINE_QUESTION_RE.search(message or ""))
    chat_only = bool(_CHAT_ONLY_RE.search(message or "")) and not tool_results
    if chat_only or not used_evidence:
        display_mode = "none"
    elif explicit_image:
        display_mode = "result_grid"
    elif inline_question:
        display_mode = "inline_images"
    else:
        display_mode = "collapsed"
    return {
        "required": used_evidence,
        "display_mode": display_mode,
        "evidence_count": evidence_count,
        "representative_evidence": rep_assets[:6],
        "all_evidence_available": bool(task.current_result_set),
        "result_set_id": task.current_result_set,
        "explicit_image_request": explicit_image,
    }


class AgentRuntime:
    """Thin tool-loop runtime. 模型调用通过传入的 chat_fn 注入。"""

    def __init__(self, *, chat_fn, profile_name: str | None = None,
                 scope_id="home-default", viewer_id="owner", conversation_id=None,
                 ocr_settings: dict | None = None):
        self.chat_fn = chat_fn
        self.profile = get_profile(profile_name)
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.conversation_id = conversation_id
        self.ocr_settings = ocr_settings or {}

    def _chat(self, messages, *, call_type: str, step_id: str):
        """Pass observation metadata when supported by the injected client."""
        try:
            return self.chat_fn(messages, call_type=call_type, step_id=step_id)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return self.chat_fn(messages)

    def _tool_descriptions(self) -> str:
        from .capability import tool_capability_summary
        lines = []
        from .tool_registry import list_tools
        allowed = set(self.profile.tools) if self.profile.tools else None
        for spec in list_tools():
            if spec.readiness == "blocked":
                continue
            if allowed is not None and spec.name not in allowed:
                continue
            capability = tool_capability_summary(spec.name)
            lines.append(f"- {spec.name}: {spec.description}"
                         f"{(' 能力=' + capability) if capability else ''}"
                         f" 输入schema={json.dumps(spec.input_schema, ensure_ascii=False)}")
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
            selected_result_set_id: str | None = None,
            conversation_summary: str = "") -> RuntimeTurn:
        """progress_callback(event: dict) 在每次新增公开进度事件后调用（C13 数据合同：stage/step_index/timestamp 增量推送）。"""
        turn = RuntimeTurn(profile=self.profile.name, budget=BudgetState(
            max_model_steps=self.profile.max_model_steps,
            max_tool_calls=self.profile.max_tool_calls,
            max_inspections=self.profile.max_inspections,
            wall_time_s=self.profile.wall_time_s,
            final_reserve_s=self.profile.final_reserve_s,
        ))
        turn.budget.start()
        policy = ToolPolicy(scope_id=self.scope_id, viewer_id=self.viewer_id, budget=turn.budget,
                            allowed_tools=set(self.profile.tools) if self.profile.tools else None)
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
        if selected_handle:
            # Phase C C15：用户点选了结果集里的照片，模型可直接用该 handle 复核/交付原图
            ctx = f"用户当前选中了照片（handle={selected_handle}）"
            if selected_result_set_id:
                ctx += f"，属于结果集 {selected_result_set_id}"
            ctx += ("。问'这张/原图/里面有几个人'时，直接用 "
                    f"get_original_photos(handle={selected_handle}) 或 "
                    f"inspect_photo(asset_handle={selected_handle})，不要重新全库搜索。")
            messages.append({"role": "system", "content": ctx})
        if history:
            messages.append({"role": "system", "content": f"最近对话：\n{history}"})
        if conversation_summary:
            messages.append({"role": "system", "content": f"本会话摘要：\n{conversation_summary}"})
        active_lines = []
        if task.active_person:
            active_lines.append(f"当前关注人物：{task.active_person}")
        if task.active_event:
            active_lines.append(f"当前关注事件：{task.active_event}")
        if task.open_questions:
            active_lines.append("未解决问题：" + "、".join(str(q) for q in task.open_questions[:5]))
        if active_lines:
            messages.append({"role": "system", "content": "当前上下文：\n" + "\n".join(active_lines)})
        messages.append({"role": "user", "content": message})
        self._emit_progress(turn, progress_callback, stage="thinking", status="running",
                            text="正在理解你的问题…")

        parse_retries = 0
        max_parse_retries = 2
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
        resolution_retries = 0
        max_resolution_retries = 2
        completion_retries = 0
        max_completion_retries = 1
        unknown_tool_retries = 0
        max_unknown_tool_retries = 1
        model_call_seq = 0
        last_model_step_id = None
        next_call_type = "agent"
        next_call_trigger = "用户问题"
        wants_visual = visual_intent(message)
        # Phase E：Adaptive Visual Budget——按问题类型放宽视觉复核预算
        wants_multi = multi_image_intent(message)
        wants_ocr = ocr_intent(message)
        adaptive_inspections = self.profile.max_inspections
        if wants_multi:
            adaptive_inspections = max(adaptive_inspections, 4)
        elif wants_ocr:
            adaptive_inspections = max(adaptive_inspections, 2)
            # read_photo_text 需要整图 + 3x3 tile 多次图片推理，放宽总预算
            # （同图 OCR 结果已缓存，只有首次需要长预算）
            turn.budget.wall_time_s = max(turn.budget.wall_time_s, 240)
        turn.budget.max_inspections = adaptive_inspections
        # Phase G G3：Completion State / Gate —— 最小动态 requirements（不重建 Planner）
        completion = CompletionState(message)
        while True:
            if not turn.budget.can_model_step():
                # Phase H H4：步骤耗尽但 OCR 已读到确定性硬值时，直接渲染交付（不依赖 12B 收尾）
                if task.tool_results:
                    try:
                        nuc = build_nucleus(task.as_dict(), message)
                        kind = classify_deterministic(message)
                        simple = render_simple(nuc, kind, message) if kind else None
                        if simple:
                            turn.steps.append({"type": "nucleus", "status": "rendered",
                                               "kind": kind, "value": simple[:40]})
                            turn.final_answer = simple
                            turn.status = "complete"
                            turn.reason = ""
                            turn.termination_reason = "deterministic_at_step_limit"
                            break
                    except Exception:
                        pass
                turn.status = "partial" if turn.steps else "timeout"
                turn.reason = "model step budget exhausted"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="预算用尽")
                break
            turn.budget.record_model_step()
            # Phase H H-A：确定性硬值约束——工具已确认的数量/日期/OCR 硬值注入，
            # 禁止 12B 在生成 final 时改写（只注入一次，约束文本很短）。
            if not turn.nucleus_injected and task.tool_results:
                try:
                    nuc = build_nucleus(task.as_dict(), message)
                    ctext = nuc.constraint_text()
                    if ctext:
                        _merge_system_constraint(messages, ctext)
                    turn.nucleus_injected = True
                except Exception:
                    turn.nucleus_injected = True
            model_call_seq += 1
            model_step_id = f"model_call_{model_call_seq}"
            call_type = next_call_type
            next_call_type = "agent"
            call_trigger = next_call_trigger
            next_call_trigger = "继续处理当前问题"
            try:
                raw = self._chat(messages, call_type=call_type, step_id=model_step_id)
            except Exception as exc:
                error_text = str(exc)
                outcome = "context_blocked" if any(
                    marker in error_text.lower()
                    for marker in ("context", "token budget", "maximum context", "preflight")
                ) else "model_error"
                turn.steps.append({"type": "model", "status": "error",
                                   "reason": error_text, "step_id": model_step_id,
                                   "call_type": call_type, "trigger": call_trigger,
                                   "turn_outcome": outcome,
                                   "parse_status": "not_applicable",
                                   "next_step": "stop"})
                # D12：恢复/后续模型调用失败时，不丢弃已产出的 final 回答
                if turn.final_answer:
                    turn.status = "partial"
                    turn.reason = f"model_error_after_final: {exc}"
                    break
                turn.status = "error"
                turn.reason = f"model_call_error: {exc}"
                break
            model_step = {"type": "model", "raw": (raw or "")[:500],
                          "step_id": model_step_id, "call_type": call_type,
                          "trigger": call_trigger,
                          "turn_outcome": None,
                          "parse_status": "failed",
                          "next_step": "recovery"}
            turn.steps.append(model_step)
            last_model_step_id = model_step_id
            action = self._parse_action(raw)
            if action is not None:
                model_step["parse_status"] = "success"
                model_step["action"] = action.get("action")
                if action.get("action") == "tool_call":
                    model_step["turn_outcome"] = "tool_call"
                    model_step["next_step"] = action.get("tool") or "tool"
                    model_step["tool"] = action.get("tool") or ""
                    model_step["arguments"] = action.get("arguments") or {}
                elif action.get("action") == "final":
                    model_step["turn_outcome"] = "final_answer"
                    model_step["next_step"] = "final_answer"
                    model_step["answer_preview"] = str(action.get("answer") or "")[:300]
            if action is None:
                model_step["turn_outcome"] = "parse_failure"
                model_step["parse_status"] = "failed"
                if parse_retries < max_parse_retries and turn.budget.can_model_step():
                    parse_retries += 1
                    messages.append({"role": "assistant", "content": raw})
                    if parse_retries >= 2:
                        # D11：第二次恢复——只要求输出最简单的 final（12B 长输出/跑题时最有效）
                        self._emit_progress(
                            turn, progress_callback, stage="recovering", status="running",
                            text="刚才的输出没有解析成功，我正在简化处理。")
                        messages.append({"role": "user", "content": (
                            "你连续两次输出的 JSON 都无法解析。现在请只输出一个最简单的 final JSON："
                            '{"action":"final","answer":"<一句话回答>","evidence_refs":["tool_call_1"]}。'
                            "answer 基于已返回的工具结果用一句自然的话；没有工具结果就如实说没有找到相关记录；"
                            "不要调用任何工具、不要写解释。"
                        )})
                    else:
                        messages.append({"role": "user", "content": (
                            "你的上一条输出不是合法的 JSON 对象，无法解析。"
                            "请严格只输出一个 JSON 对象（action 只能是 tool_call 或 final），"
                            "不要 markdown、不要多余文字、不要省略结尾的引号或括号。"
                        )})
                    next_call_type = "recovery"
                    next_call_trigger = "模型输出无法解析，要求重新输出合法 JSON"
                    continue
                turn.status = "error"
                turn.reason = "unparseable_action"
                turn.termination_reason = "parse_failure"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="输出无法解析")
                break
            if action.get("action") == "final":
                # Phase E §14：Premature Final Guard —— recommended_resolution 要求继续证据解析
                resolution = _pending_resolution(task)
                if resolution:
                    if resolution_retries >= max_resolution_retries:
                        # 模型多次忽略提示 → runtime 自动执行证据解析工具（代码兜底，不依赖 12B 自觉）
                        auto_spec = get_tool(resolution["tool"])
                        preview = (task.result_preview or []) or []
                        if auto_spec is not None and preview and turn.budget.can_tool_call():
                            tool_name = resolution["tool"]
                            tool_call_seq += 1
                            tool_call_id = f"tool_call_{tool_call_seq}"
                            auto_args = {"asset_handle": preview[0]}
                            if tool_name == "read_photo_text":
                                auto_args["question"] = message
                            self._emit_progress(
                                turn, progress_callback, stage="inspecting", status="running",
                                text=f"正在复核照片 {preview[0]}…" if tool_name == "inspect_photo"
                                else "正在读取照片中的文字…")
                            auto_t0 = time.monotonic()
                            auto_decision = policy.execute(auto_spec, auto_args, context={
                                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                                "task_state": task.as_dict(), "history": history,
                                "conversation_id": self.conversation_id,
                                "ocr_settings": self.ocr_settings,
                            })
                            auto_latency = round(time.monotonic() - auto_t0, 2)
                            auto_observation = dict(auto_decision.observation or {})
                            auto_internal_metrics = auto_observation.pop("_model_call_metrics", [])
                            turn.steps.append({
                                "type": "tool", "tool": tool_name, "arguments": auto_args,
                                "status": "ok" if auto_decision.allowed else "denied",
                                "observation": auto_observation,
                                "error": auto_decision.error, "latency_s": auto_latency,
                                "step_id": tool_call_id,
                                "parent_step_id": last_model_step_id,
                                "call_type": "automatic_recovery",
                                "internal_model_call_metrics": auto_internal_metrics,
                            })
                            if auto_decision.allowed:
                                task.update_from_tool(tool_name, auto_args, auto_observation)
                                task.record_tool_result(tool_call_id, tool_name,
                                                        auto_observation)
                                # Phase H H-A：自动解析后若已是确定性问题（价格/年份/数量/日期），
                                # 直接用 Nucleus 渲染，不再消耗模型步骤（12B 常因步骤耗尽无法收尾）。
                                try:
                                    if tool_name == "read_photo_text":
                                        nuc = build_nucleus(task.as_dict(), message)
                                        kind = classify_deterministic(message)
                                        simple = render_simple(nuc, kind, message) if kind else None
                                        if simple:
                                            turn.steps.append({"type": "nucleus", "status": "rendered",
                                                               "kind": kind, "value": simple[:40]})
                                            turn.final_answer = simple
                                            turn.status = "complete"
                                            turn.termination_reason = "deterministic_after_auto"
                                            break
                                except Exception:
                                    pass
                                messages.append({"role": "assistant", "content": _model_visible_action(action)})
                                messages.append({"role": "tool", "tool_call_id": tool_call_id,
                                                 "content": json.dumps(
                                                     _model_visible_observation(
                                                         auto_observation),
                                                     ensure_ascii=False)})
                                next_call_type = "recovery"
                                next_call_trigger = f"运行时自动执行 {tool_name} 后重新生成回答"
                                continue
                    elif resolution_retries < max_resolution_retries and turn.budget.can_model_step():
                        resolution_retries += 1
                        messages.append({"role": "assistant", "content": _model_visible_action(action)})
                        messages.append({"role": "user", "content": (
                            f"{resolution.get('reason') or '问题需要复核照片'}。"
                            f"这是完成回答的必要步骤：你必须立即调用 {resolution['tool']}"
                            "（asset_handle 用 preview 里的 handle，例如 photo_1），"
                            "先拿到实际观察，再基于观察输出 final。不调用该工具就无法正确回答。"
                        )})
                        next_call_type = "recovery"
                        next_call_trigger = f"回答前必须调用 {resolution['tool']}，模型提前结束"
                        continue
                # 视觉细节意图 + 有 preview 候选 + 未 inspect → 确定性纠正一步（不依赖 12B 随机自觉）
                if search_has_preview and not inspect_called and wants_visual \
                        and visual_retries < max_visual_retries and turn.budget.can_model_step():
                    visual_retries += 1
                    denies_found = bool(__import__("re").search(
                        r"没(?:有|找到)|未找到|没有获取到|找不到|还没有",
                        str(action.get("answer") or "")))
                    messages.append({"role": "assistant", "content": _model_visible_action(action)})
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
                    next_call_type = "recovery"
                    next_call_trigger = "视觉问题已有候选照片，但模型尚未执行 inspect_photo"
                    continue
                # Phase G G3：Completion Gate —— retrieve_evidence / deliver_media 等需求未完成时，
                # 只告诉模型“任务还没完成”，由模型自己决定下一步 Tool（不重建 Planner）。
                # 先保存模型产出的 final 文本：gate 后续模型调用失败时也不丢失已产出回答（D12）。
                turn.final_answer = str(action.get("answer") or "")
                try:
                    from .final_writer import naturalize_answer
                    turn.final_answer = naturalize_answer(turn.final_answer)
                except Exception:
                    pass
                # Phase H H-A：简单确定性问题（数量/日期/布尔）直接按 Nucleus 确定性渲染，
                # 不再把 total=5 交给 12B 自由改写（reg3 少报根因）。
                try:
                    if task.tool_results:
                        nuc = build_nucleus(task.as_dict(), message)
                        kind = classify_deterministic(message)
                        if kind:
                            simple = render_simple(nuc, kind, message)
                            if simple:
                                turn.steps.append({"type": "nucleus", "status": "rendered",
                                                   "kind": kind, "value": simple[:40]})
                                turn.final_answer = simple
                        else:
                            print(f"[nucleus] kind={kind} no_simple nuc_price={[v.label for v in nuc.all('price')] if nuc else None}", file=__import__("sys").stderr)
                except Exception as exc:
                    print(f"[nucleus] render error: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
                completion.update(task.as_dict())
                gate_prompted = False
                for req in completion.blocking():
                    if req.code == RESOLVE_OCR and _pending_resolution(task):
                        continue  # 已由上方 recommended_resolution 流程处理
                    if req.code == RESOLVE_VISUAL and search_has_preview and not inspect_called \
                            and wants_visual and visual_retries < max_visual_retries:
                        continue  # 已由上方视觉流程处理
                    if completion_retries < max_completion_retries \
                            and turn.budget.can_model_step():
                        completion_retries += 1
                        messages.append({"role": "assistant", "content": _model_visible_action(action)})
                        if req.code == RETRIEVE_EVIDENCE:
                            messages.append({"role": "user", "content": (
                                f"{req.reason} 这是完成回答的必要步骤：请先调用检索工具"
                                "（search_memories / query_memory_facts / get_core_memory / get_person_memory），"
                                "拿到工具结果后再输出 final。"
                            )})
                        elif req.code == DELIVER_MEDIA:
                            messages.append({"role": "user", "content": (
                                f"{req.reason} 请调用 {req.tool}（使用 search_memories 返回的 result_set_id）"
                                "交付可查看的照片后再输出 final。"
                            )})
                        else:
                            messages.append({"role": "user", "content": (
                                f"{req.reason} 这是完成回答的必要步骤：你必须立即调用 {req.tool}"
                                "（asset_handle 用 search_memories 返回的 preview handle，如 photo_1），"
                                "先拿到实际结果，再基于结果输出 final。"
                            )})
                        gate_prompted = True
                        break
                if gate_prompted:
                    continue
                # Phase F F1：Final Answer Writer——草稿违反 Answer Policy 时用受控事实重写
                if turn.final_answer and turn.budget.can_model_step():
                    try:
                        from .final_writer import build_final_context, needs_rewrite, rewrite_final
                        fctx = build_final_context(message, task.as_dict())
                        if needs_rewrite(turn.final_answer, fctx):
                            turn.budget.record_model_step()
                            model_call_seq += 1
                            writer_step_id = f"model_call_{model_call_seq}"
                            rewritten = rewrite_final(
                                self._chat, fctx, turn.final_answer, step_id=writer_step_id)
                            writer_status = "failed" if not rewritten else (
                                "unchanged" if rewritten == turn.final_answer else "rewritten")
                            turn.steps.append({"type": "writer", "status": writer_status,
                                               "step_id": writer_step_id,
                                               "call_type": "writer",
                                               "trigger": "候选回答违反最终回答样式策略",
                                               "answer_preview": str(rewritten or turn.final_answer)[:300]})
                            if rewritten and rewritten != turn.final_answer:
                                turn.final_answer = naturalize_answer(rewritten)
                    except Exception:
                        pass
                problems = guard.check(
                    turn.final_answer,
                    task_state={
                        "scope_id": self.scope_id,
                        "viewer_id": self.viewer_id,
                        "user_query": message,
                        "history_text": history,
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
                        "selected_asset_handle": task.selected_asset_handle or selected_handle,
                        "tool_results": task.tool_results,
                        "evidence_refs": action.get("evidence_refs") or [],
                    },
                    delivered_count=task.delivered_count,
                )
                # C9：结构化记录 L1 guard 检查结果（debug 展示用，普通用户不可见）
                turn.steps.append({
                    "type": "guard",
                    "status": "fail" if problems else "pass",
                    "codes": list(problems) if problems else [],
                    "attempt": guard_retries + 1,
                })
                # L2：L1 确定性规则通过后，有工具结果时用 12B 评审语义级真实性
                if not problems and task.tool_results and turn.budget.can_model_step():
                    turn.budget.record_model_step()
                    trusted = _confirmed_facts(task.as_dict()) + _trusted_facts(task.as_dict())
                    try:
                        model_call_seq += 1
                        judge_step_id = f"model_call_{model_call_seq}"
                        faithful, judge_problems = judge_faithfulness(
                            self._chat, query=message, tool_results=task.tool_results,
                            answer=turn.final_answer, trusted_facts=trusted,
                            step_id=judge_step_id)
                        turn.steps.append({"type": "judge", "faithful": faithful,
                                           "problems": list(judge_problems),
                                           "step_id": judge_step_id,
                                           "call_type": "faithfulness_judge",
                                           "trigger": "L1 Guard 通过且回答使用了工具事实",
                                           "attempt": guard_retries + 1})
                        if not faithful:
                            problems = judge_problems
                    except Exception as exc:
                        turn.steps.append({"type": "judge", "status": "skipped",
                                           "reason": f"model_call_error:{exc}",
                                           "step_id": judge_step_id,
                                           "call_type": "faithfulness_judge",
                                           "trigger": "L1 Guard 通过且回答使用了工具事实",
                                           "attempt": guard_retries + 1})
                if problems:
                    severity = problems.severity if hasattr(problems, "severity") else "truth"
                    if severity == "hard_block":
                        # G4 Safety Hard Block：权限/隐私/内部泄漏/非法写 —— 不可放行、不可恢复
                        turn.status = "blocked_by_guard"
                        turn.reason = "hard_block:" + ";".join(problems)
                        if task.tool_results:
                            turn.final_answer = render_emergency_summary(
                                task.as_dict(), reason="回答未通过事实校验")
                        break
                    if severity == "style":
                        # G4 Style Advisory：只做一次建议性重写；重写失败/未改变 → 放行原答案
                        # （绝不得把事实正确的答案变成 blocked_by_guard）
                        if turn.final_answer and turn.budget.can_model_step():
                            try:
                                from .final_writer import (build_final_context, needs_rewrite,
                                                          rewrite_final)
                                fctx = build_final_context(message, task.as_dict())
                                if needs_rewrite(turn.final_answer, fctx):
                                    turn.budget.record_model_step()
                                    rewritten = rewrite_final(self.chat_fn, fctx, turn.final_answer)
                                    if rewritten and rewritten != turn.final_answer:
                                        turn.steps.append({"type": "writer",
                                                           "status": "rewritten_style"})
                                        turn.final_answer = rewritten
                            except Exception:
                                pass
                        turn.status = "complete"
                        break
                    # Phase H H4：guard 拦截后若问题可确定性渲染（价格/年份/数量），直接交付硬值
                    if task.tool_results:
                        try:
                            nuc = build_nucleus(task.as_dict(), message)
                            kind = classify_deterministic(message)
                            simple = render_simple(nuc, kind, message) if kind else None
                            if simple:
                                turn.steps.append({"type": "nucleus", "status": "rendered",
                                                   "kind": kind, "value": simple[:40]})
                                turn.final_answer = simple
                                turn.status = "complete"
                                turn.reason = ""
                                turn.termination_reason = "deterministic_after_guard"
                                break
                        except Exception:
                            pass
                    # G4 Truth Recoverable → Recovery v3：rewrite → one tool recovery → natural partial
                    if guard_retries < max_guard_retries and turn.budget.can_model_step():
                        guard_retries += 1
                        self._emit_progress(
                            turn, progress_callback,
                            stage="recovering", status="running",
                            text="结果里有一处信息对不上，我正在重新核对。")
                        last_answer = (turn.final_answer or "").strip()[:300]
                        messages.append({"role": "assistant",
                                         "content": f"（你上一版 final 回答）{last_answer}"})
                        inspect_obs = [
                            tr.get("inspect_text") for tr in task.tool_results
                            if tr.get("tool") == "inspect_photo" and tr.get("inspect_text")
                        ]
                        ocr_obs = [
                            (tr.get("ocr_text") or "")[:600] for tr in task.tool_results
                            if tr.get("tool") == "read_photo_text"
                            and (tr.get("ocr_text") or "").strip()
                        ]
                        trusted = _confirmed_facts(task.as_dict()) + _trusted_facts(task.as_dict())
                        issue_lines = problems.natural_messages if hasattr(problems, "natural_messages") \
                            else [str(p) for p in problems]
                        recovery = (
                            "你的最终回答与工具结果有冲突，需要修正后重新输出 final：\n- "
                            + "\n- ".join(issue_lines) +
                            "\n\n可信事实（只能基于这些，不要重新调用昂贵工具）：\n- "
                            + "\n- ".join(trusted or ["(无工具结果)"]) +
                            ("\n注意：检索或复核没有产生可用照片时，不要声称找到候选照片，"
                             "也不要引用被拒/空的 inspect 调用；直接如实说没有找到或无法确认。"
                             if any("fabrication" in p or "fabrication_from_empty" in p
                                    or "inspection_fabrication" in p
                                    for p in problems) else "") +
                            ("\ninspect_photo 的实际观察是：" + "；".join(inspect_obs)
                             + "\n如果观察与用户假设矛盾，以观察为准回答，不要迎合用户假设。"
                             if inspect_obs else "") +
                            ("\nread_photo_text 实际读到的文字是：\n" + "\n".join(ocr_obs)
                             + "\n基于这些读到的文字直接回答具体内容（价格/店名/电话/年份），"
                               "不要笼统说'还不能确认'；检索层的不确定可单独用一句自然语言带过。"
                             if ocr_obs else "") +
                            "\n请只输出一个 JSON final（保留 evidence_refs 引用你实际使用的工具结果，"
                            "并在 evidence_refs 中列出你引用过的 inspect_photo 调用编号），"
                            "并按 query_satisfaction 如实表述（candidate_only 不能声称确认）。"
                            "final 必须先直接回答用户问题本身（地点问题直接说'是在…'，数字问题直接给数字），"
                            "不确定的其余条件用一句自然语言带过；禁止输出"
                            "'找到 N 张接近的照片；部分信息能对上；我可以继续帮你核对'这类套话。"
                        )
                        messages.append({"role": "user", "content": recovery})
                        next_call_type = "recovery"
                        next_call_trigger = "Guard 或 L2 事实一致性检查未通过"
                        continue
                    # Recovery v3 第二层：一次工具恢复——仍有证据解析需求未满足且预算允许
                    recovery_tool = None
                    if turn.budget.can_tool_call():
                        completion.update(task.as_dict())
                        for req in completion.blocking():
                            if req.code in (RESOLVE_OCR, RESOLVE_VISUAL) and req.tool:
                                recovery_tool = req.tool
                                break
                    if recovery_tool:
                        preview = (task.result_preview or []) or []
                        spec = get_tool(recovery_tool)
                        if spec is not None and preview:
                            auto_args = {"asset_handle": preview[0]}
                            if recovery_tool == "read_photo_text":
                                auto_args["question"] = message
                            self._emit_progress(
                                turn, progress_callback, stage="inspecting", status="running",
                                text=f"正在复核照片 {preview[0]}…" if recovery_tool == "inspect_photo"
                                else "正在读取照片中的文字…")
                            auto_decision = policy.execute(spec, auto_args, context={
                                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                                "task_state": task.as_dict(), "history": history,
                                "conversation_id": self.conversation_id,
                                "ocr_settings": self.ocr_settings,
                            })
                            if auto_decision.allowed:
                                task.update_from_tool(recovery_tool, auto_args,
                                                       auto_decision.observation or {})
                                task.record_tool_result(f"recovery_{recovery_tool}", recovery_tool,
                                                        auto_decision.observation or {})
                                completion.update(task.as_dict())
                                if recovery_tool == "read_photo_text" \
                                        and (auto_decision.observation or {}).get("status") == "partial":
                                    turn.ocr_partial = True
                                    turn.ocr_partial_reason = str(
                                        auto_decision.observation.get("reason") or "ocr_failed")
                                messages.append({"role": "assistant",
                                                 "content": _model_visible_action(action)})
                                messages.append({"role": "tool",
                                                 "tool_call_id": f"recovery_{recovery_tool}",
                                                 "content": json.dumps(
                                                     _model_visible_observation(
                                                         auto_decision.observation),
                                                     ensure_ascii=False)})
                                messages.append({"role": "user", "content": (
                                    "我重新读取了一次照片，请基于新的工具观察，直接输出一个修正后的 final。"
                                )})
                                continue
                    # Recovery v3 第三层：natural partial —— 展示已确认部分与缺口，不放错误答案
                    turn.status = "partial"
                    turn.reason = "truth_unresolved:" + ";".join(problems[:4])
                    turn.final_answer = _natural_partial(task.as_dict(), problems)
                    break
                self._emit_progress(
                    turn, progress_callback,
                    stage="finalizing", status="complete",
                    text="正在整理回答…")
                # G6：OCR 显式 partial —— 读文字失败且回答如实反映“没读清”时，
                # 以 natural partial 收尾（status=partial, reason=ocr_timeout），不猜、不暴露工程错误
                if turn.ocr_partial and re.search(
                        r"没(?:能|有)?(?:可靠)?读(?:出|到|清)|读不清|看不清|无法读取|没能读出|没有读到|读不出来",
                        turn.final_answer or ""):
                    turn.status = "partial"
                    turn.reason = turn.ocr_partial_reason or "ocr_timeout"
                else:
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
                    messages.append({"role": "assistant", "content": _model_visible_action(action)})
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
                    next_call_type = "recovery"
                    next_call_trigger = "模型重复调用了相同工具和参数"
                    continue
                turn.status = "partial" if turn.steps else "error"
                turn.reason = f"tool_denied:{tool_name}:duplicate_tool_call"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="重复调用被拒绝")
                break
            seen_tool_calls.add(call_signature)
            spec = get_tool(tool_name)
            if spec is None:
                if unknown_tool_retries < max_unknown_tool_retries and turn.budget.can_model_step():
                    unknown_tool_retries += 1
                    messages.append({"role": "assistant", "content": _model_visible_action(action)})
                    from .tool_registry import list_tools
                    valid = "、".join(sorted({s.name for s in list_tools()
                                              if s.readiness != "blocked"}))
                    messages.append({"role": "user", "content": (
                        f"工具「{tool_name}」不存在。可用工具只有：{valid}。"
                        "请重新选择一个可用工具，或直接输出 final。"
                    )})
                    next_call_type = "recovery"
                    next_call_trigger = f"模型请求了不存在的工具 {tool_name}"
                    continue
                turn.steps.append({"type": "tool", "tool": tool_name, "status": "error",
                                   "reason": "unknown_tool", "step_id": tool_call_id,
                                   "parent_step_id": last_model_step_id})
                turn.reason = "unknown_tool:" + tool_name
                turn.termination_reason = "tool_unavailable"
                break
            if tool_name == "inspect_photo":
                handle_arg = str(arguments.get("asset_handle") or "")
                self._emit_progress(
                    turn, progress_callback,
                    stage="inspecting", status="running",
                    text=f"正在检查照片 {handle_arg}…" if handle_arg else "正在检查照片…")
            t0 = time.monotonic()
            decision = policy.execute(spec, arguments, context={
                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                "task_state": task.as_dict(), "history": history,
                "conversation_id": self.conversation_id,
                "ocr_settings": self.ocr_settings,
            })
            latency = round(time.monotonic() - t0, 2)
            observation = dict(decision.observation or {})
            internal_model_metrics = observation.pop("_model_call_metrics", [])
            result = ToolResult(tool=tool_name,
                                status="ok" if decision.allowed else "denied",
                                observation=observation,
                                error=decision.error, latency_s=latency)
            turn.steps.append({
                "type": "tool", "tool": tool_name, "arguments": arguments,
                "status": result.status, "observation": result.observation,
                "error": result.error, "latency_s": latency,
                "step_id": tool_call_id, "parent_step_id": last_model_step_id,
                "internal_model_call_metrics": internal_model_metrics,
            })
            emit_text = public_status
            if tool_name == "inspect_photo" and result.status == "ok":
                handle_arg = str(arguments.get("asset_handle") or "")
                emit_text = f"已检查照片 {handle_arg}…" if handle_arg else "已检查照片…"
            self._emit_progress(
                turn, progress_callback,
                stage="tool_result" if result.status == "ok" else "tool_error",
                status=result.status, text=emit_text)
            if not decision.allowed:
                turn.status = "partial" if turn.steps else "error"
                turn.reason = f"tool_denied:{tool_name}:{decision.reason}"
                if task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="工具调用被拒绝")
                break
            task.update_from_tool(tool_name, arguments, result.observation or {})
            task.record_tool_result(tool_call_id, tool_name, result.observation or {})
            completion.update(task.as_dict())
            # G6：OCR 显式 partial（超时/失败）→ 记录到 turn，用于最终 natural partial 语义
            if tool_name == "read_photo_text" and (result.observation or {}).get("status") == "partial":
                turn.ocr_partial = True
                turn.ocr_partial_reason = str(result.observation.get("reason") or "ocr_failed")
                self._emit_progress(
                    turn, progress_callback, stage="tool_result", status="partial",
                    text="这次没能可靠读出照片里的文字…")
            if tool_name == "search_memories" and (result.observation or {}).get("can_inspect"):
                search_has_preview = True
            if tool_name == "inspect_photo":
                inspect_called = True
            # Observation 进入下一步模型上下文
            messages.append({"role": "assistant", "content": _model_visible_action(action)})
            messages.append({"role": "tool", "tool_call_id": tool_name, "content": json.dumps(
                _model_visible_observation(result.observation), ensure_ascii=False)})
            next_call_trigger = f"工具 {tool_name} 返回结果"

        turn.task_state = task.as_dict()
        turn.task_state["completion"] = completion.as_dict()
        turn.answer_grounding = _build_answer_grounding(
            message=message, task=task, selected_handle=selected_handle)
        turn.termination_reason = _classify_termination(turn)
        return turn


def _classify_termination(turn: RuntimeTurn) -> str:
    """D11：termination_reason 全量分类（telemetry 用）。"""
    reason = turn.reason or ""
    if turn.status == "complete":
        return "complete"
    if turn.reason == "ocr_timeout":
        return "ocr_timeout"
    if turn.status == "blocked_by_guard" or "guard" in reason:
        return "guard_recovery_exhausted"
    if "unparseable" in reason:
        return "parse_failure"
    if "model step budget" in reason:
        return "model_step_limit"
    if "wall" in reason or turn.status == "timeout":
        return "wall_time_limit"
    if "unknown_tool" in reason:
        return "tool_unavailable"
    if "tool_denied" in reason:
        return "tool_rejected"
    if "budget" in reason:
        return "tool_call_limit"
    if "l2" in reason.lower():
        return "l2_recovery_failed"
    if turn.status == "partial":
        return "partial"
    return turn.status or "unknown"
