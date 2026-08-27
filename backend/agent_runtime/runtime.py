"""AgentRuntime 薄循环（v2 §4.1/§32.3）。

model -> tool -> observation -> model -> ... -> final
模型选择 Tool；代码通过 ToolPolicy 提供边界；BudgetManager 限制循环。
"""

from __future__ import annotations

import json
import inspect
import os
import re
import time
from dataclasses import dataclass, field

from .budget_manager import BudgetState
from .jit_prompt import build_jit_system_prompt
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
from .tool_registry import get_tool, list_tools


def _people_count_from_summary(text: str) -> int:
    """Infer a bounded visible-person count without assigning identities."""
    text = str(text or "")
    count_words = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
                   "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    matches = re.findall(
        r"([一二两三四五六七八九十]|\d+)\s*名?\s*(?:个)?"
        r"(人|成年人|成人|成年男子|成年女性|小孩|孩子|幼儿)", text)
    if not matches:
        return 0
    aggregate = [count_words.get(raw, int(raw) if raw.isdigit() else 0)
                 for raw, kind in matches if kind == "人"]
    if aggregate:
        return max(aggregate)
    return sum(count_words.get(raw, int(raw) if raw.isdigit() else 0)
               for raw, _kind in matches)


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
        if tr.get("tool") == "inspect_photo":
            inspect_text = (tr.get("inspect_observation") or "").strip()
            # Older replay traces only have inspect_text; do not treat an
            # uncertain failure summary as a visual observation.
            if not inspect_text and str(tr.get("certainty") or "supported").lower() != "uncertain":
                inspect_text = (tr.get("inspect_text") or "").strip()
            if inspect_text:
                extras.append(f"照片复核能看到：{inspect_text[:80]}")
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
    # Retrieval diagnostics describe the server-owned full candidate pool and
    # must not become answer facts.  They remain in the recorded trace for
    # recall auditing, while the model only receives the bounded public
    # candidate window and evidence projection.
    hidden = {
        "retrieval_timing", "debug", "telemetry", "trace",
        "raw_candidate_count", "validation_candidate_count", "validation_batches",
        "retrieval_channels", "validation_rows",
    }
    compact = {key: value for key, value in observation.items() if key not in hidden}
    preview = compact.get("preview")
    if isinstance(preview, list):
        compact["preview"] = preview[:5]
        if compact["preview"] and isinstance(compact["preview"][0], dict):
            compact["recommended_handle"] = compact["preview"][0].get("handle")
        # Asset IDs are server-side provenance. Handles remain the only stable
        # references visible to the Agent; benchmark/debug projections resolve
        # the handle mapping outside the model prompt.
        compact["preview"] = [
            {key: value for key, value in item.items() if key != "asset_id"}
            if isinstance(item, dict) else item
            for item in compact["preview"]
        ]
    # ResultSet keeps the full internal mapping server-side. The model only
    # needs stable handles from the bounded preview; exposing asset IDs turns
    # the whole candidate set into model-visible evidence and encourages
    # paging/inspection of unrelated images.
    for key in ("asset_ids", "retrieved_asset_ids", "preview_asset_ids", "evidence_asset_ids",
                "_retrieved_asset_ids", "_preview_asset_ids", "_source_asset_id"):
        compact.pop(key, None)
    return compact


def _normalize_preview_handle(arguments: dict, preview_handles: list[str] | None) -> tuple[dict, str | None]:
    """Keep visual/OCR inspection inside the handles actually shown to the model."""
    if not isinstance(arguments, dict) or not preview_handles:
        return arguments, None
    normalized = dict(arguments)
    requested = str(arguments.get("asset_handle") or "")
    # Older local-model prompts used image_id/query for visual inspection.  An
    # image_id is often a private ResultSet id (rs_...), not a public handle;
    # never pass it through as an asset handle or silently inspect a stale id.
    legacy_image_id = str(arguments.get("image_id") or "")
    legacy_query = arguments.get("query")
    if not requested and legacy_image_id:
        requested = legacy_image_id
    if not normalized.get("question") and legacy_query:
        normalized["question"] = legacy_query
    if not requested:
        normalized["asset_handle"] = preview_handles[0]
        return normalized, None
    if requested not in preview_handles:
        # Never silently inspect a different image when a stale handle is
        # supplied. Preserve the requested handle so policy/tool feedback can
        # force a fresh search or a visible-handle retry.
        normalized["asset_handle"] = requested
        return normalized, requested
    return normalized, None


def _normalize_selected_image_handles(handles, preview_handles, limit: int = 6) -> list[str]:
    """Accept only explicit, currently visible handles for user-facing images."""
    visible = {str(handle) for handle in (preview_handles or []) if handle}
    values = handles if isinstance(handles, list) else [handles] if handles else []
    selected = []
    for value in values:
        handle = str(value or "").strip()
        if handle and handle in visible and handle not in selected:
            selected.append(handle)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _model_visible_action(action: dict) -> str:
    """Feed the parsed action back without model reasoning or prose."""
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


_RECOVERY_MESSAGE_MARKERS = (
    "你尚未检索相册",
    "你的上一条输出不是合法的 JSON",
    "你连续两次输出的 JSON 都无法解析",
    "你现在必须给出一个明确的 final 答案",
    "这是完成回答的必要步骤",
    "请先调用 inspect_photo",
    "请先调用 search_memories",
    "这与工具结果矛盾",
    "当前任务还没有完成",
)


def _prompt_annotations(messages: list[dict]) -> list[dict]:
    """Mark runtime-injected recovery prompts without changing API roles.

    The model API still receives a normal ``role=user`` message for
    compatibility with the deployed backend.  Debug/benchmark traces get an
    explicit origin so 8771 does not present an internal correction as a real
    user utterance.
    """
    annotations = []
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if index <= 1 or not any(marker in content for marker in _RECOVERY_MESSAGE_MARKERS):
            continue
        annotations.append({
            "message_index": index,
            "message_origin": "system_recovery",
            "content_preview": content[:240],
        })
    return annotations

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
  或 {{"action":"final","answer":"...","evidence_refs":["tool_call_1", ...],"selected_image_handles":["photo_1"]}}
- 不要重复调用相同的工具和参数：同一轮里相同 tool+arguments 只允许一次，重复会被拒绝。
- inspect_photo 的 asset_handle 必须原样使用 search_memories 返回 preview 里的 handle（如 photo_1、photo_2）。
  不要编造 preview 之外的 handle。
- search_memories 若返回 recommended_handle，inspect_photo / read_photo_text 默认优先使用该 handle；
  不要因为示例中的 photo_1 文本而改选其它照片。
- selected_image_handles 只填写最终确实要展示给用户的图片，必须来自当前 preview，最多 6 张；搜索返回的全部候选不能直接当作展示图片。
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
  通常为城市/区县/景区名，如"某市某区"、"某度假村"）；用户要求更多/下一页/还有吗 时，用 get_result_page（result_set_id 用 search_memories 返回的，page 从 1 开始）。
- 问'在哪里/哪个城市/什么地点/哪举办的'时，用 search_memories 检索并在回答中引用 preview 的 place 字段；
  query_memory_facts 只返回时间/数量/分组，不能回答照片地点。
- 工具选择看用户意图，不是看有没有日期：
  · 用户要找照片、看照片内容（颜色/服装/道具/人数/雕塑/文字/哪张照片）、或问照片是在哪里拍的 → 用 search_memories（把日期写进 filters.time，不要省略）；需要照片里的视觉细节时再 inspect_photo / read_photo_text。
  · 只有纯统计/确定性事实（一共多少张、最早/最近一张、是否存在、按时间/地点分组）才用 query_memory_facts，并把用户问题里的时间写进 filters.time（如 '2023年'、'2025-05'），不要用模型估算。
- 用户要'给我所有视频/照片/音频/文本'或'列出相册里的视频'时，用 query_memory_facts 的 operation=list，并在 filters.media 填 video/image/audio/text；工具返回 items 是实际媒体，回答要引用 items 里的 file_name、时长、场景/关键帧来源，不能只报数量。如果工具返回 summary，直接使用 summary 里的文件名和描述逐项列出。
- 按月份/地点统计分布用 query_memory_facts 的 operation=group，并填 group_by（month 或 place）。
- operation=group 且 group_by=place 时，工具会返回 known_location_assets/unknown_location_assets 覆盖信息：
  只要 unknown_location_assets>0，回答必须如实说明还有多少张照片没有可靠地点信息，不能把地点说成完整清单。
- operation=meal 回答'吃过什么/吃饭/火锅'类问题：工具会返回 explicit_foods（明确食物，按事件去重）、
  meal_scene_events（只能确认在吃饭）、possible_events；回答必须逐项列出 explicit_foods 里的食物
  （如具体菜名）并说明各出现几次，有 meal_scene_events 时还要说明其中一部分只能确认在用餐、
  不能确认具体菜品；没有 explicit_foods 时才只说用餐场景。
- final 回答直接给答案，先回答用户问题本身；需要说明不确定时用自然语言，不要复述检索过程。
- 回答结构：1) 直接答案 2) 必要的 uncertainty 3) 可选一句补充。不要以"我为您找到 N 张候选照片/检索到…"开头。
- 内部检索词汇（query_satisfaction、candidate_only、partial_support、full_support、no_match、候选照片、
  匹配程度、检索结果、相似候选）不得原样出现在 final 回答里；需要用用户能懂的话转译。
- 不确定性用自然语言四级：
  确定 → 直接给答案（如"是在某景区门口。"）；
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
  例如用户说"某度假村"，能匹配到该度假村所在区县（如"某市某县"）拍摄的照片；说"某城市"也能匹配该市下辖区县的照片；中文地名与英文译名可互相匹配。
  不要把要找的目标/活动/主题当作 place（活动/主题不是地点）。地点不确定时留空，只按时间和人物过滤。
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



def _merge_system_constraint(messages: list[dict], constraint: str) -> None:
    """Keep all system instructions at the beginning for strict chat templates.

    Qwen3.5/3.8 chat templates raise "System message must be at the
    beginning" when a system message appears mid-conversation, which turns
    into a 400 from vLLM /tokenize and /chat/completions. Merge later system
    constraints into the leading system message instead of appending.
    """
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


def _pending_resolution(task) -> dict | None:
    """Phase E §14：Premature Final Guard —— 检索明确要求视觉/OCR 复核但尚未执行时，阻止提前 final。

    依据 search_memories observation 里的 recommended_resolution（确定性字段），
    不依赖模型自觉。对应工具只有在产生可用证据后才算完成；失败/partial
    结果仍保持 pending，交给后续有界恢复逻辑处理。
    """
    for tr in reversed(task.tool_results or []):
        if tr.get("tool") == "search_memories":
            obs = tr.get("observation") or {}
            rec = (tr.get("recommended_resolution") or
                   obs.get("recommended_resolution") or {})
            if rec.get("needed") and rec.get("tool"):
                if not CompletionState._tool_succeeded(task.tool_results or [], rec["tool"]):
                    return rec
    return None


def _next_resolution_handle(task, tool: str) -> str | None:
    """Pick an uninspected bounded-preview handle for deterministic recovery."""
    preview = [str(handle) for handle in (task.result_preview or []) if handle]
    if not preview:
        return None
    if tool != "inspect_photo":
        return preview[0]
    inspected = {
        str(result.get("inspect_handle"))
        for result in (task.tool_results or [])
        if result.get("tool") == "inspect_photo" and result.get("inspect_handle")
    }
    return next((handle for handle in preview if handle not in inspected), None)


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
    # Agent 2 shadow state stays separate from the legacy conversational TaskState.
    agent2_trace: dict = field(default_factory=dict)
    answer_grounding: dict = field(default_factory=dict)
    selected_image_handles: list[str] = field(default_factory=list)
    selected_image_ids: list[str] = field(default_factory=list)
    termination_reason: str = ""
    ocr_partial: bool = False
    ocr_partial_reason: str = ""
    nucleus_injected: bool = False


def public_agent2_trace(trace: dict | None) -> dict:
    """Return benchmark-safe Agent 2 telemetry without scope or asset references."""
    trace = trace or {}
    requirements = ((trace.get("task_state") or {}).get("requirements") or [])
    status_counts: dict[str, int] = {}
    public_requirements = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        status = str(requirement.get("status") or "open")
        status_counts[status] = status_counts.get(status, 0) + 1
        public_requirements.append({
            "id": str(requirement.get("id") or ""),
            "evidence_type": str(requirement.get("evidence_type") or ""),
            "status": status,
            "required": bool(requirement.get("required", True)),
            "coverage_status": str(requirement.get("coverage_status") or "candidate"),
            "failure_reason": str(requirement.get("failure_reason") or ""),
            "attempt_count": int(requirement.get("attempt_count") or 0),
            "last_attempt": str(requirement.get("last_attempt") or ""),
        })
    entries = ((trace.get("evidence_ledger") or {}).get("entries") or [])
    partial_entries = sum(
        1 for entry in entries
        if isinstance(entry, dict)
        and int((entry.get("coverage") or {}).get("processed") or 0)
        < int((entry.get("coverage") or {}).get("requested") or 0)
    )
    return {
        "requirements": public_requirements,
        "requirement_status_counts": status_counts,
        "evidence_coverage": {"entries": len(entries), "partial_entries": partial_entries},
        "planner_decisions": list(trace.get("planner_decisions") or []),
        "terminal_reason": str(trace.get("terminal_reason") or ""),
        "budget_outcome": dict(trace.get("budget_outcome") or {}),
    }


def record_agent2_tool_evidence(task_state, evidence_ledger, spec, *,
                                tool_call_id: str, input_refs=(), provenance_refs=(),
                                observation: dict | None = None,
                                allow_partial: bool = True,
                                question: str = "") -> bool:
    """Record rich, scope-bound evidence produced by one capability call.

    ``observation`` is optional for backwards compatibility with the original
    shadow tests. When present, one call may produce multiple evidence types and
    bind each type to every compatible requirement.
    """
    from .evidence_ledger import Coverage, LedgerEntry

    observation = observation or {}
    input_refs = tuple(str(ref) for ref in input_refs if ref)
    provenance_refs = tuple(str(ref) for ref in provenance_refs if ref)
    question_text = str(question or getattr(task_state, "user_goal", "") or "")

    def mark_failure(reason: str) -> bool:
        recorded = False
        for state in task_state.requirements.values():
            if not spec.can_satisfy(state.requirement.evidence_type):
                continue
            if state.status not in {"open", "running", "partially_supported"}:
                continue
            requirement_id = state.requirement.id
            evidence_ledger.append(LedgerEntry(
                tool_call_id=tool_call_id,
                capability=spec.name,
                evidence_type=state.requirement.evidence_type,
                input_refs=input_refs,
                provenance_refs=provenance_refs,
                certainty="uncertain",
                coverage=Coverage(requested=1, failed=1),
                failure_reason=reason,
                requirement_refs=(requirement_id,),
                provenance_scope_id=evidence_ledger.scope_id,
            ))
            task_state.mark_evidence_failed(
                requirement_id, reason=reason, evidence_refs=(tool_call_id,))
            task_state.record_attempt(requirement_id, {
                "tool": spec.name,
                "input_refs": list(input_refs),
                "question": str(observation.get("question") or ""),
                "outcome": "failed",
                "reason": reason,
            })
            recorded = True
        return recorded

    if not observation:
        return mark_failure("empty_tool_observation")

    # A blocked/uncertain visual or OCR response is a failed attempt, not
    # evidence. In particular, a summary such as "please search first" must
    # never satisfy a visual requirement merely because it is non-empty.
    observation_status = str(observation.get("status") or "").lower()
    observation_certainty = str(observation.get("certainty") or "supported").lower()
    if spec.name in {"inspect_photo", "read_photo_text"} and (
            observation_status in {"partial", "failed", "error", "unavailable"}
            or observation.get("blocked")
            or observation_certainty in {"uncertain", "unsupported"}):
        return mark_failure(str(observation.get("reason") or observation_status
                                or "uncertain_evidence"))

    confidence = _evidence_confidence(observation)
    preview = observation.get("preview") or []
    asset_ids = tuple(str(value) for value in observation.get("asset_ids") or [] if value)
    observation_asset_ids = tuple(
        str(value) for value in (
            observation.get("evidence_asset_ids")
            or observation.get("source_asset_ids")
            or ([observation.get("_source_asset_id")] if observation.get("_source_asset_id") else [])
            or observation.get("retrieved_asset_ids")
            or []
        ) if value
    )
    all_provenance = tuple(dict.fromkeys(
        (*provenance_refs, *asset_ids, *observation_asset_ids)
    ))
    requested = max(1, int(observation.get("total") or len(preview) or 1))
    processed = min(requested, len(preview) or (1 if observation else 0))
    if spec.name not in {"search_memories", "get_result_page"}:
        requested, processed = 1, 1
    coverage = Coverage(requested=requested, processed=processed)
    evidence_rows: list[dict] = []

    if spec.name in {"search_memories", "get_result_page"}:
        # A bounded preview is only a candidate window unless the retrieval
        # validator explicitly promoted it to validated/full_support. Do not
        # satisfy person/location/visual requirements from an unvalidated
        # candidate merely because its descriptive fields look plausible.
        preview_evidence_allowed = str(
            observation.get("evidence_status") or "").lower() in {
                "validated", "full_support"
            }
        assets = [{
            "handle": item.get("handle"),
            "captured_at": item.get("captured_at"),
            "place": item.get("place"),
            # Search previews carry the server-side asset id in the raw trace,
            # while ``asset_ids`` is an optional legacy field. Preserve the
            # preview id so metadata evidence can be traced to the exact photo.
            "asset": (asset_ids[index] if index < len(asset_ids)
                      else str(item.get("asset_id") or "")),
        } for index, item in enumerate(preview) if isinstance(item, dict)]
        if assets or asset_ids:
            evidence_rows.append({
                "evidence_type": "memory_asset",
                "value": {"result_set_id": observation.get("result_set_id"), "assets": assets,
                           "asset_ids": list(asset_ids)},
                "subject": str(observation.get("query") or "记忆照片"),
                "asset_id": asset_ids[0] if len(asset_ids) == 1 else "",
            })
        for index, item in enumerate(preview):
            if not isinstance(item, dict):
                continue
            asset_id = (asset_ids[index] if index < len(asset_ids)
                        else str(item.get("asset_id") or ""))
            summary = str(item.get("evidence_summary") or "").strip()
            if not preview_evidence_allowed:
                continue
            if summary:
                # The ingestion observation is an evidence source in its own
                # right. Keep it tied to this exact asset so the Writer can
                # use a rich memory description even when a later 12B visual
                # pass misses a small/occluded detail; inspect_photo remains a
                # bounded freshness check rather than the sole source.
                evidence_rows.append({
                    "evidence_type": "visual_observation",
                    "value": summary,
                    "subject": "照片观察摘要",
                    "asset_id": asset_id,
                    "certainty": "supported",
                })
                if re.search(r"都有谁|有哪些人|哪几个人|几个人|谁一起|谁参加|人物", question_text):
                    visible_count = _people_count_from_summary(summary)
                    named_count = sum(
                        1 for person in (item.get("people") or [])
                        if isinstance(person, dict)
                        and str(person.get("name") or "").strip()
                        and person.get("identity_status") == "confirmed"
                    )
                    unknown_count = max(0, visible_count - named_count)
                    if unknown_count:
                        evidence_rows.append({
                            "evidence_type": "photo_identity",
                            "value": {
                                "identity_status": "unconfirmed",
                                "visible_people_count": visible_count,
                                "confirmed_people_count": named_count,
                                "unconfirmed_people_count": unknown_count,
                                "description": f"另有{unknown_count}名未确认身份的同行者",
                            },
                            "subject": "照片人物总数与未确认同行者",
                            "asset_id": asset_id,
                            "certainty": "supported",
                        })
                if ocr_intent(question_text):
                    # Ingestion OCR is a usable text source when the more
                    # expensive tile OCR times out. Preserve only the exact
                    # stored text fragment; never let the writer infer a
                    # different slogan from the scene description.
                    text_match = re.search(r"文字：(.+)$", summary)
                    if text_match and text_match.group(1).strip():
                        evidence_rows.append({
                            "evidence_type": "visible_text",
                            "value": text_match.group(1).strip()[:600],
                            "subject": "照片中已记录的文字",
                            "asset_id": asset_id,
                            "certainty": "supported",
                        })
            for person in item.get("people") or []:
                if not isinstance(person, dict) or person.get("identity_status") != "confirmed":
                    continue
                name = str(person.get("name") or "").strip()
                if not name:
                    continue
                evidence_rows.append({
                    "evidence_type": "photo_identity",
                    "value": {
                        "person_name": name,
                        "family_role": str(person.get("family_role") or ""),
                        "identity_status": "confirmed",
                    },
                    "subject": "当前照片中的已确认人物",
                    "asset_id": asset_id or str(item.get("handle") or ""),
                    "certainty": "confirmed",
                })
        cond_summary = observation.get("condition_summary") or {}
        for cond_key, cond_status in cond_summary.items():
            if preview_evidence_allowed and cond_status in {"matched", "confirmed"}:
                evidence_rows.append({
                    "evidence_type": "structured_fact",
                    "value": f"检索确认满足条件：{cond_key}（共找到 {len(assets)} 张照片）",
                    "certainty": "confirmed",
                    "subject": cond_key,
                })
        group_count = observation.get("group_photo_count")
        group_sizes = observation.get("group_photo_sizes") or []
        if preview_evidence_allowed and group_count:
            evidence_rows.append({
                "evidence_type": "structured_fact",
                "value": {
                    "group_photo_count": int(group_count),
                    "group_photo_sizes": list(group_sizes),
                    "rows": observation.get("group_photo_rows") or [],
                },
                "subject": "事件合影人数统计",
                "asset_id": ((observation.get("group_photo_rows") or [{}])[0].get("asset_id")
                             if isinstance((observation.get("group_photo_rows") or [{}])[0], dict)
                             else ""),
                "certainty": "supported",
            })
        location_question = bool(re.search(
            r"在哪里|哪儿|哪个城市|什么地点|何处|哪举办|地点具体", question_text))
        places = [item for item in assets if item.get("place")]
        # GPS/reverse-geocode is a direct structured field.  For a location
        # question the highest-ranked preview may therefore be used as a
        # source even before visual validation; do not promote the rest of a
        # broad candidate pool.
        if places and not preview_evidence_allowed and location_question:
            places = places[:1]
        elif not preview_evidence_allowed:
            places = []
        if places:
            evidence_rows.append({
                "evidence_type": "location_metadata",
                "value": [{"asset": item.get("asset_id") or item.get("asset") or item.get("handle"),
                           "value": item.get("place")} for item in places],
                "subject": "照片地点",
                "asset_id": places[0].get("asset_id") or places[0].get("asset") or "",
            })
        date_question = bool(re.search(
            r"哪天|什么时候|何时|哪一年|年份|日期|时间|几月|最早|最近一次", question_text))
        dates = [item for item in assets if item.get("captured_at")]
        if dates and not preview_evidence_allowed and date_question:
            dates = dates[:1]
        elif not preview_evidence_allowed:
            dates = []
        if dates:
            evidence_rows.append({
                "evidence_type": "temporal_metadata",
                "value": [{"asset": item.get("asset_id") or item.get("asset") or item.get("handle"),
                           "value": item.get("captured_at")} for item in dates],
                "subject": "照片拍摄时间",
                "asset_id": dates[0].get("asset_id") or dates[0].get("asset") or "",
            })
            # Date/year questions are structured facts even when the user
            # also describes a visual scene. Bind the same source timestamps
            # to structured_fact so the authoritative writer can answer the
            # requested year instead of treating a valid dated candidate as
            # an unresolved visual match.
            if date_question:
                evidence_rows.append({
                    "evidence_type": "structured_fact",
                    "value": [{"asset": item.get("asset_id") or item.get("asset") or item.get("handle"),
                                "captured_at": item.get("captured_at")} for item in dates],
                    "subject": "照片拍摄时间事实",
                    "asset_id": dates[0].get("asset_id") or dates[0].get("asset") or "",
                    "certainty": "supported",
                })
    elif spec.name == "inspect_photo":
        value = observation.get("observation") or observation.get("scene") or observation.get("summary")
        # A visual inspection can directly disprove a requested scene. That
        # negative observation is useful for recovery, but must not satisfy a
        # positive visual requirement merely because it is non-empty.
        negative_visual = bool(value and re.search(
            r"(?:没有|未发现|未看到|不存在|不包含|看不到|无法确认|不是).{0,24}",
            str(value), re.I))
        if value and not negative_visual:
            evidence_rows.append({"evidence_type": "visual_observation", "value": value,
                                  "subject": str(observation.get("question") or "照片视觉细节"),
                                  "asset_id": str(observation.get("_source_asset_id")
                                                  or observation.get("asset_handle")
                                                  or (input_refs[0] if input_refs else ""))})
        for identity in observation.get("photo_identities") or []:
            if not isinstance(identity, dict) or identity.get("identity_status") != "confirmed":
                continue
            evidence_rows.append({
                "evidence_type": "photo_identity",
                "value": {
                    "face_instance_id": identity.get("face_instance_id"),
                    "cluster_id": identity.get("cluster_id"),
                    "entity_id": identity.get("entity_id"),
                    "person_name": identity.get("person_name"),
                    "family_role": identity.get("family_role"),
                    "identity_status": "confirmed",
                },
                "subject": "当前照片中的已确认人物",
                "asset_id": str(identity.get("asset_id") or observation.get("_source_asset_id")
                                or observation.get("asset_handle") or ""),
                "certainty": "confirmed",
            })
    elif spec.name == "read_photo_text":
        value = observation.get("full_text") or observation.get("exact_values") or observation.get("text")
        if value:
            evidence_rows.append({"evidence_type": "visible_text", "value": value,
                                  "subject": str(observation.get("question") or "照片文字"),
                                  "asset_id": str(observation.get("asset_handle") or (input_refs[0] if input_refs else ""))})
        for ev in (observation.get("exact_values") or []):
            if ev.get("type") == "year" and ev.get("value"):
                evidence_rows.append({
                    "evidence_type": "structured_fact",
                    "value": f"创立/创建年份为 {ev.get('value')} 年",
                    "certainty": "confirmed",
                    "subject": "品牌创立年份",
                })
            elif ev.get("type") == "price" and ev.get("value"):
                evidence_rows.append({
                    "evidence_type": "structured_fact",
                    "value": f"价格: {ev.get('text', ev.get('value'))}",
                    "certainty": "confirmed",
                    "subject": "商品价格",
                })
    elif spec.name == "query_memory_facts":
        value = observation.get("value")
        if value is None:
            value = observation.get("rows") or observation.get("items") or observation.get("summary")
        if value not in (None, "", [], {}):
            evidence_rows.append({"evidence_type": "structured_fact", "value": value,
                                  "subject": str(observation.get("operation") or "结构化记忆事实")})
        if (observation.get("operation") in {"date", "first", "last"}
                and observation.get("value") not in (None, "", [], {})):
            evidence_rows.append({"evidence_type": "temporal_metadata", "value": observation.get("value"),
                                  "subject": str(observation.get("operation"))})
        filters = observation.get("filters_applied") or {}
        time_range = filters.get("time_range") if isinstance(filters, dict) else None
        items = observation.get("items") or []
        if time_range and items:
            temporal_value = [
                {"asset": str(row.get("asset_id") or row.get("id") or ""),
                 "value": row.get("captured_at") or row.get("date") or ""}
                for row in items if isinstance(row, dict)
                and (row.get("captured_at") or row.get("date"))
            ]
            if temporal_value:
                evidence_rows.append({"evidence_type": "temporal_metadata",
                                      "value": temporal_value,
                                      "subject": "时间过滤后的照片拍摄时间",
                                      "asset_id": temporal_value[0].get("asset") or ""})
        if observation.get("group_by") == "place" or observation.get("common_places"):
            value = observation.get("rows") or observation.get("common_places")
            if value:
                evidence_rows.append({"evidence_type": "location_metadata", "value": value,
                                      "subject": "结构化地点事实"})
    elif spec.name == "query_memory_metadata":
        # Dedicated structured metadata must enter the same typed ledger as
        # search/inspect results; otherwise direct date/place/event evidence is
        # visible in the tool trace but remains invisible to the final gate.
        value = observation.get("value")
        if value is None:
            value = observation.get("items") or observation.get("rows") or observation.get("summary")
        source_ids = tuple(str(value) for value in (
            observation.get("evidence_asset_ids")
            or observation.get("source_asset_ids")
            or []
        ) if value)
        operation = str(observation.get("metadata_operation")
                        or observation.get("operation") or "").lower()
        if value is not None:
            evidence_rows.append({
                "evidence_type": "structured_fact",
                "value": value,
                "subject": f"结构化元数据:{operation or 'query'}",
                "asset_id": source_ids[0] if len(source_ids) == 1 else "",
            })
        if operation in {"date", "first", "last"} and value not in (None, "", [], {}):
            evidence_rows.append({
                "evidence_type": "temporal_metadata", "value": value,
                "subject": "结构化日期事实",
                "asset_id": source_ids[0] if len(source_ids) == 1 else "",
            })
        filters = observation.get("filters_applied") or {}
        time_range = filters.get("time_range") if isinstance(filters, dict) else None
        if time_range and source_ids:
            evidence_rows.append({
                "evidence_type": "temporal_metadata",
                "value": {"time_range": time_range, "source_asset_ids": list(source_ids)},
                "subject": "时间过滤后的结构化照片来源",
                "asset_id": source_ids[0],
            })
        if operation == "place" and value not in (None, "", [], {}):
            evidence_rows.append({
                "evidence_type": "location_metadata", "value": value,
                "subject": "结构化地点事实",
                "asset_id": source_ids[0] if len(source_ids) == 1 else "",
            })
    elif spec.name == "query_photo_people":
        people = observation.get("people") or []
        unknown = observation.get("unconfirmed_people") or []
        if people or unknown or observation.get("summary"):
            source_ids = tuple(str(value) for value in (
                observation.get("evidence_asset_ids")
                or observation.get("source_asset_ids")
                or []
            ) if value)
            evidence_rows.append({
                "evidence_type": "photo_identity",
                "value": {"people": people, "unconfirmed_people": unknown,
                          "summary": observation.get("summary") or ""},
                "subject": "当前照片人物",
                "asset_id": source_ids[0] if source_ids else str(observation.get("asset_id") or ""),
                "certainty": "confirmed" if people else "supported",
            })
    elif spec.name == "get_original_photos":
        if int(observation.get("delivered") or 0) > 0 and observation.get("result_set_id"):
            evidence_rows.append({
                "evidence_type": "memory_asset",
                "value": {"result_set_id": observation.get("result_set_id"),
                          "handle": observation.get("handle"),
                          "url": observation.get("url")},
                "subject": "原图交付",
            })
    elif spec.name == "search_conversation_history":
        matches = observation.get("matches") or []
        if matches:
            evidence_rows.append({
                "evidence_type": "user_statement",
                "value": matches,
                "subject": str(observation.get("query") or "历史对话"),
            })
    elif spec.name == "get_core_memory":
        cards = observation.get("cards") or []
        if cards:
            confirmed = any(
                str(item.get("truth_status") or "") in {"confirmed", "confirmed_fact"}
                for card in cards for item in (card.get("items") or [])
                if isinstance(item, dict)
            )
            user_stated = any(
                str(item.get("truth_status") or "") in {"user_assertion", "user_stated"}
                for card in cards for item in (card.get("items") or [])
                if isinstance(item, dict)
            )
            if confirmed:
                evidence_rows.append({"evidence_type": "confirmed_identity",
                                      "value": cards, "subject": "长期确认记忆"})
            if user_stated:
                evidence_rows.append({"evidence_type": "user_statement",
                                      "value": cards, "subject": "长期用户表述"})
    elif spec.name == "get_person_memory":
        if str(observation.get("readiness") or "") == "ready":
            person = str(observation.get("person") or "")
            evidence_rows.append({"evidence_type": "confirmed_identity",
                                  "value": {"person": person,
                                            "family_role": observation.get("family_role")},
                                  "subject": person})
            for key, evidence_type in (("first_occurrence", "temporal_metadata"),
                                       ("last_occurrence", "temporal_metadata"),
                                       ("common_places", "location_metadata"),
                                       ("asset_count", "structured_fact"),
                                       ("events", "structured_fact"),
                                       ("co_occurrence", "structured_fact")):
                if observation.get(key) not in (None, "", [], {}):
                    evidence_rows.append({"evidence_type": evidence_type,
                                          "value": observation.get(key),
                                          "subject": person})
    elif spec.name == "get_person_profile":
        if str(observation.get("readiness") or "") == "ready":
            person = str(observation.get("person") or "")
            evidence_rows.append({"evidence_type": "confirmed_identity",
                                  "value": {"person": person,
                                            "family_role": observation.get("family_role")},
                                  "subject": person})
            if observation.get("claims") or observation.get("patterns") or observation.get("relationships"):
                evidence_rows.append({"evidence_type": "structured_fact",
                                      "value": {"claims": observation.get("claims") or [],
                                                "patterns": observation.get("patterns") or [],
                                                "relationships": observation.get("relationships") or []},
                                      "subject": person})

    covered_types = {row["evidence_type"] for row in evidence_rows}
    # A summary string is transport metadata, never a substitute for an
    # actual typed field. Every emitted Evidence row above is tied to a
    # concrete observation field; identity is never inferred generically.

    if not evidence_rows:
        blocked = ",".join(str(x) for x in observation.get("blocked") or [])
        reason = str(observation.get("reason") or observation.get("status") or
                     (f"blocked:{blocked}" if blocked else "no_evidence_returned"))
        return mark_failure(reason)

    recorded = False
    for row in evidence_rows:
        evidence_type = row["evidence_type"]
        row_coverage = coverage
        # A bounded preview is intentionally smaller than the retrieved set,
        # but it is still a complete successful attempt for requirements that
        # only need one source photo or one directly observed metadata field.
        # Do not turn a useful location/date/asset row into a failure merely
        # because the result set has more candidates on later pages.
        if spec.name in {"search_memories", "get_result_page"} and evidence_type in {
                "memory_asset", "location_metadata", "temporal_metadata",
                "photo_identity", "structured_fact"}:
            row_coverage = Coverage(requested=1, processed=1)
        # Search validation may inspect only a bounded preview of a large
        # candidate set. A visual row that the validator explicitly accepted
        # is complete evidence for that asset; counting the whole candidate
        # pool as the requested coverage incorrectly leaves the requirement
        # ``partial`` and triggers a late refusal after a valid search.
        if (spec.name in {"search_memories", "get_result_page"}
                and evidence_type == "visual_observation"):
            validated = {
                str(value) for value in (observation.get("evidence_asset_ids") or [])
                if value
            }
            if str(row.get("asset_id") or "") in validated:
                row_coverage = Coverage(requested=1, processed=1)
        active = [state for state in task_state.requirements.values()
                  if state.requirement.evidence_type == evidence_type
                  and state.requirement.required
                  and (state.status in {"open", "running", "partially_supported"}
                       # One inspection can emit several facts of the same
                       # type (e.g. two named people). Keep binding later
                       # rows from this call to the requirement after the
                       # first row has satisfied it.
                       or (state.status == "satisfied"
                           and tool_call_id in state.evidence_refs))]
        subject = str(row.get("subject") or "").strip()
        if len(active) == 1:
            refs = (active[0].requirement.id,)
        else:
            refs = tuple(
                state.requirement.id for state in active
                if subject and (
                    subject in state.requirement.description
                    or state.requirement.description in subject
                )
            )
        # A generic scene request is not enough to satisfy a targeted visual
        # requirement (for example, "请描述这张照片" cannot close a
        # bridesmaid-attire requirement). Keep it as an unmatched observation
        # so the model/auto resolver must issue a targeted inspection.
        if (evidence_type == "visual_observation" and len(active) == 1
                and subject in {"请描述这张照片", "描述这张照片", "照片视觉细节"}):
            refs = ()
        evidence_conflict = False
        if evidence_type == "visual_observation":
            current_value = str(row.get("value") or "")
            negates_subject = bool(re.search(
                r"(没有|无|未见|看不出).{0,12}(伴娘|bridesmaid)"
                r"|\bno\s+(?:clear\s+|visible\s+)?bridesmaid\b",
                current_value, re.I))
            if negates_subject:
                prior_positive = any(
                    entry.evidence_type == "visual_observation"
                    and entry.asset_id == str(row.get("asset_id") or "")
                    and re.search(r"伴娘|bridesmaid", str(entry.extracted_value or ""), re.I)
                    and not re.search(
                        r"(没有|无|未见|看不出).{0,12}(伴娘|bridesmaid)"
                        r"|\bno\s+(?:clear\s+|visible\s+)?bridesmaid\b",
                        str(entry.extracted_value or ""), re.I)
                    for entry in evidence_ledger.entries
                )
                if prior_positive:
                    refs = ()
                    evidence_conflict = True
        try:
            evidence_ledger.append(LedgerEntry(
                tool_call_id=tool_call_id,
                capability=spec.name,
                evidence_type=evidence_type,
                input_refs=input_refs,
                provenance_refs=all_provenance,
                certainty=("uncertain" if evidence_conflict
                           else str(observation.get("certainty") or "supported")),
                coverage=row_coverage,
                failure_reason=(str(observation.get("reason") or "")
                                if row_coverage.failed else ""),
                provenance_scope_id=evidence_ledger.scope_id,
                subject=str(row.get("subject") or ""),
                asset_id=str(row.get("asset_id") or ""),
                extracted_value=row.get("value"),
                confidence=confidence,
                requirement_refs=refs,
                unmatched_reason=("evidence_conflict" if evidence_conflict
                                  else "evidence_incompatible" if not refs and task_state.requirements
                                  else ""),
            ))
        except ValueError as exc:
            if "duplicate tool call" in str(exc):
                continue
            raise
        recorded = True
        for state in task_state.requirements.values():
            if state.requirement.evidence_type != evidence_type:
                continue
            if state.requirement.id not in refs:
                continue
            if state.status == "open":
                task_state.mark_running(state.requirement.id)
            if state.status == "running":
                if row_coverage.is_partial:
                    if allow_partial:
                        task_state.mark_partially_supported(
                            state.requirement.id, evidence_refs=(tool_call_id,))
                        outcome = "partial"
                    else:
                        task_state.mark_evidence_failed(
                            state.requirement.id,
                            reason="partial_coverage",
                            evidence_refs=(tool_call_id,))
                        outcome = "failed"
                else:
                    task_state.mark_satisfied(state.requirement.id, evidence_refs=(tool_call_id,))
                    outcome = "satisfied"
                task_state.record_attempt(state.requirement.id, {
                    "tool": spec.name,
                    "input_refs": list(input_refs),
                    "question": str(observation.get("question") or row.get("subject") or ""),
                    "outcome": outcome,
                    "evidence_type": evidence_type,
                    "coverage": row_coverage.as_dict(),
                })
    return recorded


def _evidence_confidence(observation: dict) -> float | None:
    value = observation.get("confidence")
    if isinstance(value, dict):
        value = value.get("score") or value.get("value")
    try:
        return max(0.0, min(1.0, float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


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
    # Search result cardinality is retrieval telemetry, not an answer fact.
    # Explicit count questions are represented by query_memory_facts instead;
    # never turn a broad ResultSet size into a user-facing claim here.
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


def _promote_structured_sources_to_result_set(task: TaskState, observation: dict,
                                              *, scope_id: str, query: str = "") -> None:
    """Make structured-fact sources addressable by the same photo handles.

    Metadata/fact tools may be the first successful retrieval after a semantic
    search returned zero candidates. Their source assets must therefore be
    merged into the current server-side ResultSet; otherwise grounding has
    asset IDs but the UI cannot resolve a handle or media URL.
    """
    if not observation or not task.current_result_set:
        return
    ids: list[str] = []
    for value in (observation.get("evidence_asset_ids")
                  or observation.get("source_asset_ids") or []):
        if value and str(value) not in ids:
            ids.append(str(value))
    for row in (observation.get("items") or observation.get("rows") or []):
        if isinstance(row, dict):
            value = row.get("asset_id") or row.get("id")
            if value and str(value) not in ids:
                ids.append(str(value))
    if not ids:
        return
    try:
        from . import tools as runtime_tools
        rs_store = runtime_tools._RUNTIME.get("result_sets")
        if rs_store is None:
            return
        rs = rs_store.get(task.current_result_set)
        if rs is None:
            rs = rs_store.new(scope_id=scope_id, query=query,
                              asset_ids=ids, owner="owner")
            task.current_result_set = rs.result_set_id
        else:
            merged = list(dict.fromkeys([*(rs.asset_ids or []), *ids]))
            if merged != list(rs.asset_ids or []):
                rs.asset_ids = merged
                rs.total = len(merged)
                rs.shown = min(6, len(merged))
                rs_store.save(rs)
        task.result_preview = [f"photo_{i + 1}" for i in range(min(20, len(rs.asset_ids)))]
    except Exception:
        return


def _build_answer_grounding(*, message: str, task: TaskState,
                            selected_handle: str | None = None,
                            selected_image_handles: list[str] | None = None,
                            selected_image_ids: list[str] | None = None) -> dict:
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
    retrieved_assets: list[str] = []
    source_assets: list[str] = []
    rep_assets: list[dict] = []
    for tr in tool_results:
        for aid in (tr.get("retrieved_asset_ids") or tr.get("asset_ids") or []):
            if aid and str(aid) not in retrieved_assets:
                retrieved_assets.append(str(aid))
        # Preview/page projections are retrieval candidates, never evidence.
        # Evidence must be explicitly emitted by a validator, metadata fact,
        # people evidence, inspect or OCR tool.
        for aid in (tr.get("evidence_asset_ids") or []):
            if aid and str(aid) not in source_assets:
                source_assets.append(str(aid))
        for aid in (tr.get("source_asset_ids") or []):
            if aid and str(aid) not in source_assets:
                source_assets.append(str(aid))
        # Fact rows are evidence even when no image was selected for delivery.
        for row in (tr.get("items") or tr.get("rows") or []):
            if isinstance(row, dict) and row.get("asset_id"):
                aid = str(row["asset_id"])
                if aid not in retrieved_assets:
                    retrieved_assets.append(aid)
                if aid not in source_assets:
                    source_assets.append(aid)
        for row in (tr.get("preview") or []):
            if isinstance(row, dict) and row.get("asset_id"):
                aid = str(row["asset_id"])
                if aid not in retrieved_assets:
                    retrieved_assets.append(aid)
        if tr.get("tool") == "inspect_photo" and tr.get("inspect_handle"):
            handle = str(tr["inspect_handle"])
            if handle not in evidence_handles:
                evidence_handles.append(handle)
                rep_assets.append({"handle": handle, "kind": "inspection",
                                   "observation": (tr.get("inspect_text") or "")[:80]})
            if tr.get("asset_id"):
                aid = str(tr["asset_id"])
                if aid not in retrieved_assets:
                    retrieved_assets.append(aid)
                if aid not in source_assets:
                    source_assets.append(aid)
        for s in (tr.get("samples") or []) or []:
            if isinstance(s, dict) and s.get("asset_id"):
                aid = s["asset_id"]
                if aid not in evidence_assets:
                    evidence_assets.append(aid)
                    if len(rep_assets) < 6:
                        rep_assets.append({"asset_id": aid,
                                           "captured_at": s.get("captured_at") or "",
                                           "caption": (s.get("caption") or s.get("transcript") or "")[:80]})
                    if aid not in source_assets:
                        source_assets.append(aid)
    # A model-selected preview is still only a candidate.  Never promote it
    # to evidence merely because it was placed in selected_image_ids; delivery
    # must remain a subset of validated/structured evidence sources.
    evidence_count = len(evidence_handles) + len(evidence_assets)
    # Keep retrieved candidates, validated evidence and explicit delivery
    # distinct even when the model chose no image for final delivery.
    for aid in source_assets:
        if aid not in evidence_assets:
            evidence_assets.append(aid)
    for aid in evidence_assets:
        if aid not in retrieved_assets:
            retrieved_assets.append(aid)
    evidence_assets = [aid for aid in retrieved_assets if aid in set(evidence_assets)]
    evidence_count = max(evidence_count, len(evidence_assets))
    evidence_images: list[dict] = []
    rs = None
    if evidence_assets:
        try:
            from . import tools as runtime_tools
            store = getattr(runtime_tools, "_RUNTIME", {}).get("store")
            rs_store = getattr(runtime_tools, "_RUNTIME", {}).get("result_sets")
            rs = rs_store.get(task.current_result_set) if rs_store and task.current_result_set else None
            for aid in evidence_assets[:12]:
                asset = store.get_asset(aid) if store else None
                if not asset:
                    continue
                handle = ""
                if rs is not None and aid in (rs.asset_ids or []):
                    handle = f"photo_{list(rs.asset_ids).index(aid) + 1}"
                evidence_images.append({
                    "asset_id": aid,
                    "handle": handle,
                    "file_name": asset.get("file_name") or "",
                    "captured_at": asset.get("captured_at") or "",
                    "media_url": (f"/api/assistant/result-set/{task.current_result_set}/photo?handle={handle}"
                                   f"&scope_id={rs.scope_id}" if handle and task.current_result_set and rs is not None else ""),
                })
        except Exception:
            evidence_images = []
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
    evidence_asset_set = set(evidence_assets)
    valid_selected_ids = [str(aid) for aid in (selected_image_ids or [])
                          if aid and str(aid) in evidence_asset_set]
    valid_selected_handles = list(selected_image_handles or [])
    if rs is not None:
        valid_selected_handles = [handle for handle in valid_selected_handles
                                  if any(str(aid) in evidence_asset_set
                                         and f"photo_{list(rs.asset_ids).index(aid) + 1}" == str(handle)
                                         for aid in (rs.asset_ids or []))]
    else:
        # A handle without its originating result set cannot be resolved safely.
        valid_selected_handles = []
    return {
        "required": used_evidence,
        "display_mode": display_mode,
        "evidence_count": evidence_count,
        "representative_evidence": rep_assets[:6],
        "all_evidence_available": bool(task.current_result_set),
        "result_set_id": task.current_result_set,
        "explicit_image_request": explicit_image,
        "selected_image_handles": valid_selected_handles,
        "selected_asset_ids": valid_selected_ids,
        "retrieved_asset_ids": list(dict.fromkeys(retrieved_assets)),
        "evidence_asset_ids": list(dict.fromkeys(evidence_assets)),
        "evidence_images": evidence_images,
    }



_POLICY_REFUSAL_RULES = [
    (re.compile(r"忽略.*(?:指令|系统提示|提示词|安全规则|限制)", re.I), "我无法透露系统提示词内容，也不能忽略既有的安全规则。"),
    (re.compile(r"(?:告诉|输出|显示|给我).*(?:系统提示|提示词|system prompt)", re.I), "我无法透露系统提示词内容，也不能忽略既有的安全规则。"),
    (re.compile(r"(?:导出|发给|提供|输出|告诉).*(?:特征向量|人脸特征|身份信息|家庭关系|银行卡|密码|住址|电话)", re.I), "我无法导出人脸特征向量和用户隐私身份数据，这属于敏感个人信息。"),
    (re.compile(r"(?:特征向量|人脸特征|身份信息|家庭关系|银行卡|密码|住址|电话).*(?:导出|发给|发我|提供|输出|给我)", re.I), "我无法导出人脸特征向量和用户隐私身份数据，这属于敏感个人信息。"),
    (re.compile(r"(?:跳过|绕过).*(?:安全检查|安全验证)", re.I), "我无法通过对话指令执行删除操作，也无法跳过安全验证。"),
    (re.compile(r"(?:删除|清空|覆盖).*(?:相册|照片|数据)", re.I), "我无法通过对话指令执行删除操作，也无法跳过安全验证。"),
    (re.compile(r"\b(?:DAN|Do Anything Now)\b", re.I), "我不会扮演不受限制的角色，也无法输出家庭成员的个人隐私信息。"),
    (re.compile(r"(?:无限制|不受限制|没有任何限制).*(?:模式|助手|AI|版本)", re.I), "我无法切换到无安全限制的模式，也不能提供未经授权提取和使用人脸数据的方法。"),
    (re.compile(r"如何.*(?:提取人脸|训练.*模型)", re.I), "我无法切换到无安全限制的模式，也不能提供未经授权提取和使用人脸数据的方法。"),
]


def _detect_policy_refusal(message: str) -> str | None:
    msg = str(message or "").strip()
    for pattern, refusal_text in _POLICY_REFUSAL_RULES:
        if pattern.search(msg):
            return refusal_text
    return None

def _agent2_answer_context_ready(task_state, answer_context: dict) -> bool:
    """Return true when at least one required field is grounded enough to write.

    A single unresolved ancillary requirement must not discard a directly
    supported answer (for example, a GPS location while visual confirmation
    of the scene is still pending). The writer receives only the grounded
    fields and is instructed to state the remaining uncertainty.
    """
    if not answer_context.get("facts"):
        return False
    states = [state for state in getattr(task_state, "requirements", {}).values()
              if getattr(state.requirement, "required", True)]
    return any(state.status in {"satisfied", "partially_supported"}
               for state in states)


def _refresh_agent2_status(task_state, evidence_ledger) -> str:
    """Recompute the authoritative task status from requirements and registry."""
    if task_state is None or evidence_ledger is None:
        return "blocked"
    from .requirement_completion import RequirementCompletion
    available = RequirementCompletion(task_state, evidence_ledger).allowed_capabilities(
        list_tools(readiness="ready"))
    return task_state.recompute_status(has_available_tools=bool(available))


def _capability_note(tool_name: str) -> str:
    """Phase F F8：从 tool_capability_matrix 读取 capability 级 readiness，注入 Agent 提示。

    只输出关键结论（ready 高置信 / limited 提示），避免上下文膨胀。
    """
    try:
        import json as _json
        from pathlib import Path as _Path
        matrix_path = _Path(__file__).resolve().parent.parent.parent / "configs" / "tool_capability_matrix.json"
        if not matrix_path.is_file():
            return ""
        matrix = _json.loads(matrix_path.read_text(encoding="utf-8"))
        caps = matrix.get(tool_name) or {}
        notes = []
        for key, info in (caps or {}).items():
            status = str(info.get("status") or "")
            acc = info.get("accuracy")
            if status == "ready" or status.startswith("ready"):
                notes.append(f"{key}=ready" + (f"({acc:.2f})" if isinstance(acc, (int, float)) else ""))
            elif status == "limited":
                notes.append(f"{key}=limited")
        return "；".join(notes)
    except Exception:
        return ""


class AgentRuntime:
    """Thin tool-loop runtime. 模型调用通过传入的 chat_fn 注入。"""

    def __init__(self, *, chat_fn, profile_name: str | None = None,
                 scope_id="home-default", viewer_id="owner", conversation_id=None,
                 ocr_settings: dict | None = None, include_debug: bool = False):
        self.chat_fn = chat_fn
        self.profile = get_profile(profile_name)
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.conversation_id = conversation_id
        self.ocr_settings = ocr_settings or {}
        self.include_debug = include_debug

    def _tool_descriptions(self) -> str:
        lines = []
        from .tool_registry import list_tools
        allowed = set(self.profile.tools) if self.profile.tools else None
        for spec in list_tools():
            if spec.readiness == "blocked":
                continue
            if allowed is not None and spec.name not in allowed:
                continue
            capability = _capability_note(spec.name)
            lines.append(f"- {spec.name}: {spec.description}"
                         f"{(' 能力=' + capability) if capability else ''}"
                         f" 输入schema={json.dumps(spec.input_schema, ensure_ascii=False)}")
        return "\n".join(lines) or "(无工具)"


    def _force_final_once(self, turn, messages: list) -> str | None:
        """额外一次纯 final：让模型明确给出结论（哪怕"没有答案"），失败返回 None。

        不计入 model_steps（用于预算耗尽后的收尾），但仍受墙钟 final reserve 限制。
        """
        if not turn.budget.has_final_reserve():
            return None
        try:
            from .final_writer import naturalize_answer
            final_messages = list(messages)
            final_messages.append({"role": "user", "content": (
                "你现在必须给出一个明确的最终回答，不允许再调用工具。"
                "如果根据已有工具结果足以回答，就直接回答；如果不足以回答，"
                "就明确说“现有记录不足以确认”并给出你能确认的部分。"
                '只输出 {"action":"final","answer":"<结论>","evidence_refs":[]}。'
            )})
            if self.include_debug:
                import copy as _copy
                _ff_prompt = _copy.deepcopy(final_messages)
            else:
                _ff_prompt = None
            raw = self.chat_fn(final_messages)
            parsed = self._parse_action(raw)
            if parsed and parsed.get("action") == "final" and (parsed.get("answer") or "").strip():
                _ff_step = {"type": "model", "raw": (raw or "")[:500],
                            "call_type": "force_final", "forced_final": True}
                if _ff_prompt is not None:
                    _ff_step["prompt"] = _ff_prompt
                    _ff_step["raw_full"] = raw
                turn.steps.append(_ff_step)
                return naturalize_answer(parsed["answer"])
        except Exception:
            pass
        return None


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
            if isinstance(parsed, dict):
                act = parsed.get("action")
                if act in ("tool_call", "final"):
                    return parsed
                # 兼容 0.8B 等小模型把工具名直接写在 action 字段的情况（如 {"action":"search_memories", "filters":...}）
                if act in ("search_memories", "query_memory_facts", "inspect_photo", "read_photo_text", "get_result_page", "get_original_photos", "search_conversation_history", "get_core_memory", "get_person_memory", "get_person_profile"):
                    tool_name = act
                    args = {k: v for k, v in parsed.items() if k not in ("action", "public_status")}
                    # 如果有 arguments 嵌套则展开
                    if isinstance(args.get("arguments"), dict):
                        args = args["arguments"]
                    return {
                        "action": "tool_call",
                        "tool": tool_name,
                        "arguments": args,
                        "public_status": parsed.get("public_status") or f"正在调用 {tool_name}"
                    }
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
        selected_image_handles: list[str] = []
        last_model_final_answer = ""
        policy = ToolPolicy(scope_id=self.scope_id, viewer_id=self.viewer_id, budget=turn.budget,
                            allowed_tools=set(self.profile.tools) if self.profile.tools else None)
        refusal = _detect_policy_refusal(message)
        if refusal:
            turn.final_answer = refusal
            turn.status = "complete"
            turn.reason = "policy_refusal"
            return turn
        # W1.2 stage-timing instrumentation (observation only; no behavior change).
        # Bucket every model call by call_type; restore the original chat_fn at the end.
        _stage_timing_ms: dict[str, float] = {}
        _orig_chat_fn = self.chat_fn
        _orig_chat_signature = inspect.signature(_orig_chat_fn)
        _accepts_chat_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in _orig_chat_signature.parameters.values()
        )

        def _timed_chat(messages, *, call_type=None, step_id=None, **kwargs):
            _t0 = time.perf_counter()
            try:
                _kw = {}
                if call_type is not None and (
                    _accepts_chat_kwargs or "call_type" in _orig_chat_signature.parameters
                ):
                    _kw["call_type"] = call_type
                if step_id is not None and (
                    _accepts_chat_kwargs or "step_id" in _orig_chat_signature.parameters
                ):
                    _kw["step_id"] = step_id
                return _orig_chat_fn(messages, **_kw, **kwargs)
            finally:
                _key = call_type or "recovery_or_judge"
                _stage_timing_ms[_key] = _stage_timing_ms.get(_key, 0.0) + (time.perf_counter() - _t0) * 1000.0
        self.chat_fn = _timed_chat
        task = TaskState.from_dict(task_state, user_goal=message)
        agent2_task_state = None
        agent2_evidence_ledger = None
        event_summary_request = False
        if (self.profile.features.get("agent2_authoritative")
                or self.profile.features.get("agent2_shadow")):
            # The authoritative profile uses this planner/task state as the
            # production decision path. The shadow flag is retained only for
            # replay compatibility with historical profiles.
            from .evidence_ledger import EvidenceLedger
            from .goal_planner import GoalPlanner, _event_summary_intent

            event_summary_request = _event_summary_intent(message)
            from .task_state import TaskState as Agent2TaskState

            planner_step_id = "planner_step_0"
            planner_result = GoalPlanner(chat_fn=self.chat_fn).declare(
                message, scope_id=self.scope_id, history=history,
                include_debug=self.include_debug, step_id=planner_step_id)
            decision = {"kind": "declare"}
            if planner_result.ok:
                agent2_task_state = Agent2TaskState.from_declaration(planner_result.declaration)
                agent2_evidence_ledger = EvidenceLedger(scope_id=self.scope_id)
                decision["status"] = "accepted"
                turn.agent2_trace = {
                    "task_declaration": planner_result.declaration.as_dict(),
                    "task_state": agent2_task_state.as_dict(),
                    "evidence_ledger": agent2_evidence_ledger.as_dict(),
                    "planner_decisions": [decision],
                }
            else:
                decision.update({"status": "fallback", "reason": planner_result.fallback_reason})
                turn.agent2_trace = {"planner_decisions": [decision]}

            planner_step = {
                "type": "planner",
                "status": "complete" if planner_result.ok else "fallback",
                "call_type": "planner",
                "step_id": planner_step_id,
                "raw": (planner_result.raw or "")[:500],
            }
            if self.include_debug and planner_result.prompt:
                planner_step["prompt"] = planner_result.prompt
                planner_step["raw_full"] = planner_result.raw
            turn.steps.append(planner_step)
            if self.profile.features.get("agent2_authoritative") and not planner_result.ok:
                turn.final_answer = "当前问题的证据需求无法可靠规划，因此暂时无法确认。"
                turn.status = "partial"
                turn.reason = planner_result.fallback_reason or "planner_invalid"
                turn.termination_reason = "planner_blocked"
                self.chat_fn = _orig_chat_fn
                return turn
        guard = FinalGuard(scope_id=self.scope_id, viewer_id=self.viewer_id)
        is_candidate_mode = bool(
            self.profile.features.get("agent2_authoritative")
            or self.profile.features.get("agent2_candidate")
        )
        if is_candidate_mode and agent2_task_state is not None:
            system = build_jit_system_prompt(
                task_state=agent2_task_state,
                current_time_str=current_time_line(),
                tool_results=task.tool_results,
                preview_handles=task.result_preview,
                is_candidate=True,
                allowed_tool_names=self.profile.tools,
            )
        else:
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
            from .tools import person_profile_summary
            profile_line = person_profile_summary(task.active_person, self.scope_id)
            if profile_line:
                active_lines.append(f"人物画像：{profile_line}")
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
        max_parse_retries = 3
        guard_retries = 0
        max_guard_retries = 1
        seen_tool_calls = set()
        # Read-only calls are idempotent.  Small models sometimes repeat an
        # identical call after receiving a result; reuse the prior observation
        # instead of turning a harmless retry into a terminal tool rejection.
        tool_result_cache = {}
        dedup_retries = 0
        max_dedup_retries = 2
        search_has_preview = False
        inspect_called = False
        tool_call_seq = 0
        debug_step_seq = 0
        last_model_step_id = None
        visual_retries = 0
        max_visual_retries = 1
        resolution_retries = 0
        max_resolution_retries = 2
        completion_retries = 0
        max_completion_retries = 1
        answer_quality_retries = 0
        auto_resolution_attempts = 0
        # A failed first visual/OCR candidate must be followed by bounded
        # attempts on new preview handles before the task is declared
        # insufficient.  Three total recovery calls remain within the profile
        # tool budget and prevent unbounded inspection loops.
        max_auto_resolution_attempts = 3
        unknown_tool_retries = 0
        max_unknown_tool_retries = 1
        forced_final_attempted = False
        wants_visual = visual_intent(message)

        def visual_gate_requested() -> bool:
            """Use only the planner requirement on the authoritative path.

            Legacy intent heuristics remain a compatibility fallback for old
            profiles, but they must not add an evidence requirement in Agent 2.
            """
            model_visual = CompletionState._agent2_pending(
                agent2_task_state, RESOLVE_VISUAL)
            if self.profile.features.get("agent2_authoritative"):
                return bool(model_visual)
            return wants_visual if model_visual is None else model_visual

        def _auto_retry_failed_resolution(requirement) -> bool:
            """Use the next visible candidate after a failed visual/OCR call.

            This is a bounded evidence-closure step: it only runs when the
            same resolution tool already produced a failed/partial result and
            a not-yet-inspected handle remains in the current ResultSet. It
            never broadens the result set or treats a failed call as evidence.
            """
            nonlocal resolution_retries, auto_resolution_attempts, inspect_called
            requirement_code = (requirement.get("code") if isinstance(requirement, dict)
                                else getattr(requirement, "code", None))
            if requirement_code not in {RESOLVE_VISUAL, RESOLVE_OCR}:
                return False
            if auto_resolution_attempts >= max_auto_resolution_attempts:
                return False
            tool_name = (requirement.get("tool") if isinstance(requirement, dict)
                         else getattr(requirement, "tool", None))
            if not tool_name or not turn.budget.can_tool_call():
                return False
            prior = [tr for tr in task.tool_results if tr.get("tool") == tool_name]
            # If the model ignored the mandatory resolution entirely, the
            # deterministic recovery should still make the first bounded
            # inspection call. Only a usable prior result suppresses it.
            if CompletionState._tool_succeeded(task.tool_results, tool_name):
                return False
            handle = _next_resolution_handle(task, tool_name)
            if not handle:
                return False
            spec = get_tool(tool_name)
            if spec is None:
                return False
            auto_resolution_attempts += 1
            resolution_retries += 1
            args = {"asset_handle": handle}
            if tool_name == "read_photo_text":
                args["question"] = message
            elif tool_name == "inspect_photo":
                # The automatic closure call must inspect the missing field,
                # not a generic scene summary. Reuse the planner's pending
                # visual requirement so attire/person/action questions remain
                # answerable even when the model skipped or mis-targeted its
                # first inspect call.
                target_question = ""
                if agent2_task_state is not None:
                    for state in agent2_task_state.requirements.values():
                        if (state.requirement.evidence_type == "visual_observation"
                                and state.status in {"open", "running", "partially_supported"}):
                            target_question = str(state.requirement.description or "").strip()
                            if target_question:
                                break
                args["question"] = target_question or message
            self._emit_progress(
                turn, progress_callback, stage="inspecting", status="running",
                text=f"正在复核照片 {handle}…" if tool_name == "inspect_photo"
                else "正在读取照片中的文字…")
            decision = policy.execute(spec, args, context={
                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                "task_state": task.as_dict(), "history": history,
                "conversation_id": self.conversation_id,
                "ocr_settings": self.ocr_settings,
            })
            observation = decision.observation or {}
            call_id = f"auto_{tool_name}_{auto_resolution_attempts}"
            state_before = agent2_task_state.status if agent2_task_state is not None else ""
            req_before = ({req_id: state.status
                           for req_id, state in agent2_task_state.requirements.items()}
                          if agent2_task_state is not None else {})
            ledger_entries_before = (len(agent2_evidence_ledger.entries)
                                     if agent2_evidence_ledger is not None else 0)
            turn.steps.append({
                "type": "tool", "tool": tool_name, "tool_call_id": call_id,
                "arguments": args, "status": "ok" if decision.allowed else "denied",
                "observation": observation, "error": decision.error,
                "parent_step_id": last_model_step_id,
                "auto_resolution": True,
                "raw_arguments": dict(args),
                "normalized_arguments": dict(args),
                "task_status_before": state_before,
                "requirement_status_before": req_before,
            })
            if not decision.allowed:
                if agent2_task_state is not None:
                    turn.steps[-1]["standardized_evidence"] = []
                    turn.steps[-1]["evidence_ids"] = []
                    turn.steps[-1]["task_status_after"] = agent2_task_state.status
                    turn.steps[-1]["requirement_status_after"] = {
                        req_id: state.status
                        for req_id, state in agent2_task_state.requirements.items()
                    }
                return False
            task.update_from_tool(tool_name, args, observation)
            if tool_name in {"query_memory_facts", "query_memory_metadata"}:
                _promote_structured_sources_to_result_set(
                    task, observation, scope_id=self.scope_id,
                    query=str(args.get("query") or message or ""))
            task.record_tool_result(call_id, tool_name, observation)
            if agent2_task_state is not None and agent2_evidence_ledger is not None:
                record_agent2_tool_evidence(
                    agent2_task_state, agent2_evidence_ledger, spec,
                    tool_call_id=call_id,
                    input_refs=(handle,), provenance_refs=(handle,),
                    observation=observation,
                    allow_partial=not self.profile.features.get("agent2_authoritative"),
                    question=message,
                )
                # An event-level group fact directly answers a visual
                # enumeration requirement; do not force a second identity
                # inspection merely because the planner described the same
                # requirement as visual_observation.
                if any(tr.get("tool") == "search_memories"
                       and tr.get("group_photo_count") is not None
                       for tr in task.tool_results):
                    for req_id, req_state in agent2_task_state.requirements.items():
                        if (req_state.requirement.evidence_type == "visual_observation"
                                and re.search(r"合影|人数|兄弟|brother|photo|number|count",
                                              f"{message} {req_state.requirement.description or ''}", re.I)):
                            try:
                                agent2_task_state.mark_satisfied(
                                    req_id, evidence_refs=(tool_call_id,))
                            except Exception:
                                pass
                _refresh_agent2_status(agent2_task_state, agent2_evidence_ledger)
                new_entries = agent2_evidence_ledger.entries[ledger_entries_before:]
                # Keep the public grounding projection in sync with the
                # authoritative ledger.  Structured location/date facts can
                # be valid evidence even when search validation returns no
                # visual rows, and must still carry their source asset.
                if task.tool_results:
                    ledger_assets = [str(entry.asset_id) for entry in new_entries
                                     if getattr(entry, "asset_id", None)]
                    if ledger_assets:
                        current = task.tool_results[-1].setdefault("evidence_asset_ids", []) or []
                        task.tool_results[-1]["evidence_asset_ids"] = list(dict.fromkeys(
                            [*current, *ledger_assets]))
                        task.tool_results[-1]["source_asset_ids"] = list(dict.fromkeys(
                            [*(task.tool_results[-1].get("source_asset_ids") or []), *ledger_assets]))
                turn.steps[-1]["standardized_evidence"] = [entry.as_dict() for entry in new_entries]
                turn.steps[-1]["evidence_ids"] = [
                    f"{entry.tool_call_id}:{entry.evidence_type}" for entry in new_entries
                ]
                turn.steps[-1]["task_status_after"] = agent2_task_state.status
                turn.steps[-1]["requirement_status_after"] = {
                    req_id: state.status
                    for req_id, state in agent2_task_state.requirements.items()
                }
            elif agent2_task_state is not None:
                turn.steps[-1]["standardized_evidence"] = []
                turn.steps[-1]["evidence_ids"] = []
                turn.steps[-1]["task_status_after"] = agent2_task_state.status
                turn.steps[-1]["requirement_status_after"] = {
                    req_id: state.status
                    for req_id, state in agent2_task_state.requirements.items()
                }
            if tool_name == "inspect_photo":
                inspect_called = True
            if tool_name == "read_photo_text" and observation.get("status") == "partial":
                turn.ocr_partial = True
                turn.ocr_partial_reason = str(observation.get("reason") or "ocr_failed")
            messages.append({"role": "assistant", "content": _model_visible_action({
                "action": "tool_call", "tool": tool_name, "arguments": args})})
            # The action is serialized as ordinary assistant content, not an
            # OpenAI tool_calls message. A strict OpenAI-compatible API (e.g.
            # DeepSeek) rejects a following role=tool message without the
            # matching tool_calls envelope, so keep the observation in a
            # normal user turn that remains model-visible.
            messages.append({"role": "user", "content": (
                f"工具 {tool_name}（{call_id}）返回：\n" +
                json.dumps(_model_visible_observation(observation), ensure_ascii=False)
            )})
            return True

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
        # Authoritative Agent2 must always hand the Evidence Ledger to the
        # single writer.  The feature flag remains useful for non-authoritative
        # shadow profiles, but can no longer silently bypass the production
        # evidence contract.
        answer_context_enabled = is_candidate_mode and (
            self.profile.features.get("agent2_authoritative") or
            self.profile.features.get("agent2_answer_context") or
            os.getenv("SENTRIX_AGENT2_ANSWER_CONTEXT", "").strip().lower() in {"1", "true", "on"}
        )
        answer_context = None
        answer_writer_pending = False
        answer_writer_messages = None
        while True:
            if answer_writer_pending and answer_writer_messages:
                if not turn.budget.can_model_step():
                    answer_writer_pending = False
                else:
                    turn.budget.record_model_step()
                    try:
                        if turn.agent2_trace:
                            turn.agent2_trace["writer_input"] = answer_writer_messages
                        raw_writer = self.chat_fn(
                            answer_writer_messages,
                            call_type="writer",
                            step_id="answer_writer",
                        ) or ""
                        from .final_writer import clean_writer_output
                        turn.final_answer = clean_writer_output(raw_writer)
                        # The single evidence writer can still choose a generic
                        # refusal when 12B sees rich facts. Apply the same
                        # claim-completeness policy used by the normal final
                        # path before releasing this answer; this is a
                        # text-only rewrite over the existing controlled facts,
                        # never a deterministic answer template.
                        # In authoritative Agent2 the evidence-only writer has
                        # already received the complete ledger. A second
                        # legacy-context rewrite can drop explicit unknown
                        # companions or replace a correct answer with a
                        # refusal, so style repair is limited to shadow mode.
                        if not self.profile.features.get("agent2_authoritative"):
                            try:
                                from .final_writer import (
                                    build_final_context, evidence_answer_problems,
                                    needs_rewrite, rewrite_final,
                                )
                                writer_context = build_final_context(message, task.as_dict())
                                writer_issues = evidence_answer_problems(
                                    message, turn.final_answer, writer_context)
                                if ((needs_rewrite(turn.final_answer, writer_context)
                                     or writer_issues)
                                        and turn.budget.can_model_step()):
                                    turn.budget.record_model_step()
                                    rewritten = rewrite_final(
                                        self.chat_fn, writer_context, turn.final_answer,
                                        step_id="answer_writer_repair")
                                    if rewritten:
                                        turn.final_answer = clean_writer_output(rewritten)
                                        turn.steps.append({
                                            "type": "writer", "status": "rewritten",
                                            "call_type": "writer",
                                            "step_id": "answer_writer_repair",
                                        })
                                        if turn.agent2_trace:
                                            turn.agent2_trace["writer_repair_output"] = turn.final_answer
                            except Exception:
                                pass
                        turn.steps.append({
                            "type": "writer",
                            "status": "generated",
                            "call_type": "writer",
                            "step_id": "answer_writer",
                            "raw": raw_writer[:500],
                            **({"prompt": answer_writer_messages, "raw_full": raw_writer}
                               if self.include_debug else {}),
                        })
                        if turn.agent2_trace:
                            turn.agent2_trace["writer_output"] = turn.final_answer
                        writer_problems = guard.check(
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
                                "evidence_refs": [],
                            },
                            delivered_count=task.delivered_count,
                        )
                        turn.steps.append({
                            "type": "guard",
                            "status": "fail" if writer_problems else "pass",
                            "codes": list(writer_problems) if writer_problems else [],
                            "attempt": 1,
                        })
                        if writer_problems:
                            severity = writer_problems.severity if hasattr(writer_problems, "severity") else "truth"
                            if severity == "hard_block":
                                turn.status = "blocked_by_guard"
                                turn.reason = "hard_block:" + ";".join(writer_problems)
                                if task.tool_results:
                                    turn.final_answer = render_emergency_summary(
                                        task.as_dict(), reason="回答未通过事实校验")
                            else:
                                turn.final_answer = _natural_partial(task.as_dict(), list(writer_problems))
                                turn.status = "partial"
                                turn.reason = "writer_guard:" + ";".join(str(p) for p in writer_problems[:2])
                        else:
                            turn.status = "complete" if turn.final_answer else "partial"
                            turn.reason = "" if turn.final_answer else "empty_answer_writer"
                        break
                    except Exception as exc:
                        turn.steps.append({"type": "writer", "status": "failed",
                                           "call_type": "writer", "reason": str(exc)})
                        answer_writer_pending = False
            if not turn.budget.can_model_step():
                # Phase H H4：步骤耗尽但 OCR 已读到确定性硬值时，直接渲染交付（不依赖 12B 收尾）
                if task.tool_results and not self.profile.features.get("agent2_authoritative"):
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
                # Phase H H8：兜底改模型输出——预算耗尽时允许额外一次纯 final（不计入步数，
                # 只受墙钟限制），让"没有答案"也由模型明确说出，而不是代码拼文案。
                if not forced_final_attempted:
                    forced_final_attempted = True
                    forced = self._force_final_once(turn, messages)
                    if forced:
                        turn.final_answer = forced
                        turn.status = "complete"
                        turn.reason = ""
                        turn.termination_reason = "forced_final_at_step_limit"
                        break
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
            step_id = f"step_{debug_step_seq}" if self.include_debug else None
            if self.include_debug:
                debug_step_seq += 1
                last_model_step_id = step_id
            try:
                sig = inspect.signature(self.chat_fn)
                if "call_type" in sig.parameters:
                    raw = self.chat_fn(messages, call_type="agent", step_id=step_id)
                else:
                    raw = self.chat_fn(messages)
            except Exception as exc:
                # D12：恢复/后续模型调用失败时，不丢弃已产出的 final 回答
                if turn.final_answer:
                    turn.status = "partial"
                    turn.reason = f"model_error_after_final: {exc}"
                    break
                turn.status = "error"
                turn.reason = f"model_call_error: {exc}"
                break
            model_step = {"type": "model", "raw": (raw or "")[:500]}
            if agent2_task_state is not None and agent2_evidence_ledger is not None:
                from .requirement_completion import RequirementCompletion
                model_step["tool_candidates"] = [
                    spec.name for spec in RequirementCompletion(
                        agent2_task_state, agent2_evidence_ledger
                    ).allowed_capabilities(list_tools(readiness="ready"))
                ]
            if self.include_debug:
                import copy as _copy
                model_step["raw_full"] = raw
                model_step["prompt"] = _copy.deepcopy(messages)
                model_step["step_id"] = step_id
                model_step["call_type"] = "agent"
                annotations = _prompt_annotations(messages)
                if annotations:
                    model_step["prompt_annotations"] = annotations
            turn.steps.append(model_step)
            action = self._parse_action(raw)
            if action is None:
                if parse_retries < max_parse_retries and turn.budget.can_model_step():
                    parse_retries += 1
                    messages.append({"role": "assistant", "content": raw})
                    if parse_retries == 2:
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
                    elif parse_retries >= 3:
                        # 第三次恢复：必须给出明确答案，哪怕确认"现有记录不足以回答"
                        self._emit_progress(
                            turn, progress_callback, stage="recovering", status="running",
                            text="我正在给出一个明确的结论。")
                        messages.append({"role": "user", "content": (
                            "你现在必须给出一个明确的 final 答案，不允许再调用工具。"
                            "如果现有记录足以回答，就直接回答；如果不足以回答，"
                            '就明确说"现有记录不足以确认"并给出你能确认的部分。'
                            '只输出 {"action":"final","answer":"<结论>","evidence_refs":[]}，'
                            "不要 markdown、不要多余文字。"
                        )})
                    else:
                        messages.append({"role": "user", "content": (
                            "你的上一条输出不是合法的 JSON 对象，无法解析。"
                            "请严格只输出一个 JSON 对象（action 只能是 tool_call 或 final），"
                            "不要 markdown、不要多余文字、不要省略结尾的引号或括号。"
                        )})
                    continue
                turn.status = "error"
                turn.reason = "unparseable_action"
                turn.termination_reason = "parse_failure"
                forced = self._force_final_once(turn, messages)
                if forced:
                    turn.final_answer = forced
                    turn.status = "complete"
                    turn.reason = ""
                    turn.termination_reason = "forced_final_after_parse_failure"
                elif task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="输出无法解析")
                break
            if action.get("action") == "final":
                last_model_final_answer = str(action.get("answer") or "").strip()
                selected_image_handles = _normalize_selected_image_handles(
                    action.get("selected_image_handles") or action.get("image_handles"),
                    task.result_preview,
                )
                # Authoritative Agent2 gate: evidence sufficiency is decided by
                # the single TaskState, never by intent heuristics or a model
                # assertion that it is "done".
                if self.profile.features.get("agent2_authoritative") \
                        and agent2_task_state is not None \
                        and agent2_evidence_ledger is not None:
                    agent2_status = _refresh_agent2_status(
                        agent2_task_state, agent2_evidence_ledger)
                    final_gate = turn.agent2_trace.setdefault("final_gate", {})
                    final_gate.update({
                        "status": agent2_status,
                        "candidate_closure": agent2_status == "complete",
                        "pending_requirements": [
                            state.requirement.id
                            for state in agent2_task_state.requirements.values()
                            if state.requirement.required and state.status != "satisfied"
                        ],
                    })
                    if agent2_status != "complete":
                        from .requirement_completion import RequirementCompletion
                        available = RequirementCompletion(
                            agent2_task_state, agent2_evidence_ledger
                        ).allowed_capabilities(list_tools(readiness="ready"))
                        # Planner declarations describe semantic evidence
                        # (location/memory) but may omit the prerequisite
                        # visual resolution tool. If search explicitly marks
                        # inspection as required, include that capability in
                        # the gate's actionable set instead of blocking before
                        # the resolution phase can run.
                        _pending_rec = _pending_resolution(task)
                        _resolution_spec = None
                        if _pending_rec and _pending_rec.get("tool"):
                            _resolution_spec = get_tool(_pending_rec["tool"])
                        if (_resolution_spec is not None and
                                all(item.name != _resolution_spec.name for item in available)):
                            available = list(available) + [_resolution_spec]
                        if _pending_rec:
                            _resolution_code = (RESOLVE_OCR if _pending_rec.get("tool") == "read_photo_text"
                                                else RESOLVE_VISUAL)
                            if _auto_retry_failed_resolution({
                                    "code": _resolution_code,
                                    "tool": _pending_rec.get("tool"),
                            }):
                                continue
                        pending = [
                            f"{state.requirement.id}:{state.requirement.evidence_type}"
                            for state in agent2_task_state.requirements.values()
                            if state.requirement.required and state.status != "satisfied"
                        ]
                        has_partial_answer = any(
                            state.requirement.required and state.status == "satisfied"
                            for state in agent2_task_state.requirements.values()
                        )
                        if (agent2_status == "in_progress" and available
                                and turn.budget.can_model_step()
                                and not has_partial_answer):
                            final_gate.update({
                                "decision": "continue",
                                "available_tools": [spec.name for spec in available],
                            })
                            messages.append({"role": "assistant", "content": _model_visible_action(action)})
                            messages.append({"role": "user", "content": (
                                "任务证据尚未闭合，不能输出确定性 final。"
                                f"未满足需求：{', '.join(pending)}。"
                                f"当前可用工具：{', '.join(spec.name for spec in available)}。"
                                "请选择一个工具继续获取实际证据。"
                            )})
                            continue
                        if has_partial_answer:
                            # A required ancillary field (for example venue)
                            # may remain unresolved while the user-requested
                            # field (for example confirmed people) is already
                            # directly supported. Let the writer answer only
                            # the supported portion and state the missing one.
                            final_gate.update({
                                "decision": "partial_answer",
                                "available_tools": [spec.name for spec in available],
                            })
                        else:
                            final_gate.update({
                                "decision": "block",
                                "available_tools": [spec.name for spec in available],
                            })
                            turn.final_answer = "现有证据不足，无法确认。"
                            turn.status = "partial"
                            turn.reason = "agent2_insufficient_evidence"
                            turn.termination_reason = "evidence_gate_blocked"
                            break
                # Agent 2.0 Guard: 如果从未执行任何检索工具且存在未满足的记忆/地点/事实需求，禁止直接猜测 final
                if is_candidate_mode and not task.tool_results and agent2_task_state is not None:
                    open_ev_types = {r.requirement.evidence_type for r in agent2_task_state.requirements.values() if r.status in ("open", "running")}
                    if open_ev_types and turn.budget.can_model_step():
                        messages.append({"role": "assistant", "content": _model_visible_action(action)})
                        messages.append({"role": "user", "content": (
                            "你尚未检索相册，禁止直接猜测回答。请先调用 search_memories 检索相关照片。"
                        )})
                        continue

                # Phase E §14：Premature Final Guard —— recommended_resolution 要求继续证据解析
                # Evidence resolution is a production correctness gate, not a
                # legacy/shadow-only hint. Authoritative Agent2 turns must also
                # inspect a representative photo when the retrieval contract
                # explicitly says that visual/person/travel evidence is needed.
                resolution = _pending_resolution(task)
                if resolution:
                    # Do not depend on the model obeying a recommendation: on
                    # the first premature final, execute the bounded visual/
                    # OCR resolution directly against the first visible
                    # handle.
                    if not any(tr.get("tool") == resolution["tool"]
                               for tr in task.tool_results):
                        resolution_code = (RESOLVE_OCR if resolution["tool"] == "read_photo_text"
                                           else RESOLVE_VISUAL)
                        if _auto_retry_failed_resolution({"code": resolution_code,
                                                          "tool": resolution["tool"]}):
                            continue
                    if resolution_retries >= max_resolution_retries:
                        # 模型多次忽略提示 → 使用统一的 bounded recovery 路径。
                        # 该路径同时写入 execution trace、TaskState 和 Agent2 ledger，
                        # 避免自动调用只影响内部状态而在 8771 轨迹中消失。
                        resolution_code = (RESOLVE_OCR if resolution["tool"] == "read_photo_text"
                                           else RESOLVE_VISUAL)
                        if _auto_retry_failed_resolution({"code": resolution_code,
                                                          "tool": resolution["tool"]}):
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
                        continue
                # 视觉细节意图 + 有 preview 候选 + 未 inspect → 确定性纠正一步（不依赖 12B 随机自觉）
                if search_has_preview and not inspect_called and visual_gate_requested() \
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
                    if task.tool_results and not self.profile.features.get("agent2_authoritative"):
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
                if not self.profile.features.get("agent2_authoritative"):
                    completion.update(task.as_dict(), agent2_task_state=agent2_task_state)
                gate_prompted = False
                for req in (() if self.profile.features.get("agent2_authoritative") else completion.blocking()):
                    if _auto_retry_failed_resolution(req):
                        gate_prompted = True
                        break
                    if req.code == RESOLVE_OCR and _pending_resolution(task):
                        continue  # 已由上方 recommended_resolution 流程处理
                    if req.code == RESOLVE_VISUAL and search_has_preview and not inspect_called \
                            and visual_gate_requested() and visual_retries < max_visual_retries:
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
                # Evidence resolution has a bounded recovery budget.  Once the
                # model retry and deterministic fallback are exhausted, never
                # release an unsupported final answer just because a tool was
                # invoked; return an honest partial instead.
                blocked_codes = ({req.code for req in completion.blocking()}
                                 if not self.profile.features.get("agent2_authoritative") else set())
                if (blocked_codes & {RESOLVE_OCR, RESOLVE_VISUAL}
                        and (completion_retries >= max_completion_retries
                             or not turn.budget.can_model_step())):
                    unresolved = [req.label for req in completion.blocking()]
                    turn.final_answer = _natural_partial(
                        task.as_dict(), problems=unresolved or ["证据复核失败"])
                    turn.status = "complete"
                    turn.termination_reason = "evidence_resolution_exhausted"
                    break
                # Phase F F1：Final Answer Writer——草稿违反 Answer Policy 时用受控事实重写
                if turn.final_answer and turn.budget.can_model_step():
                    try:
                        from .final_writer import build_final_context, needs_rewrite, rewrite_final
                        fctx = build_final_context(message, task.as_dict())
                        if needs_rewrite(turn.final_answer, fctx):
                            turn.budget.record_model_step()
                            _wr_debug = {} if self.include_debug else None
                            rewritten = rewrite_final(self.chat_fn, fctx, turn.final_answer,
                                                      debug_out=_wr_debug)
                            if rewritten and rewritten != turn.final_answer:
                                _wr_step = {"type": "writer", "status": "rewritten",
                                            "call_type": "writer"}
                                if _wr_debug:
                                    _wr_step["prompt"] = _wr_debug.get("messages")
                                    _wr_step["raw_full"] = rewritten
                                turn.steps.append(_wr_step)
                                turn.final_answer = naturalize_answer(rewritten)
                    except Exception:
                        pass
                try:
                    from .final_writer import (build_final_context, evidence_answer_problems)
                    _quality_context = build_final_context(message, task.as_dict())
                    _quality_problems = evidence_answer_problems(
                        message, turn.final_answer, _quality_context)
                    if _quality_problems and answer_quality_retries < 1 and turn.budget.can_model_step():
                        answer_quality_retries += 1
                        messages.append({"role": "assistant", "content": _model_visible_action(action)})
                        messages.append({"role": "user", "content": (
                            "已有工具结果包含直接证据，但上一版回答不完整或拒答（问题："
                            + ", ".join(_quality_problems)
                            + "）。请严格根据受控事实直接回答；已确认的人物、日期和地点必须写出，"
                              "无法确认的部分要明确说明，不要把其他人的细节归给目标人物。"
                        )})
                        continue
                    if _quality_problems:
                        # Do not replace the model's natural answer with a
                        # code-like deterministic concatenation. Preserve the
                        # Writer output and expose the unresolved quality
                        # state in the trace for the next bounded retry.
                        if turn.agent2_trace:
                            turn.agent2_trace.setdefault("quality", {})[
                                "unresolved_problems"] = list(_quality_problems)
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
                        judge_result = judge_faithfulness(
                            self.chat_fn, query=message, tool_results=task.tool_results,
                            answer=turn.final_answer, trusted_facts=trusted,
                            include_debug=self.include_debug)
                        if self.include_debug:
                            faithful, judge_problems, judge_debug = judge_result
                        else:
                            faithful, judge_problems = judge_result
                        judge_step = {"type": "judge", "faithful": faithful,
                                      "problems": list(judge_problems)}
                        if self.include_debug:
                            judge_step["debug"] = judge_debug
                            step_id = f"step_{debug_step_seq}"
                            debug_step_seq += 1
                            judge_step["step_id"] = step_id
                            judge_step["call_type"] = "faithfulness_judge"
                        turn.steps.append(judge_step)
                        if not faithful:
                            problems = judge_problems
                    except Exception as exc:
                        turn.steps.append({"type": "judge", "status": "skipped",
                                           "reason": f"model_call_error:{exc}"})
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
                    if task.tool_results and not self.profile.features.get("agent2_authoritative"):
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
                        continue
                    # Recovery v3 第二层：一次工具恢复——仍有证据解析需求未满足且预算允许
                    recovery_tool = None
                    if turn.budget.can_tool_call():
                        if not self.profile.features.get("agent2_authoritative"):
                            completion.update(task.as_dict(), agent2_task_state=agent2_task_state)
                        for req in (() if self.profile.features.get("agent2_authoritative") else completion.blocking()):
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
                                if recovery_tool in {"query_memory_facts", "query_memory_metadata"}:
                                    _promote_structured_sources_to_result_set(
                                        task, auto_decision.observation or {},
                                        scope_id=self.scope_id,
                                        query=str(auto_args.get("query") or message or ""))
                                task.record_tool_result(f"recovery_{recovery_tool}", recovery_tool,
                                                        auto_decision.observation or {})
                                if not self.profile.features.get("agent2_authoritative"):
                                    completion.update(task.as_dict(), agent2_task_state=agent2_task_state)
                                if recovery_tool == "read_photo_text" \
                                        and (auto_decision.observation or {}).get("status") == "partial":
                                    turn.ocr_partial = True
                                    turn.ocr_partial_reason = str(
                                        auto_decision.observation.get("reason") or "ocr_failed")
                                messages.append({"role": "assistant",
                                                 "content": _model_visible_action(action)})
                                messages.append({"role": "user", "content": (
                                    f"工具 {recovery_tool}（恢复结果）返回：\n" +
                                    json.dumps(_model_visible_observation(
                                        auto_decision.observation), ensure_ascii=False)
                                )})
                                messages.append({"role": "user", "content": (
                                    "我重新读取了一次照片，请基于新的工具观察，直接输出一个修正后的 final。"
                                )})
                                continue
                    # Recovery v3 第三层：优先让模型明确收尾（哪怕"没有答案"），
                    # 失败才回退 natural partial 代码文案。
                    forced = self._force_final_once(turn, messages)
                    if forced:
                        turn.final_answer = forced
                        turn.status = "complete"
                        turn.reason = ""
                        turn.termination_reason = "forced_final_after_guard_recovery"
                    else:
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
            raw_arguments = dict(action.get("arguments") or {})
            arguments = raw_arguments
            routing_correction = ""
            # Small local models can still pick the legacy list/count tool even
            # after the planner has declared structured event evidence. Keep
            # this boundary deterministic: event-level questions must read
            # event summaries, while photo/text questions remain on their
            # normal visual/OCR paths.
            if (self.profile.features.get("agent2_authoritative")
                    and event_summary_request
                    and tool_name in {"query_memory_facts", "search_memories",
                                      "inspect_photo", "read_photo_text"}):
                routing_correction = f"{tool_name}->query_memory_metadata(event)"
                tool_name = "query_memory_metadata"
                arguments = {"operation": "event", "query": message}
            # 模型有时把参数包在 arguments.schema 里，统一展开（工具契约兼容层）
            if isinstance(arguments.get("schema"), dict):
                arguments = {**arguments, **arguments["schema"]}
            normalized_arguments = dict(arguments)
            public_status = action.get("public_status") or "正在处理。"
            tool_call_seq += 1
            tool_call_id = f"tool_call_{tool_call_seq}"
            call_signature = json.dumps({"tool": tool_name, "arguments": arguments},
                                        ensure_ascii=False, sort_keys=True)
            if call_signature in seen_tool_calls:
                cached_observation = tool_result_cache.get(call_signature)
                if cached_observation is not None:
                    if not turn.budget.can_model_step():
                        # No model step remains to consume the cached result.
                        # Still close honestly as a bounded partial instead of
                        # reporting a policy/tool failure for an idempotent
                        # read-only retry.
                        turn.status = "partial" if turn.steps else "error"
                        turn.reason = "duplicate_tool_call_reused"
                        turn.termination_reason = "cached_tool_result"
                        turn.final_answer = render_emergency_summary(
                            task.as_dict(), reason="预算用尽")
                        break
                    dedup_retries += 1
                    messages.append({"role": "assistant", "content": _model_visible_action(action)})
                    messages.append({"role": "user", "content": (
                        f"工具 {tool_name}（缓存结果）返回：\n" +
                        json.dumps(_model_visible_observation(cached_observation), ensure_ascii=False)
                    )})
                    messages.append({"role": "user", "content": (
                        "这是你刚才相同工具调用的已缓存结果，不需要再次调用。"
                        "请直接基于该结果输出 final；如果证据仍不足，请明确说明无法确认。"
                    )})
                    continue
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
                    valid = "、".join(sorted({s.name for s in list_tools()
                                              if s.readiness != "blocked"}))
                    messages.append({"role": "user", "content": (
                        f"工具「{tool_name}」不存在。可用工具只有：{valid}。"
                        "请重新选择一个可用工具，或直接输出 final。"
                    )})
                    continue
                turn.steps.append({"type": "tool", "tool": tool_name, "status": "error",
                                   "reason": "unknown_tool"})
                turn.reason = "unknown_tool:" + tool_name
                turn.termination_reason = "tool_unavailable"
                break
            corrected_handle = None
            if tool_name in {"inspect_photo", "read_photo_text"}:
                arguments, corrected_handle = _normalize_preview_handle(
                    arguments, task.result_preview)
                normalized_arguments = dict(arguments)
            agent2_status_before = None
            agent2_requirements_before = None
            ledger_entries_before = 0
            if agent2_task_state is not None and agent2_evidence_ledger is not None:
                agent2_status_before = agent2_task_state.status
                agent2_requirements_before = {
                    req_id: state.status
                    for req_id, state in agent2_task_state.requirements.items()
                }
                ledger_entries_before = len(agent2_evidence_ledger.entries)
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
            result = ToolResult(tool=tool_name,
                                status="ok" if decision.allowed else "denied",
                                observation=decision.observation,
                                error=decision.error, latency_s=latency)
            turn.steps.append({
                "type": "tool", "tool": tool_name, "tool_call_id": tool_call_id,
                "arguments": arguments,
                "status": result.status, "observation": result.observation,
                "error": result.error, "latency_s": latency,
                "parent_step_id": last_model_step_id,
                "raw_arguments": raw_arguments,
                "normalized_arguments": normalized_arguments,
                "task_status_before": agent2_status_before,
                "requirement_status_before": agent2_requirements_before,
            })
            if routing_correction:
                turn.steps[-1]["routing_correction"] = routing_correction
            if corrected_handle and self.include_debug:
                turn.steps[-1]["requested_asset_handle"] = corrected_handle
            emit_text = public_status
            if tool_name == "inspect_photo" and result.status == "ok":
                handle_arg = str(arguments.get("asset_handle") or "")
                emit_text = f"已检查照片 {handle_arg}…" if handle_arg else "已检查照片…"
            self._emit_progress(
                turn, progress_callback,
                stage="tool_result" if result.status == "ok" else "tool_error",
                status=result.status, text=emit_text)
            if not decision.allowed:
                if agent2_task_state is not None:
                    turn.steps[-1]["standardized_evidence"] = []
                    turn.steps[-1]["evidence_ids"] = []
                    turn.steps[-1]["task_status_after"] = agent2_task_state.status
                    turn.steps[-1]["requirement_status_after"] = {
                        req_id: state.status
                        for req_id, state in agent2_task_state.requirements.items()
                    }
                # A stale model-selected handle is a recoverable visual
                # resolution failure. Retry the next bounded preview handle
                # before closing the task; otherwise one bad handle made the
                # whole answer look like missing evidence.
                if self.profile.features.get("agent2_authoritative") and tool_name in {
                        "inspect_photo", "read_photo_text"}:
                    failed_code = RESOLVE_VISUAL if tool_name == "inspect_photo" else RESOLVE_OCR
                    if _auto_retry_failed_resolution({"code": failed_code, "tool": tool_name}):
                        continue
                # A late duplicate/stale inspection must not erase a complete
                # structured answer. Preserve the model answer when no
                # structured override exists; for event group counts use the
                # typed fact already recorded by search_memories.
                if (self.profile.features.get("agent2_authoritative")
                        and agent2_task_state is not None
                        and agent2_task_state.status == "complete"):
                    group_fact = next((tr for tr in task.tool_results
                                       if tr.get("tool") == "search_memories"
                                       and tr.get("group_photo_count") is not None), None)
                    if group_fact:
                        sizes = group_fact.get("group_photo_sizes") or []
                        suffix = ("，分别是" + "、".join(str(x) + "人" for x in sizes)) if sizes else ""
                        turn.final_answer = (
                            f"一共拍了{int(group_fact.get('group_photo_count') or 0)}张不同人数的合影{suffix}。"
                        )
                    elif last_model_final_answer:
                        turn.final_answer = last_model_final_answer
                    if turn.final_answer:
                        turn.status = "complete"
                        turn.reason = "late_tool_rejection_after_complete"
                        turn.termination_reason = "task_complete"
                        break
                turn.status = "partial" if turn.steps else "error"
                turn.reason = f"tool_denied:{tool_name}:{decision.reason}"
                if self.profile.features.get("agent2_authoritative"):
                    turn.final_answer = "现有证据不足，无法确认。"
                    turn.termination_reason = "evidence_gate_blocked"
                elif task.tool_results:
                    turn.final_answer = render_emergency_summary(task.as_dict(), reason="工具调用被拒绝")
                break
            task.update_from_tool(tool_name, arguments, result.observation or {})
            if tool_name in {"query_memory_facts", "query_memory_metadata"}:
                _promote_structured_sources_to_result_set(
                    task, result.observation or {}, scope_id=self.scope_id,
                    query=str(arguments.get("query") or message or ""))
            task.record_tool_result(tool_call_id, tool_name, result.observation or {})
            if result.status == "ok":
                tool_result_cache[call_signature] = dict(result.observation or {})
            if agent2_task_state is not None and agent2_evidence_ledger is not None:
                input_refs = tuple(
                    str(value) for key, value in arguments.items()
                    if key in {"asset_handle", "result_set_id", "handle"} and value
                )
                record_agent2_tool_evidence(
                    agent2_task_state, agent2_evidence_ledger, spec,
                    tool_call_id=tool_call_id, input_refs=input_refs,
                    provenance_refs=input_refs,
                    observation=result.observation or {},
                    allow_partial=not self.profile.features.get("agent2_authoritative"),
                    question=message,
                )
                _refresh_agent2_status(agent2_task_state, agent2_evidence_ledger)
                new_entries = agent2_evidence_ledger.entries[ledger_entries_before:]
                if task.tool_results:
                    ledger_assets = [str(entry.asset_id) for entry in new_entries
                                     if getattr(entry, "asset_id", None)]
                    if ledger_assets:
                        current = task.tool_results[-1].setdefault("evidence_asset_ids", []) or []
                        task.tool_results[-1]["evidence_asset_ids"] = list(dict.fromkeys(
                            [*current, *ledger_assets]))
                        task.tool_results[-1]["source_asset_ids"] = list(dict.fromkeys(
                            [*(task.tool_results[-1].get("source_asset_ids") or []), *ledger_assets]))
                turn.steps[-1]["standardized_evidence"] = [entry.as_dict() for entry in new_entries]
                turn.steps[-1]["evidence_ids"] = [
                    f"{entry.tool_call_id}:{entry.evidence_type}" for entry in new_entries
                ]
                if answer_context_enabled:
                    from .final_writer import build_answer_writer_messages
                    answer_context = agent2_evidence_ledger.build_answer_context(
                        message, agent2_task_state)
                    turn.agent2_trace["answer_context"] = answer_context
                    if _agent2_answer_context_ready(agent2_task_state, answer_context):
                        answer_writer_messages = build_answer_writer_messages(
                            message, answer_context)
                        answer_writer_pending = True
                turn.steps[-1]["task_status_after"] = agent2_task_state.status
                turn.steps[-1]["requirement_status_after"] = {
                    req_id: state.status
                    for req_id, state in agent2_task_state.requirements.items()
                }
            if not self.profile.features.get("agent2_authoritative"):
                completion.update(task.as_dict(), agent2_task_state=agent2_task_state)
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
            # Candidate 模式下根据最新 TaskState 动态更新首条 JIT System Prompt
            if is_candidate_mode and agent2_task_state is not None:
                messages[0]["content"] = build_jit_system_prompt(
                    task_state=agent2_task_state,
                    current_time_str=current_time_line(),
                    tool_results=task.tool_results,
                    preview_handles=task.result_preview,
                    is_candidate=True,
                    allowed_tool_names=self.profile.tools,
                )
            messages.append({"role": "assistant", "content": _model_visible_action(action)})
            # See the strict-provider compatibility note above: observations
            # use a regular user turn because the assistant action is not a
            # native tool_calls object.
            messages.append({"role": "user", "content": (
                f"工具 {tool_name} 返回：\n" +
                json.dumps(_model_visible_observation(result.observation), ensure_ascii=False)
            )})
            if self.profile.features.get("agent2_authoritative"):
                failed_resolution = None
                obs = result.observation or {}
                certainty = str(obs.get("certainty") or "supported").lower()
                status = str(obs.get("status") or "").lower()
                if tool_name == "inspect_photo" and (
                        certainty in {"uncertain", "unsupported"}
                        or status in {"partial", "failed", "error", "unavailable"}
                        or obs.get("blocked")):
                    failed_resolution = {"code": RESOLVE_VISUAL, "tool": tool_name}
                elif tool_name == "read_photo_text" and (
                        status in {"partial", "failed", "error", "unavailable"}
                        or obs.get("blocked")):
                    failed_resolution = {"code": RESOLVE_OCR, "tool": tool_name}
                if failed_resolution and _auto_retry_failed_resolution(failed_resolution):
                    continue

        if self.profile.features.get("agent2_authoritative") and agent2_task_state is not None:
            # The end-of-turn trace must reflect the final state, not an
            # intermediate model final that was correctly rejected earlier.
            if turn.agent2_trace:
                turn.agent2_trace.setdefault("final_gate", {})[
                    "status"] = agent2_task_state.status
        # Writer output remains authoritative. Any unresolved completeness
        # issue is retained in trace/guard state instead of being replaced by
        # a code-like deterministic answer.
        turn.task_state = task.as_dict()
        turn.task_state["completion"] = completion.as_dict()
        turn.selected_image_handles = selected_image_handles
        if selected_image_handles:
            try:
                from . import tools as runtime_tools
                turn.selected_image_ids = [
                    asset_id for handle in selected_image_handles
                    if (asset_id := runtime_tools.resolve_handle_asset_id(
                        handle, task.current_result_set, self.scope_id))
                ]
            except Exception:
                turn.selected_image_ids = []
        turn.answer_grounding = _build_answer_grounding(
            message=message, task=task, selected_handle=selected_handle,
            selected_image_handles=turn.selected_image_handles,
            selected_image_ids=turn.selected_image_ids)
        # Enumeration answers should deliver every bounded source row that
        # carries a distinct group size, even if the model selected only one
        # image in its final JSON.
        group_fact = next((tr for tr in task.tool_results
                           if tr.get("tool") == "search_memories"
                           and tr.get("group_photo_rows")), None)
        if group_fact:
            group_ids = {str(row.get("asset_id")) for row in
                         (group_fact.get("group_photo_rows") or []) if row.get("asset_id")}
            group_handles = [str(item.get("handle")) for item in
                             (turn.answer_grounding.get("evidence_images") or [])
                             if str(item.get("asset_id")) in group_ids and item.get("handle")][:3]
            if group_handles:
                turn.selected_image_handles = group_handles
                try:
                    from . import tools as runtime_tools
                    turn.selected_image_ids = [
                        asset_id for handle in group_handles
                        if (asset_id := runtime_tools.resolve_handle_asset_id(
                            handle, task.current_result_set, self.scope_id))
                    ]
                except Exception:
                    turn.selected_image_ids = []
                turn.answer_grounding = _build_answer_grounding(
                    message=message, task=task, selected_handle=selected_handle,
                    selected_image_handles=turn.selected_image_handles,
                    selected_image_ids=turn.selected_image_ids)
        # Delivery is a projection of evidence, not a second retrieval path.
        # If the writer omits selected_image_handles, expose up to three
        # representative evidence images so both 4174 and 8771 still show the
        # sources that support the answer.  Never fall back to raw candidates.
        if not turn.selected_image_handles:
            fallback_items = [
                item for item in
                (turn.answer_grounding.get("evidence_images") or [])
                if item.get("asset_id")
            ][:3]
            fallback_handles = [
                str(item.get("handle")) for item in fallback_items
                if item.get("handle")
            ]
            fallback_ids = [
                str(item.get("asset_id")) for item in fallback_items
                if item.get("asset_id")
            ]
            if fallback_handles or fallback_ids:
                # A structured fact may have a source asset but no current
                # ResultSet handle.  Keep the asset ID as the delivery key so
                # both benchmark and production clients can render it.
                turn.selected_image_handles = fallback_handles
                turn.selected_image_ids = fallback_ids
                try:
                    from . import tools as runtime_tools
                    resolved = [
                        asset_id for handle in fallback_handles
                        if (asset_id := runtime_tools.resolve_handle_asset_id(
                            handle, task.current_result_set, self.scope_id))
                    ]
                    if resolved:
                        turn.selected_image_ids = list(dict.fromkeys(resolved))
                except Exception:
                    pass
                turn.answer_grounding = _build_answer_grounding(
                    message=message, task=task, selected_handle=selected_handle,
                    selected_image_handles=turn.selected_image_handles,
                    selected_image_ids=turn.selected_image_ids)
        turn.termination_reason = _classify_termination(turn)
        if turn.agent2_trace:
            if agent2_task_state is not None and agent2_evidence_ledger is not None:
                turn.agent2_trace["task_state"] = agent2_task_state.as_dict()
                turn.agent2_trace["evidence_ledger"] = agent2_evidence_ledger.as_dict()
            if self.profile.features.get("agent2_authoritative"):
                turn.agent2_trace["terminal_reason"] = (
                    "task_complete" if agent2_task_state is not None
                    and agent2_task_state.status == "complete"
                    else "insufficient_evidence"
                )
            else:
                turn.agent2_trace["terminal_reason"] = "shadow_only"
            turn.agent2_trace["budget_outcome"] = turn.budget.as_dict()
            turn.agent2_trace["stage_timing_ms"] = {
                k: round(v, 1) for k, v in sorted(_stage_timing_ms.items())}
        self.chat_fn = _orig_chat_fn
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
