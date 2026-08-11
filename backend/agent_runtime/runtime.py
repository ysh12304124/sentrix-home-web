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
- 检索条件已确认（condition_summary 标记 confirmed，或 query_satisfaction=full_support）时，直接当作确定事实回答，
  不要画蛇添足加"还不能完全确认"；只有 candidate_only 或关键条件 unknown 时才说"找到几张接近的，还不能完全确认"。
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
                 scope_id="home-default", viewer_id="owner", conversation_id=None):
        self.chat_fn = chat_fn
        self.profile = get_profile(profile_name)
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.conversation_id = conversation_id

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
        unknown_tool_retries = 0
        max_unknown_tool_retries = 1
        visual_intent = bool(__import__("re").search(
            r"桌上|桌面|颜色|几个|多少人|招牌|文字|天气|外套|衣服|猫|雪|小孩|穿着|穿|在做什么|"
            r"有没有|是什么|放着|写了|内容|细节", message))
        # Phase E：Adaptive Visual Budget——按问题类型放宽视觉复核预算
        multi_image_intent = bool(__import__("re").search(
            r"哪一张|哪张|哪些|哪几张|每一张|逐[一一张]|逐一|对比|还有吗|还有没有|都看|全部|每张|"
            r"所有照片|哪几张|哪几个|翻看", message))
        ocr_intent = bool(__import__("re").search(
            r"菜单|价格|多少钱|售价|招牌|店名|电话|写了什么|什么字|文字|创始于|"
            r"价位|几块钱|面单|多少钱一份", message))
        adaptive_inspections = self.profile.max_inspections
        if multi_image_intent:
            adaptive_inspections = max(adaptive_inspections, 4)
        elif ocr_intent:
            adaptive_inspections = max(adaptive_inspections, 2)
            # read_photo_text 需要整图 + 3x3 tile 多次图片推理，放宽总预算
            # （同图 OCR 结果已缓存，只有首次需要长预算）
            turn.budget.wall_time_s = max(turn.budget.wall_time_s, 240)
        turn.budget.max_inspections = adaptive_inspections
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
                # D12：恢复/后续模型调用失败时，不丢弃已产出的 final 回答
                if turn.final_answer:
                    turn.status = "partial"
                    turn.reason = f"model_error_after_final: {exc}"
                    break
                turn.status = "error"
                turn.reason = f"model_call_error: {exc}"
                break
            turn.steps.append({"type": "model", "raw": (raw or "")[:500]})
            action = self._parse_action(raw)
            if action is None:
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
                            auto_args = {"asset_handle": preview[0]}
                            if tool_name == "read_photo_text":
                                auto_args["question"] = message
                            self._emit_progress(
                                turn, progress_callback, stage="inspecting", status="running",
                                text=f"正在复核照片 {preview[0]}…" if tool_name == "inspect_photo"
                                else "正在读取照片中的文字…")
                            auto_decision = policy.execute(auto_spec, auto_args, context={
                                "scope_id": self.scope_id, "viewer_id": self.viewer_id,
                                "task_state": task.as_dict(), "history": history,
                                "conversation_id": self.conversation_id,
                            })
                            if auto_decision.allowed:
                                task.update_from_tool(tool_name, auto_args, auto_decision.observation or {})
                                task.record_tool_result(f"auto_{tool_name}", tool_name,
                                                        auto_decision.observation or {})
                                messages.append({"role": "assistant", "content": raw})
                                messages.append({"role": "tool", "tool_call_id": f"auto_{tool_name}",
                                                 "content": json.dumps(
                                                     auto_decision.observation or {}, ensure_ascii=False)})
                                continue
                    elif resolution_retries < max_resolution_retries and turn.budget.can_model_step():
                        resolution_retries += 1
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({"role": "user", "content": (
                            f"{resolution.get('reason') or '问题需要复核照片'}。"
                            f"这是完成回答的必要步骤：你必须立即调用 {resolution['tool']}"
                            "（asset_handle 用 preview 里的 handle，例如 photo_1），"
                            "先拿到实际观察，再基于观察输出 final。不调用该工具就无法正确回答。"
                        )})
                        continue
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
                # Phase F F1：Final Answer Writer——草稿违反 Answer Policy 时用受控事实重写
                if turn.final_answer and turn.budget.can_model_step():
                    try:
                        from .final_writer import build_final_context, needs_rewrite, rewrite_final
                        fctx = build_final_context(message, task.as_dict())
                        if needs_rewrite(turn.final_answer, fctx):
                            turn.budget.record_model_step()
                            rewritten = rewrite_final(self.chat_fn, fctx, turn.final_answer)
                            if rewritten and rewritten != turn.final_answer:
                                turn.steps.append({"type": "writer", "status": "rewritten"})
                                turn.final_answer = rewritten
                    except Exception:
                        pass
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
                            ("\nread_photo_text 实际读到的文字是：\n" + "\n".join(ocr_obs)
                             + "\n基于这些读到的文字直接回答具体内容（价格/店名/电话/年份），"
                               "不要笼统说'还不能确认'；检索层的不确定可单独用一句自然语言带过。"
                             if ocr_obs else "") +
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
                if unknown_tool_retries < max_unknown_tool_retries and turn.budget.can_model_step():
                    unknown_tool_retries += 1
                    messages.append({"role": "assistant", "content": raw})
                    from .tool_registry import list_tools
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
            if tool_name == "search_memories" and (result.observation or {}).get("can_inspect"):
                search_has_preview = True
            if tool_name == "inspect_photo":
                inspect_called = True
            # Observation 进入下一步模型上下文
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "tool_call_id": tool_name, "content": json.dumps(
                result.observation or {}, ensure_ascii=False)})

        turn.task_state = task.as_dict()
        turn.answer_grounding = _build_answer_grounding(
            message=message, task=task, selected_handle=selected_handle)
        turn.termination_reason = _classify_termination(turn)
        return turn


def _classify_termination(turn: RuntimeTurn) -> str:
    """D11：termination_reason 全量分类（telemetry 用）。"""
    reason = turn.reason or ""
    if turn.status == "complete":
        return "complete"
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
