"""Just-In-Time (JIT) Prompt Builder for Agent 2.0.

根据当前 TaskState 的未决需求与执行阶段，动态剪枝工具描述与系统提示词，
保证单步 System Prompt 长度控制在 150~300 tokens 以内，彻底消除小模型注意力过载。
"""

from __future__ import annotations

import json
from typing import Any
from .task_state import TaskState
from .tool_registry import get_tool, ToolSpec, list_tools


# 极简通用核心规则（约 150 tokens；工具级要求下沉到各工具描述，避免静态规则
# 提及未暴露工具——模型只在看到工具时看到该工具的规则）
BASE_SYSTEM_PROMPT = """你是 Sentrix 家庭记忆助手。根据当前待确认需求调用工具或给出结论。
规则：
1. 每次只输出一个标准 JSON 对象，不要输出 markdown 代码块或解释；
2. 调用工具格式：{"action":"tool_call","tool":"<工具名>","arguments":{...},"public_status":"<简短状态>"}
3. 结论输出格式：{"action":"final","answer":"<直接回答用户问题>","evidence_refs":["tool_call_1"],"selected_image_handles":["photo_1"]}；只有确实要给用户看的图片才填写 selected_image_handles，最多 6 张。
4. 未调用能满足当前证据需求的工具前不得猜测；已有足够事实，或已确认当前条件无法统计时，直接输出 final。
5. selected_image_handles 只能填写当前照片预览中出现的 handle；搜索候选不等于要展示的图片。
6. 不要重复调用相同工具和参数；只使用工具返回的事实回答，不编造数字或细节。
7. 内部检索词汇（query_satisfaction、candidate_only、检索结果、匹配程度等）不得原样出现在回答里，用自然语言转译；不确定性用四级：确定→直接给答案、较可能→"看起来是…"、不确定→"可能…还不能完全确定"、无依据→"现有记录里看不出来"。
8. 检索满足度与照片复核是两层，分开表述：检索层说语义条件（活动/地点/时间）是否确认；复核层（看照片细节）说照片里直接可见的内容；照片里看到的不能把"候选"说成"已确认"。"""


# 工具极简契约定义（每个工具仅保留最精简输入格式，<40 tokens）
LITE_TOOL_SCHEMAS = {
    "query_memory_facts": (
        "- query_memory_facts: 精确统计当前相册的原始图片/视频、已命名人物出现或处理状态。"
        "不支持颜色、物体、场景、活动、关系、金额或桌数；不要把检索候选数当总数。\n"
        '  输入: {"operation":"count|exists|first|last|group","subject":"asset|person_appearance|processing",'
        '"group_by":"month|place|media|status","filters":{"time":"","media":"image|video","place":"","person":""}}'
    ),
    "search_memories": (
        "- search_memories: 检索照片，每轮只调一次（返回最多 18 张候选，预览每张含地点/时间/描述/handle）。"
        "问'哪些/几种/所有不同场地或对象'等枚举聚合类问题时直接看预览描述+翻页批量判断，不要逐张 inspect；"
        "需要更多候选用 get_result_page 翻页；返回 recommended_handle 时优先复核该图。\n"
        '  输入: {"query": "关键词", "filters": {"time": "<问题时间，缺省省略>", "person": "人物", "place": "地点"}}'
    ),
    "query_photo_people": (
        "- query_photo_people: 读取当前预览中一张照片的人脸绑定人物及其家庭归属；没有稳定绑定的同行者明确标为未知。\n"
        '  输入: {"asset_handle": "photo_1", "result_set_id": "..."}'
    ),
    "inspect_photo": (
        "- inspect_photo: 复核照片视觉细节（人物/衣服颜色/物品/动作）。只用于看单张细节；跨图枚举/汇总用 search 预览描述。"
        "当前 handle 不含目标就换下一张（photo_2、photo_3…）复核，同一张最多复核一次。\n"
        '  输入: {"asset_handle": "photo_1", "question": "观察问题"}'
    ),
    "read_photo_text": (
        "- read_photo_text: 读取照片中的文字/招牌/价格/数字。优先用 search 返回的 recommended_handle。\n"
        '  输入: {"asset_handle": "photo_1", "question": "读取问题"}'
    ),
    "get_result_page": (
        "- get_result_page: 查看 search_memories 结果集的下一页（每页最多 6 张）。不要重复 search，改变查询条件才重搜。\n"
        '  输入: {"result_set_id": "...", "page": 2}'
    ),
    "get_original_photos": (
        "- get_original_photos: 交付照片原图。\n"
        '  输入: {"handle": "photo_1"}'
    ),
    "search_conversation_history": (
        "- search_conversation_history: 查询历史对话记录。\n"
        '  输入: {"query": "...", "scope": "current|recent"}'
    ),
    "get_core_memory": (
        "- get_core_memory: 读取长期家庭记忆。\n"
        '  输入: {"subject": "人物名", "topic": "话题"}'
    ),
    "get_person_profile": (
        "- get_person_profile: 读取人物高维画像（家庭归属/关系/行为规律/近期事件）；关系来源会标注模型或用户确认。\n"
        '  输入: {"person": "人物名"}'
    ),
}


def build_jit_system_prompt(
    *,
    task_state: TaskState | None,
    current_time_str: str,
    tool_results: list[dict] | None = None,
    preview_handles: list[str] | None = None,
    is_candidate: bool = False,
    allowed_tool_names: set[str] | tuple[str, ...] | None = None,
) -> str:
    """按需动态组装当前轮次的 System Prompt。"""
    tool_results = tool_results or []
    preview_handles = preview_handles or []
    
    # 1. Base Prompt
    parts = [BASE_SYSTEM_PROMPT]
    if current_time_str:
        parts.append(f"当前时间：{current_time_str}")
        
    if not is_candidate or task_state is None:
        # Fallback 到包含所有常用工具的简版
        tool_descriptions = "\n".join([
            LITE_TOOL_SCHEMAS["search_memories"],
            LITE_TOOL_SCHEMAS["query_photo_people"],
            LITE_TOOL_SCHEMAS["inspect_photo"],
            LITE_TOOL_SCHEMAS["read_photo_text"],
        ])
        parts.append(f"可用工具：\n{tool_descriptions}")
        return "\n\n".join(parts)

    # 2. 注入当前任务目标与未决状态
    goal = task_state.declaration.goal
    open_reqs = [
        f"{r.requirement.evidence_type}({r.requirement.description or '待确认'})"
        for r in task_state.requirements.values()
        if r.status in ("open", "running", "partially_supported")
    ]
    satisfied_reqs = [
        r.requirement.id for r in task_state.requirements.values()
        if r.status == "satisfied"
    ]
    
    state_desc = f"任务目标：{goal}\n"
    if open_reqs:
        state_desc += f"待确认证据：{', '.join(open_reqs)}"
    else:
        state_desc += "没有强制的证据需求：可直接 final；若答案需要从相册核实，先调用检索工具再作答，不要编造。"
    parts.append(state_desc)

    # 3. JIT 只依据统一注册表和未满足需求提供工具，不按问题关键词
    # 硬编码“先 search 再 inspect”的流程。模型仍然决定下一步调用哪个工具。
    open_types = {
        r.requirement.evidence_type for r in task_state.requirements.values()
        if r.status in ("open", "running", "partially_supported")
        and r.requirement.required
    }
    ready_specs = list_tools(readiness="ready")
    if allowed_tool_names is not None:
        allowed = set(allowed_tool_names)
        ready_specs = [spec for spec in ready_specs if spec.name in allowed]
    satisfied_types = {
        state.requirement.evidence_type
        for state in task_state.requirements.values()
        if state.status == "satisfied"
    }
    missing_prerequisites = set()
    for spec in ready_specs:
        if not any(spec.can_satisfy(evidence_type) for evidence_type in open_types):
            continue
        if "asset_handle_in_current_preview" in spec.preconditions and not preview_handles:
            missing_prerequisites.update(
                item for item in spec.prerequisite_evidence_types
                if item not in satisfied_types
            )
    selected_specs: list[ToolSpec] = []
    for spec in ready_specs:
        direct = any(spec.can_satisfy(evidence_type) for evidence_type in open_types)
        prerequisite_provider = any(
            spec.can_satisfy(evidence_type) for evidence_type in missing_prerequisites
        )
        if not direct and not prerequisite_provider:
            continue
        if "asset_handle_in_current_preview" in spec.preconditions and not preview_handles:
            continue
        selected_specs.append(spec)
    selected_tools = [spec.name for spec in selected_specs]

    # 所有需求已满足 / 无需证据：有把握就直接 final；若需要核实相册记录（具体数字/金额/
    # 年份/地点/有没有某物等），保留 search_memories 可选，避免模型"事实足够"时凭空编数字。
    if not open_reqs:
        parts.append("当前没有待确认的证据需求。可直接回答的问题请直接输出 final；"
                     "若答案需要从相册照片核实，请先调用 search_memories 检索，"
                     "再如实作答或说明无法确认，不要编造数字/细节。")
        parts.append("本步骤可用工具（按需核实，也可直接 final）：\n"
                     + LITE_TOOL_SCHEMAS["search_memories"])
    elif not selected_tools:
        # 有需求但暂无直接满足的工具：由模型按需调用已注册检索工具获取证据后再 final。
        parts.append("当前有待确认的证据需求，请按需调用合适的工具获取证据后再 final。")
    else:
        # 去重并添加工具描述
        tool_text_list = []
        for tname in dict.fromkeys(selected_tools):
            if tname in LITE_TOOL_SCHEMAS:
                tool_text_list.append(LITE_TOOL_SCHEMAS[tname])
        parts.append("本步骤可用工具（按需调用）：\n" + "\n".join(tool_text_list))
        
    return "\n\n".join(parts)
