"""Just-In-Time (JIT) Prompt Builder for Agent 2.0.

根据当前 TaskState 的未决需求与执行阶段，动态剪枝工具描述与系统提示词，
保证单步 System Prompt 长度控制在 150~300 tokens 以内，彻底消除小模型注意力过载。
"""

from __future__ import annotations

import json
import re
from typing import Any
from .task_state import TaskState
from .tool_registry import get_tool, ToolSpec, list_tools


# 极简通用核心规则（小于 100 tokens）
BASE_SYSTEM_PROMPT = """你是 Sentrix 家庭记忆助手。根据当前待确认需求调用工具或给出结论。
规则：
1. 每次只输出一个标准 JSON 对象，不要输出 markdown 代码块或解释；
2. 调用工具格式：{"action":"tool_call","tool":"<工具名>","arguments":{...},"public_status":"<简短状态>"}
3. 结论输出格式：{"action":"final","answer":"<直接回答用户问题>","evidence_refs":["tool_call_1"],"selected_image_handles":["photo_1"]}；只有确实要给用户看的图片才填写 selected_image_handles，最多 6 张。
4. 未检索相册前禁止猜测或回答；已有足够事实时直接输出 final。
5. selected_image_handles 只能填写当前 search_memories preview 中出现的 handle；搜索候选不等于要展示的图片。"""


# 工具极简契约定义（每个工具仅保留最精简输入格式，<40 tokens）
LITE_TOOL_SCHEMAS = {
    "search_memories": (
        "- search_memories: 检索照片。返回照片预览（含地点/拍摄时间/handle）。\n"
        '  输入: {"query": "关键词", "filters": {"time": "<问题中的时间，缺省省略>", "person": "人物", "place": "地点"}}'
    ),
    "query_memory_facts": (
        "- query_memory_facts: 查询统计与结构化事实（总数/最早/最近/分组）；不要用它回答视频场景做了什么或展示了什么。\n"
        '  输入: {"operation": "count|first|last|date|group|meal|list", "filters": {"time": "时间", "person": "人物", "media": "video/image"}}'
    ),
    "query_memory_metadata": (
        "- query_memory_metadata: 查询视频/事件摘要及结构化元数据。视频里做了什么、展示了什么、先后发生了什么时用 operation=event；要编排故事线、章节、旁白或剪辑方案时用 operation=timeline 读取完整时间线；\n"
        '  输入: {"operation": "event|timeline|date|place|count", "query": "视频或事件关键词"}'
    ),
    "inspect_photo": (
        "- inspect_photo: 复核照片视觉细节（人物/衣服颜色/物品/动作）。\n"
        '  输入: {"asset_handle": "photo_1", "question": "观察问题"}'
    ),
    "read_photo_text": (
        "- read_photo_text: 读取照片中的文字/招牌/价格/数字。\n"
        '  输入: {"asset_handle": "photo_1", "question": "读取问题"}'
    ),
    "get_result_page": (
        "- get_result_page: 获取结果集下一页。\n"
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
    "get_person_memory": (
        "- get_person_memory: 读取人物结构化记忆。\n"
        '  输入: {"person": "人物名", "operation": "overview|events|common_places"}'
    ),
    "get_person_profile": (
        "- get_person_profile: 读取人物高维画像（家庭角色/关系/行为规律/近期事件）。\n"
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
            LITE_TOOL_SCHEMAS["query_memory_facts"],
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
        state_desc += "所有证据已确认充分，请直接整理 final 回答。"
    parts.append(state_desc)
    if re.search(r"剪成|剪辑|故事线|章节|旁白|分镜|开场|片头|蒙太奇|b-?roll|短视频|脚本|标题|转场|配乐|选素材|剪片|vlog|视频结构", goal, re.I):
        parts.append(
            "这是视频创作编排任务。读取时间线后输出可执行方案：标题/章节、镜头顺序、每段用途、旁白或字幕、转场建议；"
            "不要只复述检索结果，也不要因为没有单张图片而拒答。只能把摘要明确支持的内容写成事实，推测内容标注待确认。"
        )

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

    # 如果所有需求都满足或无需工具，不暴露工具，直接引导 final
    if not open_reqs or not selected_tools:
        parts.append("当前已具备足够事实，请直接输出 final 结论。")
    else:
        # 去重并添加工具描述
        tool_text_list = []
        for tname in dict.fromkeys(selected_tools):
            if tname in LITE_TOOL_SCHEMAS:
                tool_text_list.append(LITE_TOOL_SCHEMAS[tname])
        parts.append("本步骤可用工具（按需调用）：\n" + "\n".join(tool_text_list))
        
    return "\n\n".join(parts)
