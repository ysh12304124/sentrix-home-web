"""Just-In-Time (JIT) Prompt Builder for Agent 2.0.

根据当前 TaskState 的未决需求与执行阶段，动态剪枝工具描述与系统提示词，
保证单步 System Prompt 长度控制在 150~300 tokens 以内，彻底消除小模型注意力过载。
"""

from __future__ import annotations

import json
from typing import Any
from .task_state import TaskState
from .tool_registry import get_tool, ToolSpec


# 极简通用核心规则（小于 100 tokens）
BASE_SYSTEM_PROMPT = """你是 Sentrix 家庭记忆助手。根据当前待确认需求调用工具或给出结论。
规则：
1. 每次只输出一个标准 JSON 对象，不要输出 markdown 代码块或解释；
2. 调用工具格式：{"action":"tool_call","tool":"<工具名>","arguments":{...},"public_status":"<简短状态>"}
3. 结论输出格式：{"action":"final","answer":"<直接回答用户问题>","evidence_refs":["tool_call_1"]}
4. 未检索相册前禁止猜测或回答；已有足够事实时直接输出 final。"""


# 工具极简契约定义（每个工具仅保留最精简输入格式，<40 tokens）
LITE_TOOL_SCHEMAS = {
    "search_memories": (
        "- search_memories: 检索照片。返回照片预览（含地点/拍摄时间/handle）。\n"
        '  输入: {"query": "关键词", "filters": {"time": "时间(如2019年7月)", "person": "人物", "place": "地点"}}'
    ),
    "query_memory_facts": (
        "- query_memory_facts: 查询统计与结构化事实（总数/最早/最近/分组）。\n"
        '  输入: {"operation": "count|first|last|date|group|meal|list", "filters": {"time": "时间", "person": "人物", "media": "video/image"}}'
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

    # 3. JIT 动态挑选候选工具 (方案 A: 分步呈现)
    selected_tools: list[str] = []
    
    # 判断是否已有检索 preview
    has_preview = bool(preview_handles)
    open_types = {
        r.requirement.evidence_type for r in task_state.requirements.values()
        if r.status in ("open", "running", "partially_supported")
    }

    # 规则 A1: 如果尚未执行检索，优先供给检索类工具
    if not tool_results:
        if "structured_fact" in open_types:
            selected_tools.append("query_memory_facts")
            selected_tools.append("search_memories")
        elif "user_statement" in open_types:
            selected_tools.append("search_conversation_history")
            selected_tools.append("search_memories")
        else:
            selected_tools.append("search_memories")
            if any(t in open_types for t in ("structured_fact", "confirmed_identity")):
                selected_tools.append("query_memory_facts")
    else:
        # 已有工具执行结果
        if has_preview:
            # 存在照片预览，检查是否需要视觉或 OCR 深入复核
            if "visible_text" in open_types:
                selected_tools.append("read_photo_text")
            if "visual_observation" in open_types:
                selected_tools.append("inspect_photo")

        # 如果仍有检索类需求未满足
        if any(t in open_types for t in ("memory_asset", "location_metadata", "temporal_metadata")):
            if not has_preview:
                selected_tools.append("search_memories")
        if "structured_fact" in open_types and not any(tr.get("tool") == "query_memory_facts" for tr in tool_results):
            selected_tools.append("query_memory_facts")

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
