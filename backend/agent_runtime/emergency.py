"""B4 / Phase C C3 — Emergency Renderer：tool-loop 未正常完成时的确定性诚实收尾（§16）。

只在 budget 耗尽 / 解析失败 / guard 冲突 / 工具被拒等失败路径触发；
输出只能基于已有 Tool Observation / TaskState，不生成新事实、不冒充完整回答。
文案面向用户自然化，禁止出现 guard 规则名 / 内部状态名（D4）。
"""

from __future__ import annotations

from .final_writer import sanitize_internal_refs

_OPERATION_LABEL = {
    "count": "数量",
    "media": "照片数量",
    "first": "最早时间",
    "last": "最近时间",
    "date": "时间",
    "exists": "是否存在",
}

_REASON_TAIL = {
    "回答未通过事实校验": "以上是我目前能确认的部分信息。",
    "guard_blocked": "以上是我目前能确认的部分信息。",
    "预算用尽": "这些是目前能确认的内容，其余信息暂时无法确定。",
    "输出无法解析": "这些信息还不足以确认你要的答案。",
    "动作无法识别": "这些信息还不足以确认你要的答案。",
    "重复调用被拒绝": "这些是目前能确认的内容。",
    "工具调用被拒绝": "这些是目前能确认的内容。",
}

_DEFAULT_TAIL = "这些信息还不足以确认你要的答案。"


def render_emergency_summary(task_state: dict, *, reason: str = "") -> str:
    """基于 TaskState 生成诚实的部分结果摘要（自然化文案）。"""
    parts = []
    fact_total = task_state.get("fact_total")
    fact_operation = task_state.get("fact_operation")
    fact_value = task_state.get("fact_value")
    if fact_total is not None:
        if fact_operation in {"count", "media"}:
            parts.append(f"查询到 {fact_value} 条符合条件的结果（共 {fact_total} 条记录）。")
        elif fact_operation in {"first", "last", "date"} and fact_value:
            label = _OPERATION_LABEL.get(fact_operation, "相关时间")
            parts.append(f"相关记录中，{label}是 {fact_value}。")
        elif fact_operation == "exists":
            parts.append("已确认存在相关记录。" if fact_value is True else "已确认不存在相关记录。")
    result_total = task_state.get("result_total")
    satisfaction = task_state.get("search_satisfaction")
    if result_total is not None:
        if result_total == 0:
            parts.append("检索没有找到符合条件的照片。")
        # A positive search total is retrieval telemetry, not a user answer
        # fact.  The bounded evidence preview is already exposed separately;
        # do not turn a broad candidate pool into "找到 N 张" fallback text.
        elif satisfaction == "candidate_only":
            parts.append("找到了一些相关照片，但还不能完全确认。")
        elif satisfaction == "partial_support":
            parts.append("找到了一些相关照片，部分信息能对上，还有细节不能完全确认。")
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            handle = tr.get("inspect_handle") or ""
            display_handle = sanitize_internal_refs(handle)
            if display_handle == "这张照片":
                parts.append(f"{display_handle}复核：{tr['inspect_text']}")
            else:
                parts.append(f"照片{(' ' + display_handle) if display_handle else ''}复核：{tr['inspect_text']}")
        if tr.get("tool") == "get_original_photos" and tr.get("total"):
            parts.append("原图交付已授权。")
    if not parts:
        parts.append("这次处理没有完成，没有产生可确认的结果。")
    tail = _REASON_TAIL.get(reason, _DEFAULT_TAIL if not reason else
                            f"这次处理因故没有完整完成，可以继续问我。")
    return sanitize_internal_refs("；".join(parts) + tail)
