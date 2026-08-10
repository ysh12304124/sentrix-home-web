"""B4 — Emergency Renderer：tool-loop 未正常完成时的确定性诚实收尾（§16）。

只在 budget 耗尽 / 解析失败 / guard 冲突 / 工具被拒等失败路径触发；
输出只能基于已有 Tool Observation / TaskState，不生成新事实、不冒充完整回答。
状态必须保留：partial / runtime_timeout / guard_failure。
"""

from __future__ import annotations


def render_emergency_summary(task_state: dict, *, reason: str = "") -> str:
    """基于 TaskState 生成诚实的部分结果摘要。"""
    parts = []
    fact_total = task_state.get("fact_total")
    fact_operation = task_state.get("fact_operation")
    fact_value = task_state.get("fact_value")
    if fact_total is not None:
        if fact_operation in {"count", "media"}:
            parts.append(f"已确认{fact_operation}结果为 {fact_value}（共 {fact_total} 条记录）。")
        elif fact_operation in {"first", "last", "date"} and fact_value:
            parts.append(f"已确认{fact_operation}为 {fact_value}。")
        elif fact_operation == "exists":
            parts.append("已确认存在相关记录。" if fact_value is True else "已确认不存在相关记录。")
    result_total = task_state.get("result_total")
    current_rs = task_state.get("current_result_set")
    satisfaction = task_state.get("search_satisfaction")
    if result_total is not None:
        if result_total == 0:
            parts.append("检索未找到符合条件的照片。")
        else:
            remaining = task_state.get("result_remaining") or 0
            parts.append(f"已找到 {result_total} 张候选照片，还有 {remaining} 张未查看。")
            if satisfaction == "candidate_only":
                parts.append("目前只是相似候选，还不能确认完全匹配。")
            elif satisfaction == "partial_support":
                parts.append("部分条件已确认，仍有条件未完全确认。")
    for tr in task_state.get("tool_results") or []:
        if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip():
            parts.append(f"照片复核：{tr['inspect_text']}")
        if tr.get("tool") == "get_original_photos" and tr.get("total"):
            parts.append("原图交付已授权。")
    if not parts:
        parts.append("这次处理没有完成，未产生可确认的结果。")
    tail = "这次没有完成全部处理，可以继续问我。"
    if reason:
        tail = f"这次因{reason}提前结束，可以继续问我。"
    return "；".join(parts) + tail
