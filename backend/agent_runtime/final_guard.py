"""L1 FinalGuard — 结构性兜底（Phase H H7 拍板后的最小集合）。

设计原则：回答"是否合格"（事实正确性、数字/词语等价、编造/矛盾/漏报）全部由 L2 模型评审
（judge.py）判断，L1 不再用词语/数字正则做合格性匹配——例如"三个人"与"3"是等价表达，
不得由死代码判错。

L1 只保留无法靠模型兜底的结构性检查：
- Safety Hard Block：权限/范围越权、内部 ID/表结构泄漏、非法写操作（不可放行）。
- 模板占位符泄漏：[地点名称]/[数量] 等未填占位直接泄漏给用户。
- 未调用任何检索工具却声称"没有找到相关记录"（流程结构错误，必须纠正后重试检索）。
- all 请求但结果还有更多（交付不完整）。
- 声称全部交付但 delivered 数不足（交付矛盾）。

事实性判断（编造/矛盾/漏报/过度声称/缺口披露）→ L2 judge；运行时异常/空观察/预算耗尽
→ runtime 的 emergency/natural partial 路径，均不在本模块。
"""

from __future__ import annotations

import re

from .guard_types import (REVISION_HARD_BLOCK, REVISION_REWRITE_ONLY,
                          GuardIssue, GuardResult)

# D12：记录级否认（"没有找到相关记录"），用于"未检索就声称找不到"检查；
# 不含"我没去过北京"这类出行否定。
_DENY_RECORDS = re.compile(
    r"没(?:有)?找到|未找到|找不到|查无|没有(?:任何|符合|相关|一张|一个|一条|拍到|拍过)?(?:照片|记录|回忆|记忆)", re.I)


def _natural_message(code: str, detail: str = "") -> str:
    """把内部规则码转成用户可读、模型可执行的恢复文案（D4：用户不看到内部规则名）。"""
    base = {
        "placeholder_leak": "回答里出现了'地点名称/数量/时间'这类未填写的占位符，必须替换成真实数据或删除",
        "denial_without_search": "你的回答声称没有找到相关记录，但本轮没有调用任何检索工具。请先调用 search_memories 或 query_memory_facts 完成检索，再基于工具结果回答",
        "all_requested_but_has_more": "用户要求全部结果，但结果集还有更多未交付。请继续取回剩余结果，或如实说明只交付了部分",
        "delivery_contradiction": "你的回答声称已经全部交付，但实际没有交付任何结果。请如实说明交付情况",
    }
    text = base.get(code, f"回答与工具结果不一致（{code}）")
    if "{expected}" in text:
        text = text.replace("{expected}", detail or "?")
    elif detail:
        text = f"{text}：{detail}"
    return text


def _issue(code: str, detail: str = "", *, revision=REVISION_REWRITE_ONLY,
           tool_ref=None, trusted_facts=None) -> GuardIssue:
    return GuardIssue(code=code, message=_natural_message(code, detail),
                      revision=revision, tool_ref=tool_ref,
                      trusted_facts=trusted_facts or [])


class FinalGuard:
    def __init__(self, *, scope_id="home-default", viewer_id="owner", result_sets=None):
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.result_sets = result_sets or {}

    def check(self, answer: str, *, task_state=None, delivered_count=None,
              observation_ids_seen=None) -> GuardResult:
        issues: list[GuardIssue] = []
        answer = answer or ""
        task_state = task_state or {}
        # G4：安全层独立校验（权限/内部结构/非法写）——任何命中都优先于恢复逻辑
        issues.extend(self._check_safety(answer, task_state))
        # 模板占位符泄漏（[地点名称1]/[数量]/[此处填入…店名] 等未填占位）
        if re.search(r"\[[^\[\]]{0,48}(?:填入|此处|占位|待填|名称|名字|姓名|朋友|数量|时间|地点|内容|数字|照片|记录|店名|电话|价格|日期|金额)[^\[\]]{0,36}\]", answer):
            issues.append(_issue("placeholder_leak"))
        # D12：声称"没有找到记录"但本轮从未调用任何检索工具 → 必须纠正后重试检索。
        tool_results = task_state.get("tool_results") or []
        retrieval_tools = {"search_memories", "query_memory_facts", "search_conversation_history",
                           "get_core_memory", "get_person_memory", "get_result_page"}
        if not any((tr.get("tool") or "") in retrieval_tools for tr in tool_results) \
                and _DENY_RECORDS.search(answer):
            issues.append(_issue("denial_without_search"))
        # all 请求但交付不完整
        mode = (task_state or {}).get("result_mode")
        has_more = (task_state or {}).get("has_more")
        if mode == "all" and has_more:
            issues.append(_issue("all_requested_but_has_more"))
        # 声称全部交付但 delivered 数不足
        if task_state.get("delivery_state") == "complete" and delivered_count == 0:
            issues.append(_issue("delivery_contradiction"))
        return GuardResult(issues)

    @staticmethod
    def _check_safety(answer: str, task_state: dict) -> list[GuardIssue]:
        """G4 Safety Hard Block：权限/范围越权、内部结构泄漏、非法写操作。

        这一层只处理不可放行的安全边界；命中即 hard_block，不走恢复。
        当前单 owner 场景下多为防御性规则，保持最小集合。
        """
        issues: list[GuardIssue] = []
        # 内部标识符显式出现在回答里
        leaks = re.findall(r"\b(asset_|obs_|entity_|mention_|claim_|turn_|conversation_)[a-f0-9]{6,}\b", answer)
        if leaks:
            issues.append(_issue("internal_id_leak", f"{sorted(set(leaks))[:3]}",
                                 revision=REVISION_HARD_BLOCK))
        if re.search(r"\b(assets|observations|entity_mentions|semantic_claims|agent_result_sets)\b", answer):
            issues.append(_issue("table_name_leak", "", revision=REVISION_HARD_BLOCK))
        # 显式内部字段名（代码级标识符）不应出现在用户可见回答
        if re.search(r"\b(scope_id|viewer_id|read_write|allowed_tools|max_model_len)\s*[:=]", answer):
            issues.append(_issue("table_name_leak", "internal_schema", revision=REVISION_HARD_BLOCK))
        # 越权提示：回答中声称可访问/看到其他相册、他人私有内容（防御性）
        if re.search(r"别的?相册|他人|别的?人[的]?(?:相册|照片|隐私)|其他用户", answer):
            issues.append(_issue("viewer_escape", "", revision=REVISION_HARD_BLOCK))
        # 非法写操作：回答中声称进行了写/删/改（runtime 只读）
        if re.search(r"(?:已经|帮你|可以)(?:写入|删除|修改|覆盖|清空)(?:了|过)?", answer):
            issues.append(_issue("write_not_allowed", "", revision=REVISION_HARD_BLOCK))
        return issues
