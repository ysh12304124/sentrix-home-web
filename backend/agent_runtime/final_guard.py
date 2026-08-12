"""L1 FinalGuard — 结构性兜底（Phase H H7 拍板后的最小集合）。

设计原则：回答"是否合格"（事实正确性、数字/词语等价、编造/矛盾/漏报）由 L2 模型评审
（judge.py）判断，L1 不做词语/数字正则的合格性匹配——例如"三个人"与"3"是等价表达，
不得由死代码判错。

L1 保留的确定性检查分两类：
1. 纯结构兜底（模型无法兜底）：
   - Safety Hard Block：权限/范围越权、内部 ID/表结构泄漏、非法写操作（不可放行）。
   - 模板占位符泄漏：[地点名称]/[数量] 等未填占位直接泄漏给用户。
   - 未调用任何检索工具却声称"没有找到相关记录"（流程结构错误，必须纠正后重试检索）。
   - all 请求但结果还有更多（交付不完整）。
   - 声称全部交付但 delivered 数不足（交付矛盾）。
2. 最小确定性存在性检查（Truth Guard，可恢复，不是词语等价判断）：
   - 工具确认存在（total>0 / exists=True）但回答整体否认存在 → omission / exists 矛盾。
   - 工具确认不存在（total=0 / exists=False）但回答明确断言已交付/找到 → fabrication。
   这两类是可确定性验证的存在性矛盾（"有结果却说没找到/没结果却说找到"），
   不涉及"三个人 vs 3"、同义词、日期格式等语义等价——等价性仍全部由 L2 模型判断。
   诚实的不确定性（"无法确认/还不能确定"）与条件级否认（"没找到能确认爬山
   的记录"）不算整体否认，不做拦截。

运行时异常/空观察/预算耗尽 → runtime 的 emergency/natural partial 路径，不在本模块。
"""

from __future__ import annotations

import re

from .guard_types import (REVISION_HARD_BLOCK, REVISION_REWRITE_ONLY,
                          GuardIssue, GuardResult)

# 检索/事实类工具（存在性判断只针对这些工具的结果）
_RETRIEVAL_TOOLS = {"search_memories", "query_memory_facts", "search_conversation_history",
                    "get_core_memory", "get_person_memory", "get_result_page"}

# 整体否认存在（带明确宾语/动作）：total>0 / exists=True 时回答却说没找到。
# 只匹配"明确否认找到照片/记录"这类整体否定，不含"我没去过北京"这类出行否定。
_DENY_EXISTS = re.compile(
    r"没(?:有)?找到|未找到|找不到|查无|"
    r"没有(?:任何|符合|相关|一张|一个|一条|拍到|拍过|去过)?(?:照片|记录|回忆|记忆|相关)", re.I)
# 只是不确定/hedge，不算整体否认（"无法确认/还不能确定/可能没有"等）
_HEDGE = re.compile(
    r"无法确认|不能确认|不确定|还不能|暂时|无法判断|记不清|可能没有|未必|没有完全|不能确定", re.I)
# 条件级否认："没找到能确认'爬山'的记录"≠"没有找到照片"，不算整体否认
_CONDITION_LEVEL_DENY = re.compile(
    r"没(?:有)?找到(?:能|办法|足够|明确|可以){0,2}(?:确认|确定|验证|证实)", re.I)
# 明确交付/找到断言（total=0 / exists=False 时回答却断言已找到/交付）
_FOUND_DELIVERY = re.compile(
    r"(?:已|为)?(?:为您|为你)?找到|已找到|找到了|"
    r"找到\s*\d+\s*张|这是(?:您|你)要找的|这就是(?:您|你)?要的|"
    r"有(?:一张|两张|几张|相关|这些)?(?:照片|记录)", re.I)
# 整句否定门槛：含明确否定词时不视为"断言找到/交付"
_ANY_DENIAL = re.compile(r"没|未|无|不|无法|尚未", re.I)


def _natural_message(code: str, detail: str = "") -> str:
    """把内部规则码转成用户可读、模型可执行的恢复文案（D4：用户不看到内部规则名）。"""
    base = {
        "placeholder_leak": "回答里出现了'地点名称/数量/时间'这类未填写的占位符，必须替换成真实数据或删除",
        "denial_without_search": "你的回答声称没有找到相关记录，但本轮没有调用任何检索工具。请先调用 search_memories 或 query_memory_facts 完成检索，再基于工具结果回答",
        "all_requested_but_has_more": "用户要求全部结果，但结果集还有更多未交付。请继续取回剩余结果，或如实说明只交付了部分",
        "delivery_contradiction": "你的回答声称已经全部交付，但实际没有交付任何结果。请如实说明交付情况",
        "omission_conflict": "工具确认存在相关结果（total>0），但你的回答却说没有找到。请基于工具结果如实回答存在的情况，不要整体否认",
        "fabrication_from_empty": "工具确认不存在相关结果（total=0），但你的回答却声称找到了/已交付。请删除没有证据的找到/交付断言",
        "fact_exists_contradiction": "工具确认存在相关记录，但你的回答却说没有找到。请改为如实说明已确认存在",
        "fact_exists_contradiction_false": "工具确认不存在相关记录，但你的回答却声称找到了。请删除没有证据的找到断言",
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
        if not any((tr.get("tool") or "") in _RETRIEVAL_TOOLS for tr in tool_results)                 and _DENY_EXISTS.search(answer):
            issues.append(_issue("denial_without_search"))
        # H7：最小确定性存在性检查（Truth Guard，可恢复）——total 与回答的存在性断言矛盾
        issues.extend(self._check_existence(answer, task_state))
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
    def _check_existence(answer: str, task_state: dict) -> list[GuardIssue]:
        """最小确定性存在性检查：工具确认存在/不存在 vs 回答的整体否认/交付断言。

        只做可确定性验证的存在性矛盾，不判断任何词语等价：
        - 任一检索工具 total>0 且回答整体否认存在 → omission_conflict
        - 任一检索工具 total=0 且回答明确断言已找到/交付 → fabrication_from_empty
        - query_memory_facts exists=True 且回答整体否认 → fact_exists_contradiction
        - query_memory_facts exists=False 且回答明确断言已找到 → fact_exists_contradiction_false
        诚实不确定性（hedge）与条件级否认不触发。
        """
        answer = (answer or "").strip()
        if not answer:
            return []
        issues: list[GuardIssue] = []
        tool_results = task_state.get("tool_results") or []
        # 整体否认（排除 hedge 与条件级否认）
        denies = bool(_DENY_EXISTS.search(answer))             and not _HEDGE.search(answer)             and not _CONDITION_LEVEL_DENY.search(answer)
        # 明确交付/找到断言（排除否定句："没有找到"不算）
        found_claim = bool(_FOUND_DELIVERY.search(answer)) and not _ANY_DENIAL.search(answer)

        retrieval_rows = [tr for tr in tool_results
                          if (tr.get("tool") or "") in _RETRIEVAL_TOOLS
                          and tr.get("total") is not None]
        if any((tr.get("total") or 0) > 0 for tr in retrieval_rows) and denies:
            issues.append(_issue("omission_conflict",
                                 f"tool={[tr.get('tool') for tr in retrieval_rows if (tr.get('total') or 0) > 0][:2]}"))
        if any((tr.get("total") or 0) == 0 for tr in retrieval_rows) and found_claim:
            issues.append(_issue("fabrication_from_empty",
                                 f"tool={[tr.get('tool') for tr in retrieval_rows if (tr.get('total') or 0) == 0][:2]}"))

        # query_memory_facts exists 操作的确定性事实
        if task_state.get("last_tool") == "query_memory_facts"                 and task_state.get("fact_operation") == "exists":
            expected = task_state.get("fact_value")
            if expected is True and denies:
                issues.append(_issue("fact_exists_contradiction", "expected=True"))
            elif expected is False and found_claim:
                issues.append(_issue("fact_exists_contradiction_false", "expected=False"))
        return issues

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
