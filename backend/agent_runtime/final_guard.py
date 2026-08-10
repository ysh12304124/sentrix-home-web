"""MVP FinalGuard（v2 §17 / Phase C C2）— 确定性可证明范围。

检查：权限 / 内部 ID 泄漏 / Tool Result 引用 / 交付一致性（all vs has_more）/
写授权。人物长总结等多事实 claim 验证后移 A6，不在 MVP。

Phase C：返回 GuardResult（codes 兼容旧调用，携带结构化 GuardIssue 与自然文案），
recoverable 问题交给 runtime 走"可信事实 + 重写"恢复循环。
"""

from __future__ import annotations

import re

from .guard_types import (REVISION_HARD_BLOCK, REVISION_REWRITE_ONLY,
                          GuardIssue, GuardResult)

# 显式否认（带宾语），用于 exists=True / 有结果却说没找到
_DENY_WITH_OBJECT = re.compile(
    r"没有(?:拍|照|找(?:到)?|任何|相关|一张|一个|一条|过)?(?:.{0,6})?(?:照片|记录|回忆|记忆|相关|任何|拍过|去)?"
    r"|不存在|未找到|没找到|没拍过|查无|找不到", re.I)
# 只是不确定/hedge，不算否认
_HEDGE = re.compile(r"无法确认|不能确认|不确定|还不能|没有完全|暂时|无法判断|记不清|可能没有|未必", re.I)
# 正向断言找到（exists=False 用），排除 没/未/无 前缀
_POSITIVE_FOUND = re.compile(
    r"(?<!没)(?<!未)(?<!无)(?:有|找到|存在|拍了?)(?:相关|任何|一些)?(?:的)?(?:照片|记录|回忆|记忆|去)")
_FOUND_CLAIM = re.compile(r"找到|为您找到|有.{0,10}(照片|记录)")


def _natural_message(code: str, detail: str = "") -> str:
    """把内部规则码转成用户可读、模型可执行的恢复文案（D4：用户不看到内部规则名）。"""
    base = {
        "fact_value_missing": "工具确认的数量是 {expected}，但你的回答没有体现这个数字",
        "fact_date_missing": "工具确认的时间是 {expected}，但你的回答没有提到",
        "fact_exists_contradiction": "工具确认存在相关记录，但你的回答却说没有找到",
        "fact_exists_contradiction_false": "工具确认不存在相关记录，但你的回答却声称找到了",
        "group_fabrication": "你的回答出现了工具结果里没有的月份或分组",
        "group_no_evidence": "你描述了分组结论，但工具结果里没有可支持的分组数据",
        "internal_id_leak": "回答里出现了内部编号，用户不应看到",
        "table_name_leak": "回答里提到了内部数据表名",
        "all_requested_but_has_more": "用户要求全部结果，但还有更多结果没有交付",
        "delivery_contradiction": "交付状态与回答不一致",
        "fabrication_from_empty": "没有检索到任何结果，回答里却出现了具体事实",
        "fabrication_from_empty_ref": "你引用的检索没有返回结果，却声称找到了内容",
        "omission_conflict": "工具实际返回了候选结果，但你的回答却说没有找到",
        "candidate_claimed_as_match": "只是相似候选，回答却把它说成了确认结果",
        "missing_disclosure": "检索只是部分确认或相似候选，回答没说明还不能完全确认",
        "inspection_fabrication": "照片复核没有产生可确认的观察，回答却断言了照片里的细节",
        "inspect_evidence_contradicted": "照片复核已经有观察，回答却否认看到了内容",
        "inspect_observation_contradicted": "照片复核的观察与你的回答直接矛盾",
        "certainty_upgrade": "有条件没有确认，回答却说得过于确定",
        "judge_unfaithful": "评审认为回答与工具观察不一致",
        "placeholder_leak": "回答里出现了'地点名称/数量/时间'这类未填写的占位符，必须替换成真实数据或删除",
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
        issues.extend(self._check_faithfulness(answer, task_state))
        # 0. query_memory_facts 事实一致性：final 必须包含工具返回的 value
        if task_state.get("last_tool") == "query_memory_facts":
            op = task_state.get("fact_operation")
            expected = task_state.get("fact_value")
            if op in {"count", "media"} and isinstance(expected, int):
                if expected == 0 and re.search(r"没有|未拍|零", answer):
                    pass
                elif not re.search(rf"(?<!\d){expected}(?!\d)", answer):
                    issues.append(_issue("fact_value_missing", f"expected={expected}"))
            elif op in {"first", "last", "date"} and isinstance(expected, str):
                date_part = expected[:10]
                nums = re.findall(r"\d+", date_part)
                if nums and not all(self._date_num_ok(n, answer) for n in nums):
                    issues.append(_issue("fact_date_missing", f"expected={date_part}"))
            elif op == "exists":
                if expected is True:
                    # 只有显式否认（带宾语/动作）才算矛盾；"没有完全确认/无法确认"这类 hedge 放行
                    if not _HEDGE.search(answer) and _DENY_WITH_OBJECT.search(answer):
                        issues.append(_issue("fact_exists_contradiction", "expected=True"))
                elif expected is False:
                    if _POSITIVE_FOUND.search(answer) and not _DENY_WITH_OBJECT.search(answer):
                        issues.append(_issue("fact_exists_contradiction_false", "expected=False"))
            elif op in {"group", "meal"}:
                issues.extend(self._check_group(answer, task_state))
        # 0.5 模板占位符泄漏（[地点名称1]/[数量] 等未填占位）
        if re.search(r"\[[^\[\]]{0,14}(?:名称|数量|时间|地点|内容|数字|照片|记录)[^\[\]]{0,14}\]", answer):
            issues.append(_issue("placeholder_leak"))
        # 1. 内部 ID 泄漏（asset_/obs_/entity_ 前缀 + 内部表名）
        leaks = re.findall(r"\b(asset_|obs_|entity_|mention_|claim_|turn_)[a-f0-9]{6,}\b", answer)
        if leaks:
            issues.append(_issue("internal_id_leak", f"{sorted(set(leaks))[:3]}",
                                 revision=REVISION_HARD_BLOCK))
        if re.search(r"\b(assets|observations|entity_mentions|semantic_claims)\b", answer):
            issues.append(_issue("table_name_leak", "", revision=REVISION_HARD_BLOCK))
        # 2. all 请求但交付不完整
        mode = (task_state or {}).get("result_mode")
        has_more = (task_state or {}).get("has_more")
        if mode == "all" and has_more:
            issues.append(_issue("all_requested_but_has_more"))
        # 3. 声称全部交付但 delivered 数不足
        if task_state.get("delivery_state") == "complete" and delivered_count == 0:
            issues.append(_issue("delivery_contradiction"))
        # 4. 明确无证据却给出具体事实
        if task_state.get("fulfillment") == "empty" and re.search(r"\d+\s*张|第一次|最后一次", answer):
            issues.append(_issue("fabrication_from_empty"))
        # 5. 检索为空却声称找到了照片/记录
        if task_state.get("fulfillment") == "empty" and not re.search(r"没|未", answer):
            if _FOUND_CLAIM.search(answer):
                issues.append(_issue("fabrication_from_empty"))
        return GuardResult(issues)

    @staticmethod
    def _check_faithfulness(answer: str, task_state: dict) -> list[GuardIssue]:
        """B2.1 Observation Faithfulness：omission / certainty upgrade / disclosure。"""
        issues: list[GuardIssue] = []
        answer = (answer or "").strip()
        refs = set(task_state.get("evidence_refs") or [])
        tool_results = task_state.get("tool_results") or []
        satisfaction = task_state.get("search_satisfaction")
        condition_summary = task_state.get("search_condition_summary") or {}
        denies = bool(__import__("re").search(
            r"没(?:有)?找到|未找到|不存在|没有(?:任何|符合|相关|一张|一个|一条)?(?:照片|记录|回忆|记忆)|查无|"
            r"无法看到.{0,6}(?:照片|内容)|看不到任何照片|无法查看(?:照片|内容)|没有(?:可|能看|看过).{0,4}照片|"
            r"没有去过任何(?:地方|地点|城市)|没去过任何(?:地方|地点|城市)|"
            r"没有(?:任何|一个|什么|一处|哪个)(?:去过)?(?:地方|地点|城市)", answer))
        # 1) 有结果却声称没有：非空结果 + 明确否认存在 → omission（与是否引用无关）
        # C8：条件级否认不算整体否认——"没找到能确认'爬山'的记录"≠"没有找到照片"
        condition_level_deny = bool(re.search(
            r"没(?:有)?找到(?:能|办法|足够|明确){0,2}(?:确认|确定|验证|证实)", answer))
        for tr in tool_results:
            total = tr.get("total")
            if total is None:
                continue
            if total > 0 and denies and satisfaction != "no_match" and not condition_level_deny:
                issues.append(_issue("omission_conflict",
                                     f"tool={tr.get('tool')},total={total}"))
            if tr.get("tool_call_id") in refs and total == 0 and not denies:
                if _FOUND_CLAIM.search(answer):
                    issues.append(_issue("fabrication_from_empty_ref"))
        inspect_texts = [tr.get("inspect_text") for tr in tool_results
                         if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip()]
        selected_handle = task_state.get("selected_asset_handle")
        # C8：inspect 只确认照片里直接可见的视觉细节，不能反向确认检索条件。
        # 用户明确点选某张照片追问时（selected_handle 存在），该照片的视觉回答以 inspect 为准，豁免。
        # 2) candidate_only 声称完全匹配（inspect 观察存在也不豁免"找到了/确认是"检索条件）
        if satisfaction == "candidate_only" and not selected_handle:
            if re.search(r"确认|确定|就是|肯定是|找到了|确认是", answer) and \
               not re.search(r"不能确认|无法确认|候选|未确认|不确定|接近|类似|还不能|没有直接证据", answer):
                issues.append(_issue("candidate_claimed_as_match"))
        # 3) partial/candidate 必须披露检索层缺口（inspect 视觉观察不替代检索层披露；点选追问除外）
        if satisfaction in {"partial_support", "candidate_only"} and not (inspect_texts and selected_handle):
            if not re.search(r"不能确认|无法确认|候选|未确认|不确定|接近|类似|还不能|没有直接证据|还需要|无法完全", answer):
                issues.append(_issue("missing_disclosure"))
        # 4.5) inspect 被拒/无观察却断言视觉细节 → inspection_fabrication
        inspect_results = [tr for tr in tool_results if tr.get("tool") == "inspect_photo"]
        for tr in inspect_results:
            if tr.get("blocked") and not (tr.get("inspect_text") or "").strip():
                # 无观察时：只有未加不确定措辞的视觉断言才算编造（si07 类已 hedge 的不算）
                if not re.search(r"无法确认|不能确认|不确定|无法判断|无法看到|看不清|看不清楚", answer):
                    if re.search(
                            r"桌上(?:放着|放了|摆着|有)|穿着|写着|天气(?:是|为|很)|有(?:雪|猫|小孩)|"
                            r"是(?:雪|猫|小孩)|(?:外套|衣服|招牌).{0,6}(?:是|穿|写)", answer):
                        issues.append(_issue("inspection_fabrication",
                                             f"blocked={tr.get('blocked')}"))
            # 4.6) inspect 有真实观察，但 Agent 却否认看到 → inspect_evidence_contradicted
            elif (tr.get("inspect_text") or "").strip():
                obs_text = tr["inspect_text"]
                if re.search(r"无法看到|看不到|没有看到任何|无法查看|没有(?:任何|可看)的照片|看不到任何照片", answer):
                    issues.append(_issue("inspect_evidence_contradicted"))
                # 4.7) 观察否定存在，回答却断言存在 → 观察与回答直接矛盾（编造）
                neg_obs = re.search(
                    r"没有(?:出现|看到|找到)?(?:任何|一个|一只)?(?:人|猫|小孩)|没有.{0,6}(?:人|猫|小孩)|无(?:人|猫|小孩)", obs_text)
                if neg_obs:
                    # 已 hedge/否定语境（"无法确认有人""并没有出现人"）不算编造
                    hedged = re.search(
                        r"无法确认|不能确认|不确定|无法判断|无法看到|看不清|看不清楚|没有|并无|未出现|未看到|没看到", answer)
                    if not hedged:
                        positive_claim = re.search(
                            r"有(?:一个|一位|两只|一只|几位|几个人)?[^，。]{0,8}(?:人|猫|小孩|红衣|红衣服)|"
                            r"穿着.{0,4}(?:红色|白色|黑色|蓝色|黄色|外套)|看到了?.{0,6}(?:人|猫|小孩)|"
                            r"确认.{0,8}(?:人|猫|小孩)|"
                            r"(?:人|猫|小孩|雪|山|外套).{0,4}(?:是|为).{0,4}(?:白色|红色|黑色|蓝色|黄色|绿色|灰色|棕色|粉色|紫色)", answer)
                        if positive_claim:
                            issues.append(_issue("inspect_observation_contradicted"))

        # 4) 条件明确 unknown/contradicted 却升级为 confirmed（标签感知：只拦"条件本身被确认"的表述，
        #    不误伤"我确认照片里没有雪"这类视觉断言）
        if condition_summary:
            unresolved = [k for k, v in condition_summary.items() if v in {"unknown", "contradicted"}]
            if unresolved:
                confirm = None
                for _m in re.finditer(r"确认是|确定是|肯定是|就是|确认了|确定了|确认到|确认过", answer):
                    before = answer[max(0, _m.start() - 4):_m.start()]
                    if not re.search(r"没|未|不|还|难|无法|不能", before):
                        confirm = _m
                        break
                label_assert = any(str(label) and str(label) in answer for label in unresolved) or \
                    bool(re.search(r"(?:这些|上述|几个)?条件.{0,8}(?:确认|确定|成立)", answer))
                if confirm and label_assert:
                    issues.append(_issue("certainty_upgrade", f"conditions={unresolved[:3]}"))
        return issues

    @staticmethod
    def _check_group(answer: str, task_state: dict) -> list[GuardIssue]:
        """group 结果一致性：回答中的月份必须都在 rows 内；没有任何 row 证据时不得断言具体分组。"""
        issues: list[GuardIssue] = []
        rows = task_state.get("fact_rows") or []
        group_by = task_state.get("fact_group_by") or "month"
        if not rows:
            return issues
        valid_months = set()
        labels = set()
        for row in rows:
            group = str(row.get("group") or "")
            labels.add(group)
            m = re.match(r"\d{4}-(\d{2})", group)
            if m:
                valid_months.add(int(m.group(1)))
        found_months = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})月", answer)]
        if found_months and valid_months:
            bad = sorted({m for m in found_months if m not in valid_months})
            if bad:
                issues.append(_issue("group_fabrication", f"months={bad}"))
        # C10：place 分组 / meal 食物分组的列举项必须在 rows 内（编造地点/食物 -> group_fabrication）
        if group_by == "place" or any(str(r.get("group") or "") for r in rows):
            known = {str(r.get("group") or "") for r in rows if str(r.get("group") or "")}
            items = FinalGuard._extract_list_items(answer, r"去过|去了|到过|前往|游玩|参观|出差|旅游")
            bad = [it for it in items if it and not any(it == k or it in k or k in it for k in known)]
            if bad:
                issues.append(_issue("group_fabrication", f"places={bad}"))
        food_rows = [r for r in rows if "food" in r]
        if not food_rows:
            value = task_state.get("fact_value") or []
            food_rows = [r for r in value if isinstance(r, dict) and "food" in r]
        if food_rows:
            known = {str(r.get("food") or "") for r in food_rows if str(r.get("food") or "")}
            items = FinalGuard._extract_list_items(answer, r"吃过|吃了|喝过|喝了|点了|吃：|吃:|喝过:|吃过:")
            bad = [it for it in items if it and not any(it == k or it in k or k in it for k in known)]
            if bad:
                issues.append(_issue("group_fabrication", f"foods={bad}"))
        if re.search(r"没|未", answer):
            return issues
        has_evidence = bool(valid_months & set(found_months)) or any(
            label and label in answer for label in labels)
        if not has_evidence and re.search(r"主要|包括|有照片|拍了|月份|地点|地方", answer):
            issues.append(_issue("group_no_evidence"))
        return issues

    @staticmethod
    def _extract_list_items(answer: str, keywords: str) -> list[str]:
        """从动词/关键词后的列举段提取短名词项（用于校验 place/meal 列举是否在工具结果内）。"""
        items = []
        for m in re.finditer(keywords, answer):
            tail = answer[m.end():m.end() + 80]
            tail = re.split(r"[。；!！?\n]", tail)[0]
            for part in re.split(r"[、，,和及与]", tail):
                part = re.split(r"[（(]", part)[0]
                part = part.strip("：:、，,和及与 ").strip()
                if 1 <= len(part) <= 8 and re.match(r"^[\u4e00-\u9fa5A-Za-z0-9]+$", part) \
                        and part not in {"以下地方", "这些地方", "以下几个", "以下", "如下",
                                         "以上", "这些", "几个", "一些地方", "一些", "几个地方",
                                         "的地方包括", "地方包括", "的地方", "包括", "去过的地方",
                                         "去过的地方包括", "有"}:
                    items.append(part)
        # 子弹列表："- 杭州市：150条记录"
        for m in re.finditer(r"(?:^|\n)\s*[-*•]\s*([\u4e00-\u9fa5A-Za-z0-9]{1,8})[：:]", answer):
            item = m.group(1)
            if item not in items and item not in {"以下地方", "这些地方", "以下几个", "以下", "如下",
                                                  "以上", "这些", "几个", "一些地方", "一些", "几个地方",
                                                  "的地方包括", "地方包括", "的地方", "包括", "去过的地方",
                                                  "去过的地方包括", "有"}:
                items.append(item)
        return items

    @staticmethod
    def _date_num_ok(num: str, answer: str) -> bool:
        if num in answer:
            return True
        if len(num) > 1 and num[0] == "0" and num[1:] in answer:
            return True
        return False
