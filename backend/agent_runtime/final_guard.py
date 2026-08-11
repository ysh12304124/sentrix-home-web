"""MVP FinalGuard（v2 §17）— 确定性可证明范围。

检查：权限 / 内部 ID 泄漏 / Tool Result 引用 / 交付一致性（all vs has_more）/
写授权。人物长总结等多事实 claim 验证后移 A6，不在 MVP。
"""

from __future__ import annotations

import re


class FinalGuard:
    def __init__(self, *, scope_id="home-default", viewer_id="owner", result_sets=None):
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.result_sets = result_sets or {}

    def check(self, answer: str, *, task_state=None, delivered_count=None,
              observation_ids_seen=None) -> list[str]:
        problems = []
        answer = answer or ""
        task_state = task_state or {}
        problems.extend(self._check_faithfulness(answer, task_state))
        # 0. query_memory_facts 事实一致性：final 必须包含工具返回的 value
        if task_state.get("last_tool") == "query_memory_facts":
            op = task_state.get("fact_operation")
            expected = task_state.get("fact_value")
            if op in {"count", "media"} and isinstance(expected, int):
                if expected == 0 and re.search(r"没有|未拍|零", answer):
                    pass
                elif not re.search(rf"(?<!\d){expected}(?!\d)", answer):
                    problems.append(f"fact_value_missing:expected={expected}")
            elif op in {"first", "last", "date"} and isinstance(expected, str):
                date_part = expected[:10]
                nums = re.findall(r"\d+", date_part)
                if nums and not all(self._date_num_ok(n, answer) for n in nums):
                    problems.append(f"fact_date_missing:expected={date_part}")
            elif op == "exists":
                if expected is True and re.search(r"没有|不存在|未找到|没找到", answer):
                    problems.append("fact_exists_contradiction:expected=True")
                elif expected is False and re.search(r"有照片|存在|找到", answer):
                    problems.append("fact_exists_contradiction:expected=False")
            elif op == "group":
                problems.extend(self._check_group(answer, task_state))
        # 1. 内部 ID 泄漏（asset_/obs_/entity_ 前缀 + 内部表名）
        leaks = re.findall(r"\b(asset_|obs_|entity_|mention_|claim_|turn_)[a-f0-9]{6,}\b", answer)
        if leaks:
            problems.append(f"internal_id_leak:{sorted(set(leaks))[:3]}")
        if re.search(r"\b(assets|observations|entity_mentions|semantic_claims)\b", answer):
            problems.append("table_name_leak")
        # 2. all 请求但交付不完整
        mode = (task_state or {}).get("result_mode")
        has_more = (task_state or {}).get("has_more")
        if mode == "all" and has_more:
            problems.append("all_requested_but_has_more")
        # 3. 声称全部交付但 delivered 数不足
        if task_state.get("delivery_state") == "complete" and delivered_count == 0:
            problems.append("delivery_contradiction")
        # 4. 明确无证据却给出具体事实
        if task_state.get("fulfillment") == "empty" and re.search(r"\d+\s*张|第一次|最后一次", answer):
            problems.append("fabrication_from_empty")
        # 5. 检索为空却声称找到了照片/记录
        if task_state.get("fulfillment") == "empty" and not re.search(r"没|未", answer):
            if re.search(r"找到|为您找到|有.{0,10}(照片|记录)", answer):
                problems.append("fabrication_from_empty")
        return problems

    @staticmethod
    def _check_faithfulness(answer: str, task_state: dict) -> list[str]:
        """B2.1 Observation Faithfulness：omission / certainty upgrade / disclosure。"""
        problems = []
        answer = (answer or "").strip()
        refs = set(task_state.get("evidence_refs") or [])
        tool_results = task_state.get("tool_results") or []
        satisfaction = task_state.get("search_satisfaction")
        condition_summary = task_state.get("search_condition_summary") or {}
        denies = bool(__import__("re").search(
            r"没(?:有)?找到|未找到|不存在|没有(?:任何|符合|相关|一张|一个|一条)?(?:照片|记录|回忆|记忆)|查无|"
            r"无法看到.{0,6}(?:照片|内容)|看不到任何照片|无法查看(?:照片|内容)|没有(?:可|能看|看过).{0,4}照片", answer))
        # 1) 有结果却声称没有：非空结果 + 明确否认存在 → omission（与是否引用无关）
        for tr in tool_results:
            total = tr.get("total")
            if total is None:
                continue
            if total > 0 and denies and satisfaction != "no_match":
                problems.append(f"omission_conflict:tool={tr.get('tool')},total={total}")
            if tr.get("tool_call_id") in refs and total == 0 and not denies:
                if __import__("re").search(r"找到|为您找到|有.{0,10}(照片|记录)", answer):
                    problems.append("fabrication_from_empty_ref")
        inspect_texts = [tr.get("inspect_text") for tr in tool_results
                         if tr.get("tool") == "inspect_photo" and (tr.get("inspect_text") or "").strip()]
        # 2) candidate_only 声称完全匹配（inspect 复核到具体观察的不算）
        if satisfaction == "candidate_only" and not inspect_texts:
            if __import__("re").search(r"确认|确定|就是|肯定是|找到了|确认是", answer) and \
               not __import__("re").search(r"不能确认|无法确认|候选|未确认|不确定|接近|类似|还不能|没有直接证据", answer):
                problems.append("candidate_claimed_as_match")
        # 3) partial/candidate 必须披露缺口（已通过 inspect 复核到具体观察的除外）
        if satisfaction in {"partial_support", "candidate_only"} and not inspect_texts:
            if not __import__("re").search(r"不能确认|无法确认|候选|未确认|不确定|接近|类似|还不能|没有直接证据|还需要|无法完全", answer):
                problems.append("missing_disclosure")
        # 4.5) inspect 被拒/无观察却断言视觉细节 → inspection_fabrication
        inspect_results = [tr for tr in tool_results if tr.get("tool") == "inspect_photo"]
        for tr in inspect_results:
            if tr.get("blocked") and not (tr.get("inspect_text") or "").strip():
                # 无观察时：只有未加不确定措辞的视觉断言才算编造（si07 类已 hedge 的不算）
                if not __import__("re").search(r"无法确认|不能确认|不确定|无法判断|无法看到|看不清|看不清楚", answer):
                    if __import__("re").search(
                            r"桌上(?:放着|放了|摆着|有)|穿着|写着|天气(?:是|为|很)|有(?:雪|猫|小孩)|"
                            r"是(?:雪|猫|小孩)|(?:外套|衣服|招牌).{0,6}(?:是|穿|写)", answer):
                        problems.append(f"inspection_fabrication:blocked={tr.get('blocked')}")
            # 4.6) inspect 有真实观察，但 Agent 却否认看到 → inspect_evidence_contradicted
            elif (tr.get("inspect_text") or "").strip():
                obs_text = tr["inspect_text"]
                if __import__("re").search(r"无法看到|看不到|没有看到任何|无法查看|没有(?:任何|可看)的照片|看不到任何照片", answer):
                    problems.append("inspect_evidence_contradicted")
                # 4.7) 观察否定存在，回答却断言存在 → 观察与回答直接矛盾（编造）
                neg_obs = __import__("re").search(
                    r"没有(?:出现|看到|找到)?(?:任何|一个|一只)?(?:人|猫|小孩)|没有.{0,6}(?:人|猫|小孩)|无(?:人|猫|小孩)", obs_text)
                if neg_obs:
                    # 已 hedge/否定语境（“无法确认有人”“并没有出现人”）不算编造
                    hedged = __import__("re").search(
                        r"无法确认|不能确认|不确定|无法判断|无法看到|看不清|看不清楚|没有|并无|未出现|未看到|没看到", answer)
                    if not hedged:
                        positive_claim = __import__("re").search(
                            r"有(?:一个|一位|两只|一只|几位|几个人)?[^，。]{0,8}(?:人|猫|小孩|红衣|红衣服)|"
                            r"穿着.{0,4}(?:红色|白色|黑色|蓝色|黄色|外套)|看到了?.{0,6}(?:人|猫|小孩)|"
                            r"确认.{0,8}(?:人|猫|小孩)|"
                            r"(?:人|猫|小孩|雪|山|外套).{0,4}(?:是|为).{0,4}(?:白色|红色|黑色|蓝色|黄色|绿色|灰色|棕色|粉色|紫色)", answer)
                        if positive_claim:
                            problems.append("inspect_observation_contradicted")

        # 4) 条件明确 unknown/contradicted 却升级为 confirmed
        if condition_summary:
            unresolved = [k for k, v in condition_summary.items() if v in {"unknown", "contradicted"}]
            if unresolved and __import__("re").search(r"确认|确定|肯定|就是", answer):
                problems.append(f"certainty_upgrade:conditions={unresolved[:3]}")
        return problems

    @staticmethod
    def _check_group(answer: str, task_state: dict) -> list[str]:
        """group 结果一致性：回答中的月份必须都在 rows 内；没有任何 row 证据时不得断言具体分组。"""
        rows = task_state.get("fact_rows") or []
        group_by = task_state.get("fact_group_by") or "month"
        if not rows:
            return []
        valid_months = set()
        labels = set()
        for row in rows:
            group = str(row.get("group") or "")
            labels.add(group)
            m = re.match(r"\d{4}-(\d{2})", group)
            if m:
                valid_months.add(int(m.group(1)))
        problems = []
        found_months = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})月", answer)]
        if found_months and valid_months:
            bad = sorted({m for m in found_months if m not in valid_months})
            if bad:
                problems.append(f"group_fabrication:months={bad}")
        if re.search(r"没|未", answer):
            return problems
        has_evidence = bool(valid_months & set(found_months)) or any(
            label and label in answer for label in labels)
        if not has_evidence and re.search(r"主要|包括|有照片|拍了|月份|地点|地方", answer):
            problems.append("group_no_evidence")
        return problems

    @staticmethod
    def _date_num_ok(num: str, answer: str) -> bool:
        if num in answer:
            return True
        if len(num) > 1 and num[0] == "0" and num[1:] in answer:
            return True
        return False
