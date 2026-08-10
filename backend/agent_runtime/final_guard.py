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
