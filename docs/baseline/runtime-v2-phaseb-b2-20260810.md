# Phase B B2/B2.1 — search_memories Truth Contract + Observation Faithfulness Guard

- 日期：2026-08-10
- 状态：✅ 代码完成；Shadow 18 实测 10→14 complete

## B2 — Truth Contract（工具层确定性计算）

- `search_memories` Observation 新增：`query_satisfaction`（`full_support/partial_support/candidate_only/no_match`）、`answerability`、`condition_summary`（每条件 `confirmed/supported/unknown/contradicted`）
- 判定逻辑 `_truth_contract()` 完全由代码基于 `packet.condition_results` 聚合，不依赖模型
- SYSTEM 提示新增表述规则：full=可确认；partial=必须披露缺口；candidate_only=禁止"找到了/确认是"，只能说"接近的候选，还不能完全确认"；no_match=禁止声称找到
- final action schema 新增 `evidence_refs`（本轮工具调用按 `tool_call_1`…编号）

## B2.1 — Observation Faithfulness Guard

`FinalGuard` 新增 4 类检查（保留原 count/date/group/exists/ID/empty 检查）：
1. `omission_conflict`：非空结果 + 明确否认存在（"没找到/不存在"）→ 拦截（s14 类）
2. `candidate_claimed_as_match`：candidate_only 却断言确认且无披露词
3. `missing_disclosure`：partial/candidate 必须含披露（不能确认/候选/接近/未确认…）
4. `certainty_upgrade`：条件 unknown/contradicted 却断言确认

Runtime 新增：Guard 冲突给模型**一次受控修正机会**（反馈具体冲突点），仍失败 → `blocked_by_guard`。工具预算耗尽不再遗留 `pending`（已修）。

## Shadow 18 实测（8105，scope=album2_e2b，ANN 已接线）

| 指标 | B0 基线 | B2/B2.1 |
|---|---:|---:|
| complete | 10/18 | **14/18** |
| avg/turn | 1.9s | 3.7s（含 guard 修正步 + ANN 检索） |
| s05/s06/s10 | 编造/空洞 | 诚实候选披露（"接近的候选，还不能完全确认"） |
| s15 | - | **guard 修正链拦截 12→66 编造**（工具值 66，模型先写 12，修正后引用工具值） |

## 已知遗留
- s18 类"帮我看看照片"宽泛请求：模型重复调用同一搜索（4 次）→ 预算耗尽 → 需 Emergency Renderer 兜底（B4）
- 时间表达式由模型自行解析（去年→2023 错误未拦）：工具层可加确定性"去年=2025"解析，Guard 无法校验语义年份（记入 B4 候选）
- 新增测试 `backend/tests/test_tool_loop_truth_contract.py`：10/10 通过
