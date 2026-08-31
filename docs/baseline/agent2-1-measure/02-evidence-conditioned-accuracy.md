# Agent 2.1 测量报告 02 — Evidence-Conditioned Answer Accuracy

基线 run：`20260820-003839-album3-gemma4-12b-it-agent2-1-ab`（38 题，gemma4-12b-it）
方法：对每题判定 `required_evidence_available`（回答所需关键事实是否已进入 Minimal Answer Context / `agent2_trace.answer_context.facts`）与 `final_answer_correct`（core = judge≥1）。GT=refuse 的 13 题不适用（N/A）。人工逐题核对 `answer_context.facts` 值。

> 只测量，未改 Agent 主链。此报告不新增线上 Agent 模型调用。

## 1. 四象限总表（25 题 GT=answer）

| Evidence 可用 | Answer 对 | 归因 | 数量 | 题目 |
| --- | --- | --- | --- | --- |
| ✓ | ✓ | 成功 | 10 | q01、q24-04、q24-05、q26-02、q26-06、q26-07、q40-01、q40-02、q40-04、q47-02 |
| ✓ | ✗ | synthesis failure | 2 | **q02、q26-01** |
| ✗ | ✗ | upstream failure | 13 | q08、q03、q24-01、q24-02、q24-07、q26-03、q47-01、q47-04、q47-03、q47-07、b2-01、b2-02、b2-03 |
| ✗ | ✓ | 猜对/标注问题 | 0 | — |

## 2. 核心指标

| 指标 | 值 |
| --- | --- |
| Required Evidence Availability Rate | **12/25 = 48.0%** |
| Evidence-Conditioned Core Accuracy | **10/12 = 83.3%** |
| Evidence-Conditioned Exact Accuracy | 5/12 = 41.7% |
| Synthesis Failure Rate（证据在但答错 / 全部可答题） | 2/25 = 8.0% |
| Upstream Failure Rate（证据没进 Context） | 13/25 = 52.0% |

## 3. Evidence 可用性 × 答案质量交叉

| Evidence | 题数 | Answer Quality Mean | Core Accuracy |
| --- | --- | --- | --- |
| ✓ | 12 | 1.25 | 83.3% |
| ✗ | 13 | 0.00 | 0% |

**关键结论**：证据可用时，答对率 83%；证据不可用时，答对率 0%。所以答案质量的主杠杆是**上游证据获取**，不是写作/合成。

## 4. 两个 synthesis failure 详解（证据在，却答错）

1. **q26-01（顶呱呱创始年份）**：`answer_context` 的 `visible_text` 里**明明已提取到 "1974年"**（参考答案=1974），但 Agent 最终拒答"照片里看不出来"，judge=0。→ 这是最典型的 **evidence 已到 Writer、Writer/决策没用它** 的合成/收尾失败。
2. **q02（找沙雕合影记录）**：`memory_asset` result set 已进 Context（照片已召回），但 `confirmed_identity`/`visual_observation` 没确认"图中是明明乐乐+沙雕"，Agent 未给出正确结论，judge=0。→ 属于部分证据到位但未绑定/未产出（E2 evidence binding + F1 收尾）。

## 5. 对"是否冻结 Writer"的判断

计划 Gate：Evidence-Conditioned Core Accuracy ≥ 95% 才冻结 Writer，≥ 90% 才过泛化门。
- 当前 **83.3% < 90%** → **不应冻结 Writer，也不应宣布合成层可收敛**。
- 但 13/25=52% 的失败是 **upstream（证据根本没进 Context）**，只有 2/25=8% 是 synthesis。
- 因此下一瓶颈的排序应是：**R1 检索召回 → V1 视觉理解 → V3 OCR**（上游），Writer/Verifier 不是首要矛盾。这和 Phase 2 的 finalization 结论一致：Agent 的可答性失败主要是"拿不到证据"，不是"拿到证据写错"。
- 唯一需要优先看的 Writer 场景是 q26-01 这类"证据已在 Context 却拒答"的案例——值得在 Error Pareto 里单独归因（P1 requirement planning / E3 answer-context compilation / F1 finalization 三类中的某类）。

## 6. 方法备注（供审阅）

- `required_evidence_available=true` 的标准：`answer_context.facts` 中存在与参考答案关键值一致/覆盖的事实。例如 q26-01 的 `visible_text` 含 "1974年" → true；q24-07 的 `visible_text` 是"长春黄旗路派出所 22048004"（≠ 参考 22048084/85）→ false（OCR 读错，key fact 未正确进入）。
- `answer_context.facts` 与 `evidence_ledger` 不同：ledger 只记 tool 产出的视觉/OCR 证据（平均 1.2 条/题），而 search_memories 的检索事实直接进 answer_context。因此本报告用 answer_context 而非 ledger 作为"evidence available"判定源。
- 13 题 GT=refuse（安全/拒答）不纳入四象限。
