# Agent 2.1 测量报告 03 — Stage Latency Waterfall + Loop Turn 指标

基线 run：`20260820-003839-album3-gemma4-12b-it-agent2-1-ab`（38 题，gemma4-12b-it）
数据源：`timing_breakdown`、`tool_trace`、`model_call_metrics`、`agent2_trace.budget_outcomes`、`answer_context.source_refs`
> 先测不优化；本报告不修改任何实现。

---

## 1. Phase 5 — Stage Latency Waterfall

### 1.1 总览（每题均值，ms）

| 阶段 | Mean(ms) | 占 agent 时间 | 说明 |
| --- | --- | --- | --- |
| **model（loop 内全部 LLM 调用）** | 9,441 | 47.3% | p50 11.6s / p95 16.8s / max 21.9s |
| **search_memories** | 5,966 | 29.9% | 41 次调用，单次 mean 5.53s / p95 7.65s / max 7.79s |
| **read_photo_text** | 3,979 | 19.9% | 8 次调用，单次 mean 18.9s，**含 76.98s 离群** |
| inspect_photo | 271 | 1.4% | 15 次调用，单次 mean 0.69s |
| get_result_page | ~0 | 0.0% | 1 次 |
| other / agent_overhead | 317 | 1.6% | |
| **agent 合计（不含 judge）** | **19,972** | 100% | p50 22.1s / p95 94.5s / max 100.6s |
| judge（回答质量+任务判断） | 3,099 | 加在 agent 之外 | wall_clock 合计 23,112 |

> 数值交叉验证：model 9,441 + search 5,966 + read_text 3,979 + inspect 271 + other 317 ≈ 19,974 ≈ agent 19,972 ✓

### 1.2 Top-3 延迟贡献者

1. **Loop LLM 调用（9.44s，47%）** —— 平均 6~7 轮模型步，每轮 ~1.4s。
2. **search_memories（5.97s，30%）** —— 每次 5.5s，含 ANN 检索（visual_ann 2.4s + text_ann 0.8s + 向量嵌入）。
3. **read_photo_text（3.98s，20%）** —— 单次 18.9s，主要是 VLM 读图；**76.98s 的离群调用**在 q24-02 类（店招 OCR）。

### 1.3 测量缺口（本 run 无法拆的更细粒度）

计划要求的 `planner_ms / final_writer_ms / evidence_ledger_ms / answer_context_ms / recovery_ms` **当前 run 没有单独记录**：
- `debug_trace` 的 planner/writer 步不含耗时字段；`model_call_metrics` 全部标记为 `role=tool_loop`，无法区分 planner/agent/writer。
- `model_ms` 是这些 LLM 调用的聚合，无法按阶段拆分。
- **结论**：要精确回答"25.3s 花在哪"，当前数据能定位到"model 47% + 两大 tool 50%"，但 planner/writer/ledger/context/recovery 的内部拆解需要给 debug_trace 加 stage timer 后再跑一次（Phase 5 的下一步就是加这个埋点，不改主链行为，只加计时）。

---

## 2. Phase 6 — 为什么平均 6.70 轮

### 2.1 每题的 loop 结构（30 题跑进 agent loop；8 题安全类直接短路无 loop）

| 指标（mean over 30 loop 题） | 值 |
| --- | --- |
| model_steps（loop 模型步） | **6.27**（p50 7 / p95 8 / max 8） |
| answer_context facts | 2.87 |
| evidence_ledger entries | 1.20 |
| tool_call 步 | 3.50 |
| final（回答步） | 1.50 |
| faithfulness judge / guard 步 | 1.70 |
| **最后一条证据进入的步**（src_last） | **1.68** |
| **证据到位后额外步数**（ms − src_last） | **4.68**（max 8） |

### 2.2 6.70 轮到底花在哪

**证据通常在 ~1.7 步内就到齐，loop 却跑满 ~6.3 步**——"证据够了还在 reasoning"成立，但要在两个子类里拆开看：

**(a) 校验/纠错环（productive，会救回错误答案）**
典型：q01 —— 证据 step 2 到位 → step 3 第一次 final 答成"北京"（错）→ guard → faithfulness judge → step 5 改为"秦皇岛市昌黎县"（对）→ judge。这类"证据后步骤"是 FinalWriter+Guard+LLM judge 在纠错，**不是浪费**，且 q01 最终 judge=2。
q47-04 出现 3 次 final 尝试（final=3）。

**(b) 重复空转 / 预算耗尽（unproductive）**
典型：q26-03 —— **8 步、7 次 tool_call、0 条 fact、0 条 ledger**，全程没拿到任何证据就撞预算（judge=0）。
q40-02（7 tool_call，证据 step3 后还多跑 5 步）、q47-01（5 tool_call 仍答错）。

### 2.3 结论

- **6.27 轮不是任务需要多步证据**（证据均值 1.7 步到位）。
- 额外 ~4.6 步里：一大部分是 **final+guard+faithfulness_judge 的校验环**（能纠错、有收益），一小部分集中在几个难例上的 **重复检索空转**（无新证据、无预算退出）。
- **对 State Guard / Completion Gate 的含义**：现有 G3 completion 在"证据满足+final 通过校验"时应当能提前收尾；当前仍在跑满 8 步的题（q26-03、q40-02、q47-01/04）说明 completion 判据对"证据不足/无进展"的题缺少提前终止路径。按计划，**在拿到进一步数据前不修改 State Guard**。
- No-op 定义建议（供后续）：`no-op turn = 该轮未新增 answer_context fact 且未关闭 requirement 且非最终 accepted final`。按此口径，judge/guard 步恒为 no-op 但属"校验用途"；真正的浪费是"tool_call 但 0 新 fact"（如 q26-03 的 7 次）。
