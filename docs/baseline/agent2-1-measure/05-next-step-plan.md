# Agent 2.1 下一步执行计划（讨论稿 v0.1）

> **For agentic workers:** REQUIRED SUB-SKILL: 执行阶段使用 superpowers:executing-plans 或 subagent-driven-development。本稿是决策级计划；每项修复的 TDD 细化计划在拆解/分析结果出来后单独立项。
>
> **Goal:** 在 Agent 2.1 orchestration 冻结前提下，修测量基础设施、拆解并修复 Reference Resolution 与 OCR、用现有 GT 建立 Paraphrase Robustness Set，为真正的 Holdout 和视觉优化铺路。
>
> **Architecture:** 三层流水线 —— ① 测量层（canonical metric + stage timer）先立；② 拆解层（R1→A/B、V1→A..E、OCR→crop/provider/recognition/fusion/budget）产出决策证据；③ 修复层（reference path、OCR tool）只做被证据支持的最小改动。
>
> **Tech Stack:** `backend/agent_runtime/`（runtime/task_state/result_set/ocr_tool）、`services/photobench/backend/benchmark_orchestrator.py`、`services/photobench/frontend/src/App.vue`、`services/photobench/data/album3/qa/`。

---

## 0. 总原则（本阶段红线）

1. **Agent 2.1 orchestration 冻结**：不开发新 Planner / Verifier / Agent Loop / Evidence Ranker / Multi-Agent / 更复杂 Completion / 更复杂 Writer / New Verifier / Claim Grounding / Self-Refine。
2. **Canonical metric 唯一口径**：`按 qa_id 去重后的 direct item judge`，denominator = 题数（38）。任何报告数字必须绑定 `run_id / dataset_version / model / profile / answer_context flag`。禁止再出现 "94.4% vs 60.5%" 的口径漂移。
3. **不搞 circular holdout**：凡是基于现有 observations/错误生成的题集，一律叫 **Paraphrase Robustness Set / Metamorphic Set**，永远不叫 Generalization Holdout。
4. **改动只碰评测与 capability**：新增 judge 只在 benchmark/offline；不新增线上 Agent 模型调用。
5. **153 工作树现状**：已脏（未提交改动 + 未跟踪文件）。新产物只放 `docs/baseline/agent2-1-measure/`（未跟踪），不碰既有未提交改动、不提交。

---

## W1 — 评测基础设施（立即执行，已批准）

> 只改观测，不改 Agent。

### W1.1 修 Answer Quality denominator（38 去重）
- **文件**：`services/photobench/backend/benchmark_orchestrator.py` `_capability_summary`（约 2268-2273 行）；`services/photobench/frontend/src/App.vue`（约 421 行面板）。
- **改动**：judge 扁平化改为 **按 qa_id 取最终 item judge**（多轮时取最后 turn 的 judge，而不是把每轮 judge 都计入），denominator = `len(qa_rows)`。面板显式显示 `Answer Quality Valid: X / {total}` + `Invalid: Y`；Invalid 输出 `{"question_id","judge_valid":false,"reason"}`。
- **验收**：AB run 上 valid=38、denominator=38、分布 {0:15,1:5,2:18}、均值 1.079；面板不再出现 41。
- **回归**：复用 AB run.json 离线验证聚合函数输出（不重跑 benchmark）。

### W1.2 Stage timer（planner/writer/ledger/context/recovery 埋点）
- **文件**：`backend/agent_runtime/runtime.py`（主链加计时字段，**不改行为**）。
- **改动**：`debug_trace` 的 planner / writer / nucleus / force_final / recovery 步补 `elapsed_ms`；`agent2_trace` 补 `stage_timing_ms`（planner_ms / agent_llm_ms / search_memories_ms / query_memory_facts_ms / inspect_photo_ms / read_photo_text_ms / result_processing_ms / evidence_ledger_ms / answer_context_ms / final_writer_ms / recovery_ms / total_ms）。
- **验收**：跑一次 38 题后，报告 03 的"测量缺口"小节能用精确 stage 数字替换。
- **依赖**：需要一次 38-q benchmark 重跑（153 生产 vLLM + judge LLM，非高峰窗口）。→ **决策点 1**。

### W1.3 Canonical metric 治理文档
- **产物**：`docs/baseline/agent2-1-measure/` 加一页 measurement-governance 说明（指标定义、绑定字段、引用口径），防止以后回归。

---

## W2 — Reference Resolution 专项（先拆解，再决定是否修）

### W2.1 拆解 R1 的 5 个错误（纯分析，立即）
- **对象**：q24-01（拍摄日期）、q26-03（汉堡价格）、b2-01/02/03（三个多轮引用）。
- **方法**：逐题看 `runtime_turns` —— 用户消息是否携带 conversation context、agent 实际搜了什么、`conversation_context_mode`、是否有 `selected_asset/current_result_set/focus_event` 可用但没用。
- **输出**：分类为
  - R1-A 真检索 miss：全库语义检索就没召回（预计 q24-01、q26-03）。
  - R1-B 引用消解失败：本应先解析 referent（"那次/那个公园/那个动物园"）到既有 ResultSet/Event/Asset，却重新全库 search → F1=0（预计 b2-01/02/03）。

### W2.2 决策门（拆解后）
- 若 R1-B = 3/5 → 立项最小 reference-resolution 修复（见 W2.3）；若 R1-A 为主 → 另议（不碰 ANN，先看检索 query 构建）。
- **关键约束**：reference fix 是 **agent 行为改动**（不是纯测量）。需确认边界 —— 只做 "referent 先解析到 selected_asset/current_result_set/focus_event/recent_result_set/conversation history，再决定是否全库搜索"，不动 Planner 决策逻辑。→ **决策点 2**。

### W2.3（若批准）最小 reference-resolution 修复
- **文件**：`backend/agent_runtime/result_set.py`、`conversation_summary.py`、`tools.py`（search 前解析 referent），`backend/tests/` 加 3 个引用题回归。
- **验收**：b2-01/02/03 在引用路径下 F1>0；38-q 回归无退化；不改 Planner。

---

## W3 — OCR 专项（准确率 + 延迟双目标）

### W3.1 调查 3 个 OCR 错误 + 76.98s 离群（纯分析，立即）
- **对象**：q03（沙雕主题名读不出）、q24-02（店名读错：大圣葱油拌面→大兴烧烤串儿）、q24-07（报警电话读错：22048084/85→22048004）、read_photo_text 单次 76.98s 离群。
- **方法**：看 `ocr_tool.py` 实际路径 —— 用了 PaddleOCR 小模型还是 VLM fallback？crop 策略？prompt？confidence？76.98s 那次是哪个调用、是否 VLM 超时。
- **输出**：每个错误归到 crop / provider / recognition / fusion / budget。

### W3.2 修复（按调查结果立项）
- 候选：crop 策略（区域 vs 全图）、provider 选择（小模型优先置信度门控）、VLM 调用预算/超时上限（压 76.98s 离群）。
- **验收**：3 个 OCR 错误至少修复 2 个，且 read_photo_text p95 明显下降；38-q 回归无退化。

---

## W4 — Paraphrase Robustness Set（75~125 题，不叫 Holdout）

### W4.1 生成
- 用现有 25 个 answer 题 GT 做 3~5 个高差异改写（语法/词汇/否定/指代变化），答案不变（如"顶呱呱创立于哪一年？"→"照片里那家店的信息显示顶呱呱是哪一年创立的？"）。
- 可 LLM 辅助生成 + 人工抽查 10~15% 样本校准；**不许**照抄已知错误模式。
- 存为 `services/photobench/data/album3/qa/paraphrase-robustness.jsonl`，`qa_id` 挂 parent（robustness:parent=validation-album3-XXX）。
- **决策点 3**：生成方式（模板式 vs LLM 辅助）、规模（75 vs 125）、是否也覆盖 refuse 类、是否在 W1.2 重跑之后再跑（保证 canonical 指标）。

### W4.2 跑批 + 报告
- 跑 paraphraase set，报告 **Paraphrase Robustness Report**：同一题不同说法的 core/exact 稳定性、Planner/JIT/Retrieval 对表述变化的鲁棒性。
- **明确命名**：结果只能叫 robustness，不写入任何"泛化通过"结论。

---

## W5 — V1 视觉拆解（不开发，只拆）

### W5.1 拆 5 个 V1 错误（纯分析，立即）
- 对象：q08（上衣颜色）、q47-01（表演地点）、q47-03（开场道具）、q47-04（标志植物）、q47-07（持火把照片）。
- 方法：逐题看 `inspect_photo` 实际调用的 asset/crop/prompt 与 VLM 原始返回；对照 capability matrix。
- 输出：按 V1-A model 真看不出 / V1-B 查错 asset / V1-C 覆盖不足 / V1-D 问题表述差 / V1-E identity binding 归类。若最终 V1-A ≤1/5 → **明确不换 VLM**。

### W5.2 仅当证据支持的低成本 tool 级修复
- 例：inspect_photo 只查了 preview/单张而漏掉关键帧（若 V1-B/C 成立）→ 可做"多候选 asset 覆盖"的小改动，立项前先给 ROI。

---

## 执行顺序（建议）

| 阶段 | 内容 | 是否改 Agent |
| --- | --- | --- |
| **Phase 1（立即，纯分析）** | W1.1 denominator 修复；W2.1 R1 拆解；W3.1 OCR 调查；W5.1 V1 拆解；W1.3 治理文档 | 否（除 W1.1 是评测代码） |
| **Phase 2（决策后）** | W1.2 stage timer + 38-q 重跑；W2.2/2.3 reference 修复；W3.2 OCR 修复 + 回归 | W2.3/W3.2 是 capability 改动 |
| **Phase 3** | W4 paraphrase set 生成 + 跑批 + 报告 | 否（新增 qa_set） |
| **任何阶段后** | Error Pareto 更新；只有真正 Holdout（新 album/新人工 QA）才谈 Generalization Gate | — |

---

## 待讨论决策点

1. **W1.2 重跑**：需要一次 38-q benchmark 重跑（153 vLLM + judge LLM）。批准非高峰窗口跑？还是先只加埋点、等下次自然重跑收集？
2. **W2.2/2.3 reference 修复边界**：这是 agent 行为改动。确认只做"referent 先解析到既有 ResultSet/Event/Asset 再决定是否全库搜索"，不动 Planner/决策；是否允许触碰 `runtime.py` 主循环（还是只在 `result_set.py`/`conversation_summary.py` 收窄）？
3. **W4 paraphrase 生成**：模板式 vs LLM 辅助 + 人工抽查？规模 75 vs 125？是否也覆盖 13 个 refuse 类？跑批是否排在 W1.2 重跑之后？
4. **W3.2 OCR 预算**：76.98s 离群是否直接加 VLM 调用硬超时（如 30s）+ 小模型置信度门控？这是 tool 层改动，确认可做。
5. **交付位置**：本计划存哪 —— 与报告一起放 `docs/baseline/agent2-1-measure/`？还是你想要的别处？

---

## 一句话总结

Agent 2.1 orchestration 暂停；下一步 = **修 measurement（denominator + stage timer）→ 拆 R1 reference resolution（先解析 referent 再全库搜索）→ 优化 V3 OCR（质量+延迟）→ 用现有 GT 建 Paraphrase Robustness Set**；真正视觉模型优化与新 Agent 模块全部等这些结果。
