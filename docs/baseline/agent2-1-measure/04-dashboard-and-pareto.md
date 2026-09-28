# Agent 2.1 测量报告 04 — Evaluation Dashboard v2 + Error Pareto + Top-3 候选

基线 run：`20260820-003839-album3-gemma4-12b-it-agent2-1-ab`（38 题 / full-album3-38q / gemma4-12b-it）
> 数据全部来自该 run 的只读重算（直接按题计数，denominator=38）。本报告不改主链。

---

## 1. Evaluation Dashboard v2（基线数据）

### 1.1 Final Quality
| 指标 | 值 | 说明 |
| --- | --- | --- |
| Answer Quality Mean | **1.079 / 2** | 直接按题 {0:15, 1:5, 2:18}（面板显示 1.049，因分母 41 膨胀） |
| Core Accuracy（整体 38） | **0.605**（23/38） | 面板 0.585 因分母膨胀 |
| Exact Accuracy（整体 38） | **0.474**（18/38） | 面板 0.463 |
| Core Accuracy（仅可答题 25） | **0.400**（10/25） | 见报告 02 |
| Exact Accuracy（仅可答题 25） | **0.200**（5/25） | 13 题 refuse 全对拉高了整体 |

### 1.2 Retrieval（38 题）
| 指标 | 值 |
| --- | --- |
| retrieval_recall_mean | 0.649 |
| Precision (micro) | 0.811 |
| Recall (micro) | 0.769 |
| F1 (micro) | 0.789 |

### 1.3 Evidence（见报告 02 + Phase 4）
| 指标 | 值 |
| --- | --- |
| Required Evidence Availability Rate | 12/25 = 48.0%（GT=answer 内） |
| Evidence-Conditioned Core Accuracy | 10/12 = 83.3% |
| Synthesis Failure Rate | 2/25 = 8.0% |
| Upstream Failure Rate | 13/25 = 52.0% |
| Evidence Judge Applicable Rate | 18/38 = 47.4% |
| Evidence Judge Mean Score | **1.056**（applicable 内） |
| Evidence Full Support Rate（score=2） | **44.4%** |
| Evidence Score=0 Rate | **38.9%** |

**Evidence × Answer Quality 交叉（Phase 4 离线 judge 实测）**

| Evidence Score | n | Answer Quality Mean | Core Accuracy |
| --- | --- | --- | --- |
| 2 | 8 | 1.625 | 100% |
| 1 | 3 | 0.667 | 67% |
| 0 | 7 | 0.000 | 0% |

结论：**可适用的 18 题里，答案质量几乎完全由证据质量解释**（单调 2→100%、0→0%）。"Evidence 好 + Answer 错"≈0（仅 q02 是 ev=1+answer 错）。7 个 evidence=0 的题，正是离线 judge 能看图看到答案（如 q24-07 报警电话 22048084、q26-01 创始 1974、q47-07 火把）而 Agent 拒答的案例——再次指向 V1/V3 上游。

### 1.4 Finalization（见报告 01）
| 指标 | 值 |
| --- | --- |
| Task Judgment Accuracy（direct） | 25/38 = 65.8% |
| GT=refuse 正确率 | 13/13 = 100% |
| GT=answer→refuse 误拒 | 11 |
| GT=answer→clarify 误澄清 | 2 |

### 1.5 Reliability
| 指标 | 值 |
| --- | --- |
| JSON parse rate | 148/149 = 99.3% |
| QA 步数内完成率 | 29/30 = 96.7% |
| Runtime error | 0（38 题均 complete / refuse 短路） |

### 1.6 Latency（详见报告 03）
| 指标 | 值 |
| --- | --- |
| agent wall（不含 judge） | 19,972 ms mean（p50 22.1s / p95 94.5s / max 100.6s） |
| wall（含 judge） | 23,112 ms mean |
| 延迟构成 | model 47% + search_memories 30% + read_photo_text 20% |

### 1.7 Tokens
| 指标 | 值 |
| --- | --- |
| prompt tokens total | 188,395（~4,958/题） |
| completion tokens total | 18,983（~500/题） |
| context tokens p95 | 2,253 |
| TTFT mean | 225 ms |
| throughput | 62 tok/s |

---

## 2. 统一 Error Attribution（38 题，primary cause）

以下只对 judge<2 的 20 题归因（15 题答错 + 5 题部分对）。GT=refuse 13 题全部正确，不计入。

### 2.1 15 题 judge=0（答错）的 primary-cause 分布

| Primary cause | 数 | 占比 | 题目 |
| --- | --- | --- | --- |
| **V1 visual_understanding_failure** | 5 | 33% | q08 上衣颜色、q47-01 表演地点、q47-04 标志植物、q47-03 开场道具、q47-07 持火把照片 |
| **R1 retrieval_miss** | 5 | 33% | q24-01 拍摄日期(f1=0)、q26-03 汉堡价格(f1=0)、b2-01/b2-02/b2-03 引用消解(f1=0) |
| **V3 ocr_failure** | 3 | 20% | q03 沙雕主题名、q24-02 店名(OCR 读错)、q24-07 报警电话(OCR 读错) |
| **V2 identity_binding_failure** | 1 | 7% | q02 合影记录（已召回但无法把明明乐乐/沙雕绑定） |
| **F1 wrong_finalization_state** | 1 | 7% | q26-01 创始年（1974 已在 answer_context 却拒答） |

### 2.2 5 题 judge=1（部分对）备注
q24-04 / q24-05 / q26-02 / q40-02 / q47-02 —— 都是"检索/证据到位但未完整确认（人物/事件）"，属 E2 evidence_binding 或回答完整性，非硬错误。

### 2.3 结论
- **V1+R1+V3 = 13/15 = 87%** 的错误在"证据获取/理解"上游；**Synthesis（S1/S3）≈ 0**，New Verifier 无数据支持。
- 唯一值得单独看的收尾案例是 q26-01（evidence-conditioned 里的 ✓✗ 合成失败）。

---

## 3. Top-3 下一阶段候选（含 ROI）

> 候选排序完全由 Error Pareto 驱动。数据不支持开发 New Verifier / Claim Grounding。
> ROI 的 Core 基准为**可答题子集 core = 40%（10/25）**。

### 候选 1：R1 检索/证据获取（含引用消解）
| 维度 | 值 |
| --- | --- |
| 影响错误 | R1（5 题：q24-01、q26-03、b2-01/02/03） |
| 可覆盖题数 | 5 |
| 预计质量收益 | Core +5/25 ≈ **+20pp**（若检索+引用消解都修好，Core 15→20/25=80%） |
| 实现成本 | 中-高（检索召回 + conversation/resultset 引用解析两处） |
| Token | 0（不新增模型调用） |
| 延迟 | 0（复用现有检索） |
| 新故障面 | 低 |
| 备注 | 引用消解(3题)是独立的 agent2 能力，建议与纯检索(2题)分开评估 |

### 候选 2：V3 OCR 证据修复
| 维度 | 值 |
| --- | --- |
| 影响错误 | V3（3 题：q03、q24-02、q24-07） |
| 可覆盖题数 | 3 |
| 预计质量收益 | Core +3/25 ≈ **+12pp**（OCR 是这三题唯一的确定性阻断点） |
| 实现成本 | 低（OCR provider 选择/预算，已有 PaddleOCR 小模型优先路径） |
| Token | 0 |
| 延迟 | 低（read_photo_text 单次 18.9s 已有离群，优化它同时降延迟） |
| 新故障面 | 低 |

### 候选 3：V1 视觉理解
| 维度 | 值 |
| --- | --- |
| 影响错误 | V1（5 题：q08、q47-01/03/04/07） |
| 可覆盖题数 | 5 |
| 预计质量收益 | Core +5/25 ≈ **+20pp**（理论最大，但依赖视觉模型能力上限） |
| 实现成本 | 高（视觉模型/提示/prompt 工程，非确定性修复） |
| Token | 中（可能增加 inspect 调用） |
| 延迟 | 中 |
| 新故障面 | 中 |
| 备注 | 收益不确定，建议只做"inspect_photo 追问策略"这类低成本改进，不做模型替换 |

### 候选 3'（替代，成本低）：F1/Completion 收尾（q26-01 类）
| 维度 | 值 |
| --- | --- |
| 影响错误 | F1（q26-01：1974 已在 Context 却拒答）+ 同类 0 题 |
| 可覆盖题数 | 1（直接），更多是预防 |
| 预计质量收益 | Core +4pp（直接）；价值在于修复"证据在却拒答"的收尾路径 |
| 实现成本 | 低（Completion Gate / FinalWriter 对 evidence 已满足时的强制输出检查） |
| Token | 0 |
| 延迟 | 0 |
| 新故障面 | 低 |

---

## 4. 对 Agent 2.2 是否存在的判断

按计划纪律：**在 Error Pareto 完成前不实现 Agent 2.2 新模块。**
本报告给出的是**回归集(dev)上的 Pareto**。结论：
- 若 holdout（待新 QA）跑出来仍以 V1/R1/V3 为主 → 下一阶段应做 **检索/引用消解 + OCR 证据修复**，**不是 New Verifier**。
- `Evidence-Conditioned Core Accuracy = 83.3% < 90%` → 不应宣布 Writer 冻结，但也无需大改 Writer（synthesis 仅 8%）。
- q26-01 这类"证据在却拒答"值得单独加一条 completion 层检查（低成本、高信号）。

---

## 5. 交付物状态总览

| # | 交付物 | 状态 | 位置 |
| --- | --- | --- | --- |
| 1 | Evaluation Dashboard v2（本文 §1） | ✅ 基线数据齐全（evidence judge 待 Phase 4 补） | 本文件 |
| 2 | Finalization Confusion Matrix + 63.6% 解释 | ✅ | 报告 01 |
| 3 | Evidence-Conditioned Accuracy Report | ✅ | 报告 02 |
| 4 | Stage Latency Waterfall（25.3s 拆解） | ✅ | 报告 03 |
| 5 | Generalization Holdout Report | ⏳ **受阻：新 100 题 QA 待用户提供** | — |
| 6 | Error Pareto + Top-3 Candidates（本文 §2-3） | ✅ | 本文件 |
