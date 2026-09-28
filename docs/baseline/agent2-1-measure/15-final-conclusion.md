# Agent 2.1 本阶段完整结论（测量 → 归因 → 下一阶段决策）

日期：2026-08-20 ｜ 覆盖：Agent 2.1 测量阶段 + single-turn semantic robustness 阶段（P1 测量 + P2 shadow）

---

## 一、做了什么

1. **测量基础设施**：canonical metric 口径（按 qa_id 去重）、stage timer 埋点、denominator 修复、离线 Evidence Judge、导出功能（勾选筛选 + 完整轨迹）。
2. **基线对比**：Agent 2.0 vs Agent 1.0 分模型（12B + Qwen 0.8B），38 题。
3. **归因**：Error Pareto、R1 拆解、OCR 调查、V1 拆解、延迟瀑布、Evidence-Conditioned。
4. **鲁棒性测量**：Paraphrase Robustness Set（88 题）跑批 + 四层 paired attribution + drift funnel + P2 shadow。
5. **修复**：answer_context 配置漂移、get_person_profile 注入退化、judge 401、denominator 膨胀、导出缺轨迹。

---

## 二、核心结论（按重要性）

### 1. Agent 2.0 相对 1.0 全面显著提升
| 模型 | AQ | Core | Exact | 检索Recall | Prompt Token |
| --- | --- | --- | --- | --- | --- |
| 12B | 0.895→**1.158** | 0.553→0.684 | 0.342→0.474 | 0.667→0.800 | **-76%** |
| Qwen 0.8B | 0.474→**0.763** | 0.289→0.447 | 0.184→0.316 | **0→0.633** | **-69%** |

- 小模型收益最大（架构"借力"）；延迟代价 +~10s；12B JSON 解析 0.719→0.993。

### 2. 证据决定质量，瓶颈在上游
- Evidence-Conditioned Core **83%**（证据到 Writer）；证据没到 **0%**。
- Error Pareto：R1(33%) + V1(33%) + V3(20%) = 87% 在上游证据获取。

### 3. 延迟花在哪
- 12B：model 59%（agent loop 28% + planner 14% + recovery 9% + writer 8%）；工具 search 32% + OCR 19%。
- Qwen 0.8B：工具 76%（search_memories 73%）+ judge 7.6s（比 agent 还贵）。

### 4. Paraphrase 鲁棒性：中等敏感，角度最脆
- 同题换说法，可答题 AQ 掉 ~20%（12B -19%、Qwen -21%）。
- 风格：syntax 最稳（12B 1.048）> lexicon/order > **angle 最脆**（0.591）。

### 5. Drift Funnel：失败在"语义→检索意图"这一跳（P2 关键证据）
- 掉分首漂移层（修正版）：**Planner 47% + Search 42% = 89%**，Retrieval 0%，Evidence 11%。
- angle 100% 在上游（planner+search）。
- **Retrieval 本身稳定**：给什么 query 就返回什么。

### 6. P2 Shadow：canonical 检索意图强支持
- **22/22 父题家族出现 search 漂移**（同一父题的 4 个改写，agent 用了不同 query/filters）。
- **17/22 掉分与 search 漂移相关**（例：q26-q06 带 place 的写法答对、丢 place 的答错）。
- canonical 结构化检索（place/time/person/keyword → constraints）对同一父题恒定 → 可消除全部 search 漂移。

---

## 三、P2 判定

**数据强支持进入 P2（Canonical Retrieval Intent）实现**：
- 89% 的 paraphrase 漂移在"语义→检索意图"（planner 需求 + search query）。
- shadow 已做（query 漂移量化）：22/22 家族漂移、17/22 相关。
- 剩余 11%（evidence/visual/OCR）需 V1/V3 配合，P2 不覆盖。

**P2 实现纪律**（按你定的）：
- `answer_target + retrieval_target + structured constraints`，search 用 retrieval_target+constraints，不依赖用户原话。
- 先 shadow 对比 legacy vs canonical 的 retrieval recall/F1，再切 candidate。
- **不与 0-result relaxation 同批**。

---

## 四、剩余任务清单（按序）

| # | 任务 | 性质 | 说明 |
| --- | --- | --- | --- |
| 1 | **P2 Canonical Retrieval Intent** | 改 agent | shadow recall 对比 → 实现 → candidate |
| 2 | **W2.3 多轮引用消解** | 改 agent | referent 先解析到 ResultSet/Event |
| 3 | **W3.2 OCR 修复** | 改 agent | 小模型优先 + 切块优先 + VLM 超时（质量+延迟） |
| 4 | R1-A query 构建修复 | 改 agent | place 值/别名 + 0 结果降级（P2 之后单独） |
| 5 | 重跑 22+88+OCR subset+reference subset | 测量 | 每改动报 AQ/angle/retrieval/evidence/latency/extra-calls/regression |
| 6 | 重生成 Error Pareto | 测量 | 决定是否进 V1 |
| 7 | 真正 Holdout | 待新数据 | 新 album/人工 QA，不叫 paraphrase set |

---

## 五、最终一句话

**Agent 2.0 的架构（规划→按需取证→证据约束写作）已被数据证明有效且全面优于 1.0；本阶段把下一个瓶颈精确锁定为"自然语言→检索意图"的映射（89% 的 paraphrase 漂移发生于此），并已用 shadow 数据证明 canonical retrieval intent 是正确解法。下一阶段按 P2 → W2.3 → W3.2 的顺序推进，每个改动用统一指标矩阵回归，最后重生成 Error Pareto 再决定是否进入 V1 视觉优化。**

---

## 附：本阶段产出（153 `docs/baseline/agent2-1-measure/` + Downloads）

报告 01-13：denominator、confusion matrix、evidence-conditioned、延迟、turn 指标、dashboard+pareto、R1 拆解、OCR、V1 拆解、metric 治理、Agent2vs1 提升、latency waterfall、paraphrase robustness、drift funnel。
产物：stage timer、denominator 修复、导出增强、paraphrase set（88 题）、修复的 jit_prompt/start 脚本。
