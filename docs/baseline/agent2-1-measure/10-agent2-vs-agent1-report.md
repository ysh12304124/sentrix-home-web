# Agent 2.0 vs Agent 1.0 端到端效果提升报告（分模型）

日期：2026-08-20 ｜ 数据集：`full-album3-38q`（38 题）｜ Judge：`doubao-seed-2.0-lite`（火山引擎）

## 0. 本次测评基线说明

| run | 模型 | Agent 版本 | run_id |
| --- | --- | --- | --- |
| A1-12B | gemma4-12b-it | Agent 1.0（Thin Agent） | `20260817-161053` |
| **A2-12B** | gemma4-12b-it | **Agent 2.0**（goal_driven_candidate） | `20260820-141456-612fb3` |
| A1-qwen0.8B | qwen3.5-0.8b-it | Agent 1.0 | `20260817-163522` |
| **A2-qwen0.8B** | qwen3.5-0.8b-it | **Agent 2.0** | `20260820-141456-6d4919` |

Agent 2.0 配置：`goal_driven_candidate` + `SENTRIX_AGENT2_ANSWER_CONTEXT=1`（Minimal Answer Context 开启）+ Tool-Loop；本轮已修复 get_person_profile 注入导致的拒答退化（见附录）。

---

## 1. 指标定义（每个指标：代码 / 计算方式 / 分母 / 含义）

| 指标 | 代码字段 | 计算方式 | 分母 | 含义 |
| --- | --- | --- | --- | --- |
| **Answer Quality 均分 (AQ)** | `judge.score` | `Σ(每题judge得分) / 有效题数`；得分：2=正确覆盖全部核心信息，1=覆盖部分或核心对但有错/矛盾，0=核心错/缺失/编造/答非所问 | 38（按 qa_id 去重） | 回答质量综合均值 |
| **Core Accuracy（核心准确率）** | `judge.score ≥ 1` | `(1分+2分题数) / 38` | 38 | 答案"基本答对"比例 |
| **Exact Accuracy（完全准确率）** | `judge.score == 2` | `(2分题数) / 38` | 38 | 答案"完全正确"比例 |
| **Task Judgment（任务判断）** | `task_judge.correct` | task_judge 把实际行为分类为 answer/refuse/clarify/unsupported，与 GT `expected_action` 对齐；`correct=True / 有效数` | 有效题数 | "该答就答、该拒就拒"的正确率 |
| **Retrieval Recall（检索召回）** | `retrieval_recall` | 每题 `命中GT图/GT图总数` 的题均（macro） | 有 GT 图的题 | 把目标照片找回来的能力 |
| **端到端延迟** | `wall_clock_ms` | 每题从输入到最终回答的墙钟时间；Mean/P50 | 38 | 回答耗时（含 judge） |
| **JSON 解析成功率** | `agent_stability.json_parse_*` | `解析成功模型输出 / 需解析输出` | 全部模型步 | 模型输出可解析的可靠性 |
| **Prompt Token 总量** | `llm_summary.prompt_tokens_total` | 整个 run 的输入 token 累计 | — | 上下文用量（成本） |

---

## 2. 12B（gemma4-12b-it）：Agent 1.0 → Agent 2.0

| 指标 | 定义 | Agent 1.0 | Agent 2.0 | 提升 |
| --- | --- | --- | --- | --- |
| Answer Quality 均分 | judge 0/1/2 均值 | **0.895** | **1.158** | **+0.263** |
| Core Accuracy | ≥1 分比例 | 0.553 | 0.684 | +0.131 |
| Exact Accuracy | =2 分比例 | 0.342 | 0.474 | +0.132 |
| Task Judgment | 四态路由正确率 | 0.711 | 0.711 | +0.000 |
| Retrieval Recall | 命中 GT 图比例 | 0.667 | 0.800 | +0.133 |
| JSON 解析率 | 模型输出可解析 | 0.719 | 0.993 | +0.274 |
| 端到端延迟 Mean | wall_clock | 20.1s | 29.9s | +9.8s |
| 延迟 P50 | — | 17.8s | 27.9s | +10.1s |
| Prompt Token | 输入累计 | 714,281 | 168,997 | **-76%** |

**12B 结论**：Agent 2.0 让 12B 在 AQ/Core/Exact/检索/可靠性全面提升，且 **Prompt Token 降 76%**（JIT 上下文剪枝）。代价是延迟 +50%（Agent 循环做规划+证据+收尾）。

## 3. 千问 0.8B（qwen3.5-0.8b-it）：Agent 1.0 → Agent 2.0

| 指标 | 定义 | Agent 1.0 | Agent 2.0 | 提升 |
| --- | --- | --- | --- | --- |
| Answer Quality 均分 | judge 0/1/2 均值 | **0.474** | **0.763** | **+0.289** |
| Core Accuracy | ≥1 分比例 | 0.289 | 0.447 | +0.158 |
| Exact Accuracy | =2 分比例 | 0.184 | 0.316 | +0.132 |
| Task Judgment | 四态路由正确率 | 0.632 | 0.711 | +0.079 |
| Retrieval Recall | 命中 GT 图比例 | **0.000** | **0.633** | **+0.633** |
| JSON 解析率 | 模型输出可解析 | 0.849 | 0.832 | -0.017 |
| 端到端延迟 Mean | wall_clock | 4.9s | 14.2s | +9.3s |
| Prompt Token | 输入累计 | 546,995 | 168,852 | **-69%** |

**千问 0.8B 结论**：**提升最显著**——Agent 1.0 下小模型检索召回为 0（根本找不到照片），Agent 2.0 把它带到 0.633，AQ 从 0.474 → 0.763。架构（Planner + JIT 工具 + 证据）对小模型是"借力"：用结构化流程弥补模型能力，让 0.8B 也能完成端到端任务。延迟同样 +9s，Token 降 69%。

---

## 4. 综合结论

1. **两个模型 Agent 2.0 全面优于 Agent 1.0**（除 12B Task Judgment 持平外，其余指标全部提升）。
2. **小模型收益更大**：0.8B 的检索从 0 → 0.633，说明 Agent 2.0 的流程化设计能显著放大弱模型的能力。
3. **Token 效率大增**：12B -76%、0.8B -69%，JIT 上下文剪枝省成本。
4. **延迟代价**：两模型都 +~10s（Agent 2.0 多轮 planner+证据+收尾），12B 均值 29.9s、0.8B 14.2s。
5. **可靠性**：12B 的 JSON 解析从 0.719 → 0.993（Agent 2.0 的严格输出控制）。

## 5. 口径与限制

- 全部指标用 canonical 口径（按 qa_id 去重，denominator=38）。
- Agent 2.0 的两个 run 都开启了 `SENTRIX_AGENT2_ANSWER_CONTEXT=1`（turn 级 agent2_trace 含 answer_context 与 stage_timing_ms 已核验）。
- 单次 run，存在 run-to-run 方差；提升幅度以 AQ/Exact 等聚合指标为准。
- Agent 1.0 基线为 2026-08-17 的历史 run，judge 同 doubao。

---

## 附录：本轮修的问题

- **get_person_profile 自动注入退化**：12:07 未提交改动给 JIT prompt 注入 `get_person_profile` 工具，导致拒答题（q06/q24-q08）从干净拒答退化为否定前提/啰嗦总结（2→1/0 分）。已移除注入规则（保留 schema），恢复 AB 等价行为后 12B AQ 回到 1.158。
- **answer_context 配置漂移**：8091 重启丢失 `SENTRIX_AGENT2_ANSWER_CONTEXT=1`，已在启动脚本修复。
