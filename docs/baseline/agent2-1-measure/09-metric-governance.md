# Agent 2.1 — Canonical Metric 治理说明

目的：杜绝"上一次 94.4%、下一次 60.5%"这类口径漂移。所有报告数字必须能追溯到单一权威口径。

## 1. Canonical 口径（唯一权威）

**Answer Quality / Core / Exact / Task Decision 一律按 `qa_id` 去重后的 direct item judge。**

| 字段 | 取值 |
| --- | --- |
| judge 来源 | `run.items[qa_id].judge.score`（多轮对话时即最后一轮 judge，item 级已封装） |
| denominator | 去重后 QA 行数（full-album3-38q = 38） |
| judge_valid | `judge.status == "completed"` 且 `score in {0,1,2}` |
| AQ mean | `sum(valid_score) / judge_valid_count` |
| Core Accuracy | `(score≥1 的 valid 数) / judge_valid_count` |
| Exact Accuracy | `(score==2 的 valid 数) / judge_valid_count` |
| Task Judgment Accuracy | `item.task_judge.correct` 为真的 valid 数 / valid 数 |

> 禁止把 conversation 各轮的 judge 摊平计数（那会把分母从 38 撑到 41）。已修复：`benchmark_orchestrator._capability_summary`。

## 2. 报告必带绑定字段

任何指标数字出现时，必须同时给出：

```
run_id            # 如 20260820-003839-agent2-1-ab
dataset_version   # 如 full-album3-38q（含 manifest/qa 文件 sha256 可回溯）
model             # 如 gemma4-12b-it
profile           # 如 goal_driven_candidate（SENTRIX_AGENT_PROFILE）
answer_context flag # SENTRIX_AGENT2_ANSWER_CONTEXT=1/0
judge_model       # 回答/任务判断 judge 是什么
denominator       # 显式写 valid/total（如 38/38）
```

缺任一字段的数字视为"不可比较"，不得跨 run 直接对比。

## 3. 三类口径（本阶段用 A）

| 口径 | 定义 | 用途 |
| --- | --- | --- |
| **A. Canonical（推荐）** | 按 qa_id 取 item judge，denom=题数 | 所有报告/比较 |
| B. 面板旧口径 | conversation 轮次摊平（denom 会 > 题数） | 已废弃 |
| C. 直接重算 | 脚本对 run.json 逐题重算（等价于 A） | 复算校验 |

## 4. 相关指标口径备注

- **Evidence-Conditioned**（报告 02）：`required_evidence_available` 判定基于 `answer_context.facts` 是否含参考答案关键值；`final_answer_correct` 用 Core（score≥1）。四象限只对 GT=answer 的题算。
- **Evidence Judge**（报告 04）：`evidence_judge.score` 只统计 `applicable=true` 的题；`Evidence Mean / Full Support / Score=0` 均以 applicable 为分母；另报 `Applicable Rate = applicable/total`。
- **Latency**：`agent_wall_ms`（不含 judge）= model+tool+other；`wall_clock_ms`（含 judge）。报告要注明是否含 judge。
- **Tokens**：`prompt_tokens_total` / `completion_tokens_total` 按整个 run 累计，平均按题。

## 5. 本阶段基线（canonical，20260820-003839）

- AQ mean **1.079**（{0:15, 1:5, 2:18}，38/38 valid）
- Core **0.605**（整体）/ **0.400**（仅 25 可答题）
- Exact **0.474**（整体）
- Task Judgment **0.658**（25/38）
- 可答题子集：Evidence Available 12/25=48%，Evidence-Conditioned Core 83.3%
