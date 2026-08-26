# Agent 2.1 W1.2 — Stage-Level Latency Waterfall（精确版）

基线：Agent 2.0 干净配置（answer_context=1 + 无 get_person_profile 注入 + stage timer 埋点）
数据源：`agent2_trace.stage_timing_ms`（turn 级，本轮新增埋点）+ `tool_trace` + `timing_breakdown`
> 只测量，未改 Agent 行为。

## 1. 12B（gemma4-12b-it）— agent 均 20.75s

| 阶段 | 代码/来源 | Mean | 占 agent | P50 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| **agent loop LLM** | `stage_timing_ms.agent` | 5,832ms | 28.1% | 6.0s | 9.7s | 10.2s |
| **search_memories** | `tool_trace` | 5,526ms | 32.4% | — | — | —（46 次调用，单次均 5.5s） |
| **read_photo_text（OCR）** | `tool_trace` | 16,851ms | 19.2% | — | — | —（9 次调用，单次均 16.9s，hotspot） |
| planner | `stage_timing_ms.planner` | 2,854ms | 13.8% | 3.0s | 3.8s | 3.9s |
| recovery_or_judge | `stage_timing_ms.recovery_or_judge` | 1,897ms | 9.1% | 2.1s | 3.0s | 3.3s |
| writer | `stage_timing_ms.writer` | 1,697ms | 8.2% | 0.9s | 6.7s | 6.7s |
| inspect_photo | `tool_trace` | 804ms | 1.8% | — | — | —（18 次） |
| **agent 合计** | `timing_breakdown.agent_wall_ms` | **20,750ms** | 100% | — | — | — |
| judge（回答+任务判断） | `timing_breakdown.judge_ms` | 9,178ms | 加在 agent 之外 | — | — | — |
| **wall 合计（含 judge）** | `wall_clock_ms` | **29,928ms** | — | 27.9s | — | — |

## 2. 千问 0.8B（qwen3.5-0.8b-it）— agent 均 6.55s

| 阶段 | 代码/来源 | Mean | 占 agent |
| --- | --- | --- | --- |
| **search_memories** | `tool_trace` | 6,098ms | **73.3%**（单次均 6.1s，ANN 检索模型无关） |
| agent loop LLM | `stage_timing_ms.agent` | 965ms | 14.7% |
| planner | `stage_timing_ms.planner` | 258ms | 3.9% |
| recovery_or_judge | `stage_timing_ms.recovery_or_judge` | 245ms | 3.7% |
| read_photo_text | `tool_trace` | 737ms | 2.3% |
| inspect_photo | `tool_trace` | 164ms | 0.6% |
| writer | `stage_timing_ms.writer` | 61ms | 0.9% |
| **agent 合计** | `agent_wall_ms` | **6,552ms** | 100% |
| **judge** | `judge_ms` | **7,649ms** | 加在 agent 之外（比 agent 还高） |

## 3. 关键结论（"25.3s / 延迟到底花在哪"）

1. **12B：模型 LLM 是主成本（59%），其中 agent loop 28% + planner 14%**；工具里 search_memories（32%）和 OCR（19%）是大头。
2. **Qwen 0.8B：工具是绝对主成本（76%），search_memories 单它就占 73%** —— 小模型 LLM 快（0.97s），瓶颈全在检索工具（ANN 与模型无关）。
3. **OCR 是独立 hotspot**：read_photo_text 单次 16.9s（12B），这是 VLM 读图成本；和报告 07（OCR 专项）一致——它同时是质量瓶颈（V3）和延迟瓶颈。
4. **judge 成本不可忽略**：12B 的 judge 9.2s（= agent 的 44%）、Qwen 的 judge 7.65s（> agent 的 6.55s）。小模型场景 judge 是最大单项。
5. **本轮新埋点补齐了报告 03 的"测量缺口"**：planner 2.85s / writer 1.70s / recovery 1.90s（12B）现在都有精确值。证据/context 阶段仍归入 overhead（占 agent <3%，无需单独拆）。

## 4. 优化优先级线索（纯观测，不据此改）

- 12B：压缩 agent loop 步数（28%）或 JIT 剪枝 prompt 减 LLM 时间 → 延迟收益最大。
- 两模型：search_memories 5.5-6.1s/次高是因为 ANN（visual_ann 2.4s + text_ann 0.8s + 嵌入）——若缓存嵌入可显著降。
- OCR（read_photo_text 16.9s/次）→ 对应 W3.2 的 OCR 优化（小模型优先 + VLM 超时）。
- Qwen：judge 比 agent 贵 → 小模型场景可考虑更便宜的 judge 或抽样评分。
