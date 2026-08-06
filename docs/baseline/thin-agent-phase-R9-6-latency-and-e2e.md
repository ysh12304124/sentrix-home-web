# R9-6 · GPU 与真实端到端验收 — 报告（代码就绪）

**日期**：2026-08-06
**性质**：stage trace 基础设施 + 全阶段延迟测量脚本 + 端到端 case 验证脚本。**最终性能验收依赖 GPU 修复窗口（D10，用户 infra）**。
**测试**：本地 493 通过（R9-5 489 + 新增 4）。

## 1. 交付

| 文件 | 改动 |
|---|---|
| `backend/thin_agent.py` | `SENTRIX_AGENT_STAGE_TRACE=1` 时，每响应带 `perf` 块（thread-local `_Perf`）：`explicit_detector/parser/router/probe/query_spec/retrieval/answer/claim/complex_chain` 阶段耗时 + `model_calls`（parser/repair/answer/claim）；关时不产生开销（nullcontext） |
| `backend/query_parser.py` | `call_counts`（parser/repair）供 model_calls |
| `scripts/benchmarks/measure_agent_latency.py`（新，替代 measure_latency.py） | §10.2 十条路径 × cold/warm p50/p95；真实 `perf.model_calls` 计数；每阶段 p50；timeout/fallback/路由异常计数；**real-HTTP 检查**（warm normal_chat 必须含 1 次真实 answer 调用）；统一 deadline 20s 对照 |
| `scripts/benchmarks/e2e_r9_cases.py`（新） | §10 端到端 10 条 case 对真实 API 断言 route/evidence/披露/model_calls：人物介绍（必进 evidence，杜绝 0.001s 假通过）、写作零记忆、短语无命中→clarify、照片里写着→家庭、简单 evidence 免 Writer、为什么+日期+人物→evidence、海豚→近似披露、贵阳→strict_empty 拒答、会话后续、概念→none |
| `backend/tests/test_agent_latency_trace.py`（新） | perf collector 计时/计数、关标志无 perf、trace 附加 + model_calls 断言 |

## 2. 验收状态

**代码就绪 ✅**：stage trace、测量、E2E 脚本、测试全部落地；本地 493 绿；审计 runtime D/E=0。

**待 153 实测（用户/部署）**：
- [ ] 153 apply R9 代码 → 本地/153 测试绿 → 重启 `start_sentrix_api_8091.sh`（`SENTRIX_AGENT_STAGE_TRACE=1` + `SENTRIX_AGENT_MODEL_PROFILE=quality_12b`）
- [ ] `measure_agent_latency.py`（10 路径 warm/cold）
- [ ] `e2e_r9_cases.py`（10 条断言）
- [ ] **GPU driver mismatch 修复**（独立维护窗口，D10）→ 修复后对照（12B Parser / 普通聊天 12B / 简单 evidence / 复杂人物完整链）
- [ ] 12B 全角色 20s 门槛仅可在 GPU 修复或明确替代部署后宣称（R9 §10.4 / D10）

## 3. 下一步

R9-7：汇总报告 + 关闭决策 A/B/C/D（依赖上述 153 实测数据）。
