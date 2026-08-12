# Agent Runtime v2 — 前端思考过程 Agent 化 + 原始证据折叠框 完成与测试报告

- 日期：2026-08-11
- 范围：前端思考过程（类编程 agent 实时工具轨迹）+ 原始证据折叠框（tool-loop 证据样本链路）
- 153 正式分支：`psh`（含 `0527639`/`8f91025`/`f758811` 合并）
- 本地工作分支：`psh-runtime-v2`（HEAD `4dd1a10`）

## 1. 思考过程 Agent 化

- `src/app.js`：`buildThinkingSteps` 合并实时 `public_progress`（SSE 增量，stage/status/step_index）与完成后的 `toolTrace`；`agentStepHtml` 渲染工具卡（💭 阶段 / 🔧 工具 + ✓/…/! 状态 + 耗时）。
- 加载中实时渲染 `data-live-progress` 步骤流；回答后保留「思考过程 · N 步」折叠块，成功收起、失败/guard 自动展开。
- 后端：tool-loop 分支补 `tool_trace` 生成（此前从未产出）；非 admin 下 `toolTrace` 裁剪为 `tool/status/latency_s` 对普通用户可见，参数与 observation 明细保持管理员可见。

## 2. 原始证据折叠框 + 证据样本

- `structured_memory._sample_observations`：按过滤条件取代表 observation（asset_id/caption/captured_at）。
- `tools._query_memory_facts`：事实类 operation 与 meal 附加 `samples`；**分组类查询不附任意样本**（避免与分组内容不匹配造成误导）。
- `result_set.record_tool_result` 持久化 samples；`tool_policy` sanitizer 放行 `samples`。
- 前端 `toolLoopEvidence` 渲染可点击证据卡片；`requiresEvidence` 兼容 tool-loop（有 samples 或结果集即显示「原始证据 · N 项」折叠框，默认展开）。

## 3. 测试结果

- 153 全量：`768 passed | 3 failed（已知事件归并，用户拍板不改）| 4 skipped`，无回归。
- 前端：`38 tests | 0 failures`。
- 线上实测（4174 → 8091，scope=全部相册）：
  - 明哥最早是什么时候来的 → complete，`samples: 3`（可点照片），toolTrace 含 `query_memory_facts`。
  - 去年去过哪里 → complete，分组场景 samples=0（不误导），回答城市列表 + 覆盖披露。
  - 找一些2024年的照片 → complete，ResultSet 52 张、preview 6、toolTrace 含 `search_memories`。
  - SSE 实时事件经 4174 流式转发正常（thinking → tool_result → recovering → finalizing）。
