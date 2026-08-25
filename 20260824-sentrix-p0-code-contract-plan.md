# Sentrix P0 代码契约与轨迹一致性修复计划

日期：2026-08-24  
权威验证环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）

## 目标

在继续 P1 证据闭环、检索候选治理和记忆完整性之前，先修复工具输出、图片 handle、证据类型和 8771 轨迹之间的代码契约问题。P0 不改变模型、并发、检索排序或记忆生成策略。

## 已确认的代码事实

### 1. `search_memories` 描述字段的分层丢失

153 当前运行时的 `backend/agent_runtime/tools.py` 已通过 `_observation_summary()` 生成 `preview[].evidence_summary`。最新 100QA 的 `runtime_turns[].debug_trace` 中可以看到完整描述，例如“婚礼现场……舞台、音响设备……”。

但 `services/photobench/backend/benchmark_orchestrator.py::_evaluate_one()` 当前只从 response 的 `retrieval_trace` 和 `tool_trace` 构建顶层轨迹；最新 run 中 `retrieval_trace` 为空，完整工具 observation 只保存在 `runtime_turns[].debug_trace`，顶层 `item.tool_trace` 因此只有工具名、状态和时延。结果是 8771 某些轨迹视图看不到描述。

### 2. `photo_N` 泄漏路径未完全关闭

`photo_1` 是模型内部使用的 ResultSet handle，本身不是错误；错误是它出现在用户可见的最终答案。`final_writer.py` 已有 `sanitize_internal_refs()`，但普通 final 路径先调用 `naturalize_answer()`，没有统一经过 sanitize。最新 run `album3v4-029` 仍出现“根据对照片 photo_1……”，证明端到端边界尚未关闭。

### 3. 证据类型与工具编号/轨迹字段混用

后端 `EvidenceRequirement.evidence_type` 和 `ToolSpec.produces_evidence` 使用规范字符串，例如 `visual_observation`、`visible_text`、`memory_asset`，当前代码没有合法的数字工具编号协议。数字 `0/1/2` 在 PhotoBench 中是 evidence judge 分数，不是 evidence type。

最新样本还暴露出真实的语义契约问题：planner 要求 `confirmed_identity`，但 `search_memories` 产生 `memory_asset`、`location_metadata`、`temporal_metadata`，因此账本 `requirement_refs` 为空，需求保持 open。8771 前端的 `agent2EvidenceTypeLabel()` 只覆盖旧短名（`visual/text/temporal/...`），未覆盖当前规范字符串，进一步增加误读风险。

### 4. 仍需纳入 P0 的已知代码问题

- `TaskState.update_from_tool()` 目前只要 `search_memories` 返回 `result_set_id` 就覆盖 `current_result_set`，即使 `total=0`；空结果可能覆盖之前有效结果集，后续 `inspect_photo/read_photo_text` 产生 `unknown_handle`。
- 事件锚点路径已经改为调用 `ResultSetStore.new()`，但必须用回归测试确认它和普通 search 路径的 handle 生命周期一致。
- `_resolve_time_expression()` 仍是有限白名单；“国庆”“那次”“回乡”“晚上”等表达可能原样进入严格时间解析并导致错误过滤。P0 先补可复现测试和显式 `time_parse_status`，不直接扩大正则词表。
- `role=user` 的 completion/recovery 提示是 runtime 插入的内部控制消息，不是用户真实输入。8771 目前只展示原始 role，容易误判为用户说过这句话。轨迹需要增加 `message_origin=system_recovery` 和明确的展示标签。
- `filters.place` 是否把活动主题误填为地点，需要用原始模型 action 与 canonical intent 的差异做审计，不能仅凭最终过滤结果判断。

## P0-1：建立统一轨迹契约

涉及文件：

- `backend/agent_runtime/runtime.py`
- `services/photobench/backend/benchmark_orchestrator.py`
- `services/photobench/backend/test_trace_contract.py`
- `services/photobench/tests/test_image_extraction.py`

统一每一个新轨迹步骤的字段：

```text
stage / type
call_type
step_id
parent_step_id
conversation_turn
tool
arguments
observation
status
tool_call_id
evidence_refs
```

规则：

1. runtime 的 `debug_trace`、`retrieval_trace`、`tool_trace` 必须来自同一份步骤数据，不能各自重新按位置推断。
2. 8771 若 response 没有 `retrieval_trace`，必须从 `debug_trace` 构建完整 `execution_trace` 和工具步骤，而不是留下空轨迹。
3. 有 `step_id/parent_step_id` 时只按 ID 绑定；绑定失败要标记 `binding_source=unresolved`，不能静默按数组位置绑定。
4. 每个工具步骤必须保留真实 observation；摘要字段只能作为展示摘要，不能替代原始工具结果。

验收：新 run 中每个工具调用都能在 8771 页面绑定到唯一模型步骤，工具名、参数、结果和顺序一致；不得出现无原因的 `unbound` 工具。

## P0-2：修复 `search_memories` 结果展示

涉及文件：

- `backend/agent_runtime/tools.py`
- `services/photobench/backend/benchmark_orchestrator.py`
- `services/photobench/frontend/src/App.vue`

具体工作：

1. 保留 `preview[].evidence_summary`，无描述时显式写入 `description_status=missing` 和原因，不能静默省略。
2. 8771 的“工具输出”优先显示真实 observation；顶层工具性能摘要只显示时延和状态。
3. 页面同时展示 `tool`、`tool_call_id`、`parent_step_id`、`evidence_type` 和 `binding_source`，避免把工具序号当证据类型。

## P0-3：修复 `photo_N` 用户可见泄漏

涉及文件：

- `backend/agent_runtime/runtime.py`
- `backend/agent_runtime/final_writer.py`
- `backend/tests/test_agent2_answer_context.py`
- `backend/tests/test_phaseg_guard_tiers.py`

具体工作：

1. 在最终 response 的统一出口调用 `sanitize_internal_refs()`，覆盖普通 final、writer 重写、guard recovery、emergency summary、nucleus 渲染路径。
2. 内部 debug trace 可以保留 `photo_N`，但用户答案和 8771 的回答正文不能出现 `photo_N`。
3. 保持 `photo_N → asset_id` 的服务端映射，不把真实数据库 ID 暴露给模型或用户。

验收：使用包含 `photo_1`、`photo_12`、多个 handle 和括号表达的回归样本，所有最终答案均无内部 handle。

## P0-4：统一证据类型协议

涉及文件：

- `backend/agent_runtime/task_state.py`
- `backend/agent_runtime/tool_registry.py`
- `backend/agent_runtime/goal_planner.py`
- `backend/agent_runtime/runtime.py`
- `services/photobench/frontend/src/App.vue`

具体工作：

1. 证据类型只允许规范字符串；禁止用数字表示 evidence type。
2. 工具编号、模型步骤编号、evidence type、Judge score 使用不同字段和不同显示标签。
3. planner 需求必须与工具能力兼容；不兼容时记录 `evidence_incompatible` 并触发重新规划，不能生成 `requirement_refs=[]` 后继续伪装为已覆盖。
4. 前端补齐规范字符串映射：`memory_asset`、`memory_reference`、`visual_observation`、`visible_text`、`structured_fact`、`location_metadata`、`temporal_metadata`、`confirmed_identity`、`user_statement` 等。
5. 账本每条 entry 必须带 `capability`、`evidence_type`、`tool_call_id` 和对应 requirement 引用；如果没有匹配需求，明确显示“非当前需求证据”。

验收：固定样本中 planner 的 evidence type、实际工具、账本 evidence type、8771 展示标签四者一致；`confirmed_identity` 不得被 `search_memories` 的普通召回结果错误满足。

## P0-5：最小复现与验证顺序

先使用已保存的三个样本：

- `album3v4-004`：planner 要求 `confirmed_identity`，search 只产生 metadata 类型。
- `album3v4-021`：search 返回 `evidence_summary`、20 个召回候选和 OCR 路由。
- `album3v4-029`：最终回答泄漏 `photo_1`。

顺序：

1. 先写失败测试，确认轨迹缺 observation、handle 泄漏和 evidence type 不兼容均可复现。
2. 只修 P0-1/P0-2，验证 8771 轨迹完整性。
3. 只修 P0-3，验证所有最终输出路径的 handle 清理。
4. 只修 P0-4，验证 evidence type/工具/账本/前端标签一致。
5. 在 153 上运行定向回归、10QA smoke，再运行严格 100QA。

P0 完成门槛：

- 新 run 的工具步骤 100% 有真实 observation 或明确失败原因。
- 新 run 不出现无来源的 positional tool binding。
- 8771 展示的工具名与实际执行工具 100% 一致。
- 用户答案中 `photo_\\d+` 泄漏为 0。
- `requirement_refs` 不再因规范类型不匹配而静默为空。
- P0 不降低当前 Qdrant 健康状态和 0.628 retrieval recall 基线。

## P0 完成后的后续顺序

只有 P0 通过后，才进入：

1. P1 证据闭环与自适应重规划；
2. P1 检索候选治理；
3. P2 原始图像/关键帧记忆完整性；
4. 最终模型和检索 A/B。
