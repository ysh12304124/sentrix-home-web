# 单相册结构化事实工具执行计划

> **目标：** 恢复一个受限、可审计的 `query_memory_facts` 工具，让 Agent 能精确回答当前单个相册内的基础统计问题；不恢复旧版语义/视觉聚合分支，也不引入跨相册统计。

> **架构：** 模型仅声明操作与显式筛选条件；Runtime 从会话注入 `scope_id`；`StructuredMemoryExecutor` 以 SQLite 全量聚合执行；工具只回传标量、有限分组和统计口径。`structured_fact` 未被尝试时，Runtime 在模型首次 final 前仅软提醒一次。

> **技术栈：** Python 3、SQLite、现有 `MemoryStore`、Agent 2 Tool Registry / JIT Prompt / Evidence Ledger、pytest。

## 已确认的产品约束

1. 统计范围恒为当前单个相册；模型参数里没有 `scope_id`，也不得跨 scope。
2. “照片”默认是 `assets.media_type='image'` 且非派生资产；视频单独统计；视频关键帧、缩略图等派生资产不计入任一默认媒体总数。
3. 模型不接收原始资产列表、内部 ID 或 SQL，只接收本次聚合的事实对象与最多三条代表性来源。
4. 可统计事实仅限 `asset`、`person_appearance`、`processing`；不支持颜色、物体、场景、活动、关系、OCR 金额、桌数或自由事件名。
5. 统计需求的 final 门禁为一次性软提醒：模型漏调工具时有一次补调用机会，之后可直接如实 final，绝不循环阻塞。

## 任务 1：为基础统计建立可执行的资产口径

**文件：**
- 修改：`backend/structured_memory.py`
- 修改：`backend/tests/test_structured_memory.py`

1. 在 `backend/tests/test_structured_memory.py` 新增失败测试：同一 scope 中插入一张原始图片、一段原始视频和一张 `derived_kind='video_keyframe'` 图片；断言图片统计只返回原始图片数、视频统计只返回原始视频数。
2. 运行：
   ```bash
   pytest -q backend/tests/test_structured_memory.py
   ```
   预期：新测试失败，证明当前执行器尚未排除派生资产。
3. 在 `StructuredMemoryExecutor._base_query()` 无条件加入 `COALESCE(a.derived_kind, '') = ''`，使所有默认事实统计只扫描原始媒体；保留 scope、时间、地点、媒体、实体的既有参数化筛选。
4. 在 `_filters_applied()` 输出 `asset_kind: 'original_only'`，让 Final Writer 和审计 trace 可见口径。
5. 重跑同一测试，预期全部通过。

## 任务 2：恢复受限的 `query_memory_facts` 工具和输出契约

**文件：**
- 修改：`backend/agent_runtime/tools.py`
- 修改：`backend/agent_runtime/tool_policy.py`
- 修改：`backend/agent_runtime/jit_prompt.py`
- 修改：`backend/agent_runtime/profile.py`
- 修改：`backend/agent_runtime/goal_planner.py`
- 修改：`backend/tests/test_phase_d_tools.py`
- 修改：`backend/tests/test_tool_registry_contract.py`

1. 新增 `test_query_memory_facts_counts_only_current_scope`：用两个 scope 的 `MemoryStore` 数据绑定工具，调用 `count/asset/media=image`，断言只计当前 context scope 且结果含 `fact_type='asset_count'`、`unit='original_image'`、`coverage.complete=True`。
2. 新增 `test_query_memory_facts_rejects_visual_filter`：传入 `filters={'color':'红色'}`，断言返回 `status='unsupported_filter'` 与 `coverage.complete=False`，且没有数值事实。
3. 运行上述测试，预期因工具不存在而失败。
4. 在 `tools.py` 恢复最小 `_query_memory_facts()`，仅接受：
   - `operation`: `count|exists|first|last|group`
   - `subject`: `asset|person_appearance|processing`
   - `group_by`: `month|place|media`；`processing` 额外允许 `status`
   - `filters`: `time|media|place|person`
   未知字段返回 `unsupported_filter`；不实现 `meal/list/event` 或自由语义条件。
5. 工具从 `context['scope_id']` 取 scope、从 `context['viewer_id']` 取 viewer，调用 `_draft_from_filters()`、`_spec_for()`、`StructuredMemoryExecutor.execute()`；不接受外部 scope 参数。
6. 将数据库结果映射为稳定事实：`fact_type`、`subject`、`value`、`unit`、`filters_applied`、`coverage`、`samples`。人物无法精确解析返回 `unresolved_person`，不退化为全相册。
   对 `count|exists|first|last`，忽略模型附带的无害 `group_by`，避免小模型填充可选枚举字段而拒绝本可精确回答的问题。
7. 注册 `ToolSpec(name='query_memory_facts')`，描述明确列出允许统计的事实和“不支持颜色、物体、场景、活动、关系、OCR 金额、桌数”的禁止边界；`produces_evidence=('structured_fact',)`。
8. 从 `search_memories.produces_evidence` 删除 `structured_fact`，确保候选召回总数不能闭合全量统计需求。
9. 在 `ToolPolicy._TOOL_ALLOWED` 为该工具加入 `fact_type`、`subject`、`value`、`unit`、`filters_applied`、`coverage`、`rows`、`samples`、`status`、`reason`；在 `LITE_TOOL_SCHEMAS` 加入相同能力边界的简版描述。
10. 在所有可使用家庭记忆工具的 Agent profile 中加入 `query_memory_facts`，让 JIT Prompt 和 ToolPolicy 都能提供该工具。
11. 修改 Planner 规则：单相册的媒体/时间/地点/已命名人物/处理状态统计声明 `structured_fact`；视觉、语义、关系与 OCR 金额不声明它。
12. 重跑任务 2 的测试与 `test_tool_registry_contract.py`，预期通过。

## 任务 3：让证据账本只接受完整的统计事实

**文件：**
- 修改：`backend/agent_runtime/runtime.py`
- 修改：`backend/tests/test_agent2_shadow_runtime.py`

1. 添加失败测试：`TaskState` 要求 `structured_fact`，`search_memories` 返回带 `total` 的预览；断言该要求仍为 open。该测试将 `search_memories` 工具契约设为不产生 `structured_fact`。
2. 添加失败测试：同样的 `TaskState`，`query_memory_facts` 返回 `status='unsupported_filter'`，断言需求未满足；返回 `status='ok'` 且 `coverage.complete=True`、有 `value` 时，断言需求满足。
3. 在 `record_agent2_tool_evidence()` 恢复按工具/状态收窄逻辑：
   - `query_memory_facts` 仅可尝试 `structured_fact`；
   - `status!='ok'`、`coverage.complete!=True` 或 `value is None` 时记录失败/不确定，不能关闭需求；
   - 成功时将 `value` 作为 `structured_fact` 账本条目，带 `filters_applied` 和 `fact_type`；检索预览、OCR 和人物画像不得再产生该证据类型。
4. 运行上述测试，预期通过。

## 任务 4：接入一次性 structured fact 软提醒

**文件：**
- 修改：`backend/agent_runtime/runtime.py`
- 修改：`backend/tests/test_agent2_shadow_runtime.py`

1. 添加失败测试：Planner 声明一个 `structured_fact`，脚本模型第一次直接 final、第二次调用 `query_memory_facts`、第三次 final；断言 Runtime 只注入一次提醒，且最终运行完成。
2. 添加失败测试：同一声明下，模型连续两次 final；断言第二次 final 被接受而不是再注入提醒或被强制改写。
3. 在 `AgentRuntime.run()` 初始化 `structured_fact_reminder_prompted=False`。
4. 在 authoritative Agent2 的 final 分支、一般 `unattempted` 提醒前，检测是否存在未尝试的 `structured_fact` 且 `query_memory_facts` 未调用；若满足，追加一次恢复消息：要求调用该工具获得全量统计，或在没有支持口径时直接 final 如实说明；设置 trace `final_gate.decision='continue_structured_fact_once'` 并 `continue`。
5. 该分支仅可执行一次；第二个 final 无条件进入现有放行路径。
6. 运行新增 runtime 测试，预期通过。

## 任务 5：回归、语法与 153 运行验证

**文件：** 无功能修改。

1. 在 153 运行：
   ```bash
   pytest -q backend/tests/test_structured_memory.py \
     backend/tests/test_retrieval_strategy.py \
     backend/tests/test_phase_d_tools.py \
     backend/tests/test_tool_registry_contract.py \
     backend/tests/test_agent2_shadow_runtime.py \
     backend/tests/test_agent2_evidence_recording.py
   python -m py_compile backend/structured_memory.py backend/agent_runtime/tools.py backend/agent_runtime/tool_policy.py backend/agent_runtime/runtime.py
   ```
   预期：全部通过。
2. 重启 153 的 8091 Sentrix 服务，使用其实际当前相册 scope 调用 Agent 或测试脚本验证：
   - 图片数不包含视频关键帧；
   - 视频数单列；
   - 不支持的视觉统计返回诚实说明；
   - 漏调统计工具时只出现一次软提醒。
3. 使用 `git diff --check`、`git status --short` 审核，只提交本项涉及的受跟踪文件；不处理既有未跟踪文件。
4. 在 153 的 `psh` 分支提交实现与计划，提交信息：`feat(agent): restore scoped structured fact queries`。
