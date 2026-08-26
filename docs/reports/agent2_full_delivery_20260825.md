# Sentrix Agent2 完整执行与交付报告

日期：2026-08-25  
权威代码与测试环境：`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`  
本地副本：`/Users/rm001/Sentrix-Home-Web-psh-a43ac327`  
模型：8100 `gemma4-12b-it`（12B，`max_num_seqs=12`）  
生产服务：8091，Agent2 authoritative，非 shadow 决策

## 1. 交付结论

本轮计划已完成代码落地、153 验证、三种候选召回策略的 100QA、轨迹审计和报告交付。当前 153 生产配置为：

```text
SENTRIX_AGENT_PROFILE=goal_driven_candidate
SENTRIX_CANDIDATE_STRATEGY=relevance_head_then_event_diversity
SENTRIX_VECTOR_BACKEND=qdrant
SENTRIX_AGENT2_ANSWER_CONTEXT=1
```

`goal_driven_candidate` 在当前代码中已是 authoritative Agent2；名称保留是兼容旧配置，生产路径不再把 Agent2 当作 shadow 观察层。旧 shadow profile 仅保留给历史回放和兼容测试。

最终健康检查：8091 `/api/health` 为 `ok`；Qdrant 346 collections、45,330 points、无降级错误；8101 BGE-M3 health 为 `ok`、1024 维；8100 进程确认为 12B、并发上限 12。

## 2. 已发现问题与处理结果

| 问题 | 根因 | 处理结果 |
|---|---|---|
| Agent2 规划只写 telemetry，未真正阻断 | planner/task state 与旧 CompletionState 脱节 | 已接入 authoritative TaskState；只有 required evidence 全部 satisfied 才允许确定性 final |
| 公开 evidence type 不统一 | `memory_reference` 等内部类型混入公共合同 | 新增唯一公共合同，统一 planner、registry、ledger、trace；公共集合为 `memory_asset/location_metadata/temporal_metadata/confirmed_identity/photo_identity/visual_observation/visible_text/structured_fact/user_statement` |
| 工具能力与证据需求没有统一注册 | JIT 依赖类型硬编码，缺少 prerequisite/precondition | ToolSpec 增加 `required_inputs/preconditions/prerequisite_evidence_types`；JIT 按 registry 做依赖选择 |
| inspect_photo 没有严谨身份来源 | 视觉观察和已确认人物身份混为一谈 | 增加只读 SQL identity join，仅返回 confirmed entity/mention 信息；没有确认身份时不声称身份 |
| photo1/内部 handle 泄漏 | emergency renderer 未经过 writer sanitizer | `sanitize_internal_refs()` 覆盖 `photo_N`、内部 asset/table 标识；所有 emergency 输出统一清理 |
| 只读重复调用被当作工具失败 | runtime 对完全相同的 read-only call 直接终止 | 增加调用结果缓存；有预算时复用观察结果，无预算时输出诚实 partial，不再伪装成工具拒绝 |
| JIT 显示了 profile 不允许的工具 | JIT 使用全局 ready registry，ToolPolicy 之后才拒绝 | JIT 增加 `allowed_tool_names`，只展示当前 profile 允许的工具，消除“模型看到的工具”和轨迹工具集合不一致 |
| 轨迹 evidence type/工具编号错位风险 | 显示层自行解释数字类型，且 trace 使用中间状态 | trace 使用 registry 字符串类型；回合结束覆盖为最终 TaskState；ResultSet 完整集仍服务端持有，preview 与 full projection 分离 |
| search_memories 描述缺失 | 返回字段没有稳定描述状态/摘要 | 已补 `evidence_summary/description_status` 等公开字段，并保留完整 ResultSet 服务端映射 |
| 结果集过大遮蔽关键图片 | preview 与 full ResultSet 混在模型上下文 | 已实现 `head_only`、`event_diversity`、`relevance_head_then_event_diversity`；模型只见 bounded preview，分页和 full 集合由服务端处理 |
| BGE/向量通道疑似无声退化 | sidecar 熔断与 Qdrant 锁降级路径可观测性不足 | 本轮完成只读诊断；当前 sidecar healthy，生产请求 visual/text ANN 均 ready/qdrant；P1 建议补熔断计数和 active-backend trace |

## 3. 代码验收

已同步并核对的关键文件包括：

- `backend/agent_runtime/evidence_contract.py`
- `task_state.py`、`tool_registry.py`、`tools.py`、`goal_planner.py`、`jit_prompt.py`
- `completion.py`、`runtime.py`、`tool_policy.py`、`profile.py`
- `final_writer.py`、`emergency.py`
- `scripts/runtime/start_sentrix_api_8091.sh`

本地与 153 对上述文件做 SHA-256 逐文件核对，无 mismatch。

153 验证结果：

```text
python3 -m py_compile（全部改动文件）    PASS
Agent2/registry/result-set targeted suite 128 tests, 0 failures, 0 errors
```

这 128 项覆盖 public evidence contract、TaskState 完成门控、registry prerequisite、JIT、preview handle scope、identity SQL、evidence recording、旧回放兼容和 result-set contract。

全量历史测试没有作为“全部通过”宣称：此前全量 738 项中有环境缺失导致的 hnswlib/qdrant/video 错误，以及 153 基线已有的 event segmentation/memory-space/OCR 等失败；这些不是本轮改动引入，已与改动前结果对照确认。

## 4. 8771 三策略 100QA

三轮均使用同一 album/scope、同一 100qa-full、同一 8100 12B、QA 并发 12、Judge 并发 12。所有 run 均在 153 完成，`run_valid=true`。

| 策略 | Run ID（末段） | Recall mean | Precision micro | F1 micro | AQ mean | Exact | Core | 100QA 完成内步率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| head_only（初始） | `35beb5` | 0.651 | 0.054 | 0.098 | 0.567 | 0.216 | 0.351 | 0.03 |
| head_only（中间复测） | `cbb251` | 0.635 | 0.073 | 0.129 | 0.398 | 0.153 | 0.245 | 0.06 |
| event_diversity | `0d57ff` | 0.626 | 0.060 | 0.106 | 0.464 | 0.196 | 0.268 | 0.04 |
| relevance_head_then_event_diversity | `a50068` | 0.636 | 0.064 | 0.112 | 0.521 | 0.202 | 0.319 | 0.08 |

说明：首轮 head_only 在 emergency/重复调用修复前运行，存在 19 个 `tool_rejected` 和 15 个 `photo_N` 泄漏，不能直接与修复后的策略作严格公平比较。修复后 event/hybrid 两轮均把 photo handle 泄漏降为 0，重复调用不再形成 duplicate-tool rejection；剩余 3 个拒绝均为 profile 不允许工具或预算耗尽的分页，不是 handle/evidence 类型错位。

在修复后可公平比较的两轮中，hybrid 的 answer quality、exact、core 均高于 event_diversity，因此已直接上线 hybrid。它的检索召回略高于 event，延迟约 64.4 秒/题；后续 P1 应优化 planner/证据闭环，而不是再扩大 preview 数量。

## 5. 仍未解决但已定位的问题

这些不是本轮 P0 代码错误，不能通过继续添加正则掩盖：

1. 12B planner 偶尔声明过宽或错误的 required evidence（例如计数问题声明成 memory_asset），导致检索正确但任务状态无法闭合。
2. 视觉复核返回 uncertain/partial 后，若当前候选没有新 handle 或预算耗尽，任务会诚实进入 `insufficient_evidence`；当前还没有“按 attempt ledger 自动切换下一个未尝试候选”的完整闭环。
3. `inspect_photo` 的人物数、OCR 数字、小物体等能力矩阵仍是 limited/experimental；Agent2 现在会阻止无证据的确定性回答，但不会提升底层视觉模型能力。
4. Qdrant 锁被其他进程占用时，非服务进程会降级 SQLite 全表扫；生产 8091 当前锁健康且使用 qdrant。hnswlib 在该远端环境不可用，不能用当前 SQLite 分支做等价 ANN 质量基线。

## 6. 下一阶段执行计划

### P1-A：证据闭环

- 在 requirement state 中记录 `tool + input_refs + normalized_question + outcome` 的 attempt ledger。
- 同一 handle/问题失败后自动选择当前 ResultSet 中未尝试的 handle。
- 每次失败重新执行 planner/JIT，输入未满足需求、已尝试集合、预算和可用工具；无新候选或预算耗尽才 partial。
- 验收：第一候选不确定→第二候选成功；全部候选失败→明确不足；两条轨迹都不得出现无新输入的重复调用。

### P1-B：向量与 sidecar 稳定性

- 在公开 debug trace 记录 qdrant lock、active backend、visual/text embedder availability、熔断计数。
- sidecar 连续失败时加入一次恢复探测，区分 `embedder_unavailable` 与“真实零召回”。
- 若要比较 Qdrant/SQLite，先在隔离目录构建同模型同维度索引，再测固定 query 集 Recall@K/延迟，禁止把全表扫结果当 ANN 基线。

### P1-C：planner 泛化

- 优先改 GoalPlanner taxonomy/examples 和结构化 evidence 约束，不再为 benchmark 新场景添加 intent 正则。
- `intent.py` 继续只承担安全兜底、预算和最终硬约束；`judge_faithfulness` 的 fail-open 按既定决策保持不变。

## 7. 交付物

- 本报告：`docs/reports/agent2_full_delivery_20260825.md`
- Wave2 诊断：`docs/reports/agent2_wave2_diagnostic_20260825.md`
- 153 当前生产进程已重启并验证，最终策略为 hybrid；无 shadow 层参与生产决策。

