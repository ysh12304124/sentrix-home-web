# Sentrix Agent2 证据闭环完整交付报告

日期：2026-08-25  
执行节点：`192.168.0.153`（权威代码与全部运行验证节点）  
模型：`gemma4-12b-it`（8100，12B，`max_num_seqs=12`）

## 1. 交付结论

P0–P8 的代码闭环已落地到 153：Planner 声明证据需求，统一 Evidence Contract 与 Tool Registry，JIT 只暴露当前未满足能力，TaskState 与 Evidence Ledger 绑定需求—证据，inspect_photo 只接受检索链中的只读身份，Final Gate 决定是否允许完成，唯一 Writer 只消费账本，8771 保存完整可审计轨迹。

代码一致性已核对。关键文件本地与 153 的 SHA256 一致；所有 benchmark 与服务验证均在 153 执行。

质量门结论以最终三策略结果为准：若没有策略同时达到计划阈值，报告会明确标记为“结构闭环通过、质量门未通过”，不会把低质量策略宣称为质量达标生产版本。

## 2. P0：基线与代码级问题

- 修复证据类型与工具编号/轨迹错位：轨迹使用注册表中的 canonical tool name 与 evidence type，不再依赖前端或旧编号解释。
- 修复 `search_memory`/`search_memories` 结果描述缺失：结果集含 `evidence_summary`、`description_status`、时间、地点和可审计 `result_set_id`。
- 修复 `photo1` 身份链：检索返回的 handle 必须原样进入 `inspect_photo`，禁止把展示序号、asset id、文件名混用。
- 修复状态覆盖：每个工具步骤记录调用前后 TaskState/requirement 状态，证据 ledger 追加而非覆盖。
- 修复同类型证据误满足：只有 `requirement_refs` 显式绑定到当前 requirement 的证据才能满足它。
- 增加 compile、contract、identity、trace 审计脚本与测试。

## 3. P1–P6：实现内容

### P1 统一合同与注册表

`EvidenceContract` 是唯一公共类型集合；`ToolSpec.produces_evidence_types` 与旧字段兼容但输出统一 canonical 字段。Planner、JIT、Ledger、Completion 和 Final Gate 都从同一注册表取能力。

### P2 检索—图片—检查身份链

`search_memories` 生成 result set 和 preview handle；`inspect_photo` 仅接受当前 scope、当前 result set 已返回的 handle。规范化前后参数、候选链、result_set_id、preview handles 均写入轨迹。越界句柄不会执行。

### P3 TaskState 与 Evidence Ledger

Requirement id 唯一；EvidenceEntry 记录 capability、type、input/provenance refs、scope、certainty、coverage、failure reason 和 requirement refs。需求状态只允许合法迁移，增加 `unavailable` 终态；attempt_count 只在真实尝试时递增。

### P4 Planner/JIT

Planner 只声明最小可回答证据；数量/存在/分组使用 `structured_fact`，日期使用 `temporal_metadata`，地点使用 `location_metadata`，视觉细节使用 `visual_observation`，文字使用 `visible_text`。JIT 根据未满足需求和注册能力动态给出工具，不再固定 legacy 工具序列。

### P5 Final Gate/Writer

权威模式只允许 Final Gate 根据 TaskState、Ledger、可用能力和预算做完成/不足判断；旧 intent/`_pending_resolution` 不能覆盖权威判断。唯一 Writer 强制消费 `EvidenceLedger.build_answer_context()`；权威模式自动启用该上下文。

### P6 防循环与完整轨迹

工具签名去重、失败尝试计数、预算耗尽和不可用能力均有终止路径。每个 model/tool/writer/gate 步骤记录输入、规范化参数、状态前后、证据 ids、writer 输入输出和 gate 决策。

## 4. 测试与验证

153 关键契约测试：21 tests，全部通过；编译检查通过。关键文件 SHA256 与本地一致。

本地完整 unittest 共 710 项，结果为 674 通过、20 skipped、8 failures、28 dependency/import errors；失败集中在既有 event segmentation/evidence bundle/delete/recovery 回归和本机缺少 pytest/cv2/hnswlib，不能作为 Agent2 契约通过的依据。153 完整 discover 另有既有模块路径导入冲突，已记录，不冒充通过。

## 5. 100QA 三策略

测试集：`album3-max`、`album_cba01be9502b`、`100qa-full`、8100 gemma4-12b-it、并发 12、8771。

| 策略 | Run ID | AQ | Exact | Core | 状态 |
|---|---|---:|---:|---:|---|
| head_only | `20260825-102230-album3-max-gemma4-12b-it-reuse-e5b8e3` | 0.536 | 0.227 | 0.309 | 已完成 |
| event_diversity | `20260825-095632-album3-max-gemma4-12b-it-reuse-e9aa53` | 0.505 | 0.211 | 0.295 | 已完成 |
| relevance_head_then_event_diversity | `20260825-101035-album3-max-gemma4-12b-it-reuse-520769` | 0.562 | 0.229 | 0.333 | 已完成 |

结构审计脚本：`scripts/benchmarks/audit_agent2_trace_contract.py`。最终 hybrid run 审计结果为：100/100 trace，缺失字段 0，非法证据类型 0，句柄越界 0，重复调用 0，Final Gate 非完成 0，Writer 缺失 0，`structural_pass=true`。head/event 的早期 run 含自动恢复分支旧轨迹；自动恢复和拒绝分支字段已在最终代码中补齐，后续运行按同一审计口径。

## 6. 生产选择与回滚

计划质量阈值：AQ ≥ 0.810、Exact ≥ 0.320、Core ≥ 0.490，且结构审计为零错误。三种策略均未达到质量阈值；最高为 hybrid（0.562/0.229/0.333），因此 8091 当前生产选择为 `relevance_head_then_event_diversity`（通过 `SENTRIX_CANDIDATE_STRATEGY` 设置），但明确标记为“结构闭环上线、质量门未通过”，不把它宣称为质量达标版本。证据身份、状态和轨迹修复不回滚。

生产冒烟（153/8091）已完成：健康检查通过；真实请求返回 `status=complete`，结果含 `agent2_trace.final_gate`、`evidence_ledger`、`writer_input`、`writer_output`，Planner→tool→Writer→guard 轨迹完整。该请求的回答因检索结果不足而诚实返回“现有照片里看不出来”，符合证据门语义。

## 7. 未解决问题与下一阶段

当前主要剩余风险是 12B Planner/工具选择和检索结果过宽导致的回答质量，而非并发（并发已固定为 12）。下一阶段优先：对 `structured_fact`/日期工具路由做离线题型回放；压缩 search preview 并保留高价值多样性；对视觉/文字需求增加受控 inspect/OCR 预算；把 100QA 的失败题按“规划错误、检索错误、证据不足、Writer 误答”分桶后再做针对性改动。

## 8. 交付物

- Agent2 运行时与 Evidence Contract/Registry/TaskState/Ledger/Final Gate/Writer 代码（本报告所在提交工作区及 153）。
- P0–P6 contract/identity/production tests。
- `scripts/benchmarks/audit_agent2_trace_contract.py`、既有 100QA root-cause 审计脚本。
- 本报告及三策略 100QA 原始 run/results/trace 文件。
