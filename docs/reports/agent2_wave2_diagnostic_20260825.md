# Agent2 Wave 2 只读诊断报告

日期：2026-08-25  
基线：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）  
服务：8091 Agent2 authoritative、8100 `gemma4-12b-it`（12B，`max_num_seqs=12`）

## 结论摘要

1. 153 的 BGE-M3 sidecar（8101）当前健康，返回 `status=ok`、维度 1024；本轮 100QA 的检索轨迹没有出现 `embedder_unavailable`。因此“文本检索经常退化”的主要证据不是当前 sidecar 故障，而应继续关注熔断窗口和故障时的可观测性。
2. 8091 当前使用 `SENTRIX_VECTOR_BACKEND=qdrant`。100QA 轨迹中 visual/text ANN 均报告 `status=ready, backend=qdrant`；单独启动第二个 Python 进程时因 Qdrant 文件锁无法取得，`vector_search_status()` 会明确降级到 SQLite 全表扫描。hnswlib 依赖在该远端环境不可用，SQLite 分支没有可用 ANN 索引，不能把该分支当作等价的质量基线。
3. Wave 1 的模型优先门控已经生效，但仍存在“工具已得到候选、视觉复核不确定后，预算耗尽即收尾”的证据闭环缺口。第一轮最终 100QA 中可见 `task_state.status=in_progress`、终止原因 `insufficient_evidence` 的样本；这不是并发 12 的问题，而是没有基于未满足需求和未复核候选继续规划的新一轮闭环。
4. 发现一个独立的代码级问题：相同的只读工具调用会被 runtime 当作重复调用拒绝，造成 `tool_rejected:duplicate_tool_call`。已在 P0 直接修复为复用相同调用的缓存观察结果，并将 emergency 文案统一清理 `photo_1` 等内部 handle。该改动已同步 153，语法和针对性测试通过，后续第二轮 100QA 用于实测收益。

## 证据

### Sidecar 与通道

- 153：`curl http://127.0.0.1:8101/health` 返回 BAAI/bge-m3、1024 维。
- 8091 环境：`SENTRIX_TEXT_EMBEDDER=bge`、`SENTRIX_VECTOR_BACKEND=qdrant`。
- 第一轮 100QA（`20260825-075324-album3-max-gemma4-12b-it-reuse-35beb5`）的 81 次 search 轨迹中，visual_ann/text_ann 都是 `status=ready`；未出现 `embedder_unavailable`。
- 非服务进程尝试同时打开 Qdrant 时记录了锁降级；这说明“锁竞争导致 SQLite 兜底”是可复现的运行模式，但不应在服务持锁时再启动第二个 Qdrant 使用者。

### 证据闭环

第一轮中，任务级汇总为：`task_complete=78`、`insufficient_evidence=33`（按 planner terminal reason 汇总）；部分样本在工具已经返回候选后，视觉观察仍为 `uncertain` 或 `partial_coverage`，最终状态保持 `in_progress`。现有 `RequirementCompletion.allowed_capabilities()` 只按未满足 evidence type 返回能力，没有记录“已尝试过哪个 handle/问题”并据此切换到下一个候选；这解释了“一次性放弃/重复相同复核”的现象。

## P1 实施建议（本轮只诊断，不在此报告内改动）

### P1-A 证据闭环

- 为每个 requirement 增加 attempt ledger：`tool_name + input_refs + normalized_question + outcome`。
- 视觉/OCR 复核失败时，从当前 ResultSet 选择尚未尝试的 handle；同一 handle 同一问题不再重复调用。
- 每次失败后重新执行 planner/JIT，输入必须包含未满足需求、已尝试集合、剩余预算和可用工具；只有无新候选或预算耗尽才进入 `insufficient_evidence`。
- 验收：同一问题至少覆盖“第一候选不确定→第二候选成功”和“所有候选失败→诚实不足”两条轨迹；不得出现无新输入的重复调用。

### P1-B 文本/视觉通道稳定性

- 保持 Qdrant 为 8091 唯一生产向量后端；将 Qdrant 锁状态、active backend、每通道 embedder 状态写入公开 debug trace。
- sidecar 连续失败/熔断时，增加一次健康恢复探测和明确的 `embedder_unavailable` 计数，避免把空召回误判为“没有相关记忆”。
- 不把当前不可用的 hnswlib/SQLite 分支作为质量对照；若要比较，先在隔离目录构建同模型同维度索引，再用固定查询集测 Recall@K/延迟。

## P0 修复验收

- `runtime.py`：相同只读调用有缓存观察结果时复用，不再直接终止为 duplicate_tool_call。
- `emergency.py`：所有 emergency 输出经过 `sanitize_internal_refs()`，不泄漏 `photo_N`、asset/table 内部标识。
- 153：目标文件 `py_compile` 通过；`test_agent2_production_contract`、`test_agent2_evidence_recording`、`test_agent2_shadow_runtime` 共 16 项通过。

