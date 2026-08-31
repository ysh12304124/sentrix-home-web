# Sentrix Agent2 当前规划最终交付报告

日期：2026-08-24  
权威基线与验证环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`，分支 `psh`）  
本地副本：`/Users/rm001/Sentrix-Home-Web-psh-a43ac327`

## 1. 交付结论

本轮计划的代码实现、153 部署、服务重启、回归测试、Qdrant 索引验证和 100QA 实测已经形成闭环。最终代码已同步到 153，8091 服务正在使用新代码，Qdrant 为当前活动检索后端，100QA 最终运行 100/100 完成且 Judge 100/100 有效。

需要明确的边界是：本轮已经修复“失败证据被当作成功”和“证据不足时继续输出无依据结论”的主要路径，但没有宣称全量证据闭环已经完成。最终 100QA 仍记录 32 个 `open/candidate` 任务状态，说明下一阶段仍需做状态闭环与工具失败后的自适应重规划；该残余问题已在第 7 节列出。

## 2. 与 153 的一致性

- 153 是本轮唯一权威代码与测试环境。
- 本地工作区已同步 153 本轮涉及的运行时代码、检索代码、维护脚本和新增测试。
- 关键文件在同步后逐一做 SHA-1 对比；报告生成后再次执行同步和一致性校验。
- 没有创建提交或推送分支，保留当前工作区修改供主 agent 审阅。

## 3. 已发现问题与解决情况

### 3.1 已解决

1. `GoalPlanner/agent2_shadow` 只写 telemetry、不参与 CompletionState 门控。现在模型结构化证据需求优先进入门控，`intent.py` 仅在模型判断缺失时作为兜底，避免继续按 benchmark 场景堆叠正则。
2. `_search_recommends()` 只读嵌套 `observation`，读不到扁平化 `recommended_resolution`。现在兼容扁平和旧嵌套格式。
3. 工具结果只要出现工具名就被视为成功。现在 OCR 必须有非空 `ocr_text/full_text/exact_values/text_regions`，视觉检查必须有实际 observation；`failed/partial/uncertain/unavailable` 不再满足需求。
4. 工具失败没有进入证据账本，任务状态会被错误推进。现在失败会写入 `Coverage(failed=1)`、`failure_reason`，并保持需求 active/open。
5. 证据结果在 `TaskState` 中没有 candidate/supported/confirmed/failed 区分。现在 `RequirementState` 显式保留 `coverage_status` 和 `failure_reason`，并在 trace 中公开。
6. 视觉/OCR 需求在失败后可能重复取同一候选或预算耗尽后继续生成答案。现在有未检查候选选择、一次有界自动补证；在 completion retry 用尽或模型预算不足时，转为自然的证据不足回答，避免继续编造。
7. `photo_1` 等短内部 handle 可能泄漏到最终回答。`final_writer` 增加内部 handle 清理规则。
8. Qdrant 批量维护能力不足、全量重建容易长时间占用资源。新增 `upsert_many` 和按 scope 重建；本次仅重建 `album_cba01be9502b`，未修改 SQLite 数据。
9. Qdrant 可用但空结果时，原逻辑静默落到通用 SQLite 兜底，无法区分“没有结果”和“后端异常”。现在记录 `qdrant_no_collection`/`qdrant_empty` 等显式原因。
10. OCR 失败没有 worker 级统计，且 telemetry snapshot 对顶层标量处理错误。现在记录 attempts/successes/failures、provider failure reasons，`/api/telemetry/ocr` 已恢复 200。

### 3.2 按用户决策保持不变

- 没有继续扩张 `intent.py` 场景正则。
- 没有改变 `judge_faithfulness` 的 fail-open 取舍。
- 没有把并发 12 当成根因，也没有修改 benchmark 并发配置。
- 没有把 Judge 离线评测器与生产 Agent Judge 混为一谈。

## 4. 代码落地范围

核心修改集中在：

- `backend/agent_runtime/completion.py`
- `backend/agent_runtime/runtime.py`
- `backend/agent_runtime/task_state.py`
- `backend/agent_runtime/result_set.py`
- `backend/agent_runtime/ocr_tool.py`
- `backend/agent_runtime/final_writer.py`
- `backend/qdrant_memory.py`
- `backend/db.py`
- `scripts/maintenance/sync_qdrant_vectors.py`

配套增加/更新了 Agent2、ResultSet、TaskState、Qdrant 和 OCR telemetry 测试，以及 100QA 根因审计脚本和诊断脚本。

## 5. 验证结果

### 5.1 代码级回归

153 上执行：

```text
PYTHONPATH=. python3 -m unittest -q \
  backend.tests.test_agent_task_state \
  backend.tests.test_agent2_shadow_runtime \
  backend.tests.test_result_set_contracts \
  backend.tests.test_requirement_completion \
  backend.tests.test_goal_planner \
  backend.tests.test_agent_runtime_trace_contract \
  backend.tests.test_agent_contracts_v2 \
  backend.tests.test_tool_loop_truth_contract \
  backend.tests.test_phase_d_runtime_contracts \
  backend.tests.test_phase_d_d12_place_retrieval \
  backend.tests.test_phaseg_guard_tiers \
  backend.tests.test_runtime_message_order \
  backend.tests.test_agent2_evidence_recording
```

结果：**103 tests passed，0 failures**。

Qdrant 专项：`.venv/bin/pytest -q backend/tests/test_qdrant_memory.py`，结果：**7 passed**。

另外，对本轮涉及 Python 文件执行 `py_compile`，并执行 `git diff --check`，均通过。全量项目测试中仍有 22 个失败，但已核实为本地缺少 `hnswlib`、153 原有失败或环境问题，不是本轮新增回归；本报告不把它们冒充为全量通过。

### 5.2 Qdrant 与服务健康

153 上对 scope `album_cba01be9502b` 做定向重建后：

- Qdrant collections：346
- Qdrant points：45,330
- 该 scope SQLite vectors：3,620
- 10 个 benchmark 查询 top-k overlap：**1.0**
- SQLite 平均查询：约 85.026 ms；Qdrant 平均查询：约 4.336 ms
- SQLite p95：约 112.175 ms；Qdrant p95：约 6.554 ms

8091 当前 `/api/health` 显示：

- `status=ok`
- `vectorIndex.backend=qdrant`
- `qdrant_available=true`
- `degraded=false`
- `active_backend=qdrant`
- `collections=346`、`points=45330`
- 最近 `text_ann`/`visual_ann` 路由均为 Qdrant，未出现降级错误

`/api/telemetry/ocr` 返回 HTTP 200；独立失败注入 smoke test 能记录 `asset_path_unavailable` 及失败计数。

### 5.3 最终 100QA

最终 run：`20260824-203300-album3-max-qwen3.5-0.8-lora-v2-reuse-f2a498`，相同 album scope、模型和 QA 集合。

| 指标 | 基线 run `20260824-143818` | 最终 run | 变化 |
|---|---:|---:|---:|
| retrieval recall mean | 0.616 | 0.628 | +0.012 |
| answer quality mean | 0.76 | 0.77 | +0.01 |
| exact accuracy | 0.33 | 0.32 | -0.01 |
| core accuracy | 0.43 | 0.45 | +0.02 |
| QA completion | — | 100/100 | — |
| Judge valid | — | 100/100 | — |
| within-step rate | — | 0.83 | — |

最终路由统计：`search_memories=97`、`inspect_photo=28`、`read_photo_text=10`；没有 page 调用。审计文件位于 `docs/reports/20260824-p1-final-100qa-audit.json`。

结果说明：检索召回、答案质量和 core accuracy 有小幅改善，但 exact accuracy 有 0.01 波动，不能据此宣称整体问答已经解决。最终运行中 83 项正常 complete，12 项触发 tool-call limit，3 项 parse failure，2 项 tool rejected；Judge 本身全部有效。

## 6. 根因判断

本轮证据支持的主要根因不是并发，而是：

1. 证据需求状态和工具真实结果之间存在语义断层；
2. 失败/空结果缺少显式 failure ledger，导致门控误判或无法继续规划；
3. 检索后端之前缺少可观测的降级原因，Qdrant 与 SQLite 路径行为不透明；
4. benchmark 中 Agent2 仍可能在需求保持 `open/candidate` 时结束，说明“证据需求 → 下一工具 → 证据确认/降级”的闭环尚未完全收敛。

## 7. 残余问题与下一步优先级

### P1：证据闭环继续收口（最高优先级）

最终 100QA 的 Agent2 trace 仍有 32 个 `open/candidate` 需求状态。下一轮应把“候选已看过但不支持”“工具失败”“模型预算耗尽”“工具调用被拒绝”统一为可判定状态，并由状态机选择下一候选或明确 partial，不让 `open` 状态直接成为正常 complete 的隐含终点。

### P1：工具结果与最终 writer 的统一边界

最终样本中仍能看到少量历史/路径上的内部引用（例如 `photo_1`）或带猜测措辞的回答，说明 writer 清理和 benchmark 输出路径还需要端到端断言，而不能只测独立函数。

### P2：检索工具返回集合治理

Qdrant 路径已经稳定且性能显著更好，但召回集合仍可能过大。下一步应在工具层增加候选分层、去重、每组上限和“先摘要/再精查”的契约，并用 100QA 的 full-recall 分层指标验证，而不是单纯提高 top-k。

### P2：原始图像/关键帧记忆完整性

本轮没有改写图像记忆生成策略。应单独建立“原图/关键帧 → 结构化记忆”的字段覆盖评测，区分视觉对象、人物、文字、时间地点、关系和不确定性，避免摘要阶段的信息损失传递到检索和回答。

## 8. 交付物

- 本报告：`docs/reports/20260824-sentrix-agent2-final-delivery.md`
- 100QA 审计：`docs/reports/20260824-p1-final-100qa-audit.json`
- Wave2 诊断：`docs/reports/20260824-sentrix-wave2-diagnostic.md`
- P1 过程记录：`docs/reports/20260824-sentrix-p1-evidence-gate-progress.md`

报告文件已写入本地并同步至 153；代码和测试修改均保留在工作区，未创建 commit。
