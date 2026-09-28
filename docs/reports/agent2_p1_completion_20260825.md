# Sentrix Agent2 P1 完成与验收报告

日期：2026-08-25  
权威环境：`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`  
模型：8100 `gemma4-12b-it`（12B，vLLM `max_num_seqs=12`）  
生产服务：8091；Agent2 authoritative；检索策略 `relevance_head_then_event_diversity`

## 结论

P1 的代码任务、153 部署、定向测试、生产 smoke 和完整 100QA 回归已完成。P1 代码改动已直接上线，没有新增 shadow 层。最终 100QA 回归有效完成 100/100，运行 ID：

`20260825-091056-album3-max-gemma4-12b-it-reuse-55e49e`

P1 解决了“失败证据没有可审计输入/结果”“失败后没有有界换候选重试”“向量通道降级不可见”“planner 重复/过宽声明缺少约束”这些代码和可观测性问题；但它没有提升底层 12B 处理复杂多证据任务的能力，因此回答质量指标没有优于 P0 已选 hybrid 基线。这一结果本身已验证并记录，不能把 P1 宣称为 benchmark 质量提升。

## 已实现

### P1-A 证据闭环

- `RequirementState` 增加有界 attempt ledger（最近 8 次），记录 `tool`、`input_refs`、问题、证据类型、coverage、outcome、failure reason。
- `record_agent2_tool_evidence()` 对成功、partial、失败和空 observation 都写入具体尝试。
- visual/OCR 失败或不确定时，runtime 在当前结果集选择尚未尝试的 preview handle，最多自动复核 3 次；无新 handle、无工具预算或全部失败才进入 partial/insufficient evidence。
- 同一只读工具调用继续使用缓存观察结果，不再伪装成工具失败；公开 trace 增加 `required/attempt_count/last_attempt`，完整 debug trace 保留 attempt 细节。
- 修正 attempt 计数语义：只有实际工具尝试才递增，`mark_running()` 不再把状态迁移误计为一次尝试。

### P1-B 向量与 sidecar 可观测性

- embedding router 暴露 visual/text 的 configured、available、model、dimension 状态。
- BGE-M3 暴露连续失败次数、熔断状态和 health check 状态。
- 每次 visual/text ANN 轨迹写入 embedding status；vector health 写入 Qdrant lock、active backend 和 degraded reason。
- 153 当前生产检查：Qdrant 346 collections、45,330 points、服务进程持锁，检索使用 Qdrant；没有把 SQLite fallback 当作质量基线。

### P1-C planner

- GoalPlanner prompt 约束最小充分证据：结构化问题优先 `structured_fact`，视觉/文字/地点/身份使用对应类型，禁止无关 `user_statement` 和同类型重复需求。
- 归一化层去重 evidence type，并保留模型显式 `required` 标记；需求均需能由注册表能力或 prerequisite 满足。

## 验证结果

153 定向测试：

```text
Ran 130 tests in 60.999s
OK
```

本地 P1 关键测试：21/21 通过。全部 P1 目标文件与 153 SHA-256 一致；远端 py_compile 通过。8091 重启后 health 和 Qdrant Level-1 检查通过。

生产 smoke 验证了：模型调用来自 8100 12B；轨迹包含 planner、search、inspect、TaskState requirements、attempt ledger；embedding visual/text 均报告 `available=true`、backend `qdrant`。对于没有直接支持的候选，系统输出 partial/无法确认，而不是编造结论。

## 100QA 回归

P1 回归运行：`20260825-091056-album3-max-gemma4-12b-it-reuse-55e49e`，100/100 完成，run valid。

| 指标 | P1 回归 |
|---|---:|
| retrieval recall mean | 0.696 |
| retrieval precision micro | 0.065 |
| retrieval F1 micro | 0.118 |
| answer quality mean | 0.398 |
| exact accuracy | 0.153 |
| core accuracy | 0.245 |
| task decision accuracy | 0.405 |
| within-step completion | 0.050 |
| terminal `task_complete` / `insufficient_evidence` | 67 / 44 |
| search / inspect / OCR calls | 142 / 35 / 4 |

与已选 P0 hybrid 回归（AQ 0.521、Exact 0.202、Core 0.319）相比，P1 代码没有带来 benchmark 质量提升。原因从轨迹可定位为：12B 仍会为复杂问题声明过宽的 required evidence，搜索 preview 常为 partial coverage，且人物身份等能力不能由普通视觉观察替代；系统现在会把这些状态如实标为未完成，而不是错误放行。该差异不是并发问题，8100 并发仍为 12，Qdrant/sidecar 也健康。

## 当前上线状态与后续优先级

当前 153 生产继续使用已验证的 hybrid，不回退到 head_only，也不新增 shadow。P1 已闭环的代码无需再做 shadow 验证。

后续真正能提升 AQ/Exact/Core 的工作应是 P2：

1. 给 planner 增加按问题类型的最小证据模板和身份/视觉能力边界校验，避免声明不可满足的 required evidence。
2. 对 partial ResultSet 自动优先走 `get_result_page` 或按候选 handle 复核，而不是重复改写 query；把“候选覆盖不足”和“没有匹配”分开。
3. 对 photo identity 建立 confirmed face/entity 的独立证据链；视觉模型只回答“看到了什么”，不能推导人名。
4. 以固定 QA 子集单独评估上述 planner 改动，验收标准为 AQ、Exact、Core 至少不低于 P0 hybrid，且 `photo_N` 泄漏、证据类型/轨迹错位、重复同输入调用均为 0。

