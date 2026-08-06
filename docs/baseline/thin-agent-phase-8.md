# Phase 8 · 灰度切换与 flag 布线

**完成时间**：2026-08-05
**Shadow diff report**：`/tmp/phase8-shadow.json`

## Feature Flags 全景

| Flag | 阶段 | 默认 | 说明 |
|---|---|---|---|
| `SENTRIX_THIN_AGENT_V1` | Phase 4 骨架 → 全阶段 | off | Thin Agent 运行时总开关（旧 agent.py 与新 thin_runtime 之间） |
| `SENTRIX_SEMANTIC_QUERY_PARSER_V1` | Phase 2R | on-by-code | QueryParser 恒开；此 flag 作为影子指标 marker |
| `SENTRIX_EVIDENCE_RETRIEVAL_V1` | Phase 3 | off | 派生投影读取路径；off 时走 canonical Observation 扫描 |
| `SENTRIX_ANN_INDEX_V1` | Phase 3.5 | off | ANN 索引路径；十万张压力回放通过后可开 |
| `SENTRIX_LLM_CLAIM_EXTRACTOR_V1` | Phase 4 复杂路径 | off | 复杂人物/事件路径接入 LLMClaimExtractor + Writer + Verifier |
| `SENTRIX_CORE_MEMORY_V1` | Phase 5 | off | Contextual 模式使用 Core Memory Cards 而非 confirmed entity 占位 |
| `SENTRIX_MEMORY_CORRECTION_V1` | Phase 6 | off | 记忆纠正 propose/apply API（Agent-owned revisions） |
| `SENTRIX_ADVANCED_MEMORY_TOOLS_V1` | Phase 7 | off | 高级工具（summarize_person/timeline/compare/pattern） |
| `SENTRIX_EXPLICIT_IMAGE_REINSPECTION` | Phase 4 | off | 显式原图重读授权（默认 off，用户明确要求"仔细核实"时开） |

## 生产切换顺序（原计划 §14）

1. `SENTRIX_SEMANTIC_QUERY_PARSER_V1` — QuerySpec 只记录不生效（shadow）
2. `SENTRIX_EVIDENCE_RETRIEVAL_V1` — 新旧检索影子对比通过后启用
3. `SENTRIX_ANN_INDEX_V1` — 完成 ANN 选型 + 十万规模压力回放
4. 小范围切换 evidence 查询
5. `SENTRIX_THIN_AGENT_V1` — 切换 Thin Agent runtime
6. `SENTRIX_LLM_CLAIM_EXTRACTOR_V1` — 启用复杂路径
7. `SENTRIX_CORE_MEMORY_V1` — 启用 Core Memory
8. `SENTRIX_MEMORY_CORRECTION_V1` — 启用 Correction
9. `SENTRIX_ADVANCED_MEMORY_TOOLS_V1` — 启用高级工具（最后）

每一步在 153 上跑一次真实旧相册回放（明哥、家人介绍、日期、负样本、原图请求）——全部符合规则再前进。

## 回滚

- 关任一 flag → 旧路径立即恢复
- MemoryStore 与 FMA 5173 永远不因 Agent flag 停止
- Phase 3+ 新增的 Agent-owned 表在 flag 关闭时也保持存在，不影响 canonical

## Shadow diff 结果（`shadow_run_diff.py`）

```
case_count=4 divergent_count=2

kitchen_month  diverges=True  old evidence=0   thin evidence=1  # thin correctly time-filters
person_intro   diverges=False old evidence=1   thin evidence=1  # equivalent
writing_prompt diverges=False old evidence=0   thin evidence=0  # both correctly refuse
empty_result   diverges=True  old evidence=0   thin evidence=0  # thin runs evidence-mode path
```

两处 diverge 均是**新版本更正确**：
- `kitchen_month`：老 agent 缺月份硬过滤，thin agent 正确锚定
- `empty_result`：老 agent 直接进 none，thin agent 走 evidence 模式后正确返回空

不存在 thin agent 比 old agent 更差的 case。

## 已通过的量化门槛

- 单元测试全绿（334 tests，仅 pre-existing PIL 错误）
- Phase 1 benchmark `evaluate_evidence_retrieval.py`：Thin Agent V1 ON = 10/10
- Phase 2R semantic benchmark `evaluate_thin_agent_semantic.py`：24/24（paraphrase 15/15 + contrast 8/8 + composite 1/1）
- Shadow diff：2 divergences, 均是改进
- 语法（`compileall`）与 `git diff --check` 通过

## 待用户点头的最后事项

1. **Phase 3.5.1 ANN 选型**：需要在 153 上跑 `evaluate_ann_libraries.py` 输出报告后由用户选库
2. **Phase 3.5.3 十万张压力**：ANN 选型后跑 `replay_hundred_thousand_scale.py`
3. **生产切换时段**：flag 逐个开启时间
4. **153 同步**：本地 `psh` 领先 8 个 commit（Phase 0/1/2R/3/3.5/4/5/6/7/8）+ 未提交的 model_clients.py 与 PROJECT_MEMORY.md 冲突处理

## 未完成 / 后续增强

- `MemoryStore.apply_authorized_revision` 未直接落到 db.py（当前经 `MemoryCorrections` 外部服务，未来可迁）
- Core Memory 初次建卡的 `scripts/maintenance/build_core_memory.py` 未落地（Phase 5 只完成运行时接入，历史数据首次建卡待补）
- ANN 真实库集成（FAISS/HNSW）待用户选型后加实现分支

## 结论

Thin Agent 与证据检索内核**核心搭建完成**：Phase 0-8 全部落地。所有 flag 默认 off，可按 §14 顺序小范围切换。语义正确性通过 24/24 semantic benchmark、10/10 evidence retrieval benchmark 验证。
