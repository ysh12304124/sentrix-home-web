# Phase R3 + R3B · 词法/条件证据/融合 + Seed-based Adjacency & 去重分组

**日期**：2026-08-06
**状态**：代码 + 单测完成（407 tests 全绿，含新增 25）

## R3 交付物

### `evidence_retrieval.py`
| 改动 | 说明 |
|---|---|
| `_contains` 只留完整子串（P0-6） | 删 tokenized all-match；`_contains("卧室睡衣自拍","浅黄色拼接毛绒睡衣自拍")=False`；杜绝 album1-01 单字 FP |
| `_MATCHED_SOURCE_TYPES` 白名单（P1-2） | `asset_metadata / observation_field_exact / confirmed_bridge / entity_bridge_confirmed / subject_binding` 才能 matched；`_evaluate` 对白名单外 source 降级 matched→possible |
| `_evaluate_single_value` / `_evaluate_activity` matched source → `observation_field_exact` | place/activity 精确命中才算直接证明 |

### `retrieval/fusion.py`（R2 已建，R3 测试补齐）
RRF(k=60) + evidence_class 分级（anchor boost / semantic rank fusion / expander 继承）。

### 测试
`test_evidence_bundle` 扩展（ContainsSemantics 4 + MatchedSourceWhitelist 3）、`test_retrieval_fusion`（10）。

## R3B 交付物

### `retrieval/adjacency.py`（P0-9 实装）
- `expand(seed_asset_ids, filters, limit)`：event 扩展（`event_observations` join）/ 时间窗扩展（captured_at ±window）/ batch 扩展（source_album_id / source_device_id）；每边预算；scope/media/time 重过滤；seed 自身排除。
- **不是第一轮并行 retriever**：Kernel 只在 primary recall → 评估 → **seed-quality gate**（exact/strong）后调用 `expand`。

### `evidence_retrieval.py` `_retrieve_multi` 重构
primary recall → `_evaluate_fused`（评估 + attributions + fusion_score）→ seeds（exact/strong）→ adjacency expand → 再评估（跳过已存在）→ 合并排序。

### `retrieval/near_duplicate.py`（P0-13）
- SHA-256（content_sha256）分组；CLIP cosine 辅助阈值来自 config。
- **只注解不删结果**：`best/top_k` 折叠展示由 UI 做（"组内还有 N 张"）；`all_relevant` 保留全部；**retrieval metrics 不因分组改变召回率**。
- `thin_agent._evidence_answer` 对 evidence 项注解 `near_duplicate_group/size`。

### 测试
`test_adjacency_retriever`（5：event/time-window/scope/no-seeds/seed-excluded）、`test_near_duplicate`（3：sha 分组 / annotate / all 保留）。

## 本地验证

```
unittest discover backend.tests → 407 OK (skipped=1)
```

## 工程门槛（审阅 §7.4，R7 前不要求 90%/95%）

- hard violation = 0（kernel 硬过滤 + runner 独立复检）
- strict-empty FP = 0（empty_policy 在 R6 落实）
- 与 R1A 比 Recall/MRR 明确提升（153 真实数据，待跑）
- 无 case-specific 规则（`test_no_benchmark_runtime_dependency` 守护）
