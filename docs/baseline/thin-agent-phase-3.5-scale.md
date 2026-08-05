# Phase 3.5.3 · 十万张压力回放报告

**完成时间**：2026-08-05
**环境**：153 `asus@192.168.0.153`, `.venv` (hnswlib 0.8.0, faiss-cpu 1.15.0, numpy 2.2.6)
**脚本**：`scripts/benchmarks/replay_hundred_thousand_scale.py`

## 参数

- 合成向量：**100,000**（`dim=512`，高斯分布，`seed=42`）
- 查询：**100 次**（hnswlib）/ **20 次**（SQLite baseline，因单次 ~3s）
- Top-K：10
- hnswlib 参数：`M=16, ef_construction=200, ef_search=50, max_elements=100000`

## 结果

| Backend | build_ms | p50 query ms | p95 query ms | p50 ≤ 2s | p95 ≤ 5s |
|---|---:|---:|---:|:-:|:-:|
| **hnswlib** | 11,162 | **0.141** | **0.204** | ✅ | ✅ |
| SQLite full scan | 1,393 | 2,969 | 3,101 | ❌ | ✅ |

## 分析

### hnswlib 目标达成

- **p50 = 0.141 ms**：低于门槛 **14,000 倍**（2s / 0.14ms）
- **p95 = 0.204 ms**：低于门槛 **24,000 倍**（5s / 0.2ms）
- 100 次查询无异常，索引稳定
- 一次性 build 11 秒，可用维护窗口全量重建

### SQLite baseline 超门槛证据

- **p50 = 2969 ms 违反 p50 ≤ 2s**
- 与 Phase 3.5.1 选型报告的线性外推预测（~5000 ms）在同一量级
- p95 = 3101 ms 勉强符合 p95 ≤ 5s，但 p50 已超，实际用户可感知的延迟不可接受

### 加速比

| 指标 | 加速比 |
|---|---:|
| p50 latency | **21,059×**（2969 ms / 0.141 ms） |
| p95 latency | **15,209×**（3101 ms / 0.204 ms） |

## 验收结论

- ✅ **hnswlib 满足原计划 §11 性能门槛 p50 ≤ 2s、p95 ≤ 5s**
- ✅ **`SENTRIX_ANN_INDEX_V1` 可以开启的最后一个先决条件已满足**
- ✅ 十万张切换前的 ANN 选型 + 增量 + 回表 + 重建 + 压力回放**全部完成**

## 生产切换的注意事项

1. **首次索引冷启动**：build 11 秒（100k 向量）。生产切换建议在维护窗口先跑一次完整 build 并 save 到磁盘；service 重启后走 `HnswlibIndex.load()` 从磁盘恢复，规避冷启动成本。
2. **max_elements 预留**：建议 `max_elements=200_000` 或 `2 * 当前向量数`（`HnswlibIndex._DEFAULT_MAX_ELEMENTS`）；hnswlib 支持 `resize_index` 但会 lock，最好一次给足。
3. **scope 回表**：`EvidenceRetrievalKernel` 已经在硬过滤阶段做 scope 检查，ANN 只召回候选。当 scope 命中稀疏时（例如 `all_authorized` 跨 3 相册），index 内部会 `k * 4` 过取候选，代码再筛选。
4. **delete 的持久化**：`HnswlibIndex.save/load` 保存了 `deleted_labels`，加载后能正确恢复删除状态。
5. **flag 顺序**：`SENTRIX_ANN_INDEX_V1` 应在 `SENTRIX_EVIDENCE_RETRIEVAL_V1` 之后开启（原计划 §14），因为 ANN 走 kernel → 派生投影 → ANN 三层。

## 报告 JSON

- hnswlib：`asus@192.168.0.153:/tmp/phase3.5.3-hnswlib-100k.json` → `/tmp/phase3.5.3-hnswlib-100k.json`
- sqlite baseline：`asus@192.168.0.153:/tmp/phase3.5.3-sqlite-100k.json` → `/tmp/phase3.5.3-sqlite-100k.json`

## Phase 3.5 完成

- ✅ 3.5.1 库选型：hnswlib（用户选定）
- ✅ 3.5.2 backend 实装：`backend/retrieval_ann.py:HnswlibIndex`，11 tests 全绿（5 baseline + 6 hnswlib）
- ✅ 3.5.3 十万张压力：p50 0.141 ms, p95 0.204 ms 双双达标

至此 Phase 0-8 的 8 大阶段 + 补充计划 2R 全部完成。剩余待办：

1. `MemoryStore.apply_authorized_revision` 迁进 db.py（当前 `MemoryCorrections` 服务实现等价）
2. `scripts/maintenance/build_core_memory.py` — 从 canonical 派生首次 Core Memory Cards
3. 生产切换：按原计划 §14 顺序逐个开 flag，每一步跑真实相册回放
