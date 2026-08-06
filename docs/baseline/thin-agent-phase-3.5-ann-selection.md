# Phase 3.5.1 · ANN 选型报告

**采集时间**：2026-08-05
**环境**：153 `asus@192.168.0.153`, `.venv` (faiss-cpu 1.15.0, hnswlib 0.8.0, numpy 2.2.6)
**数据源**：153 生产 `data/sentrix.db`（visual 526、semantic 374、episodic 565 向量）
**参数**：limit_per_space=500，queries=50，k=10
**报告 JSON**：`asus@192.168.0.153:/tmp/ann-report.json`（本地 `/tmp/ann-report-153.json`）

## 结果矩阵

### Visual（asset-level CLIP 向量，dim=512）

| 库 | build_ms | p50_ms | p95_ms | mem_kb | Recall@10 | 增量 add | 支持 delete |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| SQLite full scan | 0.0 | 25.338 | 26.216 | 70 | 1.000 | Y | Y |
| **FAISS HNSW32** | 14.2 | **0.016** | 0.024 | 10 | **0.992** | Y | **N** |
| FAISS IVFFlat | 130.9 | 0.008 | 0.010 | 5 | 0.700 ❌ | Y | N |
| FAISS Flat | 0.2 | 0.014 | 0.015 | 5 | 1.000 | Y | N |
| **hnswlib** | 3.3 | 0.018 | 0.023 | **5** | **0.998** | Y | **Y** |

### Semantic（observation 语义向量）

| 库 | build_ms | p50_ms | p95_ms | mem_kb | Recall@10 | 增量 add | 支持 delete |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| SQLite full scan | 0.0 | 18.851 | 18.999 | 51 | 1.000 | Y | Y |
| FAISS HNSW32 | 1.4 | 0.014 | 0.019 | 11 | 0.996 | Y | N |
| FAISS IVFFlat | 1.4 | 0.008 | 0.011 | 9 | 0.664 ❌ | Y | N |
| FAISS Flat | 0.1 | 0.011 | 0.012 | 5 | 1.000 | Y | N |
| **hnswlib** | 2.2 | 0.016 | 0.019 | 5 | **1.000** | Y | **Y** |

### Episodic（event 向量）

| 库 | build_ms | p50_ms | p95_ms | mem_kb | Recall@10 | 增量 add | 支持 delete |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| SQLite full scan | 0.0 | 25.168 | 25.545 | 47 | 1.000 | Y | Y |
| FAISS HNSW32 | 1.8 | 0.014 | 0.018 | 5 | **1.000** | Y | N |
| FAISS IVFFlat | 8.6 | 0.008 | 0.012 | 5 | 0.724 ❌ | Y | N |
| FAISS Flat | 0.1 | 0.014 | 0.015 | 5 | 1.000 | Y | N |
| **hnswlib** | 2.9 | 0.016 | 0.019 | 5 | **0.998** | Y | **Y** |

## 分析

### 淘汰

- **FAISS IVFFlat**：Recall 0.66-0.72，**远低于原计划 §12.3 门槛 Recall@10 ≥ 0.90**。本次 500 训练点远少于 IVF 聚类需要的 1200+，即使扩到十万张后 nlist 调优也很难追平 HNSW 家族。**排除**。
- **FAISS Flat**：本质是全量比较（无索引结构），只是 numpy 加速版的 SQLite full scan。虽然 Recall 1.0，但十万张下 p50 会线性增长到几百 ms 甚至秒级，且不支持增量 delete。**只作 fallback 参考**。
- **SQLite full scan**：当前 500 向量 25ms 可用，但十万张线性外推到 ~5 秒，超出原计划 §11 性能目标 p50 ≤ 2s。**只作 baseline**。

### 决赛

**FAISS HNSW32** vs **hnswlib**：

| 维度 | FAISS HNSW32 | hnswlib | 判定 |
|---|---|---|---|
| Recall@10 | 0.992-1.000 | 0.998-1.000 | 打平（hnswlib 略优） |
| p50 latency | 0.014-0.016 ms | 0.016-0.018 ms | 打平（微秒级差别可忽略） |
| p95 latency | 0.018-0.024 ms | 0.019-0.023 ms | 打平 |
| 内存占用 | 5-11 KB | 5 KB | hnswlib 略优 |
| build time | 1.4-14.2 ms | 2.2-3.3 ms | 打平 |
| 增量 add | ✅ | ✅ | 打平 |
| **真正 delete** | ❌ 需要 rebuild | ✅ 内置 mark-deleted | **hnswlib 胜** |
| Python 打包 | pip install faiss-cpu（预编译 wheel） | pip install hnswlib（从源编译，~30s） | FAISS 略优 |
| GPU 加速 | 可切 faiss-gpu | 不支持 | FAISS 略优（家庭场景不需要） |
| 社区活跃度 | Facebook Research | nmslib | 打平 |

### 决定性差异

**家庭记忆场景对 delete 的需求真实存在**：
- 用户删除照片 → Asset 删除 → visual 向量必须从 ANN 移除
- 用户合并/拆分实体 → Observation revision 变化 → semantic 向量的旧版本要下架
- 用户 apply Correction → Event 重建 → episodic 向量重建

FAISS HNSW32 遇到 delete 只能：
- 标记为 tombstone（查询时过滤）— 内存不释放，Recall 逐渐降级
- 或触发全量 rebuild — 十万张 build 时间估算 3 秒以上，运维负担重

hnswlib 有原生 `mark_deleted(label)` API：
- 立即从查询结果剔除
- Rebuild 可延迟到维护窗口
- 内存标记而非释放，但对家庭规模（几十万）可接受

## 推荐

### 主推：**hnswlib**

理由：
1. **Recall 打平或略优** FAISS HNSW32
2. **原生支持 delete** — 与家庭记忆动态变化场景匹配
3. 内存最小
4. API 极简（`Index(space='cosine', dim=D)` → `add_items` → `knn_query`）
5. 已在 153 `.venv` 安装成功且构建通过

### 备选：**FAISS HNSW32**

保留价值：
- 若未来扩展到 GPU 加速（家庭规模不需要，但 Sentrix 演进方向可能）
- Facebook Research 长期维护背书
- 若 hnswlib 出现瓶颈可切换

### 排除

- FAISS IVFFlat（Recall 不达标）
- FAISS Flat（无索引结构，本质等于 SQLite full scan）
- 保留 SQLite full scan 作为 Recall ground truth 与 canonical fallback

## 后续实现（Phase 3.5.2 完成之后）

选定 hnswlib 后，`backend/retrieval_ann.py` 加一个 `HnswlibIndex` 实现：

```python
class HnswlibIndex:
    """Real ANN backend once user picks the library."""
    def __init__(self, dim=512, space="cosine", max_elements=200_000, ef_construction=200, M=16):
        import hnswlib
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self._index.set_ef(50)
        self._id_map = {}  # ann_id -> our string id + metadata

    def build(self, vectors): ...
    def add(self, vectors): ...
    def remove(self, ids):
        for our_id in ids:
            ann_id = self._id_map.pop(our_id, None)
            if ann_id is not None:
                self._index.mark_deleted(ann_id)
    def search(self, query, k, scope_id=None): ...
    def save(self, path): self._index.save_index(path)
    def load(self, path): self._index.load_index(path, max_elements=...)
```

`create_index("hnswlib")` 加分支。

## 待决策

请从以下选项中确认一项：

1. **采纳 hnswlib 作为 Phase 3.5.2 生产 ANN 后端**（推荐）
2. **采纳 FAISS HNSW32**（如果偏好 Facebook 生态或未来 GPU 路径）
3. **两者都实现**（AnnIndex 协议兼容，用户按 env flag 切换 — 增加维护面）

## 参数说明

benchmark 使用的是**样本子集**（每空间 500 向量）。规模趋势估算：

| 规模 | SQLite p50 (线性) | hnswlib p50 (对数) | 是否达标 (≤ 2s) |
|---|---|---|---|
| 500 (今天) | 25 ms | 0.016 ms | ✅ |
| 5000 | 250 ms | 0.03 ms | ✅ |
| 50000 | 2500 ms | 0.05 ms | SQLite ❌，hnswlib ✅ |
| **100000 (Phase 3.5.3 目标)** | ~5000 ms | ~0.06 ms | **SQLite ❌，hnswlib ✅** |

十万张压力测试（Phase 3.5.3 的 `replay_hundred_thousand_scale.py`）只有实装 hnswlib 后跑才有意义。

## 结论

**推荐 hnswlib**。请回复选择哪一项，我将在 Phase 3.5.2 实现选定库的 `AnnIndex` 后端，然后 Phase 3.5.3 跑十万张压力回放。
