# Phase B B1 — ANN 恢复与检索 Readiness 报告

- 日期：2026-08-10
- 状态：✅ 完成（视觉/文本/语义/情节索引全部恢复并接入 runtime）

## 1. 根因

`data/ann/` 为空与 `data/models/clip/ViT-B-32.bin` 缺失同源：**数据目录丢失**（`data/` 已被 `.gitignore` 排除，不随代码恢复）。`memory_vectors` 表（2528 行）留存，是可重建的素材；CLIP checkpoint 在 `/home/asus/Github/stmem-bak/models/` 有备份。

## 2. 重建结果（`rebuild_ann_indices.py`，原子 swap + manifest）

| 空间 | 向量数 | 维度 | 模型 | 说明 |
|---|---:|---:|---|---|
| visual | 747（全量图片资产） | 768 | Chinese-CLIP ViT-L/14 | 用 `--visual-embedder chinese_clip` 从图片文件重嵌入；**替代被 R1B 判定中文近似随机（AUC~0.51）的旧 ViT-B-32 视觉空间** |
| semantic | 581 | 512 | ViT-B-32 | 存量 observation 向量 |
| episodic | 944 | 512 | ViT-B-32 | 存量 observation/event 向量 |

- 每个空间生成 `.hnsw` + `.meta.json` + `.manifest.json`（model_id/dim/count/checksum/built_at）
- 视觉索引此前只有 587 个资产向量（部分覆盖）→ 重嵌入后 747 全量
- 文本通道恢复：从 `stmem-bak` 恢复 `data/models/clip/ViT-B-32.bin`，`text_ann` status=ready（"爬山"命中 5 条，cos≈0.85+）

## 3. Runtime 接线修复（关键）

- 发现 `app.py _tool_loop_turn` 的 `bind_runtime(store, gamma=gamma)` **未传 embedding_router** → tool-loop 的 `search_memories` 一直走 `EvidenceRetrievalKernel(store)` 单通道线性扫描，**从未使用 ANN**
- 修复：`app.py` 与 `evaluate_tool_loop_shadow.py` 均构建 `EmbeddingRouter.from_clip(ClipAdapter())` + `RetrievalConfig()` 传入
- 验证（scope=`album2_e2b`）：爬山 16 候选 / 红色衣服 10 / 猫咪 4（此前为空→模型编造）

## 4. Retrieval Regression 重跑（R8 44 例集，38 例含 GT）

| 指标 | 记录（R8 visual_backbone） | 重跑（B1） | Δ |
|---|---:|---:|---:|
| Recall@10 | 0.891 | 0.836 | -0.055 |
| Recall@20 | 0.926 | 0.859 | -0.067 |
| MRR | 0.711 | 0.764 | **+0.053** |
| Precision@5 | 0.377 | 0.416 | **+0.039** |
| all_relevant | 33 | 31 | -2 |

- 6 例无 GT（记录即空，不含在指标）；差距集中在 3 个 place 类查询（album1-05/album3-02/album3-03，时间/地点条件）
- 1 例（album2-01）为**记录集 GT 与当前 DB scope 资产映射不一致**（GT 属于 album2_e2b），非真实回归
- 结论：**ANN 已恢复且排序质量（MRR/P@5）优于记录；Recall@10/20 略低 ~6pt，归因于 place 类查询与索引构建细节，无系统性回退**

## 5. 通道 Readiness（写进 B4 正式矩阵）

| 通道 | 状态 | 依据 |
|---|---|---|
| search_memories.visual | **ready** | Chinese-CLIP 全量索引 + 回归数字 |
| search_memories.text | **limited** | 通道可用，但英文 CLIP 中文语义弱（R8-4 记录 recall@10=0.216 不合格，低权重 append-only） |
| semantic/episodic ANN | ready | 索引存在 + 测试 23/23 |
| ANN health 读取 | ready | `retriever.status`（ready/incompatible/embedder_unavailable）+ manifest 校验 |

## 6. 验证与产物

- `test_ann_manifest` / `test_visual_ann_retriever` / `test_text_ann_retriever` / `test_ann_index`：23/23 通过（153 venv）
- `scripts/benchmarks/regression_retrieval_r8.py`：可复现回归 runner（新增）
- `data/ann/*.manifest.json`：索引健康证据
- 8097 启动脚本补充：`SENTRIX_IMAGE_EMBEDDER=chinese_clip CLIP_DEVICE=cuda SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1 SENTRIX_RETRIEVER_RANKING=visual_backbone`
