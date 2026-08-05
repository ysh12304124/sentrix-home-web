# Phase R2 · 接通多路检索到生产 Kernel

**日期**：2026-08-06
**状态**：代码 + 单测完成（382 tests 全绿，含新增 41）；153 真实数据验收待跑

## 交付物

### 新增 `backend/embeddings/`
| 文件 | 作用 |
|---|---|
| `base.py` | `VisualQueryEmbedder` / `TextQueryEmbedder` Protocol（P0-3） |
| `clip_visual.py` | CLIP 视觉 query embedder（对齐 Asset image vectors） |
| `clip_text.py` | CLIP 文本 query embedder（对齐 Observation/Event text vectors） |
| `bge_text.py` | bge-m3 备选（D3，默认不激活，import gate 保护） |
| `router.py` | `EmbeddingRouter`：双槽独立 + env 选型（`SENTRIX_IMAGE_EMBEDDER`/`SENTRIX_TEXT_EMBEDDER`） |

### 新增 `backend/retrieval/`
| 文件 | 作用 |
|---|---|
| `base.py` | `CandidateHit`（P0-19 score direction）/ `HardFilterContext` / `RetrievalQuery` / `Retriever` Protocol |
| `metadata.py` | 结构化召回：时间/媒体/scope；**无正向结构化条件时返回空**（避免纯语义查询被全 scope 污染） |
| `entity.py` | confirmed entity → person_bridge → 候选 asset |
| `lexical.py` | FTS 预分词召回（P0-1）；FTS 空时惰性自愈重建 |
| `visual_ann.py` | text→visual ANN→Asset；Manifest 校验（P0-4）+ 搜索时维度交叉校验 |
| `text_ann.py` | text→semantic/episodic ANN→Observation→Asset |
| `adjacency.py` | 接口 + `expand()`（R3B 实装） |
| `fusion.py` | RRF(k=60) + evidence_class 分级（P1-1：anchor boost / semantic rank fusion / expander 继承） |
| `config.py` | `RetrievalConfig` 双层配置（P1-4） |
| `probes.py` | `NeutralProbe` 接口（R4 实装） |
| `__init__.py` | `build_default_retrievers` 工厂（6 通道逐个 flag 开关） |

### 修改
| 文件 | 改动 |
|---|---|
| `evidence_retrieval.py` | `EvidencePacket.channel_trace` 字段；`retrieve()` 按 flag 分派 `_retrieve_multi`（prefilter→recall→merge→condition→postfilter→fusion）或旧 `_retrieve_single` |
| `retrieval_indexes.py` | `pre_tokenize()`（整词+latin 词+CJK bigram，单 CJK 字不 token）；FTS5 虚表 `observation_search_fts`；`refresh_from_observation` 双写；`search_fts`（逐 token MATCH，exact whole boost）；`rebuild_all` 重建 FTS |
| `retrieval_ann.py` | `HnswlibIndex` 加 Manifest（model_id/checkpoint_hash/dimension/normalized/id_map_checksum/space/source_count…）+ `validate()` + atomic save/load |
| `thin_agent.py` | 构造收 `embedding_router`/`retrieval_config`；多路 flag on 时构建 retrievers；trace 加 `channels` 段 |
| `agent.py` | `MemoryAgent.__init__` 从 clip 自动建 `EmbeddingRouter` |
| `app.py` | 无需改（MemoryAgent 自建 router） |

### 测试（新增 6 文件 + 1 集成）
`test_retriever_contracts` / `test_lexical_retriever` / `test_ann_manifest` / `test_visual_ann_retriever` / `test_text_ann_retriever` / `test_embedding_router` / `test_multi_retriever_kernel`

## 本地验证

```
unittest discover backend.tests → 382 OK (skipped=1)
evaluate_retrieval_kernel --channels lexical → exit 0
thin_runtime 多路 smoke → lexical 命中 asset_demo_1, channel_trace {metadata:0, entity:0, lexical:1}
```

## 关键行为变化

1. **Kernel 只评估被召回的候选**：`_retrieve_multi` 里只有至少一个 retriever 命中的 asset 才会进 condition pass（旧路径全 scope 逐对）。
2. **单 CJK 字不再是 token**：`pre_tokenize("色")` → `[]`，杜绝单字 all-match。
3. **ANN 有 Manifest 校验**：query embedder 与索引 model/dim 不一致 → 通道 `index_incompatible` 跳过，trace 记录原因。
4. **生产 ANN 故障不降级全表扫描**（P0-10）：`visual_ann`/`text_ann` 失败返回空 + 状态，不触发 `store.search_vectors`。

## 153 真实数据验收（待跑）

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmarks/evaluate_retrieval_kernel.py \
  --db data/sentrix.db --spec-source cached --exclude-hidden docs/baseline/hidden_set_manifest.json \
  --channels full_hybrid --report docs/baseline/retrieval_R2_full_hybrid.json
# 消融各通道
for c in lexical visual text structured hybrid_no_adjacency; do
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/evaluate_retrieval_kernel.py \
    --db data/sentrix.db --channels $c --report docs/baseline/retrieval_R2_$c.json
done
```

验收目标：trace 每个启用通道 `invoked=true`（或明确不可用原因）；Recall/MRR 比 R1A 明确提升、FP 相对下降。
