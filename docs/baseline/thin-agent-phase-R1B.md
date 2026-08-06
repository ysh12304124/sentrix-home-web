# Phase R1B · Visual/Text Embedding 独立能力评估

**日期**：2026-08-06
**状态**：评估器 + 测试 + 输入构建器完成（本地 smoke 通过）；153 真实模型验收待跑

## P0-2 关键拆分

不共用"一个 ClipAdapter 证明一切"：
- **Visual Cross-modal**：query text → text embed → image embed 余弦 → 正确图片 rank。独立于文本检索。
- **Text Retrieval**：query → caption/activity/place/object/clothing/ocr/event_summary 字段 → 目标记录 rank。按字段分。

## 交付物

| 文件 | 作用 |
|---|---|
| `scripts/benchmarks/evaluate_embedding_quality.py` | `visual_crossmodal(images, queries, embedder)` + `text_retrieval(corpus, queries, embedder)` 两个独立评估器；Recall@1/5/10、MRR、正负样本间隔 AUC；`--embedder clip/stub`；CLI 读 JSON |
| `scripts/benchmarks/build_embedding_eval_input.py` | 在 153 从真实 DB 生成 images-json（真实图片路径）与 corpus-json（observation 字段 + 可选 event summary） |
| `backend/tests/test_embedding_quality.py` | 用确定性 stub embedder（bigram hash）验证评估逻辑：重叠文本 top-1、字段召回、无关文本不压过目标、AUC 形状 |

## 本地验证

```
unittest: backend.tests.test_embedding_quality → 5 ok
```

## 153 验收配方（待跑）

```bash
# 1. 从真实 DB 构建候选
PYTHONPATH=. .venv/bin/python scripts/benchmarks/build_embedding_eval_input.py \
  --db data/sentrix.db --images-json /tmp/dev_images.json --corpus-json /tmp/dev_corpus.json --event-text

# 2. 作者标注 Development labels（查询文本 → 目标）
#    visual 目标 = 图片 id；text 目标 = observation id。标签源 = 图片 canonical caption 的自匹配 + 少量改写。
#    **不得使用 benchmark query_cn。**

# 3. 跑真实 CLIP
PYTHONPATH=. .venv/bin/python scripts/benchmarks/evaluate_embedding_quality.py \
  --images-json /tmp/dev_images.json --queries-json /tmp/dev_visual_labels.json \
  --corpus-json /tmp/dev_corpus.json --queries-json /tmp/dev_text_labels.json \
  --embedder clip --report docs/baseline/embedding_quality_report.json
```

## 判定规则（D3 / D8）

| 评估器 | 不合格阈值 | 动作 |
|---|---|---|
| Visual cross-modal | AUC < 0.7 或 Recall@10 < 0.6 | 提交 Chinese-CLIP（`OFA-Sys/chinese-clip-vit-base-patch16`）选型报告，用户点头后换 adapter + 索引重建 |
| Text retrieval | AUC < 0.7 或 Recall@10 < 0.6 | 提交 bge-m3 选型报告，同上 |

两者独立选型，独立 adapter，不互相绑架。**在 153 报告出之前不假设当前 ViT-B-32 中文可用。**
