# R9-4 · bge-m3 Text ANN Shadow Sidecar — 报告

**日期**：2026-08-06
**性质**：bge-m3 改为 **Sidecar** 隔离（主 API 不 import torch/sentence-transformers），独立 text ANN 空间 + shadow 对照基础设施。
**测试**：本地 489 通过（R9-3 483 + 新增 6）。

## 1. 交付

| 文件 | 改动 |
|---|---|
| `scripts/maintenance/text_embedder_sidecar.py`（新） | 独立 `.venv-text` 内运行的 HTTP 服务：`GET /health`、`POST /embed`（1024-d）；CPU/GPU 切换（`SENTRIX_TEXT_EMBEDDER_DEVICE`）；线程锁；stdlib 实现 |
| `backend/embeddings/bge_text.py` | **改为 SidecarClient**：主进程仅 httpx；`available`=health 探活；**熔断**（连续 3 次失败→30s 冷却返回 []）；`dimension=1024` |
| `requirements-text.txt`（新） | sentence-transformers/transformers/torch/numpy（`PYTHONNOUSERSITE=1` 隔离） |
| `scripts/maintenance/build_text_ann_space.py`（新） | 镜像 pipeline 文本组装（caption/activity/place/ocr/transcript/clothing/facts）→ sidecar 嵌入 → 独立 `data/ann/text_bge_{semantic,episodic}.hnsw` + manifest（model_id=BAAI/bge-m3, dim=1024）；不碰运行时 memory_vectors |
| `scripts/maintenance/probe_text_embedder.py`（新） | 探活 + 嵌入回程，非零退出供编排告警 |
| `scripts/benchmarks/evaluate_text_paraphrase.py` | `--embedder bge` 现走 SidecarClient（原有对照逻辑复用） |
| `backend/tests/test_text_ann_shadow.py`（新） | sidecar health/嵌入/熔断/重置 + **bge shadow 不位移 visual top-K**（visual_backbone 尾部追加）+ CLIP 缺失不崩 |

## 2. 隔离边界

```text
主 API venv（无 torch/sentence-transformers）
  backend/embeddings/bge_text.py = HTTP SidecarClient
        │  SENTRIX_TEXT_EMBEDDER_URL=http://127.0.0.1:8101
        ▼
.venv-text（requirements-text.txt, PYTHONNOUSERSITE=1）
  text_embedder_sidecar.py  (BAAI/bge-m3, device=cpu|cuda)
```

- 依赖真正隔离；sidecar 崩溃→熔断→`text_available=False`（不拖垮主 API）；可独立切 CPU/GPU；独立健康检查。
- 主进程对 bge 无 torch 依赖；`TextAnnRetriever` 按 `embedding_router.text.model_id` 校验空间 manifest。

## 3. 接入门槛（§8.3，待 153 实测）

- [ ] `.venv-text` 建立 + sidecar 起服（`probe_text_embedder.py` 探活）
- [ ] `build_text_ann_space.py` 构建 bge text 空间
- [ ] `evaluate_text_paraphrase.py --embedder bge`：Recall@10 明显 >0.216（CLIP 基线）
- [ ] Dev/Hidden 新增 GT / 新增 FP；strict-empty FP 不增；不位移 visual top 候选
- [ ] 达标 → `SENTRIX_TEXT_BGE_ACTIVE` 低权重补召回；不达标 → 关闭 Text ANN

## 4. 下一步

R9-5：Hidden Acceptance（盲跑 → `hidden_predictions.json` → 用户持 GT 离线评分）。
