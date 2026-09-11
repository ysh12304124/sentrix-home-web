# Sentrix 部署对齐（153）· 一键安装 / 启动

本目录脚本用于把仓库拉起为与 153 当前运行一致的一套服务。

## 环境与选型（对齐表，代码为准）

| 项 | 选型/取值 | 由谁决定 |
|---|---|---|
| 图像嵌入 IMAGE | `chinese_clip` → `chinese-clip-ViT-L-14`(768d)，进程内；需 `cn-clip` 包 + 权重 | `backend/embeddings/scheme.py` |
| 文本嵌入 TEXT | `bge` → `BAAI/bge-m3`(1024d)，@8101 sidecar | `scheme.py` + `scripts/maintenance/text_embedder_sidecar.py` |
| 生成/Agent 模型 | `gemma4-12b-it` @8100（vLLM，`max_model_len=8192`） | vllm manager 8500 + `configs/sentrix_vllm_registry_192_168_0_153.json` |
| Judge | doubao（volcengine ark）远端 | `services/photobench/config/runtime_connection.json`（含 provider） |
| Sentrix API | 8091 · `backend.app`（uvicorn，`.venv`） | `scripts/deploy/start_all.sh`（env 对齐 start_sentrix_api_8091.sh） |
| orchestrator | 8771 · `benchmark_orchestrator.py` | 同上 |
| 存储 | SQLite `data/sentrix.db`（scope/向量/FTS 一体） | 仓库内 |

> 说明：`start_all.sh` 里把 `SENTRIX_TEXT_EMBEDDER=bge`（153 旧脚本写 clip，但查询侧已由
> `scheme.py` 锁定 bge，此处显式 bge 消除歧义）；ANN 索引均按 chinese-clip/bge 重建。

## 图像向量一致性（重要，勿跳过）

写入侧与查询侧的 IMAGE embedder 必须是同一个模型，否则检索会**静默退化**：

- 写入侧（图片导入）历史用通用 CLIP `ViT-B-32`，查询侧锁定 `chinese-clip-ViT-L-14`；
- `visual_ann` 检索前要求库中存在"查询模型"的向量，缺了就跳过该通道 —— **不报错**，
  只是图片语义召回全空（文本/词法通道还在，所以表面上"还能用"）。

因此 `start_all.sh` 在 8091 启动前（Qdrant 目录锁释放的窗口）做两件事：

1. **preflight**：确认 `.venv` 有 `cn_clip`、且 chinese-clip 权重存在；缺失直接报错退出，
   绝不静默降级；
2. **一致性检查**：只读扫描各 scope 的"图片数 vs chinese-clip 向量数"，有缺失就告警并给出补齐命令。

补齐命令（只补缺失的，已补齐的零成本跳过）：

```bash
.venv/bin/python scripts/maintenance/ensure_visual_vectors.py --apply --scope <scope_id,...>
```

也可以在启动时用 `SENTRIX_ENSURE_SCOPES=<scope_id,...>` 让脚本自动补这几个 scope。
**不指定 scope 时脚本只检测告警、不自动全量重嵌**——全量在 CPU 上可能数千张/数小时，
不能堵在启动路径上。新导入的相册跑 `backend/scope_finalize` 收尾时即会补上该 scope 的向量。

## 步骤
1. 前置（外部，不在本仓库）：
   - vLLM runtime：sentrix-vllm（manager 8500），profile `gemma4-12b-it` → 8100；
   - Judge doubao API key（runtime_connection.json 的 judge_provider 配置）；
   - chinese-clip 权重 `~/.cache/clip/clip_cn_vit-l-14.pt`（约 1.6GB，install_env.sh 会检查并给出下载地址）。
2. `bash scripts/deploy/install_env.sh` —— 建 `.venv`（后端 + `cn-clip`）、`.venv-text`（bge sidecar）、构建前端 dist。
3. `bash scripts/deploy/start_all.sh` —— 起 8100(12B) → 8101(bge) → [preflight + 向量一致性] → 8091(API) → 8771(orchestrator) → health/检索探测。

> **8091 不在此脚本里另写 env**：它直接调用生产启动脚本 `scripts/runtime/start_sentrix_api_8091.sh`。
> 实测教训——另抄一份 env 时漏掉 `SENTRIX_VECTOR_BACKEND`/`SENTRIX_QDRANT_PATH`，向量层会
> 静默退化成 SQLite 全表扫（检索"没报错但明显变差"）；漏掉 `HF_HUB_OFFLINE=1`，bge sidecar
> 会反复连 huggingface 超时起不来。缺失该生产脚本时脚本直接报错退出，拒绝降级启动。

可覆盖变量：`SENTRIX_PORT` / `ORCH_PORT` / `TEXT_EMBED_PORT` / `SENTRIX_ENSURE_SCOPES` / `CHINESE_CLIP_CHECKPOINT`。

## 与 153 当前一致性的校验点
- 8091 监听、`/api/assets?scope_id=…` 返回资产（含 keyframe `parent_asset_id`/`derived_kind`）；
- 8101 `/embed` 返回 1024 维 bge-m3；
- 8100 `/v1/models` 含 `gemma4-12b-it`；
- 8771 `/api/runs/…/memory-effectiveness` 返回 `hit_match_source=使用 scope 资产清单…`；
- 向量一致性：`ensure_visual_vectors.py`（只读）输出 `missing_scope_count: 0`。
