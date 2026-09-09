# Sentrix 部署对齐（153）· 一键安装 / 启动

本目录脚本用于把仓库拉起为与 153 当前运行一致的一套服务。

## 环境与选型（对齐表，代码为准）

| 项 | 选型/取值 | 由谁决定 |
|---|---|---|
| 图像嵌入 IMAGE | `chinese_clip` → `chinese-clip-ViT-L-14`(768d)，进程内 | `backend/embeddings/scheme.py` |
| 文本嵌入 TEXT | `bge` → `BAAI/bge-m3`(1024d)，@8101 sidecar | `scheme.py` + `scripts/maintenance/text_embedder_sidecar.py` |
| 生成/Agent 模型 | `gemma4-12b-it` @8100（vLLM，`max_model_len=8192`） | vllm manager 8500 + `configs/sentrix_vllm_registry_192_168_0_153.json` |
| Judge | doubao（volcengine ark）远端 | `services/photobench/config/runtime_connection.json`（含 provider） |
| Sentrix API | 8091 · `backend.app`（uvicorn，`.venv`） | `scripts/deploy/start_all.sh`（env 对齐 start_sentrix_api_8091.sh） |
| orchestrator | 8771 · `benchmark_orchestrator.py` | 同上 |
| 存储 | SQLite `data/sentrix.db`（scope/向量/FTS 一体） | 仓库内 |

> 说明：`start_all.sh` 里把 `SENTRIX_TEXT_EMBEDDER=bge`（153 旧脚本写 clip，但查询侧已由
> `scheme.py` 锁定 bge，此处显式 bge 消除歧义）；ANN 索引均按 chinese-clip/bge 重建。

## 步骤
1. 前置（外部，不在本仓库）：
   - vLLM runtime：sentrix-vllm（manager 8500），profile `gemma4-12b-it` → 8100；
   - Judge doubao API key（runtime_connection.json 的 judge_provider 配置）。
2. `bash scripts/deploy/install_env.sh` —— 建 `.venv`（后端）、`.venv-text`（bge sidecar）、构建前端 dist。
3. `bash scripts/deploy/start_all.sh` —— 起 8100(12B) → 8101(bge) → 8091(API) → 8771(orchestrator)。

## 与 153 当前一致性的校验点
- 8091 监听、`/api/assets?scope_id=…` 返回资产（含 keyframe `parent_asset_id`/`derived_kind`）；
- 8101 `/embed` 返回 1024 维 bge-m3；
- 8100 `/v1/models` 含 `gemma4-12b-it`；
- 8771 `/api/runs/…/memory-effectiveness` 返回 `hit_match_source=使用 scope 资产清单…`。
