# Sentrix Home 项目记忆

## 文档维护规则

本文档是 Sentrix 唯一的当前状态交接文档。它记录产品定义、架构、数据契约、
模块实现、运行边界、实测状态、已完成工作与待办事项，供新接手者快速恢复上下文。

- 本文档及后续维护一律使用中文；代码标识、路径、命令、端口、模型名、API、
  数据库表名和精确指标保持原样。
- 架构、数据契约、模型边界、部署、验收或重要缺陷变化时，必须在同一变更中更新
  本文档。只记录已核验事实；设计、推测和未运行基准不能写成已完成。
- 不记录密码、令牌、`.env` 内容、原始家庭身份数据、临时提示词或未经验证指标。
  数据库备份、模型缓存、日志和 `.ollama-sentrix/` 属于运行状态，不进入 Git。
- `README.md` 只保存运行入口与本文档链接；设计和实施计划保存于
  `docs/superpowers/`，阶段验收报告与实测 JSON 保存于 `docs/baseline/`，历史实现
  由 Git 历史保存，不再维护第二份当前架构文档。

## 产品定义

Sentrix 是本地优先的家庭记忆系统。它将原始图片、音频、文本和视频
转化为可回溯原始证据的事件记忆、人物语义记忆和视觉记忆。Cognee 仅提供
`remember`、`cognify`、`recall`、`improve` 的设计参考，不是运行依赖。

硬约束：

1. 所有事件、人物事实、回答和反馈均可沿证据链回溯到 `Observation` 与原始
   `Asset`。
2. 导入源只允许提供原图、拍摄时间、拍摄地点和相册来源。事件名、活动、人名、
   家庭角色、关系、`event_id`、`activity_hint`、人脸标注和查询标准答案不得写入
   家庭记忆。
3. 人脸聚类只表示候选身份。用户确认姓名之前，不得把它写成命名人物、关系或
   人物语义事实。
4. 相册归属仅是来源证据，不是画面人物、家庭角色、关系或事件分组信号。
5. 图片查询必须返回可打开的原始 Asset，不得只输出文件名或用生成缩略图替代证据。
6. 家庭记忆助手以自然对话和长期家庭记忆为中心：先判断普通聊天、记忆查询、
   反馈/纠正或澄清，再按需调取分层记忆；只有输出家庭具体事实时才附带来源评分
   与可展开证据。无证据时输出 `gap`/`no_result`，不编造。

## 当前视频部署（2026-08-13 实测）

- 200 仓库：`/home/sscy/GitHub/hpq/sentrix-home-web-worldmm`
- Web：`http://192.168.0.200:4174`，代理本部署 API `http://127.0.0.1:8091`
- 原始数据、派生关键帧、SQLite、WorldMM 输出均在该独立工作副本的 `data/` 下。
- VLM：进程内惰性加载本机 Qwen3-VL-4B，提供 12 GiB GPU 显存使模型完整驻留；
  WorldMM 使用 GPU 0。InsightFace、CLIP 保持 CPU，避免影响该机其他现有服务。
- 真实 `IMG_3957.MOV` 完整 GPU 全流程耗时 27.811 秒（CPU 卸载模式 118.796 秒，
  原全 CPU 运行 390.356 秒）。

## 历史基线（2026-08-10 实测）

- 153 仓库：`/home/asus/Github/Sentrix-Home-Web`
- 正式后端提交分支：`psh`
- 当前提交：`15039f0` (`fix(api): scope dedup in /api/ingest (same album only, not cross-space)`，2026-08-10)
- 当前工作树：干净。
- Web：`http://192.168.0.153:4174`，代理 `http://127.0.0.1:8091`
- 生产 Agent API：`8091`（AgentRuntime Tool-Loop 全栈 + `data/sentrix.db`）；`8090` 为旧实例保留
- RX 验证实例：`127.0.0.1:8092`（`SENTRIX_RX_V1=1` + `SENTRIX_12B_FULL_CHAIN_VALIDATION=1` + `AGENT_MODEL_PROFILE=quality_12b`）
- Ollama RX 实例：`127.0.0.1:8096`（`SENTRIX_RX_V1=1` + `AGENT_MODEL_PROFILE=quality_12b`）
- 本地开发栈（`/home/asus/Github/ysh/sentrix-home-web`）：`11000` Web / `11001` API
- 主模型：vLLM `gemma4-12b-it`，`127.0.0.1:8100`（生产默认）
- E2B LoRA 服务：`127.0.0.1:8101`（可选，默认不启用）
- FMA Web：端口 `5173`，与 Sentrix 无关，禁止停止、修改或重启；本次检查未监听
- 当前数据库：`/home/asus/Github/Sentrix-Home-Web/data/sentrix.db`
- 2026-08-13 已实现视频链路：真实 ffprobe 元数据、仓库内置 WorldMM-a 关键帧/DMD/Scene、
  派生图片 Asset、按 Scene 强制绑定的 Event、逐帧绝对拍摄时间、GPS 继承、Timeline
  图片堆和关键帧回跳原视频。部署在 `192.168.0.200`，未操作旧 153 服务。

## 模型与运行隔离

生产推理使用 vLLM，模型 profile 由 vLLM registry 单例管理；旧 Ollama 专用监听
`11435` 当前未运行，环境变量保留为兼容。

- 主 VLM/LLM：vLLM `gemma4-12b-it`；模型 `/home/asus/models/gemma-4-12B-it`；
  `dtype=bf16` + bitsandbytes 4-bit；`max_model_len=4096`；`max_num_seqs=4`；
  `gpu_memory_utilization=0.68`；OpenAI 兼容端点 `http://127.0.0.1:8100/v1`。
- vLLM registry：`configs/sentrix_vllm_registry_192_168_0_153.json`。profiles 包括
  `gemma4-12b-it`、`gemma4-e2b-it`、`gemma4-e2b-it-lora-v2`、`qwen3-instruct` 等；
  切换管理器 `/home/asus/sentrix-vllm/bin/sentrix_vllm_manager.py`，`--wait-ready`
  等待端点就绪后才激活；`/api/model-profiles/switch` 为唯一切换入口（单例锁）。
- 图像向量：Chinese-CLIP（`SENTRIX_IMAGE_EMBEDDER=chinese_clip`）。
- 文本向量：CLIP `ViT-B-32`（`SENTRIX_TEXT_EMBEDDER=clip`）。
- 语音：FunASR，`paraformer-zh`、`fsmn-vad`、`ct-punc`。
- 人脸检测及关键点：InsightFace `buffalo_l`。
- 人脸身份向量：AdaFace `ir_50`；权重路径：
  `/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt`。
- 逆地理：PyGeoCN（天地图行政边界数据）离线返回省/市/区，不联网。
- 可选 E2B LoRA：`services/e2b_server`，默认 `127.0.0.1:8101`
  （`gemma-4-E2B-it` + `gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47`）。
- 共享 Ollama `127.0.0.1:11434` 属于其他项目，禁止修改或停止；当前运行中且
  `/api/ps` 无模型。
- 专用 Ollama `127.0.0.1:11435`：启动脚本仍保留 `OLLAMA_BASE_URL` /
  `OLLAMA_MODEL=gemma4:12b` / `OLLAMA_KEEP_ALIVE=-1` 环境变量，但当前未运行。
- 硬件：RTX 3090 24GB（12B-FC 修复后 Driver 595.84 / CUDA 13.2）。
- `nvidia-smi` 受 NVML 用户态/内核版本不一致影响不可用；以 Ollama/vLLM 运行日志与
  `/api/model-profiles` 状态作为 GPU/模型验证来源。

## 仓库结构

```text
backend/             FastAPI + Thin Agent 运行时 + 记忆库
  app.py             路由与编排
  thin_agent.py      回答入口 answer_turn
  router.py          确定性最终路由（8 步决策树 + NeutralProbe）
  query_parser.py    12B 开放词表解析（sanitize/repair/回退）
  query_contracts.py QuerySpec 契约（HARD/SEMANTIC/PREFERENCE）
  model_routing.py   角色路由、RequestDeadline、circuit breaker
  evidence_retrieval.py  六路检索内核
  retrieval/         通道与融合（metadata/entity/FTS5/visual/text/adjacency、加权 RRF）
  retrieval_ann.py    HNSW ANN 索引
  retrieval_indexes.py 派生检索投影重建
  embeddings/        视觉/文本 embedder 与路由
  answer_brief.py    检索 -> 受控 AnswerBrief 边界
  answer_composer.py 最终回答组装
  response_plan.py / response_validator.py / response_writer.py
  structured_memory.py  确定性结构化回答执行器（TFPE）
  task_contracts.py  只读工具计划与白名单
  core_memory.py     核心记忆卡片（agent_core_memory_*）
  memory_corrections.py 纠正提议/授权修订（agent_memory_correction_*）
  graph_memory.py        MAGMA 视频记忆图派生索引（不替代主检索）
  advanced_memory_tools.py 进阶记忆工具（EvidencePacket）
  claim_extractor.py / complex_answer.py / narrative_context.py
  visible_evidence.py / memory_gate.py / agent_annotations.py
  geocoding.py       PyGeoCN 离线逆地理
  model_clients.py   vLLM/CLIP/AdaFace/FunASR 适配器
  db.py / pipeline.py / face_embeddings.py / face_clustering.py / person_appearance.py
  validation/        full_chain_profile、model_call_ledger
services/e2b_server/ E2B LoRA 模型服务器（Ollama 兼容，8101）
configs/sentrix_vllm_registry_192_168_0_153.json
scripts/runtime/     API/Web/Ollama/E2B 启动脚本（8090/8091/8092/8096 等实例）
scripts/benchmarks/  只读评估（parser slots、hidden、latency、E2E、12B-FC、结构化 QA）
scripts/maintenance/ 重建、索引投影、GPS 回填等运维脚本
docs/baseline/       阶段验收报告与实测 JSON（Thin Agent、R7-R9、12B-FC、RX、benchmark）
docs/plans/          长期计划（semantic-entity-roadmap、digital-memory-steward）
docs/superpowers/    设计与实施计划
```

本仓库另有 `scripts/maintenance/rebuild_graph_memory.py` 和
`docs/MAGMA_GRAPH_MEMORY_INTEGRATION.md`，用于显式重建 MAGMA 视频记忆图。
图数据只写入 `graph_memory_nodes`、`graph_memory_edges`、`graph_memory_builds` 三张
可删除派生表；`assets`、`observations`、`events`、`entities`、`relationships`、
`memory_vectors` 等 canonical 表保持不变。

## 数据模型与证据链

```text
MemorySpace -> Asset -> Observation -> FaceInstance / MemoryVector
                                  -> Event / EventParticipant
                                  -> SemanticProfile / SemanticClaim / Relationship
                                  -> Agent recall（证据校验后回答）
```

人物外观声明要求更强证据链：

```text
Asset -> Observation -> FaceInstance -> PersonAppearanceEvidence -> SemanticClaim
                             \-> 已确认 Person -> Event
```

| 对象 | 责任 | 下游用途 |
| --- | --- | --- |
| `memory_spaces` / `ingest_batches` | 空间与导入批次 | 隔离全部读写与检索；批次完成后再总结事件 |
| `assets` | 原始文件、SHA-256、时间地点、`captured_location`、`batch_id` | 文件访问、处理、证据 UI |
| `observations` | 模型观察、原始 JSON、对象、场景衣物、OCR、转写 | 事件、向量、原始证据 |
| `face_instances` / `face_clusters` / `face_prototypes` | 人脸框、质量、姿态、AdaFace 向量、多原型候选 | 人工确认、合并、拆分 |
| `entities` / `entity_mentions` / `entity_observations` | 人物与非人物实体及其提及 | 事件参与者、画像、检索 |
| `entity_properties` / `entity_merge_candidates` | 实体属性与语义合并候选 | 实体治理（非人物语义组为派生投影） |
| `events` / `event_revisions` / `event_observations` / `event_participants` | 事件边界、总结、修订和证据成员 | 时间线、语义、检索 |
| `trips` / `trip_revisions` | 行程候选（跨日、地点序列、同行者） | 行程确认/驳回 |
| `persons` | 已确认人物（多名称、别名、merged-into） | 人物页、改名、合并 |
| `person_event_memory` / `person_patterns` | 人物事件投影与跨事件模式 | 人物时间线和 Agent |
| `semantic_profiles` / `semantic_claims` / `facts` | 画像、版本化声明与事实状态 | 人物页、知识页、Agent |
| `person_appearance_evidence` | 人脸、上半身裁剪、目标衣物和原始证据 | 人物衣物声明 |
| `relationships` | 人工确认或维护的实体关系 | 关系图、Agent |
| `memory_vectors` | `visual`、`episodic`、`semantic` 向量（HNSW ANN） | 相似检索 |
| `query_gaps` / `memory_feedback` | 查询缺口和人工反馈 | 维护闭环 |
| `dialogue_states` | 会话焦点（活动实体/事件/地点/证据/未解决歧义） | Thin Agent 连续对话 |
| `agent_core_memory_cards` / `agent_core_memory_items` / `agent_query_accesses` | Agent 核心记忆卡片（epistemic 类型 + 源修订引用） | 长期记忆与上下文 |
| `agent_memory_correction_proposals` / `_revisions` / `_audit` | 纠正提议、授权修订与审计 | 记忆修正闭环 |
| `rebuild_runs` | 重建范围、状态、统计和错误 | 运维审计 |
| `stories` / `invites` / `runtime_settings` | 故事、邀请与运行设置 | 页面与运维 |

## 端到端管线

1. **资料导入**：`IngestionPipeline.create_asset()` 计算 SHA-256，只保留时间、
   地点、设备、相册归属等白名单来源；缺失时读取 EXIF；在同一 `scope_id` 内去重
   （`15039f0` 起 `/api/ingest` 去重仅限同一相册，禁止跨空间）。
2. **媒体处理**：图片支持 HEIC；先写入 AdaFace、CLIP、元数据和事件的快速证据，
   再以 896px、禁用思考模式、320 token 的核心中文 JSON 完成语义丰富；音频经 FunASR
   后走文本分析；文本走同一文本分析路径；视频上传立即返回并在后台执行 ffprobe、
   WorldMM 关键帧/DMD/Scene，再让派生关键帧进入现有图片理解。批量导入使用
   `batch_id`，批次完成且资产终态后再总结事件一次。
3. **证据规范化**：保存中文规范字段、原始模型 JSON、对象、场景衣物、空间关系、
   OCR、转写、人脸实例和模型版本。
4. **视觉与身份索引**：事件选择前写入 CLIP/Chinese-CLIP 向量；`buffalo_l` 检测、
   对齐后用 AdaFace 写入身份向量。低质量检测保留为证据。
5. **事件构建**：按时间地点召回，综合活动/类型、对象、视觉地点、已确认人物、
   CLIP 相似度与 GPS 距离线性评分；同时间同地点给时空加成；仅当活动和类型均冲突、
   无确认人物/对象桥接且向量可比时，低视觉兼容度触发保守拆分保护。GPS 原始坐标
   只留在 `captured_location`/`geo`，事件标题、地点和总结只用图片语义地点。
6. **事件总结**：导入热路径只建立可检索的事件和 Observation 证据；批次/维护任务
   （`/api/maintenance/summarize-events`）生成中文标题、类型、活动和摘要。导入标签
   绝不参与。
7. **人物确认**：确认姓名和可选角色后，刷新实体提及、参与者、事件摘要、
   `person_event_memory`、`person_patterns`、画像和声明。原始 Observation 的匿名
   模型描述不被改写。人物支持多名称、全局改名、同名人自动合并、拒绝即删除。
8. **人物外观**：每个关联事件最多选择一个高质量已确认人脸，裁剪头部和上半身并
   分析目标人物衣物；场景衣物不会因共现升级为人物事实。
9. **Agent 回答**：生产走 AgentRuntime Tool-Loop（`backend/agent_runtime/`，默认
   `SENTRIX_AGENT_PROFILE=tool_loop`）：模型自主选工具（记忆事实/检索/翻页/复核原图），
   ToolPolicy/BudgetManager 限制循环，FinalGuard + LLM judge 兜底诚实性；回答必须
   可回溯证据，缺口写入 `query_gaps`，反馈写入 `memory_feedback`。旧 Thin Agent 路径
   仅保留给 benchmark/回归测试。
10. **视频记忆图**：`backend/graph_memory.py` 从已有视频 asset、video scene event、
    关键帧 observation 和 entity 事实构建 `EPISODE/SESSION/EVENT/ENTITY` 派生图；
    通过 `BELONGS_TO_SESSION`、`PART_OF`、单向 `PRECEDES`、`REFERS_TO`、
    `RELATED_TO` 保留证据链。当前不把相邻或时间顺序推断为因果，`causal_edges=0`。

## 模块实现

| 模块 | 实现方式 | 权威输出 |
| --- | --- | --- |
| `backend/app.py` | FastAPI 路由、范围筛选、人物确认、模型 profile 切换、原图与维护 | HTTP API 编排 |
| `backend/db.py` | SQLite 加法迁移、事务、事件评分、空间隔离、聚类与人物投影 | 唯一权威记忆库 |
| `backend/pipeline.py` | 白名单元数据、媒体适配、向量/人脸先写入再归并事件 | Asset、Observation、事件候选 |
| `backend/model_clients.py` | vLLM/CLIP/AdaFace/FunASR 薄适配器 | 带版本结构化输出 |
| `backend/agent_runtime/` | AgentRuntime 薄循环：profile、ToolRegistry、ToolPolicy、BudgetManager、FinalGuard、LLM judge、ResultSet | Agent 回答与轨迹（生产唯一路径） |
| `backend/thin_agent.py` | 旧 `answer_turn` 编排（仅 benchmark/测试使用） | 回归基准 |
| `backend/router.py` | 确定性最终路由 8 步决策树 + NeutralProbe | 最终 route |
| `backend/query_parser.py` / `query_contracts.py` | 12B 开放词表解析、sanitize/repair/回退、QuerySpec | 结构化查询规格 |
| `backend/model_routing.py` | parser/answer/verify 角色、RequestDeadline、circuit breaker | 模型调用边界 |
| `backend/evidence_retrieval.py` / `retrieval/` | 六路检索 + 加权 RRF（visual 2.5）+ 排序 | EvidencePacket |
| `backend/retrieval_ann.py` / `retrieval_indexes.py` | HNSW ANN、派生投影重建 | ANN 索引 |
| `backend/answer_brief.py` / `response_*` | AnswerBrief 边界与计划/校验/写作 | 受控回答内容 |
| `backend/structured_memory.py` | TFPE 确定性 SQL 执行 | 精确结构化答案 |
| `backend/core_memory.py` / `memory_corrections.py` | 核心记忆卡片、纠正提议/授权修订 | 长期记忆与修正 |
| `backend/geocoding.py` | PyGeoCN 离线逆地理 | 省/市/区地点 |
| `backend/face_embeddings.py` / `face_clustering.py` | AdaFace、质量姿态门控、多原型全局重聚类 | 人脸实例、簇、原型 |
| `services/e2b_server/` | E2B LoRA 服务器（Ollama 兼容） | 可选 2B parser/主模型 |
| `configs/sentrix_vllm_registry_192_168_0_153.json` | vLLM profiles + 状态文件 | 模型切换事实源 |

## API 与交互约定

- 规范人物确认入口：`POST /api/face-clusters/{cluster_id}/confirm`；兼容入口
  `POST /api/persons/{person_id}/confirm` 会先将原生 `entity_id` 解析到活跃人脸簇。
- 人物改名：`POST /api/people/{person_id}/rename`（多名称/别名、merged-into 标记）；
  `POST /api/persons/{person_id}/reject` 拒绝即删除。
- 非人物实体：`GET /api/entity-groups`（派生语义组）、`GET /api/entity-merge-candidates`、
  `POST /api/entity-merge-candidates/{candidate_id}/confirm|reject`、
  `PUT /api/entities/{entity_id}/properties/{property_key}`。
- 模型管理：`GET /api/model-profiles`、`GET /api/model-profiles/current`、
  `POST /api/model-profiles/switch`（单例 vLLM 切换，`--wait-ready` 后才激活）；
  `GET /api/vlm-backend` 保留为只读兼容，切换功能已退役（写路径返回 410）。
- 行程：`GET /api/trips`、`POST /api/trips/{trip_id}/confirm|reject`。
- 地理：`GET /api/geo-places`（离线逆地理地点视图）。
- 导入：`POST /api/ingest`（同一相册内去重）、`POST /api/import`、
  `POST /api/import/server-directory`、`POST /api/ingest-batches`、
  `POST /api/ingest-batches/{batch_id}/complete`。
- `POST /api/assistant/turn` 接收 message、`conversation_id`、feedback、`scope_id` 和
  可选 `selected_entity_id`，返回 response_mode、memory_used、evidence、trace；
  `POST /api/search` 是兼容包装器。
- Agent 记忆回答必须返回 `memory_used`、`evidence_required`、`evidence_status` 和
  `evidence_presentation`；有具体事实时状态为 `anchored` 并至少绑定 Event、Observation
  或 Asset 证据，无依据时状态为 `gap` 并返回查询缺口。普通聊天明确标记为
  `memory_used=false`，不读取家庭记忆。
- 证据默认折叠展示但不可省略；用户明确要求原图时
  `original_evidence_requested=true`，前端直接展示可打开的原始媒体。普通用户界面
  隐藏内部 ID / 检索 trace / 原始 JSON；管理员 `?debug=1` 可见。
- 图片结果带 `asset_id`、`observation_id`、文件名、时间、caption、`media_url`，前端
  展示 `/api/assets/{asset_id}/file` 原图和打开入口；头像统一使用
  `/api/face-instances/{face_instance_id}/crop`。
- 反馈目标和 `query_gap` 均按 `scope_id` 校验；未确认目标、跨空间目标或缺少依据时
  不写入事实。
- 前端五秒轮询仅刷新计数；当前页面、媒体 DOM、表单和弹窗输入不得被静默重建。

## 记忆空间与家庭基准

家庭基准源：`/Users/rm001/Downloads/samples`（历史导入路径；当前生产库已含更多
用户相册）。

| 空间 | 历史图片数 | 备注 |
| --- | ---: | --- |
| `album1` | 64 | 62 assets / 25 events（08-10 实测） |
| `album2` | 58 | 50 assets / 34 events（08-10 实测，资产数较历史 58 有变化） |
| `album3` | 69 | 69 assets / 36 events（08-10 实测） |
| `album*_e2b` | - | 端到端验证空间（62/26、66/35、69/35） |
| `album_<hash>` | - | 用户新建独立相册空间 |

- 导入器支持根目录 `metadata.json` 和嵌套 `metadata/metadata.json`，只导入
  `images/` 实际存在的文件。
- `face_info_cn.json`、`face_info_en.json`、`face_id_images`、`query.json` 仅供评估；
  不得自动确认人名、创建关系、生成事件名或写入模型提示。
- 三个基准相册为独立 `MemorySpace`；Asset、Observation、事件、实体、向量、语义和
  Agent 读取均按 `scope_id` 过滤，禁止跨空间事件归并。
- 受控基准曾显式设定 `FACE_IDENTITY_MIN_QUALITY=0.35`；全局默认仍更严格。

## 当前 153 数据库状态（2026-08-10 实测）

`data/sentrix.db`，`PRAGMA integrity_check = ok`；全部 841 assets 状态为
`processed`：

| 记录 | 计数 |
| --- | ---: |
| Asset / Observation | 841 / 841 |
| Event / EventObservation | 595 / 841 |
| FaceInstance / FaceCluster | 543 / 144 |
| Entity / EntityMention | 2959 / 108 |
| SemanticProfile / SemanticClaim | 38 / 742 |
| PersonEventMemory / PersonPattern | 81 / 303 |
| Relationship | 6636 |
| MemoryVector | 2781 |
| QueryGap / MemoryFeedback | 195 / 0 |
| EventParticipant / IngestBatch | 81 / 76 |
| RebuildRun | 4 |

主要空间计数（资产/事件）：`album1=62/25`、`album2=50/34`、`album3=69/36`、
`album1_e2b=62/26`、`album2_e2b=66/35`、`album3_e2b=69/35`，另有多个
`album_<hash>` 用户相册与 `gps-location-validation-20260807`、
`photobench-e2e-20260807` 等验证空间。

说明：`album2` 资产数（50）与 2026-08-05 记录的 58 不一致，属用户侧数据变化，
未做推断。上述计数是运行态事实，不是质量验收结论。

## 已完成工作

- 原始资料、SHA-256 去重、EXIF 时间/GPS/设备回退、相册来源白名单、HEIC 支持、
  文件夹/服务器目录导入与批次导入。
- `MemorySpace` 隔离的相册导入、查询和网页选择；用户可创建独立相册。
- 图片、音频、文本 Observation 管线；视频保留原始 Asset，并生成可回溯原视频秒数的
  派生图片 Asset / Observation；一个 WorldMM Scene 固定映射一个 Event。
- `buffalo_l` 检测、AdaFace 向量、质量/姿态元数据、多原型全局聚类、低质量人脸
  证据化、候选簇确认/拒绝/合并/拆分与审计。
- 人脸裁剪证据接口与统一头像显示；人物多名称、别名、全局改名、同名自动合并、
  拒绝即删除。
- 事件多信号评分（含 GPS 距离线性评分、视觉地点、时空加成）、CLIP 可比性检查、
  保守拆分保护、事件级中文总结和相册内兼容事件合并；GPS 不出现在事件展示文本。
- 人物确认后的参与者刷新、保守重分段、事件摘要刷新和人物语义重建；人物事件投影、
  跨事件模式、版本化声明、画像和事实状态管理；目标人物上半身外观证据。
- Thin Agent 全栈：语义解析 -> 确定性路由 -> 六路检索 -> AnswerBrief -> 证据回答；
  结构化记忆执行器（时间/计数/首末/地点/媒体精确回答）；核心记忆卡片；纠正提议/
  授权修订；HNSW ANN 索引；检索缺口与反馈闭环。
- 12B Evidence Answer 生产默认（vLLM `gemma4-12b-it`）与无降级验证（ModelCallLedger、
  NO_FALLBACK 等）；RX 回答体验（`visible_assets` 唯一决定可见图片、普通用户隐藏
  内部 ID/trace）。
- 离线逆地理（PyGeoCN 省/市/区）与基于地点的照片视图；PyGeoCN 依赖不可用时优雅回退。
- 原图、事件、人物、知识、资产、故事、行程、导入、设置等网页入口；家庭关系图、
  相册管理、模型 profile 切换 UI。
- 独立 vLLM profile registry 单例管理；E2B LoRA 服务器（可选）。
- 家庭基准交集生成器和只读评估器；LFW 主脸隔离基准 coverage `0.9917`、pairwise
  F1 `0.9916`（只评估，不写入正式记忆）。

### 已通过的受控历史验收

- 120 张虚拟家庭相册曾完整重建，外部 manifest 人脸候选评估为 precision/recall/F1
  `1.0000`；148 个检测人脸中 84 个高质量形成 4 簇，64 个低质量保留证据。
- 一次受控确认生成头像/画像、五条事件活动声明、事件参与者和
  `PersonAppearanceEvidence`；活动、衣物和原图 Agent 查询均有证据返回。
- 已验证事件导入不读取 `event_id`、`activity_hint` 等评估标签。
- Thin Agent 生产切换（2026-08-05）：8091 一次性开启 Phase 0-8 V1 flags，冒烟通过
  （`memoryUsed=false`、零家庭读取）。
- RX 双轨验收（2026-08-06）：E2E 14/14、人工盲测新>=旧 91.7%（泄漏 0、文图矛盾 0）、
  本地单测 597 全绿、前端 27 全绿。
- 12B-FC（2026-08-06）：全角色探针 100% 存活、全链 E2E 11/12（唯一失败为“人物介绍”
  完整链缺 claim 角色）、简单证据 p95 8.1s、API<=20s、故障注入证明无静默降级。
- R9（2026-08-06）：153 E2E 10/10、parser slots 12B mode/date/JSON 全 1.0、
  文字规则审计 runtime 语义规则=0；R8 检索基线 Recall@10 0.891 / r20 0.926、
  strict-empty fp=0、hard violation=0。
- 结构化记忆 QA（2026-08-07）：40 例 15 通过（routed 0.4、exact 0.375）为当前基线，
  非最终质量结论。

## 最近验证（2026-08-10）

- `psh` 当前提交为 `15039f0`，工作树干净。
- Node 回归：`node --test test/*.test.js` 31/31 通过（本次同步修正了 3 处陈旧断言：
  `server.js` 默认后端改为项目本地 `11001`、E2B LoRA 标签为“蒸馏后+LoRA”、
  app.py vLLM 状态改为 `_load_vllm_state`/`--wait-ready` 契约）。
- `node --check src/app.js`、`node --check src/api.js`、`node --check server.js`、
  `.venv/bin/python -m compileall -q backend scripts`、`git diff --check` 均通过。
- `/api/health`（8090/8091）返回 200：`mode=sentrix-local-backend`，VLM/LLM 指向
  vLLM `8100` 且 `gemma4-12b-it` running；AdaFace、`buffalo_l`、FunASR、CLIP 报告 ready。
- Web `4174` 返回 200；FMA `5173` 本次未监听（外部服务，未做任何操作）。
- 完整 Python 套件本次未重跑（避免与生产 vLLM `8100` 争用）；最近文档基线：
  RX 阶段本地单测 597 全绿、前端 27 全绿、R9 基线 494 pass / 1 skip（均为 08-06 记录）。

## 当前未完成事项

### P0：MVP 门槛

1. **真实相册人脸验收**：当前真实相册标签缺 bbox/face-instance 级对齐，无法宣称
   F1>=0.95 且覆盖率>=0.95；需要补齐标注或显式人工对齐审阅后再评估。
2. **CLIP 生产验收**：用真实生产 checkpoint 强制执行一次图片和文本 embedding，
   确认不是随机初始化并记录跨模态/相似度结果；向量存在不构成质量验收。
3. **全量端到端性能验收**：以同一三相册 manifest、固定硬件和可审计旧基线运行隔离
   全量管线；实际平均速度比 >=5x 才可关闭性能目标。
4. **12B 全链人物 claim 场景**：完整链（writer->claim->verify）修复并重跑 11/12 中
   唯一失败 case。
5. **结构化 QA 覆盖率**：提升 40 例 15 通过基线；校准 parser action/facet recall
   指标（标注集过度指定导致的偏差）。
6. **Hidden acceptance**：16 条 predictions-only 待用户持 GT 用
   `score_hidden.py` 离线评分。
7. **查询和反馈闭环验收**：用三相册 query 集评估原图命中率、答案事实性和证据链；
   实际执行一次视觉补全后二次查询与一次接受/纠正反馈。

### P1：产品可用性与语义质量

1. 完善人物簇合并/拆分的人工审阅流程与端到端 UI 回归。
2. 制定审核过的中文属性词表后，再做保留证据的同义归一化。
3. 用真实家庭问题集校准 Agent 排序、置信度和查询缺口策略。
4. 扩大来源验证、失败阶段重试、异常恢复和重建可观测性。

### P2：明确延后

- 主动式家庭建议和会话外长期推荐。
- MagFace 对比及是否替换 AdaFace 的生产决策。
- API 与 vLLM/Ollama 的正式托管服务、重启策略、监控和主机级所有权确认。

## 运行与验证命令

```bash
cd /home/asus/Github/Sentrix-Home-Web

# vLLM registry 单例切换（生产默认 gemma4-12b-it）
/home/asus/sentrix-vllm/bin/sentrix_vllm_manager.py switch gemma4-12b-it --wait-ready

# 生产 API（8090/8091 使用同一启动脚本，端口由 SENTRIX_API_PORT 决定）
SENTRIX_API_PORT=8091 scripts/runtime/start_sentrix_api.sh

# Web（4174 代理 8091）
SENTRIX_BACKEND_URL=http://127.0.0.1:8091 PORT=4174 npm run dev

# 可选 E2B LoRA
scripts/runtime/start_sentrix_e2b.sh
```

提交或宣称完成前：

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
node --check server.js
.venv/bin/python -m compileall -q backend scripts
git diff --check
```

模型切换前先确认 `/api/model-profiles/current` 状态；重建会替换派生记忆，执行前
必须确认数据库目标和 `scope_id`：

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py \
  --root . --source /path/to/source-album
```

## 关键历史里程碑

| 日期 | 提交/阶段 | 结果 |
| --- | --- | --- |
| 2026-07-29 | `6ad4ae8` | 首次提交证据驱动家庭记忆项目，建立项目记忆和基础数据契约。 |
| 2026-07-30 | AdaFace、聚类、人物确认、事件评分、独立 Ollama | 完成受控 120 图验证。 |
| 2026-07-31 | `bf286a1` | 引入 MemorySpace、三相册基准、人物事件投影和只读评估工具。 |
| 2026-08-02 | `853ff66` | 对 153 工作树、服务、数据库和未完成 MVP 门槛完成当前状态核验。 |
| 2026-08-03 | `878579b`、`d588c53` | 视觉模型门禁、核心语义快速路径 5.5x、延迟事件总结与并发导入修复。 |
| 2026-08-04 | album1 干净重建 | 62 assets / 25 events；自动导入记忆生成、描述恢复、批次导入修复；三相册语义链路验收 189/189。 |
| 2026-08-05 | Thin Agent 生产切换 | 8091 一次性开启 Phase 0-8 V1 flags；统一 4174->8091->data/sentrix.db。 |
| 2026-08-06 | R9 / 12B-FC / RX | 路由收口、12B 无降级 11/12、RX E2E 14/14、盲测 91.7%。 |
| 2026-08-07 | 人物/地理/检索/vLLM | 多名称与自动合并、PyGeoCN 逆地理、GPS 事件聚类、六路检索 + HNSW ANN、vLLM registry。 |
| 2026-08-08 | vLLM 生产默认 | `gemma4-12b-it` 在 `8100` 常驻；事件时空加成与视觉地点事件。 |
| 2026-08-10 | `15039f0` | `/api/ingest` 同一相册内去重；同步项目记忆与 Node 断言。 |
| 2026-08-13 | WorldMM 视频场景 | `IMG_3957.MOV` 原版算法实测 11.735s / 3 关键帧 / 1 Scene；3 派生 Asset、3 Observation、1 video_scene Event，时间/GPS/原视频秒数链路通过。 |

## 历史阶段记录（2026-08-03 至 2026-08-05）

- 2026-08-03：正式 SQLite 备份 `data/backups/sentrix-before-household-rerun-20260803.db`；
  三相册 `189/189` 均 `processed`，`94/94` 事件持久化原始图片派生封面；8090 由
  `scripts/runtime/start_sentrix_api.sh` 启动并配置 CUDA providers。
- 2026-08-04：album1 干净重建 62 assets / 25 events；修复自动导入记忆生成（首轮空
  `semantic.objects` 时从描述恢复响应投影）、描述恢复物品兼容（`8cd5e6b`）、批次
  导入事件总结（`224c9f7`，`ingest_batches` + 批次完成接口）。
- 2026-08-04：三相册语义链路验收 191 文件全部处理，189 assets/observations，
  95 事件（25/34/36），失败 0、OOM 0；GPS 仅保留在 `geo`，事件文本 GPS 坐标 0；
  向量 `episodic=284`、`semantic=189`、`visual=263`；正式 `.venv` 语义/模型/数据库/
  管道测试 58/58。
- 2026-08-05：统一运行拓扑 `4174 Web -> 8091 Agent API -> data/sentrix.db`；8090 继续
  作为直接 API 入口；FMA 5173 未改动；Node 27/27。GitHub 仓库
  `ysh12304124/sentrix-home-web` 发布快照（`psh-archive-20260805` 保留旧远端 tip）。

## 实体模型待办

当前可回溯实现：`Person`、`Place`、`Object`、`Time`（拍摄日期）、`Event`、`Mood`
（模型情绪字段），另有 `Trip` 候选与派生语义实体组。所有实体必须绑定 Observation，
并在实体详情提供关联事件、原图、置信度和算法证据。

1. **P0 实体目录与人工维护**：人物页展示全部有样本的待命名簇，单张样本标为谨慎
   确认；确认后即时刷新事件、人物摘要与语义声明。语义页按人物、地点、物件、日期、
   情感展示实体，历史 `superseded` 人物声明不作为当前画像展示。
2. **P1 地点与时间质量**：稳定 GPS 地点、`geo` 属性和视觉 `scene_type` 已区分；
   PyGeoCN 离线逆地理已接入；继续补齐历史 GPS 属性回填。
3. **P1 人物档案**：在已确认的人脸证据基础上维护首次/最后出现、高频地点、高频活动、
   同框关系和经用户确认的角色；不从场景描述推断人物外貌或亲密度。
4. **P2 行程 Trip**：仅在跨日事件、地点序列、城市和同行者均有可追溯证据后聚合，
   不用普通 Event 冒充行程。
5. **P2 情感与叙事**：完善 Mood 的置信度、视觉风格与用户纠正链。

完整七类实体实施顺序、数据契约和验收条件见 `docs/plans/semantic-entity-roadmap.md`。
数字人产品定位：中性家庭记忆管家，先调用结构化、语义、事件和原始证据工具，再由
模型组织回答；歧义时澄清，证据不足记录查询缺口，写入必须由用户显式确认并保留
审计。完整协议见 `docs/plans/digital-memory-steward.md`。

## Phase E 待办（2026-08-11）

1. **OCR 显存/时间超预算（P0 待办）**：`read_photo_text` 当前 = 整图 + 3x3 tile（2x 放大）
   共 10 次 12B VLM 图片推理，首图实测 ~150s，GPU 已用 19.2/24GB，端侧不可行。
   候选方案：① 换专用轻量 OCR（PaddleOCR/RapidOCR，CPU/小显存，Numeric Exact Match
   目标 ≥95%）并保留 12B 仅做语义兜底；② 压缩 VLM 用量（mosaic 单次推理 / 2x2 tile /
   scale 1.5）；③ OCR 走 CPU/小核，12B 按需加载。用现有 7 题 spike 集 + final3 4 个
   成功题复测 exact match 后定案。
2. **Answer Style 未达标**：final3 retrieval jargon 泄漏 15 条、direct rate 57.7%
   （目标 0 / 95%）；根因是 final 仍拼接 evidence 摘要文本，需重构 response writer。
3. **inspect_result_set 无结论**：未做 spike、未上线；现由 adaptive budget（多图→4）+
   `read_photo_text` 部分替代，需正式实测结论。
4. **benchmark 产物缺口**：tool sequence（`evaluate_search_inspect_e2e.py`）与
   synthesis faithfulness 无运行输出；capability matrix 未接入 health/Agent prompt；
   dashboard 每题未显示 R/V/O/T/S/G/J 分层标签。
5. **Product wrong 未下降**：final3 wrong 15 > Phase D 基线 11；V 层（招牌/小字）与
   S 层回避是主要贡献。

## Benchmark 人物解析待办（2026-08-21）

1. **P0 benchmark scope 下 Agent 人名→实体解析失效**：`backend/agent_runtime/tools.py` 的 `_resolve_entity`（约 159 行）依赖 `store.list_entities(status="confirmed")`，而 `db.py` 的 `list_entities`（约 3651 行）会排除 `memory_spaces.include_in_people = 0` 的 scope。benchmark 编排器创建的评测 scope（如 `album_06bd13b819ea`，album3-max 100qa run）默认标记 0，导致人名无法解析成实体，`query_memory_facts` 的人物查询拿不到结果。同文件 `_event_resolution`（约 605 行）已用 entities 表直查绕过该过滤（注释注明"benchmark scope 也能用"），`_resolve_entity` 属同类漏修。
   - 实测证据（2026-08-21，data/sentrix.db）：`list_entities(status="confirmed", scope_id="album_06bd13b819ea")` 返回 0；SQL 直查同条件 person 实体返回 8（我/明明/张晓莉/乐乐/王建国/芳芳/雪儿/强子，均 confirmed）。8090 旧代码进程同库可见全部人物，8091（18:22 重启加载新代码）people/entities 均为空。
   - 修复方案：`_resolve_entity` 改为与 `_event_resolution` 相同的 entities 表直查（按 scope_id + entity_type='person' + status='confirmed' 匹配 canonical_name）；或 `list_entities` 增加绕过 include_in_people 过滤的参数。
   - 验收：修复后评测 scope 的人名解析返回 8 人；Agent 问答"明明参加过哪些活动"能解析实体并返回事实。
   - 影响：run `20260821-184608-album3-max-gemma4-e2b-it-6da9f0`（100qa-full）人物题成绩受污染（部分"未明确记录"由此导致）；修复前跑其他模型的 100QA 会同样不公平。

### 2026-08-21 变更记录（benchmark 侧维护）

- `sentrix-vllm/bin/sentrix_vllm_api.py`（8500 Manager）新增 `GET /process-memory`，对齐 `services/vllm_manager/app.py` 的同名接口：state 文件取 root_pid，`_process_tree` 追踪 vLLM 进程树，nvidia-smi compute-apps 按 PID 汇总进程显存，limit 取 `gpu_memory_utilization * 整卡显存`，并从模型端口 `/metrics` 提取 kv_cache_usage/requests 指标。原文件备份 `sentrix_vllm_api.py.bak-process-memory-20260821`；8500 已重启（8100 模型进程为独立进程组不受影响），实测 qwen3.5-0.8b 启动时返回 19096 MiB / 上限 20398.1 MiB。修复 photobench 走 8500 链路时 GPU 面板"模型进程显存/KV Cache"为空的问题。

## 接手原则

1. 先读取本文档、`README.md`、最新 Git 提交和实际 153 服务/数据库，再修改代码。
2. 后端正式提交只在 153 `psh`；本地副本仅用于同步、编辑和验证，不形成第二条
   后端提交线。
3. 传输前比较 Git 状态；不得传输 `.env`、凭据、日志、数据库、模型、备份或未审查
   实验文件。
4. 先写会失败的回归测试，再做最小修复；始终保留证据边界和 scope 隔离。
5. 不得用相册来源、评估标签、匿名模型人物描述或场景衣物替代已确认人物证据。
6. 生产推理依赖 vLLM `8100` 单例；模型切换/重启需避开在线服务窗口，并先确认
   `/api/model-profiles/current` 状态。
7. 完成后先运行新鲜验证，再更新本文档中的当前状态、实测结果和待办，并提交到
   153 `psh`。

## 同步记录

- 2026-08-10：本文档按 153 `psh` `15039f0` 实测状态全面更新（Thin Agent、vLLM、
  运行拓扑、数据库计数、验证结果与待办）；同步修正 `test/project-structure.test.js`
  三处陈旧断言（Node 31/31 通过）。
- 项目记忆双层存储约定不变：153 本文档为完整权威来源，Project Memory MCP
  `projects/sentrix-home-web/` 保存精炼共享摘要、任务、决策与 handoff；重要提交后
  两边同时更新。

## 8091 安全重启 SOP 与检索健康探测（2026-08-23 benchmark 侧部署）

- 背景：2026-08-22 排查确认，重启 8091 若只按端口杀监听进程，进程树中持
  qdrant 目录锁的进程会存活，导致新进程向量层降级 SQLite 全表扫（恒定
  20-25s、无报错）。同日代码已补可观测性（`476f649b`：qdrant 锁失败打限流
  ERROR、`/api/health` 的 `memory.vectorIndex` 暴露 degraded/按路 telemetry）。
- 本机脚本（`scripts/`，由 benchmark 项目同名脚本改编，去 ssh 层）：
  - `restart_sentrix_8091_153.sh`：安全重启 SOP——8771 活跃 run 拦截
    （`--force` 跳过）→ 按完整命令行精确 pkill（`[b]ackend` 字符类防自匹配）
    → pgrep 复核全灭（残留强杀）→ `scripts/runtime/start_sentrix_api_8091.sh`
    拉起 → health 就绪 → 自动跑 Level 1 检索探测。重启 8091 一律用本脚本，
    禁止 `lsof -ti :8091 | xargs kill`。
  - `probe_sentrix_retrieval.py`：两级检索探测。Level 1 查 health 的
    vectorIndex（qdrant 可用/未降级/锁无失败/p95，秒级、无需主模型）；
    `--live` 追加 5 个固定问题走 assistant/turn 断言单次 search_memories
    < 3s（需 8100 主模型在线）。改动 8091/向量库后必跑。
- 用法（153 本机）：
  `bash scripts/restart_sentrix_8091_153.sh`（常规）或 `--force`；
  `python3 scripts/probe_sentrix_retrieval.py --host 127.0.0.1:8091 [--live]`。
- 降级判断速查：`curl -s localhost:8091/api/health | python3 -m json.tool`
  看 `memory.vectorIndex.degraded`；日志 `logs/sentrix-api-8091.log` 搜
  "qdrant dir lock"。
