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
  `docs/superpowers/`，历史实现由 Git 历史保存，不再维护第二份当前架构文档。

## 产品定义

Sentrix 是本地优先的家庭记忆系统。它将原始图片、音频、文本和预留的视频入口
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

## 当前基线

- 153 仓库：`/home/asus/Github/Sentrix-Home-Web`
- 正式后端提交分支：`psh`
- 当前提交：`d588c53` (`perf: accelerate core image semantics`，2026-08-03)。
- 当前工作树：干净。
- Web：`http://192.168.0.153:4174`
- API：端口 `8090`
- FMA Web：端口 `5173`，与 Sentrix 无关，禁止停止、修改或重启。
- 当前数据库：`/home/asus/Github/Sentrix-Home-Web/data/sentrix.db`
- 视频只保留 `VideoMemoryAdapter` 接口；未实现视频解码、关键帧、时序 Observation、
  视频向量和视频事件。

## 模型与运行隔离

Sentrix 使用项目独立 Ollama：`127.0.0.1:11435`。共享系统 Ollama 位于
`127.0.0.1:11434`，属于其他项目，禁止修改或停止。

- 主多模态模型：`gemma4:12b`。
- 语音：FunASR，`paraformer-zh`、`fsmn-vad`、`ct-punc`。
- 人脸检测及关键点：InsightFace `buffalo_l`。
- 人脸身份向量：AdaFace `ir_50`；权重路径：
  `/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt`。
- 视觉向量：CLIP `ViT-B-32`；当前数据库图像向量为 512 维。
- Sentrix 进程必须设置：`OLLAMA_BASE_URL=http://127.0.0.1:11435`、
  `OLLAMA_MODEL=gemma4:12b`；生产默认 `OLLAMA_KEEP_ALIVE=0`。
- `keep_alive=0` 会在请求结束后卸载 12B 模型，防止同一时间与其他项目常驻两个
  12B 模型而 OOM。长任务前需检查 `11434` 和 `11435` 的 `/api/ps`；当前
  `11435` 无常驻模型。
- 2026-07-30 已验证独立运行器识别 RTX 3090，并将 Gemma `49/49` 层卸载至
  `CUDA0`。`nvidia-smi` 受 NVML 用户态/内核版本不一致影响不可用；以 Ollama CUDA
  运行日志作为 GPU 验证来源。
- 当前 API health 显示 AdaFace、`buffalo_l`、FunASR、CLIP ready。health 字段
  `gamma4_12B` 是历史命名，实际配置模型为 `gemma4:12b`。
- `scripts/runtime/start_sentrix_api.sh` 在进程启动前配置 NVIDIA pip runtime
  动态库，并优先使用 `CUDAExecutionProvider`。当前正式 `8090` 仍为旧直接
  `uvicorn` 进程；切换须在维护窗口执行。

## 仓库结构

```text
backend/                 FastAPI、SQLite、处理管线、模型适配器、聚类和 Python 测试
src/                     浏览器端 JavaScript、API 包装器和样式
scripts/runtime/         运行时工具，例如独立 Ollama 启动器
scripts/maintenance/     显式、耗时或可能破坏数据的维护命令
scripts/benchmarks/      只读评估和家庭基准交集生成工具
scripts/fixtures/        可复现公开测试资料和元数据生成器
test/                    Node 前端与目录结构回归测试
docs/                    本文档和已批准的设计/实施记录
server.js                静态文件服务与同源 `/api/*` 代理
```

`server.js` 只代理到 `backend.app`，不包含 mock API、Cognee 回退或第二套记忆实现。
`backend.app` 与 `MemoryStore` 是唯一 HTTP 和持久化权威边界。

## 数据模型与证据链

基础证据链：

```text
MemorySpace
  -> Asset -> Observation -> Event -> Person/Entity -> SemanticClaim/Profile
       \_________________________ 原始证据 __________________________/
```

人物外观声明要求更强证据链：

```text
Asset -> Observation -> FaceInstance -> PersonAppearanceEvidence -> SemanticClaim
                             \-> 已确认 Person -> Event
```

基准相册使用人物事件投影：

```text
MemorySpace / Household
  -> Asset -> Observation -> Event
  -> PersonEventMemory
  -> PersonPattern / SemanticClaim
  -> 原始 Asset 或 FaceInstance 证据
```

| 对象 | 责任 | 下游用途 |
| --- | --- | --- |
| `memory_spaces` | 记录家庭或基准相册空间与来源 | 隔离全部读写与检索 |
| `assets` | 原始文件、SHA-256、时间地点、允许来源 | 文件访问、处理、证据 UI |
| `observations` | 模型观察、原始 JSON、对象、场景衣物、OCR、转写 | 事件、向量、原始证据 |
| `face_instances` | 人脸框、质量、姿态、AdaFace 向量、模型版本 | 头像和人物候选 |
| `face_clusters` / `face_prototypes` | 质量感知多原型候选身份与审计状态 | 人工确认、合并、拆分 |
| `entities` / `entity_mentions` | 人物实体及其 Observation 提及 | 事件参与者、画像、关系 |
| `events` / `event_observations` | 事件边界、总结和证据成员 | 时间线、语义、检索 |
| `event_participants` | 已确认人物在事件中的角色和证据 | 人物事件投影 |
| `person_event_memory` | 人物在单个事件的活动、地点、时间、共现人与证据 | 人物时间线和 Agent |
| `person_patterns` | 跨事件活动、地点、共现人和外观模式 | 长期人物画像 |
| `semantic_profiles` / `semantic_claims` | 跨事件画像和版本化详细声明 | 人物页、知识页、Agent |
| `person_appearance_evidence` | 人脸、上半身裁剪、目标衣物和原始证据 | 人物衣物声明 |
| `relationships` | 人工确认或维护的实体关系 | 关系图、Agent |
| `memory_vectors` | `visual`、`episodic`、`semantic` 向量，带 scope 和模型信息 | 相似检索 |
| `query_gaps` / `memory_feedback` | 查询缺口和人工反馈 | 维护闭环 |
| `rebuild_runs` | 重建范围、状态、统计和错误 | 运维审计 |

## 端到端管线

1. **资料导入**：`IngestionPipeline.create_asset()` 计算 SHA-256，只保留时间、
   地点、设备、相册归属等白名单来源；缺失时读取 EXIF；在同一 `scope_id` 内去重。
2. **媒体处理**：图片先写入 AdaFace、CLIP、元数据和事件的快速证据，再以
   896px、禁用思考模式、320 token 的核心中文 JSON 完成语义丰富；音频经 FunASR
   后走文本分析；文本走同一文本分析路径；视频只返回预留状态。
3. **证据规范化**：保存中文规范字段、原始模型 JSON、对象、场景衣物、空间关系、
   OCR、转写、人脸实例和模型版本。
4. **视觉与身份索引**：事件选择前写入 CLIP Asset 向量；`buffalo_l` 检测、对齐后
   用 AdaFace 写入身份向量。低质量检测保留为证据，但不必进入候选聚类。
5. **事件构建**：先按时间地点召回，再综合活动/类型、对象、视觉地点、已确认人物、
   CLIP 相似度和地理邻近度。仅当活动和类型均冲突、无确认人物/对象桥接且向量可比
   时，低视觉兼容度才触发保守拆分保护。
6. **事件总结**：导入热路径只建立可检索的事件和 Observation 证据；待总结事件由
   `POST /api/maintenance/summarize-events` 的后台维护任务生成中文标题、类型、活动
   和摘要，避免同一事件随每张图片重复推理。导入标签绝不参与。
7. **人物确认**：确认姓名和可选角色后，刷新实体提及、参与者、事件摘要、
   `person_event_memory`、`person_patterns`、画像和声明。原始 Observation 的匿名
   模型描述不被改写。
8. **人物外观**：每个关联事件最多选择一个高质量已确认人脸，裁剪头部和上半身并
   分析目标人物衣物；场景衣物不会因共现升级为人物事实。
9. **检索维护**：Agent 优先召回人物画像、模式、声明、事件、Observation 和向量，
   校验证据 ID，并返回轨迹、原图和缺口；反馈写入
   `query_gaps`/`memory_feedback`，不伪装成普通查询。

## 模块实现

| 模块 | 实现方式 | 权威输出 |
| --- | --- | --- |
| `backend/app.py` | FastAPI 路由、范围筛选、人物确认、检索对话、原图与维护 | HTTP API 编排 |
| `backend/db.py` | SQLite 加法迁移、事务、事件评分、空间隔离、聚类与人物投影 | 唯一权威记忆库 |
| `backend/pipeline.py` | 白名单元数据、媒体适配、向量/人脸先写入再归并事件 | Asset、Observation、事件候选 |
| `backend/model_clients.py` | Gemma、FunASR、`buffalo_l`、AdaFace、CLIP 薄适配器 | 带版本结构化输出 |
| `backend/face_embeddings.py` / `face_clustering.py` | AdaFace、质量姿态门控、多原型全局重聚类 | 人脸实例、簇、原型 |
| `backend/person_appearance.py` | 从脸向下扩展的确定性上半身裁剪 | 外观裁剪坐标 |
| `backend/agent.py` | 意图分类、混合召回、证据校验、局部视觉补全、对话反馈 | 答案、图片结果、轨迹、缺口 |
| `scripts/maintenance/rebuild_memory.py` | 批量重建，可按来源或 benchmark manifest 运行 | 重建审计与派生记忆 |
| `scripts/benchmarks/prepare_household_benchmark.py` | 图片与元数据/人脸/查询标注取交集 | 可审计导入 manifest |
| `scripts/benchmarks/evaluate_household_benchmark.py` | 只读计算人脸、事件、空间和图片查询指标 | 外部评估报告 |
| `src/api.js` / `src/app.js` | 同源 API、范围选择、证据浏览、Agent 对话、静默轮询 | 浏览器本地状态 |

## API 与交互约定

- 规范人物确认入口：`POST /api/face-clusters/{cluster_id}/confirm`。
- 兼容入口：`POST /api/persons/{person_id}/confirm` 会先将原生 `entity_id` 解析到
  活跃人脸簇，再执行同一传播链，避免实体 ID 被旧 `persons` 路由误判为 404。
- 人物证据查看与确认分开：查看显示人脸裁剪、原图、关联事件和状态；确认才打开
  姓名/角色表单。
- `POST /api/assistant/turn` 接收 message、`conversation_id`、feedback、`scope_id`，
  区分 `query`、`feedback`、`clarification`。`POST /api/search` 是兼容包装器。
- 图片结果带 `asset_id`、`observation_id`、文件名、时间、caption、`media_url`，前端
  展示 `/api/assets/{asset_id}/file` 原图缩略图和打开入口。
- 头像统一使用 `/api/face-instances/{face_instance_id}/crop`，不使用整图冒充头像。
- 前端五秒轮询仅刷新计数；当前页面、媒体 DOM、表单和弹窗输入不得被静默重建。

## 记忆空间与家庭基准

当前家庭基准源：`/Users/rm001/Downloads/samples`。

| 空间 | 当前图片数 | 原始 metadata 条目 | 原始 face-map 条目 | 查询数 |
| --- | ---: | ---: | ---: | ---: |
| `album1` | 64 | 1069 | 78 | 20 |
| `album2` | 58 | 1466 | 638 | 20 |
| `album3` | 69 | 1047 | 257 | 20 |

- 导入器支持根目录 `metadata.json` 和嵌套 `metadata/metadata.json`，只导入
  `images/` 实际存在的文件，并记录未匹配标注。
- `face_info_cn.json`、`face_info_en.json`、`face_id_images`、`query.json` 仅供评估；
  不得自动确认人名、创建关系、生成事件名或写入模型提示。
- 三个相册为独立 `MemorySpace`。Asset、Observation、事件、实体、向量、语义和
  Agent 读取均按 `scope_id` 过滤，禁止跨空间事件归并。
- 本次受控基准显式设定 `FACE_IDENTITY_MIN_QUALITY=0.35`；全局默认仍更严格，
  此设置不等于生产默认已经放宽。

## 当前 153 数据库状态

以下为 2026-08-02 live SQLite 实测值，不是目标值，也不是质量验收结论：

| 记录 | 总计 | `album1` | `album2` | `album3` |
| --- | ---: | ---: | ---: | ---: |
| Asset | 189 | 62 | 58 | 69 |
| 已处理 Asset | 183 | 60 | 57 | 66 |
| 失败 Asset | 6 | 2 | 1 | 3 |
| Observation | 183 | 60 | 57 | 66 |
| 活跃 Event | 92 | 26 | 34 | 32 |
| Event-Observation 链接 | 183 | 60 | 57 | 66 |
| EventParticipant | 14 | 1 | 13 | 0 |
| FaceInstance | 73 | 10 | 38 | 25 |
| FaceCluster | 11 | 5 | 4 | 2 |
| 人物 Entity | 11 | 5 | 4 | 2 |
| SemanticProfile | 3 | 1 | 2 | 0 |
| SemanticClaim | 86 | 7 | 79 | 0 |
| PersonEventMemory | 14 | 1 | 13 | 0 |
| PersonPattern | 29 | 2 | 27 | 0 |
| Fact | 2 | 1 | 1 | 0 |
| Relationship | 0 | 0 | 0 | 0 |
| MemoryVector | 714 | 216 | 243 | 255 |
| QueryGap / MemoryFeedback | 0 / 0 | - | - | - |
| RebuildRun | 3 | 1 | 1 | 1 |

实体状态：`album1` 为 1 已确认、4 待确认；`album2` 为 2 已确认、2 待确认；
`album3` 为 2 待确认。三次重建均为 `completed_with_failures`，所以当前数据不完整，
不得把查询、事件、人物关系统计当作最终质量结论。

## 已完成工作

- 原始资料、SHA-256 去重、EXIF 时间/GPS/设备回退、相册来源白名单。
- `MemorySpace` 隔离的相册导入、查询和网页选择。
- 图片、音频、文本 Observation 管线；视频原始 Asset 预留。
- `buffalo_l` 检测、AdaFace 向量、质量/姿态元数据、多原型全局聚类、低质量人脸
  证据化、候选簇确认/拒绝/合并/拆分与审计。
- 人脸裁剪证据接口与统一头像显示。
- 事件多信号评分、CLIP 可比性检查、保守拆分保护、事件级中文总结和相册内兼容
  事件合并。
- 人物确认后的参与者刷新、保守重分段、事件摘要刷新和人物语义重建。
- 人物事件投影、跨事件模式、版本化声明、画像和事实状态管理。
- 目标人物上半身外观证据；无该证据时不生成个人衣物事实。
- 证据优先 Agent：人物、地点、日期、活动、物体、衣物、空间关系、图片结果、
  局部视觉补全、缺口、反馈和有界对话。
- 原图、事件、人物、知识、资产、故事、导入、设置等网页入口；图片搜索不会只列
  文件名。
- 独立 GPU Ollama `11435`，自动释放 12B 模型；仓库根目录重复实现已清理，脚本已
  按运行、维护、夹具和基准分类。
- 家庭基准交集生成器和只读评估器已实现。
- LFW 主脸隔离基准与自动质量门禁：120 图 coverage `0.9917`、pairwise F1 `0.9916`，
  超过 95% 目标；基准只评估，不写入正式记忆。
- 分层 Agent 已按语义、事件、Observation/Asset 证据检索；时间、地点、人物、物体等
  结构化命中可跳过向量检索，置信度不足时降级为事件与原始证据回答。
- 地点、物体、情感实体、`entity_observations` 和带证据 ID 的实体关系已进入记忆库；
  搜索页显示可折叠的算法判断依据。
- 核心视觉语义已加入只读 `evaluate_vision_model.py` 门禁；并发后台导入为每任务独立
  SQLite 连接，初始化锁避免 schema migration 和 GPU 模型重复加载竞争。

### 已通过的受控历史验收

以下说明特定受控数据和当时版本曾通过，不替代当前三相册基准验收：

- 120 张虚拟家庭相册曾完整重建，外部 manifest 人脸候选评估为 precision `1.0000`、
  recall `1.0000`、F1 `1.0000`；148 个检测人脸中，84 个高质量样本形成 4 簇，64 个
  低质量样本保留证据。
- 一次受控确认生成头像/画像、五条事件活动声明、事件参与者和
  `PersonAppearanceEvidence`；活动、衣物和原图 Agent 查询均有证据返回。
- 已验证事件导入不读取 `event_id`、`activity_hint` 等评估标签。LFW 衍生
  “生日/维修”标签没有可观察画面差异，不能作为模型应该切开的有效测试。
- 已验证独立 Ollama GPU 卸载和闲置后模型卸载。

## 最近验证（2026-08-03）

- `psh` 当前提交为 `d588c53`，工作树干净。正式 API `8090`、Web `4174` 与 FMA
  `5173` 正常；FMA 未修改。专用 Ollama `11435` 已卸载模型。
- 人脸门禁：`evaluate_lfw_clusters.py` 在隔离库输出 coverage `0.9917`、pairwise
  F1 `0.9916`，通过 95% 目标。
- 核心视觉模型门禁：三张真实场景图均返回完整中文 JSON、必需字段和证据字段，均值
  `3.2791s/图`，相对 `18.14s/图` 旧视觉基线为 `5.5321x`。
- 临时 GPU API `8095` 的预热稳定态三图完整语义：`9.8922s`，对比旧同步完整路径
  `54.4271s`，为 `5.502x`；三个 Asset 均为 `processed`，日志确认
  `CUDAExecutionProvider`。临时 API 已停止，测试仅写入 `/tmp`。
- 相关 Python 回归：69 项通过，包括并发 SQLite、FaceAdapter 契约、管道、模型请求
  参数和视觉门禁。正式 `8090` 尚未重启加载本轮代码。
- `GET /api/health` 返回 `200`；AdaFace、`buffalo_l`、FunASR、CLIP 均报告 ready；
  `GET http://127.0.0.1:11435/api/ps` 返回空模型列表。
- Web `4174` 返回 `200`；FMA `5173` 未改动。
- `node --test test/*.test.js`：10 项通过。
- `node --check src/app.js`、`node --check src/api.js`、
  `.venv/bin/python -m compileall -q backend scripts`、`git diff --check` 均通过。
已知测试警告：隔离测试构造的 `ClipAdapter` 输出
`No pretrained weights loaded for model 'ViT-B-32'. Model initialized randomly.`。
这不证明生产运行随机初始化，但也不能替代生产 checkpoint 图文推理验收。

## 当前未完成事项

### P0：MVP 门槛

1. **生产切换与烟雾验证**：在维护窗口将 `8090` 从直接 `uvicorn` 切换到
   `scripts/runtime/start_sentrix_api.sh`，确认 health、CUDA provider、快速证据、核心
   语义和待总结事件维护任务；准备并验证回滚命令，不影响 FMA。
2. **清理失败资料并完成可复现重建**：定位 `album1` 2 个、`album2` 1 个、
   `album3` 3 个失败 Asset 的阶段和错误，针对性重试或明确记录不可处理原因；确保
   每个可用图片都有 Observation、事件链接、空间和处理状态；所有重建不再为
   `completed_with_failures` 后才可谈完整性验收。
3. **运行真实基准并设定门槛**：对当前数据库运行事件分割、空间隔离、
   查询原图召回的只读评估；按相册记录输入交集、precision、recall、F1、误合并、
   漏合并、事件拆分/合并、hit rate 和证据正确性。在有可观察的同时间同地点不同
   活动资料前，不改事件拆分 `0.45` 阈值或放宽生产规则。
4. **完成 CLIP 生产验收**：用真实生产 checkpoint 强制执行一次图片和文本 embedding，
   确认不是随机初始化，记录跨模态/相似度结果和向量维度；714 条向量的存在不构成
   质量验收。
5. **完成人物身份闭环**：审阅 8 个待确认簇，按真实情况命名、拒绝或拆分；确认后
   检查 UI 与 API 的 Observation、Event、Pattern、Claim、Appearance 影响数量。关系
   必须人工明确确认，当前 `relationships=0`。
6. **完成查询和反馈闭环验收**：用三相册 query 集评估原图命中率、答案事实性和
   证据链；实际执行一次视觉补全后二次查询，及一次接受/纠正反馈，验证
   `query_gaps`、`memory_feedback` 与后续结果。

### P1：产品可用性与语义质量

1. 知识 UI 从平铺 Claim 改为清晰的
   `Person -> Event -> PersonPattern / SemanticClaim -> Evidence`，展示时间线、模式、
   外观、关系和原图路径。
2. 完善人物簇合并/拆分的人工审阅流程与端到端 UI 回归。
3. 制定审核过的中文属性词表后，再做保留证据的同义归一化，例如“深色西装外套”
   与“黑色西装外套”；当前不得武断合并。
4. 用真实家庭问题集校准 Agent 排序、置信度和查询缺口策略。
5. 扩大来源验证、失败阶段重试、异常恢复和重建可观测性。

### P2：明确延后

- 视频解码、关键帧、时序证据、视频向量和视频事件记忆。
- 主动式家庭建议和会话外长期推荐。
- MagFace 对比及是否替换 AdaFace 的生产决策。
- API 与独立 Ollama 的正式托管服务、重启策略、监控和主机级所有权确认。

## 运行与验证命令

```bash
cd /home/asus/Github/Sentrix-Home-Web

scripts/runtime/start_sentrix_ollama.sh

FACE_MODEL_ROOT=$PWD/data/face-models \
FACE_MODEL_NAME=buffalo_l \
FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
FACE_EMBEDDING_MODE=adaface \
ADAFACE_ARCHITECTURE=ir_50 \
ADAFACE_DEVICE=cuda \
ADAFACE_MODEL_PATH=/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt \
ADAFACE_REPO_ROOT=/home/asus/models/AdaFace \
OLLAMA_BASE_URL=http://127.0.0.1:11435 \
OLLAMA_MODEL=gemma4:12b \
OLLAMA_KEEP_ALIVE=0 \
scripts/runtime/start_sentrix_api.sh

SENTRIX_BACKEND_URL=http://127.0.0.1:8090 PORT=4174 npm run dev
```

提交或宣称完成前：

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
git diff --check
```

重建会替换派生记忆，执行前必须确认数据库目标和 `scope_id`：

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py \
  --root . --source /path/to/source-album
```

## 关键历史里程碑

| 日期 | 提交/阶段 | 结果 |
| --- | --- | --- |
| 2026-07-29 | `6ad4ae8` | 首次提交证据驱动家庭记忆项目，建立项目记忆和基础数据契约。 |
| 2026-07-30 | AdaFace、聚类、人物确认、事件评分、独立 Ollama | 完成受控 120 图验证；早期 34 簇碎片化方案被淘汰。 |
| 2026-07-30 | `064d952` | 强化基于证据的事件分割。 |
| 2026-07-31 | `4fced260` | 完成人物命名兼容、对话 Agent、图片结果和反馈分流。 |
| 2026-07-31 | `bf286a1` | 引入 MemorySpace、三相册基准、人物事件投影和只读评估工具。 |
| 2026-07-31 | `d00cd02` 至 `dab0f66` | 改进重建期间页面更新、基准默认相册、事件合并、轮询输入/媒体保护、增量相册隔离和人物事实清理。 |
| 2026-08-02 | `853ff66` | 对 153 工作树、服务、数据库和未完成 MVP 门槛完成当前状态核验。 |
| 2026-08-03 | `878579b`、`d588c53` | 加入视觉模型门禁、核心语义快速路径、延迟事件总结与并发导入修复；隔离稳定态完整语义达到 `5.502x`。 |
| 2026-08-03 | `cefb3f1`、`065f784`、`8c291d5` | 三相册使用 GPU 管线重跑并替换正式派生库：`189/189` Asset 均已处理，全部人物保持待命名。身份查询直接返回可审阅簇、原图证据和 `identity` 查询缺口，跳过向量与大模型；待确认簇内部标识不再从 API/UI 泄漏。 |

## 当前生产结果（2026-08-03）

- 正式 SQLite 已备份为 `data/backups/sentrix-before-household-rerun-20260803.db`，不纳入 Git。
- 当前 `data/sentrix.db` 完整性为 `ok`：`album1=62/62`、`album2=58/58`、`album3=69/69`，合计 `189/189` 均为 `processed`；所有人物实体为 `pending`，确认人物数为 `0`。
- `8090` 由 `scripts/runtime/start_sentrix_api.sh` 启动，进程环境含
  `FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` 与 NVIDIA runtime 库路径；Web `4174` 和 FMA `5173` 均在本次切换后返回 `200`，未改动 FMA。
- 真实相册标签没有人脸框，且存在漏检和标注人数不一致；不能以此宣称三相册人脸 F1 达到 95%。LFW 受控门禁结果仍为 coverage `0.9917`、F1 `0.9916`。

## 接手原则

1. 先读取本文档、`README.md`、最新 Git 提交和实际 153 服务/数据库，再修改代码。
2. 后端正式提交只在 153 `psh`；本地副本仅用于同步、编辑和验证，不形成第二条
   后端提交线。
3. 传输前比较 Git 状态；不得传输 `.env`、凭据、日志、数据库、模型、备份或未审查
   实验文件。
4. 先写会失败的回归测试，再做最小修复；始终保留证据边界和 scope 隔离。
5. 不得用相册来源、评估标签、匿名模型人物描述或场景衣物替代已确认人物证据。
6. 完成后先运行新鲜验证，再更新本文档中的当前状态、实测结果和待办，并提交到
   153 `psh`。
