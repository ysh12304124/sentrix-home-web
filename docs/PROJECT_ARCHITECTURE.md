# Sentrix Home 项目架构图

> 本文是项目级纯文本架构图，覆盖当前代码结构、数据流、运行边界，以及已实现和待实现模块。本文不展开具体算法和代码实现。

## 1. 系统边界

```text
+------------------------- 用户与家庭设备 --------------------------+
| 浏览器 / 手机端 / 局域网上传                                      |
| 图片、音频、文本、视频                                             |
+-------------------------------+---------------------------------+
                                |
                                v
+------------------------- Sentrix Home -----------------------------+
| Web UI                                                            |
| Node Web Server                                                   |
| Sentrix API                                                       |
| Ingestion Pipeline                                                |
| Native SQLite Memory Store                                        |
| Agent / Retrieval / Evidence                                      |
+----------------------+----------------------+-------------------+
                       |                      |
                       v                      v
              本地模型与运行时              原始文件与派生记忆
              Gemma / FunASR              data/media
              InsightFace / AdaFace       data/sentrix.db
              CLIP                       vectors / audit records

外部参考边界：Cognee 只作为 remember / cognify / recall / improve 的设计
参考，不作为 Sentrix 在线运行依赖，也不接管家庭原始证据和隐私边界。

并行运行边界：FMA 使用 5173 端口，属于外部系统，Sentrix 不修改、不停止。
```

## 2. 运行分层

```text
+-------------------------------------------------------------------+
| 展示层                                                            |
| index.html                                                        |
| src/app.js / src/api.js / src/normalizers.js / styles.css         |
| 事件、人物、实体、知识、资产、检索结果、证据、反馈和导入交互         |
+-----------------------------------+-------------------------------+
                                    v
+-------------------------------------------------------------------+
| Web 接入层                                                        |
| server.js                                                         |
| 静态文件服务、同源 API 代理、健康检查、导入兼容入口、Cognee 适配入口  |
+-----------------------------------+-------------------------------+
                                    v
+-------------------------------------------------------------------+
| API 与应用编排层                                                  |
| backend/app.py                                                    |
| 资源 API、导入 API、搜索 API、人物确认、事件修订、事实反馈、维护任务  |
| MemoryAgent + IngestionPipeline                                  |
+----------------------+----------------------+-------------------+
                       |                      |
                       v                      v
+--------------------------+       +------------------------------+
| 记忆处理层               |       | 模型适配层                   |
| backend/pipeline.py      |       | backend/model_clients.py    |
| Asset -> Observation     |       | Gemma 视觉与文本             |
| Observation -> Event     |       | FunASR 音频转写              |
| Event -> Fact / Claim    |       | InsightFace 人脸检测         |
| 事件摘要与人物记忆回写    |       | AdaFace / MagFace 身份向量   |
|                          |       | CLIP 图像与文本向量          |
+-------------+------------+       +------------------------------+
              v
+-------------------------------------------------------------------+
| 记忆存储层                                                        |
| backend/db.py                                                     |
| SQLite 权威数据、原始证据关系、实体关系、语义声明、向量索引、审计记录 |
| backend/face_clustering.py / backend/face_embeddings.py            |
| 人脸质量、多视角 prototype、全局聚类、可替换 embedding 边界          |
+-------------------------------------------------------------------+
```

## 3. 主要模块

| 层 | 模块 | 职责 | 状态 |
| --- | --- | --- | --- |
| Web | `src/app.js` | 主界面、导航、事件/人物/知识/资产/检索视图、交互反馈 | 已实现 |
| Web | `src/api.js` | 浏览器到 Sentrix API 的请求封装 | 已实现 |
| Web | `src/normalizers.js` | 后端数据的统一展示结构 | 已实现 |
| Web | `styles.css` | Web 页面布局和样式 | 已实现 |
| 接入 | `server.js` | 静态服务、同源代理、兼容 API、Cognee 适配入口 | 已实现；兼容入口保留 |
| API | `backend/app.py` | 资源、导入、搜索、人物、事件、事实和维护 API | 已实现；持续扩展 |
| 编排 | `backend/pipeline.py` | 多媒体摄取、观察生成、事件合并、事实和向量写入 | 已实现 |
| 编排 | `backend/agent.py` | 问题检索、证据组装、视觉补全、回答和检索轨迹 | 已实现；排序待校准 |
| 存储 | `backend/db.py` | SQLite schema、查询、实体、语义、向量和审计 | 已实现 |
| 人脸 | `backend/face_embeddings.py` | AdaFace/MagFace embedding adapter 和质量信号 | 已实现；MagFace 未部署 |
| 人脸 | `backend/face_clustering.py` | 多视角聚类、质量门控、桥接样本保护、指标 | 已实现 |
| 模型 | `backend/model_clients.py` | Gemma、FunASR、InsightFace、AdaFace、CLIP 适配 | 已实现；依赖运行环境 |
| 重建 | `scripts/rebuild_memory.py` | 从源数据重新建立派生记忆 | 已实现 |
| 评估 | `scripts/evaluate_lfw_clusters.py` | LFW 人脸聚类指标评估 | 已实现；数据标注待校准 |
| 测试数据 | `scripts/build_virtual_family_album.py` | 生成固定身份的虚拟相册和审计元数据 | 已实现，仅用于测试 |
| 测试 | `backend/tests/`、`test/` | 后端和前端回归测试 | 已实现 |
| 计划模块 | 视频记忆、session memory、MagFace 对照 | 后续扩展，不改变现有证据 ID | 待实现 |

## 4. 核心数据实体

```text
Asset                         原始文件、来源、时间、地点、设备和哈希
  |
  +--> Observation             单个 Asset 的视觉、音频或文本观察
          |
          +--> Event           按时间、地点和内容聚合的事件
          |      |
          |      +--> EventParticipant  已确认人物及事件角色
          |
          +--> Fact            有证据的通用事实及状态
          |
          +--> FaceInstance    单个检测到的人脸及 embedding
                    |
                    +--> FaceCluster  待确认或已确认的人物簇
                                  |
                                  +--> Entity / Person
                                          |
                                          +--> SemanticProfile
                                          +--> SemanticClaim
                                          +--> EntityMention

辅助实体：Relationship、MemoryVector、Story、Invite、QueryGap、
MemoryFeedback、EntityRevision、RebuildRun。

Asset / Observation / Event / FaceInstance / Claim 均保留回指原始证据的 ID。
Agent 回答不得脱离这条证据链。
```

## 5. 端到端导入与记忆管线

```text
[1] 文件进入
    |
    +--> Web 上传 / API 导入 / 批量重建脚本
    v
[2] Asset 建立
    |
    +--> 原始路径、MIME、SHA-256 内容身份与去重
    +--> EXIF 时间 / GPS / 设备
    +--> 来源成员 / 设备 / 相册 provenance
    v
[3] 媒体分流
    |
    +--> 图片 ------+
    +--> 音频 ------+--> Observation 生成
    +--> 文本 ------+
    +--> 视频 ------> 原始 Asset + video-extraction-reserved
                         当前不做视频解码和视频向量
    v
[4] 多模态观察
    |
    +--> Gemma：caption / activity / place / objects / people /
    |           clothing / spatial_relations / OCR / event_type
    +--> FunASR：语音转写、VAD、标点和时间片段
    +--> InsightFace：人脸检测、关键点、姿态、质量特征
    +--> AdaFace：五点对齐后的人脸身份 embedding
    +--> CLIP：图像向量和文本向量
    v
[5] Observation 持久化
    |
    +--> canonical observation
    +--> 原始模型输出与模型版本
    +--> 原始 Asset、来源和时间地点关联
    v
[6] Event 候选与边界判断
    |
    +--> 时间窗口、地点/GPS、视觉地点
    +--> 来源设备/相册
    +--> 活动、物体、OCR 和人物重合
    +--> 同地点同时间不同活动的冲突保护
    |
    +--> 分数足够：合并到既有 Event
    +--> 分数不足：创建新 Event
    +--> 候选接近：保存 ambiguity 供后续处理
    v
[7] 事件后处理
    |
    +--> 事件摘要与标题
    +--> Fact 写入及冲突版本管理
    +--> EventParticipant 写入
    +--> Episodic / Semantic / Visual 向量写入
    v
[8] 人物与语义记忆
    |
    +--> FaceInstance 全局重聚类
    +--> FaceCluster 待确认
    +--> 用户确认 / 拒绝 / 合并 / 拆分
    +--> EntityRevision 审计
    +--> Person -> Event -> Claim / Profile 回写
    v
[9] 可查询家庭记忆
```

## 6. 人脸身份管线

```text
InsightFace buffalo_l 检测
        |
        +--> bbox / kps / pose / detection score / face size
        v
五点关键点对齐到标准人脸输入
        v
AdaFace embedding + quality signal
        v
质量门控
        +--> 高质量样本：参与身份聚类
        +--> 低质量样本：弱证据 / pending / noise
        v
多视角 prototype
        +--> frontal / profile_left / profile_right / unknown
        v
全局约束聚类
        +--> 模型和版本隔离
        +--> bridge sample 防误合并
        +--> confirmed person 防自动跨人合并
        +--> pairwise precision / recall / F1
        v
待确认人物簇
        +--> confirm：建立人物实体和语义回写
        +--> reject：保留证据但不建立人物关系
        +--> merge：合并并记录 revision
        +--> split：拆分并记录 revision
```

## 7. Agent 查询与证据管线

```text
用户自然语言问题
        v
问题维度识别：人物 / 时间 / 地点 / 活动 / 物品 / 衣物 / 空间关系
        v
并行候选召回
        +--> lexical：事件、观察、事实、人物、声明
        +--> vector：episodic / semantic / visual
        +--> entity：Person / Entity / Relationship
        +--> evidence：Observation / Asset 原始证据
        v
候选排序与证据校验
        +--> 词法相关性、精确命中、向量相似度
        +--> 时间、地点、人物约束
        +--> 证据 ID 可回溯校验
        v
证据是否足够？
   |                         |
   | 是                      | 否且属于视觉细节问题
   v                         v
Gemma 基于证据回答       专项视觉分析
   |                         +--> 更新 Observation 版本 / 向量
   |                         +--> 创建 QueryGap
   |                         +--> 等待用户反馈或修订
   v
分层返回：答案 -> 人物/事件 -> Claims/Facts -> Observations -> Assets -> Gaps
```

## 8. 人工反馈闭环

```text
人脸簇确认 / 事实确认 / 事件编辑 / 查询反馈
        v
审计记录与版本链
        +--> EntityRevision
        +--> semantic claim supersedes / status
        +--> MemoryFeedback
        +--> QueryGap resolution
        v
局部或人物级记忆重建
        +--> EventParticipant
        +--> SemanticProfile
        +--> SemanticClaim
        +--> Event summary
        +--> Retrieval evidence
```

## 9. 已实现范围

- 图片、音频、文本的 Asset -> Observation -> Event 基础管线。
- 视频原始 Asset 登记和 `video-extraction-reserved` 保留接口。
- SHA-256 去重、EXIF 时间/GPS/设备边界和来源 provenance。
- 事件候选评分、同时间同地点不同活动的分离保护、歧义元数据。
- SQLite 原生实体、事实、语义声明、人物画像、关系和向量存储。
- AdaFace 适配器、官方 checkpoint 加载、五点对齐、质量信号和模型版本记录。
- 多视角 prototype、低质量样本隔离、桥接样本保护和确认人物保护。
- 人脸簇确认、拒绝、合并、拆分及 revision 审计。
- 人物确认后的事件角色、语义声明、人物画像和事件摘要回写。
- Agent 的词法检索、向量检索、实体/事件检索、证据校验和结构化检索轨迹。
- 查询缺口、专项视觉补全、反馈写回和分层结果展示。
- 事件、人物、知识、资产、原始证据和人脸簇的 Web 交互。
- 批量重建、重建运行审计、失败清理和回归测试。

## 10. 待实现或继续完善

- AdaFace 在真实家庭相册上的阈值和侧脸/正脸指标校准。
- LFW/虚拟相册 benchmark 的多脸真值标注和聚类 purity 评估完善。
- MagFace 对照实验，以及在 AdaFace 不足时的模型选择决策。
- 语义声明更丰富的中文谓词、习惯阈值和冲突策略。
- Agent 查询排序、时间地点约束和视觉补全触发条件继续校准。
- 更完整的来源 provenance 校验和异常数据检测。
- 153 `psh` 正式分支整理、提交和受控发布流程。
- 生产派生记忆重建前的指标门禁、回滚和小批量验收自动化。
- 视频镜头切分、视频观察、视频向量和时间戳检索。
- 本地 session memory；在核心证据检索稳定后再引入。

## 11. 部署与数据边界

```text
本地开发：phone app / local branch psh2

153 正式后端：
  /home/asus/Github/Sentrix-Home-Web
  Sentrix API 8090
  Sentrix Web 4174
  formal backend commit target: psh

独立外部服务：FMA 5173（不可停止、不可修改）

敏感或运行期数据：.env、凭据、日志、数据库、模型权重和实验数据
不进入代码同步或架构文档。
```

## 12. 架构原则

```text
原始 Asset 是事实边界
Observation 是可追溯观察
Event 是有证据的聚合
Entity 是经过确认的人物或对象
Claim 是带状态、置信度和证据的语义记忆
Agent 只能基于已检索证据回答
Cognee 只提供设计启发，不接管 Sentrix 本地证据存储
```

## 13. 当前文件模块清单

```text
项目入口与运行：
  README.md
  index.html
  server.js
  package.json
  styles.css

后端核心：
  backend/app.py
  backend/agent.py
  backend/db.py
  backend/pipeline.py
  backend/model_clients.py
  backend/face_embeddings.py
  backend/face_clustering.py
  backend/requirements.txt

前端：
  src/app.js
  src/api.js
  src/normalizers.js

数据与重建脚本：
  scripts/build_virtual_family_album.py
  scripts/download_test_data.py
  scripts/evaluate_lfw_clusters.py
  scripts/ingest_face_benchmark.py
  scripts/prepare_test_metadata.py
  scripts/rebuild_memory.py

后端测试：
  backend/tests/test_agent.py
  backend/tests/test_entities.py
  backend/tests/test_face_clustering.py
  backend/tests/test_memory_store.py
  backend/tests/test_model_clients.py
  backend/tests/test_pipeline.py
  backend/tests/test_rebuild.py

前端测试：
  test/no-demo-data.test.js
  test/normalizers.test.js

架构、计划和记忆文档：
  docs/architecture.md
  docs/implementation-plan.md
  docs/implementation-plan-person-centered.md
  docs/test-datasets.md
  docs/PROJECT_MEMORY.md
  docs/PROJECT_ARCHITECTURE.md
```
