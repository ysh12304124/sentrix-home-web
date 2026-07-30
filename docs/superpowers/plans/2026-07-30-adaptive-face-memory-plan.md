# Sentrix 自适应人脸与人物记忆实施计划

## 目标

在不破坏 Asset、Observation 和原始媒体证据链的前提下，完成第一阶段的
质量感知人脸身份基础：引入 AdaFace/MagFace 可替换 embedding 边界，保存
可审计的人脸质量数据，支持多视角人物原型，并建立能够测量侧脸漏合并和
异人误合并的评估基线。随后按已批准的顺序推进事件聚合、人物实体、语义
记忆、Agent 检索和查询页面。

## 架构

```text
image Asset
  -> SCRFD/buffalo_l detector + landmarks
  -> aligned face crop
  -> FaceEmbeddingAdapter (AdaFace primary, MagFace comparator)
  -> quality score + pose bucket + embedding record
  -> multi-prototype global clusterer
  -> pending/confirmed Person entity
  -> EventParticipant + SemanticClaim + SemanticProfile
```

检测器和身份 embedding 解耦。第一阶段不直接替换生产数据，不修改 153
数据库；所有重建在本地临时数据库或显式指定的测试数据库中进行。

事件聚合使用候选评分，而不是“找到第一个候选就合并”：时间、地点、来源、
活动、对象、视觉地点和已确认人物重合分别产生可解释分数，低于合并阈值时
创建新事件，处于歧义区间时保留候选而不静默合并。

## 技术栈

- Python 3、SQLite、unittest
- InsightFace/SCRFD 负责检测和关键点
- AdaFace 作为主身份 embedding 适配器；MagFace 保留比较适配器
- OpenCV/Pillow 负责图片读取、裁剪和清晰度指标
- Sentrix 原生 SQLite entity、event、claim、profile 和 vector 表
- Node.js 原生测试验证查询结果页面和 API 数据整形

## 实施任务

### 1. 建立失败测试和适配器契约

文件：

- `backend/tests/test_model_clients.py`
- `backend/tests/test_entities.py`
- 新增 `backend/tests/test_face_clustering.py`
- 新增 `backend/face_embeddings.py`

操作：

1. 测试 embedding adapter 返回固定维度、归一化向量、模型名和版本。
2. 测试质量记录包含 detection score、face area、pose、sharpness、quality
   和 pose bucket。
3. 测试正脸、左侧脸、右侧脸被同一人物的多个 prototype 接收。
4. 测试低质量单样本不能创建高置信身份簇。
5. 测试桥接样本不能因为单个 pair 达到阈值而合并两个紧密簇。
6. 先运行新增测试，确认因接口尚不存在而失败。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_face_clustering -v
```

预期：新增测试在实现前失败，失败原因必须是缺少目标行为，而不是导入或
测试环境错误。

### 2. 实现 embedding adapter 和质量持久化

文件：

- `backend/face_embeddings.py`
- `backend/model_clients.py`
- `backend/db.py`
- `backend/pipeline.py`
- `backend/tests/test_model_clients.py`

操作：

1. 定义 `FaceEmbeddingAdapter` 协议，统一 `embed(aligned_crop)`、
   `model_name`、`model_version` 和 `quality_signal`。
2. 实现 `AdaFaceAdapter`，从 `ADAFACE_MODEL_PATH` 和
   `ADAFACE_MODEL_NAME` 读取配置；模型不可用时明确报告 unavailable，不能
   静默伪装成 buffalo_l embedding。
3. 实现 `MagFaceAdapter` 同一接口，用于离线比较。
4. 保留 SCRFD/buffalo_l 检测和关键点，生成对齐 crop 后送入身份模型。
5. 扩展 `face_instances` 保存 `quality_json`、`pose_json`、`area_ratio`、
   `sharpness`、`pose_bucket`、`embedding_model` 和 `embedding_version`。
6. 在迁移逻辑中为旧数据库增加这些列，旧记录保持可读但标记旧模型版本。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_model_clients backend.tests.test_face_clustering -v
```

### 3. 实现多视角 prototype 和全局聚类

文件：

- `backend/db.py`
- 新增 `backend/face_clustering.py`
- `backend/pipeline.py`
- `scripts/rebuild_memory.py`
- `scripts/evaluate_lfw_clusters.py`
- `backend/tests/test_face_clustering.py`

操作：

1. 为每个 pending/confirmed person cluster 保存多个 prototype，每个
   prototype 绑定 pose bucket、质量分数、来源 face instance 和模型版本。
2. 在线匹配使用最高质量 prototype 与 top-k 邻居综合分数，不再更新单一
   均值代表。
3. 全局重聚类使用质量加权的约束层次聚类；簇内最大距离和最小支持数共同
   约束合并，低密度样本保留为 noise/pending。
4. confirmed entity 的 prototype 只允许高置信新样本加入，不能被普通
   pending 样本覆盖。
5. 合并、拆分和重新分配时写入 revision，更新 face vector 的 cluster 元数据。
6. 评估脚本按 benchmark identity label 计算 pairwise precision、recall、
   F1、singleton ratio、false merge rate 和 missed merge rate。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_face_clustering backend.tests.test_entities -v
python3 scripts/evaluate_lfw_clusters.py --db data/face-benchmark.db
```

### 4. 实现事件候选评分

文件：

- `backend/db.py`
- `backend/tests/test_memory_store.py`
- `backend/tests/test_pipeline.py`

操作：

1. 将 `_event_candidates` 拆成候选生成和候选评分两个函数。
2. 记录时间差、地点匹配、来源设备/相册、活动相似度、对象重合、视觉地点
   和已确认人物重合的分项分数。
3. 最高分超过合并阈值才自动合并；多个候选接近时保留 ambiguity metadata。
4. 事件摘要只在分组完成后生成，保留原始观察证据。
5. 新增同地点不同活动不合并、跨设备同一活动合并、无人物图片保留来源
   关系的回归测试。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_memory_store backend.tests.test_pipeline -v
```

### 5. 完成人物确认和语义记忆回写

文件：

- `backend/db.py`
- `backend/app.py`
- `backend/tests/test_entities.py`

操作：

1. 统一 legacy `persons` 与 native `entities` 的人物读取和确认路径。
2. 确认、拒绝、合并、拆分都写入 `entity_revisions`。
3. 人物确认后重建 entity mentions、event participants、semantic claims、
   semantic profile 和相关 event summary。
4. 来源成员只作为 provenance；只有用户确认或已有实体映射后才能成为
   `captured_by` 人物角色。
5. 冲突身份属性进入版本链，活动、地点、衣物和出席保持多值声明。

### 6. 完成 Agent 与查询排布

文件：

- `backend/agent.py`
- `backend/app.py`
- `src/app.js`
- `src/api.js`
- `src/normalizers.js`
- `styles.css`
- `backend/tests/test_agent.py`
- `test/normalizers.test.js`

操作：

1. 将问题解析为人物、时间、地点、活动、物品、衣物和空间关系约束。
2. 并行检索 lexical、vector、person/entity、event、claim 和 visual evidence。
3. 对候选证据重排，模型只能引用已检索证据 ID。
4. 查询缺口触发专项视觉分析并保存 `query_gap` 和反馈。
5. 页面按“答案 -> 人物/事件 -> 语义声明 -> Observation -> 原始 Asset ->
   查询缺口”展示。

### 7. 153 受控验收

操作：

1. 对当前 153 Git 状态和数据库做只读快照记录。
2. 在本地临时数据库运行完整重建和人脸 benchmark。
3. 只把经过测试的代码和文档同步到 153，不同步 `.env`、数据库、日志、模型
   权重或实验数据。
4. 在 153 以小批量数据测试 AdaFace 可用性、API、网页和 FMA 5173 隔离。
5. 只有在指标和证据完整性通过后，才讨论 153 全量派生记忆重建。

## 完成门槛

- 新增失败测试先红灯，再绿灯；全量测试无回归。
- AdaFace 不可用时服务健康接口明确报告模型状态，不产生伪 AdaFace 数据。
- 侧脸与正脸的召回、误合并和漏合并有可比较指标。
- 所有新 Claim、EventParticipant 和 Agent 答案都能回到原始 Asset。
- 153 服务、FMA 5173 和用户现有数据库不因本地开发被停止或覆盖。
- 每次架构、模型、数据链路或验收状态变化都同步更新
  `docs/PROJECT_MEMORY.md`。
