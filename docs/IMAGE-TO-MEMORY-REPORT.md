# Sentrix 图片到记忆完整链路报告

更新时间：2026-08-04

本文只描述 Sentrix 后端从图片进入系统，到形成可追溯事件、语义实体、人脸记忆和向量记忆的实现。本文不包含 Agent 检索、对话、回答生成、查询重排或网页展示链路。

## 1. 范围与原则

Sentrix 的图片记忆生成遵循以下边界：

1. 原始图片是唯一不可替代的证据源。
2. 导入元数据只允许提供文件、拍摄时间、GPS、设备、相册来源和作用域。
3. 相册名称、文件名、GPS 坐标和外部标签不能直接变成事件名称、人物或活动。
4. 模型输出分为原始观察、规范化字段和语义实体三层，不能只保留模型最终选择。
5. 每个事件、实体、声明和向量都必须能够回到 Observation 与 Asset。
6. GPS 是地点锚点和 `geo` 属性，不是地点实体名称；地点实体名称来自图片语义。
7. 人脸聚类在用户确认前只是候选身份，不能自动变成命名人物。

## 2. 系统边界

```text
浏览器选择图片
       |
       v
POST /api/ingest
       |
       v
原始文件 + Asset + 来源元数据
       |
       v
图片级并行处理
  |          |          |
 Face      CLIP      Gamma视觉
  |          |          |
  +---- Observation 临时证据 ----+
       |
       v
事件候选选择与实体投影
       |
       v
Gamma 语义补全 + CLIP 文本向量
       |
       v
批次完成后按事件总结
       |
       v
Event / Entity / Claim / Profile / Vector / Audit
```

主要实现模块：

| 模块 | 职责 |
| --- | --- |
| `backend/app.py` | 上传接口、后台任务、批次完成接口、处理任务编排 |
| `backend/pipeline.py` | Asset 处理、图片观察、语义丰富、事件总结、向量写入 |
| `backend/model_clients.py` | Gemma、Face、FunASR、CLIP 适配器；本文只使用图片相关部分 |
| `backend/semantic_taxonomy.py` | 地点、物品、氛围的主类、细节和别名规范化 |
| `backend/db.py` | SQLite schema、事务、事件候选、实体投影、向量存储和审计 |
| `scripts/maintenance/rebuild_memory.py` | 从原始相册重新建立派生记忆 |
| `src/api.js` / `src/app.js` | 创建上传批次、上传文件、发送批次完成信号 |

## 3. 图片导入

### 3.1 浏览器侧批次

用户一次选择多个文件时，浏览器生成一个唯一 `batch_id`。每个文件请求都携带同一个 `batchId`，文件上传完成后再调用：

```text
POST /api/ingest-batches/{batch_id}/complete
```

这个完成信号只表示“本批文件已经全部提交”，不表示后台处理已经完成。后台仍会等待批次内所有 Asset 进入终态。

终态包括：

- `processed`
- `failed`
- `video-extraction-reserved`（图片链路不会产生）

批次状态：

```text
open -> complete -> summarizing -> completed
```

批次完成之前，图片可以并行进行 Face、CLIP 和 Gamma 处理，但不能触发该图片所属事件的最终总结。

### 3.2 `/api/ingest`

接口接收到图片后按以下顺序处理：

1. 使用 `Path(file.filename).name` 清理文件名，避免路径穿越。
2. 分配 `asset_id`。
3. 将原图保存到 `data/media/{asset_id}_{file_name}`。
4. 根据 MIME 类型确认是 image。
5. 计算 SHA-256 内容哈希。
6. 读取 EXIF 中的拍摄时间、设备和 GPS。
7. 创建或复用当前 `scope_id` 下的 Asset。
8. 将 `batch_id` 写入 `assets.batch_id` 和 `metadata_json`。
9. 返回 HTTP `202`。
10. 通过 FastAPI `BackgroundTasks` 安排后台处理。

重复文件按照内容哈希去重。重复上传不会再次生成一份原始 Asset；它可以作为该批次的已存在证据返回。

### 3.3 允许进入记忆的导入元数据

`IngestionPipeline.create_asset()` 只允许以下来源字段：

```text
content_sha256 / sha256
exif
captured_at
captured_location
source_owner_id
source_owner_label
source_device_id
source_album_id
source_confidence
scope_id
batch_id
```

导入元数据不会允许写入 `event_id`、`event_hint`、`activity_hint`、姓名、家庭角色、关系或评估标签。

## 4. Asset 与 Observation

### 4.1 Asset

`assets` 保存原始媒体及其来源边界：

```text
id
scope_id
batch_id
file_name
media_type
path
mime_type
size_bytes
status
metadata_json
source_owner_id
source_device_id
source_album_id
content_sha256
captured_at
captured_location
created_at
updated_at
```

Asset 是原始文件指针和处理状态，不是图片语义本身。

### 4.2 Observation

Observation 是一张图片在某一处理版本下的可验证观察：

```text
asset_id
captured_at
source_type
caption
activity
place
people
objects
ocr_text
event_type
clothing
spatial_relations
confidence
raw_json
canonical_json
revision
```

图片快速阶段会先创建一条临时 Observation：

```text
source_type = image_fast_evidence
caption = ""
canonical.semantic_status = pending
```

这样原始图片可以先进入事件和视觉证据索引。语义阶段完成后使用 `enrich_observation()` 更新字段、增加 `revision`，并保留原始模型 JSON。

## 5. 图片级并行处理

### 5.1 并行分支

`process_fast_image()` 和完整图片路径都会把相互独立的任务分开：

```text
FaceAdapter.detect(path)       
ClipAdapter.embed_image(path)  } ThreadPoolExecutor 并行
GammaClient.analyze_image(path) }
```

Face 和 CLIP 不写 SQLite；所有数据库写入在模型结果返回后完成，以避免多个线程同时修改同一个事务。

Gamma 请求在多个上传任务之间可以并发提交。实际并发度受 Ollama 推理槽位、显存和 GPU 吞吐限制，当前代码没有把 Gamma 强制改成串行。

### 5.2 Face 分支

图片进入 `FaceAdapter.detect()` 后：

1. InsightFace `buffalo_l` 检测人脸、边框、关键点和检测置信度。
2. 计算人脸面积占比、清晰度、姿态和质量信号。
3. 使用 AdaFace `ir_50` 生成身份向量。
4. 记录模型名称、权重版本、质量和是否满足身份聚类门槛。
5. 通过 `MemoryStore.add_face_instance()` 保存 `face_instances`。
6. 高质量且模型可用的人脸进入在线候选簇。
7. 低质量人脸仍保留为原始证据，但不能单独制造强身份桥接。

人脸向量不直接写入普通语义实体。它首先形成 `FaceInstance` 和 `FaceCluster`，用户确认后才传播为命名 Entity。

### 5.3 CLIP 图像分支

`ClipAdapter` 使用本地 CLIP `ViT-B-32`：

1. 首次使用时加载项目配置的 checkpoint。
2. 根据 `CLIP_DEVICE` 选择 CUDA 或 CPU。
3. 读取原图并使用 CLIP preprocess。
4. 在 `torch.no_grad()` 下计算图像向量。
5. 归一化后写入 `memory_vectors` 的 `visual` 空间。

当前生产图像向量为 512 维。

## 6. Gamma 图片观察

### 6.1 输入编码

原始 Asset 不会被修改。Gemma 输入副本经过：

1. PIL 打开图片。
2. 转换为 RGB。
3. 最长边缩放到 `VISION_CORE_MAX_DIMENSION`，默认 `896`。
4. JPEG quality `90` 编码。
5. Base64 放入 Ollama `/api/chat` 请求。

当前 Gamma 配置：

```text
model       = gemma4:12b
endpoint    = http://127.0.0.1:11435
stream      = false
temperature = 0
think       = false
num_ctx     = 4096
num_predict = 320
keep_alive  = -1
```

`keep_alive=-1` 在客户端必须作为数值发送，不能发送字符串 `"-1"`，否则 Ollama 返回 HTTP 400。

### 6.2 核心图片提示词

实际提示词的核心内容如下，地点选项会由 `PLACE_PRIMARY_TYPES` 动态追加：

```text
你是家庭记忆观察器。仅根据图片和元数据抽取可验证的核心观察，不猜测姓名。
严格返回简体中文 JSON 对象，不要解释。
caption、activity、place、event_type 是必须同时输出的自然语言观察字段；
即使能够选择 semantic，也不能只输出 semantic 选择。
画面能判断时不要留空；确实看不清才用空数组或空字符串。

字段固定为：
caption、activity、place、scene_type、semantic、people、objects、
clothing、emotions、spatial_relations、ocr_text、event_type、facts。

semantic.place.primary 只能选择地点主类，details 从图片可观察的地点细节中多选；
semantic.objects 是物品记录数组，每项包含 primary、label、details；
semantic.atmosphere.labels 和 details 都是可观察画面氛围的多选值，不描述人物心理。

不要把来源成员当成画面人物，也不要推测拍摄者姓名；
source_owner 只作为事件来源候选。
metadata: {...}
```

字段限制：

- `caption` 最多 20 字。
- `activity`、`place`、`event_type` 各最多 10 字。
- `people`、`objects`、`clothing`、`emotions`、`spatial_relations` 各最多 2 项。
- `facts` 最多 1 项。
- `ocr_text` 最多 20 字。

### 6.3 描述恢复提示词

如果首轮只有语义选择，没有自然语言描述，则追加一次恢复请求：

```text
首轮图片结果只有分类或为空，请补齐可验证的自然语言观察。
只根据图片，不猜测姓名，不输出坐标。

严格返回简体中文 JSON：
caption（图片中看到什么，20字内）、
activity（正在发生什么，10字内）、
place（语义地点描述，如家中客厅/餐厅/公园，10字内）、
event_type（10字内）、people（最多2项）、objects（最多4项）、
ocr_text（20字内）。

画面确实看不清才留空；不要只返回分类字段。
```

恢复结果只补充空字段。已有非空的受控语义选择优先保留；如果首轮物品语义为空而恢复结果有顶层 `objects`，恢复物品会投影到语义物品层。

### 6.4 中文规范化提示词

当输出包含明显拉丁文本时，追加一次中文规范化请求：

```text
把下面的家庭图片观察规范化为简体中文 JSON。
只翻译和整理已有内容，不新增人物、物体、活动或事实，不猜测姓名。
保留字段 caption、activity、place、scene_type、semantic、people、objects、
clothing、spatial_relations、ocr_text、event_type、facts。
semantic 必须保留地点主类、地点细节、物品记录和可观察画面氛围。
scene_type 必须保留为受控地点主类之一。
原始观察：{parsed_json}
```

### 6.5 输出规范化

模型 JSON 经过以下确定性处理：

1. 标量字段转换为字符串。
2. 数组字段统一为空数组或列表。
3. 置信度支持数字、百分比和中文等级。
4. 受控地点主类通过别名和关键词归一化。
5. 受控物品主类、标签和细节分层保存。
6. 情绪字段只在能映射到氛围主类时进入氛围层。
7. 无法进入词表的词进入 `raw_labels`，不被静默丢弃。

## 7. 语义分类体系

### 7.1 地点主类

```text
居住空间、餐饮空间、商业空间、公园与花园、滨水空间、山地与自然景观、
街道与广场、交通空间、文化与展览、运动与休闲、演出与活动、办公与学习、
医疗与公共服务、宗教与纪念、工业与工程、住宿空间、农场与乡村、其他或不确定
```

地点细节包括：

```text
室内、室外、客厅、卧室、厨房、阳台、门口、正餐、咖啡或茶、烘焙、
有餐桌、露天座位、商场、超市、市场或摊位、展厅、舞台、候车区、
酒店房间、多人停留、开放空间、自然环境
```

GPS 处理：

```text
Asset.captured_location -> Event 地理锚点
                       -> Place Entity.geo
图片 semantic.place.primary -> Place Entity.canonical_name
图片 place 自由描述 -> Observation 证据/raw_labels
```

因此 GPS 坐标不会成为用户看到的地点名称。

### 7.2 物品主类

```text
食品与饮品、餐具与容器、电子设备、家具与家居、服饰与配件、植物与花卉、
动物与宠物、交通工具、建筑与公共设施、文字与标识、玩具与娱乐、书籍与文具、
礼物与纪念物、其他或不确定
```

物品记录结构：

```json
{
  "primary": "食品与饮品",
  "label": "蛋糕",
  "details": ["桌面"]
}
```

### 7.3 氛围主类

```text
温馨、热闹、轻松、平静、安静、活跃、忙碌、庄重、节庆、自然开阔、其他或不确定
```

氛围细节包括明亮、昏暗、暖色光线、冷色光线、空间开阔、空间拥挤、多人聚集、少人、画面整洁、画面繁杂、庆祝活动、观看活动、休息状态和自然光。

氛围只描述画面呈现，不等于人物心理，也不直接生成情绪事实。

## 8. 事件形成与批次总结

### 8.1 事件候选

每条 Observation 完成快速证据后调用 `merge_observation_into_event()`：

1. 读取原始拍摄时间。
2. 读取原始 GPS/地点锚点。
3. 读取视觉地点、活动和事件类型。
4. 读取对象、视觉向量和已确认人物桥接。
5. 对候选事件计算时间、地理、视觉、语义和人物重叠分数。
6. 选择候选或创建 `待总结事件`。
7. 写入 `event_observations`。
8. 重新选择该事件的原图封面。

事件聚类和事件总结是两个不同阶段。事件聚类决定哪些 Observation 属于同一事件；事件总结只对已经确定成员的 Observation 描述进行归纳。

### 8.2 批次内事件总结

带 `batch_id` 的图片处理完成后：

```text
Observation 完成
  -> Event 成员关系完成
  -> 不调用 summarize_event
  -> 资产进入 processed/failed

batch_complete
  -> 等待批次资产全部终态
  -> 找到批次涉及的 event_id
  -> 每个 event_id 调用一次 summarize_event
  -> 用该事件全部 Observation 描述构造 JSON
  -> 更新 Event
  -> 生成事件文本向量
```

事件总结输入是所有成员 Observation 的结构化文字，不是某一张图片：

```json
{
  "time_start": "...",
  "time_end": "...",
  "place": "...",
  "observations": [
    {
      "observation_id": "obs_1",
      "caption": "...",
      "activity": "...",
      "people": [],
      "objects": [],
      "ocr_text": "...",
      "clothing": [],
      "spatial_relations": []
    }
  ]
}
```

当前事件总结提示词：

```text
你是家庭事件总结器。下面是一组已经按拍摄时间和地点聚类的图片观察。
只能使用给定观察，不得把元数据地点以外的信息当作事实，
不得猜测未确认人物姓名；如果观察彼此不足以支持具体事件，
使用保守、描述性的标题。

严格返回简体中文 JSON：
title（不超过20字）、event_type、activity、
summary（包含时间地点范围和可验证活动）、confidence。
事件：{event_json}
```

批次完成接口具有幂等状态迁移。并发完成回调只有一个连接能把 `complete` 领取为 `summarizing`，避免同一事件被重复总结。

不带批次的旧导入继续即时总结，供旧客户端兼容；网页相册导入统一使用批次接口。

## 9. 语义实体投影

`maintain_observation_entities()` 读取规范化 Observation 和原始 Gamma 结果：

1. Place：使用语义地点主类作为稳定名称。
2. Object：使用物品 `label`，主类和细节保存为语义属性。
3. Atmosphere：使用氛围主类，保留氛围细节。
4. Time：由拍摄日期生成日期实体。
5. 每个实体通过 `entity_observations` 绑定 Observation。
6. 每个 Place 的 GPS 写入 `entity_properties(property_key='geo')`。
7. 每个属性保存来源、置信度、证据 ID、版本和 supersedes 关系。

人物实体只由人脸候选和用户确认流程产生；图片中的“一个成年人”“孩子”等描述不能自动成为命名人物。

## 10. 语义声明与人物扩展

图片观察可以为已确认人物生成事件级语义投影：

```text
confirmed Person
  -> entity_mentions
  -> event_participants
  -> person_event_memory
  -> person_patterns
  -> semantic_profiles
  -> semantic_claims
```

这些投影必须保留：

- 支撑 Observation ID。
- 支撑 Event ID。
- 置信度。
- 来源模型或人工来源。
- 声明版本和 superseded 状态。

场景级衣物不直接成为人物衣物事实。只有确认人物后，从该人物关联事件中选择高质量人脸并生成目标人物裁剪，才会进入 `person_appearance_evidence`。

## 11. 向量化处理

当前没有使用外部向量数据库。

### 11.1 图像向量

```text
原图 -> CLIP ViT-B/32 encode_image -> 512 维向量
     -> L2 normalize
     -> memory_vectors(space='visual', source_type='asset')
```

### 11.2 Observation 文本向量

文本由以下字段拼接：

```text
caption + activity + place + ocr_text + transcript + clothing + facts
```

再通过 CLIP `encode_text`：

```text
-> episodic / observation
-> semantic / observation
```

### 11.3 Event 文本向量

事件总结后拼接：

```text
title + event_type + activity + summary
```

再写入：

```text
memory_vectors(space='episodic', source_type='event')
```

### 11.4 SQLite 向量表

`memory_vectors` 主要字段：

```text
scope_id
space
source_type
source_id
vector_json
model_name
metadata_json
created_at
updated_at
```

唯一约束为：

```text
(space, source_type, source_id, model_name)
```

向量写入使用 upsert，不会因为同一 Observation 更新而无限新增重复向量。当前检索实现读取同一空间向量并在 Python 中计算余弦相似度；这不是当前图片导入速度的主要瓶颈，但数据量扩大后需要本地向量索引。

## 12. SQLite 记忆存储

数据库使用 SQLite WAL 模式和独立连接：

```text
MemorySpace
  -> assets
  -> observations
  -> face_instances / face_clusters / face_prototypes
  -> events / event_observations / event_participants
  -> entities / entity_observations / entity_properties
  -> semantic_profiles / semantic_claims
  -> memory_vectors
  -> rebuild_runs / revisions / feedback
```

图片处理后台任务为每个任务创建自己的 SQLite 连接，避免共享请求连接导致事务嵌套错误。数据库开启 `foreign_keys=ON`，结构迁移是加法迁移。

原始媒体仍保存在文件系统，数据库只保存路径、哈希和派生记录。所有派生删除和重建都不应删除原始来源目录。

## 13. 失败处理与可重建性

### 13.1 单图失败

任意处理异常时：

1. 删除该 Asset 产生的 Observation、Face、Event projection、Vector 和实体观察关系。
2. Asset 标记为 `failed`。
3. 在 `metadata_json.error` 保存错误。
4. 后续可通过维护重试。

### 13.2 模型输出失败

- JSON 解析失败时使用空对象并进入默认规范化。
- 图片描述为空时执行一次恢复提示词。
- 中文规范化失败不应伪造新事实。
- 事件总结失败时保留已有事件和 Observation 描述，事件不会被删除。

### 13.3 重建

批量重建命令：

```bash
cd /home/asus/Github/Sentrix-Home-Web
.venv/bin/python scripts/maintenance/rebuild_memory.py \
  --root . \
  --benchmark-manifest data/household-benchmark-manifest.json
```

该脚本会：

1. 初始化或清空派生数据库。
2. 从 manifest 读取三个 MemorySpace。
3. 只导入 manifest 指定的原图和允许的时间/GPS来源。
4. 使用 `pipeline.process(..., summarize_event=False)` 处理图片。
5. 对每个作用域执行事件合并。
6. 对每个事件执行一次文字事件总结。
7. 清理无证据事实。
8. 执行人脸全局重聚类。
9. 写入 `rebuild_runs` 统计和失败信息。

重建前必须使用 SQLite backup API 保存一致性备份，并确认目标 scope、输入 manifest 和 FMA 状态。

## 14. 当前性能分析与优化

### 14.1 已确认瓶颈

`album1` 历史 62 张已处理图片的处理时间统计：

```text
Gemma vision mean   19.232s
Gemma vision median 20.345s
Face mean            0.146s
CLIP image mean      0.202s
```

主要瓶颈是 Gemma 多模态推理，不是 SQLite、Face 或 CLIP。

### 14.2 已实施优化

1. Gemma 专用 Ollama 使用 `11435`，不触碰共享 `11434`。
2. `gemma4:12b` 使用 `keep_alive=-1` 常驻，避免每次请求重复加载模型。
3. 图片输入压缩至 896px，限制上下文 4096 和输出 320 token。
4. `think=false` 和 temperature 0，减少无关推理。
5. Face、CLIP、Gamma 图片级任务并行。
6. 批次内不对每一张图片重复总结同一个事件；批次完成后每个事件只总结一次。
7. Asset 使用 SHA-256 去重，重复导入不重复建立原始 Asset。
8. 向量使用 upsert，Observation 更新不会无界增殖向量。

### 14.3 后续可测优化

这些优化不能在没有质量基准的情况下直接启用：

1. 对 Gamma 图片任务测试并发度 1、2、3、4，选择真实吞吐最高的配置。
2. 测试 `896px`、`768px`、`672px` 对地点、物品、氛围和 caption 的影响。
3. 测试 `num_predict=160/192/320` 的完整字段率和速度。
4. 将完整相册处理从 FastAPI 临时后台任务迁移到有界持久队列。
5. 对 `sha256 + model + prompt_version` 建立语义缓存。
6. 事件数量增大后增加本地向量索引，避免 JSON 全表扫描。
7. 修复 NVIDIA NVML 用户态与内核版本不一致，建立更可靠的 GPU 监控。

优化验收必须同时观察：

- 每张图片处理时延。
- 整个相册墙钟时间。
- Gamma 超时率。
- 图片自然语言字段非空率。
- 地点/物品/氛围受控词表命中率。
- 事件过度合并和过度拆分。
- Observation、Entity、Event、Vector 的数量和证据完整性。

## 15. 重建验收标准

每个相册必须满足：

1. manifest 文件数、成功数、失败数可解释。
2. 所有成功 Asset 状态为 `processed`。
3. 每个成功 Asset 至少有一条 Observation。
4. 每条 Observation 的 `raw_json` 和 `canonical_json` 可解析。
5. 图片可判断时 `caption`、`activity`、`place`、`event_type` 不能全部为空。
6. 地点实体名称不能是 GPS 坐标。
7. GPS 只能出现在 Asset 地理字段或 Place 的 `geo` 属性。
8. 物品实体至少有主类、标签或受控细节之一。
9. 氛围实体只使用受控氛围主类或明确的其他类。
10. 每个事件的 Observation 数量与 `event_observations` 一致。
11. 每个事件总结只调用一次批处理总结路径，且输入包含该事件全部 Observation 描述。
12. 视觉、Observation 文本和事件向量均有正确 source 指针。
13. 人脸簇和人脸实例的模型版本、质量、证据 ID 完整。
14. `PRAGMA integrity_check` 返回 `ok`。
15. 重建状态为 `completed`，或每个失败项都有明确可重试原因。

