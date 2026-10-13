# 事件中心的稀疏时序图记忆改造方案

## 1. 文档目的

本文档描述如何在不偏离项目现有远程仓库主流程的前提下，改进图结构记忆的构建与检索，使系统更好地处理：

- 事件级问题；
- 先后顺序、后来、之前、之后等时序问题；
- 人物、地点、对象、事件之间的多跳问题；
- 图片与视频之间的跨媒体关联；
- 召回阶段的证据覆盖率和最终回答质量。

本方案的核心不是增加一套独立的时间节点，而是把已有的事件总结提升为图中的上层语义节点，把关键帧作为下层可验证证据，并通过少量有方向的时间边连接事件。

## 2. 参考论文与源码

实现时优先参考以下项目的相应部分，不要整体复制其数据模型或模型调用流程：

1. [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
   - Knowledge Graph + Personalized PageRank；
   - 实体作为检索入口；
   - 多实体激活和多跳路径传播。

2. [LightRAG](https://github.com/HKUDS/LightRAG)
   - 实体和关系归一化；
   - local/global/mix 检索模式；
   - 图检索和向量检索双通道融合。

3. [VideoRAG](https://github.com/HKUDS/VideoRAG)
   - 视频、片段和关键帧的层级组织；
   - 视频语义与视觉证据联合检索。

4. [MG²-RAG](https://github.com/Daboolu/MG2-RAG)
   - 多粒度节点；
   - entity-driven visual grounding；
   - Personalized PageRank 和多粒度证据汇聚。

5. [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
   - 实体/关系合并和层级组织思想。
   - 当前项目不建议直接引入完整社区摘要流程。

## 3. 当前结构与问题

当前项目已经具备以下基础：

- `FRAME -> CLIP -> VIDEO` 层级；
- `ENTITY` 节点；
- `TEMPORAL`、`SEMANTIC`、`ENTITY` 等关系；
- Visual ANN、Text ANN、Lexical、Metadata、Entity、Adjacency、Graph 等检索通道；
- weighted RRF 融合。

重点代码位置：

- `backend/graph_memory/keyframe_memory_builder.py`
- `backend/graph_memory/service.py`
- `backend/graph_memory/keyframe_query_engine.py`
- `backend/retrieval/graph_expander.py`
- `backend/retrieval/entity.py`
- `backend/retrieval/adjacency.py`
- `backend/retrieval/fusion.py`
- `backend/retrieval/ranking.py`

当前的关键问题：

1. 关键帧被当作 `EVENT` 使用时，事件总结和单帧观察容易混在一起；
2. `event_title`、`event_summary` 主要是帧文本字段，没有充分成为图上的共享父节点；
3. 图扩展依赖普通检索先产生 seed，初始 ANN 失误时图无法独立补召回；
4. 时间信息主要是属性或简单邻接，缺少事件级的有方向时序关系；
5. 人物、地点、对象的别名和同义表达可能使路径断裂；
6. 目前缺少稳定的 `事件 -> 多帧证据` 和 `事件 -> 事件` 多跳路径。

## 4. 总体目标结构

### 4.1 两层事件模型

不新增冗余的时间节点，使用两层事件语义：

```text
EVENT_SUMMARY       # 跨关键帧的完整事件
FRAME_EVENT         # 单个关键帧代表的局部观察
```

推荐图结构：

```text
VIDEO
  └── CLIP
        └── EVENT_SUMMARY
              ├── FRAME_EVENT_1
              ├── FRAME_EVENT_2
              └── FRAME_EVENT_3

PERSON ────────> EVENT_SUMMARY
PLACE ─────────> EVENT_SUMMARY
OBJECT ────────> FRAME_EVENT
```

如果现有 `NodeType` 不适合立即新增枚举，第一版使用已有节点类型，并增加逻辑属性：

```json
{
  "subtype": "event_summary"
}
```

关键帧节点使用：

```json
{
  "subtype": "frame_event"
}
```

节点 ID 必须稳定、可重复构建，不能使用每次随机生成的 UUID 作为事件聚合键。

### 4.2 EVENT_SUMMARY 节点

事件总结节点建议包含：

```json
{
  "event_id": "event_wedding_001",
  "subtype": "event_summary",
  "event_title": "河北户外婚礼",
  "event_summary": "多人在河北参加户外婚礼仪式",
  "date": "2017-10-04",
  "start_time": 120.5,
  "end_time": 245.8,
  "duration": 125.3,
  "video_uid": "video_001",
  "clip_uid": "clip_001",
  "frame_ids": ["frame_001", "frame_002", "frame_003"],
  "person_ids": ["person_mingming"],
  "place_ids": ["place_hebei"],
  "confidence": 0.91
}
```

时间字段使用事件区间，不单独建立每个时间点的节点。

### 4.3 FRAME_EVENT 节点

关键帧节点仍然是最终视觉验证的最小证据单元：

```json
{
  "frame_id": "frame_001",
  "subtype": "frame_event",
  "video_time_sec": 132.4,
  "clip_time_sec": 12.4,
  "event_id": "event_wedding_001",
  "caption": "婚礼现场有人站在舞台旁边",
  "objects": ["舞台", "鲜花"],
  "persons": ["person_mingming"]
}
```

关键帧不可被事件摘要替代，因为回答最终仍需要具体图片或视频证据。

## 5. 边类型设计

### 5.1 事件归属边

```text
FRAME_EVENT -> BELONGS_TO -> EVENT_SUMMARY
EVENT_SUMMARY -> CONTAINS -> FRAME_EVENT
```

实现时只需保留一个物理方向，另一方向通过图查询反向访问即可，避免重复存储。

### 5.2 层级边

保留现有层级逻辑：

```text
FRAME_EVENT -> CLIP -> VIDEO
EVENT_SUMMARY -> CLIP -> VIDEO
```

如果当前图模型只允许原有的 `FRAME -> CLIP -> VIDEO`，事件节点可以额外挂到 `CLIP`，但不能删除原有层级边。

### 5.3 实体边

```text
PERSON -> PARTICIPATED_IN -> EVENT_SUMMARY
PLACE -> OCCURRED_AT -> EVENT_SUMMARY
EVENT_SUMMARY -> HAS_PERSON -> PERSON
EVENT_SUMMARY -> HAS_PLACE -> PLACE
OBJECT -> OBSERVED_IN -> FRAME_EVENT
```

实体节点应带有：

```json
{
  "canonical_id": "person_mingming",
  "entity_type": "PERSON",
  "aliases": ["明明", "明明姐"],
  "confidence": 0.9
}
```

### 5.4 稀疏帧级时序边

帧级只建立相邻关键帧边：

```text
FRAME_1 -> NEXT -> FRAME_2 -> NEXT -> FRAME_3
```

边属性：

```json
{
  "delta_seconds": 2.5,
  "same_event": true,
  "same_clip": true,
  "confidence": 1.0
}
```

不建立所有帧之间的 `BEFORE` 边，不建立帧级全连接图。

### 5.5 事件级时序边

同一视频中按时间排序的相邻事件之间建立：

```text
EVENT_A -> TEMPORAL_ORDER -> EVENT_B
```

边属性：

```json
{
  "relation": "before",
  "delta_seconds": 3600.0,
  "same_video": true,
  "confidence": 1.0
}
```

只保存一个有方向的 `TEMPORAL_ORDER`，查询时根据方向解释为 before/after，不同时存储 `BEFORE` 和 `AFTER`。

`SAME_DAY` 直接使用日期字段过滤，不建立全图日期边。

### 5.6 跨媒体边

当图片和视频在人物、地点、日期或事件上有足够证据时，建立：

```text
IMAGE_FRAME -> RELATED_TO -> VIDEO_CLIP
IMAGE_FRAME -> SAME_EVENT -> EVENT_SUMMARY
```

每条跨媒体边必须记录来源和置信度：

```json
{
  "source": "shared_person_place_date",
  "confidence": 0.84,
  "evidence": ["person_x", "河北", "2017-10-04"]
}
```

## 6. 事件构建流程

### 6.1 事件 ID 优先级

构建事件时按照以下顺序确定聚合键：

1. 使用数据中已有的稳定 `event_id`；
2. 使用 `video_uid + clip_uid + 标准化 event_title`；
3. 使用 `video_uid + 时间区间 + 事件类型`；
4. 仅在没有事件信息时，使用轻量规则聚类。

不允许每个关键帧都随机生成一个新的事件总结。

### 6.2 轻量事件聚类

没有现成事件 ID 时，事件归属可以使用：

- 时间接近；
- 人物重合；
- 场景重合；
- 对象重合；
- 文本标题/摘要相似；
- 同一视频或 clip。

可使用以下可解释分数：

```text
event_affinity =
    0.35 * person_overlap
  + 0.25 * scene_overlap
  + 0.20 * object_overlap
  + 0.15 * text_similarity
  + 0.05 * temporal_proximity
```

聚类阈值必须配置化，并在小规模数据上验证，避免把整段视频错误合并为一个事件。

### 6.3 事件时间范围

事件的时间范围由其子帧计算：

```text
start_time = min(frame.video_time_sec)
end_time   = max(frame.video_time_sec)
```

如果数据已有事件时间范围，优先使用数据字段，并保留来源信息。

## 7. 图检索流程

### 7.1 普通问题

```text
实体/关键词
  -> EVENT_SUMMARY
  -> FRAME_EVENT
  -> 具体视觉证据
```

例如“明明参加过哪些婚礼？”先命中人物和事件，再下钻到关键帧。

### 7.2 多跳问题

```text
PERSON
  -> EVENT_SUMMARY_A
  -> TEMPORAL_ORDER
  -> EVENT_SUMMARY_B
  -> PLACE
  -> FRAME_EVENT
```

例如“明明参加婚礼之后去了哪里？”应优先进行有方向的事件级遍历。

### 7.3 视频内时序问题

```text
FRAME_EVENT_A
  -> NEXT
  -> FRAME_EVENT_B
  -> OBJECT
```

例如“他后来拿起了什么？”使用相邻帧和对象关系，而不是只依赖图像向量相似度。

### 7.4 图传播

参考 HippoRAG 的 Personalized PageRank，但要做成有界、关系感知的传播：

- 最大跳数：2~3；
- 最大候选节点数：配置化；
- 最大耗时：配置化；
- 只遍历与查询意图匹配的边；
- 图失败时不能阻断 ANN/FTS 主流程。

不同查询类型使用不同边权：

```text
时间问题：TEMPORAL_ORDER / NEXT 权重高
人物问题：PARTICIPATED_IN / HAS_PERSON 权重高
地点问题：OCCURRED_AT / HAS_PLACE 权重高
多跳问题：EVENT_SUMMARY / ENTITY / TEMPORAL_ORDER 权重高
```

候选图分数可以采用：

```text
graph_score =
    0.30 * pagerank_score
  + 0.20 * relation_match
  + 0.20 * temporal_match
  + 0.15 * entity_match
  + 0.10 * path_diversity
  + 0.05 * source_confidence
```

具体权重必须通过消融实验调整，不得直接修改 ground truth 或评分公式。

## 8. 证据输出

图检索返回的候选必须保留可解释路径：

```json
{
  "asset_id": "asset_001",
  "path": [
    {"node": "person_mingming", "type": "PERSON"},
    {"edge": "PARTICIPATED_IN"},
    {"node": "event_wedding_001", "type": "EVENT_SUMMARY"},
    {"edge": "CONTAINS"},
    {"node": "frame_001", "type": "FRAME_EVENT"}
  ],
  "path_count": 2,
  "relation_match": 1.0,
  "temporal_match": 1.0,
  "source_confidence": 0.91
}
```

证据压缩要求：

- 同一事件只保留少量代表帧；
- 同一视频的重复近邻去重；
- 保留不同路径支持的候选；
- 不因去重删除唯一能证明时间关系的帧；
- 不改变现有 evidence API 的兼容字段。

## 9. 实施顺序

### 阶段 A：只增加事件父节点

- 从已有 `event_id/event_title/event_summary` 构建 `EVENT_SUMMARY`；
- 将多个关键帧挂到同一事件；
- 保留所有旧节点和旧边；
- 增加构建统计：事件数、平均帧数、孤立帧数。

### 阶段 B：增加稀疏时序边

- 关键帧建立相邻 `NEXT`；
- 事件建立相邻 `TEMPORAL_ORDER`；
- 时间问题才启用时序遍历；
- 普通查询不增加额外时间图开销。

### 阶段 C：增加实体归一化

- canonical entity ID；
- 中文别名和同义词；
- 人物、地点、对象、事件类型分层；
- 实体到事件的直接路径。

### 阶段 D：增加关系感知图传播

- 实体直达召回；
- 2~3 跳 Personalized PageRank；
- 查询意图对应的边权；
- 与现有 ANN/FTS 使用 weighted RRF 融合。

## 10. 兼容性要求

- 不删除 Visual ANN、Text ANN、Lexical、Metadata、Entity、Adjacency；
- 不改变现有 retrieval metric 定义；
- 不修改 QA ground truth 和文件映射；
- 不通过硬编码测评问题提高分数；
- 不引入完整 GraphRAG 社区摘要流程；
- 所有新增功能提供配置开关；
- 默认关闭实验性行为时，旧流程应保持可运行；
- 图构建失败不能阻断普通检索和回答；
- 所有节点 ID 和边 ID 可重复生成，避免重复建图。

## 11. 验收与消融实验

在相同数据、相同模型、相同 top-k 和相同并发配置下比较：

1. baseline；
2. baseline + 事件父节点；
3. baseline + 事件父节点 + 稀疏时序边；
4. baseline + 实体归一化；
5. baseline + Personalized PageRank；
6. full graph improvement。

每组必须记录：

- `retrieval_recall_mean`；
- `retrieval_precision`；
- `retrieval_f1`；
- `answer_quality_mean`；
- `exact_accuracy`；
- `core_accuracy`；
- `judge_valid_count`；
- `failed_count`；
- 延迟；
- 图节点数、边数、平均路径长度；
- 各类问题的分项召回率：普通、人物、地点、对象、时序、多跳、跨媒体。

先完成 487 QA 回归，确认没有明显退化，再使用复用的 487 memory 运行 1167 QA 全量测评。

## 12. 成功标准

只有同时满足以下条件才认为改造成功：

- 时序问题的召回率相对 baseline 提升；
- 多跳问题的召回率相对 baseline 提升；
- 普通图片问题不出现明显退化；
- `answer_quality_mean` 可正常计算，Judge 不再大面积未评分；
- 图检索失败不会拖垮 ANN/FTS 主流程；
- 487 QA 回归无明显功能回归；
- 1167 QA 全量测评完成并保存可复现结果。
