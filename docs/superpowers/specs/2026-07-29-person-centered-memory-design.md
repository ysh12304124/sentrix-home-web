# Sentrix 人物中心家庭记忆设计

## 目标

将 Sentrix 的记忆系统从“图片描述加零散事实”升级为人物中心的家庭记忆系统：

- 人物是语义记忆的主轴；
- 事件是跨家庭成员设备聚合的经历；
- 图片可以没有可见人物，但必须通过来源成员或事件上下文归属到至少一个人物；
- 用户确认人物后，所有相关观察、事件、事实和向量都重新规范化；
- Agent 在查询失败时回退到原始证据，触发针对性视觉补全，并将补全结果纳入后续检索；
- 所有面向用户的描述和语义值使用简体中文；
- 视频继续只保留接口，不在本阶段执行视频提取。

## 设计原则

### 人物中心

语义知识必须能沿着以下路径回到人物：

```text
Person -> Event -> Evidence
Person -> Activity / Place / Object / Habit -> Evidence
```

活动、地点、物品和习惯不允许成为没有人物或事件归属的孤立知识。底层可以继续使用统一的 `entities` 表，但 `entity_type=person` 的数据不在普通实体页展示，人物使用独立的 People 页面和 API。

### 事件与语义分工

- Episodic Memory 保存一次具体经历：时间、地点、人物、角色、行为、资产和证据。
- Semantic Memory 保存跨事件形成的长期人物知识：人物画像、活动模式、地点习惯、物品/衣物特征、偏好和明确关系。
- Visual Memory 保存原始图片、局部视觉细节、人脸实例、图像向量和补全分析结果。

### 证据优先

模型描述不是事实。任何人物、事件、属性和习惯都必须引用 `asset_id`、`observation_id`、`event_id` 或用户确认记录。原始模型结果不可变，规范化结果可以版本化更新。

## 领域模型

### Asset

资产保留原始文件和来源信息：

```text
Asset
- asset_id
- media_type
- original_uri
- captured_at
- captured_location
- source_owner_id       # 来自哪个家庭成员的手机/账户
- source_device_id
- source_album_id
- uploader_id
- source_confidence
- metadata
```

`source_owner_id` 是资产来源证据，不自动等同于画面中的人物。只有来源设备明确归属时，才可以高置信度推断 `captured_by`。

### Person

人物从普通实体列表中独立出来：

```text
Person
- person_id
- canonical_name
- aliases
- family_role
- identity_status
- face_cluster_ids
- source_owner_device_ids
- profile_summary_zh
- confidence
- revision
```

人物状态包括 `pending`、`confirmed`、`rejected`、`merged` 和 `split`。人脸簇只是身份候选，不等于人物。

### Observation

每个 Observation 分为原始观察和规范观察：

```text
Observation
- observation_id
- asset_id
- event_id
- raw_observation_json
- canonical_observation_json
- visible_people
- source_owner_id
- inferred_captured_by
- clothing
- objects
- activity
- place
- ocr_text
- confidence
- model_info
- revision
```

`visible_people` 只表示画面中看见的人；`source_owner_id` 和 `inferred_captured_by` 表示资产来源和拍摄角色，两者不能混用。

### Event

事件不再只保存字符串 participants，而使用角色化参与者：

```text
Event
- event_id
- title_zh
- event_type
- time_start
- time_end
- location
- summary_zh
- participant_roles
- evidence_asset_ids
- evidence_observation_ids
- confidence
- revision
```

```text
EventParticipant
- event_id
- person_id
- role
- evidence_ids
- confidence
```

角色包括 `celebrant`、`attendee`、`visible_subject`、`captured_by`、`organizer`、`gift_giver`、`speaker` 和 `mentioned_person`。

“共同出现在同一观察中”不再创建语义关系，只保存在事件参与者和共现证据中。

### SemanticProfile 与 SemanticClaim

语义层围绕 Person 组织，不再用大量互相孤立的三元组：

```text
SemanticProfile
- profile_id
- person_id
- summary_zh
- activity_summary_zh
- place_summary_zh
- appearance_summary_zh
- preference_summary_zh
- revision
- evidence_ids
```

```text
SemanticClaim
- claim_id
- person_id
- dimension          # activity/place/object/clothing/preference/habit/relationship
- predicate
- value_entity_id or value_text
- valid_from
- valid_to
- supporting_event_ids
- evidence_ids
- status
- confidence
- supersedes_claim_id
- revision
```

示例：

```text
人物：妈妈
维度：clothing
谓词：穿着
内容：蓝色外套
时间：2025-06-03
关联事件：event_x
证据：asset_x
```

家庭角色优先作为 Person 属性。只有夫妻、亲子、兄弟姐妹、照顾者等不能仅由角色推导的关系，才写入显式关系表。

## 跨设备事件构建

### 资产导入

每个家庭成员导入或同步相册时必须提供来源成员和设备身份。系统读取 EXIF 时间、GPS、设备和相册信息，并将上传时间与拍摄时间分开保存。

### 事件候选聚合

事件候选综合以下信号：

- 拍摄时间窗口；
- GPS 或地点距离；
- 来源相册和设备；
- 可见人物重合；
- 来源成员重合；
- 活动/物品/文本语义相似度；
- 图片视觉相似度；
- 音频和 OCR 的时间对齐；
- 用户已经确认的事件边界。

同一生日事件可以吸收多个成员设备上的照片、视频和音频。事件合并必须保留每个资产的来源成员和证据，不覆盖原始资产。

### 无人物图片

蛋糕、礼物、风景等图片可以没有 `visible_people`，但仍然可以属于一个有人物参与的事件：

```text
visible_people = []
source_owner = 儿子
event = 母亲生日
event_roles = 儿子: captured_by, 母亲: celebrant
```

如果没有来源成员、事件上下文或用户确认，系统只能生成“发现生日蛋糕”等中性观察，不能生成“儿子为母亲拍摄”。

## 人脸聚类与确认

### 聚类流程

1. buffalo_l 检测人脸并提取 ArcFace embedding；
2. 按脸尺寸、检测置信度、清晰度和姿态过滤低质量实例；
3. 同一资产内进行重复脸去重；
4. 对全量 embedding 执行全局聚类，避免按输入顺序在线合并；
5. 使用高质量 medoid 作为簇代表；
6. 用 LFW 验证集校准阈值，报告 precision、recall、F1 和 singleton 比例；
7. 低置信度簇进入用户确认队列；
8. 支持用户拆分、合并和撤销，并保留修订历史。

### 用户确认

用户确认人物时填写姓名和家庭角色。确认动作触发：

```text
FaceCluster
 -> Person
 -> FaceInstance / Observation
 -> EventParticipant
 -> Canonical Observation
 -> Event Summary
 -> Semantic Profile / Claim
 -> episodic / semantic / visual vectors
```

事件标题和摘要必须重新生成中文规范描述，不能只在原描述后追加名字。

## 图片理解与中文规范化

图片理解输出固定为中文结构化字段：

```text
caption
event_type
activity
place
visible_people
clothing
objects
food
spatial_relations
ocr_text
confidence
```

要求：

- 所有规范字段值使用简体中文；
- 不确定字段为空；
- 不能凭外观猜姓名和家庭关系；
- 英文或格式不合格结果不能直接进入语义层；
- 原始模型输出保留在 raw observation 中；
- 中文规范化结果单独版本化。

## Agent 查询与记忆补全

Agent 使用循环检索：

```text
问题解析
 -> 人物/时间/事件/属性识别
 -> SemanticProfile 检索
 -> Event 检索
 -> Visual / OCR / 原始 Asset 检索
 -> 证据充分性判断
 -> 必要时专项视觉重分析
 -> 更新 Observation / Event / SemanticClaim
 -> 返回答案和证据
```

当用户询问衣服、物品或画面细节而现有描述缺失时，Agent 触发专项分析，例如 `clothing`、`object` 或 `spatial_relation` extractor。专项分析结果必须引用原始资产，并经过置信度和用户反馈治理，不能直接把一次模型猜测变成长期事实。

系统保存：

```text
QueryGap
- query
- missing_dimension
- candidate_asset_ids
- resolution
- evidence_ids

MemoryFeedback
- query_id
- accepted_answer
- correction
- target_claim_id
- user_id
```

重复出现的查询缺口可以提升该维度的提取优先级，但不修改全局 Prompt 文本。

## 网页模块

- People：人物确认、头像样本、家庭角色、合并/拆分、人物画像和相关事件；
- Timeline：跨设备聚合后的事件，显示事件角色和全部证据；
- Semantic Knowledge：选择人物后显示活动、地点、物品、衣物、偏好、习惯和修订状态；
- Evidence：查看原始资产、Observation、模型版本、来源成员和时间地点；
- Agent：答案、检索轨迹、补全动作、原始证据和用户反馈。

普通实体列表只展示非人物概念，人物不再与活动、地点和物品混排。

## 数据迁移与重建

已有派生记忆不直接兼容新语义投影。迁移步骤为：

1. 保留原始 Asset 和原始媒体；
2. 删除旧 Observation、Event、Fact、Entity 和 Vector 派生数据；
3. 从 Asset 重新读取来源成员、EXIF 和模型配置；
4. 按新结构生成 Observation、Event、Person Candidate 和 SemanticClaim；
5. 重新生成全部向量；
6. 对旧用户确认记录进行 ID 映射；
7. 通过证据链和计数一致性检查后发布。

## 验收标准

- 每个资产都有来源成员或明确的未知来源状态；
- 事件可以吸收多个家庭成员设备的资产；
- 无人物图片可以正确归属到有人物参与的事件；
- 用户确认一个人物后，相关 Observation、Event、SemanticProfile 和向量都会更新；
- 人物页面与普通语义知识页面分离；
- 语义层不存在没有人物或事件归属的孤立知识；
- 用户查询衣物等细节时，Agent 能回退到原始图片并补充记忆；
- 所有规范化描述为中文；
- 每个结论都能回到原始证据；
- 视频只返回 `video-extraction-reserved`，不生成视频编码记忆。
