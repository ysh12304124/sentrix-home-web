# Sentrix 人物中心记忆实现计划

## 目标

把当前 Sentrix 从“图片观察 + 扁平事实”升级为人物中心的家庭记忆系统，支持跨家庭成员设备的事件聚合、人物确认后的全量事件重建、中文规范化观察、语义画像和 Agent 记忆补全。

## 架构

- `backend/db.py`：SQLite 权威存储，增加资产来源、事件角色、语义画像、语义声明、查询缺口和反馈。
- `backend/pipeline.py`：从 Asset 生成结构化中文 Observation，按时间/地点/来源/内容合并到跨设备 Event。
- `backend/model_clients.py`：Gemma 中文结构化抽取、针对衣物/物品/空间关系的专项视觉抽取；buffalo_l 质量过滤和 embedding。
- `backend/agent.py`：人物、事件、语义声明、原始资产多阶段检索；证据不足时触发专项分析并写回记忆。
- `backend/app.py`：人物、事件角色、语义知识、反馈和补全 API。
- `src/app.js`、`src/api.js`、`styles.css`：人物独立页面、语义知识页面、事件角色和证据交互。
- `scripts/rebuild_memory.py`：从原始资产重新建立新版本派生记忆。

## 实施任务

### 1. 固定失败测试和数据契约

文件：

- `backend/tests/test_memory_store.py`
- `backend/tests/test_entities.py`
- `backend/tests/test_agent.py`
- `backend/tests/test_pipeline.py`

新增测试覆盖：

- 资产保存 `source_owner_id` 和来源设备；
- 事件可以保存 `captured_by`、`celebrant`、`attendee` 等角色；
- 人物确认会更新 Observation、Event、SemanticProfile 和 SemanticClaim；
- 无可见人物的蛋糕图片仍能通过来源成员进入生日事件；
- 冲突语义声明保留旧版本；
- Agent 在模型拒答时回退原始资产并保存 QueryGap；
- 中文规范化结果拒绝英文值。

验证命令：

```bash
python3 -m unittest discover -s backend/tests -v
node --test
```

预期：新增测试先失败，说明测试确实锁定了未实现行为。

### 2. 扩展原生 SQLite 记忆模型

文件：`backend/db.py`

新增表：

- `event_participants(event_id, person_id, role, evidence_ids_json, confidence, revision)`；
- `semantic_profiles(person_id, summary_zh, activity_summary_zh, place_summary_zh, appearance_summary_zh, preference_summary_zh, revision)`；
- `semantic_claims(person_id, dimension, predicate, value_text, value_entity_id, valid_from, valid_to, supporting_event_ids_json, evidence_ids_json, status, confidence, supersedes_claim_id, revision)`；
- `query_gaps(query, missing_dimension, candidate_asset_ids_json, resolution, evidence_ids_json, status, created_at)`；
- `memory_feedback(query_gap_id, user_id, accepted_answer, correction, target_claim_id, created_at)`。

扩展 `assets`：`source_owner_id`、`source_device_id`、`source_album_id`、`source_confidence`、`captured_at`、`captured_location`。

扩展 `observations`：`canonical_json`、`source_owner_id`、`inferred_captured_by`、`clothing_json`、`spatial_relations_json`、`revision`。

实现方法：

- 增加 `upsert_event_participant`、`list_event_participants`；
- 增加 `upsert_semantic_profile`、`maintain_semantic_claim`、`list_person_knowledge`；
- 所有声明按照 `(person_id, dimension, predicate, value)` 合并，冲突时创建新版本并保留 `supersedes_claim_id`；
- 从普通实体列表中排除 `entity_type=person` 的展示结果；
- 保留原始字段，不删除旧证据。

验证命令：`python3 -m unittest discover -s backend/tests -v`。

### 3. 改进人脸质量、全局聚类和身份回写

文件：

- `backend/model_clients.py`
- `backend/db.py`
- `scripts/rebuild_memory.py`
- `scripts/evaluate_lfw_clusters.py`

实现方法：

- `FaceAdapter.detect` 返回脸尺寸、姿态、清晰度和质量分数；
- 过滤低于最小脸尺寸、检测置信度和质量阈值的实例；
- 重建脚本先收集全部脸 embedding，再执行全局聚类；
- 使用高质量 medoid 更新 `face_clusters`，不使用输入顺序相关的在线均值；
- LFW 评估输出纯度、singleton 比例、同人召回、异人误合并和 F1；
- 确认、拒绝、合并和拆分都写入 `entity_revisions`；
- 确认人物后调用统一的 `rebuild_person_memory(person_id)`，更新所有 Observation、EventParticipant、Event 摘要和 SemanticProfile。

验证命令：

```bash
python3 -m unittest discover -s backend/tests -v
python scripts/evaluate_lfw_clusters.py
```

### 4. 实现来源成员和跨设备事件构建

文件：

- `backend/pipeline.py`
- `backend/db.py`
- `backend/app.py`

实现方法：

- 导入时保存 multipart 的 `source_owner_id`、`source_device_id` 和 EXIF 时间/GPS；
- Event 匹配综合时间窗口、地点距离、活动类型、人物重合、来源成员、物体和 OCR；
- 不把一张图片强制当成一个事件；
- 图片无可见人物时保留空 `visible_people`，但通过来源成员和已存在事件建立关系；
- 将来源成员转换成 `captured_by` 候选角色，置信度随设备归属和用户确认变化；
- 事件摘要使用中文结构化参与者和角色重新生成；
- 视频仍返回 `video-extraction-reserved`。

验证命令：运行事件角色、跨设备生日和无人物图片测试，并通过 `GET /api/events/{id}` 检查所有 Asset/Observation 证据。

### 5. 重构中文图像观察和 Agent 补全

文件：

- `backend/model_clients.py`
- `backend/pipeline.py`
- `backend/agent.py`
- `backend/app.py`

实现方法：

- Gemma 图像 Prompt 固定输出中文结构化字段：人物、衣物、物体、活动、地点、空间关系、OCR；
- 对包含大量英文或不符合字段契约的结果执行一次中文规范化；
- Agent 首先检索 Person、SemanticClaim、Event 和原始 Observation；
- 发现缺失 `clothing`、`object` 或 `spatial_relation` 时调用专项图像分析；
- 专项分析写入同一 Observation 的版本化 canonical 字段和视觉向量；
- 创建 QueryGap，保存缺失维度和候选证据；
- 用户反馈只修订指定 Claim，不直接修改原始模型结果；
- 返回答案时总是带 Asset、Observation、Event 和 Claim 证据。

验证命令：

```bash
python3 -m unittest discover -s backend/tests -v
curl -X POST http://127.0.0.1:8090/api/search -H 'content-type: application/json' -d '{"query":"某天妈妈穿的蓝色外套"}'
```

### 6. 拆分人物和语义知识网页

文件：

- `src/app.js`
- `src/api.js`
- `styles.css`
- `backend/app.py`

新增页面和接口：

- `GET /api/people`：只返回人物及确认状态；
- `GET /api/people/{id}/profile`：人物画像、声明、事件和证据；
- `GET /api/knowledge?person_id=`：人物关联的活动、地点、物品、衣物、偏好和习惯；
- `GET /api/events/{id}`：事件角色和全部原始证据；
- `POST /api/query-gaps/{id}/feedback`：保存用户反馈。

网页要求：人物、事件、语义知识三个入口独立；语义页显示声明来源、置信度、有效时间、修订状态和证据；事件页按角色显示人物，图片只有物体时显示来源成员而不伪造画面人物。

验证命令：`node --test`，并通过 4174 端口请求主页、health、people、knowledge、events 和 search。

### 7. 空库重建和 153 验收

执行顺序：

1. 停止 Sentrix 后端；
2. 删除 `data/sentrix.db` 和派生媒体目录；
3. 用全量家庭相册重新摄取；
4. 重新执行 LFW 人脸聚类评估；
5. 启动 8090 后端和 4174 网页；
6. 验证数据库无 failed Asset、孤儿 Observation 和无证据 Claim；
7. 验证人物确认、事件角色、语义画像和 Agent 回退查询；
8. 保留 5173 FMA 服务不变。

最终验收：

- 家庭资产全部有来源成员或明确未知状态；
- 生日跨设备图片能够归并到一个事件；
- 无人物蛋糕图片能够绑定生日事件和拍摄者角色；
- 人物命名后事件摘要出现规范中文姓名；
- 语义知识页面显示人物画像和带证据的声明；
- 衣物缺失查询能够回退图片并生成下一次可检索的细节；
- 视频导入只返回 `video-extraction-reserved`。
