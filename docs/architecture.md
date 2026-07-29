# Sentrix Home 初版架构契约

## 记忆边界

### 事件记忆

图片、音频、文本和未来的视频共同写入。事件对象必须包含时间范围、参与者、地点、活动、证据 ID、置信度和修订版本。

### 语义记忆

图片、音频、文本和未来的视频共同写入。维护 Person、Place、Relationship、Habit 和 Fact。模型抽取不是最终事实，必须保留来源和人工修订记录。

### 视频编码记忆

视频独享。第一版只建立原始 Asset 和导入状态，不进行镜头切分、动作抽取、视频向量或时间戳视觉检索。接口名固定为 `video_memory_adapter`，后续替换实现不应改变事件和证据 ID。

## 后端数据流

```text
Asset
  -> Observation
  -> Event Candidate
  -> Canonical Event
  -> Semantic Fact
  -> Sentrix native semantic graph and vector index
  -> Agent retrieval trace
```

## 接口约定

### `POST /api/search`

请求：

```json
{ "query": "去年春节全家去了哪里？", "spaceId": "home-default" }
```

响应：

```json
{
  "answer": "...",
  "confidence": 0.86,
  "memories": ["episodic", "semantic", "visual-evidence"],
  "retrievalTrace": [],
  "evidence": []
}
```

### `POST /api/import`

网页通过 multipart 上传真实文件，后端保存原始 Asset 并返回稳定的 `assetId`。所有详情页面通过 Asset、Observation 和 Event ID 回到原始证据。

```json
{ "fileName": "birthday.m4a", "mediaType": "audio" }
```

视频响应必须支持：

```json
{ "assetId": "asset_x", "status": "video-extraction-reserved" }
```

## 153 模型适配

模型具体加载方式不固化在网页端。后端通过适配器访问：

- `gamma4_12B`：事件抽取、语义事实归纳、Agent 回答；
- `FunASR`：音频转写、VAD、标点和时间片段；
- 人脸模型：人脸实例、候选聚类和人物确认任务。

所有适配器都需要返回 `modelName`、`modelVersion`、`confidence` 和 `sourceIds`。
