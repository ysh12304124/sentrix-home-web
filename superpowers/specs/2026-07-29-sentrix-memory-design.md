# Sentrix Home 独立记忆架构设计

## 目标

在 153 上建立不依赖 FMA 业务代码的本地优先家庭多模态记忆系统。第一版真实支持图片事件记忆、音频事件记忆、可版本化的语义事实维护和带证据的 Agent 问答；视频只登记原始资产并保留 `video_memory_adapter` 协议。

## 记忆边界

- 事件记忆：图片、音频、文本共享，未来视频也通过 Observation 进入。事件按时间、地点、人物、活动和证据聚合，可被新观察更新。
- 语义记忆：图片、音频、文本共享，维护人物、地点、关系、习惯和一般事实。事实有状态、置信度、证据和修订链，不直接覆盖历史。
- 视频编码记忆：视频独享。第一版不解码、不切片、不生成动作向量；导入只生成 Asset 和等待状态。

## 数据流

```text
Asset -> Observation -> Event candidate -> Event revision
                             |
                             +-> Fact proposal -> Fact revision
                             |
                             +-> Sentrix native semantic index / graph
```

Sentrix 自己的 SQLite、原生向量索引和关系图是事实来源。Cognee 只作为设计参考，不作为运行时依赖，不接收数据投影，也不参与线上检索。

## 模型适配

- Ollama `gemma4:12b`：图片观察抽取、文本事实归纳、Agent 回答。
- FunASR：Paraformer 转写、FSMN-VAD 活动检测和 CT-Punc 标点恢复。
- 153 上的 `buffalo_l`：人脸检测与 embedding，不依赖 FMA 的代码、数据库或 API。
- `video_memory_adapter`：固定接口，第一版返回 `video-extraction-reserved`。

### 原生记忆存储

- `SQLite MemoryStore`：Asset、Observation、Event、Entity、Relationship、Fact、Evidence 和修订历史。
- `NativeVectorStore`：在 Sentrix 自己的数据库中保存向量、向量空间、来源 ID 和模型版本，提供余弦相似度检索；不依赖 Cognee。
- `MemoryGraph`：使用实体与关系表维护可审计的家庭知识图，关系必须绑定证据和状态。

## Agent 契约

Agent 先解析问题，再从 Sentrix 原生的事件索引、实体关系图、事实版本库和向量索引检索，最后将有来源的上下文交给 Gemma。模型只能使用上下文回答；证据不足时必须明确说明，并返回证据 ID、置信度和检索轨迹。

## 隐私与部署

原始文件、SQLite、人物特征和模型调用都留在局域网。网页服务运行在 153；不运行 Cognee 作为 Sentrix 依赖；不复制 FMA 项目或其数据。
