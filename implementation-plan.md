# Sentrix Home 实际记忆接入计划

## 目标

让 153 上的 Sentrix Web 通过自己的本地记忆后端接入 `gemma4:12b`、FunASR 和 153 上的 `buffalo_l`。参考 Cognee 的 DataPoint、图关系和检索编排思想，但不依赖 Cognee 项目、不向 Cognee 投影数据。第一阶段交付图片事件记忆、原生实体/关系/事实维护和本地向量索引；视频只登记原始资产，不启动视频编码记忆。

## 架构

- Sentrix Web：网页 Portal 和同源 API 网关。
- Sentrix Backend：Sentrix 自有 SQLite、图片/音频摄取、Gemma 结构化观察、事件合并、人物识别适配、事实更新和 Agent 编排。
- FunASR：Sentrix 自己的本地 Paraformer、FSMN-VAD 和 CT-Punc 适配器。
- Ollama `127.0.0.1:11434`：`gemma4:12b` 多模态理解和 Agent 回答。
- Sentrix `MemoryStore`：SQLite 事实、事件、实体、关系和证据的权威存储。
- Sentrix `NativeVectorStore`：本地向量保存和相似度检索，按 `episodic`、`semantic`、`visual` 分空间。
- `video_memory_adapter`：保留协议，不在第一阶段调用。

## 实施步骤

1. 用单元测试固定 Sentrix Event、Person、Observation 和 Agent 上下文的数据契约。
2. 添加本地 SQLite 记忆层、实体关系图和原生向量索引。
3. 添加本地 Gamma Agent 调用；要求答案只能来自上下文，返回答案、置信度和证据 ID。
4. 将真实 dashboard、人物、时间线和搜索结果绑定到网页。
5. 下载小型公开 COCO 图片样本到未跟踪数据目录，通过 Sentrix `/api/ingest` 摄取，验证事件观察、人脸候选、事件聚合。
6. 触发事件重检、人物聚类和实体/事实维护接口，验证新增素材能够更新事件和知识状态。
7. 在 153 远端运行测试、启动服务并验证浏览器 API；不提交模型、数据、日志或 `.env`。

## 验收标准

- `node --test` 通过。
- Sentrix dashboard 能返回至少一个真实事件和一条图片观察。
- Agent 具体问题返回答案或明确的证据不足，而不是无来源编造。
- 新图片摄取后事件数量、观察数量或事件证据发生可追踪变化。
- 视频导入返回 `video-extraction-reserved`，不会生成视频编码结果。
