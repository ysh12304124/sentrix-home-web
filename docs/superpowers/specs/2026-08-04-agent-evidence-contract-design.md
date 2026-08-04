# Agent Evidence Contract Design

## Goal

将 Sentrix Agent 从固定检索器推进为中性的家庭记忆助手。所有涉及家庭记忆的具体回答都必须带可追溯证据入口；证据默认可折叠，但不能从响应或界面中省略。用户明确要求原始证据时，直接返回可打开的原始 Asset、Observation 或媒体。

## Scope

本设计只覆盖 Agent、assistant API、对话前端和相关测试：

- `backend/agent.py`
- `backend/app.py` 中 assistant 相关请求和响应适配
- `src/api.js`
- `src/app.js`
- `src/styles.css`
- Agent、API、前端回归测试和本设计/实施文档

不修改记忆生成、媒体导入、模型推理、人脸、事件生成、重建或生产数据库。若现有记忆接口无法提供所需的只读证据，先提出接口请求，不在本任务中修改生成管线。

## Product Rules

1. 普通聊天不读取家庭记忆，不返回记忆证据或媒体。
2. 记忆回答必须包含 `evidence` 或明确的 `query_gap`/澄清结果；不能返回无来源的具体家庭事实。
3. 证据至少能沿着 `semantic claim/profile -> event -> observation -> asset` 回溯；缺失层级时保留实际可用的证据层级，并明确缺口。
4. 证据默认折叠展示，但前端必须渲染证据入口。
5. 明确要求照片、原图、原始资料或查看依据时，直接展示可打开的原始媒体或 Observation，不只返回折叠入口。
6. 向量候选不能单独支撑事实，必须回到事件或原始观察文本锚定。
7. 所有读取和会话状态受 `scope_id` 限制。
8. 反馈只有在目标明确且用户确认后才能改变事实，普通对话不能写入事实。

## Internal Turn Contract

每轮 Agent 先产生受限计划，再由后端白名单执行。响应保留机器可验证字段，前端将其转换为自然对话：

```json
{
  "mode": "chat|memory|feedback|clarify",
  "answer": "自然语言回答",
  "memory_used": true,
  "evidence": [],
  "evidence_layers": {"claims": [], "events": [], "observations": [], "assets": []},
  "original_evidence_requested": false,
  "images": [],
  "tool_trace": [],
  "confidence": 0.0,
  "insufficient_evidence": false,
  "clarification_candidates": [],
  "query_gap_id": null,
  "dialogue_state": {}
}
```

`evidence` 是记忆回答的强制字段。无证据时不得用空证据包装具体事实，必须转为缺口或澄清状态。

## Tool Boundary

允许的只读工具：`resolve_constraints`、`describe_entity`、`find_events`、`trace_timeline`、`compare_memories`、`suggest_recall`、`open_evidence`、`request_clarification`。

治理工具 `record_feedback` 只接受显式目标和确认后的写入。模型不能访问 SQLite、生成事实或自行选择未验证的证据。

## Retrieval Order

```text
semantic entity group
  -> stable source entities
  -> anchored events
  -> observations/assets
  -> vector candidates only when structure and lexical evidence are insufficient
```

输出证据按查询覆盖、实体/事件直接匹配、Observation 支持度、来源置信度和时间地点约束排序。图片是证据的一种表现，不是默认答案主体。

## Acceptance Gates

- 普通聊天的家庭记忆读取次数为 0。
- 记忆回答缺失证据入口为 0。
- 无证据事实回答为 0。
- 跨 `MemorySpace` 读取为 0。
- 明确要求原始证据却未返回原始媒体为 0。
- 未确认反馈写入事实为 0。
- 介绍、时间线、比较、推荐、澄清和反馈均有 Agent 回归测试。
