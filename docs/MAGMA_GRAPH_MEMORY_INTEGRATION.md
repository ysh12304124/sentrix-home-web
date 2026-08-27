# MAGMA 视频记忆图接入说明

## 目标与边界

本次接入把 MAGMA 的“图结构与记忆结构”以加法方式接入 Sentrix 现有
SQLite 记忆库。这里的“图”是视频记忆图，不是图片文件、图片缩略图或新的
图片处理链。原始视频、视频场景、关键帧观察和实体仍由现有 Sentrix 管线产生；
图只是可删除、可重建的派生索引。

明确保持不变：`backend/video/hybrid_keyframe.py`、
`backend/video/mlt_keyframe.py`、`backend/video/processor.py`、WorldMM 适配器、
场景边界、关键帧预算/数量、`search_memories`、RRF、ANN、Agent 工具选择与 QA
排序，以及 MAGMA 的 `keyframe_server.py`、Flask 和 Qdrant 输入。

## 记忆映射

| Sentrix 事实 | 图节点 | 说明 |
| --- | --- | --- |
| MemorySpace | `EPISODE` | 每个 scope 有稳定根节点，承接非视频事件 |
| 原始视频 `assets` | `EPISODE` | 仅选择 `media_type=video` 且不是派生 asset |
| `events` 中的视频场景 | `SESSION` | 通过 `source_asset_id/source_scene_index` 保留视频血缘 |
| `observations`（通常来自关键帧） | `EVENT` | attributes 保留 `observation_id/asset_id/event_id` |
| `entities` | `ENTITY` | 人物和非人物实体都保留原 scope |

关系只从已有事实生成：`BELONGS_TO_SESSION`、`PART_OF`、单向
`PRECEDES`、`REFERS_TO` 和 `RELATED_TO`。关键帧顺序优先使用
`source_timestamp_sec`，再用 `captured_at`、`source_frame_index` 和稳定 ID
消歧；视频场景顺序使用 `source_scene_index/source_start_sec`。相邻不等于因果。
视频相邻场景会生成单独的 `LEADS_TO_CANDIDATE` 待确认候选，但只有来源元数据
明确提供 `causal_edges`/`causal_relations` 时才生成正式的 `CAUSAL:LEADS_TO`
或 `CAUSAL:ENABLES`，因此 `causal_edges` 仍可能为 0。

## 派生表与重建

`backend/graph_memory.py` 只创建以下表，不改 canonical 表，也不建立到 canonical
表的外键，便于安全删除后重建：

- `graph_memory_nodes`
- `graph_memory_edges`
- `graph_memory_builds`

节点和边 ID 由 `scope_id + source/type` 稳定哈希生成；单 scope 重建在一个
`BEGIN IMMEDIATE` 事务中先删旧边、再删旧节点，失败时回滚并记录 `failed` build。
重建有进程内 `db_write_guard` 保护，查询严格按 scope 隔离。

## API

后端权威入口是 `backend/app.py`：

- `GET /api/graph-memory/stats?scope_id=home-default`（含 `causal_edges` 正式边和 `causal_candidates` 待确认候选）
- `POST /api/graph-memory/rebuild`，body：`{"scope_id":"home-default"}`
- `POST /api/graph-memory/query`，body：`query/scope_id/limit/expand_depth/node_types`
- `GET /api/graph-memory/nodes/{node_id}?scope_id=home-default`
- `GET /api/graph-memory/subgraph`，支持 `max_depth<=5`、`max_nodes<=500`、边类型和最小置信度过滤

查询先做轻量词法锚点匹配（包含中文二元词），再从最高 5 个锚点做双向有界 BFS，
返回 `anchors`、`expanded`、`matched_terms`、`graph_score` 和 `depth`。它不是
现有主检索/RRF/ANN 的替代品。

## 运维命令

默认只查看统计，不会重建：

```bash
PYTHONPATH=. python3 scripts/maintenance/rebuild_graph_memory.py \
  --db data/sentrix.db --scope-id home-default --stats
```

明确执行重建：

```bash
PYTHONPATH=. python3 scripts/maintenance/rebuild_graph_memory.py \
  --db data/sentrix.db --scope-id home-default --apply
```

生产操作前应使用仓库已有的 `scripts/maintenance/backup_sentrix.sh` 备份数据库，
确认 `SENTRIX_DB_PATH` 和现有服务实例后再执行；不启动独立数据库服务、不新增端口，
也不直接修改 canonical 表。

## 测试与回滚

覆盖测试位于 `backend/tests/test_graph_memory.py`，验证视频映射、单向排序、scope
隔离、检索扩展、幂等性、事务回滚、canonical 计数和 `foreign_key_check`。
若需回滚，只需停止现有服务并回退本次 Git 提交；图派生表本身也可由
`DROP TABLE graph_memory_edges`、`graph_memory_nodes`、`graph_memory_builds` 后
再次运行 `--apply` 恢复，canonical 记忆不受影响。
