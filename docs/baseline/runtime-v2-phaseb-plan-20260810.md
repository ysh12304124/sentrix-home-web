# Agent Runtime v2 — Phase B 执行计划

- 日期：2026-08-10
- 对照计划：`Sentrix_Agent_Runtime_v2_PhaseB_Tool真实性与Canary准备_规划输入.md`（v1.0）
- 代码基线：153 `psh` = `bc46460`；本地分支 `psh-runtime-v2`
- 阶段目标：把 Tool-Loop 从"架构证明成功"推进到"结构化事实可灰度、图片类 Tool 真实性可控、结果交付可用、生产迁移证据完整"

## 0. 环境核验（2026-08-10 实测，B0 冻结基准）

| 项 | 实测 |
|---|---|
| 153 psh HEAD | `bc46460`（A1-A5 + 完成度报告已合入） |
| production 8091 | 运行中，`pipeline` profile，默认行为未切 Tool-Loop |
| 8092 validation | 运行中，`SENTRIX_LLM_BACKEND=vllm SENTRIX_VLLM_BASE_URL=http://127.0.0.1:8100/v1 SENTRIX_VLLM_MODEL=gemma4-12b-it` |
| `data/ann/` | 空（0 文件）—— visual/text/semantic/episodic ANN 全部缺失 |
| `memory_vectors` | 2528 行在 `data/sentrix.db`（重建素材存在） |
| GPU | 单卡 24GB，used 2.2GB，空闲约 22GB |
| vLLM manager | `current=none`；8100/8105 当前均无响应；12B 模型 `gemma4-12b-it`（bitsandbytes 4-bit）配置存在，按需启动 |
| 前端 | `src/app.js`/`src/api.js` 已有 progress 渲染；Node 测试 31/31 |
| 后端测试 | `backend/tests/` 80+ 文件，`npm run backend:test`（unittest discover） |
| 重建脚本 | `scripts/maintenance/rebuild_ann_indices.py`（hnswlib，原子 swap + manifest）已存在 |
| ANN 运行时 | `backend/retrieval_ann.py`：hnswlib 原生删除 + manifest 校验 + `index_incompatible` 降级 |

## 1. 代码 Agent 建议（已采纳进计划）

1. **Omission 检测用结构化 evidence 引用，不靠字符串猜测**：final action 增加可选字段 `evidence_refs: ["tool_call_id"]`，FinalGuard 校验"每个非空 ToolResult 至少被引用一次 / 引用的 id 必须存在"。s14 类"有结果却说没有"由此变成确定性可判。
2. **Truth Contract 由 Tool 层确定性计算，不交给模型**：`search_memories` 根据 `packet.condition_results`（matched/unknown）+ gaps 计算 `query_satisfaction`（full_support / partial_support / candidate_only / no_match / blocked），Observation 直接带 verdict；模型只负责照实叙述。
3. **ANN 重建做"最小但诚实"**：直接复用 `rebuild_ann_indices.py`（hnswlib + 原子 swap + manifest）；不做复杂回滚。重建后按空间实测 recall/通道可用性，`tool_readiness_matrix.json` 如实标 ready/limited/blocked，绝不"文件存在即 ready"。
4. **Live progress 用后台任务 + 轮询端点**：`/api/assistant/turn` 改为异步执行（POST 立即返回 `turn_id`），新增 `/api/assistant/turn/{turn_id}` 轮询，返回增量 `public_progress` 与最终结果；前端每 500ms 轮询。比 SSE 改动小、可测 `Time to First Progress`，SSE 留待后续。
5. **ResultSet 保持内存 + TTL，不建新 DB 表**：TTL 30 分钟 + owner conversation + 稳定排序（asset_id 键序）+ `get_result_page(result_set_id, cursor)` 工具。当前规模（几十~几百候选）足够，避免过度设计。
6. **Structured Canary 用实例隔离，不做流量切分**：8092 作为唯一 canary 面（`SENTRIX_AGENT_PROFILE=tool_loop`），8091 生产不动；响应带 `profile_used` + `tool_loop_status`，回滚 = 重启 8092 为 `pipeline`。canary telemetry 落 trajectory 表。
7. **12B 约束解码按"先量化失败分布、再试 vLLM guided_json"推进**：先统计 18 例修复失败的具体模式（截断/围栏/畸形键值），再对照 `extra_body={"guided_json": schema}`、更短 schema、`max_tokens` 与 temperature；以实测 repair 率决定，不强行迁移 native tool calling。
8. **Emergency renderer 覆盖 budget/timeout/unparseable/guard-failure 四类**：只基于 TaskState + 已存在 Tool Observation 生成确定性诚实摘要；guard 冲突时先给模型一次受控修正机会，仍失败才走 emergency。

## 2. 阶段划分与验收

### B0 — Fresh Baseline 与环境冻结
- 记录冻结清单（HEAD/profile/DB 校验/ANN/GPU/模型/前端构建/旧 flags）→ `docs/baseline/runtime-v2-phaseb-b0-20260810.md`
- 跑：backend unittest 全套、前端 `node --test`、`compileall`、structured QA、retrieval regression、RX cases、Tool Selection 55 例、Shadow 18 例、latency
- 产出：`tool_loop_baseline_manifest.json`（含 raw action repair 分布基线）
- 启动 12B 于 8105（不占用 8100/8092 使用的 8100 端口）

### B1 — ANN Recovery & Retrieval Readiness
- 诊断 `memory_vectors` 空间/模型/维度分布；`rebuild_ann_indices.py --apply` 重建；校验 manifest/checksum/query 兼容
- 每个空间跑 recall 回归（`test_visual_ann_retriever` / `test_text_ann_retriever` / retrieval kernel benchmark），记 `ann_health.json`
- `search_memories.visual/text` readiness 如实更新到 registry + matrix
- 验收：`data/ann/` 有索引 + manifest；health 可被 runtime 读取；retrieval regression 数字记录在报告

### B2 — `search_memories` Truth Contract 重构
- Tool Observation 增加：`result_set_id / total / query_satisfaction / condition_summary{confirmed,supported,unknown,contradicted} / answerability / has_more / remaining`
- 状态命名：`full_support / partial_support / candidate_only / no_match / blocked`
- SYSTEM 规则 + Guard 约束：candidate_only 不得声称"找到了 X"；no_match 不得声称找到；partial_support 必须披露缺口
- 验收：s05/s07/s08/s09/s17 等图片类案例中 candidate_only 不再被说成 full match（shadow 重跑对比）

### B2.1 — Observation Faithfulness Guard
- FinalGuard 新增：`evidence_refs` 校验、non-empty→omit 检测、unknown→confirmed 检测（certainty upgrade）、empty→found、count/date/group 已有项保留
- Runtime 对 guard 冲突给模型一次修正机会（追加"你的回答与工具结果冲突"反馈），仍失败 → partial/emergency
- 验收：7 个既有编造仍全拦；s14 类 omission 被拦或修正

### B2.2 — Observation Faithfulness Benchmark
- 新建 40-60 条 Tool-result-conditioned cases（exact count / partial / empty / candidate_only / found-but-omit / all-has_more / group / conflicting）
- 指标：Faithfulness>=95%、FP Fulfillment=0、FN/Omission<=2%、Certainty Upgrade=0、Required Disclosure>=98%
- 产出 `faithfulness_benchmark_12b.json` + 报告

### B2.3 — Action Serialization Hardening
- 先统计 18 例（+faithfulness cases）raw JSON 失败分布
- 对照实验：free JSON vs vLLM `guided_json` vs 更短 schema vs `max_tokens`/temperature
- 目标：raw valid >=95%、repair/retry <=5%；产出 `action_serialization_experiments.json`

### B3 — Search → Inspect → Answer E2E（≥10 例）
- 用例：桌上有什么/衣着颜色 unknown/OCR 复核/人数/小物体等
- Guard：inspect 每轮上限（现有 budget）、只允许结果集内 handle、ephemeral 不写 Formation
- 验收：Agent 自主 inspect 选择 Precision 高、不必要 inspect<=10%、inspection 断言可追溯（进 final evidence）

### B3.1 — ResultSet 持久化/分页
- `ResultSet` 增加：owner_conversation / TTL（30min）/ stable ordering / cursor；`ResultSetStore` 加过期清理
- 新增 Tool：`get_result_page(result_set_id, cursor, count)` → 返回 `page/has_more/next_cursor/handles`
- 验收：连续对话"第二张/还有吗/下一页"稳定引用同一 result_set_id

### B3.2 — Original Photo Delivery E2E
- 核验 `get_original_photos` → handle→asset 授权链 → `/api/assets/{id}/file` scope 检查
- 用例：成功/无权限/asset 不存在/result_set 过期/handle 错误/多图
- 验收：前端渲染授权原图；越权 0

### B3.3 — Frontend Agent Delivery UX
- 前端：ResultSet 卡片（total/has_more/剩余）、"下一页/还有吗"、选中照片、原图交付、partial/candidate 标签自然呈现
- 不实现确认 UI（接口预留）；Node 测试全绿

### B3.4 — Live Public Progress
- app.py：turn 后台执行 + `/api/assistant/turn/{turn_id}` 轮询；进度事件含"正在查什么/已得到什么/下一步"
- 前端：轮询渲染增量 progress（替代仅最终返回）
- 指标：`Time to First Progress` 实测记录；不暴露私有推理/内部 ID

### B4 — Canary Hardening
- Emergency Renderer（四类触发，诚实 partial）
- `tool_readiness_matrix.json` 正式化（按工具/子操作）
- `agent_profile_manifest.json` + runbook（启动/健康/验证/切换/回滚）
- `structured_memory_coverage.json` 由 A0 数据 + 新查证一键产出
- canary telemetry：turn 记录 profile/tool calls/guard/final/latency/progress/fallback/category
- 验收：8092 canary 可一键回滚 `pipeline`；静默 fallback=0

### B5 — Structured Facts Limited Canary 决断
- 满足 §15 前置后，8092 以 `tool_loop` 运行第一批（count/exists/first/last/media/date-group，排除 place/person 语义）
- 退出门槛对照 §20.3；产出"结构化 canary 可启动/阻塞原因"
- 图片类 canary 决断（对照 §21，预计被 ANN/semantic 阻塞，给出明确数字）

### 并行子轨
- Semantic Evidence：failure taxonomy（_contains 失败类型/深色/做晚饭/garment/place/activity）+ baseline benchmark + 候选方法建议（不强制上线）
- Model Optimization：guided decoding/量化/温度对照（并入 B2.3）
- Person/Core/Write：只产出进入条件文档，不实现

## 3. §30 问题回答表（计划内给出答案的项）

| # | 问题 | 计划答案 |
|---|---|---|
| 1 | `data/ann/` 为何为空 | 数据目录丢失/未重建；脚本存在但未在 153 执行过（B1 重建并记录根因） |
| 2 | manifest 字段 | model_id/dim/space/count/checksum/source_revision/created_at（沿用 `retrieval_ann.py` 现有 manifest） |
| 3 | ANN missing 时通道 | `EvidenceRetrievalKernel` 多通道 prefilter→recall→fusion；ANN retriever 抛 `index_incompatible` 时该通道记 0 候选并进 channel_trace；lexical/structured 仍可用（B1 用 channel_trace 实测） |
| 4 | ANN 恢复后哪些 case 会变 | s06/s08/s09/s10/s17 等视觉语义候选质量与数量；用 shadow 重跑 diff |
| 5-6 | candidate vs supported / unknown 展示 | Tool Observation 自带 `query_satisfaction`/`condition_summary`，模型只叙述（B2） |
| 7 | s14 类漏报检测 | `evidence_refs` 结构化引用 + FinalGuard（B2.1） |
| 8 | place group 遵从 | 先测 schema 失败分布；候选：group_by 枚举化 + 更短 schema（B2.3 实测） |
| 9-10 | repair 分布 / guided_json 降幅 | B2.3 实测产出 |
| 11-12 | inspect 绑定 ResultSet / evidence 进 final | handle 必须解析自当前 result_set；inspection 结果作为 tool observation 进上下文，final 需引用（B3） |
| 13-14 | ResultSet 持久化 / cursor | 内存+TTL30min；asset_id 键序 offset cursor（B3.1） |
| 15 | 原图授权端点 | `get_original_photos` + `/api/assets/{id}/file` scope 检查（B3.2 审计） |
| 16-17 | live progress / total-next | 轮询端点 + 前端卡片（B3.3/B3.4） |
| 18 | emergency 覆盖类型 | budget/timeout/unparseable/guard-failure（B4） |
| 19 | pipeline fallback 标记 | 响应 `profile_used` + `tool_loop_status=fallback` 显式标记（B4） |
| 20 | 第一批 query 分类 | count/exists/first/last/media/date-group，排除 place/person（B5） |
| 21 | 旧 flags 收敛 | 只收敛到 3 个 profile + manifest；25+ 旧 flag 标记 deprecated 但不删除（B4.2） |
| 22 | semantic 高频失败 | taxonomy 实测后取 top3（并行子轨） |
| 23 | 减少 fabrication 的模型优化 | 候选：guided_json + observation 完整放行 + faithfulness 反馈修正；以实测定（B2.3/B2.1） |
| 24 | 图片 canary 数字门槛 | 对照 §21 量化项逐项打勾（B5） |

## 4. 风险与真实阻塞

- **12B 服务**：8100 被其他会话使用风险；统一用 8105，若显存被占则顺延（B0 阻塞项，需机器空闲）
- **inspect_photo 多模态**：A0.6 已验证 12/12，但依赖 gamma 图像通道；若 8105 模型不支持图像需切换通道（B3 首日验证）
- **ANN 重建依赖 hnswlib/embedder 可用**：153 venv 需确认包；visual 空间优先用存量 memory_vectors（source_type=asset），不强制重新 embed
- **前端改动面**：Node 测试约束；分页/原图/轮询一次性合入，最后统一跑测试

## 5. 部署与回滚

- 开发：本地 `psh-runtime-v2`；验证：同步到 153 `/home/asus/runtime-v2-work` 跑离线评估；全部完成后推送并合并 153 `psh`
- 运行验证用新端口（如 8097），不碰 8091/8092 现有实例
- 回滚：8091 默认 `pipeline` 不变；canary 面只有 8092，重启即回滚

## 6. DoD（§32）映射

DoD 1-6 → B0/B1；7-9 → B2/B2.1；10-11 → B2.2/B2.3；12 → B3；13-15 → B3.1/B3.2/B3.3；16 → B3.4；17 → B4；18-20 → B4/B5；21 → B5 决断；22 → 全程回归；23 → Final report。
