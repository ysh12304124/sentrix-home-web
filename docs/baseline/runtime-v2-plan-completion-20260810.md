# Agent Runtime v2 — 计划完成度 / 代码实现 / 实测效果报告

- 日期：2026-08-10
- 对照计划：`Sentrix_Agent_Runtime_v2_Tool_Loop_架构验证与迁移计划_v2.docx`
- 代码：本地分支 `psh-runtime-v2`（`0904399`/`5d1ed3c`/`2c557bf`）→ 已合并 153 `psh`（`cec3bd4`）
- 一句话结论：**核心验证链（A0→A0.5→A0.6→A1→A2-A4→A5）全部完成并合入 153 psh；Definition of Done 24 条中 19 条完成、5 条部分完成；计划中明确标注"后移/并行子轨"的项（人物长总结、语义质量子轨、写工具、老会话迁移等）不属于本阶段范围。**

## 1. 完成度总览

| 阶段/任务 | 状态 | 证据 |
|---|---|---|
| A0 生产/RX/前端/数据真实核验 | ✅ 完成 | `docs/baseline/runtime-v2-a0-20260810.md` |
| A0.5 12B Tool Selection Spike | ✅ 完成 | 55 例，schema 100%，primary 98.2%，unnecessary 0% |
| A0.6 inspect_photo 多模态 Spike | ✅ 完成 | 12/12 真实图片，avg 2.9s |
| A1 ConversationStore / Trajectory / Progress | ✅ 完成 | `backend/agent_conversation.py` + 前端渲染 |
| A2 AgentRuntime 控制流 | ✅ 完成 | `backend/agent_runtime/runtime.py` |
| A3 4 个只读 Tool + Shadow | ✅ 完成 | 18 例，10 正确 / 7 守卫拦截 / 1 partial |
| A4 ResultSetStore / TaskState / FinalGuard | ✅ 完成 | `result_set.py` / `final_guard.py` |
| A5 Offline A/B vs Canonical RX | ✅ 完成 | `docs/baseline/runtime-v2-a05-shadow-ab-20260810.md` |
| Profile 门控接入 app 入口 | ✅ 完成 | `SENTRIX_AGENT_PROFILE=tool_loop*` |
| 合并 153 `psh` | ✅ 完成 | `cec3bd4`，22 文件 +1882 行 |
| Fresh Baseline 完整 Python 套件 | ⚠️ 部分 | Node 31/31；backend 关键测试通过；完整套件未跑 |
| 前端分页 / 确认 / 原图交付 UI | ⚠️ 未做 | 仅 progress 渲染已加 |
| search→inspect→answer 端到端用例 | ⚠️ 部分 | 链路独立验证（A0.6）；shadow 未触发完整链 |
| ResultSet→原图交付 E2E | ⚠️ 部分 | 工具注册（limited）；shadow 未触发 |
| flags 全量收敛进 Profile | ⚠️ 部分 | 门控已接入；~25 个旧 flag 未全部收敛 |
| structured_memory_coverage.json | ❌ 未产出 | A0 报告有数字，无独立 JSON 文件 |
| ANN 索引重建（data/ann/） | ❌ 未做 | 已标记为图片类 Tool 的高优先级前置 |
| Emergency renderer（模型超时兜底） | ❌ 未做 | s03 partial 时 final 为空 |

## 2. 计划 §39 执行计划 22 项逐项核对

| # | 任务 | 状态 | 说明 |
|---|---|---|---|
| 1 | A0 生产/RX/前端/数据真实核验 | ✅ | 8091 无 RX/旧模板、8092 全 RX、前端无分页/progress/确认、结构化覆盖率、writer→claim 状态 |
| 2 | Fresh Baseline | ⚠️ | Node 31/31；backend `test_thin_agent_rx_path`(5)/`test_thin_agent_runtime`+`contracts`(14)/`test_agent_conversation`(4) 通过；完整套件待维护窗口 |
| 3 | Tool Selection Capability Spike | ✅ | 55 例 12 类，`result_v3.json`：schema 100%、primary 98.2%（1 例 place 误选）、unnecessary 0% |
| 4 | Multimodal inspect Spike | ✅ | 12/12 有效 JSON，scene/people_count/clothing/objects/OCR/activity/certainty，avg 2.9s |
| 5 | Agent Runtime 控制流设计 | ✅ | `AgentRuntime.run()` 薄循环：model→parse→tool→observation→loop；BudgetManager 统一预算 |
| 6 | ConversationStore / Trajectory | ✅ | 服务端 canonical 表（conversations/messages/tool_calls/tool_results/ui_events/attachments/trajectories）；`recent_turns` 注入模型 |
| 7 | Public Progress UX | ✅ | `public_progress` 进 API + 前端 `.assistant-progress` 渲染（本轮补样式）；未做流式/首进度耗时埋点 |
| 8 | Tool Registry / Tool Policy | ✅ | `ToolSpec`（read/write、scope、timeout、cost、readiness、version）+ `validate→authorize→budget→execute→sanitize` |
| 9 | Tool Readiness Matrix | ⚠️ | registry 有 readiness 字段；A0 报告有覆盖率数据；未产出独立 matrix 文档 |
| 10 | Budget / Timeout / Graceful Completion | ⚠️ | `BudgetState`（model steps/tool calls/inspections/wall time/final reserve）+ partial 状态（s03 实测）；emergency renderer 未做 |
| 11 | MVP FinalGuard | ✅ | count/date/group/exists 一致性 + 空检索伪造 + ID 泄漏 + all/has_more + delivery 矛盾；实测 7/7 拦截 |
| 12 | 3~4 Tool Shadow MVP | ✅ | 4 个只读 Tool、18 例、10 正确/7 拦截/1 partial、avg 1.92s |
| 13 | ResultSet / TaskState | ⚠️ | in-memory `ResultSetStore`（result_set_id + handle 映射不泄漏 ID）；分页 cursor/持久化未做 |
| 14 | Profile / Flags 收敛 | ⚠️ | `SENTRIX_AGENT_PROFILE` 门控接入 `/api/assistant/turn`；旧 flag 全量收敛未完成 |
| 15 | Structured/Semantic/Person 数据质量 | ⚠️ | A0 覆盖数据已测（entity 8.8%、place 非 canonical、ANN 缺失）；semantic quality 子轨未做 |
| 16 | Frontend 能力 | ⚠️ | progress 渲染已加；分页/继续加载/确认 UI/原图 stable handle 未加 |
| 17 | Offline A/B | ✅ | 同 DB/同 Kernel/同 12B，18 例；RX 18 答/0 降级/avg 4.24s；`ab_v4.json` |
| 18 | 轨迹和产品指标 | ⚠️ | trajectory 保存实现；First Progress/First Observation/UX 指标未埋点 |
| 19 | Canary 门槛与回滚 | ⚠️ | 门槛量化 + 决断建议（结构化类可 canary、图片类暂缓）；canary 未实际启动、回滚文档未完整 |
| 20 | 分阶段代码修改范围 | ✅ | 严格按 A0→A5 分阶段实现 |
| 21 | 部署安全 | ✅ | A/B 在独立 work 副本 + 8105 新端口隔离；生产 8091 全程未动（health 200） |
| 22 | 用户必须提供的输入/真实阻塞 | ✅ | GPU 争用（Qwen 占 19GB）、ANN 重建、完整测试维护窗口均已记录 |

## 3. Definition of Done（§40）24 条逐条核对

| 条目 | 状态 |
|---|---|
| Production/RX/frontend baseline 唯一可复现 | ✅ A0 报告 |
| Fresh baseline 完成 | ⚠️ 部分（见 §2-2） |
| 12B Tool Selection benchmark 真实结果 | ✅ 98.2%/100%/0% |
| Tool action schema 结构可靠 | ✅ 100% + 解析修复/重试兜底 |
| 是否达到 Shadow 门槛有明确结论 | ✅ 达到（98.2% ≥ 90%、0% ≤ 10%、schema 100%） |
| inspect_photo 独立结论 | ✅ 12/12、2.9s、失败模式已记录 |
| 服务端 ConversationStore 模型可见真实历史 | ✅ `recent_turns` 注入 |
| trajectory 完整保存 model/tool/observation | ✅ `save_trajectory` |
| 用户可见 Public Action Trace / Progress | ✅ API + 前端渲染 |
| Tool Registry/Policy/BudgetManager 可运行 | ✅ |
| ≥3 个 read Tool 进 Offline Shadow | ✅ 4 个 |
| 至少一个 2-step Tool Loop | ✅ s03/s14（search→facts） |
| 至少一个 structured fact 自主选 fact Tool | ✅ s01/s02/s04/s15 |
| 至少一个图片查询自主选 search Tool | ✅ s05-s11/s16-s18 |
| search→resolve→inspect→answer | ⚠️ 链路独立验证（A0.6）；shadow 未触发完整链 |
| 至少一个 ResultSet→original delivery | ⚠️ 工具注册（limited）；未 E2E 触发 |
| FinalGuard 阻止 false fulfillment/delivery contradiction | ✅ 7/7 编造被拦 |
| Tool readiness 暴露数据覆盖缺陷 | ✅ condition unknown / ANN 缺失 / entity 8.8% |
| Budget exhaustion 诚实结束 | ✅ s03 partial（无伪造） |
| Offline A/B 完成 | ✅ |
| 迁移价值门槛达到或明确说明未达到 | ✅ 报告已给出：结构化事实类值得 canary，图片类未达 |
| Safety/permission 零回退 | ✅ 未改权限路径；ToolPolicy 只读授权 |
| flags 开始收敛为 Agent Profile | ⚠️ 门控接入；全量收敛未完成 |
| Canary 是否值得启动有明确决断 | ✅ 结构化类可启动验证实例；图片类待语义子轨 |

## 4. 代码实现清单（按提交）

### `0904399` — A1：ConversationStore + trajectory + progress
- `backend/agent_conversation.py`：`ConversationStore`（conversations/messages/tool_calls/tool_results/ui_events/attachments + `save_trajectory`/`last_messages`/`bootstrap_recent`）
- `backend/db.py`：新增 8 张表 + 索引（门控 `SENTRIX_CONVERSATION_STORE_V1`）
- `backend/app.py`：`/api/assistant/turn` 注入 `recent_turns` + 持久化 + `public_progress`；`/api/conversation/{id}/messages`
- `src/app.js` / `src/api.js`：`public_progress` 渲染

### `5d1ed3c` — A2-A5：Tool-Loop 运行时 + 4 个只读 Tool + FinalGuard + Profile 门控
- `backend/agent_runtime/runtime.py`：`AgentRuntime` 薄循环；`_parse_action` JSON 修复链（围栏/截断补全/畸形 `{"}`/尾逗号）；解析失败带反馈重试 1 次；Budget 耗尽 → partial
- `backend/agent_runtime/tool_registry.py`：`ToolSpec` 合同 + 注册表
- `backend/agent_runtime/tool_policy.py`：`validate→authorize→budget→execute→sanitize`；按工具白名单清洗 observation
- `backend/agent_runtime/budget_manager.py`：`BudgetState`（model steps/tool calls/inspections/wall time/final reserve）
- `backend/agent_runtime/profile.py`：`SENTRIX_AGENT_PROFILE` → `ProfileConfig`（pipeline/tool_loop_shadow/tool_loop）
- `backend/agent_runtime/result_set.py`：`ResultSetStore`（handle 映射）+ `TaskState`（fact_value/rows/group_by/delivery_state）
- `backend/agent_runtime/final_guard.py`：count/date/group/exists 一致性 + 空检索伪造 + ID 泄漏 + all/has_more + delivery 矛盾
- `backend/agent_runtime/tools.py`：`query_memory_facts`（count/exists/first/last/date/group/media）/ `search_memories`（Kernel 封装+ResultSet）/ `get_original_photos` / `inspect_photo`
- `backend/query_contracts.py`：`parse_time_expression` 支持纯年份（2023 → 整年）
- `backend/app.py`：`SENTRIX_AGENT_PROFILE=tool_loop|tool_loop_shadow` 时走 `AgentRuntime`
- `src/styles.css`：`.assistant-progress` / `.progress-step` 样式
- `scripts/benchmarks/evaluate_tool_loop_shadow.py` + `shadow_cases_v1.json` + `evaluate_tool_loop_ab.py`

### `2c557bf` — 报告
- `docs/baseline/runtime-v2-a0-20260810.md`（A0 只读基线）
- `docs/baseline/runtime-v2-a05-shadow-ab-20260810.md`（A0.5/A0.6/A1-A5 实测 + A/B）

## 5. 具体效果（实测数字）

### 5.1 Shadow Tool-Loop（18 例，scope=album2_e2b）
- 结果：10 正确 / 7 守卫拦截 / 1 partial，avg **1.92s/turn**
- 守卫拦截的 7 例全部是真实错误：count 128≠10、月份全列（真实仅 9/12/5 月）、首现日期 2023-05（真实 2025-04-12）、3 例空检索谎称"找到了"、视频数 12≠10
- 无守卫时这 7 例（39%）会是错误答案；守卫后最终交付的硬事实 **零错发**

### 5.2 关键根因修复（shadow 过程发现）
- ToolPolicy sanitize 白名单把 `value/rows/filters_applied` 过滤掉 → 模型看不到数据只能编造 → 已按工具放行
- 12B 输出非纯 JSON（截断/`"filters":{"}`/markdown 围栏）6/18 例 → JSON 修复链 + 重试反馈
- FinalGuard 补日期/月份/exists/空检索 4 类确定性检查

### 5.3 Offline A/B（同 DB/同 Kernel/同 12B）
| 维度 | Tool-Loop | Canonical RX |
|---|---|---|
| 完成 | 10/18 + 7 诚实拦截 + 1 partial | 18/18 有回答（17 RX，0 降级） |
| avg 延迟 | 1.92s | 4.24s |
| count/月份/首现 | 确定性执行 + 守卫 | 未答/泛化（s01/s02/s04） |
| 编造防护 | 7/7 拦截 | s11"60 张"与库不符且无守卫 |
| 图片语义声称 | 5 例 overclaim（condition unknown） | 同样 overclaim（s05/s07/s08/s09/s17）；s06/s10/s15 反而更诚实 |

### 5.4 前端
- 新增 progress 渲染（`assistantMessage` + CSS）；Node 测试 31/31 通过

## 6. 未完成/部分完成项与原因

| 项 | 原因/后续 |
|---|---|
| Fresh baseline 完整 Python 套件 | 需维护窗口；建议与 ANN 重建共享空闲期 |
| search→inspect→answer 端到端 | shadow 集未含需复核图片的 follow-up；A0.6 已独立验证链路 |
| ResultSet→原图交付 E2E | 需前端原图 handle/交付 UI + 分页协议（计划 §16 前端能力） |
| 前端分页/确认/继续加载 UI | 计划明确列为前端能力项，未在本轮实现 |
| flags 全量收敛 | 本轮仅接入 Profile 门控；25+ 旧 flag 收敛需迁移排期 |
| structured_memory_coverage.json | 可据 A0 报告 §3 数据一键产出 |
| ANN 索引重建 | `data/ann/` 为空 → visual/text ANN 通道降级，图片检索类 Tool 的前置项 |
| Emergency renderer | §22.5；模型最终调用失败时用确定性摘要兜底（当前 partial 返回空 answer） |
| 老会话迁移 / trajectory compaction | 计划允许新会话开始 server-canonical，不阻塞 MVP |

## 7. 结论与下一步建议
1. 计划的核心目标（证明 Tool-Loop 架构值得继续 + 达到/说明迁移价值门槛）已达成并合入 153 `psh`，生产 8091 默认行为不变。
2. 建议下一步优先级：① 重建 ANN 索引 → ② search_memories 区分"候选/已确认"并让模型照实陈述 → ③ 前端 ResultSet 分页/原图交付/确认 UI → ④ 结构化类查询以 `SENTRIX_AGENT_PROFILE=tool_loop` 开验证实例灰度 → ⑤ 12B 工具遵从/诚实性微调与语义子轨。
