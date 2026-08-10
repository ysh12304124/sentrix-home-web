# Agent Runtime v2 — 代码审查清理 + GitHub main 合并 + 全量验证报告

- 日期：2026-08-10
- 153 正式分支：`psh`（HEAD `0ab8929`，由 `bc1274c` merge + 文档跟踪提交组成）
- 验证基线：合并/清理前 `92e5f1f`、合并+清理后 `1d64977`（内容等价 `bc1274c`）

## 1. 代码审查与清理（commit `1d64977`）

### 后端 backend/app.py
- 删除从未被调用的死代码：`_check_ollama_health`、`_check_e2b_health`、`_fire_and_forget_post`、`_schedule_backend_transition`、`VLM_BACKENDS`。
- 保留 GET/POST `/api/vlm-backend` 兼容壳（外部客户端可能仍在调用；POST 仍返回 410 指向 `/api/model-profiles/switch`）。

### 后端 backend/agent_runtime/profile.py
- 删除无人引用的 `tool_loop_active()`、`tool_enabled()`；`PROFILES` 收敛为 `tool_loop` / `tool_loop_shadow` 两个 profile。

### 前端 src/app.js / src/api.js
- **修复遗留 bug**：`submitProactiveOutcome`（主动回忆按钮）此前走同步 `assistantTurn` 直接消费结果，而后端 `/api/assistant/turn` 已是纯异步（返回 `turn_id` 供轮询），会导致主动回忆交互拿到错误结果。现统一为异步轮询路径。
- 抽出共享 `runAssistantTurn()`，`submitSearch` 与 `submitProactiveOutcome` 复用同一轮询逻辑（150 次 × 700ms 超时、实时进度、错误/超时兜底）。
- `src/api.js` 删除：重复的 `assistantTurn`（与 `assistantTurnAsync` 同实现）、已退役的 `getVlmBackend`/`setVlmBackend`、未使用的 `getCurrentModelProfile`、`importAsset`。

### 脚本与配置
- `scripts/runtime/start_sentrix_api_8097_phaseb.sh`：默认 profile 从已删除的 `pipeline` 改为 `tool_loop_shadow`。
- `scripts/runtime/start_sentrix_api.sh`：移除无人读取的死 flag `SENTRIX_ADVANCED_MEMORY_TOOLS_V1`（全仓 grep 确认无任何读取方）。
- `scripts/benchmarks/emit_phaseb_artifacts.py` + `docs/phaseb/agent_profile_manifest.json`：去掉 `pipeline` profile，更新回滚文案。

### 文档
- `docs/PROJECT_MEMORY.md`：Agent 层架构描述更新为「生产唯一路径 = AgentRuntime Tool-Loop（`backend/agent_runtime/`），Thin Agent 仅保留给 benchmark/回归测试」。

### 153 工作区清理
- 删除 6 个游离旧文件（均未跟踪）：`backend/runtime.py`、`backend/result_set.py`、`backend/agent_runtime/app.py`、`backend/agent_runtime/evaluate_result_set_e2e.py`、`backend/agent_runtime/evaluate_search_inspect_e2e.py`、空文件 `memory.db`。

## 2. GitHub main 合并（commit `2dea7c0`）

- 合并 `github/main`（`58cbbeb`）：geo fallback（国际 `reverse_geocoder` 兜底）+ 事件摘要保留 GPS。
- `backend/pipeline.py` 冲突采用 main（占位地点集合更全；这是图片导入数据管道，保留）。
- 顺带移除已废弃的 `pipeline` agent profile 与 `/api/search`（前一阶段 commit `c489ac6`）。

## 3. 全量后端测试（153 venv，psh 工作树）

```
705 tests ran | 7 failures | 0 errors | 2 skipped
```

- 对比：合并/清理前基线 692 tests | 7 failures | 3 errors。**3 个 geocoding error 已消除**（GitHub main 的 reverse_geocoder 兜底生效）。
- 用 worktree 在 153 上对合并前提交 `92e5f1f` 复跑同一失败集：**7 个失败与合并/清理后完全一致** → 主分支合并与本次清理均**无新增回归**。
- 遗留 7 个失败均为历史遗留（非本次引入）：
  1. `test_query_parser::test_model_scope_viewer_entity_ids_are_ignored` — sanitizer 未剥离 `raw_json` 中的身份字段
  2. `test_semantic_routing::test_model_ids_are_discarded` — 同上（`parser_raw` 含 attacker 字段）
  3. `test_model_budget::test_gamma_client_role_endpoint` — 断言端点无 `/v1` 后缀，当前实现带 `/v1`
  4. `test_memory_store::test_event_does_not_use_album_provenance_to_cancel_semantic_conflict` — 事件归并 provenance 语义
  5. `test_memory_store::test_same_location_different_activities_do_not_merge_without_shared_evidence` — 同上
  6. `test_event_segmentation::test_album_provenance_does_not_override_semantic_conflict_without_vectors` — 同上（事件 ID 重复）
  7. `test_capture_metadata::test_gps_is_retained_and_reverse_geocode_fills_location` — 依赖在线逆地理编码，测试环境无该服务

## 4. 结构化 Canary（album2_e2b 66 assets，直连 vLLM 8105）

- 结果：**7/8（87.5%）**，与 PhaseB B5 基线一致。
- `tool_selection_rate=100%`、`safety_critical_errors=0`、`silent_fallback=0`、`guard_blocked=1`。
- 两次运行分别为 6/8 与 7/8，差异来自 `exists` 用例的模型措辞抖动（"候选无法完全确认"），属已知保守行为，非代码回归。
- 失败用例：`exists`（模型倾向 hedged 表达）、`media` 复合查询（guard 拦截 count=0 的编造尝试）。

## 5. 线上 QA（8091 异步链路，10 题，scope=album2_e2b）

| id | 问题 | 结果 |
| --- | --- | --- |
| q01 | 去年拍了多少张照片 | complete：2023 年 10 张（已知"去年"解析缺陷，实际 2025 为 45 张） |
| q02 | 最早照片时间 | complete：2023-05-12 ✓ |
| q03 | 2023年5月拍过照片吗 | blocked_by_guard（`fact_exists_contradiction` 保守误拦，已知） |
| q04 | 最近照片桌上放了什么 | complete：真实桌面物品（沙拉、蓝盘、甜品杯、切好的水果） |
| q05 | 照片里有几个人 | blocked_by_guard（L2 judge 拦下"3人" vs 观察"2人"的编造）✓ |
| q06 | 去年十月爬山照片有雪吗 | complete：如实说未找到该记录 |
| q07 | 第一张照片有人穿红衣吗 | complete：如实说没有 ✓ |
| q08 | 招牌/文字写了什么 | complete："被坚定的爱着 / LOVE YOU" ✓ |
| q09 | 人物相关事实条数 | complete：66 条 ✓ |
| q10 | 2024年春天去过公园吗 | complete：有记录 ✓ |

- 全部 10 题异步链路完成（POST turn → 轮询 → 最终结果）；guard 正确拦截 2 例，无编造漏网。
- 结论：与 QA 基线一致，**无回归**。

## 6. 部署状态（153）

- 分支：`psh`（`0ab8929`），工作树干净。
- 实例（全部跑最新代码，vLLM `gemma4-12b-it`@8105）：
  - `8091` 生产 API：默认 `tool_loop`（无 `SENTRIX_AGENT_PROFILE`）
  - `8097` canary：`tool_loop_shadow`（经 `start_sentrix_api_8097_phaseb.sh`）
  - `8098` shadow：`tool_loop_shadow`
  - `4174` Web 网关 → `8091`；已确认前端 JS 为清理后版本（`runAssistantTurn` 存在、`getVlmBackend` 移除）
- 模型切换逻辑未改动，`/api/model-profiles/*` 保持可用（前端仍用 `getModelProfiles`/`switchModelProfile`）。

## 7. 遗留与后续建议

- 7 个历史失败建议单独排期修复（sanitizer 剥离 `raw_json`/`parser_raw`、`/v1` 断言、事件归并 provenance 语义、GPS 在线测试标记 skip）。
- `ResultSetStore.cleanup()`（TTL 清理）目前无人调用，可接入周期任务或删除。
- `backend/agent.py`（MemoryAgent）与 `backend/thin_agent.py` 仅剩测试/benchmark 引用，保留作为回归基准，不建议在生产路径恢复。
