# Agent Runtime v2 — Phase C 代码完成与测试结果报告

- 日期：2026-08-10
- 范围：C1（时间解析）/ C2（Guard 恢复）/ C3（自然失败文案）/ C4（地点聚合）/ C5（饮食活动）/ C6（Work Trace）/ C7（图片结果 UX）/ C12（Profile 收敛）/ C15（图片与回答联动）
- 153 正式分支：`psh`（HEAD `848851f1`，由 `ffed0304` 合并组成）
- 本地工作分支：`psh-runtime-v2`（HEAD `ffed030`，6 个提交）
- 验证基线：上一轮清理提交 `1d64977`

## 1. 代码实现总览（6 个提交，+1174/-161 行）

### 提交 1：`c9746bd` — Phase C 主提交（C1-C7/C12/C15）

**C1 统一时间基准与相对时间解析**
- 新增 `backend/agent_runtime/time_context.py`：固定 `Asia/Shanghai`（可用 `SENTRIX_TIMEZONE` 覆盖），`now()` 返回带时区当前时刻，`current_time_line()` 注入系统提示，明确「相对时间一律由系统换算，模型只把原话写进 filters.time」。
- `backend/query_contracts.py::parse_time_expression` 支持年/月范围；`backend/agent_runtime/tools.py::_resolve_time_expression` 支持去年/今年/前年/这两年/去年X月（含中文数字）/季节/上月/最近一年。
- 修复「去年十月」中文数字月份解析。

**C2 Guard 恢复循环 + C3 自然文案**
- 新增 `backend/agent_runtime/guard_types.py`：`GuardResult` 结构化（`GuardIssue`：code + 自然 message + revision + tool_ref + trusted_facts），`REVISION_REWRITE_ONLY` / `REVISION_HARD_BLOCK` 两级修订策略。
- `final_guard.py`：`_natural_message()` 把内部规则码转成用户可读、模型可执行的恢复文案（D4：用户不看到内部规则名）；修复 exists 误拦（hedge 表达放行、明确否认拦截）；新增 `placeholder_leak` 规则（见提交 5）。
- `runtime.py`：Guard 失败后进入恢复循环——提取 TaskState 可信事实（`_trusted_facts`）、注入恢复提示「只重写回答，不重新做昂贵 Tool」、`max_guard_retries=1`；恢复预算耗尽才进入 `blocked_by_guard`。
- `emergency.py`：紧急/部分回答文案自然化。

**C4 地点聚合**
- `structured_memory.py`：place 分组改为 geocode 城市优先（CASE 表达式），`_query_memory_facts` group=place 返回 `known/unknown_location_assets` coverage。

**C5 饮食/活动总结**
- `tools.py`：`query_memory_facts` 新增 `operation=meal`，事件级去重，返回 `explicit_foods / meal_scene_events / possible_events` 分层证据；sanitizer 白名单放行这些字段。

**C6 Agent Work Trace（实时 SSE）**
- `runtime.py`：进度事件化（`{stage, step_index, timestamp}`），写入 `progress_events`。
- `app.py`：新增 `GET /api/assistant/turn/{id}/events` SSE 端点（`text/event-stream`，增量推送）。
- `server.js`：`/events` 或 `text/event-stream` 走流式代理。
- `src/app.js`：`EventSource` 订阅 + SSE 失败回退轮询；Work Trace 用 `<details>` 折叠（成功收起 / 失败展开）。
- `src/styles.css`：details 折叠样式。

**C7/C15 图片结果 UX v2 + 选中联动**
- `app.py`：`TurnRequest` 增加 `selected_asset_handle / selected_result_set_id`，`_tool_loop_turn` 注入选中上下文。
- `tools.py`：原图 URL 带 `original=1`。
- `src/app.js`：ResultSet 卡片修复（`?张` / `还有0张` / 下一页）、可点击缩略图 → selected 状态 → 下一轮带 handle → 原图按钮。

**C12 Profile 与工具 readiness**
- `/api/health` 返回 agent profile + 5 个工具的 readiness。

### 提交 2：`596ed94` — 全部相册 scope 修复
- `tools.py` 不再把空 scope 强转 `home-default`（前端「全部相册」模式）；search/facts/meal/result-page/original/inspect 尊重 `all_authorized`。
- inspect/result-set scope 检查在全部授权时跳过；meal SQL scope 子句条件化；实体解析器回退所有 scope。
- `/api/health` 幂等注册工具，首个 turn 前即可见 readiness。

### 提交 3：`a15bca8` — geocode 优先 + meal sanitizer + 时间过滤契约
- place 分组信任层级：`reverse_geocode.city` → observation place → GPS marker → 未知（C4）。
- sanitizer 放行 meal 字段（`explicit_foods/event_count/tiers`）；meal 返回 `answer_type/value/total`。
- `search_memories` 描述要求时间放 `filters.time`（不再从 query 文本猜）。
- `SYSTEM_TEMPLATE`：地点覆盖披露 + meal 分层使用规则；内部 trace 含工具参数（管理员调试用）。

### 提交 4：`bd704a1` — 空查询 metadata 检索 + 结果截断
- 空 query 搜索（仅时间/地点条件）绕过 ANN 多检索器（空 query 召回 0），走纯硬筛选 `_search_metadata_only`，带 `full_support`。
- place group 观察截断到 top-12 行（`rows_truncated`），稳定 12B 输出。
- `SYSTEM_TEMPLATE`：meal 回答必须列出 explicit_foods 及出现次数。

### 提交 5：`ffed030` — 占位符泄漏 L1 Guard + meal 可信事实恢复
- L1 拦截含未填充模板占位符（`[地点名称1]`/`[数量]`/`[X]`）的回答，走可恢复重写。
- 恢复可信事实包含 meal explicit_foods，重写时列真实食物而非含糊 hedge。
- judge trace 暴露 faithful/problems（管理员调试）。

### 提交 6：`e17a60b` — 前置清理（承接上轮）
- sanitizer 剥离模型 echo 的身份字段（`raw_json`/`parser_raw`）；更新过时的 `/v1` + GPS 测试断言。

## 2. 测试结果

### 后端（153 psh 工作树，全量）
```
727 passed | 3 failed | 4 skipped（153 psh 工作树，echo venv + hnswlib，734 项收集）
```
- 新增 `backend/tests/test_phasec_time_guard_food.py`（+21 用例）：时间解析（去年/这两年/去年十月中文数字）、Guard 恢复（exists 误拦修复、占位符泄漏）、地点覆盖披露、meal 分层证据。
- 3 个遗留失败全部是事件归并 provenance 语义（`test_event_segmentation` ×1 + `test_memory_store` ×2），非本轮引入，用户已拍板「事件归并逻辑现在是对的，不用改」。
- 附注：若运行环境缺 `hnswlib`，另会多出 12 个 ANN 相关失败（`ModuleNotFoundError`），属环境问题非代码回归；本次重跑已安装 hnswlib 排除该因素。

### 前端（本地）
```
35 tests | 0 failures
```
- 新增 `test/phase-c-agent-ux.test.js`：SSE 事件订阅、Work Trace 折叠、ResultSet 卡片、selected handle 传递等结构性断言。

## 3. 线上 QA 实测（153 生产 8091，scope=全部相册）

| # | 问题 | 结果 | SSE 事件 | 延迟 |
| --- | --- | --- | --- | --- |
| Q1 | 去年去过哪里？ | 城市列表（杭州150/绍兴34/济南28/深圳20/北京12/上海4）+ 「12 张照片没有可靠地点信息」覆盖披露 | 5 | 7.0s |
| Q2 | 这两年吃过什么？ | 饮料7/咖啡4/茶4/蛋糕4/菜3/点心2/tea1/汤1/沙拉1/烤肉1（首次模型编造火锅 → guard 拦截 → 可信事实重写为真实食物） | 5 | 5.4s |
| Q3 | 2023年5月拍过照片吗？ | 「是的，系统记录显示…有拍摄过照片」（exists 误拦已修复） | 4 | 2.5s |
| Q4 | 去年春天去了哪里？ | 杭州100 条 + 体育场馆等地点（占位符被 L1 拦截 → 恢复后列真实地点） | 5 | 5.4s |
| Q5 | 找一些2024年的照片 | 52 张，preview photo_1..4，ResultSet 卡片正常（total=52，无 `?张`） | - | 4.0s |
| Q6 | 选中 photo_1「这张照片里有几个人？」 | 正确传 selected_asset_handle，调用 inspect_photo；回答保守「无法确定具体人数」 | - | 4.0s |

说明：
- Q1/Q2 的 progress 中可见「结果里有一处信息对不上，我正在重新核对。」→ 即 Guard 恢复循环真实生效（拦截 → 可信事实 → 重写）。
- Q6「无法确定」是 inspect 模型保守或图中信息不清晰，如实披露，未进代码。

## 4. 部署状态（153）

- 分支：`psh`（`848851f1`），工作树干净。
- 实例（全部最新代码，vLLM `gemma4-12b-it`@8105）：
  - `8091` 生产 API：默认 `tool_loop`（无 `SENTRIX_AGENT_PROFILE`）
  - `8097` canary：`tool_loop_shadow`
  - `8098` shadow：`tool_loop_shadow`
  - `4174` Web 网关 → 8091，前端 JS 为最新（EventSource 订阅 + 轮询回退）

## 5. 已知问题与下一轮建议

- Q6 inspect「无法确定」：可能是图片本身无人或 inspect 模型保守，如实披露，不进代码。
- `_search_metadata_only` 目前只走 `result_requirement` 默认 best；all/representative 需补充（C8 覆盖）。
- 下一轮：C8 Search→Inspect→Result UX（search certainty 与 inspect certainty 分开、失败路径自然化、trace 自动折叠）。
