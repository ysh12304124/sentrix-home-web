# Agent Runtime v2 — Phase C/C8 Search→Inspect→Result UX 完成与测试报告

- 日期：2026-08-10
- 范围：C8（search certainty 与 inspect certainty 分层、inspect 不得反向确认检索条件、representative 预览、选中照片复核展示）
- 153 正式分支：`psh`（HEAD `be648891`，由 `1a331ff` 合并组成）
- 本地工作分支：`psh-runtime-v2`（HEAD `1a331ff`）
- 验证基线：Phase C 报告提交 `85d2dd4d`

## 1. 代码实现（commit `1a331ff`，+210/-18 行）

### 后端：检索层与复核层确定性分离

**`backend/agent_runtime/final_guard.py`**
- 规则 2 `candidate_claimed_as_match`：检索为 `candidate_only` 时，「找到了/确认是」检索条件仍然拦截——inspect 观察存在**不再豁免**；仅当用户明确点选某张照片追问（`selected_asset_handle` 存在）时豁免，此时该照片的视觉回答以 inspect 为准。
- 规则 3 `missing_disclosure`：`partial_support/candidate_only` 必须披露检索层缺口，inspect 视觉观察不替代检索层披露（点选追问除外）。
- 规则 4 `certainty_upgrade` 改为**标签感知**：只有回答中出现未确认条件标签（如「爬山」）且用断言式措辞（「确认是/确定是/肯定是/就是/确认了…」）时才拦截；修复潜在误伤——「我确认照片里没有雪」这类视觉断言不再被当成条件升级。
- 新增条件级否认豁免：`omission_conflict` 不再把「没找到能确认'爬山'的记录」误判为整体否认（C8 目标回答形态）。

**`backend/agent_runtime/tools.py`**
- `_inspect_photo` 返回 `confirms_visual_only: true`，明确标记复核观察只确认照片里直接可见的视觉细节。
- `_search_metadata_only` 支持 `mode=representative`：预览按时间**均匀采样**（`_even_indices` 包含首尾），不再只取最新 6 张；`mode=best/all` 行为不变。

**`backend/agent_runtime/runtime.py`**
- SYSTEM_TEMPLATE 增加 C8 分层规则：检索满足度只描述语义条件是否确认；inspect 只确认照片可见细节，不能反推确认「爬山/去公园」等检索条件；给出目标回答示例。
- `_trusted_facts` / 恢复提示带 inspect handle（「照片 photo_1 复核观察：…」）。
- inspect 调用前推送实时进度「正在检查照片 photo_1…」，结果后推送「已检查照片 photo_1…」（SSE 增量可见）。

**`backend/agent_runtime/result_set.py` / `tool_policy.py` / `emergency.py`**
- TaskState 新增 `selected_asset_handle`（跨轮恢复、进入 guard 上下文）；`record_tool_result` 记录 `inspect_handle` + `confirms_visual_only`。
- sanitizer 放行 `confirms_visual_only`；emergency 摘要带复核 handle。

### 前端
- `src/app.js`：ResultSet 卡片对 inspect 过的缩略图显示「已复核」徽标，并在卡片下方列出复核观察（`photo_1 · 复核：…`）。
- `src/styles.css`：已复核描边 + 徽标样式。
- Work Trace 保持「完成自动折叠 / 失败展开」（C6 已实现，C8 未改动）。

## 2. 测试结果

### 后端（153 psh 工作树，全量，echo venv + hnswlib）
```
738 passed | 3 failed | 4 skipped（745 项收集）
```
- 3 个失败仍是已知事件归并 provenance 语义（用户已拍板不改），非本轮引入。
- 本轮新增 11 个用例（`test_phasec_time_guard_food.py` 22 → 33）：
  - `SearchInspectCertaintyTests`：inspect 不能反向确认检索条件、分层自然回答通过、视觉回答需披露、点选追问豁免、视觉断言不误伤、条件标签断言拦截。
  - `TaskStateInspectHandleTests`：inspect_handle/confirms_visual_only 记录、selected_asset_handle 跨轮恢复。
  - `RepresentativePreviewTests`：`_even_indices` 均匀覆盖首尾、13 资产 representative 预览跨时间分布、has_more/remaining 正确。

### 前端（本地）
```
36 tests | 0 failures
```
- 新增结构性断言：已复核徽标、复核观察区块、inspect_handle 渲染路径、CSS 样式。

## 3. 线上 QA 实测（153 生产 8091，scope=全部相册，C8 代码已重启生效）

| # | 问题 | 结果 |
| --- | --- | --- |
| C8-1 | 去年十月爬山的照片山上有雪吗？ | search→inspect 完整链路；satisfaction=partial_support（2025年 confirmed / 爬山 unknown）；回答分层披露「10 张候选、只能部分确认是去年十月爬山」，复核 photo_1「正在进入洞穴、无雪迹」；SSE 7 事件（含「正在检查照片 photo_1…」→「已检查照片 photo_1…」→ guard 恢复 → 最终）；未把「爬山」反向确认 |
| R1 | 2024年的照片，按时间均匀挑几张 | representative 生效：total=52，preview=photo_1/11/21/32/42/52（均匀跨年、含首尾） |
| R2 | 选中 photo_52「有什么特别的细节？」 | 正确传 selected_asset_handle=photo_52，inspect_photo 调用成功，inspect row 含 `inspect_handle=photo_52` + `confirms_visual_only=true`，回答与观察一致 |
| C8-4 | 2024年一共拍了几张照片？ | 52 张（确定性事实路径，无需披露） |

## 4. 部署状态（153）

- 分支：`psh`（`be648891`），工作树干净。
- 实例全部重启到 C8 代码：`8091`（tool_loop 生产）、`8097/8098`（tool_loop_shadow）、`4174`（Web → 8091），`/api/health` 正常、5 工具 ready。

## 5. 已知问题与后续

- C8-2「每个月份都看看」被模型理解为月份分布统计（query_memory_facts group）而非 representative 检索——语义合理，非缺陷；representative 路径已由 R1 直测覆盖。
- 前端「已复核」徽标由 `task_state.tool_results` 驱动，已通过结构性测试；浏览器视觉确认可后续人工复核。
- 下一轮建议：C9 Guard/Debug UX 分层（普通用户只见自然文案，管理员可见 L1/L2 规则码与恢复步数）。
