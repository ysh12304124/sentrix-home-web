# Phase B B0 — Fresh Baseline 与环境冻结报告

- 日期：2026-08-10
- 基线：153 `psh` = `bc46460`（含 runtime-v2 A1-A5）；本地分支 `psh-runtime-v2`
- 目标：冻结环境、跑通完整测试矩阵、产出可复现基线数字，并暴露上一阶段遗留问题

## 1. 冻结环境（B0 实测）

| 项 | 值 |
|---|---|
| 153 psh HEAD | `bc46460` |
| production 8091 | 运行中（12:15 启动，进程仍是合并前旧代码）；`pipeline` profile |
| 8092 validation | 运行中但 **LLM 后端 8100 已下线 → 当前退化**（规则回退，structured QA 0/40 的原因） |
| 8097 Phase B 实例 | 新启动（本阶段交付物）：`SENTRIX_AGENT_PROFILE=pipeline|tool_loop`、模型 `http://127.0.0.1:8105/v1` gemma4-12b-it、RX+structured+conversation flags |
| 12B 模型 | `gemma4-12b-it`（vLLM bitsandbytes 4-bit）@ 8105，ctx 4096，max_tokens 800 |
| GPU | 单卡 24GB；空闲 ~22GB；12B 占用 ~8GB |
| `data/ann/` | 空（visual/text/semantic/episodic 索引全部缺失） |
| `memory_vectors` | 2528 行（重建素材存在） |
| DB | `data/sentrix.db`：assets 747、memory_vectors 2528、scope `album2_e2b` 66 assets |
| 前端 | `src/app.js`/`api.js`，Node 测试 31/31 |

## 2. Fresh Baseline 结果

### 2.1 后端完整测试（153 venv，692 tests）
- **7 failures + 3 errors + 2 skipped**；在基线提交 `8792d3b`（worktree）上复跑同一批模块结果**完全一致** → 全部为历史遗留/环境问题，非 runtime-v2 引入
- 失败项：`test_geocoding`×3（离线地理索引）、`test_event_segmentation`×1、`test_memory_store`×2（事件合并）、`test_model_budget`×1（`/v1` 后缀期望）、`test_capture_metadata`×1、`test_semantic_routing`+`test_query_parser`×2（sanitizer 未剥离 `raw_json` 内 identity 字段）
- 本地环境不可用于全量：缺 `httpx`/`hnswlib`（153 venv 齐全）

### 2.2 前端测试
- `node --test`：31/31 通过；`compileall backend scripts/maintenance`：干净

### 2.3 Structured Memory QA（8097 直连，40 例）
- **31/40 passed**：routed 0.975、exact 0.85、no_visual 0.975、no_images 0.975、no_leak 1.0、mention 0.9
- 9 例失败集中在：`exact_value`（person/entity 类如"女儿乐乐"）、`mentions_value`（答案未带数字）
- 8092（8100 已死）上同脚本为 0/40 → 全部归因于模型通道，非逻辑问题

### 2.4 Tool Selection（55 例，8105）
- schema validity **100%**（55/55）；primary action **98.2%**（54/55）；unnecessary tool call **0%**
- 与 A0.5 数字完全一致，确认 8105 与 8100 行为等价

### 2.5 Shadow 18（8105，scope=`album2_e2b`）
- **10/18 complete / 7 blocked_by_guard / 1 partial，avg 1.9s/turn**（与 A5 一致）
- **raw action JSON repair 基线：50%（36 个模型输出中 18 个需修复）**
- 失败模式分布：markdown 围栏 14/18、未闭合字符串 2、other 2（`analyze_action_repair.py` 产出）
- 修复涉及案例：s02/s03/s05/s06/s07/s08/s09/s10/s14/s15/s16/s17/s18（13 例）

## 3. B0 暴露的关键回归（已修复）

**A1 引入：pipeline 路径 `recent_turns` 崩溃**
- `backend/app.py` 无条件向 `agent.answer_turn(..., recent_turns=...)` 传参，但 `MemoryAgent.answer_turn` 无该参数 → 当前 psh 代码下 pipeline 路径每个请求 TypeError
- 8091 未炸只因进程是合并前旧代码；一旦重启生产即全挂
- 修复：`backend/agent.py` `answer_turn` 增加 `recent_turns=""` 并转发 `thin_runtime.answer_turn(recent_turns=...)`；8097 验证结构化 QA 31/40 正常

## 4. 基线产物

| 文件 | 说明 |
|---|---|
| `/tmp/b0_tool_selection.json` | Tool Selection 55 例明细 |
| `/tmp/b0_shadow.json` | Shadow 18 例明细（含 raw 输出） |
| `scripts/benchmarks/analyze_action_repair.py` | repair 率/失败模式分析工具（本阶段复用） |
| `scripts/benchmarks/evaluate_tool_selection.py` + `tool_selection_set.json` | A0.5 脚本纳入仓库（可复现） |
| `scripts/runtime/start_sentrix_api_8097_phaseb.sh` | Phase B 验证/canary 实例启动脚本 |
| `docs/baseline/structured-qa-e2e.json` | 结构化 QA 明细 |

## 5. 结论与 B1 入口

- 基线成立：模型行为与 A5 一致；完整测试矩阵首次在 153 跑通并确认遗留失败集合
- **B1 前必做**：修复 8092（8100 下线）或明确弃用 8092 改用 8097；`data/ann/` 重建
- B2.3 主攻方向明确：repair 50% 中 78% 是 markdown 围栏 → guided_json/提示词可大幅消除
- structured QA 9 例失败与 person/entity 覆盖相关，纳入 canary 范围裁剪依据
