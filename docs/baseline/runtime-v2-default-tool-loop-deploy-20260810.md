# Sentrix Agent Runtime v2 — 默认 agent 切换 tool-loop + 部署实测报告

**日期：** 2026-08-10
**代码基线：** 153 `psh` = `59ad64e`（`merge(agent-runtime-v2): default agent profile -> tool_loop`）
**本次范围：** 后端全部重启到最新代码、修复 8091 模型通道退化、验证模型切换逻辑、默认 agent 由 `pipeline` 切为 `tool_loop`、网页链路（4174→8091）实测 QA。

---

## 1. 本次代码变更

| 变更 | 说明 |
|---|---|
| 默认 profile 切换 | `backend/agent_runtime/profile.py:58` `active_profile()` 默认 `pipeline` → `tool_loop`；`backend/app.py` 4 处默认值同步（`_tool_loop_turn`、`assistant_turn` 分发、异步 job 记录、profile 上报） |
| pipeline 保留 | `PROFILES["pipeline"]` 原样保留；`SENTRIX_AGENT_PROFILE=pipeline` 显式启用即回退；`8097` 实例继续 pipeline |
| 工件同步 | `docs/phaseb/agent_profile_manifest.json` 的 `default` 更新为 `tool_loop`（`scripts/benchmarks/emit_phaseb_artifacts.py` 同步） |
| 提交链 | 本地 `psh-runtime-v2` `859055b` → 153 `psh` 合并 `59ad64e` |

**默认 tool_loop 生效后的行为**：未显式设置 `SENTRIX_AGENT_PROFILE` 的实例（如 8091）走异步 turn（POST 立即返回 `turn_id`，前端轮询 `GET /api/assistant/turn/{turn_id}`），由 AgentRuntime 自主执行 5 个只读工具 + L1/L2 guard。

---

## 2. 部署与实例状态（153）

| 实例 | Profile | 说明 | 状态 |
|---|---|---|---|
| `8091` | tool_loop（默认） | 4174 网页后端，生产主链路 | ✅ gamma=8105，异步 turn 生效 |
| `8097` | pipeline（显式保留） | pipeline 基线/验证实例 | ✅ gamma=8105 |
| `8098` | tool_loop_shadow | Phase B canary | ✅ gamma=8105 |
| `4174` | — | 前端 Node（静态服务，无需重启） | ✅ HTTP 200，经 8091 转发正常 |

**本次修复的存量问题**：旧 8091 进程 env 缺失 `SENTRIX_VLLM_BASE_URL`，`GammaClient` 落到默认 `http://127.0.0.1:8100/v1`（该端口 vLLM 已下线）→ 之前网页表现为规则退化回答（如“我在听”）。重启时注入 `SENTRIX_VLLM_BASE_URL=http://127.0.0.1:8105/v1` 后恢复正常真实模型回答。

---

## 3. 模型切换逻辑验证（新代码中生效）

- **代码链路**：`chat_fn` 闭包每次调用读取全局 `gamma`（`backend/app.py:1255`）；`runtime_tools.bind_runtime` 每个 turn 重新绑定当前 `gamma`；L2 judge 复用同一 `chat_fn`（`backend/agent_runtime/runtime.py:275`）→ 切换模型后 tool-loop 全链路自动跟随。
- **API 实测**：`POST /api/model-profiles/switch {"profile":"gemma4-12b-it","dry_run":true}` → `accepted=true`，runtime `base_url=http://127.0.0.1:8105/v1`、`model=gemma4-12b-it`、`backend=openai`、state profile 一致。
- **运行实证**：8091/8097/8098 三个实例 health 均上报 vlm endpoint=`http://127.0.0.1:8105/v1`（vLLM gemma4-12b-it 4-bit 实际进程）。

---

## 4. 实测 QA 结果（8091 网页链路 = tool_loop，scope=`album2_e2b`，66 assets）

| ID | 类型 | 问题 | 回答摘要 | 状态 | 耗时 |
|---|---|---|---|---|---|
| q01 | count | 去年拍了多少张照片 | “2023年一共拍了10张照片。” | complete | 3.5s |
| q02 | first | 最早的一张照片是什么时候拍的 | “2023-05-12 拍摄” | complete | 3.5s |
| q03 | exists | 2023年5月拍过照片吗 | “已确认存在相关记录” + 事实校验未通过披露 | blocked_by_guard | 2.8s |
| q04 | inspect_object | 最近照片桌上放了什么 | 沙拉盘/蓝色盘子/甜点杯/切开水果 + 候选披露 | complete | 19.1s |
| q05 | inspect_people | 照片里有几个人 | “20 张候选、14 张未查看、相似候选未确认”；复核为 2 人 | blocked_by_guard | 11.6s |
| q06 | inspect_scene | 去年十月爬山照片山上有雪吗 | 诚实回复未找到爬山记录 | complete | 12.8s |
| q07 | inspect_clothing | 第一张照片有人穿红色衣服吗 | 未看到红色衣服 | complete | 15.6s |
| q08 | inspect_ocr | 招牌或文字写了什么 | “被坚定的爱着”“LOVE YOU” | complete | 17.7s |
| q09 | count_facts | 人物相关事实数量 | “66 条” | complete | 3.5s |
| q10 | place | 2024年春天去过公园吗 | 有去过公园的记录 | complete | 2.8s |

**统计：10/10 全部返回回答；结构化 5/5；图片 inspect 5/5（真实调用 inspect_photo）；guard 拦截 2 例并如实披露。**

**对照（8098 tool_loop_shadow 同 10 题）：结果与 8091 完全一致**（q01/q02/q04/q06-q10 complete，q03/q05 blocked_by_guard，latency 2.8-19.1s）。

**切换前 8091 pipeline 对照（同 scope）**：

| ID | 问题 | pipeline 回答 | 差异 |
|---|---|---|---|
| p01 | 去年拍了多少张照片 | “找到 10 条完全符合确定条件的照片记录” | 只报计数、无自然语言；同样解析为 2023 |
| p02 | 最早照片时间 | “找到 1 条” | 未给出具体日期 |
| p04 | 2024春公园 | “找到 3 条” | 未确认是否去过 |
| p05 | 照片里有几个人 | “未找到已确认人物” | 意图路由错误（路由到人物介绍） |
| p06 | 红色衣服 | “未找到已确认人物” | 同上，答非所问 |

---

## 5. Guard 实弹效果（本次 QA 直接观测）

1. **L2 judge 拦截编造**（q05）：工具观察明确“照片里有两个人”，模型最终回答为“3 个人”→ `judge_contradiction` 拦截，前置披露“相似候选、未完全确认”，未将编造输出给用户。
2. **L1 规则保守拦截**（q03）：`fact_exists_contradiction: expected=True` —— 模型称“已确认存在”，事实校验期望为 True，但回答未满足披露要求 → 拦截并披露“因回答未通过事实校验提前结束”。
3. **候选状态披露**（q04/q05）：检索仅为候选时，回答主动携带“基于相似度候选、还不能完全确认”，不再把 candidate 说成 match。

---

## 6. 已知问题（如实记录）

1. **“去年”时间解析缺陷**：q01 两个实例均答“2023 年 10 张”，真实去年（2025）为 45 张（album2_e2b 按年分布：2023×10、2024×11、2025×45）。模型对查询结果的复述是忠实的，guard 不拦截 → 属时间表达式解析的当前日期基准问题，建议下一步优化。
2. **L1 规则存在保守误拦**：q03 真实存在却因 `fact_exists_contradiction` 被拦（答案未满足披露模板），可考虑放宽 exists 类规则。
3. **pipeline 路径意图路由问题**：图片细节/人物计数类问题路由到人物介绍并返回 gap（p05/p06）——已通过默认 tool_loop 规避；pipeline 仍保留供回退。
4. **153 工作区游离文件**：`backend/runtime.py`、`backend/agent_runtime/app.py`、`backend/agent_runtime/evaluate_result_set_e2e.py`、`backend/agent_runtime/evaluate_search_inspect_e2e.py`、`backend/result_set.py`、`memory.db`（0 字节）为早期开发旧拷贝，未跟踪未清理。

---

## 7. 回退方式

- **单实例回退**：8091 重启时设 `SENTRIX_AGENT_PROFILE=pipeline`。
- **代码回退**：`git revert 59ad64e`（153 psh）或全部实例显式设置 profile。
- **模型通道回退**：`POST /api/model-profiles/switch` 切换到 registry 中其他 profile（e2b/lora/qwen3），8091 默认 tool_loop 会自动跟随新 gamma。
