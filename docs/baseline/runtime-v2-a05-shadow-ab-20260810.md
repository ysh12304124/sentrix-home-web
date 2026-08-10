# Agent Runtime v2 — Phase A0.5/A0.6/A1-A5 实测与离线 A/B 报告

- 日期：2026-08-10
- 模型：`gemma4-12b-it`（vLLM，4-bit bitsandbytes，max_model_len 4096）
- 数据源：153 `data/sentrix.db`，scope=`album2_e2b`（66 assets；2023 年 10 张；视频 0）
- 代码：本地分支 `psh-runtime-v2`（`0904399` A1、`5d1ed3c` A2-A5），待合并 153 `psh`
- 性质：只读评估 + 新代码；未触碰生产 8091 行为（默认 `SENTRIX_AGENT_PROFILE=pipeline`）

## 1. A0.5 Tool Selection 能力门槛（实测）

- 测试集：55 条中文家庭问题（chat/writing/count/time/place/first_last/find_images/hybrid/all/vague/followup/original）
- 结果（`/home/asus/runtime-v2-a05/result_v3.json`）：

| 指标 | 目标 | 实测 |
|---|---|---|
| Action schema validity | 100% | **100%**（55/55） |
| Primary action accuracy | >= 90% | **98.2%**（54/55，1 例 place 误选 query_memory_facts） |
| Unnecessary tool call rate | <= 10% | **0%** |

- 结论：12B 达到进入 Shadow Runtime 的工程门槛。未达 100% 的 place 类问题进入后续 model/prompt 优化子轨。

## 2. A0.6 inspect_photo 多模态链路（实测）

- 12 张真实图片，image→12B→structured inspection evidence：`valid_json 12/12`，`avg latency 2.9s`（2.09–3.59s）
- 观察到能力：scene / people_count / clothing / objects / visible_text(OCR) / activity / certainty
- 已知失败模式：夜间人群含灯饰误计、低光抽象物体误读（黑色干花→食物）
- 结论：链路真实可用，`inspect_photo` 注册为 Tool（每 turn 1 次，结果 ephemeral 不写库）

## 3. A1-A4 实现状态

- A1：`ConversationStore`+trajectory（服务端 canonical 会话历史）+ `/api/conversation/{id}/messages` + 前端 progress 渲染（`src/app.js:368`，本次补充 `.assistant-progress` 样式）
- A2：`backend/agent_runtime/` 薄循环：`AgentRuntime`（model→tool→observation→final）+ `ToolRegistry`/`ToolPolicy`/`BudgetManager`/`ProfileConfig`
- A3：4 个只读 Tool：`query_memory_facts` / `search_memories` / `get_original_photos` / `inspect_photo`
- A4：`ResultSetStore`（result_set_id + handle 映射，不泄漏内部 asset_id）+ `TaskState` + `FinalGuard`
- Profile 门控：`SENTRIX_AGENT_PROFILE=tool_loop|tool_loop_shadow` 时 `/api/assistant/turn` 走 AgentRuntime；默认 `pipeline` 行为不变

## 4. Shadow Tool-Loop 评估（18 例，`evaluate_tool_loop_shadow.py`）

- 结果：**10 complete / 7 blocked_by_guard / 1 partial**，平均 1.92s/turn（v12 跑数）
- 守卫拦截的 7 例（全部为真实错误，未放行）：

| 例 | 查询 | 工具实际返回 | 模型编造 | 拦截原因 |
|---|---|---|---|---|
| s01 | 去年拍多少张 | count=10 | 128 张 | fact_value_missing |
| s02 | 去年哪些月份 | rows=[9月7,12月2,5月1] | 1-12 月全列 | group_fabrication |
| s04 | 明哥第一次出现 | 2025-04-12 | 2023-05 | fact_date_missing |
| s05 | 去年十月爬山 | total=0 | 声称找到 | fabrication_from_empty |
| s07 | 去年十月上海 | total=0 | 声称找到 | fabrication_from_empty |
| s11 | 去年十月上海都给我 | total=0 | 声称找到 | fabrication_from_empty |
| s15 | 去年几个视频 | count=10(未过滤)/0(视频) | 12 个视频 | fact_value_missing |

- s03（去年主要在哪些地方）：模型两次 group 都拿不到地点维度（未传 `group_by=place`），预算耗尽 → `partial`（诚实无答案，未伪造）

## 5. Shadow 过程中发现并修复的根因

1. **ToolPolicy sanitize 白名单过窄**：`value/rows/filters_applied/operation` 被过滤，模型看不到 count/分组/日期数据 → 只能编造。改为按工具白名单放行（`tool_policy.py`）。
2. **模型输出非纯 JSON**：截断/缺闭合括号/`"filters":{"}`/markdown 围栏 → `_parse_action` 增加修复候选链 + 解析失败带反馈重试 1 次（`runtime.py`）。
3. **FinalGuard 覆盖不足**：新增 first/last/date 日期一致性、group 月份包含性、exists 布尔矛盾、空检索谎称找到（`fabrication_from_empty`）四类确定性检查（`final_guard.py`）。

## 6. 离线 A/B：Tool-Loop vs Canonical RX（18 例同集）

- Canonical RX：`SENTRIX_RX_V1=1` + 全套 RX flags，同 DB/同 Kernel/同 12B（`evaluate_tool_loop_ab.py`，base=8105）
- 实测：RX 18/18 有回答、17/18 走 RX、0 降级，平均 4.24s/turn

### 逐例对照（TL=Tool-Loop，RX=Canonical RX）

| 例 | TL | RX |
|---|---|---|
| s01 count | 守卫拦 128≠10 | 未给数量（"有几张照片记录在案"） |
| s02 time | 守卫拦月份编造 | 未列月份 |
| s03 place | partial（预算耗尽） | 泛化照片 |
| s04 first_last | 守卫拦日期编造 | 仅"记录中提到了明哥" |
| s05 爬山 | 守卫拦空检索谎称 | 交付照片（未核语义） |
| s06 红衣 | 声称找到（condition unknown） | **诚实 no_result** |
| s07 上海十月 | 守卫拦空检索谎称 | 交付照片 |
| s08 水族馆 | 声称找到（unknown） | 交付照片 |
| s09 做饭 | 声称找到（unknown） | 交付照片 |
| s10 全家福 | 声称找到（unknown） | **诚实 no_result** |
| s11 all | 守卫拦空检索谎称 | "已找到并展示 60 张"（fallback，数量与库不符） |
| s12 chat | 直接 final | chat ✓ |
| s13 writing | 直接 final（1 步） | 走 evidence 路径但答出改写 |
| s14 明哥介绍 | **漏报**（检索 8 张却说没找到） | 简答"提到了明哥" |
| s15 视频数 | 守卫拦 12≠10 | **诚实拒绝**（无视频记录） |
| s16 杭州 | 声称找到（unknown） | "3 张相关照片" |
| s17 猫咪 | 声称找到（unknown） | 交付照片 |
| s18 vague | 诚实澄清 | 直接交付照片（未澄清） |

### A/B 结论

- **硬事实可信性：Tool-Loop 占优**。7 个可验证的编造（数字/日期/月份/空检索）全部被 FinalGuard 拦截，最终零错发硬事实；RX 侧 s11 "60 张" 与库内 2023 总量（10）不符且无守卫。
- **结构化问题回答能力：Tool-Loop 占优**。count/月份/首现日期由 `query_memory_facts` 确定性执行；RX 对 count/time/first_last 多为泛化或未答。
- **图片语义声称：两侧同病**。检索命中 `level=approximate, condition=unknown` 时，12B 仍会"找到了 X 的照片"；RX 以 deliver handle 形式同样未核语义（s05/s07/s08/s09/s17）。s06/s10/s15 上 RX 反而更诚实（no_result）。
- **模型自身诚实性仍是主变量**：12B 倾向编造（128、12 视频、全月份、空检索声称找到），并存在 s14 漏报（不作为）——守卫能拦"作为的谎言"，拦不住"不作为的漏报"。
- **延迟**：TL 1.92s vs RX 4.24s（平均每 turn；TL 单次调用 12B，RX 固定多次模型调用）。

## 7. Definition of Done 对照

| 条目 | 状态 |
|---|---|
| Production/RX/frontend baseline 唯一可复现 | ✅ A0 |
| Fresh baseline | ✅ 部分（完整 Python 套件待维护窗口） |
| 12B Tool Selection benchmark | ✅ 98.2%/100%/0% |
| inspect_photo 真实可用 | ✅ 12/12，2.9s |
| 服务端 ConversationStore 可见历史 | ✅ A1（`recent_turns` 注入 + trajectory） |
| Public Action Trace / Progress | ✅ `public_progress` API + 前端渲染 |
| Tool Registry/Policy/Budget 可运行 | ✅ |
| >=3 个 read Tool 进 Shadow | ✅ 4 个 |
| 2-step Tool-Loop turn | ✅（s03/s14 等 search→facts 多步） |
| Agent 自主选 fact Tool | ✅（s01/s02/s04/s15） |
| Agent 自主选 search Tool | ✅（s05-s11/s16-s18） |
| ResultSet→original delivery | ✅（get_original_photos 注册，A4 完整化） |
| FinalGuard 阻止 false fulfillment | ✅ 7/7 编造被拦 |
| Tool readiness 暴露数据覆盖缺陷 | ✅（condition unknown / ANN 缺失 / entity 8.8%） |
| Budget 耗尽诚实结束 | ✅ s03 partial |
| Offline A/B 完成 | ✅ 本报告 |
| flags 收敛为 Profile | ✅ `SENTRIX_AGENT_PROFILE` 门控接入 app |
| Canary 决断 | 见 §8 |

## 8. Canary 结论

- **值得 canary（结构化事实类）**：count / 月份 / 地点分组 / first-last / 视频-照片区分。`query_memory_facts` 确定性执行 + FinalGuard 一致性校验，质量高于 RX 泛化回答，且延迟更低。
- **暂不无条件 canary（图片检索/混合类）**：`search_memories` 在语义条件未验证（unknown）时模型仍过度声称；检索召回在 2023-10+爬山/上海 上返回 0（A0 §9 ANN 索引缺失为高优先级前置）。这类需要：重建 ANN 索引、检索 condition 验证状态显式暴露给模型并让模型照实陈述、12B 语义遵从微调/提示强化。
- 建议迁移路径：先以 `SENTRIX_AGENT_PROFILE=tool_loop` 开独立验证实例（非 8091），灰度结构化事实类查询；图片类继续走 RX 直至语义子轨达标。

## 9. 已知限制与下一步

- 12B 输出 JSON 稳定性仍需重试兜底（本轮 18 例触发 6 次解析修复/重试）。
- FinalGuard 无 "不作为漏报" 检查（s14），建议后续加：search total>0 但答案声称"没找到"时的矛盾检查。
- group 的 place 维度守卫未做（模型须传 `group_by=place` 才返回地点分组；s03 暴露该参数遵从问题）。
- `search_memories` 的 `condition_summary.unknown` 需要工具层把"候选 vs 已确认"分开呈现（readiness 文档化）。
- ANN 索引重建（`data/ann/`）是图片类 Tool 与 RX 共同的前置项。
