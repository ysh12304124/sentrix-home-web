# Agent 2.1 W2.1 — R1 拆解报告（Reference Resolution vs 真检索 miss）

基线 run：`20260820-003839-agent2-1-ab` ｜ 对象：15 题答错中的 5 个 R1 错误
> 纯分析，未改代码。核对了 runtime_turns、conversation mode、tool_trace、search 参数、agent2_trace 与 DB 存储值。

## 结论先行

**R1-B（引用消解失败）= 3/5，用户假设成立**。且另外 2 个 R1-A 也不是 ANN/索引质量问题，而是"工具参数构建"和"hybrid 检索路径缺口"。

| 题 | 单轮/多轮 | 分类 | 根因 |
| --- | --- | --- | --- |
| b2-01 沙雕前那次 | 多轮 | **R1-B 引用消解** | turn 0 "和孩子的合影"已检索失败(0 结果)；turn 1 重新全库搜 referent 文本，无 conversation 锚定 → F1=0 |
| b2-02 邯郸那个公园 | 多轮 | **R1-B 引用消解** | 同上：turn 0 "夜景照片"失败；turn 1 未用对话上下文，重新搜 → F1=0 |
| b2-03 泰国动物园晚上 | 多轮 | **R1-B 引用消解** | 同上：turn 0 "国外的照片"失败；turn 1 重新搜 → F1=0 |
| q24-01 工作留影拍摄日期 | 单轮 | R1-A 检索路径 bug | place filter "上海普陀区江宁路"元数据可命中 408 张，但 hybrid(mode=best) 返回 0 |
| q26-03 汉堡套餐价格 | 单轮 | R1-A 工具参数构建 | place filter 填了商家名"顶呱呱炸鸡店"，存储无此值 → 0 结果，且重复搜 3 次 |

## 1. R1-B：引用消解失败（3/5，确认）

三题都是 `conversation_context_mode=shared_conversation_id`、2 轮。核对了 turn 1 的 agent prompt：

- 系统确实传了 `当前结果集：rs_xxx，共 0 张` 和 `当前上下文：当前关注人物：孩子`。
- **但 turn 0 本身就检索失败了**（"和孩子的合影"/"夜景照片"/"国外的照片"→ 0 结果），所以 ResultSet 是空的，referent 无锚可解析。
- turn 1 由 Planner 重新声明全新 goal（"Identify the specific photo of Mingming and Lele..."），完全重新全库搜索 referent 文本 → 0 结果。

**判定**：不是"turn 0 有结果但 turn 1 没解析到"那么简单，而是**多轮引用在系统里根本没有被解析**——turn 1 应先把 referent 映射到既有 Event/ResultSet/Asset（利用 turn 0 意图 + 对话历史），而不是重新 semantic search。现状是 conversation 上下文只透传了"关注人物"，没有透传已检索结果/已确认事件。

> 附带发现：turn 0 的模糊查询（"和孩子的合影"等）本身就该能检索到（album3 有大量人物合影），却返回 0——这和第 2 点 q24-01 同属"检索路径对自然语言召回不足"。

## 2. R1-A 两个案例都不是 ANN 质量问题

### q26-03（place 参数错误）
- 工具调用：`search_memories {"query":"顶呱呱炸鸡店 汉堡单人套餐 价格","filters":{"time":"2022-7-16","place":"顶呱呱炸鸡店"}}`
- 同一天同一菜单：q26-01 用 `place="上海市闵行区青杉路"` 检索成功(f1=1.0)。
- DB 实测：album3 里 **不存在 o.place="顶呱呱炸鸡店"**（place 都是场景类型"快餐店/奶茶店/餐厅内部"），reverse_geocode label="上海市闵行区"。place filter 填商家名 → 硬过滤 0 命中。
- Agent 还用同样参数重复搜了 3 次（step_0/2/3），撞预算。
- **修复方向**：工具参数构建 guidance（place 应填地址/区名，不是商家名）+ 检索 0 结果的自动降级（去 place 重试 / 用商家名走 entity 通道）。

### q24-01（hybrid 检索路径缺口）
- 工具调用：`search_memories {"query":"上海普陀区江宁路店铺 拍摄 工作留影合影","filters":{"place":"上海普陀区江宁路"}}`（无 time filter，合理，因为用户在问日期）
- DB 实测：按 `_place_clause` 的 SQL 复现，place="上海普陀区江宁路" 能命中 **408 张** album3 照片（普陀区 district instr 匹配）。数据侧完全能命中。
- 但 live hybrid 检索(mode=best) 返回 **0** 张。→ **问题在 hybrid 检索路径**（top-k 候选生成后硬过滤，或 place 过滤与 query 检索通道组合有缺口），不在数据、也不在模型 query 构建。
- **修复方向**：单独调试 mode=best 对"place-only 过滤 + 自然语言 query"的召回；可能需要 place 过滤先于/并行于 ANN，或非空 query 时也走 metadata 通道兜底。

## 3. 对 W2.2/2.3 决策门的影响

- **R1-B = 3/5 → 立项 Reference Resolution 修复（优先级最高）**。最小改动：多轮 turn 1 先解析 referent（selected_asset / current_result_set / focus_event / recent_result_set / conversation history），命中则直接基于已有结果作答或在其上增量检索，不重新全库搜。
- **R1-A 两个案例不碰 ANN**：
  - q26-03 → 工具参数 guidance + 零结果降级；
  - q24-01 → 单独排查 hybrid 检索路径（这是一个**高价值单轮 bug**：query 正确、数据匹配，却 0 召回）。
- 建议 W2.3 拆成两个子任务：W2.3a 引用消解、W2.3b 检索路径修复（q24-01 类）。

## 4. 待办/证据缺口

- b2-01/02/03 的 turn 0 模糊查询为何 0 召回，需顺带确认（可能与 q24-01 同根因）。
- q26-03 的"零结果降级"是否会改变 Agent 行为（会）——需用户确认作为 capability 修复进入实现，符合"可以改"授权。
