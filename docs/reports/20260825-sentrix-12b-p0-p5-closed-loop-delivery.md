# Sentrix Agent2 12B：P0–P5 闭环交付报告

日期：2026-08-25  
权威代码、数据与测试环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）  
模型：`gemma4-12b-it`（12B，vLLM bitsandbytes，`max_num_seqs=12`）

## 结论

P0–P5 已完成并在 153 验证。当前可交付版本已修复记忆生成截断、人物字段丢写、旧视觉参数格式、检索描述缺失、证据类型/轨迹错配等代码级问题；检索索引已重建，100QA 三种候选窗口策略均完成，修复后的 head-only 与 event-diversity 也各完成一轮全量 100QA。

建议生产默认使用 `head_only`：在同一 12B、同一相册、同一 QA 条件下，最终 head-only 的回答质量和 exact 较高；`event_diversity` 的检索 precision/recall 略高，适合作为宽泛找图或候选过度集中时的备用策略。两者都不能单独解决回答层的事实完整性问题。

## 已发现并已解决的问题

### P0 代码合同与轨迹

1. `search_memories` 预览曾只有 handle，没有描述。现在每个预览显式返回 `evidence_summary` 与 `description_status`，空描述和传输缺失可区分。
2. 结果集完整列表与模型可见候选窗口分离；窗口策略可通过 `SENTRIX_CANDIDATE_STRATEGY` 做 A/B，服务端仍保留完整 ResultSet。
3. 证据类型编号/展示错配已修复。最终 head-only 513 条 Agent2 证据账本记录与需求类型逐条核对，类型错配为 0。
4. 视觉/OCR handle 必须来自当前结果集可见 preview。最终 head-only 115 次视觉/OCR 调用中，越界 handle 为 0。
5. 少数模型仍会产生旧参数 `image_id=rs_xxx, query=...`。兼容层现将其绑定到当前 preview 合法 handle，并将 query 转为 question；最终 head-only 触发并修正 3 次，越界为 0。

### P1 记忆生成与写入

1. 根因：`VISION_CORE_NUM_PREDICT` 默认 320，完整 observation JSON 经常被截断，`parse_json_response` 得到 `{}`，造成“模型调用成功但记忆为空”。已提高默认预算到 800，并保留环境变量覆盖。
2. 根因：`MemoryStore.enrich_observation()` 漏写 `people_json`，12B 已生成的人物字段只进入 canonical，无法被人物检索使用。已补写 `people_json` 并增加回归测试。
3. 12B 补全任务两轮均完成且失败为 0；最终覆盖率（363 张/363 条 observation）：caption/activity/place/detail 100%，objects 99.45%，people 76.86%，clothing 47.11%，spatial_relations 63.09%，OCR 42.42%。人物、衣着和 OCR 仍有真实视觉不确定性，空值未被编造填充。
4. `observation_search_terms` 已按 363 条 observation 重建，索引覆盖率 100%。

## 100QA 验证结果

所有运行均为 `album3-max` 的 `100qa-full`、复用 scope `album_cba01be9502b`、12B 当前模型、并发 12；不同策略只改变候选窗口策略。

| 运行 | 策略 | answer quality | exact | core | retrieval recall | retrieval precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260825-003251...66158a` | head-only（修复后最终） | 0.810 | 0.320 | 0.490 | 0.718 | 0.125 | 0.199 |
| `20260825-004354...4ec3b1` | event-diversity（修复后最终） | 0.768 | 0.313 | 0.455 | 0.724 | 0.129 | 0.205 |
| `20260825-235838...6006e3` | default（修复旧参数前） | 0.768 | 0.283 | 0.485 | 0.713 | 0.126 | 0.200 |

修复后最终 head-only：JSON 解析成功率 0.990，任务判断准确率 0.685，99/100 Judge 有效；event-diversity：JSON 解析成功率 0.996，99/100 Judge 有效。完整召回但 Judge=0 的比例仍约 0.49–0.50，说明主要剩余问题已从“找不到图”转向“候选找到了但答案不完整/过度推断/事实选择错误”。

## 当前剩余问题与原因判断

1. **检索 precision 仍低（约 0.125–0.129）**：事件相册中大量近重复照片被召回，候选窗口虽有策略差异，但搜索排序本身仍偏宽；这是 P1 检索质量问题，不是并发问题。
2. **完整召回仍可能答错**：full-recall AQ0 约 45%–50%。典型表现为年份选择错误、地点只回答大区域、视觉细节过度具体、数量/布置回答不完整。根因分为结构化事实与视觉观察混用、模型在多个相似候选中选错、以及 writer 没有严格按证据边界收敛。
3. **候选闭合不等于事实闭合**：Agent2 常以 `candidate_closure` 终止，但需求仍有 `open/partially_supported`，最终回答可能过早结束。需要把“候选已遍历”和“核心需求已满足”分成两个终止条件。
4. **OCR/inspect 仍有低价值重复**：最终运行约 110 次视觉/OCR 调用，部分题目会先拿到大候选集再反复尝试；后续应让工具结果携带“不支持原因/已检查候选”，减少重复调用。

## 下一阶段执行计划

### P1（优先）

1. 为 `search_memories` 增加问题类型感知的 precision rerank：时间/地点/事件硬约束先过滤，再用视觉摘要和事件多样性排序；保留完整 ResultSet，不把窗口截断当作检索结果。
2. 对 temporal/location/structured fact 建立确定性冲突检查：当 observation 时间、事件摘要、用户问题约束冲突时，禁止 writer 直接选择单一年份/地点，转为列出候选或明确无法确认。
3. 将 Agent2 终止条件改为：所有核心 requirement `satisfied` 才允许 `candidate_closure` 直接收尾；否则必须继续补证据或输出受约束 partial。
4. 强化 inspect prompt：先描述可直接观察属性，再回答角色/语义判断；对“不确定”输出结构化 uncertainty，禁止用角色常识补全衣着、人数和装饰。

### P2

1. 对人物、衣着、OCR 低覆盖字段做增量补全队列和质量抽样，不在每次请求时同步调用 12B。
2. 增加按 QA 角度的回归集（年份、地点、数量、OCR、视觉服装/布置、找图）和候选窗口 A/B 自动报告。

### P3

1. 将检索 precision、full-recall AQ0、open requirement rate、legacy-argument rate、handle 越界率纳入 8771 发布门禁。
2. 继续保留完整 trace：planner、tool arguments（规范化前后）、tool result、evidence ledger、writer/judge 输入必须可逐条关联。

## 验证记录

- 远端 `backend.tests.test_model_clients`：27 tests OK。
- 远端 `backend.tests.test_result_set_contracts`：29 tests OK；本地同组 30 tests OK。
- 远端 `backend.tests.test_entities`：49 tests OK。
- 153 8091 重启后 Qdrant：346 collections、45,330 points，健康探测通过。
- 最终 head-only 轨迹：证据类型错配 0，视觉/OCR 越界 handle 0，旧参数规范化 3 次且均成功。

详细原始审计文件：

- [最终 head-only 审计](20260825-gemma12b-final-head-only-audit.json)
- [最终 event-diversity 审计](20260825-gemma12b-final-event-diversity-audit.json)
- [12B 后记忆覆盖率审计](20260824-memory-coverage-album-cba-after-12b-people-fix.json)

