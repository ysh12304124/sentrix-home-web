# Sentrix Agent2 闭环修复完成报告

日期：2026-08-25  
权威节点：`192.168.0.153`  
模型：`gemma4-12b-it`，8100，12B，并发 12

## 1. 结论

本轮计划已完整落地并在 153 完成验证：

- 人物身份已进入 `search_memories` 的模型可见预览和 Agent2 证据账本；不暴露 face id、embedding 等私有字段。
- 检索候选、模型选择、最终图片展示已分层；8771 不再把整个搜索候选集自动当成用户要看的图片。
- Qwen/vLLM tokenizer 的 502/连接型预检失败会重试一次并降级到保守本地估算；硬性 4xx 仍按配置决定是否阻断。
- 8771 已提供按“相册 + 模型基座”隔离的复用选择，并修复历史跨模型复用污染导致的基座错配。
- 本轮 12B `album3-max / 100qa-full` 复用回归已完成 100/100，结果和轨迹均已落盘。

质量结论仍需实事求是：结构和数据链路问题已修复，但 100QA 的最终质量仍不达既定质量门，主要剩余瓶颈是检索召回/规划与证据不足，不是并发。

## 2. 本轮实际修改

### 2.1 人物身份进入 Agent 上下文

涉及：`backend/agent_runtime/tools.py`、`backend/agent_runtime/runtime.py`、`backend/retrieval/entity.py`。

`_preview_entry` 从只读人脸映射读取已确认姓名和家庭角色，输出紧凑字段：

```json
{"name":"明明","family_role":"孩子","identity_status":"confirmed"}
```

待确认或无法确认的簇不会被伪装成确定身份。`record_agent2_tool_evidence` 将这些信息作为 `photo_identity` 证据写入 ledger。另修复了 `observation_search_terms` 表存在但为空时提前返回的问题，并兼容 observation 中的字典型 `people`，因此 album3-max 按“明明”检索可实际命中。

153 真实烟测：按 `明明` 检索命中 10 张照片，前 3 张预览均包含已确认人物字段。

### 2.2 候选与最终图片选择分离

涉及：`backend/agent_runtime/runtime.py`、`jit_prompt.py`、`tools.py`、`backend/app.py`、`services/photobench/backend/benchmark_orchestrator.py`。

- 搜索预览增加 `priority_rank` 和 `selection_reason`，第一张是相关性最高，其余是事件多样性补充。
- Final action 允许显式 `selected_image_handles`，只接受当前结果集中的句柄，去重并限制最多 6 张。
- 8771 的图片提取只信任显式 selected handles/asset ids、answer grounding 或明确 image delivery；忽略搜索的 `debug_asset_ids` 全候选集合。
- `selected_image_handles` 与 `selected_asset_ids` 写入 `answer_grounding`，便于轨迹审计。

### 2.3 tokenizer 预检 502 降级

涉及：`backend/model_clients.py`、`backend/tests/test_model_clients.py`。

`/tokenize-current` 遇到 5xx/连接型瞬时失败重试一次，仍失败则使用基于字符/JSON 的保守估算，并标记 `budget_source=local_estimate`、`preflight_status=fallback`、`preflight_fallback_reason`。正常路径标记 `vllm_tokenize/ok`；4xx 仍遵守 `SENTRIX_TOKEN_BUDGET_REQUIRED`。

### 2.4 8771 复用相册基座

涉及：`services/photobench/backend/benchmark_orchestrator.py`、`services/photobench/frontend/src/App.vue`。

`GET /api/memory-spaces` 新增 `reuse_bases`，按精确 `album_id + model_profile` 聚合 scope 和来源 run。scope 名称与历史来源 run 双重校验，避免同一个 scope 被污染后把 e2b 基座误显示成 12B。复用模式直接绑定已有 scope；full 模式保持原有建库链路。

153 当前接口返回 113 个空间、16 个复用基座；album3-max 的 `gemma4-12b-it` 正确指向 `album_cba01be9502b`，没有被 e2b 历史记录污染。前端构建已通过并由 8771 dist 提供。

## 3. 153 验证结果

### 3.1 自动化测试与编译

在 153 执行：

```text
python3 -m unittest \
  backend.tests.test_photo_identity_readonly \
  backend.tests.test_agent2_evidence_recording \
  backend.tests.test_agent2_production_contract \
  backend.tests.test_entity_retriever \
  services.photobench.tests.test_image_extraction \
  services.photobench.tests.test_suite_control \
  backend.tests.test_model_clients
```

结果：`Ran 94 tests ... OK`。随后 `compileall -q backend services/photobench/backend` 通过。

### 3.2 12B 100QA 回归

Run：`20260825-113704-album3-max-gemma4-12b-it-reuse-88204c`  
模式：`reuse`；scope：`album_cba01be9502b`；数据：`album3-max / 100qa-full`。

| 指标 | 结果 |
|---|---:|
| 完成题数 | 100/100 |
| Answer quality mean | 0.546 |
| Exact accuracy | 0.227 |
| Core accuracy | 0.320 |
| Retrieval recall mean | 0.090 |
| Retrieval precision (micro) | 0.480 |
| Retrieval recall (micro) | 0.039 |
| Retrieval F1 (micro) | 0.072 |
| JSON parse success rate | 0.905 |
| 平均 Agent loop calls | 6.4 |
| Planner fallback | 0 |
| Token preflight fallback | 0（797/797 次模型调用均 `vllm_tokenize/ok`） |
| 证据覆盖记录 | 693（其中 partial 409） |
| terminal `task_complete` / `insufficient_evidence` | 47 / 64 |

### 3.3 轨迹与图片选择审计

该 run 的 `results.jsonl` 中：

- 100/100 有 answer grounding。
- 86 个题目的搜索预览至少包含一条已确认人物信息；实际出现的姓名包括 `我`、`芳芳`、`明明`、`乐乐`、`王建国`、`张晓莉`、`雪儿`、`强子`。
- 809 个预览条目均带 rank/selection reason；模型看到的候选最多 6 张。
- 16 个题目显式选择图片；16/16 的 `predicted_images` 与 `answer_grounding.selected_asset_ids` 完全一致，0 个候选泄漏到最终展示。
- 工具调用统计：`search_memories` 146、`inspect_photo` 47、`get_result_page` 31、`query_memory_facts` 29、`read_photo_text` 14；工具总体没有因本轮改动产生批量失败。

原始轨迹和汇总位于 153：
`services/photobench/results/20260825-113704-album3-max-gemma4-12b-it-reuse-88204c/`。

## 4. 当前仍未解决的质量问题

100QA 结果说明本轮修复解决了数据链路和可审计性问题，但不会自动把低召回变成高召回：

1. 召回仍是主要瓶颈：micro recall 仅 0.039，许多题先得到宽泛候选，缺少直接支持。
2. `insufficient_evidence` 仍多于 `task_complete`；模型在证据不足时能更诚实地拒答，但对可回答题仍会过早停止。
3. `read_photo_text` 在该回归中成功率低于其他工具，日期/文字问题仍需更强的受控 OCR 路由。
4. 16 个题目产生显式图片选择，说明“只展示模型选择图片”的链路已经生效；其余题目没有显式图片需求，因此不应自动把候选图全量展示。

这些是下一轮质量优化项，不应回滚本轮已验证的身份、状态、轨迹、预算降级和复用基座修改。

## 5. 交付状态

- 代码已同步到 153，8091 与 8771 已重启加载最新代码。
- 8091 健康检查通过；8771 复用基座接口与前端构建通过。
- 94 个定向测试、编译检查和 100QA 全量回归均在 153 完成。
- 本轮可直接上线的修改：人物身份注入、候选/展示分离、tokenizer 502 降级、8771 复用基座；无需 shadow 层。
- 质量门结论：结构闭环通过，质量门尚未通过；下一阶段应按题型对低召回、过早停止和 OCR 失败做针对性修复。
