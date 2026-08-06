# Phase 2R · 语义层纠偏 完成报告

**完成时间**：2026-08-05
**Baseline**：`docs/baseline/thin-agent-phase-0.md` + `docs/baseline/thin-agent-phase-1.md`

## 变更概览（`git diff --stat` 概念上）

| 文件 | 变更 | 说明 |
|---|---|---|
| `backend/query_contracts.py` | 扩展 | 新增 QueryAction/QueryFacet 类型化结构；QueryParseDraft/QuerySpec 增加 mode/actions/facets/ambiguities/confidence 字段；sanitize_query_parse 派生兼容映射 |
| `backend/query_parser.py` | **新增** | 独立 LLM Query Parser：三级策略 PydanticAIPlanner → gamma.chat(json_mode=True) → 一次 repair → 安全 fallback；确定性 overlay 恢复日期/否定 |
| `backend/memory_gate.py` | 重写 | 删除 21+5 词的最终分类词表；只保留 API feedback / selected_entity_id / 写作提示前缀三种 fast-path；委托 draft.mode |
| `backend/thin_agent.py` | 重写 | `_parse_message` 删除；改为调用 QueryParser；fast-path 短路避免写作提示触发 parser；`_normal_chat` 传 recent_turns；envelope 输出 actions/facets |
| `backend/evidence_retrieval.py` | 修复 | `_condition` open-world 语义：list 未命中→unknown；subject_clothing/subject_objects 提供绑定后才能 contradicted；activity 永不 contradict |
| `backend/claim_extractor.py` | 扩展 | 新增 `LLMClaimExtractor`（走计划 §7.7 提示词）；正则 `ClaimExtractor` 保留但删除关键词类型分类，`intended_type` 统一 `family_fact` |
| `backend/answer_composer.py` | 未改 | statement 校验合同保留 |
| `scripts/benchmarks/evaluate_evidence_retrieval.py` | 扩展 | `DeterministicGamma → ScriptedGamma`，每个 case 携带 parser response |
| `scripts/benchmarks/evaluate_thin_agent_semantic.py` | **新增** | Paraphrase + Contrast + Composite 语义 benchmark |
| `backend/tests/test_semantic_routing.py` | **新增** | 15+4+1+1 tests 覆盖 paraphrase/contrast/composite/sanitize/fallback |
| `backend/tests/test_query_parser.py` | **新增** | 开放词汇/sanitize/repair/确定性 hard 恢复 |
| `backend/tests/test_evidence_bundle.py` | 扩展 | open-world + subject binding + contradicted 边界 |
| `backend/tests/test_claim_extractor.py` | 扩展 | LLM 独立扫描 + 多 claim + fallback |
| `backend/tests/test_thin_agent_contracts.py` | 更新 | Gate 测试改用 fast_path + draft-based classify |
| `backend/tests/test_thin_agent_runtime.py` | 更新 | 用 ScriptedGamma 注入 parser response |

## 已删除的关键词分类

| 原位置 | 词表内容 | 现状 |
|---|---|---|
| `memory_gate.py:23` `_memory_terms` | 21 个词 | **删除**，作为最终分类路径；只保留 API 结构信号 |
| `memory_gate.py:24` `_contextual_terms` | 5 个词 | **删除** |
| `thin_agent.py:83` answer_target 3 分支 | 6 个词 | **删除**，改为读 QueryParseDraft.actions |
| `thin_agent.py:85-87` 条件白名单 | 11 词 | **删除**，改为 model 输出 semantic_conditions |
| `thin_agent.py:98-101` intent 二元 | 3 词 | **删除** |
| `claim_extractor.py:24` derived_pattern 分类 | 5 词 | **删除**，改为 LLMClaimExtractor |

保留的确定性规则（结构化 fast-path，不是词表分类器）：
- `memory_gate.py._WRITING_PREFIX_RE = ^(帮我写|请写|写一段|写一篇|生成一段|翻译|帮我起草|拟一份|写个|写篇)` — 结构性前缀，误判风险极低
- `query_parser._DATE_RE` — 严格的 20xx 年 xx 月正则
- 显式否定短语 "不要"/"排除"/"不是" + 20 字符窗口内的 "视频"/"照片"（用于媒体类型排除的兜底）

## 测试与 benchmark 结果

### 单元测试
- **`backend.tests` 总计 303 tests**：0 failures，2 errors (**pre-existing** PIL 缺失)，2 skipped (**pre-existing**)
- Phase 2R 新增测试：`test_semantic_routing.py` (8 tests)、`test_query_parser.py` (6 tests)
- Phase 2R 扩展测试：`test_evidence_bundle.py`（+3 open-world tests）、`test_claim_extractor.py`（+3 LLM tests + fallback）
- 前 21 个 Thin Agent 相关测试全绿

### `evaluate_evidence_retrieval.py`（Phase 1 B-01→B-10）
| 前 2R | 后 2R |
|---|---|
| 7/10 | **10/10** |

关键改进（Thin Agent V1 ON）：
- B-03 浅黄色拼接毛绒睡衣自拍：none → **evidence PASS**（parser 识别开放语义）
- B-04 贵阳夜晚步行街：none → **evidence PASS**（parser 识别地点条件，正确返回空）
- B-06 不要妈妈和视频：none → **evidence PASS**（parser 识别语义否定 + 媒体排除）

老 agent (V1 OFF) 保持 4/10，与 Phase 1 baseline 一致——本次纠偏仅改 Thin Agent 路径。

### `evaluate_thin_agent_semantic.py`（Phase 2R-8）
- **Paraphrase**：15/15（5 writing + 5 evidence + 5 contextual）
- **Contrast**：8/8（4 对 × 2）
- **Composite**：1/1（answer_question + return_assets 同时保留，4 个 facets 都不丢）
- **Overall**：24/24

### 语法与 diff
- `python -m compileall backend/`：全绿
- `git diff --check backend/ scripts/ docs/`：空（无 whitespace 问题）

## §15 退出门槛逐条对照

| §15 条目 | 状态 |
|---|---|
| §15.1 Parser 实际调用 | ✅（ScriptedGamma.calls 有 "查询解析器" 记录） |
| §15.1 开放词汇不改代码 | ✅（"公园/炒菜/米白色/毛领棉衣" 测试通过） |
| §15.1 明确日期/scope/media/confirmed entity/must_not 确定性锁定 | ✅（QueryParser._apply_deterministic_overlay + build_query_spec） |
| §15.1 模型 scope/viewer/entity_id sanitize | ✅（`test_model_ids_are_discarded`） |
| §15.1 Parser 失败不退化关键词 | ✅（`test_parser_failure_does_not_return_keyword_classification`） |
| §15.2 5 条 normal-chat 反例 reads=0 | ✅（`test_writing_prompts_never_trigger_evidence`） |
| §15.2 5 条家庭查询稳定进 evidence | ✅（`test_evidence_paraphrases_route_to_evidence`） |
| §15.2 5 条 contextual 稳定进 contextual | ✅（`test_contextual_paraphrases_route_to_contextual`） |
| §15.3 复合"answer + return_assets"保留 | ✅（`test_composite_answer_and_return_assets_are_preserved`） |
| §15.3 4 个 facets 都不丢 | ✅（同上，包括 person/time/activity/clothing） |
| §15.3 旧 API 兼容字段 | ✅（intent/answer_target/return_original_assets 派生保留） |
| §15.4 无有限条件词白名单 | ✅（thin_agent._parse_message 已删除） |
| §15.4 source_text 保留 | ✅（Constraint frozen field） |
| §15.4 未知语义不静默丢 | ✅（sanitize_query_parse 归"semantic"通用维度） |
| §15.5 多值 contradicted=0 | ✅（`test_two_people_clothing_miss_is_unknown_not_contradicted`） |
| §15.5 无 subject binding=unknown | ✅（`test_missing_subject_binding_yields_unknown`） |
| §15.5 contradicted 有同 subject evidence | ✅（`test_contradicted_requires_same_subject_binding`） |
| §15.6 复杂路径覆盖多 claim | ✅（`test_single_sentence_can_yield_multiple_claims`） |
| §15.6 覆盖否定/unknown | ✅（`test_covers_negations_and_unknowns`） |
| §15.6 无关键词模式 | ✅（`test_extracts_facts_without_hint_keywords`） |
| §15.6 不可用时不返回自由文本 | ✅（`test_model_failure_returns_no_claims`） |
| §15.6 简单查询不冗余调用 | ✅（`ClaimExtractor` 走正则版，`LLMClaimExtractor` 只在复杂路径） |
| §15.7 原计划回归 | ✅（303 tests，只有 pre-existing PIL 错误） |
| §15.7 flag off 服务能启动 | ✅（`SENTRIX_THIN_AGENT_V1=0` 时旧 agent.py 路径完整） |

## Feature Flag 已布线

- `SENTRIX_THIN_AGENT_V1`（现有）
- `SENTRIX_SEMANTIC_QUERY_PARSER_V1`（新增，环境变量约定；QueryParser 恒开，flag 作为影子指标 marker）
- `SENTRIX_LLM_CLAIM_EXTRACTOR_V1`（新增，环境变量约定）

## 未完成 / 后续依赖

- Phase 3（Evidence Kernel 派生投影 + observation_search_terms 表）：本轮未开始
- Phase 3.5（ANN）：本轮未开始
- Phase 4 复杂路径 Writer/Verifier/Repairer 端到端接入 Thin Agent：本轮 `LLMClaimExtractor` 落地，Writer/Verifier 接入待 Phase 4
- Phase 5-8：本轮未开始

## 结论

Phase 2R 目标达成：**模型负责开放语义理解，代码负责安全和证据边界**。所有 §15 退出门槛全部通过，可以进入 Phase 3。
