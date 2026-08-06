# R9-1 · Parser 与 Router 解耦 — 报告

**日期**：2026-08-06
**性质**：Parser `proposed_mode` 降为建议；Router 确定性最终决策；写作/general 规则收窄；单一 mode 字段；复用 dialogue_states 焦点。
**测试**：本地 473 通过（基线 448 + 新增 25）。
**审计**：`runtime_text_rule_inventory.json` runtime semantic_routing=0、semantic_extraction=0、review=0。

---

## 1. 交付

| 文件 | 改动 |
|---|---|
| `backend/routing_rules.py`（新/扩展） | 单源锚正则、写作前缀、`_WRITING_COMPOSE_RE`（句内写作结构，排除"写着/写下"）、`_CONCEPT_VERB_RE`（含"介绍一下"，但仅作 Router 最后兜底）、follow-up 标记、`HOUSEHOLD_DIMENSIONS`、`message_anchored/has_general_verb/is_writing_compose/has_household_signal` |
| `backend/router.py`（新） | `GateDecision`（从 memory_gate 迁入）、`ExplicitOperationDetector`（feedback/selected_entity/不查/写作前缀，protocol）、`Router.route`（8 步决策树）、`resolve_after_probe`、`RouteDecision.as_gate_decision` |
| `backend/memory_gate.py` | 降为薄包装（fast_path→Detector、classify→Router）；删除 `_WRITING_ANYWHERE_RE`、`_explicit_general_task`、本地锚正则 |
| `backend/query_contracts.py` | `QueryParseDraft.proposed_mode` 唯一可写；`mode` 改为派生 property；sanitize 不再从 action 强制 mode |
| `backend/query_parser.py` | `_safe_fallback`→proposed_mode；`_draft_and_validate` 移除"evidence 无 action→repair"，改为**结构性自洽修复**（家庭信号但 actions 空→repair 一次，与 mode 无关） |
| `backend/thin_agent.py` | answer_turn 用 `Router.route` 替换旧 gate 编排；`_ambiguous_path` 改为 probe 分流（upgrade/clarify/no_household_match）；新增 `_evidence_path/_merge_focus/_load_focus/_save_focus/_message_entity_ids`；复用 `store.get/save_dialogue_state` |
| `scripts/benchmarks/audit_runtime_text_rules.py` | 修正 AST `re.compile` 抽取、`__all__` 过滤、inline 扫描；补新符号分类；删 `_explicit_general_task` 手工条目 |

## 2. Router 决策树（最终顺序）

```text
1. 显式操作（feedback / selected_entity / 不查 / 写作前缀）        → 直定
2. 写作前缀 + 无家庭语境                                           → none
2.5 句内写作结构 + parser-none + 无家庭信号                        → none
3. 强家庭信号（显式证据 action / 强 target answer / 时间 / 媒体 / 否定 / entity_names） → evidence
3.5 parser 提议 contextual + 无显式证据诉求                        → contextual
4. confirmed person（draft entity_names/人物 facet 或 原文命中已确认实体） → evidence
5. 会话后续（dialogue_states 焦点 + follow-up 标记）                → evidence(focus_ids)
6. general 概念（parser-none + 无家庭信号 + general 动词 + 无锚点）  → none
7. 弱家庭信号（锚点 / facets / semantic_conditions / 裸名词短语）   → ambiguous(probe)
8. 无信号                                                        → ambiguous(probe)

resolve_after_probe: upgrade→evidence / clarify→clarify /
  no_household_match→(明确 general→none，否则→clarify)
```

**关键修复（对应 10 项修订）**：
- `_WRITING_ANYWHERE_RE` **删除**；"照片里写着什么？"（media facet）→ evidence；"帮我写下那次明哥穿的衣服"→ 明确。句内写作由 `_WRITING_COMPOSE_RE`（"写一篇/起草/编个…"）处理，且仅在 parser-none + 无家庭信号时生效 → "我想写一篇明哥的虚构故事"→ none。
- general 动词（含"介绍一下"）**只作第 6 步兜底**，且前提是 parser-none + 无家庭信号 + 无锚点；confirmed-entity 检查（第 4 步，含原文命中）在 general 之前 → "介绍一下明哥"（确认人物）→ evidence；"介绍一下家庭相册这个产品概念"→ none。
- 无命中短语（银色心形手镯/燕园/八戒）→ **clarify**（`resolve_after_probe` no_household_match + 无明确 general → clarify），绝不进入普通聊天。
- 单一 `proposed_mode`；`mode` 仅序列化层派生。
- Focus 复用 `db.py` `dialogue_states` 表（active_entity_ids/active_event_ids/unresolved_ambiguity），不再新增平行 SessionFocus；thin 路径读写同一来源。

## 3. 测试

- 新增 `backend/tests/test_router_decision.py`（25 断言）：写作零检索、句内写作、confirmed 人物介绍→evidence、general 动词不判 none、概念问题→none、裸名词→probe、probe 分流、会话后续 focus 复用、contextual 保持。
- 新增 `backend/tests/test_runtime_text_rule_audit.py`：inventory runtime D/E=0、review=0、protocol/normalization 有测试引用、legacy 语义词表必须标 remove。
- `test_gate_probe.py` / `test_thin_agent_contracts.py`：构造改 `proposed_mode=`，person-facet 行为按新合同更新。

## 4. 审计结果

`runtime_text_rule_inventory.json`（21 条）：runtime semantic_routing **0**、semantic_extraction **0**、review **0**；仅 legacy `agent.py` `_is_entity_introduction_query` / `_is_comparison_query` 2 条标 `remove_or_retire`（Thin 路径不采用）。

## 5. 阻塞与下一步

无代码阻塞。下一步 R9-2：NeutralProbe v2（`no_household_match` / 会话焦点 / media hint / index_health / confirmed-entity 通道）。
