# R9-0 · 只读路由与文字规则审计报告

**日期**：2026-08-06
**性质**：R9 第一阶段只读审计——真实路由调用图、运行时文字规则分类、错误路由复现。
**产物**：`runtime_text_rule_inventory.json`（24 条规则，review=0）。

---

## 1. 当前真实路由调用图

```text
POST /api/assistant/turn  (backend/app.py:680)
  → Agent.answer_turn (backend/agent.py:1842; SENTRIX_THIN_AGENT_V1=1 且无 feedback/selected_entity)
  → ThinAgentRuntime.answer_turn (backend/thin_agent.py:62)
      │
      ├─ [thin_agent.py:69] gate.fast_path(message, api_signals)
      │     · feedback→evidence / selected_entity_id→evidence / 写作前缀 / "不用查"→none
      │     · fast none → _safe_fallback() (无 LLM) → _normal_chat  [thin_agent.py:70-72]
      │
      ├─ [thin_agent.py:73] parser.parse(message) → QueryParseDraft
      │     · _PARSER_PROMPT → sanitize_query_parse → _draft_and_validate → repair → 确定性 overlay
      │
      ├─ [thin_agent.py:74] gate.classify(draft) → GateDecision
      │
      ├─ [thin_agent.py:75-76]  decision.mode=="none" → _normal_chat   ← R9-1 移除点
      ├─ [thin_agent.py:77-78]  mode=="contextual" → _contextual
      ├─ [thin_agent.py:79-80]  mode=="ambiguous" → _ambiguous_path
      │     · draft 有 facets/semantic_conditions/time/media → 直接 retrieve (thin_agent.py:105-124)
      │     · 否则 _run_probe (thin_agent.py:126) → kernel.probe → NeutralProbe.run
      │           upgrade → retrieve → _evidence_answer
      │           clarify → 澄清文案信封
      │           其余   → _normal_chat
      ├─ else evidence (thin_agent.py:81-96) → build_query_spec → kernel.retrieve → _evidence_answer
      │     · _gate_packet_approximate(anchored=thin_agent._query_anchored)
      └─ 输出信封 _envelope
```

**关键耦合点**：
1. `thin_agent.py:75` `if decision.mode == "none": return _normal_chat` —— Parser `mode` 一票否决（R9-1 必删）。
2. `memory_gate.classify` 以 `draft.mode` 为主输入；`_explicit_general_task` 用"介绍/解释/为什么/假设"直接判 none。
3. `_WRITING_ANYWHERE_RE` 句中"写"会把家庭查询误判写作。
4. 无命中短语（probe 非 upgrade 且非 clarify 分支）→ `_normal_chat`。
5. `_ANCHOR_*` 在 memory_gate 与 thin_agent 双份实现（R9-0 已单源到 `backend/routing_rules.py`，thin_agent 改 import；memory_gate 合并到 R9-1）。

---

## 2. 运行时文字规则清单

分类规则：**A prompt**（模型提示词）/ **B protocol**（高精度产品操作）/ **C normalization**（确定性格式与硬约束）/ **D semantic_routing**（开放语义词直判 mode，禁止）/ **E semantic_extraction**（开放语义词表，禁止）。

`docs/baseline/runtime_text_rule_inventory.json`（24 条）汇总：

| 类别 | 数量 | 说明 |
|---|---:|---|
| A prompt | 1 | `_PARSER_PROMPT`（含开放语义类别，示例须不含 benchmark 题） |
| B protocol | 4 | `_WRITING_PREFIX_RE`（memory_gate + routing_rules）、`_NO_LOOKUP_RE`（×2） |
| C normalization | 15 | 锚正则/日期/否定/媒体枚举/schema 白名单/probe 阈值映射 |
| **D semantic_routing** | **4** | `_WRITING_ANYWHERE_RE`(runtime)、`_explicit_general_task`(runtime)、agent.py 旧路径 `_is_entity_introduction_query`/`_is_comparison_query`(legacy) |
| E semantic_extraction | 0 | 无硬编码开放语义词表 |
| review | 0 | 全部已分类 |

**runtime D 类 2 条（必须处理）**：
1. `memory_gate._WRITING_ANYWHERE_RE` → decision **remove_or_narrow**。"照片里写着什么？"/"帮我写下那次明哥穿的衣服"含"写"但属家庭查询。
2. `memory_gate._explicit_general_task` → decision **narrow**。"介绍一下明哥"/"为什么去年春节没有小黑的照片"含 general 动词但必须检索。

**legacy D 类 2 条**：`agent.py` 旧非 Thin 对话路径（`SENTRIX_THIN_AGENT_V1` 关闭时生效）用 "介绍/是谁/了解/档案/画像"、"比较/区别/对比" 判意图。Thin 路径不采用；标记 `remove_or_retire`（随 Thin Agent 正式化移除，或保留但加反例测试）。

---

## 3. 错误路由复现（R9 必须修复）

| 输入 | 当前行为 | 根因 | R9 期望 |
|---|---|---|---|
| 介绍一下明哥 | parser-none + 无锚点 + 无 facet → `_explicit_general_task`("介绍一下")→ none → **normal_chat** | general 动词先于 confirmed-entity 判定 | evidence（confirmed 人物可解析 → 复杂人物链） |
| 照片里写着什么？ | 命中 `_WRITING_ANYWHERE_RE`("写") → none → **normal_chat** | 句中"写"误判写作 | evidence/probe |
| 银色心形手镯 | probe 无命中 → **normal_chat**（可能编造产品描述） | 无命中短语缺少 clarify 分支 | clarify |
| 为什么去年春节没有小黑的照片 | parser 若漏日期/人物 → 可能 none | general 动词 + 弱 parser 双重风险 | evidence |

---

## 4. R9-0 交付

- [x] `backend/routing_rules.py`：单源锚正则 + `message_anchored()` + `HOUSEHOLD_DIMENSIONS`（`_ANCHOR_PERSON_TOKENS` 保持无"家人/全家"，避免 R8 写作误触发复发）。
- [x] `backend/thin_agent.py`：删重复锚正则，`_query_anchored` 复用 `message_anchored`（spec 人物/时间/地点 constraint 优先）。
- [x] `scripts/benchmarks/audit_runtime_text_rules.py`：AST 抽取模块级 regex/词表 + inline 成员检查 + 手工清单，分类产出 inventory。
- [x] `docs/baseline/runtime_text_rule_inventory.json`：24 条，review=0，runtime D=2 / legacy D=2。
- [x] 本地 38 个相关测试绿。

**下一步（R9-1）**：Router 解耦——收窄/删除 `_WRITING_ANYWHERE_RE`、`_explicit_general_task` 降为"排除家庭语境后"的兜底、`proposed_mode` 单一化、confirmed-entity 前置、无命中短语→clarify、复用 `dialogue_states` 焦点。完成后重跑审计确认 runtime D=0。
