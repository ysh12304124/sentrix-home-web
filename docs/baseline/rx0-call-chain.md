# RX-0 · 当前用户可见回答链审计（Phase RX 前置）

**日期**：2026-08-06　**性质**：只读核验产物（基于当前代码 + 12B-FC 实测 `sentrix-12b-full-chain-cases.json`）
**用途**：RX 阶段的基线事实，说明当前回答如何生成、哪些字段用户可见、内部信息如何泄漏。

---

## 1. 当前回答生成调用图

```text
用户消息
  │
  ├─ ExplicitOperationDetector（protocol 快速路径：no-lookup/写作/反馈/选人）→ mode=none → _normal_chat
  │
  ├─ QueryParser.parse → QueryParseDraft（proposed_mode 仅建议）
  │     └─ Router.route（router.py）→ none | contextual | ambiguous | clarify | evidence
  │           └─ ambiguous → NeutralProbe → resolve_after_probe（upgrade/clarify/no_household_match）
  │
  └─ _evidence_path（thin_agent.py:197）
        ├─ build_query_spec → QuerySpec
        ├─ EvidenceRetrievalKernel.retrieve → EvidencePacket（assets/exact/strong/approximate/gaps/channel_trace）
        ├─ _gate_packet_approximate（recall 强度过滤 + max_count=20）
        └─ _evidence_answer（thin_agent.py:466）
              ├─ person → _person_summary_via_complex_or_fallback（Writer→Claim→Verify→Repair）
              ├─ clothing_gap → 确定性"无法确认"
              ├─ 12B 证据回答 → _validation_evidence_answer（matched/possible/unknown + asset_id 直接进 prompt）
              ├─ 无证据 → 确定性拒答模板
              ├─ compose_answer（语句边界校验）
              ├─ ClaimExtractor.scan（确定性句子切分）
              ├─ _envelope + _image_results（全部 image 证据）
  │
  ▼
app.assistant_response（app.py:640）→ 加 retrievalTrace/toolTrace/evidencePresentation/claims 等
  │
  ▼
前端 src/app.js（assistantMessage/assistantEvidence/imageResults/algorithmEvidence/toolTrace）
```

## 2. 数据流：EvidencePacket → Answer → API → Frontend

1. `EvidencePacket.assets[]`：`asset_id`、`observation_ids`、`condition_results`（`{key:{status}}`）、`level`、`recall_strength`、`near_duplicate_group`、`observation_fields`。
2. `_evidence_answer` 把 `condition_results`/`level`/`asset_id` 拼给 12B prompt 或确定性模板 → `answer`。
3. API envelope：`answer`、`evidence[]`、`image_results[]`、`claims`、`claim_evidence_index`、`retrieval_trace`、`tool_trace`、`evidence_presentation`、`evidence_layers`、`validation`、`model_call_ledger`。
4. 前端渲染：`assistantAnswer`（正文）、`assistantEvidence`（证据卡 + 折叠"查看这次回答的依据"）、`imageResults`（图片）、`algorithmEvidence`/`toolTrace`（内部 trace，无条件渲染）。

## 3. 内部信息进入用户路径的 4 个入口

| 入口 | 位置 | 内容 |
|---|---|---|
| 12B 证据 prompt | `_validation_evidence_answer`（thin_agent.py:611） | `level`、`condition_key=status`（如 `2025年10月=matched`）、`asset={asset_id}` → 模型可原样写进回答 |
| 确定性模板 | `_simple_answer`/`_allowed_facts`（thin_agent.py:647/681） | `找到 N 条完全符合确定条件…`、`记录中有「X」`、英文状态 |
| `_image_results` | thin_agent.py:709 | 全量 image evidence 的 `asset_id`/`media_url` |
| 前端证据卡/trace | src/app.js `evidenceCard`(184)/`algorithmEvidence`(216)/`toolTrace`(222)/`assistantEvidence`(265) | `evidence.raw` JSON、`asset_id`/`observation_id`、`retrievalTrace` stage/counts、tool trace |

## 4. 回答形态决定逻辑（当前）

- 由 `_evidence_answer` 内部分支机械决定：person→人物摘要；clothing_gap→确定性；`SENTRIX_EVIDENCE_ANSWER_12B`→12B 证据回答；else 无证据→拒答模板；else→`_simple_answer`。
- **不读取用户目标**：`spec.actions`/`result_requirement` 只在 `return_assets_requested` 时控制 `image_results`，不改变回答形态（"把原图给我"仍走 12B 证据分析模板）。

## 5. 与 RX 目标的差距

- EvidencePacket 直接进入回答 prompt（违反 DoD#1）。
- `image_results` 与正文两个独立来源 → 文图矛盾可发生（实测 continuous_t3）。
- 无"用户目标→回答形态"映射（缺 ResponsePlan）。
- 无"可见图片子集"选择（缺 VisibleEvidence；近似展示 Top-10）。
- 无回答后一致性校验（缺 Response Validator）。
- 前端无用户/管理员分层。
