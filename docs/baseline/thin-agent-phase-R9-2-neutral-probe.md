# R9-2 · NeutralProbe 重构 — 报告

**日期**：2026-08-06
**测试**：本地 479 通过（R9-1 473 + 新增 6）。

## 1. 交付

| 文件 | 改动 |
|---|---|
| `backend/retrieval/probes.py` | `ProbeOutcome` 增 `channel_agreement / top_candidates / conflicts / index_health`；decision 增 **`no_household_match`**；`run` 接受 `focus / media_hint / index_health`；**无任何通道命中且无 lexical exact → no_household_match**（不再一律 clarify）；`conflicts`=与 shared 集不一致的通道 |
| `backend/evidence_retrieval.py` | `kernel.probe` 返回 **`(channel_hits, index_health)`**，每通道记录 ok/error + hits；接受 `focus/media_hint`（透传）；confirmed-entity 信号由 primary `entity` 检索器覆盖 |
| `backend/thin_agent.py` | `_run_probe` 解包元组 + 传 `focus`（dialogue_states）与 `media_hint`（从 draft.media_expressions 派生 image/media） |
| `backend/tests/test_neutral_probe.py`（新） | no_household_match、agreement+health+candidates、conflicting channels、focus/media_hint 记录、probe 不产事实 |
| `backend/tests/test_gate_probe.py` | `test_no_hits_clarifies` → `test_no_hits_is_no_household_match`（R9-2 语义） |

## 2. 行为

- **upgrade**：≥2 通道有候选 / lexical 全短语命中 / 校准分过线。
- **clarify**：有候选但弱/冲突。
- **no_household_match**：无任何通道候选且无 lexical exact → Router `resolve_after_probe` 判定（明确 general→none，否则→clarify），**绝不编造**。
- Probe 仍不产生 EvidencePacket/家庭事实/confirmed match（守卫测试）。

## 3. 下一步

R9-3：12B Agent Model Profile（resolve_specs 支持 profile + 各角色 model env；`_endpoint_for` 增 ollama backend→11434；role→推理参数表；`evaluate_parser_slots.py` 12B 验收 + 2B 对照）。
