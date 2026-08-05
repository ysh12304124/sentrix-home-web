# Phase R6 · Answer 去重、拒答、近似说明

**日期**：2026-08-06
**状态**：完成（433 tests 全绿）

## 交付物

### `thin_agent.py`
| 改动 | 说明 |
|---|---|
| `_allowed_facts` 按 condition_key 去重 | 同 condition 多 asset 命中只出一条 fact，evidence_ids 并集；杜绝 album1-01 "记录支持X"×10 |
| `_human_condition_text(key, status)` | 人类可读模板：matched→"记录中有「X」"、possible→"记录中可能有「X」，但无法完全确认"、unknown→"目前无法确认…"。**内部 condition_key / ANN 分数 / trace / 表名禁止出现在用户可见文本** |
| 空 EvidencePacket 强制拒答 | "当前记忆中没有找到足够匹配的原始证据。"——家庭证据查询空结果**禁止**走 normal chat 编造（§12.4 / P0-16 strict_empty） |

### 说明
- approximate（allow_approximate）仍展示但用 "无法完全确认" 标注差异（P0-16）
- `compose_answer` fallback 在去重后不再重复模板

## 测试
`test_answer_composer_dedup`（5：跨 asset 去重 / 人类可读 / possible 标注不确定性 / compose 去重后无重复 / 无内部 key 泄漏）

## 本地验证
```
unittest discover backend.tests → 433 OK (skipped=1)
```
