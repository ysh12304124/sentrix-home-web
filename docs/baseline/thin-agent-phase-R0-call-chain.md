# Phase R0 · 真实调用链核验 + 6 case 逐步解释

**日期**：2026-08-06
**工作副本**：`/Users/rm001/Sentrix-Thin-Agent-work-20260805`（grep/read 核验，未改代码）
**前置文档**：`thin-agent-benchmark-findings.md`（6 case 实测）、`thin-agent-benchmark-report.md`（60 case 全表）

## 1. 生产调用链（代码核验结果）

API `POST /api/assistant/turn`（`app.py:680`）→ `agent.answer_turn`（`app.py:684`，`MemoryAgent`）

`MemoryAgent.answer_turn`（`agent.py:1837`）：
```python
if SENTRIX_THIN_AGENT_V1=1 and not feedback and not selected_entity_id:
    return self.thin_runtime.answer_turn(...)
```
→ `ThinAgentRuntime.answer_turn`（`thin_agent.py:27`）

`ThinAgentRuntime` 构造（`thin_agent.py:19-25`）：
```python
def __init__(self, store, gamma=None):
    self.gate = MemoryGate()
    self.kernel = EvidenceRetrievalKernel(store)   # ← 只持 store，无 clip / ANN / search_terms
    self.parser = QueryParser(gamma=gamma)
    self.complex_builder = ComplexAnswerBuilder(gamma=gamma)
```

每轮流程（`thin_agent.py:27-59`）：
1. `gate.fast_path(message, api_signals)` → 写作前缀 / feedback / selected_entity fast-path（`memory_gate.py:45-65`）
2. 否则 `parser.parse(message)`（`query_parser.py:97`）→ 一次 Ollama JSON 调用（`gemma4:12b`），失败一次 repair，再失败 `_safe_fallback()` 返回 `mode=none`
3. `gate.classify(...)`（`memory_gate.py:67-98`）直接读 `draft.mode` 决定 none / contextual / evidence
4. evidence → `build_query_spec`（`query_contracts.py:240`）→ `kernel.retrieve(spec)`（`evidence_retrieval.py:83`）
5. `_evidence_answer`（`thin_agent.py:120`）→ `_simple_answer` 或 `_person_summary_via_complex_or_fallback`

**关键核验事实**：

| 事实 | 证据 |
|---|---|
| Kernel `retrieve()` 只 `list_assets` + `list_observations` 逐对 evaluate | `evidence_retrieval.py:83-122`；无 `retrieval_ann`/`retrieval_indexes`/`search_vectors` import |
| `retrieval_ann.py`（HnswlibIndex）在 backend 里除自身外零调用 | `grep retrieval_ann/HnswlibIndex/create_index backend/*.py` |
| `retrieval_indexes.py`（observation_search_terms）零查询方 | 同 grep |
| `ThinAgentRuntime` 从不持有 clip | `thin_agent.py:19-25`；`app.py:30` 把 `pipeline.clip` 只喂旧 `MemoryAgent` |
| 旧 `agent.py` 走 CLIP 但用 SQLite 逐行余弦 | `agent.py:1065-1066`：`clip.embed_text(query)` + `store.search_vectors(...)` |
| `_contains` 分词 all-match 过宽 | `evidence_retrieval.py:69-76`：`re.findall(r"[\w一-鿿]+")` 长度>1 + `all(term in haystack)` |
| 多值字段 miss=unknown 已修 | `evidence_retrieval.py:158-161` `_OPEN_WORLD_LIST_DIMENSIONS` |
| 每轮模型调用 | 简单路径 2 次 Ollama（parser + answer）；complex 4-5 次（+writer/claim/verify/repair） |
| 生成模型 | `OLLAMA_MODEL=gemma4:12b`（`model_clients.py:194`）；VRAM 驻留 189MB / 8GB |

## 2. 6 个实测 case 逐步解释

### album1-01 · 浅黄色拼接毛绒睡衣自拍 · FAIL（10 张全 FP）

1. Parser（`query_parser.py`）：拆成 `[clothing:浅黄色, clothing:拼接, clothing:毛绒, clothing:睡衣, visual:自拍]` 或类似词粒度（探针实测）。
2. Gate：mode=evidence → 走 kernel。
3. `build_query_spec`：5 条 SEMANTIC constraint（`query_contracts.py:267-269`）。
4. Kernel `_evaluate`→`_condition`：clothing 是多值字段，`_contains` 对"毛绒/睡衣/浅黄"逐条 all-match。因为 `_contains` 把"浅黄色"分词后 `["浅黄","黄色"]` 要求 `all(term in haystack)`，任何 caption/clothing 含"色""毛"的 asset 都可能命中 → 10 张 FP。
5. 真 GT `IMG_4350.JPG`：若 canonical caption 是"卧室睡衣自拍"，字段级"浅黄色/拼接"无法命中；CLIP 向量存在但**根本没被查询** → 该 asset 不一定进 top-10。
6. Answer：`_allowed_facts`（`thin_agent.py:250-258`）对每 asset×每 condition 生成"记录支持X"，10 asset 重复 10 次。

**归因**：C 链（`_contains` 过宽）+ A 链（CLIP 未接）。

### album1-07 · 贵阳夜晚步行街 · PASS（空 GT，mode=none 幻觉）

1. Parser 返回 mode=none（探针多次 none）→ Gate none → `_normal_chat`（`thin_agent.py:61`）生成"贵阳夜生活丰富…建设路花溪路…"。
2. 这是**幻觉**：不是正常聊天，是家庭证据查询但被 parser 误判 none。

**归因**：B 链（parser none 单点终止检索）。

### album2-05 · 上海市自己和王明的照片 · ERROR timeout

1. Complex 路径（人物+地点复合）：parser → kernel → `_person_summary_via_complex_or_fallback` → `ComplexAnswerBuilder`（Writer/Claim/Verify/Repair 4-5 次 Ollama）→ 累计 > 300s。
2. Parser 还一度判 mode=none（探针），若 none 则完全不查。

**归因**：D 链（12B 延迟）+ B 链（none 误判风险）。

### album2-06 · 夜晚车内的明哥搂着我 江西省 · PASS（空 GT，mode=none）

同 album1-07：mode=none 幻觉 → normal chat"那段在江西的深夜…"。

**归因**：B 链。

### album3-01 · 银色心形手镯 · FAIL（mode=none 最坏）

1. Parser 3 次探针：1 次 evidence、2 次 none。实测那次返回 none。
2. Gate none → `_normal_chat`："听起来是一个非常精致的设计…"。
3. **这是最坏失败**：明确照片查询，agent 完全不查记忆。

**归因**：B 链（parser 不稳定）+ D 链（Ollama 波动）。

### album3-14 · 水族馆海豚跃出水面 · PASS（空 GT，mode=none）

同 album1-07：none 幻觉 → 常识描述。

**归因**：B 链。

## 3. 阶段结论

- 6 case 里 3 个"PASS"全是空 GT + parser none 幻觉，**正样本命中 0/3**。
- 失败集中在 **A**（CLIP/ANN/search_terms 未接）、**B**（parser none 单点终止）、**C**（`_contains` 过宽 + answer 去重缺失）、**D**（12B 延迟）。
- 层 A 骨架合同（sanitize / EvidencePacket / Constraint / statement×evidence）无回归。
- 对应 Phase R 四条修复线：R2 接 A、R4 修 B、R3+R6 修 C、R5 修 D。

## 4. 工具

- `scripts/benchmarks/inspect_retrieval_case.py`：query-only / query+expected / asset-only 诊断；`--spec-json` 纯检索回放；`--semantic/--hard/--time/--media/--exclude` 确定性构造（无模型）。
- `scripts/benchmarks/audit_benchmark_cases.py`：60 case answerability + GT 一致性；本地 smoke 已检出 **2 个 GT 不一致**（album3-07 `9/8`、album3-18 `3/1`）与 28 个外部地理依赖 case。

**下一步**：R1A 建立 Retrieval-only runner + 通道消融基线（153 真实数据）；R1B 独立验证 Visual/Text embedding。
