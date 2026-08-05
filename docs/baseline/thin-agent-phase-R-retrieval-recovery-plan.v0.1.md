# Sentrix 检索可用性恢复执行计划 (Phase R)

**版本**：0.1 (draft, 待用户确认)
**日期**：2026-08-06
**输入**：`Sentrix_检索可用性恢复与通用化设计_规划输入报告.md` (下称"输入报告")
**基线**：`docs/baseline/thin-agent-benchmark-findings.md` (6 case 实测 3 pass / 2 fail / 1 timeout，正样本命中率 0/3)
**工作副本**：`/Users/rm001/Sentrix-Thin-Agent-work-20260805`，交付到 `asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web` 走 `psh` 分支
**执行原则**：本文件为计划稿，用户点头前**不改任何生产代码**；除 R0 的只读核验和测试骨架外，每阶段结束需 153 上单元测试全绿 + Retrieval-only benchmark 通过再前进下一阶段

---

## 0. 不变量与本阶段范围冻结

保留（输入报告 §0 + 原计划保留项）：
- Asset 是最终证据源
- `QuerySpec` / `Constraint` 三层 strictness / `EvidencePacket` / statement×evidence 校验 / `sanitize_query_parse` 剔除模型越权字段
- 模型负责开放语义，代码负责 scope/viewer/时间/人物/媒体/否定/证据边界
- 普通聊天不读家庭事实、未授权不重读原图、未授权不改记忆
- `raw_json` 不动、FMA 5173 不动、e2b_server 相关 uncommitted 改动不动

本阶段范围冻结（输入报告 §3）——以下模块**保留代码但不作为完成声明**：
- Core Memory 上线宣告
- Correction 端到端上线宣告
- 高级人物/事件/时间线工具完成宣告
- 主动回忆、多 viewer、Answer Writer 风格优化、Formation pipeline 大规模改造

本阶段允许修改的文件：
`memory_gate.py`、`thin_agent.py`、`query_parser.py`、`query_contracts.py`（兼容扩展）、`evidence_retrieval.py`、`retrieval_indexes.py`、`retrieval_ann.py`、`answer_composer.py`、Agent 模型调用编排、benchmark/replay 脚本、相关 API trace 与 feature flag、单元与集成测试。**新增**目录 `backend/retrieval/`（多路 retriever）。

---

## 1. 当前调用链真实核验（本地代码 grep 结果）

**核验方法**：`grep` 于 `/Users/rm001/Sentrix-Thin-Agent-work-20260805/backend/*.py`，交叉验证 evidence_retrieval / thin_agent / retrieval_ann / retrieval_indexes 的实际 import 与调用点。

| 事实 | 证据 |
|---|---|
| `EvidenceRetrievalKernel.retrieve()` 仅逐对扫描 Asset×Observation，未走任何索引 | `evidence_retrieval.py:83-122`：只 `store.list_assets` + `store.list_observations`；无 `retrieval_ann`/`retrieval_indexes`/`search_vectors` 引用 |
| `backend/retrieval_ann.py` 建了 `HnswlibIndex`，但 `backend/*.py` 除自身外**零调用** | `grep "retrieval_ann\|HnswlibIndex\|create_index" backend/*.py` 结果只出现在 `retrieval_ann.py` 自己 |
| `backend/retrieval_indexes.py` `observation_search_terms` 派生表**建了但无查询方** | 同 grep，除自身外零调用；`evidence_retrieval.py` 未 import |
| `ThinAgentRuntime` 构造只接受 `store` + `gamma`，**从未持有 clip** | `thin_agent.py:__init__`（无 clip 参数）；`app.py:30` 用 `pipeline.clip` 只喂给旧 `MemoryAgent` |
| 旧 `agent.py` 走 CLIP 通道，用 SQLite 逐行余弦扫描 | `agent.py:1065-1066`：`self.clip.embed_text(query)` + `store.search_vectors("episodic",...)` + `search_vectors("semantic",...)` |
| Kernel `_contains` 分词后 `all(term in ...)` | `evidence_retrieval.py:69-76`：`re.findall(r"[\w一-鿿]+")` + 长度>1 过滤 + `all()`。中文双字组合仍然过宽 |
| `_condition` 对多值字段 miss=unknown 已修 | `evidence_retrieval.py:158-161`：`_OPEN_WORLD_LIST_DIMENSIONS` 已定义 |
| 生产 API 8091 flag 全开（含 `SENTRIX_ANN_INDEX_V1=1`）但 kernel 从未调用 | flag on 只使 ANN 索引**能被创建/保存**，不代表 retrieve 路径读它 |

**核心失败面**（输入报告 §1 的四条独立链，均得到证据支撑）：

- **A. 生产检索路径不完整**：已建 ANN/视觉向量/文本向量/派生投影都不在 `EvidenceRetrievalKernel.retrieve` 里。
- **B. Gate 单点失败**：Parser 返回 `mode=none` 后 `thin_agent._normal_chat` 直接接管，短家庭短语（"银色心形手镯"/"八戒"）失去检索机会。
- **C. 语义匹配错误**：`_contains` 双字 all-match 造成 album1-01 命中 10 张无关；`_allowed_facts` 无按 condition 去重 → "记录支持 X" 重复 10 次。
- **D. 模型基础设施不可用**：Ollama gemma4:12b `size_vram=189MB/size=8GB` → 单次调用 9-90s 波动，端到端 240s（硬上限 20s）。

---

## 2. 四条修复线的接口与验收

四条线**分别修复、分别测量**，不允许"接上 CLIP 其他会自然好"这类简化结论（输入报告 §1）。

### 2.1 修复线 A · 生产检索路径接通

**代码位置**：
- 新增目录 `backend/retrieval/`，模块化多路 retriever（见 §5.1）
- 改造 `backend/evidence_retrieval.py::EvidenceRetrievalKernel.retrieve` 分为 prefilter → recall → merge → condition → postfilter → fusion → level (输入报告 §4 目标调用链)
- 新增 `backend/retrieval_query.py` 定义 `RetrievalQuery` / `CandidateHit` / `HardFilterContext` 数据类
- 生产 API `retrievalTrace` 每阶段附 `channel_counts: {metadata, entity, lexical, visual_ann, text_ann, adjacency}`

**接口**（输入报告 §6.1）：
```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]: ...

@dataclass(frozen=True)
class CandidateHit:
    asset_id: str
    retriever: str          # "metadata" | "entity" | "lexical" | "visual_ann" | "text_ann" | "adjacency"
    score: float            # 通道内归一化前的原始分（fusion 层再做 RRF/归一化）
    source_id: str          # observation_id / event_id / asset_id
    source_revision: str | None
    matched_text: str | None
    metadata: dict          # {"field": "clothing", "distance": 0.23, ...}
```

**验收**：
- `retrievalTrace.channels` 至少出现 `metadata`、`entity`、`lexical`、`visual_ann`、`text_ann` 五路的候选计数
- 每个进入 EvidencePacket 的 asset 至少能被回指到一个 retriever（`hit.retriever` 字段回填到 packet.asset.attributions）
- `SENTRIX_RETRIEVER_<NAME>` 单 flag off 可让该 retriever 停摆但整体不崩

### 2.2 修复线 B · Gate 不再是硬二元

**代码位置**：`memory_gate.py`、`thin_agent.py`、`query_parser.py`、`query_contracts.py`

**接口**（输入报告 §5）：
```python
@dataclass(frozen=True)
class GateDecision:
    proposed_mode: str        # "none" | "contextual" | "evidence" | "ambiguous"
    memory_relevance: float
    general_task_confidence: float
    household_query_confidence: float
    allow_probe: bool
    reason: str
```
- `MemoryGate.decide(message, draft, request_context)` 返回 `GateDecision`，不再直接返回 mode 字符串
- `ThinAgentRuntime.answer` 根据 `proposed_mode` + `allow_probe` 决定是否进入 Retrieval Probe
- Probe = 低成本 Kernel 调用（只跑 metadata + lexical + visual_ann/text_ann Top-K=5，不走 Writer 链），返回是否升级 evidence 或询问澄清

**验收**：
- Test set：paraphrase pool（每类 5 条）不因 parser mode 波动而失去检索
- 显式写作 / 翻译 / 假设不误触发 probe（`_WRITING_PREFIX_RE` fast-path 依旧存在）
- Probe 返回强候选 → 升级 evidence；候选弱 → 询问澄清；不允许 Probe 高分直接生成家庭事实

### 2.3 修复线 C · 语义匹配

**代码位置**：`evidence_retrieval.py::_contains` / `_condition`；`answer_composer.py`；`thin_agent.py::_allowed_facts`

**改动**：
- 删除单字 contains 路径。词法匹配交给 §5.3 `LexicalRetriever` (FTS5)；`_contains` 只保留完整子串验证，作为 condition 阶段的证据校验
- `_condition` 收紧：多值字段 miss=unknown 已实现（保留）；追加 `clothing`/`object` 需 `subject_clothing`/`subject_objects` binding 才允许 contradicted（保留 §2R-6 语义）
- Answer Composer 去重（输入报告 §12）：
  - `_allowed_facts` 按 `condition_key` 一次，同 condition 多 asset 命中只出一条文本
  - 面向用户的"记录支持 X"改成人类可读模板：`匹配到 {N} 张、其中 {n_exact} 张与"{condition_surface}"直接对应`
  - 空 EvidencePacket 且 QuerySpec 是家庭查询 → 强制回答"当前记忆中没有找到足够匹配的原始证据"，禁止走 normal chat

**验收**：
- `_contains` 单元测试：只有完整 needle 是 haystack 子串或与 haystack 字段完整一致才返回 True
- Answer 单元测试：album1-01 类型输入不再有重复模板；空 GT 强制拒答
- Retrieval-only benchmark：无关图片率显著下降（数值门槛见 §7）

### 2.4 修复线 D · 模型基础设施

**代码位置**：`backend/model_clients.py::GammaClient`；新增 `backend/model_routing.py`；`scripts/runtime/start_sentrix_api.sh`

**改动**（输入报告 §10）：
- `model_routing.py` 定义 3 类模型角色：
  - `QueryInterpreterModel`（parser）— 目标：JSON 稳、延迟低、中文可用
  - `AnswerModel`（simple + Writer 共用）— 目标：自然表达
  - `ClaimVerifierModel`（复杂路径）— 目标：正确性优先
- `GammaClient` 增加 `parse_model` / `answer_model` / `verify_model` 可分离配置；默认全部指向现有 gemma4:12b 保持兼容；flag `SENTRIX_MODEL_SPLIT_V1` on 时读取 `SENTRIX_PARSE_MODEL` / `SENTRIX_ANSWER_MODEL` / `SENTRIX_VERIFY_MODEL`
- 每类调用加 `seed` / `num_ctx` / `num_predict` 显式参数、`httpx timeout=30`（当前 180）、cold/warm 直探脚本 `scripts/maintenance/probe_model_health.py`
- 简单 evidence 路径生成模型调用数硬预算 ≤ 2（parser 1 + answer 0-1）；短名词+图片请求走确定性回答，0 次生成调用

**验收**：
- Retrieval-only benchmark p95 ≤ 5s（不含生成模型）
- Simple evidence E2E p95 ≤ 12s，硬上限 ≤ 20s；生成模型调用 ≤ 2
- 若硬件不达标，`docs/baseline/thin-agent-phase-R-infra-blocker.md` 输出实测阻塞（VRAM / queue / cold latency / JSON 合法率），**不通过删除正确性步骤伪造达标**（输入报告 §17）

---

## 3. Benchmark 隔离协议（输入报告 §2）

**核心原则**：Benchmark 是测试和诊断工具，不是运行时设计输入。

### 3.1 三集合分层

| 集合 | 内容 | 用途 | 允许运行时读取? |
|---|---|---|:---:|
| **Regression Set** | 现有 60 case + 6 case 实测 | 防回归 / 通道消融 / 硬约束违反检查 | ❌ |
| **Development Set** | 用户提供或本阶段新增的合成 case（不复制现有题目） | 选融合策略、默认阈值 | ❌ |
| **Hidden Acceptance Set** | 用户保留完整答案，代码 Agent 只跑不看细节 | 最终验收，检查过拟合 | ❌ |

三集合的 **query 原文、GT 文件名、Asset ID、专用词表、专用权重、专用分支** 全部禁止出现在 `backend/*.py`。

### 3.2 运行时禁入清单（输入报告 §2.1 + §20）

以下写入 `backend/*.py` 判定为不合格：
- benchmark 题目原句片段
- benchmark 文件名（`IMG_4350.JPG`、`IMG_3726.JPG`…）
- benchmark Asset ID
- 针对某 case 的关键词分支
- 针对某相册的 if 分支
- 为通过当前题目手写的同义词白名单
- 手工按 60 case 调出的 case-specific 权重
- "如果查询包含某句，就返回某结果"的 hard-coded map
- Parser few-shot 里出现 benchmark 真实题目

### 3.3 校验手段

**新增测试** `backend/tests/test_no_benchmark_runtime_dependency.py`：
- 扫描 `backend/*.py`，禁止出现任何 benchmark query 字符串或 GT 文件名
- 扫描 Parser few-shot 常量，禁止匹配 `samples/album*/query.json` 中的任何 query_cn
- 扫描 `scripts/runtime/` 与 `scripts/maintenance/`，禁止 hard-code benchmark 路径
- 该 test **跑在 CI 前置**，任何阶段 red gate

### 3.4 使用 benchmark 的正当方式（输入报告 §2.3）

允许：
- 计算 Recall / Precision / MRR / 拒答率 / 硬条件违反率
- 分析某 GT Asset 未被召回时通道 rank 分布
- 从失败 case 抽象**通用**修复原则（"多值字段未命中返回 unknown"）

不允许：
- 因单个 case 失败就写 case-specific 规则
- 对某 GT Asset 特殊 boost
- 用 benchmark 文件名建立别名

---

## 4. 目标生产调用链（输入报告 §4）

```text
用户消息
  │
  ▼
Request Interpreter (memory_gate + query_parser)
  ├─ 显式普通任务 (writing prefix / translation / hypothesis) ─── 明确 none ──▶ 普通聊天路径 (无 retrieval)
  ├─ 显式家庭查询 (structural signals：日期+confirmed entity / selected_entity / correction payload / 当前对话追问已检索证据) ─── 明确 evidence
  └─ 其他 → parser draft 输出 proposed_mode + confidence + ambiguities
                                        │
                                        ▼
                        低成本 Retrieval Probe (metadata + lexical + visual_ann + text_ann, Top-K=5, 单通道 timeout≤500ms)
                                        │
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
                无候选/弱候选        强候选一致           冲突/歧义
                    │                   │                    │
                普通回应或询问        升级 evidence         询问澄清（不生成家庭事实）
                                        │
                                        ▼
                        Immutable QuerySpec (sanitize 后冻结)
                                        │
                                        ▼
                        Evidence Retrieval Kernel
                          1. hard prefilter (scope/viewer/time/media/must_not)
                          2. multi-retriever recall (metadata/entity/lexical/visual_ann/text_ann/adjacency)
                          3. Asset merge (同 asset 多 retriever 命中合并)
                          4. condition evidence (逐条 constraint × 候选 evaluate)
                          5. hard postfilter (contradicted 拦截 / must_not 拦截)
                          6. fusion / ranking (RRF or normalized weighted, 见 §5.4)
                          7. level = exact / strong / approximate / possible
                                        │
                                        ▼
                        EvidencePacket (含 excluded_count / gaps / per-channel attribution)
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
        简单确定性回答 (0 次生成)            可选单次 answer_model (≤ 1 次生成)
                    │                                        │
                    └───────────────┬────────────────────────┘
                                    ▼
                        回答 + 授权证据入口 (image_results 只在 explicit request 时)
```

**Probe 边界**（输入报告 §4）：内部机制，不是用户可见第四种模式。**不得**返回具体家庭事实、绕过权限、自动显示照片、自动重读原图、把向量高分当事实、对明确普通任务执行检索。

---

## 5. 模块设计与文件级改动清单

### 5.1 新目录 `backend/retrieval/`

多路 retriever 模块化，每个文件单一职责（输入报告 §19）：

```text
backend/retrieval/
  __init__.py          # 只导出接口和工厂
  base.py              # Retriever Protocol / RetrievalQuery / CandidateHit / HardFilterContext
  metadata.py          # MetadataRetriever（时间 / media / scope / GPS 范围 / batch / device）
  entity.py            # EntityRetriever（confirmed entity / face-bridge / event participant）
  lexical.py           # LexicalRetriever（FTS5 based，见 §5.3）
  visual_ann.py        # VisualAnnRetriever（query text → CLIP text embed → hnswlib top-K on visual index）
  text_ann.py          # TextAnnRetriever（query text → CLIP text embed → hnswlib top-K on semantic/episodic index）
  adjacency.py         # AdjacencyRetriever（seed asset → 同 event / 同 batch / 时间窗 / 近重复组）
  fusion.py            # RRF / 归一化加权融合；来源可靠度分层；多样性/近重复抑制
```

**每个 Retriever 的守恒规则**：
- 只做**召回和评分**，不做条件评估；条件评估仍在 `EvidenceRetrievalKernel._condition` 里
- 输出 `CandidateHit.score` 通道内可比，跨通道**不可直接比较**（交给 `fusion.py`）
- 每个 retriever 都必须遵守 `HardFilterContext` 里的 scope/viewer/time/media/must_not

### 5.2 `evidence_retrieval.py` 改造

新版 `retrieve()` 骨架（伪代码）：
```python
def retrieve(self, spec: QuerySpec) -> EvidencePacket:
    filters = HardFilterContext.from_spec(spec)
    query = RetrievalQuery.from_spec(spec, embedder=self.embedder)  # 编码 whole query + facets

    # Recall — 多通道并发（asyncio 或 ThreadPool）
    candidates_by_channel = {r.name: r.retrieve(query, filters, limit=spec.recall_limit)
                             for r in self.retrievers if r.enabled}

    # Asset merge — 同 asset 多通道 hit 聚合
    merged: dict[str, MergedCandidate] = merge_candidates(candidates_by_channel)

    # Condition evidence — 用 merged.observations 逐 constraint evaluate
    for cand in merged.values():
        cand.condition_results = self._evaluate_conditions(cand.asset, cand.observations, spec)

    # Hard postfilter — condition 结果里 contradicted or must_not 命中 → 剔除
    filtered = [c for c in merged.values() if not c.hard_violated]

    # Fusion + ranking
    ranked = self.fusion.rank(filtered, spec)

    # Level 分类 → EvidencePacket
    return build_packet(ranked, spec)
```

**渐进接入**：R2 阶段先并行运行"新旧 retrieve"，`SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=off` 时保持旧行为不变（regression 兜底）；shadow diff 记录到 `retrievalTrace.shadow_multi_retriever_diff`。

### 5.3 `LexicalRetriever` 与 FTS5

删除单字 contains 语义（输入报告 §9）。

- 依赖 SQLite FTS5（stdlib 自带；不需要引入新依赖）
- 新增 `observation_search_fts` 虚表，字段：`asset_id`、`observation_id`、`scope_id`、`field_type`、`normalized_value`
- Tokenizer：`unicode61 remove_diacritics 2 tokenchars '一-鿿'`；中文按字 unigram + bigram 索引；检索时 query 走同一 tokenizer
- 支持 whole-query 与 facet 双路（输入报告 §8.2）：
  - Whole query：`SELECT ... MATCH ? ORDER BY bm25(...)` 用完整原句
  - Facets：对每个 semantic constraint 单独 MATCH，score 单独归一化
- FTS 同步：`retrieval_indexes.py::refresh_from_observation` 内追加 FTS upsert；`rebuild_all` 全量重建 FTS
- **不写 benchmark 专用同义词**；同义扩展由 embedding 或独立通用词法组件（未来）提供

### 5.4 融合策略 `retrieval/fusion.py`

输入报告 §8：初版用**可解释、通用**融合，禁止手工权重表。

**默认**：Reciprocal Rank Fusion (RRF) with k=60（Cormack 2009 常量，非 case-specific）
```python
def rrf_score(rank_by_channel: dict[str, int], k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in rank_by_channel.values())
```

**扩展**：`SENTRIX_RETRIEVER_FUSION="rrf" | "weighted_norm"`，`weighted_norm` 时通道权重从 `configs/retrieval_fusion.json` 读，**该文件不进 `backend/*.py`**，由 Development Set 训出、Hidden Acceptance Set 验证。

**向量证据等级**（输入报告 §8.3）：向量命中只能升级为 `possible`，不能直接产生 `matched` / 人物确认 / 具体日期 / 具体地址 / confirmed activity / memory correction。这条通过 `_condition` 的 source_type 白名单实现：
```python
_MATCHED_SOURCE_TYPES = {"asset_metadata", "observation_field_exact", "entity_bridge_confirmed", "ocr_exact"}
# vector_visual / vector_text 只能返回 possible
```

### 5.5 Gate 与 Parser 改动

**`memory_gate.py`**：
- 保留 `_WRITING_PREFIX_RE` fast-path（写作/翻译）→ `proposed_mode="none"` 且 `allow_probe=False`
- 保留 evidence fast-path：显式 `feedback` / `selected_entity_id` / 明确日期+confirmed entity → `proposed_mode="evidence"` 且 `allow_probe=False`
- 其余进 `QueryParser` → 用 draft 的 mode/confidence 决定；`draft.confidence < 0.7` 或 `draft.ambiguities` 非空 → `proposed_mode="ambiguous"` 且 `allow_probe=True`

**`query_parser.py`**：
- Prompt 缩短 + 通用结构 few-shot（**禁用真实 benchmark 题目**，改用合成占位）
- 显式 `seed=42` + `num_ctx=4096` + `num_predict=512`
- `_validate` 收紧：
  - `mode=="none"` 但用户消息非 writing prefix 且长度 > 6 → 触发 repair
  - `actions == []` 且 `mode=="evidence"` → 触发 repair
  - `draft.confidence` 缺失 → 补 0.5
- Repair 失败降级到"安全 spec"：`mode="ambiguous"`, `allow_probe=True`, `actions=[]`
- **不允许**返回"关键词分类器"作为 fallback（输入报告 §20.5）

### 5.6 Thin Agent 与 Answer Composer 改动

**`thin_agent.py`**：
- `_evidence_answer` 分派：short-noun-phrase 且候选强 → 走"简单确定性回答"（0 生成调用）；复杂查询 → 走 `complex_answer.ComplexAnswerBuilder`（保留现状）
- Probe 只调 `EvidenceRetrievalKernel.retrieve(spec, mode="probe")`（recall_limit=5，跳过 Writer 链）

**`answer_composer.py`**：
- `_allowed_facts` 按 `condition_key` 去重，`dict.fromkeys` 保留顺序
- 面向用户的模板改成人类可读（输入报告 §12.3）：内部 condition_key、ANN 分数、retrieval trace、DB 表名**禁止**出现在用户可见回答里
- 空 EvidencePacket 且 QuerySpec 是家庭查询 → 强制模板`"当前记忆中没有找到足够匹配的原始证据"`，不走 normal chat（输入报告 §12.4）

---

## 6. 阶段拆分（R0 → R7）

**执行原则**：
- 每阶段本地全部单元测试 + Retrieval-only benchmark 通过 → 本地 git commit（按 §14 拆分原子 commit）→ scp 到 153 → 153 上重跑 → 153 git commit → 重启 API 8091
- 任一阶段 red gate 未过，**不进入下一阶段**
- 每阶段结束在 `docs/baseline/thin-agent-phase-R{X}.md` 输出报告（改动清单 + 测试结果 + benchmark 对比 + flag 状态 + 未完成项）

### R0 · 真实调用链核验（只读，无生产代码改动）

**产出**：
- `docs/baseline/thin-agent-phase-R0-call-chain.md`：Kernel 当前使用的 retriever、ANN load/query 调用点、search_terms 调用点、parser mode 决策、每轮模型调用；用 6 实测 case 全流程逐步展开
- Asset 级诊断脚本 `scripts/benchmarks/inspect_retrieval_case.py`：对给定 `(scope_id, query, gt_asset_id)` 输出 asset metadata / observation 原始字段 / canonical 字段 / event / visual vector 是否存在 / text vector 是否存在 / index revision / 各 retriever rank / 硬过滤结果 / 条件矩阵 / 被排除原因

**退出条件**：能解释 6 个实测 case 的**每一步**（含 IMG_4350/IMG_3726 为何 rank 落到 GT 之外）；分类为 recall path 未接 / embedding 不支持 / lexical 不命中 / formation 缺字段 / 人物 binding 缺失 / GT 有歧义。

### R1 · Retrieval-only 基线（**不改生产代码，只加评测脚本**）

**产出**：
- `scripts/benchmarks/evaluate_retrieval_kernel.py`：给定 `(query, expected_asset_ids, forbidden_asset_ids, scope_id, cached_query_spec?)` → 输出 ranked Asset IDs、Recall@1/5/10/20、MRR、Precision@5、all_relevant recall、empty-GT FP、硬约束违反、per-channel 贡献、GT rank
- `scripts/benchmarks/evaluate_parser_retrieval.py`：Parser + Retrieval 组合，比较 cached QuerySpec vs real Parser
- 通道消融模式（输入报告 §7.3）：`--channels lexical`、`visual`、`text`、`structured`、`hybrid_no_adjacency`、`full_hybrid`
- 每次 run 输出 `docs/baseline/retrieval_baseline_YYYYMMDD.json` 存档；不含生成模型延迟

**退出条件**：完整 60 case 可在**不调用回答大模型时**运行（p95 ≤ 5s），输出通道消融基线；Regression Set 数据可复现。

### R2 · 接通已建索引到 Kernel

**变更清单**：
- 新增 `backend/retrieval/base.py`、`metadata.py`、`entity.py`、`lexical.py`、`visual_ann.py`、`text_ann.py`
- `LexicalRetriever` 读 `observation_search_terms`；追加 FTS5 虚表 + 迁移
- `VisualAnnRetriever` / `TextAnnRetriever` 读磁盘 hnswlib 索引 `data/ann/{visual,semantic,episodic}`
- `EvidenceRetrievalKernel.__init__` 注入 retrievers 列表；`retrieve()` 新增多通道 recall 路径，`SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1` 时启用
- `ThinAgentRuntime.__init__` 增加 `clip` 参数（可选，缺省 `None`，缺省时 visual/text retriever 自动跳过并 trace 中标注 `clip_disabled`）；`app.py:30` 传入 `pipeline.clip`
- `retrievalTrace` 结构扩展：`channels: {name, candidate_count, latency_ms, top_score}`

**验收**（输入报告 §6.3）：
- 生产 Kernel trace 里五个通道均出现候选计数（非零 / 或明确 `disabled` 原因）
- Regression Set 60 case Retrieval-only Recall@10 ≥ 现有基线 + 10 pp（保守下限）
- 每个进入 packet 的 asset 都能回指至少一个 retriever

### R3 · 词法与融合修复

**变更清单**：
- 删除 `evidence_retrieval.py::_contains` 单字/双字 all-match 路径；`_contains` 只保留完整子串验证
- `retrieval/fusion.py` 实装 RRF；`SENTRIX_RETRIEVER_FUSION="rrf"` 默认
- `_condition` matched 白名单：仅 `asset_metadata` / `observation_field_exact` / `entity_bridge_confirmed` / `ocr_exact` 可返回 matched；vector 通道返回 possible
- 近重复抑制：融合层按 CLIP visual embedding cosine > 0.98 视为同图组，只保留组内最高分
- Answer composer 去重 + 空 EvidencePacket 强制拒答

**验收**（输入报告 §17）：
- Regression Set Retrieval-only Recall@10 ≥ 90% / Recall@20 ≥ 95% / empty-GT FP = 0 / 硬条件违反 = 0
- Hybrid 不低于任一单通道 Recall@10
- 无关图片率显著下降（数值门槛与 R1 基线对比 ≥ 50% 相对下降）
- 无 benchmark 专用规则（`test_no_benchmark_runtime_dependency.py` 绿）

### R4 · Gate + Probe

**变更清单**：
- `memory_gate.py::MemoryGate.decide` 返回 `GateDecision`
- `thin_agent.py`：`mode=="ambiguous"` 且 `allow_probe=True` → 调 `EvidenceRetrievalKernel.retrieve(spec, mode="probe")`；probe 结果强 → 升级 evidence；probe 弱 → 询问澄清
- Probe 阈值不硬编码，从 `configs/probe_thresholds.json`（Development Set 训出）读；缺文件时 fail-safe 到"总是询问澄清"
- Parser `_validate` 收紧规则实装

**验收**（输入报告 §17）：
- 明确普通任务 → mode=none 准确率 ≥ 99%
- 明确家庭请求进入 evidence/probe ≥ 99%
- 短名词（"银色心形手镯"/"八戒"）GT 查询因 `none` 丢失率 = 0
- Probe 误触发率在 Development/Hidden Set 上受控并记录

### R5 · 模型预算

**变更清单**：
- `backend/model_routing.py`；`GammaClient` 支持 parse/answer/verify 模型分离
- 简单 evidence 路径生成调用 ≤ 2（parser + 可选 answer）
- 短名词 + 图片请求路径生成调用 = 0（走确定性模板）
- `httpx timeout=30`；circuit breaker：3 次 timeout 内该通道自动 fallback
- `scripts/maintenance/probe_model_health.py`：定期探测 cold/warm latency + JSON 合法率，输出 `docs/baseline/model_health_YYYYMMDD.json`

**验收**（输入报告 §17）：
- Simple evidence E2E p95 ≤ 12s，硬上限 ≤ 20s
- 简单 evidence 路径生成调用 ≤ 2
- 若硬件不达标，`docs/baseline/thin-agent-phase-R-infra-blocker.md` 有实测阻塞报告

### R6 · Answer

**变更清单**：
- `answer_composer.py` 去重
- 空 GT / 空 EvidencePacket 强制拒答
- 用户可见文本禁止暴露内部 condition_key / retrieval_trace / DB 表名
- 简单查询走确定性回答；只有 person/event summary 才走 Writer 链

**验收**：
- Regression Set 6 case（含 album1-01 / album3-01）不再出现重复模板
- 空 GT 三个 case（album1-07 / album2-06 / album3-14）返回明确拒答而非通用描述
- E2E benchmark 通过

### R7 · 隐藏验收 + 三集合切齐

**变更清单**：
- 无生产代码改动
- `docs/baseline/thin-agent-phase-R7-report.md`：Regression / Development / Hidden Acceptance 三集合结果、通道贡献、GT rank 分布、模型健康、剩余 Formation 层归因
- 若完整 Hybrid 后仍失败的 case → 分类为 formation / embedding / parser / fusion / filter / ranking / GT 歧义 → 汇总为 Formation 层改造 Phase F1 的输入（本阶段不动 Formation）

**退出条件**（输入报告 §21 Definition of Done 全部满足）：
1-18 项逐条打勾（见 §7）

---

## 7. Feature Flag 拓扑与回滚

**新增 flag**（默认全部 off，本地测试用 env 显式开启，153 生产按阶段推进逐个 on）：

| Flag | 阶段 | 作用 | Off 时行为 |
|---|:-:|---|---|
| `SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1` | R2 | 启用多通道 recall | 走旧单 Kernel 扫描 |
| `SENTRIX_RETRIEVER_METADATA` | R2 | 单通道开关 | 该通道不参与召回 |
| `SENTRIX_RETRIEVER_ENTITY` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_LEXICAL` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_VISUAL_ANN` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_TEXT_ANN` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_ADJACENCY` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_FUSION` | R3 | `"rrf"` \| `"weighted_norm"` | 默认 rrf |
| `SENTRIX_GATE_PROBE_V1` | R4 | 启用 ambiguous → probe | Gate 保持二元 mode |
| `SENTRIX_MODEL_SPLIT_V1` | R5 | parser/answer/verify 分离模型 | 全部走同一模型 |
| `SENTRIX_PARSE_MODEL` | R5 | parser 模型名 | 用 `gamma_model` |
| `SENTRIX_ANSWER_MODEL` | R5 | answer 模型名 | 用 `gamma_model` |
| `SENTRIX_VERIFY_MODEL` | R5 | verify 模型名 | 用 `gamma_model` |

**已有 flag 保留**（输入报告 §0 不变量 + Phase 8 flag 布线）：
`SENTRIX_THIN_AGENT_V1`, `SENTRIX_SEMANTIC_QUERY_PARSER_V1`, `SENTRIX_EVIDENCE_RETRIEVAL_V1`, `SENTRIX_LLM_CLAIM_EXTRACTOR_V1`, `SENTRIX_CORE_MEMORY_V1`, `SENTRIX_MEMORY_CORRECTION_V1`, `SENTRIX_ADVANCED_MEMORY_TOOLS_V1`, `SENTRIX_ANN_INDEX_V1`, `SENTRIX_EXPLICIT_IMAGE_REINSPECTION`

**回滚协议**：任一 flag off 立即回退到旧路径；`MemoryStore` 和 FMA 5173 永不因 Agent flag 停止。

---

## 8. 拟删除 / 拟保留 / 拟替换

| 模块 | 决策 | 说明 |
|---|---|---|
| `evidence_retrieval.py::_contains` 单字/双字 all-match | **删** | 由 `LexicalRetriever` FTS5 替代；`_contains` 只保留完整子串校验 |
| `evidence_retrieval.py::retrieve()` 单 Kernel 扫描 | **替** | 保留旧实现在 flag off 时可用；on 时走多通道 |
| `agent.py` 里旧 `MemoryAgent.evidence_answer` 用 `search_vectors` | **保** | 老路径继续存在，作为 flag off 时的兜底 |
| `store.search_vectors` SQLite 逐行余弦 | **保** | 作为 hnswlib 出错时 fallback；不再是主路径 |
| `answer_composer._allowed_facts` 无去重版本 | **替** | 按 condition_key 去重 |
| Parser prompt 关键词 fallback（若还有残留） | **删** | 输入报告 §20.5 明令禁止 |
| `configs/probe_thresholds.json` / `configs/retrieval_fusion.json` | **新增** | 由 Development Set 训出、Hidden Acceptance Set 验证；`backend/*.py` 不直读 case 数据 |

---

## 9. 中文 CLIP / 文本 embedding 能力独立验证方案（输入报告 §7.2、§18.1）

**问题**：当前 `CLIP_MODEL_NAME="ViT-B-32"` 的中文文本能力未证明。若中文弱，接通了也没用。

**验证脚本** `scripts/benchmarks/evaluate_embedding_quality.py`（不依赖 benchmark 60 case）：
- 输入：合成中文短语集合（**必须新造，不复用 samples/album*/query.json 里的 query_cn**），覆盖：
  - 物品（"银色心形手镯"、"墨绿色马克杯"、"黄色雨伞"）
  - 衣着 + 颜色（"浅黄色针织衫"、"藏青色羽绒服"）
  - 场景 + 活动（"厨房做饭"、"海边看日落"）
  - 地点（"贵阳步行街"、"西湖木船"）
  - 负样本对（不同语义应远离）
  - 同义改写对（应接近）
- 每对计算 cosine similarity；输出直方图 + 阳性/阴性对分离度（AUC）
- 判定：AUC < 0.7 → 中文能力不合格 → 需替换 embedding

**替换 adapter 骨架**（输入报告 §6.2 Visual/Text ANN Retriever）：
- `backend/model_clients.py::ClipAdapter` 抽象 `TextEmbedder` / `ImageEmbedder` 接口
- 备选实现（不在本阶段落地，只留 adapter hook）：多语言 CLIP（如 `mCLIP` / `Chinese-CLIP`）、独立多语言文本 encoder（如 `bge-m3`）、双路 embedding
- flag `SENTRIX_TEXT_EMBEDDER` / `SENTRIX_IMAGE_EMBEDDER` 选实现；架构必须允许替换而**不修改 QuerySpec 和 Kernel**

**验收**：
- `evaluate_embedding_quality.py` 输出 AUC ≥ 0.7 → 沿用 ViT-B-32
- AUC < 0.7 → 输出 blocker 报告，用户决定选备选 adapter；本阶段不擅自切换

---

## 10. 性能预算与模型调用预算（输入报告 §10、§17）

| 路径 | 生成模型调用 | Retrieval 延迟 | E2E 延迟 |
|---|:-:|:-:|:-:|
| 普通聊天 / 写作 / 翻译 | 0（parser 也可 fast-path 跳过） | 0 | ≤ 5s |
| 短名词 + 图片请求 | 0（确定性模板） | ≤ 5s | ≤ 8s |
| 简单 evidence | ≤ 2（parser + optional answer） | ≤ 5s | ≤ 12s |
| Complex (person / event / compare) | ≤ 4（parser + writer + claim + verify + 最多 1 repair） | ≤ 5s | ≤ 20s（硬上限） |
| Probe | 0 生成 | ≤ 2s | 计入所属路径 |

**硬门槛**：
- Retrieval-only p95 ≤ 5s
- Simple evidence E2E p95 ≤ 12s
- API 硬上限 ≤ 20s
- ANN 查询不得退化为十万向量 Python 全扫描（`test_ann_index.py::test_hnswlib_reload_recall` 已守护）

---

## 11. 单元测试与集成测试矩阵

**新增**：
| 文件 | 阶段 | 覆盖 |
|---|:-:|---|
| `backend/tests/test_retriever_contracts.py` | R2 | Retriever Protocol / CandidateHit / HardFilterContext 契约 |
| `backend/tests/test_retrieval_fusion.py` | R3 | RRF / 归一化加权 / 近重复抑制 |
| `backend/tests/test_lexical_retriever.py` | R2-R3 | FTS5 tokenizer / 中文 unigram+bigram / whole+facet |
| `backend/tests/test_visual_ann_retriever.py` | R2 | text→embed→ANN top-K；scope 回表；index 不 ready 时 fallback |
| `backend/tests/test_text_ann_retriever.py` | R2 | 同上 |
| `backend/tests/test_adjacency_retriever.py` | R2 | 同 event / 同 batch / 时间窗；不放宽硬条件 |
| `backend/tests/test_gate_probe.py` | R4 | GateDecision / probe upgrade / 询问澄清 |
| `backend/tests/test_model_budget.py` | R5 | 简单 evidence ≤ 2 次生成；probe = 0 |
| `backend/tests/test_no_benchmark_runtime_dependency.py` | **贯穿** | runtime 不含 benchmark query/GT/Asset ID |
| `backend/tests/test_embedding_quality.py` | R2 | 合成短语集合 AUC ≥ 0.7 |
| `backend/tests/test_answer_composer_dedup.py` | R6 | 去重 + 空 EvidencePacket 强制拒答 |

**扩展**：
- `test_evidence_retrieval_benchmark.py`：加通道消融维度
- `test_thin_agent_runtime.py`：Probe 分支 + 短名词路径
- `test_query_parser.py`：`_validate` 新收紧规则
- `test_memory_gate.py`（拆分自 `test_thin_agent_contracts.py`）：GateDecision

---

## 12. 输入报告 §18 十八项必答问题的对应答案位置

| # | 问题 | 计划位置 |
|:-:|---|---|
| 1 | Visual vector 模型/checkpoint/维度/中文能力如何验证 | §9 `evaluate_embedding_quality.py` |
| 2 | Observation/Event text vector encoder | §9；当前复用 ClipAdapter，AUC 决定是否切 |
| 3 | HNSW ID 如何稳定映射回 scope 下 Asset | §5.1 `visual_ann.py` sidecar JSON + scope 回表 |
| 4 | Kernel 在哪里并行调用 retriever | §5.2 `retrieve()` 骨架 concurrent recall |
| 5 | `observation_search_terms` 如何进入 lexical retriever | §5.3 FTS5 虚表同步 |
| 6 | 用哪种通用 fusion / 为何不过拟合 | §5.4 RRF k=60 / configs 训在 Dev、验在 Hidden |
| 7 | Whole text + facets 如何同时进召回 | §5.3 whole + facet 双路；§5.4 fusion 融 |
| 8 | Parser `none` 时什么情况下 probe | §5.5 `mode=="ambiguous"` 且 `allow_probe=True` |
| 9 | 明确普通写作 vs 短名词家庭查询区分 | §5.5 fast-path + Parser confidence + 结构性信号 |
| 10 | Probe 阈值 Dev/Hidden 分离 | §5.4 + §6 R4 |
| 11 | Vector hit 只作 candidate/possible | §5.4 `_MATCHED_SOURCE_TYPES` 白名单 |
| 12 | Contradicted 证据要求 | §2.3 subject binding 才允许 contradicted |
| 13 | Retrieval-only vs E2E 分离 | §6 R1 evaluate_retrieval_kernel.py |
| 14 | 简单 evidence ≤ 2 次生成 | §10 预算 + `test_model_budget.py` |
| 15 | 12B 每次 9-90s 缓解 | §2.4 model_routing.py + §9 probe_model_health |
| 16 | 独立回滚 flag | §7 flag 拓扑 |
| 17 | Runtime 不含 benchmark 内容 | §3.3 `test_no_benchmark_runtime_dependency.py` |
| 18 | 完整 Hybrid 后仍失败 case 归 Formation | §6 R7 report 分类 |

---

## 13. 零容忍门槛（对齐输入报告 §16）

以下任一违反 → 该阶段 red gate，不可宣布完成：

| # | 门槛 | 检测方式 |
|:-:|---|---|
| 1 | Benchmark query/file/Asset ID 出现在 runtime 规则 | `test_no_benchmark_runtime_dependency.py` |
| 2 | Case-specific 同义词或 boost | Code review + grep patterns |
| 3 | 明确普通写作触发家庭检索 | `test_gate_probe.py` writing prefix |
| 4 | 家庭短语因 parser `none` 永久失去检索 | `test_gate_probe.py` short-noun |
| 5 | 向量高分直接升级 confirmed fact | `_MATCHED_SOURCE_TYPES` 白名单 |
| 6 | 单字 contains 作为 matched 支持 | `test_lexical_retriever.py` |
| 7 | 多值字段未命中直接 contradicted | `test_evidence_bundle.py` |
| 8 | 空 GT 进入 normal chat 编造具体场景 | `test_answer_composer_dedup.py` |
| 9 | 已建 ANN 未被生产 Kernel 调用却宣称接入 | trace 里必须有 `visual_ann.candidate_count > 0` 的实际请求 |
| 10 | 普通查询自动原图重读 | `SENTRIX_EXPLICIT_IMAGE_REINSPECTION=0` 时 image_results 为空 |
| 11 | 用户可见完全无关 Asset | Regression Set FP 门槛 |
| 12 | 硬时间/人物/scope/media/must_not 违反 | 现有 constraint 校验 |

---

## 14. Commit 拆分建议（每阶段独立 push 到 153）

**R0**：
1. `docs: baseline phase R0 call chain audit`
2. `feat(scripts): inspect_retrieval_case for asset-level diagnosis`

**R1**：
1. `feat(benchmarks): retrieval-only kernel evaluator with channel ablation`
2. `feat(benchmarks): parser+retrieval evaluator with cached query spec`
3. `test: no benchmark runtime dependency guard`

**R2**：
1. `feat(retrieval): Retriever protocol + RetrievalQuery + CandidateHit`
2. `feat(retrieval): metadata / entity retrievers`
3. `feat(retrieval): lexical retriever backed by FTS5`
4. `feat(retrieval): visual_ann / text_ann retrievers reading hnswlib`
5. `feat(retrieval): adjacency retriever`
6. `feat(evidence): multi-retriever recall behind SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1`
7. `feat(app): pass pipeline.clip to ThinAgentRuntime`

**R3**：
1. `refactor(evidence): drop single-char contains; keep only full-substring check`
2. `feat(retrieval): RRF fusion default`
3. `feat(evidence): matched source-type whitelist; vectors return possible`
4. `feat(retrieval): near-duplicate suppression`
5. `refactor(answer): dedup allowed facts + empty-packet forced refusal`

**R4**：
1. `feat(gate): GateDecision with confidence and probe`
2. `feat(agent): ambiguous → retrieval probe path`
3. `feat(parser): tighter validator + safe fallback`

**R5**：
1. `feat(model): model_routing splitting parse/answer/verify`
2. `feat(model): timeout=30 + circuit breaker`
3. `feat(scripts): model health prober`

**R6**：
1. `refactor(answer): user-visible template hardening`
2. `feat(agent): deterministic simple-evidence path`

**R7**：
1. `docs: phase R7 hidden acceptance report + formation-layer input`

---

## 15. 仍需用户确认的问题（本次会话结束前需要点头）

**不重复询问** 已确认的产品方向（8 阶段全做、hnswlib 已选、本地推进per-phase到153）。以下**必须**在实施前拿到答复：

1. **本地 vs 153 数据依赖**：Retrieval-only benchmark 需要真实图片记忆数据（Observation/Asset/vector），本地工作副本 `data/sentrix.db` 是否有可用副本？还是全部在 153 上跑？
   - **推荐**：R0/R1 在 153 上跑（真实数据在那里），本地只写代码 + 骨架单元测试；R2 起本地也需要一份小规模真实数据副本用于开发
2. **Hidden Acceptance Set 从哪里来**：
   - 选项 A：用户手工新造 20-30 case（推荐；能验证泛化）
   - 选项 B：从现有 60 case 里随机划出 20 case 冻结，代码 Agent 后续不看细节（次优）
   - 选项 C：先不做 Hidden Acceptance，R7 前只有 Regression + Development（最快但过拟合风险高）
3. **中文 embedding 备选**：AUC < 0.7 时用户希望采用哪个方向？
   - Chinese-CLIP（视觉能力强，需要额外权重）
   - bge-m3（纯文本，视觉需另外方案）
   - 或先不切，接受"视觉召回仅英文短语可用"
4. **模型基础设施优先级**：R5 之前如果 Ollama gemma4:12b VRAM 问题仍未解决（189MB/8GB），是否允许在本阶段直接切分 parse 模型到更小模型（如 qwen2.5:3b / gemma3:4b）？
5. **配置文件位置**：`configs/probe_thresholds.json` 与 `configs/retrieval_fusion.json` 放在仓库 `configs/` 还是 `data/configs/`？前者进 git，后者不进 git；因不是 benchmark 数据，倾向 `configs/` 进 git

---

## 16. 阶段完成声明模板（每阶段结束时对齐输入报告 §21）

```markdown
# Phase R{X} 完成声明

## 修改文件清单 + git diff --stat
...

## 新增/扩展测试
- unit: 数量 / 结果
- integration: 数量 / 结果
- benchmark: Retrieval-only 前后对比

## Benchmark 对比
- Regression Set: Recall@10 / @20 / MRR / empty-GT FP / hard violation
- Development Set: 同上
- Channel ablation: 各通道贡献

## 已布线的 flag
...

## 153 上重启后的健康检查
- API 8091 = 200
- Web 4174 = 200
- FMA 5173 = 200

## 未完成项 / 真实阻塞 / 下一阶段依赖
...

## §21 Definition of Done 对照（累积打勾）
1. [ ] 生产 Kernel 接入 metadata/entity/lexical/visual/text/adjacency
2. [ ] ANN 在真实请求 trace 提供候选
...
18. [ ] Hybrid 后仍失败被准确分类
```

---

## 17. 本计划的 out-of-scope

本阶段**不做**（输入报告 §3 + §20）：
- Core Memory 完整上线宣告（保留代码，暂停宣布完成）
- Correction 端到端 UI 联调
- 主动回忆入口 `SENTRIX_PROACTIVE_MEMORY`
- 多 viewer 完整产品化
- Answer Writer 风格进一步优化
- Formation pipeline 大规模改造（R7 输出 Formation 输入报告，Formation 改造留给 Phase F1）
- 修改图片记忆生成语义 / `raw_json`
- 修改 FMA 5173 / e2b_server 相关 uncommitted 改动
- 视频解码 / 镜头切分 / 视频向量

**尤其禁止**：在 Retrieval 基础未通过时扩展主动回忆和复杂人物画像；用 375 个合同测试全绿证明实际 Agent 可用；用空 GT"凑巧没返回图"当作正确 Agent 行为（输入报告 §20）。

---

## 附录 A · 本计划与原 Phase 0-8 计划的关系

- **保留**：Phase 0-8 合同层骨架（Constraint 三层 / sanitize / EvidencePacket / statement×evidence 校验 / raw_json 只读 / flag 隔离）。
- **纠偏**：Phase 3 Kernel 单扫路径 → Phase R2 多通道；Phase 3.5 ANN 建了未接 → Phase R2 接通；Phase 4 简单回答重复模板 → Phase R6 去重 + 强制拒答；Phase 2R Gate 二元 → Phase R4 加 probe。
- **暂停宣告**：Phase 5-7（Core Memory / Correction / 高级工具）代码保留、flag 保留、**不再作为完成声明**，等 Retrieval 基础层通过后重新评估。
- **新增独立阻塞**：模型基础设施（R5）单独作为阻塞项，不与检索精度混淆。

