# Phase 0 · Thin Agent 与证据检索内核 基线冻结

**采集时间**：2026-08-05
**采集人**：Agent（接手前一位代码 Agent 的实现）
**目的**：为 Phase 0→8 全部工作建立 diff 基线，让后续每一步能明确"改了什么"。

## 1. 服务与仓库状态

### 1.1 · 153 psh 状态

- **HEAD**：`25728234834f42d4c069a76c2a7b0e2cb41f01cd`
- **最新 commit**：`2572823 docs(runbooks): vlm backend switch operator guide`
- **分支**：`psh`
- **工作树**：clean
- **最近 10 个 commit**：

```
2572823 docs(runbooks): vlm backend switch operator guide
6fc5188 feat(runtime): add start_sentrix_e2b.sh and E2B_BASE_URL env
755f94d style(web): add .model-switcher and .ready-label.warn styles
1455139 feat(web): add vlm backend switcher and migrate to health.models.vlm
f99a469 feat(api): add /api/vlm-backend GET/POST and migrate health.models to vlm
c67d398 feat(model_clients): wire E2BBackend to :8100 and add GammaClient routing
7c67164 docs(services): add e2b_server README
ce6a68d feat(services): e2b_server FastAPI app on port 8100
3651493 feat(services): e2b_server model loader with peft lora mount
0c79706 feat(services): e2b_server ollama_shape pure conversion helpers
```

### 1.2 · 服务健康检查

| 端口 | 状态 |
|---|---|
| API 8090 | HTTP 200 |
| API 8091 | HTTP 200 |
| Web 4174 | HTTP 200 |
| FMA 5173 | HTTP 200（本计划不触碰） |

### 1.3 · 本地工作副本状态

- **副本路径**：`/Users/rm001/Sentrix-Thin-Agent-work-20260805/`
- **本地 HEAD**：`fa4e5837261e1aa7238d8910cab1711af7957dc7`（`docs(plan): gemma-4 e2b-it + lora v2 integration implementation plan`）
- **本地分支**：`psh`
- **本地领先/落后 153**：本地落后 153 共 **13 个 commit**
- **本地 uncommitted modified**：
  - `backend/agent.py`（thin runtime 集成）
  - `backend/model_clients.py`（thin agent 相关）
  - `docs/PROJECT_MEMORY.md`
- **本地 untracked（前一位 Agent 的 thin agent 骨架）**：
  - `backend/answer_composer.py`
  - `backend/claim_extractor.py`
  - `backend/evidence_retrieval.py`
  - `backend/memory_gate.py`
  - `backend/query_contracts.py`
  - `backend/thin_agent.py`
  - `backend/tests/test_claim_extractor.py`
  - `backend/tests/test_evidence_bundle.py`
  - `backend/tests/test_thin_agent_contracts.py`
  - `backend/tests/test_thin_agent_runtime.py`

### 1.4 · 153 领先的 13 个 commit（合并风险区）

```
2572823 docs(runbooks): vlm backend switch operator guide
6fc5188 feat(runtime): add start_sentrix_e2b.sh and E2B_BASE_URL env
755f94d style(web): add .model-switcher and .ready-label.warn styles
1455139 feat(web): add vlm backend switcher and migrate to health.models.vlm
f99a469 feat(api): add /api/vlm-backend GET/POST and migrate health.models to vlm
c67d398 feat(model_clients): wire E2BBackend to :8100 and add GammaClient routing
7c67164 docs(services): add e2b_server README
ce6a68d feat(services): e2b_server FastAPI app on port 8100
3651493 feat(services): e2b_server model loader with peft lora mount
0c79706 feat(services): e2b_server ollama_shape pure conversion helpers
e9984b4 feat(model_clients): add E2BBackend stub
1cfa10c refactor(model_clients): extract OllamaBackend from GammaClient
f7d19b8 feat(db): add runtime_settings kv table
```

**重要合并风险**：
- `c67d398` 与 `1cfa10c` 都动了 `backend/model_clients.py`，把 `GammaClient` 拆成 `OllamaBackend` + `E2BBackend`。本地 `backend/model_clients.py` 有 uncommitted 修改，同一文件多处冲突可能。
- `f7d19b8` 新增了 `runtime_settings` KV 表（DB schema 增量）。
- `f99a469` 新增 `/api/vlm-backend` API + `health.models.vlm` 结构变更。

**传输 153 的对策**：每阶段完成后 `git fetch origin psh` + 有冲突文件手工 merge，不用 `--force` 覆盖 153 已有工作。

## 2. 测试基线

### 2.1 · 本地 Python `PYTHONPATH=. python -m unittest discover backend.tests`

- 总数：**278 tests**
- 结果：**FAILED (errors=2, skipped=2)**
- 唯一失败原因：`ModuleNotFoundError: No module named 'PIL'` 影响 `test_face_clustering.FaceEmbeddingContractTests` 2 项
- 结论：核心业务测试全绿，PIL 缺失只影响 face fixture mock，不影响 Thin Agent 相关模块

### 2.2 · 本地 Node `node --test test/*.test.js`

（未在此次采集执行，Phase 1 时补跑）

### 2.3 · 153 上测试

（Phase 0 不动 153；Phase 2R-1 传输后在 153 全量重跑）

## 3. 语义层僵化点（6 张关键词表）

对应本 Agent 只读审查的结论。这些是 Phase 2R 要修的具体位置。

### 3.1 · `backend/memory_gate.py:23-37`

```python
class MemoryGate:
    _memory_terms = ("照片", "图片", "相册", "回忆", "记得", "去年", "前年",
                     "家里", "人物", "家人", "原图", "证据", "时间线", "比较",
                     "关系", "穿什么", "衣服", "颜色", "外观", "介绍一下", "是谁")
    _contextual_terms = ("想", "想念", "怀念", "突然有点", "今天很累")

    def classify(self, message: str, conversation: str = "", *, proactive_enabled: bool = False):
        value = str(message or "").strip()
        if self._is_contextual(value):
            return GateDecision("contextual", "natural_person_mention", core_memory_reads=1)
        if not any(term in value for term in self._memory_terms):
            return GateDecision("none", "general_chat")
        target = "person" if any(term in value for term in ("介绍", "是谁", "人物", "家人", "关系")) else "general"
        original = any(term in value for term in ("给我原图", "直接看照片", "打开原始证据", "原始图片"))
        return GateDecision("evidence", "specific_household_question", answer_target=target, ...)
```

**僵化点**：`_memory_terms` 21 词、`_contextual_terms` 5 词、原图授权 4 短语、person target 5 词全部硬编码。

### 3.2 · `backend/thin_agent.py:83`（answer_target 分类）

```python
target = "person" if any(token in value for token in ("介绍", "是谁", "人物", "关系")) \
    else "clothing" if any(token in value for token in ("穿什么", "衣服", "衣着")) \
    else "general"
```

### 3.3 · `backend/thin_agent.py:85-87`（11 词条件白名单）

```python
for dimension, term in (("place", "厨房"), ("place", "餐厅"),
                        ("activity", "做晚饭"), ("activity", "做饭"),
                        ("visual", "自拍"), ("clothing", "浅黄色"),
                        ("clothing", "黄色"), ("clothing", "毛绒"),
                        ("clothing", "睡衣"), ("clothing", "红色"),
                        ("clothing", "灰色")):
```

### 3.4 · `backend/thin_agent.py:98-101`（intent 二元决策）

```python
"intent": "find_assets" if any(token in value for token in ("照片", "图片", "原图")) else "answer"
```

### 3.5 · `backend/claim_extractor.py:24`

```python
"intended_type": "derived_pattern" if any(token in sentence for token in
    ("经常", "通常", "喜欢", "总是", "往往")) else "family_fact"
```

### 3.6 · `backend/evidence_retrieval.py:178-179`（多值字段假否定）

```python
if field_value and constraint.dimension != "activity":
    return ("contradicted", "observation", ...)
```

## 4. GammaClient 未真正调用的证据

`backend/thin_agent.py` 中 `self.gamma` 出现位置：

```
22:        self.gamma = gamma                                    # __init__
55:        if self.gamma and hasattr(self.gamma, "chat"):        # _normal_chat
58:                answer = str(self.gamma.chat(prompt, ...))    # _normal_chat（唯一实际调用）
```

**结论**：`_parse_message`（line 75-109）**从不调用 `self.gamma`**——所有 QuerySpec 抽取来自本地词表，计划 §7.1 Query Parser 提示词完全没接。

## 5. 现有 flag 清单

| Flag | 用途 |
|---|---|
| `SENTRIX_THIN_AGENT_V1` | Thin Agent runtime 开关（本次工作前唯一 Thin Agent 相关 flag） |
| `SENTRIX_AGENT_FRAMEWORK_MODEL` | 可选 PydanticAI planner model 名 |
| `SENTRIX_ANNOTATION_STORE` | annotation store 可用性 |
| `SENTRIX_PROACTIVE_MEMORY` | 主动回忆（本计划冻结不解锁） |
| `SENTRIX_API_HOST/PORT`、`SENTRIX_DATA_DIR`、`SENTRIX_DB_PATH`、`SENTRIX_OLLAMA_HOST/MODELS`、`SENTRIX_PARALLEL_IMAGE_ANALYSIS`、`SENTRIX_PYTHON`、`SENTRIX_TEXT_EMBED_MODEL` | 环境/运行配置，非阶段 flag |

**待补 Phase 2R+3+3.5+4+5+6+7+8 布线的 flag**（默认全部 off）：

```
SENTRIX_QUERY_SPEC_V1            (Phase 2 完成)
SENTRIX_SEMANTIC_QUERY_PARSER_V1 (Phase 2R)
SENTRIX_EVIDENCE_RETRIEVAL_V1    (Phase 3)
SENTRIX_ANN_INDEX_V1             (Phase 3.5)
SENTRIX_LLM_CLAIM_EXTRACTOR_V1   (Phase 4)
SENTRIX_CORE_MEMORY_V1           (Phase 5)
SENTRIX_MEMORY_CORRECTION_V1     (Phase 6)
SENTRIX_ADVANCED_MEMORY_TOOLS_V1 (Phase 7)
SENTRIX_EXPLICIT_IMAGE_REINSPECTION (Phase 4)
```

## 6. 数据库 schema 快照（153 `data/sentrix.db`）

```
agent_annotation_visibility     event_entities       person_appearance_evidence
agent_claim_conflicts           event_observations   person_event_memory
agent_impressions               event_participants   person_patterns
agent_proactivity_preferences   event_revisions      persons
agent_scene_cooldowns           events               query_gaps
agent_schema_migrations         face_clusters        rebuild_runs
agent_user_assertions           face_instances       relationships
assets                          face_prototypes      runtime_settings
dialogue_states                 facts                semantic_claims
entities                        ingest_batches       semantic_profiles
entity_mentions                 invites              stories
entity_merge_candidates         memory_feedback      trip_revisions
entity_observations             memory_spaces        trips
entity_properties               memory_vectors
entity_revisions                observations
```

**Phase 5/6 schema 基础已具备**：`entity_revisions`、`event_revisions`、`trip_revisions`、`agent_schema_migrations`、`agent_user_assertions` 已存在。Phase 5 Core Memory 需要新增 `agent_core_memory_cards/items/query_accesses`；Phase 6 Correction 需要新增 `agent_memory_correction_proposals/revisions/audit`。

**Phase 3 派生投影**：需要新增 `observation_search_terms`。

**Phase 3.5 ANN 相关**：现有 `memory_vectors` 表存整个 JSON 向量，需要建独立 ANN 索引磁盘文件。

## 7. 冻结范围（本计划不修改的部分）

- ❌ 图片记忆生成管线的语义（`backend/pipeline.py` 图片入库路径）
- ❌ `raw_json` 字段（所有 canonical 表）
- ❌ FMA 服务 5173
- ❌ `SENTRIX_PROACTIVE_MEMORY` 相关的主动回忆代码（保持默认 off）
- ❌ 视频解码/镜头切分/视频向量
- ❌ e2b_server 相关的 uncommitted 改动（那是 VLM backend 切换任务的工作，与 Thin Agent 无关）
- ❌ 修改 canonical `entities`/`observations`/`events` 表结构（只新增 Agent-owned 表）

## 8. 结论

Phase 0 冻结完成。当前状态清晰记录，可以开始 Phase 1 检索 benchmark 建设。

**下一步**：Phase 1 · 新增 `scripts/benchmarks/evaluate_evidence_retrieval.py` + `backend/tests/test_evidence_retrieval_benchmark.py`，覆盖 B-01 到 B-10 全部 case。
