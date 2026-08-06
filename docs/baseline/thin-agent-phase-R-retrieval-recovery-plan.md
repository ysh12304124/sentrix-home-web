# Sentrix 检索可用性恢复执行计划 (Phase R) — v0.3

**版本**：0.3 (v0.2 对齐用户确认 D6-D8：153 2B 切换逻辑接线 / 不预留 scope 增长 / R1B 流程确认)
**日期**：2026-08-06
**输入**：
- 《Sentrix_检索可用性恢复与通用化设计_规划输入报告.md》(下称"输入报告")
- 《Sentrix_Phase_R执行计划审阅与修改建议.md》(下称"审阅报告")
**基线**：`docs/baseline/thin-agent-benchmark-findings.md` (6 case 实测 3 pass / 2 fail / 1 timeout，正样本命中率 0/3)
**工作副本**：`/Users/rm001/Sentrix-Thin-Agent-work-20260805`，交付到 `asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web` 走 `psh` 分支
**执行原则**：本文件为计划稿，用户点头前**不改任何生产代码**；每阶段结束需 153 上单元测试全绿 + Retrieval-only benchmark 通过再前进

## 0. 用户已确认的决定 (2026-08-06)

| # | 决策 | 落点 |
|:-:|---|---|
| D1 | **本地允许持有数据副本** | R0/R1 在 153 只读诊断；本地可放脱敏只读副本或人工 fixture 用于开发 (审阅 §6.1) |
| D2 | **Hidden Acceptance Set 从现有 60 case 划出（规模 15-20 已确认）** | 记录：审阅 §6.2 推荐用户新造 20-30 case（更能验证泛化）；用户选择从 60 case 划分 → 按划分执行，R7 报告会注明过拟合风险等级 |
| D3 | **Embedding 备选由 Agent 定** | Visual 备选 **Chinese-CLIP** (`OFA-Sys/chinese-clip-vit-base-patch16`)；Text 备选 **bge-m3**。仅在 R1B 评估不合格时启用；两者是独立 adapter |
| D4 | **主 Answer 模型继续 gemma4:12b** | 12B 用于 Answer/Verify；Parser 切小模型走 153 现有 2B 切换逻辑（D6） |
| D5 | **配置文件需要** | 双层：`configs/retrieval/defaults.json` 进 git + `data/configs/retrieval.local.json` 不进 git |
| D6 | **153 已有 2B 小模型 + 已写切换逻辑（用户确认）** | 153 有 `gemma4:e2b-it` 2B（8100 e2b_server，`start_sentrix_e2b.sh`，`/api/vlm-backend POST active=e2b_lora`）；切换逻辑已写好但属**另一任务的 uncommitted 改动**（本计划不修改，只接线）。本地工作副本 `GammaClient` **无** E2B 逻辑（已 grep 核实）→ R5 需在 153 核验该路径，`model_routing` 设计成可插拔 backend 选择，本地默认单 backend 通过测试 |
| D7 | **ANN scope 策略不预留增长** | 采用"按 scope 独立索引"为主，不预留全局+oversampling（用户确认"不用预留"） |
| D8 | **R1B 切换流程已确认** | "评估报告 → 用户点头 → 换 adapter + 索引重建"，不要求预装权重 |

**当前模型**：`gemma4:12b`，来源 `OLLAMA_MODEL` 环境变量（`model_clients.py:194`，默认 `gemma4:12b`；`start_sentrix_api.sh:26` 同样默认）。12B 全量 8GB 但当前 VRAM 仅 189MB 驻留。Parser 小模型候选在 153 侧已存在（D6）。

## 0.1 审阅报告采纳状态

- **P0 全部 20 条**：P0-1 至 P0-20 全部采纳，逐条落实位置见 §5/§6/§7。
- **P1 全部 8 条**：P1-1 至 P1-8 采纳，落实位置见 §5.4 / §5.3 / §5.2 / §5.1 / §6 R4 / §5.5 / §6 R1A / §6 R0。
- 阶段顺序按审阅 §5 调整：新增 **R1B（Embedding 独立能力评估）** 与 **R3B（Seed-based Adjacency + duplicate grouping）**。

---

## 1. 不变量与本阶段范围冻结

保留（输入报告 §0 + 原计划保留项 + 审阅 §4 第 1-16 条）：
- Asset 是最终证据源
- `QuerySpec` / `Constraint` 三层 strictness / `EvidencePacket` / statement×evidence 校验 / `sanitize_query_parse` 剔除模型越权字段
- 模型负责开放语义，代码负责 scope/viewer/时间/人物/媒体/否定/证据边界
- 普通聊天不读家庭事实、未授权不重读原图、未授权不改记忆
- `raw_json` 不动、FMA 5173 不动、e2b_server 相关 uncommitted 改动不动
- Result level 合同**保持** exact / strong / approximate / excluded，**不新增 possible** (P0-14)

本阶段范围冻结（输入报告 §3）——以下保留代码但不作为完成声明：
- Core Memory 上线宣告
- Correction 端到端上线宣告
- 高级人物/事件/时间线工具完成宣告
- 主动回忆、多 viewer、Answer Writer 风格优化、Formation pipeline 大规模改造

本阶段允许修改的文件：
`memory_gate.py`、`thin_agent.py`、`query_parser.py`、`query_contracts.py`（兼容扩展）、`evidence_retrieval.py`、`retrieval_indexes.py`、`retrieval_ann.py`、`answer_composer.py`、Agent 模型调用编排、benchmark/replay 脚本、相关 API trace 与 feature flag、单元与集成测试。**新增**目录 `backend/retrieval/` 与 `backend/embeddings/`。

---

## 2. 当前调用链真实核验（本地代码 grep 结果）

**核验方法**：`grep` 于 `/Users/rm001/Sentrix-Thin-Agent-work-20260805/backend/*.py`。

| 事实 | 证据 |
|---|---|
| `EvidenceRetrievalKernel.retrieve()` 仅逐对扫描 Asset×Observation | `evidence_retrieval.py:83-122`：只 `store.list_assets` + `store.list_observations`；无 `retrieval_ann`/`retrieval_indexes`/`search_vectors` 引用 |
| `backend/retrieval_ann.py` 建了 `HnswlibIndex`，`backend/*.py` 除自身外零调用 | grep `retrieval_ann\|HnswlibIndex\|create_index` 结果只在 `retrieval_ann.py` 自己 |
| `backend/retrieval_indexes.py` `observation_search_terms` 建了但无查询方 | 同 grep，除自身外零调用 |
| `ThinAgentRuntime` 构造只接受 `store`+`gamma`，从未持有 clip | `thin_agent.py:__init__`；`app.py:30` 用 `pipeline.clip` 只喂旧 `MemoryAgent` |
| 旧 `agent.py` 走 CLIP，但用 SQLite 逐行余弦 | `agent.py:1065-1066`：`self.clip.embed_text(query)` + `store.search_vectors(...)` |
| Kernel `_contains` 分词后 `all(term in ...)` | `evidence_retrieval.py:69-76`：`re.findall(r"[\w一-鿿]+")` + 长度>1 + `all()`；中文双字组合仍过宽 |
| `_condition` 多值字段 miss=unknown 已修 | `evidence_retrieval.py:158-161`：`_OPEN_WORLD_LIST_DIMENSIONS` |
| 生产 API 8091 flag 全开但 kernel 从未调 ANN | flag on 只让索引可创建/保存，不代表 retrieve 读它 |
| 当前生成模型 | `model_clients.py:194`：`OLLAMA_MODEL` 默认 `gemma4:12b`；12B 全量 8GB / VRAM 驻留 189MB |

**核心失败面**（输入报告 §1 四条独立链 + 审阅三个典型错误）：
- **A. 生产检索路径不完整**：已建 ANN/视觉向量/文本向量/派生投影都不在 `retrieve()` 里。← 审阅错误 1
- **B. Gate 单点失败**：`mode=none` 后 `thin_agent._normal_chat` 直接接管，短家庭短语失去检索机会。
- **C. 语义匹配错误**：`_contains` 双字 all-match 造成 album1-01 命中 10 张无关；`_allowed_facts` 无按 condition 去重。
- **D. 模型基础设施不可用**：12B partial VRAM → 单次 9-90s 波动，端到端 240s（硬上限 20s）。
- **审阅附加错误风险**：用文本相似度实验冒充 CLIP 跨模态能力（P0-2）；为修模型不稳又加长度/前缀/固定阈值分类器（P0-6，审阅 §9.3）；索引存在但生产路径不查（P0-4）。

---

## 3. Benchmark 隔离协议（输入报告 §2 + 审阅 P0-16/P0-17）

### 3.1 三集合分层

| 集合 | 内容 | 用途 | 允许运行时读取? |
|---|:---:|:---:|:---:|
| **Regression Set** | 现有 60 case + 6 case 实测 | 防回归 / 通道消融 / 硬约束违反 | ❌ |
| **Development Set** | 用户/本阶段新增合成 case + **从 60 case 划出的 Hidden 之外的子集** | 选融合策略、默认阈值 | ❌ |
| **Hidden Acceptance Set** | **从现有 60 case 划出（用户 D2）**，代码 Agent 冻结答案不看细节 | 最终验收，检查过拟合 | ❌ |

**划出规则（D2 落实）**：
- 从 60 case 中随机抽 15-20 case 冻结为 Hidden，覆盖查询类型分布（时间 / 地点 / 人物 / 物品 / 衣着 / 复合 / 空 GT 各至少 1）
- 划分脚本 `scripts/benchmarks/split_hidden_set.py` 在 R1A 前运行一次，`docs/baseline/hidden_set_manifest.json` 记录划出与理由；**实现期间该 manifest 的 GT 内容对 Agent 加密**（只有 key 与 answerability，无 file list）
- 记录审阅 §6.2 提醒：从现有 60 case 划分的 Hidden 泛化力弱于全新 user case；R7 报告按"划分型 Hidden"标注结论可信度

### 3.2 Case 标注扩展（P0-16 / P0-17）

每个 case 从 `{ground_truth: []}` 扩展为：

```json
{
  "query_cn": "...",
  "exact_asset_ids": [],
  "acceptable_approximate_asset_ids": [],
  "forbidden_asset_ids": [],
  "empty_policy": "strict_empty|allow_approximate",
  "answerability": {
    "metadata": true|false,
    "confirmed_entity": true|false,
    "lexical_observation": true|false,
    "visual_semantic": true|false,
    "text_semantic": true|false,
    "external_geo_required": true|false,
    "formation_missing": true|false,
    "ambiguous_gt": true|false
  }
}
```

- `strict_empty`：用户可见 Asset 必须为 0；`allow_approximate`：可展示标注过的 approximate，不计 FP
- **R1 前**必须完成 60 case 的 answerability 标注 + GT 不一致冻结（现有 `9/8`、`3/1` 等逐条定解释规则），防止把"系统从未存禹城市"归因为 ANN 失败
- 指标至少分别报告：理论可回答子集 / 需外部地理编码子集 / Formation 缺失子集 / 全集

### 3.3 运行时禁入清单（输入报告 §2.1 + §20 + 审阅 §7.2）

以下写入 `backend/*.py` 判定为不合格：
- benchmark 题目原句片段、GT 文件名、Asset ID、case-specific 分支、相册 if 分支、同义词白名单、case-specific 权重、hard-coded query→result map
- Parser few-shot 出现真实 benchmark 题目
- 配置默认值含 benchmark 数据

**校验手段**：`backend/tests/test_no_benchmark_runtime_dependency.py`，扫描 `backend/*.py`、`configs/retrieval/defaults.json`、`scripts/runtime/`、`scripts/maintenance/`，禁止匹配任何 case 的 query_cn 或 GT 文件名。CI 前置，任何阶段 red gate。

### 3.4 使用 benchmark 的正当方式

允许：Recall/Precision/MRR/拒答率/硬条件违反率；分析某 GT 未召回的通道 rank；从失败抽象**通用**修复原则（"多值未命中→unknown"）。
不允许：因单 case 失败写 case-specific 规则；对 GT Asset 特殊 boost；用文件名建别名。

---

## 4. 目标生产调用链（输入报告 §4 + 审阅 P0-6/P0-7/P0-8）

```text
用户消息
  │
  ▼
Request Interpreter (memory_gate + query_parser)
  ├─ 明确一般任务（writing prefix / 翻译 / 假设 / "不用查我的记忆"）→ 明确 none ──▶ 普通聊天路径
  ├─ 明确家庭证据动作（日期+confirmed entity / selected_entity / correction payload /
  │   当前对话围绕已检索证据追问 / 明确原图请求）→ 明确 evidence ──▶ 直接检索
  └─ 其他 → parser draft（mode + actions + facets + ambiguities）
                        │
                        ▼
        Gate 综合判定（不依赖长度、不单独信自报 confidence）：
          - 一般任务结构  - 家庭证据动作  - 对话焦点
          - actions/facets 冲突  - selected entity/feedback  - 可访问候选
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
     明确 none                 evidence/强信号            其余（短名词 / 不确定）
        │                         │                         │
   普通聊天路径                正式检索                Neutral Retrieval Probe
   （无 parser、无 retrieval）   │                    （raw text 中性查询，见下）
        │                         │                         │
        │                         ▼                    ┌────┼───────────────┐
        │                 Immutable QuerySpec          无候选  强候选(多信号)  冲突/弱
        │                         │                    │       │              │
        │                         ▼                 普通回应  升级 evidence   询问澄清
        │              Evidence Retrieval Kernel        │       │（不生成事实）│
        │            1. hard prefilter（可前置的过滤）    │       ▼              │
        │            2. 多通道 recall                    │  重新生成完整 QuerySpec
        │               metadata/entity/lexical/         │       │
        │               visual/text（Primary）           │       ▼
        │            3. Asset merge                      │  正式检索（同 Kernel）
        │            4. seed quality gate ──▶ 5. adjacency 扩展（第二阶段）
        │            6. condition evidence（候选→逐条件 evaluate）
        │            7. hard postfilter（contradicted/must_not 剔除）
        │            8. fusion/ranking（evidence_class 分级，非平权）
        │            9. level = exact/strong/approximate/excluded
        │                         │
        │                         ▼
        │                   EvidencePacket
        │                         │
        │          ┌──────────────┴──────────────┐
        │          ▼                             ▼
        │   简单确定性回答                    可选单次 AnswerModel
        │   （0 生成）                            （≤1 生成）
        │          └──────────────┬──────────────┘
        │                         ▼
        │             回答 + 授权证据入口（image_results 仅显式请求）
        │
        ▼
  （probe 失败的普通回应：禁止用通用描述掩盖，可问"你是想找相关照片，还是聊这个？"）
```

**Neutral Retrieval Probe 规则**（P0-7 / P0-8）：
- 使用 **raw user text** 构建中性查询，不依赖可能错误的 QuerySpec：
  - scope/viewer 来自请求上下文
  - media 默认允许 image
  - `whole_query` 保留完整原文
  - **不构造未经确认的 hard semantic constraints**
- 走共享的 retriever 通道（同一索引/权限/scope/embedding/trace），但策略不同：候选预算小、不跑 adjacency、不做复杂 condition conclusion、不产生用户事实、只输出 upgrade/clarify 信号（P1-5）
- **强候选判定为多信号组合，不是单一 raw cosine 阈值**（P0-8）：
  - 通道内校准分；Top-1 vs Top-2 margin；多通道一致性（≥2 通道 agreement）
  - 当前 scope；是否存在 exact lexical phrase；向量空间质量状态；结构化冲突；候选是否稳定出现在 whole query 与 facet 两个 view
  - 校准配置按空间分别保存（`configs/probe_thresholds.json`），Development Set 训、Hidden Set 冻结验证
- Probe 结果**不得直接变成最终 EvidencePacket**；升级后重新生成完整 QuerySpec 再正式检索

---

## 5. 模块设计与文件级改动清单

### 5.1 新目录 `backend/retrieval/` 与 `backend/embeddings/`

```text
backend/retrieval/
  __init__.py          # 导出接口 + 工厂
  base.py              # Retriever Protocol / RetrievalQuery / CandidateHit / HardFilterContext
  metadata.py          # MetadataRetriever（时间/media/scope/GPS 范围/batch/device）
  entity.py            # EntityRetriever（confirmed entity/face-bridge/event participant）
  lexical.py           # LexicalRetriever（FTS5 预分词投影，见 §5.3）
  visual_ann.py        # VisualAnnRetriever（text→visual embed→visual ANN top-K）
  text_ann.py          # TextAnnRetriever（text→text embed→semantic/episodic ANN top-K）
  adjacency.py         # AdjacencyRetriever（seed 后扩展器，见 §5.5）
  fusion.py            # RRF + evidence_class 分级融合（见 §5.4）
  config.py            # RetrievalConfig 统一配置对象（P1-4）
  probes.py            # NeutralProbe 信号聚合（P0-8）

backend/embeddings/
  base.py              # VisualQueryEmbedder / TextQueryEmbedder Protocol（P0-3）
  clip_visual.py       # ClipAdapter 的视觉 text-embed 适配（ViT-B-32 或 Chinese-CLIP）
  clip_text.py         # ClipAdapter 的文本 embed 适配（仅文本）
  bge_text.py          # bge-m3 文本 embed（备选，R1B 不合格才启用）
```

**EmbeddingRouter**（P0-3 关键改动，替代"把 clip 注入 ThinAgent"）：

```text
Application Composition Root
  → EmbeddingRouter（持 VisualQueryEmbedder + TextQueryEmbedder）
  → EvidenceRetrievalKernel(retrievers, embedding_router)
  → ThinAgentRuntime(evidence_service)   # ThinAgent 不知道 CLIP/HNSW/维度
```

```python
class VisualQueryEmbedder(Protocol):
    model_id: str
    dimension: int
    def embed_query(self, text: str) -> list[float]: ...

class TextQueryEmbedder(Protocol):
    model_id: str
    dimension: int
    def embed_query(self, text: str) -> list[float]: ...
```

- Visual ANN 必须用与 Asset image vectors 相同的图文兼容模型（ViT-B-32 或 Chinese-CLIP）
- Text ANN 可用独立多语言文本 embedding（bge-m3）
- Thin Agent、Gate、QuerySpec 对 embedding 实现零依赖
- `app.py` Composition Root 组装 router + kernel；`ThinAgentRuntime` 接收的是 `evidence_service`（封装 kernel）而不是 `clip`

### 5.2 `evidence_retrieval.py` 改造（含 P0-5/P0-14/P0-18/P0-19）

新版 `retrieve()` 骨架：

```python
def retrieve(self, spec: QuerySpec) -> EvidencePacket:
    filters = HardFilterContext.from_spec(spec)          # 可前置过滤部分
    query = RetrievalQuery.from_spec(spec, embedder=self.embedding_router)

    # 并发策略（P0-18）：
    #   DB 型 retriever（metadata/entity/lexical）串行或独立只读连接
    #   ANN/embedding 型（visual/text）并行
    primary = {r.name: r.retrieve(query, filters, limit=spec.recall_limit)
               for r in self.retrievers if r.kind == "primary" and r.enabled}

    # Asset merge —— 同 asset 多通道命中聚合（保留每个 hit 的通道与 rank）
    merged = merge_candidates(primary)

    # Seed quality gate（P0-9）→ adjacency 扩展
    seeds = select_seeds(merged, gate=filters)
    expanded = adjacency.retrieve(query, seeds, filters) if adjacency.enabled else {}
    merged = merge_candidates({**primary, "adjacency": expanded})

    # Condition evidence —— 候选→逐条件 evaluate（这里保持 _condition 合同）
    for cand in merged.values():
        cand.condition_results = self._evaluate_conditions(cand.asset, cand.observations, spec)

    # Hard postfilter —— contradicted/must_not/scope 剔除
    filtered = [c for c in merged.values() if not c.hard_violated]

    # Fusion + ranking（evidence_class 分级，P1-1）
    ranked = self.fusion.rank(filtered, spec)

    # Level = exact/strong/approximate/excluded（不新增 possible，P0-14）
    return build_packet(ranked, spec)
```

**并发 SQLite 策略（P0-18）**：
- 每个并发 DB retriever 使用独立只读 SQLite 连接；或 DB 查询串行、仅 ANN/embedding 并行
- 模型调用不得发生在 SQLite 写事务内
- 检索超时正确取消任务并释放连接

**CandidateHit 分数方向标准化（P0-19）**：

```json
{
  "asset_id": "asset_x",
  "retriever": "visual_ann",
  "raw_score": 0.23,
  "score_kind": "cosine_distance",
  "higher_is_better": false,
  "rank": 1,
  "calibrated_score": null,
  "source_id": "obs_x",
  "source_revision": "rev_x",
  "matched_text": null,
  "metadata": {}
}
```

- RRF 主要用 `rank`；Probe 只能用空间校准后的 `calibrated_score`
- `raw_score` 与 `higher_is_better` 分离，不再只有一个裸 `score: float`

**渐进接入**：`SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=off` 时保持旧行为；shadow diff 写入 `retrievalTrace.shadow_multi_retriever_diff`。

### 5.3 `LexicalRetriever` 与中文 FTS（P0-1 重写）

**废除**：单字/双字 `all(term in haystack)` contains 语义。`_contains` 只保留完整子串校验。

**方案 A：预分词投影（第一版采用）**：
- 在 `observation_search_terms` 或独立投影中，每行生成：
  - 完整规范化字段
  - 通用词级 token（通用 tokenizer，不含 benchmark 专用同义词）
  - CJK bigram（必要时 trigram）
  - 字段类型
- 空格分隔写入 FTS，示例（仅演示通用规则，非 benchmark 数据）：
  ```
  原字段：浅黄色毛绒睡衣
  索引 token：浅黄色 黄色 毛绒 睡衣 浅黄 黄色 色毛 毛绒 绒睡 睡衣
  ```
- Tokenizer 必须独立可测（`test_lexical_retriever.py` 断言实际 token，不只测 SQL 能跑）

**方案 B：FTS5 trigram tokenizer** — 作为对照，验证中文短语/两字对象/长组合/索引大小/Precision 后决定。

**FTS 表设计**：
- `asset_id` / `observation_id` / `scope_id` 为 **UNINDEXED 元数据**
- 更新机制**明确二选一**（不留"upsert"模糊）：delete+insert 增量，或全量 rebuild（脚本 `scripts/maintenance/rebuild_lexical_fts.py`）
- FTS 只负责**候选召回**；是否支持某条件仍由 condition evidence 决定；单汉字不能作为 `matched` 事实依据
- whole-query 与 facet 双路（P1-2 / §8.2 输入报告）：`SELECT ... MATCH ? ORDER BY bm25(...)` 用完整原句 + 对每个 semantic constraint 单独 MATCH 归一化
- 同义扩展由 embedding 或独立通用词法组件完成，不写 benchmark 同义词

**召回来源 vs 证明来源分离（P1-2）**：`CandidateHit.matched_text` 只解释为什么被召回；`ConditionResult` 单独记录 proof source / proof strength / subject binding / source revision / allowed certainty，**不把 FTS 召回文本直接复制成 matched**。

### 5.4 融合策略 `retrieval/fusion.py`（P0-13/P1-1/P1-4）

**RRF 默认**：`k=60`，主要用 rank。

**evidence_class 分级（P1-1）**，不所有通道平权：
- **Hard constraints**（时间/media/scope/must_not）：只过滤，**不参加 RRF**
- **Deterministic soft anchors**（metadata 的精确项、confirmed entity）：允许稳定 boost
- **Lexical / Visual / Text**：进行 rank fusion
- **Adjacency**：只能继承 seed 的等级，不得独立获得高等级（P0-9）

```python
def rrf_score(rank_by_channel: dict[str, int], k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in rank_by_channel.values())
```

**RetrievalConfig（P1-4）**：统一配置对象替换零散 env 组合：

```json
{
  "schema_version": 1,
  "multi_retriever": true,
  "channels": {
    "metadata": true, "entity": true, "lexical": true,
    "visual": true, "text": true, "adjacency": false
  },
  "fusion": "rrf",
  "adjacency": {"max_seeds": 8, "per_seed_budget": 6, "event": 4, "batch": 2, "time_window": 2}
}
```

`SENTRIX_RETRIEVER_FUSION="rrf" | "weighted_norm"` 仍可用作切换，但权重从配置读，不硬编码。

### 5.5 `AdjacencyRetriever`（P0-9 第二阶段扩展器）

**不参与第一轮并行**。顺序：
```text
Primary Recall（metadata/entity/lexical/visual/text）
→ Asset merge
→ seed quality gate
→ Adjacency Expansion
→ 新候选重新执行全部硬过滤和条件评估
→ Fusion
```

预算：最大 seed 数、每 seed 最大扩展数、同 Event/batch/时间窗分别的预算；`all_relevant` 与 `best` 模式不同预算；**禁止仅因邻接成为 matched**；邻接新候选必须重过硬条件。

### 5.6 ANN 接入与 Manifest（P0-4/P0-5/P0-10）

**Manifest**（`retrieval_ann.py` 侧边 JSON 升级）：

```json
{
  "space": "visual",
  "model_id": "ViT-B-32",
  "checkpoint_hash": "...",
  "dimension": 512,
  "normalized": true,
  "distance": "cosine",
  "source_type": "asset",
  "source_count": 100000,
  "source_revision": "...",
  "id_map_checksum": "...",
  "built_at": "...",
  "index_version": 1
}
```

- **加载时校验**：模型、维度、归一化、ID map 一致；不一致 → 拒绝启用该通道，trace 记 `index_incompatible`，**不得静默查询旧索引**
- **全量 rebuild 用临时文件，成功后 atomic swap**
- 增量更新记录 index revision；stale hit 回表时检查 source revision，过期结果不进 packet
- **scope/硬过滤策略（P0-5）**：明确选型 ——
  - scope 数量有限（相册边界明确）→ 按 scope 建独立索引（**主方案**）
  - 或全局索引 + 自适应 oversampling：取 `K × oversample_factor` → 回 SQLite 执行 scope/time/media 过滤 → 不足时扩大 ef/K 重试 → 到候选预算或硬上限停止
  - 明确哪些过滤 ANN 前可执行（scope、media type 可入索引 label/独立索引）、哪些只能 ANN 后（时间区间、must_not）
  - 空 scope 含义与权限边界在 `HardFilterContext` 文档化
- **生产 fallback（P0-10）**：ANN 故障时**关闭该通道**，继续 structured/lexical 等健康通道，EvidencePacket gaps/trace 说明语义通道不可用，必要时请用户加线索。**SQLite 全量余弦扫描只允许用于**离线 benchmark / ANN recall 对照 / 小规模 fixture / 显式运维诊断；**不得在普通生产请求自动启用**。

**Index readiness 状态（P1-3）**：每个通道 `ready | rebuilding | stale | incompatible | unavailable | disabled`；API trace 区分 `invoked=true, result_count=0` 与 `invoked=false, reason=index_incompatible`。验收不要求每请求每通道 `candidate_count>0`（"调用成功但无候选"是合法结果）。

### 5.7 Gate 与 Parser 改动（P0-6）

**`memory_gate.py`**：
- **删除**"消息长度 >6 → repair"（P0-6：长消息可能是"请解释量子纠缠"）
- **删除**"`draft.confidence < 0.7 → ambiguous`"（P0-6：模型自报置信度未校准）
- `draft.confidence` 只进 trace，**不单独决定路由**
- Gate 综合判定（不依赖单一信号）：
  - 明确一般任务结构（writing prefix / 翻译 / 假设 / "不用查记忆"）→ none
  - 明确家庭证据动作（日期+confirmed entity / selected_entity / correction payload / 已检索证据追问 / 原图请求）→ evidence
  - parser actions/facets 与家庭对象冲突时即使 mode=none → ambiguous
  - parser mode=none 且是明确一般任务 → none
  - 其他（尤其短名词短语）→ 允许 neutral probe

**`query_parser.py`**：
- Prompt 缩短 + **通用结构 few-shot**（合成占位内容，不含真实 benchmark，P1-6）
- `seed=42` + `num_ctx=4096` + `num_predict=512` 显式参数
- `_validate` 收紧（**不用长度阈值**）：
  - `actions==[]` 且 `mode=="evidence"` → repair
  - 明确日期/人物/原图语法出现在原文但 draft 丢失 → 确定性 overlay 恢复
  - repair 失败 → 降级安全 spec：`mode="ambiguous"`, `allow_probe=True`, `actions=[]`
- **不允许**恢复关键词分类器作 fallback（输入报告 §20.5）

### 5.8 Thin Agent 与 Answer Composer 改动（P0-13/P0-14/P0-16）

**`thin_agent.py`**：
- 构造改为接收 `evidence_service`（封装 kernel + embedding_router），**不再直接依赖 clip**（P0-3）
- `_evidence_answer` 分派：短名词+候选强 → 简单确定性回答（0 生成）；复杂 → `complex_answer.ComplexAnswerBuilder`
- Probe 只调 `EvidenceRetrievalKernel.retrieve(spec, mode="probe")`

**`answer_composer.py`**：
- `_allowed_facts` 按 `condition_key` 去重（`dict.fromkeys` 保序）
- 用户可见模板人类可读（内部 condition_key / ANN 分数 / retrieval trace / DB 表名禁止出现）
- **空结果按 empty_policy（P0-16）**：
  - `strict_empty` → 强制回答"当前记忆中没有找到足够匹配的原始证据"
  - `allow_approximate` → 可展示标注 approximate 并说明差异，不计 FP
  - 空 EvidencePacket 且是家庭查询**禁止**走 normal chat 编造通用描述

**近重复（P0-13）**：
- 优先用已有 SHA-256 / 感知哈希 / 明确 near-duplicate group；CLIP cosine 只作辅助分组依据；阈值在 Development Set 验证（不硬编码 0.98）
- `best/top_k`：可展示代表图并标记"组内还有 N 张"
- `all_relevant`：**不删除组员**，折叠成组但允许全部展开
- Retrieval metrics 计算保留原 Asset，不因 UI 分组改变召回率

---

## 6. 阶段拆分（R0 → R7，审阅 §5 顺序）

**执行原则**：每阶段本地全部单元测试 + Retrieval-only benchmark 通过 → 按 §9 交付协议传 153 → 153 重跑 → 153 `psh` 唯一正式 commit → 重启 API 8091 → 输出 `docs/baseline/thin-agent-phase-R{X}.md` 报告。

### R0 · 真实调用链 + case answerability 核验（只读，无生产代码改动）

**产出**：
- `docs/baseline/thin-agent-phase-R0-call-chain.md`：Kernel retriever、ANN load/query 调用点、search_terms 调用点、parser mode 决策、每轮模型调用；用 6 实测 case 逐步展开
- `scripts/benchmarks/inspect_retrieval_case.py`（P1-8）：支持 `query-only` / `query+expected` / `asset-only` 三种模式（不强制 gt_asset_id，可诊断真实用户问题与 Hidden 集）
  - 输出：asset metadata / observation 原始字段 / canonical 字段 / event / visual vector 存在性 / text vector 存在性 / index revision / 各 retriever rank / 硬过滤结果 / 条件矩阵 / 排除原因
- **60 case answerability 标注 + GT 一致性冻结（P0-17）**：脚本 `scripts/benchmarks/audit_benchmark_cases.py` 输出 `docs/baseline/benchmark_case_audit.json`，逐 case 标注 answerability 字段 + 冻结 `9/8`、`3/1` 等解释规则

**退出条件**：能解释 6 个实测 case 的每一步；case 分类完成（recall path 未接 / embedding 不支持 / lexical 不命中 / formation 缺字段 / 人物 binding 缺失 / GT 歧义 / 外部地理依赖）。

### R1A · Retrieval-only runner + 现有通道基线

**产出**：
- `scripts/benchmarks/evaluate_retrieval_kernel.py`：给定 `(query, expected, forbidden, scope, empty_policy, cached_query_spec?)` → ranked Asset IDs、Recall@1/5/10/20、MRR、Precision@5、all_relevant recall、empty-GT FP、硬约束违反、per-channel 贡献、GT rank
- `scripts/benchmarks/evaluate_parser_retrieval.py`：cached QuerySpec vs real Parser 组合
- 通道消融模式：`--channels lexical` / `visual` / `text` / `structured` / `hybrid_no_adjacency` / `full_hybrid`
- 每次 run 输出 `docs/baseline/retrieval_baseline_YYYYMMDD.json` 存档
- 运行位置：**153 只读**（真实数据）；本地用脱敏副本/人工 fixture 做开发

**退出条件**：完整 60 case 可在不调用回答大模型时运行（p95 ≤ 5s/query），输出通道消融基线；划分脚本 `split_hidden_set.py` 已冻结 Hidden manifest。

### R1B · Visual/Text Embedding 独立能力评估（P0-2 新增阶段）

**两个独立评估器，不共用"一个 ClipAdapter 证明一切"**：

**Visual Cross-modal Evaluation**（`scripts/benchmarks/evaluate_visual_crossmodal.py`）：
- 输入：独立 Development 图片 + 通用查询标注（**非 Regression 原句**）
- 流程：query text → text encoder → image ANN/brute-force → 正确图片 rank
- 指标：Recall@1/5/10、MRR、正负样本间隔、中文改写稳定性、颜色/材质/物品/活动/场景/组合描述
- 不得用 Regression Set 原句证明通用能力（只可最终 regression 评估时测）

**Text Retrieval Evaluation**（`scripts/benchmarks/evaluate_text_retrieval.py`）：
- 分别验证 query → caption / activity/place/object/clothing / Event summary / OCR
- 指标：Recall@K、同义改写、否定/无关文本、字段级贡献

**结论规则**：
- Visual encoder 不合格 → 只换 Visual adapter（Chinese-CLIP，D3）
- Text encoder 不合格 → 只换 Text adapter（bge-m3，D3）
- 两者独立选型，不共用隐含假设

**退出条件**：两套评估器在 153 运行，输出 AUC/Recall/MRR 报告；不合格 → 生成 Adapter 选型决策（模型大小/本地部署/索引重建/延迟），用户确认后再换，**不继续假设当前向量可用**。

### R2 · 接通 Metadata / Entity / Lexical / Visual / Text

**变更清单**：
- `backend/embeddings/base.py` + `clip_visual.py` + `clip_text.py`（可选 `bge_text.py`）
- `backend/retrieval/{base,metadata,entity,lexical}.py`；`visual_ann.py` / `text_ann.py` 读磁盘 hnswlib + **Manifest 校验**（P0-4）
- FTS5 预分词投影（P0-1 方案 A）+ `observation_search_terms` 查询方
- `EvidenceRetrievalKernel.retrieve()` 多通道 recall，`SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1=1` 启用；shadow diff
- `app.py` Composition Root 组装 EmbeddingRouter + Kernel；`ThinAgentRuntime` 收 `evidence_service`
- ANN scope 策略（按 scope 独立索引为主，P0-5）
- `retrievalTrace` 扩展：`channels: {name, status, invoked, candidate_count, latency_ms, top_calibrated_score}`

**验收（审阅 §7.1）**：
- 真实请求 trace 显示每个启用通道被调用或给出明确不可用原因（`invoked=false, reason=...`）
- ANN Manifest 与 query embedder 一致；不一致拒绝启用
- `observation_search_terms`/FTS 在生产路径实际被查询
- 与 R1A 基线比 Recall/MRR 明确提升、FP 相对下降
- 无硬约束违反；每个失败有层级归因

### R3 · 词法、条件证据、融合与结果等级

**变更清单**：
- 删除 `_contains` 单字/双字 all-match；只保留完整子串校验
- `_MATCHED_SOURCE_TYPES` 白名单（P1-2）：`asset_metadata / observation_field_exact / entity_bridge_confirmed / ocr_exact` 可 matched；向量/FTS 只能 possible → 结果归 approximate
- `fusion.py`：RRF + evidence_class 分级（P1-1）
- `_condition` 收紧（保留 2R-6：多值 miss=unknown、subject binding 才 contradicted）

**验收（审阅 §7.4 阶段工程门槛，与 R7 最终指标分离 P0-15）**：
- 通道消融完整；hard violation = 0；strict-empty case user-visible FP = 0；approximate policy 符合 case 标注
- 与 R1A 比 Recall/MRR 明确提升、FP 相对下降
- 所有失败有归因；无 case-specific 规则

### R3B · Seed-based Adjacency + duplicate grouping（P0-9/P0-13）

**变更清单**：
- `adjacency.py` 作为第二阶段扩展器，预算可配（§5.4 RetrievalConfig）
- near-duplicate grouping（§5.8）：SHA/感知哈希主、CLIP 辅助、all_relevant 不删除

**验收**：仅 seed 质量通过 gate 后扩展；邻接新候选重过硬条件；`all_relevant` 组折叠可全展开、metrics 不丢 Asset。

### R4 · Gate + Neutral Probe

**变更清单**：
- `memory_gate.py::MemoryGate.decide` 返回 `GateDecision`（含 proposed_mode / allow_probe / reason）
- `retrieval/probes.py`：Neutral Probe 多信号聚合（P0-8）
- `thin_agent.py`：ambiguous → probe → upgrade/clarify
- Probe 阈值从 `configs/probe_thresholds.json`（Development 训、Hidden 冻结验证）；缺文件 fail-safe 到"总是询问澄清"
- Parser `_validate` 收紧（不用长度阈值）

**验收（审阅 §7.3）**：
- 明确普通任务不 probe；明确家庭请求不因 parser `none` 丢失
- 短名词用 neutral probe；模型自报 confidence 不单独决定路由
- Weak probe 对短家庭短语优先澄清，不生成通用幻觉回答
- 短名词 GT 查询因 `none` 丢失率 = 0；普通任务误触发率受控并记录

### R5 · 模型路由、统一 deadline、调用预算（P0-11/P0-12）

**变更清单**：
- `backend/model_routing.py`：ParserModel / AnswerModel / VerifyModel 三角色；backend 选择可插拔（`ollama_local` / `e2b`）
- **接线 153 现有 2B 切换逻辑（D6）**：
  - 本地工作副本 `GammaClient` 无 E2B → `model_routing.py` 定义 `BackendRouter`，默认 `ollama_local`（12B 全角色）
  - 153 侧核验 `/api/vlm-backend POST active=e2b_lora`（8100）可用后，用 `SENTRIX_PARSE_BACKEND=e2b` 路由 Parser 到 2B，Answer/Verify 保持 12B
  - **不修改** 153 上 e2b_server 的 uncommitted 改动；只在其上新增调用方
  - `GammaClient` 支持 `parse_model` / `answer_model` / `verify_model` + `parse_backend` 分离
- **统一 request deadline（P0-11）**：`request_deadline = now + 20s`；每阶段从剩余时间拿预算：

| 阶段 | 预算 |
|---|---|
| Parser | 2-4s |
| Retrieval | ≤5s |
| Answer | 4-7s |
| 序列化 + API 余量 | ≥2s |

- 超时行为：Parser 超时 → neutral probe 或安全 fallback；单通道超时 → 取消该通道保留其他；Answer 超时 → 确定性 evidence summary。**不等 30s 才触发 API fallback**
- Circuit breaker 按模型角色与错误类型独立记录（不"连等 3 次 30s"）
- 主 Answer 模型保留 gemma4:12b（D4）；Parser 切小模型走 D6 现有 2B 路径（flag `SENTRIX_MODEL_SPLIT_V1` + `SENTRIX_PARSE_BACKEND`）
- `scripts/maintenance/probe_model_health.py`：cold/warm latency + JSON 合法率 → `docs/baseline/model_health_YYYYMMDD.json`

**修正后的调用预算表（P0-12）**：

| 路径 | Parser | Retrieval | Answer/Writer | Claim/Verify |
|---|:-:|:-:|:-:|:-:|
| 普通聊天/写作 | 0 | 0 | 1 | 0 |
| 明确图片短查询 | 0-1 | 1 | 0-1 | 0 |
| 简单 evidence | 0-1 | 1 | 0-1 | 0 |
| 复杂人物/比较（正常） | 1 | 1 | 1 | 2 |
| 复杂（Repair 触发） | 1 | 1 | 1 | 2 + 1 repair = 最大 5 |

**验收（审阅 §7.5）**：Retrieval-only p95 按"每个 query"计算；统一 API deadline ≤20s；simple evidence 生成调用 ≤2（parser + optional answer）；若硬件不达标输出 `thin-agent-phase-R-infra-blocker.md` 实测阻塞，不删除正确性步骤伪造达标。

### R6 · Answer

**变更清单**：
- `answer_composer.py` 去重 + empty_policy 拒答 + 人类可读模板
- 近重复 grouping 用户可见呈现

**验收**：Regression 6 case 不再重复模板；空 GT 按 empty_policy 处理（strict 拒答 / allow_approximate 说明差异）；E2E benchmark 通过。

### R7 · 隐藏验收 + 三集合切齐

**变更清单**：无生产代码改动。
**产出**：
- `docs/baseline/thin-agent-phase-R7-report.md`：Regression / Development / Hidden 三集合结果、通道贡献、GT rank 分布、模型健康
- **最终产品门槛（P0-15）**：Recall@10 ≥ 90%、Recall@20 ≥ 95%、empty-GT FP=0、hard violation=0、Hybrid 不低于任一单通道；Hidden 与 Dev 不发生明显崩溃
- 完整 Hybrid 后仍失败 → 分类为 formation / embedding / parser / fusion / filter / ranking / GT 歧义 → 汇总为 Formation Phase F1 输入（本阶段不动 Formation）

**退出条件（审阅 §7 + 输入报告 §21）**：18 项 DoD 逐条打勾。

---

## 7. Feature Flag 拓扑与回滚（含 P1-3/P1-4）

**新增 flag**（默认全部 off，153 按阶段逐个 on）：

| Flag | 阶段 | 作用 | Off 时行为 |
|---|:-:|---|---|
| `SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1` | R2 | 多通道 recall | 旧单 Kernel 扫描 |
| `SENTRIX_RETRIEVER_METADATA` | R2 | 单通道 | 该通道不召回 |
| `SENTRIX_RETRIEVER_ENTITY` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_LEXICAL` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_VISUAL_ANN` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_TEXT_ANN` | R2 | 同上 | 同上 |
| `SENTRIX_RETRIEVER_ADJACENCY` | R3B | seed 后扩展 | 无邻接扩展 |
| `SENTRIX_RETRIEVER_FUSION` | R3 | `"rrf"` \| `"weighted_norm"` | 默认 rrf |
| `SENTRIX_GATE_PROBE_V1` | R4 | ambiguous → probe | Gate 保持二元 |
| `SENTRIX_MODEL_SPLIT_V1` | R5 | parser/answer/verify 分离 | 全部走同一模型 |
| `SENTRIX_PARSE_MODEL` | R5 | parser 模型名 | 用 `gamma_model` |
| `SENTRIX_ANSWER_MODEL` | R5 | answer 模型名 | 用 `gamma_model` |
| `SENTRIX_VERIFY_MODEL` | R5 | verify 模型名 | 用 `gamma_model` |

**统一 RetrievalConfig（P1-4）**：内部用 `configs/retrieval/defaults.json`（进 git）+ `data/configs/retrieval.local.json`（不进 git，部署覆盖）；env flag 是开关，config 是参数，避免 env 组合不可复现。

**已有 flag 保留**：`SENTRIX_THIN_AGENT_V1`, `SEMANTIC_QUERY_PARSER_V1`, `EVIDENCE_RETRIEVAL_V1`, `LLM_CLAIM_EXTRACTOR_V1`, `CORE_MEMORY_V1`, `MEMORY_CORRECTION_V1`, `ADVANCED_MEMORY_TOOLS_V1`, `ANN_INDEX_V1`, `EXPLICIT_IMAGE_REINSPECTION`。

**回滚协议**：任一 flag off 立即回退旧路径；`MemoryStore` 和 FMA 5173 永不因 Agent flag 停止。

---

## 8. 性能预算与模型调用预算（P0-11/P0-12）

| 路径 | Parser | Retrieval | Answer/Writer | Claim/Verify | E2E 目标 |
|---|:-:|:-:|:-:|:-:|:-:|
| 普通聊天/写作 | 0 | 0 | 1 | 0 | ≤5s |
| 明确图片短查询 | 0-1 | 1 | 0-1 | 0 | ≤8s |
| 简单 evidence | 0-1 | 1 | 0-1 | 0 | ≤12s |
| 复杂人物/比较（正常） | 1 | 1 | 1 | 2 | ≤20s |
| 复杂（Repair） | 1 | 1 | 1 | 2+1 | ≤20s |

- Retrieval-only p95 ≤ 5s/query
- 统一 request deadline = 20s（P0-11，阶段预算见 §6 R5）
- ANN 查询不得退化为十万向量 Python 全扫描（P0-10）
- `none` 普通聊天不得调用家庭 retrieval

---

## 9. 本地与 153 Git 交付流程（P0-20 统一）

**正式历史只存在于 153 `psh`**（与本项目此前约定一致）。本地**不创建正式 commit**。

```text
本地开发
  ├─ 修改文件
  ├─ 本地语法检查 + 单元测试 + 脚本 dry-run（临时工作区，不入本地 git 历史）
  ├─ git diff --check 干净
  ├─ 输出 patch：git format-patch 或 git diff > /tmp/phaseR{r}-{commit}.patch
  └─ scp patch（或 rsync 变更文件清单）到 153
        │
        ▼
153（asus@192.168.0.153）
  ├─ git apply --check / git am --3way 到 psh
  ├─ PYTHONPATH=. .venv/bin/python -m unittest discover backend.tests
  ├─ node --test test/*.test.js（若有前端侧变更）
  ├─ 通过 → 153 git add <明确文件> → git commit（唯一正式提交位置）
  └─ 重启 API 8091 → 健康检查 200×3
```

**备选方案**（若用户后续要本地正式历史）：本地建 commit → git push 或 format-patch → 153 `git am`（cherry-pick 同一 commit 内容），**不得**"本地 commit + 153 再重新 commit"双轨（P0-20 原话）。

**数据传输**：不传 `.env` / `data/`（除显式白名单的脱敏 fixture）/ `logs/` / `.venv/` / 未跟踪试验文件 / 模型权重。

---

## 10. 拟删除 / 拟保留 / 拟替换

| 模块 | 决策 | 说明 |
|---|---|---|
| `evidence_retrieval.py::_contains` 单字/双字 all-match | **删** | 由 LexicalRetriever FTS5 替代；`_contains` 只留完整子串校验 |
| `evidence_retrieval.py::retrieve()` 单 Kernel 扫描 | **替** | flag off 保留旧实现在 shadow 侧 |
| `agent.py` 旧 `MemoryAgent.evidence_answer` 用 `search_vectors` | **保** | 老路径继续存在，flag off 兜底 |
| `store.search_vectors` SQLite 逐行余弦 | **降级** | **不再作为生产 fallback**（P0-10）；仅离线 benchmark/对照/诊断 |
| `answer_composer._allowed_facts` 无去重版本 | **替** | 按 condition_key 去重 |
| Parser 关键词 fallback（若残留） | **删** | 输入报告 §20.5 明令禁止 |
| `configs/retrieval/defaults.json` | **新增** | 进 git，通用默认 + schema version（P1-4） |
| `data/configs/retrieval.local.json` | **新增** | 不进 git，部署覆盖 |
| `configs/probe_thresholds.json` | **新增** | Dev 训、Hidden 验证（P0-8） |

---

## 11. 单元测试与集成测试矩阵

**新增**：
| 文件 | 阶段 | 覆盖 |
|---|:-:|---|
| `backend/tests/test_retriever_contracts.py` | R2 | Retriever Protocol / CandidateHit（含 score direction）/ HardFilterContext |
| `backend/tests/test_retrieval_fusion.py` | R3 | RRF / evidence_class 分级 / 近重复 grouping |
| `backend/tests/test_lexical_retriever.py` | R2-R3 | FTS 预分词 token 断言 / 中文 unigram+bigram / whole+facet / 单汉字不算 matched |
| `backend/tests/test_visual_ann_retriever.py` | R2 | text→embed→ANN top-K；**Manifest 不一致拒绝启用**；scope 回表 |
| `backend/tests/test_text_ann_retriever.py` | R2 | 同上 |
| `backend/tests/test_adjacency_retriever.py` | R3B | 仅 seed 通过 gate 后扩展；不放宽硬条件；all_relevant 组不删 |
| `backend/tests/test_embedding_router.py` | R2 | Visual/Text embedder 解耦；ThinAgent 零依赖 clip |
| `backend/tests/test_gate_probe.py` | R4 | GateDecision / 无长度阈值 / 无 confidence 单点 / probe upgrade |
| `backend/tests/test_model_budget.py` | R5 | 简单 evidence ≤2 生成；普通聊天 =1 answer；deadline 阶段预算 |
| `backend/tests/test_no_benchmark_runtime_dependency.py` | **贯穿** | runtime/configs 不含 benchmark query/GT/Asset ID |
| `backend/tests/test_embedding_quality.py` | R1B | 两套评估器（visual cross-modal / text retrieval）|
| `backend/tests/test_answer_composer_dedup.py` | R6 | 去重 / empty_policy 拒答 / 人类可读模板 |
| `backend/tests/test_ann_manifest.py` | R2 | manifest 校验 / atomic swap / stale hit revision |

**扩展**：`test_evidence_retrieval_benchmark.py`（通道消融维度）、`test_thin_agent_runtime.py`（probe 分支）、`test_query_parser.py`（新 _validate）、`test_memory_gate.py`（GateDecision 拆分）、`test_evidence_bundle.py`（proof source 白名单）。

---

## 12. 输入报告 §18 十八项必答问题对应位置

| # | 问题 | 计划位置 |
|:-:|---|---|
| 1 | Visual vector 模型/checkpoint/维度/中文能力如何验证 | R1B visual cross-modal；Manifest §5.6 |
| 2 | Observation/Event text vector encoder | R1B text retrieval；独立 adapter §5.1 |
| 3 | HNSW ID 如何稳定映射回 scope 下 Asset | §5.6 Manifest + id_map_checksum + scope 回表/独立索引 |
| 4 | Kernel 哪里并行调用 retriever | §5.2 并发策略（DB 串行/独立连接，ANN 并行）|
| 5 | `observation_search_terms` 如何进 lexical | §5.3 FTS 预分词投影 |
| 6 | 用哪种通用 fusion / 为何不过拟合 | §5.4 RRF + evidence_class；Dev 训、Hidden 验证 |
| 7 | Whole text + facets 如何同时进召回 | §5.3 whole+facet 双路；§5.4 融合 |
| 8 | Parser `none` 时什么情况 probe | §5.7 Gate 综合判定 → ambiguous → probe |
| 9 | 明确普通写作 vs 短名词家庭查询 | §5.7 一般任务结构 vs 家庭证据动作 vs 其余→probe |
| 10 | Probe 阈值 Dev/Hidden 分离 | §4 Neutral Probe + §6 R4 |
| 11 | Vector hit 只作 candidate/possible | §6 R3 `_MATCHED_SOURCE_TYPES` 白名单 |
| 12 | Contradicted 证据要求 | §5.2 subject binding 才允许 contradicted |
| 13 | Retrieval-only vs E2E 分离 | §6 R1A evaluate_retrieval_kernel.py |
| 14 | 简单 evidence ≤2 次生成 | §8 预算 + `test_model_budget.py` |
| 15 | 12B 每次 9-90s 缓解 | R5 model_routing + probe_model_health |
| 16 | 独立回滚 flag | §7 flag 拓扑 + RetrievalConfig |
| 17 | Runtime 不含 benchmark 内容 | §3.3 `test_no_benchmark_runtime_dependency.py` |
| 18 | 完整 Hybrid 后仍失败归 Formation | §6 R7 report 分类 |

---

## 13. 零容忍门槛（输入报告 §16 + 审阅 §7）

| # | 门槛 | 检测 |
|:-:|---|---|
| 1 | Benchmark query/file/Asset ID 出现在 runtime/configs | `test_no_benchmark_runtime_dependency.py` |
| 2 | Case-specific 同义词或 boost | code review + grep |
| 3 | 明确普通写作触发家庭检索 | `test_gate_probe.py` |
| 4 | 家庭短语因 parser `none` 永久失去检索 | `test_gate_probe.py` 短名词 |
| 5 | 向量高分直接升级 confirmed fact | `_MATCHED_SOURCE_TYPES` 白名单 |
| 6 | 单字 contains 作为 matched 支持 | `test_lexical_retriever.py` |
| 7 | 多值字段未命中直接 contradicted | `test_evidence_bundle.py` |
| 8 | 空 GT 进入 normal chat 编造具体场景 | `test_answer_composer_dedup.py` + empty_policy |
| 9 | 已建 ANN 未被生产 Kernel 调用却宣称接入 | trace 必须有 `visual_ann.invoked=true` 的实际请求 |
| 10 | 普通查询自动原图重读 | `EXPLICIT_IMAGE_REINSPECTION=0` 时 image_results 空 |
| 11 | 用户可见完全无关 Asset | Regression FP 门槛 |
| 12 | 硬时间/人物/scope/media/must_not 违反 | 现有 constraint 校验 |
| 13 | ANN 不一致索引被静默查询 | `test_ann_manifest.py`，不一致记 `index_incompatible` |
| 14 | 生产请求自动全表向量扫描 | P0-10，grep `search_vectors` 在生产路径 |
| 15 | 结果等级意外新增 `possible` | `test_evidence_bundle.py` + API contract 测试 |

---

## 14. Commit 拆分建议（每阶段唯一正式 commit 在 153）

**R0**：`docs: phase R0 call chain + benchmark case audit`
**R1A**：`feat(benchmarks): retrieval-only kernel evaluator + channel ablation`、`feat(benchmarks): split hidden set`、`test: no benchmark runtime dependency guard`
**R1B**：`feat(benchmarks): visual cross-modal + text retrieval embedding eval`、`docs: embedding capability decision`
**R2**：`feat(retrieval): retriever protocol + embeddings router`、`feat(retrieval): metadata/entity/lexical retrievers`、`feat(retrieval): visual/text ann retrievers with manifest`、`feat(evidence): multi-retriever recall + shadow diff`、`feat(app): composition root wiring`
**R3**：`refactor(evidence): drop single-char contains`、`feat(retrieval): rrf + evidence_class fusion`、`feat(evidence): matched source whitelist`
**R3B**：`feat(retrieval): seed-based adjacency + duplicate grouping`
**R4**：`feat(gate): GateDecision + neutral probe`、`feat(parser): tighter validator without length heuristics`
**R5**：`feat(model): model_routing + unified deadline`、`feat(scripts): model health prober`
**R6**：`refactor(answer): dedup + empty_policy + human-readable templates`
**R7**：`docs: phase R7 hidden acceptance report + formation input`

---

## 15. 用户确认结果（2026-08-06 已全部答复）

**§0 D1-D8 全部确认**：
1. **Hidden 规模 15-20 可接受**（D2）→ 从 60 case 划 15-20，Regression 剩 40-45。
2. **不预留 scope 增长**（D7）→ ANN 主方案"按 scope 独立索引"，不实现全局+oversampling。
3. **Parser 小模型用 153 现有 2B**（D6）→ 不再选 qwen2.5/gemma3；直接接线 153 `gemma4:e2b-it` 2B + 已写切换逻辑。
4. **R1B 切换流程确认**（D8）→ 评估报告 → 用户点头 → 换 adapter + 索引重建。

**当前无未决问题**。计划冻结，进入实施。

---

## 16. 阶段完成声明模板（对齐审阅 §7 + 输入报告 §21）

```markdown
# Phase R{X} 完成声明

## 修改文件清单 + git diff --stat
## 新增/扩展测试（数量 + 结果）
## Benchmark 对比
  - Regression Set: Recall@10/@20/MRR/empty-GT FP/hard violation
  - Development Set: 同上
  - Channel ablation: 各通道贡献
## 已布线的 flag（+ RetrievalConfig 状态）
## 153 上重启后健康检查（8091/4174/5173 均 200）
## 未完成项 / 真实阻塞 / 下一阶段依赖
## 审阅报告 §7 验收标准逐条打勾
```

---

## 17. 本计划的 out-of-scope

本阶段**不做**（输入报告 §3 + §20 + 审阅 §4）：
- Core Memory 完整上线宣告、Correction 端到端 UI 联调、主动回忆、多 viewer 完整产品化
- Answer Writer 风格进一步优化
- Formation pipeline 大规模改造（R7 输出 Formation 输入报告，改造留给 Phase F1）
- 修改图片记忆生成语义 / `raw_json` / FMA 5173 / e2b_server 相关 uncommitted 改动
- 视频解码 / 镜头切分 / 视频向量

**尤其禁止**：Retrieval 基础未通过时扩展主动回忆和复杂人物画像；用 375 合同测试全绿证明 Agent 可用；用空 GT"凑巧没返回图"当正确行为；为修模型不稳再引入长度/前缀/固定阈值分类器（审阅 §9.3）；用文本相似度实验证明 CLIP 跨模态能力（P0-2）。

---

## 附录 A · 与审阅报告 P0/P1 的对照表

| P0 | 主题 | 落实位置 |
|:-:|---|---|
| P0-1 | FTS 中文 token 重写（预分词投影） | §5.3 |
| P0-2 | Visual/Text embedding 评估分离 | §6 R1B |
| P0-3 | 删除 ThinAgent 对 clip 直接依赖 | §5.1 EmbeddingRouter |
| P0-4 | ANN Manifest / revision / atomic swap / stale 校验 | §5.6 |
| P0-5 | ANN scope/硬过滤 + oversampling 策略 | §5.6 |
| P0-6 | 删除长度>6 repair 与 confidence 单点 | §5.7 |
| P0-7 | Probe 用 raw text neutral query | §4 |
| P0-8 | Probe 强候选多信号标准 | §4 |
| P0-9 | Adjacency 改 seed 后扩展器 | §5.5 / R3B |
| P0-10 | 禁止生产全表向量扫描 fallback | §5.6 / §10 |
| P0-11 | 统一 request deadline | §6 R5 / §8 |
| P0-12 | 修正调用预算表 | §6 R5 / §8 |
| P0-13 | 近重复 grouping 不删 all_relevant | §5.8 / R3B |
| P0-14 | 不新增 result level `possible` | §0.1 / §5.2 |
| P0-15 | R3 工程门槛与 R7 最终指标分离 | §6 R3 / R7 |
| P0-16 | 空 GT strict/approximate policy | §3.2 / §5.8 |
| P0-17 | case answerability + GT 一致性审计 | §6 R0 / §3.2 |
| P0-18 | 并发 SQLite 连接策略 | §5.2 |
| P0-19 | CandidateHit score direction 标准化 | §5.2 |
| P0-20 | 本地/153 Git 流程统一 | §9 |

| P1 | 主题 | 落实位置 |
|:-:|---|---|
| P1-1 | Fusion evidence_class 分级非平权 | §5.4 |
| P1-2 | 召回来源 vs 证明来源分离 | §5.3 / §6 R3 |
| P1-3 | Index readiness 健康状态 | §5.6 |
| P1-4 | 统一 RetrievalConfig | §5.4 / §7 |
| P1-5 | Probe 与正式检索共享 retriever 不同策略 | §4 |
| P1-6 | Parser few-shot 用合成结构例子 | §5.7 |
| P1-7 | 普通聊天性能以实际 Answer Model 为准 | §8 |
| P1-8 | inspect_retrieval_case 支持 query-only | §6 R0 |
