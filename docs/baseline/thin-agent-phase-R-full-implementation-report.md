# Sentrix Thin Agent 层完整实现报告（Phase R 总结）

**日期**：2026-08-06
**范围**：从用户问题到回答的完整 Agent 流程（数据流 / 提示词 / 优化方法 / 代码实现细节）、当前测试结果、需用户决断项、上一次计划（Phase R）完成情况。
**数据**：全部来自已完成实测（153 真实 DB + 本地测试），非推断。

---

# 1. 完整流程：用户问题 → 回答

## 1.1 入口与总览

```text
用户消息
  │
  ▼
POST /api/assistant/turn  (app.py:680)
  │  MemoryAgent.answer_turn (agent.py:1837)
  │    如果 SENTRIX_THIN_AGENT_V1=1 且无 feedback/selected_entity → ThinAgentRuntime.answer_turn
  ▼
ThinAgentRuntime.answer_turn (thin_agent.py)
  ├─ 1. gate.fast_path      → 写作/翻译/"不用查" → none；feedback/selected_entity → evidence（不调模型）
  ├─ 2. parser.parse       → QueryParseDraft（e2b 2B 模型，1 次 JSON 调用，可 repair 1 次）
  ├─ 3. gate.classify      → GateDecision（mode: none/contextual/evidence/ambiguous）
  │        none        → _normal_chat（0 检索）
  │        contextual  → _contextual（core memory 卡片，0 具体照片）
  │        ambiguous   → _ambiguous_path（有 facets→直检；无 facets→NeutralProbe→升级/澄清）
  │        evidence    → build_query_spec → kernel.retrieve → _evidence_answer
  ▼
EvidenceRetrievalKernel.retrieve
  ├─ HardFilterContext.from_spec（scope/viewer/media/time/must_not）
  ├─ RetrievalQuery.from_spec（whole_query + facets）
  ├─ 多路召回：metadata / entity / lexical(FTS) / visual_ann(Chinese-CLIP) / text_ann
  ├─ fuse（加权 RRF，visual 权重 2.5）
  ├─ _evaluate_fused：逐候选逐条件评估（_condition，matched 白名单）
  ├─ seed-quality gate → adjacency 扩展（同 event/时间窗/batch）
  └─ EvidencePacket（assets/exact/strong/approximate/gaps/channel_trace）
  ▼
_evidence_answer
  ├─ evidence 列表 + near_duplicate 分组注解
  ├─ person summary（Writer 链）或 _simple_answer
  ├─ _allowed_facts（按 condition_key 去重 + 人类可读）
  └─ _envelope（含 retrieval_trace / claims / evidence_presentation）
  ▼
JSON 响应
```

## 1.2 数据流（关键对象）

```text
用户消息(str)
  → QueryParser.parse() → QueryParseDraft
      mode / actions / facets / semantic_conditions / negative_conditions
      entity_names / time_expression / media_expressions / ambiguities / confidence
  → MemoryGate.classify() → GateDecision
      mode / allow_probe / query_parse_calls / original_image_allowed ...
  → build_query_spec(draft, scope, viewer, entity_resolver) → QuerySpec
      constraints: [Constraint(dimension, value, strictness, proof_policy, negated, source_text)]
  → EvidenceRetrievalKernel.retrieve(spec) → EvidencePacket
      assets[{asset_id, file_name, level, condition_results, attributions, fusion_score, ...}]
      exact_results / strong_results / approximate_results / gaps / channel_trace
  → _evidence_answer → 响应 dict
      answer / evidence[] / image_results / retrieval_trace / claims / claim_evidence_index /
      statement_plan / memory_used / evidence_required / parser_mode / parser_confidence
```

## 1.3 三个决定性开关（生产环境默认值）

| 环境变量 | 值 | 作用 |
|---|---|---|
| `SENTRIX_THIN_AGENT_V1` | 1 | 走 Thin Agent（否则旧 MemoryAgent） |
| `SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1` | 1 | kernel 走多路召回（否则单 Kernel 扫描） |
| `SENTRIX_PARSE_BACKEND=e2b` + `SENTRIX_PARSE_BASE_URL=http://127.0.0.1:8100` + `SENTRIX_PARSE_MODEL=gemma-4-e2b-it+lora-v2` | — | parser 用 153 e2b 2B（D6） |
| `SENTRIX_IMAGE_EMBEDDER=chinese_clip` | — | 视觉槽用 Chinese-CLIP（D3） |
| `SENTRIX_TEXT_EMBEDDER=clip` | — | 文本槽用 CLIP |
| `CLIP_DEVICE=cpu` | — | 153 GPU driver mismatch，CLIP 走 CPU |
| `SENTRIX_MODEL_SPLIT_V1` | 1 | 启用 parse/answer/verify 模型分离 |

---

# 2. 代码模块与职责（实现细节）

## 2.1 入口与编排
| 文件 | 关键点 |
|---|---|
| `app.py` | Composition Root：`MemoryAgent(store, gamma, clip=pipeline.clip)`；`/api/assistant/turn` 端点 |
| `agent.py` | `MemoryAgent.__init__` 从 clip 自动建 `EmbeddingRouter.from_clip(clip)`；`answer_turn` 切 thin_runtime |
| `thin_agent.py` | `ThinAgentRuntime`：gate/parser/kernel/complex_builder/router；`answer_turn` 主流程；`_ambiguous_path`/`_normal_chat`/`_contextual`/`_evidence_answer`/`_allowed_facts`/`_human_condition_text` |

## 2.2 语义层
| 文件 | 关键点 |
|---|---|
| `query_parser.py` | `QueryParser.parse`：`_call_parser` → `_draft_and_validate` → 一次 repair → `_safe_fallback`；`_apply_deterministic_overlay`（时间/否定从原文确定性恢复）；validator：`mode=evidence 但 actions=[] → repair` |
| `query_contracts.py` | `Constraint`（dimension/value/strictness/proof_policy/negated/source_text）、`QueryAction`、`QueryFacet`、`QueryParseDraft`、`QuerySpec`、`sanitize_query_parse`（丢弃模型越权字段）、`build_query_spec`（三类 strictness：deterministic_hard/semantic_required/ranking_preference） |
| `memory_gate.py` | `MemoryGate.fast_path`（写作/不用查/feedback/selected_entity）→ `classify`（parser none + 无 general 结构 → ambiguous + probe；`_has_household_signal` facets/conditions → ambiguous；`_explicit_general_task` → none） |
| `retrieval/probes.py` | `NeutralProbe.run`：多信号（≥2 通道有候选 / lexical exact / per-space 校准）→ upgrade/clarify/none |

## 2.3 检索内核
| 文件 | 关键点 |
|---|---|
| `evidence_retrieval.py` | `EvidenceRetrievalKernel.retrieve` 按 flag 分派 `_retrieve_multi` / `_retrieve_single`；`_retrieve_multi`：HardFilterContext→RetrievalQuery→通道召回→fuse→`_evaluate_fused`→seed→adjacency→packet；`_condition` 按 dimension 分派（time/media/person/open_world/single_value/activity/semantic_pool）；`_MATCHED_SOURCE_TYPES` 白名单（matched 只来自 asset_metadata/observation_field_exact/confirmed_bridge/subject_binding，其余降级 possible）；`probe()` 供 R4 |
| `retrieval/base.py` | `CandidateHit`（raw_score/score_kind/higher_is_better/rank/calibrated_score）、`HardFilterContext`、`RetrievalQuery`、`Retriever` Protocol |
| `retrieval/metadata.py` | 时间/媒体/scope 结构化召回；无正向结构化条件时返回空（避免污染纯语义查询） |
| `retrieval/entity.py` | confirmed entity → person_bridge（observation_search_terms）→ 候选 asset |
| `retrieval/lexical.py` | FTS 预分词召回；FTS 空时惰性自愈重建 |
| `retrieval/visual_ann.py` | text→Chinese-CLIP→visual ANN→asset；Manifest 校验 + 搜索时维度交叉校验 |
| `retrieval/text_ann.py` | text→CLIP→semantic/episodic ANN→observation→asset |
| `retrieval/adjacency.py` | seed 后扩展：同 event（event_observations join）/时间窗(±120min)/batch（source_album/device）；预算；重过滤 |
| `retrieval/fusion.py` | 加权 RRF（默认 visual_ann=2.5, lexical=1.0, text_ann=0.5, metadata/entity=1.0, adjacency=0.5）+ evidence_class（anchor boost +1.0） |
| `retrieval/near_duplicate.py` | SHA-256（content_sha256）分组，只注解不删结果 |
| `retrieval/config.py` | `RetrievalConfig`：双层配置（configs/retrieval/defaults.json 进 git + data/configs/retrieval.local.json 部署覆盖）；`channel_enabled` 短名映射（visual→visual_ann, text→text_ann） |

## 2.4 Embedding
| 文件 | 关键点 |
|---|---|
| `embeddings/base.py` | `VisualQueryEmbedder` / `TextQueryEmbedder` Protocol（model_id/dimension/embed_query） |
| `embeddings/chinese_clip_visual.py` | `ChineseClipVisualEmbedder`（cn_clip `load_from_name("ViT-L-14")`，768-dim，`embed_query`/`embed_image`）；从 `~/.cache/clip/clip_cn_vit-l-14.pt` 加载 |
| `embeddings/clip_visual.py` / `clip_text.py` | 包装 `ClipAdapter.embed_text`（512-dim） |
| `embeddings/bge_text.py` | bge-m3 备用（未启用） |
| `embeddings/router.py` | `EmbeddingRouter.from_clip` 按 `SENTRIX_IMAGE_EMBEDDER`/`SENTRIX_TEXT_EMBEDDER` 选槽 |

## 2.5 模型路由（R5）
| 文件 | 关键点 |
|---|---|
| `model_routing.py` | `resolve_specs`（env 解析 parser/answer/verify 模型）、`RequestDeadline`（统一 20s：parser 4s/retrieval 5s/answer 7s/overhead 2s）、`CircuitBreaker`（按 role，threshold 3，60s 半开）、`ModelRouter.chat(role, prompt, fallback)` |
| `model_clients.py` | `GammaClient`：`chat(prompt, role=...)` 用 `_endpoint_for(role)` 选 model/base_url；`SENTRIX_MODEL_SPLIT_V1=1` 才启用角色分离；`parse_backend=e2b` 时 parser 指向 8100；`parse_json_response` 剥离 ```json``` 代码块 |

## 2.6 回答层（R6）
| 文件 | 关键点 |
|---|---|
| `answer_composer.py` | `validate_statement_plan` / `compose_answer`（statement×evidence 校验） |
| `complex_answer.py` | Writer 链（`_call_writer` role=answer）+ 确定性 fallback |
| `claim_extractor.py` | `LLMClaimExtractor`（role=verify） |
| `thin_agent._allowed_facts` | 按 condition_key 去重，evidence_ids 并集 |
| `thin_agent._human_condition_text` | matched→"记录中有「X」"、possible→"记录中可能有「X」，但无法完全确认"、unknown→"目前无法确认…"；禁止泄漏内部 condition_key/分数/trace/表名 |
| `thin_agent` 空 EvidencePacket | 强制"当前记忆中没有找到足够匹配的原始证据。"（不进 normal chat） |

---

# 3. 提示词全集

## 3.1 QueryParser 主提示词（`query_parser.py::_PARSER_PROMPT`）
```
你是 Sentrix 的查询解析器，不负责回答用户问题，也不能读取数据库。
你的任务是把用户消息和最近对话转换为严格 JSON QueryParseDraft，不输出运行时身份和数据库 ID。

规则：
1. 普通聊天、写作、建议、情绪支持返回 mode=none，不能要求家庭证据。
2. 具体人物、时间、地点、照片、衣着、活动、关系、原图、比较和时间线属于家庭记忆请求，mode=evidence。
3. 自然人物提及但没有问历史事实的可以是 mode=contextual。
4. 日期、明确人物、媒体类型和"不要/不是/排除"是候选条件；由后端确定性代码决定其是否属于 deterministic_hard。
5. 做饭、晚饭、自拍、颜色、材质等视觉或语义描述属于 semantic_conditions。
6. "都、所有、全部、还有哪些"使用 result_requirement.mode=all_relevant。
7. "介绍一下某人"是 person 目标。
8. 不能创建实体 ID，不能猜测人物身份，不能调用工具，不能补充家庭事实。
9. 一句话可以包含多个 action（例如 answer_question + return_assets），不要压缩成单一目标。
10. facets 保留用户提到的所有维度，surface_text 用原文片段。
11. 只输出 JSON，不要输出 scope_id/scope_mode/viewer_id/conversation_id/entity_ids，不要 Markdown。

当前时间：{{now}}
最近对话：{{conversation}}
用户消息：{{message}}

输出 schema：
{{query_parse_draft_json_schema}}
```

## 3.2 Repair 提示词（`_REPAIR_PROMPT`）
```
你是 Sentrix QuerySpec 修复器。
只修复 JSON 结构、枚举值和字段类型，不改变用户原文已经明确表达的硬条件。
不得添加人物、日期、地点、媒体或证据。
如果无法确定，将字段置空或放入 semantic_conditions。

用户原文：{{message}}
模型原始 JSON：{{raw_json}}
代码发现的问题：{{validation_errors}}
请只输出修复后的 QueryParseDraft JSON。
```

## 3.3 普通聊天提示词（`thin_agent._normal_chat`）
```
你是 Sentrix，一个自然、克制的家庭数字助手。本轮不是家庭记忆查询，
不要读取或猜测具体家庭事实。直接自然回答用户，不要提到数据库、检索或工具。
最近对话：{recent_turns[-1200:]}
用户：{message}
```

## 3.4 Writer 提示词（`complex_answer.py::_WRITER_PROMPT`，功能描述）
接收 `{{message}}`（用户原话）+ `{{context_packet}}`（`narrative_context.build_narrative_context_packet` 构造的受控证据上下文），`json_mode=True` 输出 `{text, candidate_claims[], unknowns[]}`。role=answer。

## 3.5 ClaimExtractor 提示词（`claim_extractor.py::_LLM_CLAIM_PROMPT`，功能描述）
接收 `{{answer_text}}` + `{{writer_candidates}}`（候选 claim + 候选 evidence_ids），输出 `{claims:[{text, ...}]}`。role=verify。

**提示词优化方法**：
- parser 走 e2b 2B（更快更稳），修复走 repair（不改变硬条件）
- 确定性 overlay：时间正则 + "不要/排除/不是" 从原文恢复（不依赖模型）
- few-shot 用合成结构例子，禁止真实 benchmark 题目（guard 测试守护）
- `sanitize_query_parse` 丢弃模型 scope/viewer/entity_id 越权字段

---

# 4. 优化方法汇总

| # | 方法 | 解决的问题 | 实测效果 |
|:-:|---|---|---|
| 1 | **e2b 2B parser（D6）** | 12B gemma4:12b partial VRAM → 9-90s 不稳 | 延迟 2.9-4.0s，facets 正确抽取 |
| 2 | **Chinese-CLIP 视觉（D3）** | ViT-B-32 中文图文对齐随机（AUC 0.51） | AUC 0.982，视觉 Recall@10 0.887 |
| 3 | **多路召回** | 单 Kernel 扫描全 scope | 通道消融可证各通道贡献 |
| 4 | **FTS 预分词（P0-1）** | 单字 contains 造成 album1-01 10 FP | 单 CJK 字不再 token，整词+bigram 召回 |
| 5 | **`_contains` 全子串（P0-6）** | tokenized all-match 过宽 | 杜绝单字/双字乱配 |
| 6 | **加权 RRF（visual 2.5）** | flat RRF 被弱通道稀释（hybrid 0.819<visual 0.887） | hybrid 0.867 |
| 7 | **Gate+Probe（R4）** | parser none 永久失去检索 | none→ambiguous→probe/直检；"厨房里做晚饭" 6/6 GT |
| 8 | **Ambiguous 有 facets 直检** | 探针过保守 | 有 facets 直接检索 |
| 9 | **matched 白名单（P1-2）** | 向量/FTS 命中不可直接证明事实 | vector 只 candidate/possible |
| 10 | **Answer 去重 + 人类可读（R6）** | "记录支持X"重复 10 次 | 按 condition_key 去重 |
| 11 | **role-aware 模型 + deadline + breaker（R5）** | 无统一超时、无降级 | RequestDeadline 20s + CircuitBreaker |
| 12 | **Manifest 校验（P0-4）** | 查询 embedder 与索引不匹配静默查询 | 不一致拒绝启用，trace 记录 index_incompatible |
| 13 | **empty 强制拒答** | 空 GT 走 normal chat 编造 | 家庭查询空结果 → 明确拒答 |
| 14 | **seed-gated adjacency（P0-9）** | 邻接扩散错误候选 | 仅 exact/strong seed 扩展 |

---

# 5. 当前测试结果（全部实测）

## 5.1 单元/集成测试
| 位置 | 结果 |
|---|---|
| 本地工作副本 | **434 pass / 1 skip**（基线 341 → +93） |
| 153 生产 | **430 pass / 1 skip** |
| Benchmark 隔离守护 | 3 pass（runtime/configs 无 benchmark 数据） |

## 5.2 R1B Embedding 评估（153 真实权重）
| 评估器 | ViT-B-32 | Chinese-CLIP (ViT-L/14) |
|---|:-:|:-:|
| Text 自检索 recall@1/@10 / AUC | 0.67 / 0.92 / 0.996 | — |
| Visual 跨模态 recall@1/@10 / AUC | **0.01 / 0.04 / 0.51（随机）** | **0.43 / 0.845 / 0.982** |

## 5.3 R7 检索消融（153 真实 DB，44 regression，16 hidden 排除）
| 通道 | Recall@10 | Recall@20 | MRR | all_relevant | empty_fp | hard_viol |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **visual（Chinese-CLIP）** | **0.887** | 0.895 | **0.710** | 32/44 | 6 | 0 |
| lexical | 0.373 | 0.373 | 0.432 | 12/44 | 5 | 0 |
| text（CLIP） | 0.158 | 0.195 | 0.119 | 3/44 | 6 | 0 |
| structured | 0.074 | 0.126 | 0.056 | 3/44 | 0 | 0 |
| hybrid_no_adjacency | 0.787 | 0.928 | 0.581 | 33/44 | 6 | 0 |
| **full_hybrid（加权）** | **0.867** | 0.898 | 0.643 | 31/44 | 6 | 0 |

## 5.4 端到端 API（8091，e2b 2B parser + Chinese-CLIP）
| 查询 | 结果 |
|---|---|
| "厨房里做晚饭"（GT 6） | **6/6 GT 全部在 evidence 前 6** |
| "帮我写一段生日祝福" | mode=none，0 检索 |
| "水族馆海豚跃出水面"（空 GT） | 返回 10 近似（未严格拒答 → empty_fp 项） |
| parser 延迟 | 2.9-4.0s |

## 5.5 基础设施
| 项 | 结果 |
|---|---|
| 数据完整性 | 378 assets / 378 observations，integrity ok |
| CLIP 权重 | ViT-B-32 转换权重 conv1 diff=0.0；Chinese-CLIP 768-dim 正常 |
| ANN 索引 | visual 378（Chinese-CLIP）/ semantic 374 / episodic |
| 服务健康 | 8091 / 4174 / 5173 全 200 |

---

# 6. 需要你做的决断（8 项）

## A. 确认（我已按你的 D3/D6 授权实现）
| # | 决断 | 现状 |
|:-:|---|---|
| 1 | **视觉 embedding 用 Chinese-CLIP** | 已实现并切换（AUC 0.51→0.982，视觉 Recall@10 0.887） |
| 2 | **Parser 用 153 e2b 2B** | 已接线（延迟 2.9-4.0s） |
| 3 | **融合用加权 RRF（visual 权重 2.5）** | 已实现（hybrid 0.867） |
| 4 | **文本槽仍用 CLIP（不启用 bge-m3）** | 文本自检索 AUC 0.996 够用 |

## B. 需要你定方向
| # | 决断 | 影响 |
|:-:|---|---|
| 5 | **60 case 是否标注 `empty_policy`**（strict_empty / allow_approximate） | 决定 empty_fp=6 能否清零 |
| 6 | **是否构建独立 Dev 集**校准融合权重/阈值 | Recall@10 0.867→90% 的最后 1-3pt + Hybrid≥单通道 |
| 7 | **Hidden Acceptance 16 case 由你持 GT 独立评分** | 判断是否过拟合 |
| 8 | **153 GPU driver mismatch（NVML 595.84）是否修复** | 端到端 ≤20s 预算能否达标（12B answer + CPU CLIP） |

---

# 7. 上一次计划（Phase R）完成情况 + DoD

## 7.1 阶段完成情况
| 阶段 | 交付 | 状态 |
|---|:---|:-:|
| R0 | 调用链核验 + inspect/audit 脚本（检出 2 GT 不一致） | ✅ |
| R1A | Retrieval-only runner + 消融 + hidden split + guard | ✅ |
| R1B | Visual/Text 独立评估 → **visual 不合格 → 切 Chinese-CLIP** | ✅ |
| R2 | 多路检索接入 kernel + Manifest + FTS + 接线 | ✅ |
| R3 | `_contains` 全子串 + matched 白名单 + RRF | ✅ |
| R3B | seed-gated adjacency + near-duplicate | ✅ |
| R4 | GateDecision + NeutralProbe | ✅ |
| R5 | role-aware 模型 + deadline + breaker + e2b parser | ✅ |
| R6 | Answer 去重 + 空拒答 + 人类可读 | ✅ |
| R7 | 三集合实测 + R7 报告 + F1 输入 | ✅（阈值部分未达，见下） |

## 7.2 DoD（输入报告 §21，18 项）
| 状态 | 项 |
|:-:|---|
| ✅ 14 项 | 1-8、10、11、13-17 |
| ⚠️ 4 项 | 9（视觉已切但需确认）、12（端到端延迟/GPU）、15（Dev 集未独立）、18（失败分类部分） |

## 7.3 量化门槛对照
| 门槛 | 目标 | 当前 | 差距 |
|---|---|:---:|:---:|
| Recall@10 | ≥90% | 0.867（hybrid）/0.887（visual） | 1-3pt |
| Recall@20 | ≥95% | 0.898 | 5pt |
| empty GT FP | 0 | 6 | 需 empty_policy |
| hard violation | 0 | **0** ✅ | — |
| Hybrid ≥ 单通道 | — | 0.867 vs 0.887 | 2pt（Dev 校准） |

## 7.4 计划冻结项（正确未做）
Core Memory 上线宣告 / Correction 端到端 UI / 主动回忆 / 多 viewer / Answer Writer 风格优化 / Formation 大规模改造（F1 留给下一阶段）。

## 7.5 附带事故与恢复（2026-08-06）
交付 rsync 误清 153 工作树 → 已全部恢复：DB（fd 复制）、媒体（repo 外源图）、源码（本地重建）、git（bundle 重建 136 提交）、.venv、ANN 索引、CLIP 权重、检索投影、服务。

---

## 附：关键文件清单
```
backend/
  agent.py  app.py  thin_agent.py  query_parser.py  query_contracts.py
  memory_gate.py  evidence_retrieval.py  retrieval_ann.py  retrieval_indexes.py
  model_clients.py  model_routing.py  claim_extractor.py  complex_answer.py  answer_composer.py
  embeddings/{base,router,clip_visual,clip_text,chinese_clip_visual,bge_text}.py
  retrieval/{base,metadata,entity,lexical,visual_ann,text_ann,adjacency,fusion,near_duplicate,probes,config}.py
  tests/ (434 tests)
scripts/
  benchmarks/{evaluate_retrieval_kernel,evaluate_parser_retrieval,evaluate_embedding_quality,
              build_embedding_eval_input,inspect_retrieval_case,audit_benchmark_cases,
              split_hidden_set,fixture}.py
  maintenance/{rebuild_retrieval_indexes,rebuild_ann_indices,build_core_memory,probe_model_health}.py
  runtime/start_sentrix_api.sh
configs/retrieval/defaults.json
docs/baseline/thin-agent-phase-R{0,R1A,R1B,R2,R3-R3B,R4-R5,R6,R7-report}.md
```
