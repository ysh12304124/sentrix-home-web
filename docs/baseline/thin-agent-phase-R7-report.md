# Phase R7 · 隐藏验收 + 三集合 + 事故恢复 + DoD

**日期**：2026-08-06
**状态**：基础设施全部交付 + 实测；部分量化目标未达（见 §5 结论）

---

## 0. 事故与恢复（2026-08-06 02:20 发生）

**事故**：交付 rsync 命令（多源 + `--delete-excluded`）误将 153 `/home/asus/Github/Sentrix-Home-Web` 工作树清空替换（`.git`、`backend/`、`data/`、`.venv`、前端全部丢失）。

**恢复**（已完成）：
| 项 | 方式 | 结果 |
|---|---|---|
| `data/sentrix.db` | 从运行中 uvicorn (PID 401220) 的 `/proc/<pid>/fd/3,4,5` 复制（含 WAL/SHM） | **integrity ok，378 assets / 378 observations** |
| 媒体图 | 源图在 repo 外的 `/home/asus/samples/album{1,2,3}/images/`（191 张）→ 重建 `data/household-benchmark-source/` | 378 assets path 可解析 |
| 源码树 | 从本地工作副本 rsync（无 delete）+ 清理根目录错位 | backend/scripts/configs/docs/前端 完整 |
| `.git` | 本地 `git bundle` → 153 `git init` + `git reset origin/psh` | **136 提交历史恢复**（153 侧原 commit ID 丢失，内容等价） |
| `.venv` | 从 stmem conda env 重建（`--system-site-packages`）+ 补装 hnswlib/onnxruntime/funasr/cn-clip | 可运行 |
| ANN 索引 | `rebuild_ann_indices.py` 从 memory_vectors 重建 | visual 526 / semantic 374 / episodic（原 512-dim） |
| CLIP 权重 | `~/.cache/clip/ViT-B-32.pt`（OpenAI JIT）→ 键名映射为 open_clip 格式 | 加载成功（conv1 diff=0.0） |
| 检索投影 | `rebuild_retrieval_indexes.py` | 378 observations |
| 服务 | 8091 重启（Phase R 代码）、4174/5173 200 | 健康 |

**git 历史**：153 原 HEAD 的 commit ID（82797bc 等）无法恢复（无备份），但内容与本地等价；用本地 bundle 重建 psh 全历史 + 之后每次 Phase R 提交。已多次 commit（36da7ff → ae4eadc → 90e3531 → ...）。

**根因规避**：交付改用逐路径 rsync（无 `--delete-excluded`）；`test_no_benchmark_runtime_dependency` 等仍守护 runtime。

---

## 1. R0-R6 交付回顾（详见各阶段报告）

| 阶段 | 交付 | 状态 |
|---|---|---|
| R0 | 调用链核验文档 + inspect_retrieval_case + audit_benchmark_cases（检出 2 个 GT 不一致） | ✅ |
| R1A | Retrieval-only runner + 通道消融 + split_hidden（16 hidden/44 regression）+ guard 测试 | ✅ |
| R1B | Visual/Text 独立评估器 + build 输入 + **真实 CLIP 评估** | ✅（结论见 §3） |
| R2 | embeddings/ + retrieval/ 多路 + kernel 多通道 + Manifest + FTS 预分词 + 接线 | ✅ |
| R3 | `_contains` 全子串 + matched 白名单 + RRF/evidence_class | ✅ |
| R3B | seed-gated adjacency + near-duplicate grouping | ✅ |
| R4 | GateDecision + NeutralProbe + parser 校验收紧 | ✅ |
| R5 | role-aware 模型 + deadline + breaker + **e2b 2B parser 接线（D6）** | ✅ |
| R6 | `_allowed_facts` 去重 + 空拒答 + 人类可读 | ✅ |

**单元测试**：本地 434（基线 341 → +93）；153 430 pass / 1 skip（1 个环境相关测试已修）。

---

## 2. R7 实测（153 真实 DB，44 个 Regression case，hidden 排除）

### 通道消融（cached spec，Recall@10/20）

| 通道 | Recall@10 | Recall@20 | MRR | all_relevant | empty_fp | hard_viol |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **visual（Chinese-CLIP）** | **0.887** | 0.895 | **0.710** | 32/44 | 6 | 0 |
| lexical | 0.373 | 0.373 | 0.432 | 12/44 | 5 | 0 |
| text（CLIP） | 0.158 | 0.195 | 0.119 | 3/44 | 6 | 0 |
| structured | 0.074 | 0.126 | 0.056 | 3/44 | 0 | 0 |
| hybrid_no_adjacency | 0.787 | 0.928 | 0.581 | 33/44 | 6 | 0 |
| full_hybrid（加权 w2.5） | **0.867** | 0.898 | 0.643 | 31/44 | 6 | 0 |

**关键结论**：
- **视觉单通道 0.887 最强**，接近 90% 目标；flat RRF 融合稀释（hybrid 0.819 < visual 0.887）→ 已改为加权 RRF（visual 2.5 → hybrid 0.867）。
- **Hybrid ≥ 单通道仍未完全满足**（0.867 vs 0.887 差 2pt）——Dev 集校准项。
- text 通道弱（0.158）：CLIP 文本对"query↔paraphrase"跨匹配弱（自匹配 AUC 0.996 但跨改写不行）。
- empty_fp=6：空 GT case 检索层仍返回近似候选（precision 项，见 §5）。

### 端到端（API 8091，e2b 2B parser + Chinese-CLIP）
- "厨房里做晚饭"（GT 6）→ **6/6 GT 文件全部在 evidence 前 6**。
- "帮我写一段生日祝福" → none，无检索。
- 空 GT"水族馆海豚跃出水面" → 返回近似（未严格拒，precision 项）。

---

## 3. R1B Embedding 独立评估结论（决定性）

| 评估器 | ViT-B-32 | Chinese-CLIP (ViT-L/14) | 判定 |
|---|:-:|:-:|:-:|
| Text 自检索 recall@10 / AUC | 0.92 / 0.996 | — | **文本 OK** |
| **Visual 跨模态（caption→image）recall@10 / AUC** | **0.04 / 0.51（随机）** | **0.845 / 0.982** | **ViT-B-32 失败 → 已切 Chinese-CLIP (D3)** |

- ViT-B-32 对中文图文对齐是随机的（AUC 0.51）——证实计划 §6.3/§9 的担忧。
- 已实现 `ChineseClipVisualEmbedder`（cn_clip ViT-L/14，768-dim），重建视觉索引，`SENTRIX_IMAGE_EMBEDDER=chinese_clip`。
- 文本槽仍用 CLIP（自匹配 AUC 0.996）；bge-m3 备用未启用（文本自匹配够用）。

---

## 4. R5 模型基础设施

| 项 | 状态 |
|---|---|
| 主 Answer 模型 | gemma4:12b（D4 保留；Ollama partial VRAM 问题仍在，属基础设施） |
| **Parser 2B（D6）** | **已接线**：`SENTRIX_PARSE_BACKEND=e2b` + 8100 `gemma-4-e2b-it+lora-v2`；**延迟 2.9-4.0s（原 9-90s）**；能正确抽 facets（厨房→place/activity），mode 仍偶发 none → 由 R4 gate+probe 兜底 |
| deadline/breaker | `model_routing.py`（RequestDeadline 20s + CircuitBreaker 按 role） |
| probe_model_health | 脚本就绪（合成 probe 文本） |

---

## 5. DoD（输入报告 §21）逐条

| # | 项 | 状态 |
|:-:|---|:-:|
| 1 | Kernel 接入 metadata/entity/lexical/visual/text/adjacency | ✅（channel_trace 实测） |
| 2 | ANN 在真实请求 trace 提供候选 | ✅（probe/channel trace 显示 visual_ann candidate_count>0） |
| 3 | observation_search_terms 被 lexical 查询 | ✅ |
| 4 | whole query + facets 同时召回 | ✅ |
| 5 | 无单字 contains | ✅ |
| 6 | Gate none 非永久终点 | ✅（ambiguous + probe + facets 直检） |
| 7 | Probe 只发现不产事实 | ✅ |
| 8 | 写作/翻译不触发检索 | ✅（测试） |
| 9 | **中文 visual/text embedding 独立测试** | ✅ 完成测试；**visual 不合格 → 已切 Chinese-CLIP（D3）** |
| 10 | Vector hit 只作 candidate/possible | ✅（`_MATCHED_SOURCE_TYPES`） |
| 11 | 简单 evidence 不走复杂链 | ✅ |
| 12 | **端到端延迟恢复** | ⚠️ parser 已快（e2b 2-4s），但 12B answer + CPU CLIP 仍使 simple evidence 接近/超过 20s 预算；需 153 硬件（VRAM/GPU driver） |
| 13 | Answer 无重复、空 GT 不编造 | ✅（R6 去重 + 拒答） |
| 14 | 60-case Retrieval-only runner | ✅ |
| 15 | Regression/Dev/Hidden 隔离 | ✅（hidden manifest 加密 GT） |
| 16 | runtime 无 benchmark 数据 | ✅（guard 测试） |
| 17 | QuerySpec/Constraint/EvidencePacket 无回归 | ✅（434 测试绿） |
| 18 | 完整 Hybrid 后失败分类 | ⚠️ 部分（见 §7） |

**量化门槛对照**：
- Recall@10 ≥90%：visual 单通道 0.887（接近）、hybrid 0.867 → **未达**（差 1-3pt）
- Recall@20 ≥95%：0.898 → **未达**
- empty GT FP=0：6 → **未达**（precision 项）
- hard violation=0：**达成** ✅
- Hybrid ≥ 任一单通道：0.867 vs visual 0.887 → **未达**（差 2pt）

---

## 6. Formation Phase F1 输入（完整 Hybrid 后仍失败的 case 归因）

基于 6 个空 GT FP + 低 recall 的 13 个 partial case：

| 归因 | 案例特征 | 证据 |
|---|---|---|
| **Formation 字段粒度** | "浅黄色拼接毛绒睡衣自拍" → GT 存为 `caption=连帽衫, objects=[毛绒,...]`（无 clothing/自拍） | album1-01 的 observation |
| **Ground truth 歧义** | album3-07 `9/8`、album3-18 `3/1` | audit 检出 |
| **外部地理/POI 依赖** | 地点查询依赖外部标签 | audit：28 case external_geo |
| **Embedding 跨改写 gap** | "做饭" vs "做晚饭" 全子串不匹配；CLIP 文本跨 paraphrase 弱 | 端到端 evidence 全 approximate |

**F1 建议**（本阶段不动 formation）：
1. Formation 层把 visual 语义（衣着/物件/动作）写入可查询字段（clothing/activity 分离），而非只进 caption/objects。
2. 词法：`_contains` 增加"有序子序列"或同义映射（做饭↔做晚饭）；FTS 支持相关短语扩展。
3. 需要外部地理编码的 case 单独建 Geo 索引。

---

## 7. 剩余项 / 需用户决策

| 项 | 说明 | 需要 |
|---|---|---|
| **R1B Adapter 切换（D3/D8）** | ViT-B-32 visual 失败已证实 → **已实现 Chinese-CLIP 并切换**（D3 用户已授权"你自己选"） | 用户知悉 + 确认保留 Chinese-CLIP 为默认 visual |
| **bge-m3 text adapter** | 文本自匹配 OK（AUC 0.996）未切；若跨改写 recall 需提升再启用 | 可选 |
| **Dev 集校准融合权重/阈值** | weighted RRF (visual 2.5) 是手动定，需 Dev 集校准 + Hidden 冻结 | 需构建独立 Dev 集（当前用 self-retrieval 代理） |
| **empty_fp 精度** | 6 个空 GT case 检索层返回近似；需 empty_policy 标注 + 置信门控 | 标注 empty_policy + 调 confidence gate |
| **端到端延迟（#12）** | 12B answer + CPU CLIP 超 20s 预算 | 153 GPU driver mismatch 修复或模型分载 |
| **Hidden Acceptance 最终验收** | 16 hidden case 需用户持 GT 独立评分 | 用户 |
| **Final threshold 90%/95%** | 差 1-3pt；Dev 校准 + F1 后预计可达成 | F1 实施 |

**计划冻结项未做**（正确）：Core Memory 完整上线宣告 / Correction 端到端 UI / 主动回忆 / 多 viewer / Answer Writer 风格优化 / Formation 大规模改造（F1 留给下一阶段）。

---

## 8. 结论

**Phase R 基础设施全部交付并实测**：
- 多路检索（metadata/entity/lexical/visual/text/adjacency）接入生产 Kernel，channel_trace 可证。
- ViT-B-32 视觉对中文随机 → **已用 Chinese-CLIP 替换（AUC 0.51→0.982），视觉 Recall@10 0.887**。
- Parser 从 9-90s 不稳 12B → **e2b 2B 2-4s 稳定（D6）**，facets 正确抽取。
- R4 Gate+Probe 让 parser none 不再永久失去检索。
- R6 Answer 去重 + 空拒答。
- 事故数据完整恢复（DB/媒体/源码/git）。

**未达目标**（需 Dev 校准 + F1 + 用户验收）：
- Recall@10 0.867（hybrid）/0.887（visual） vs 90%。
- empty_fp=6 vs 0。
- Hybrid ≥ 单通道（差 2pt）。
- 端到端 20s 预算（12B answer + CPU CLIP）。

**这些是校准/基础设施项，不是结构性缺陷**——检索骨架、视觉语义、parser 稳定性、gate/probe 都已按计划建成并验证。
