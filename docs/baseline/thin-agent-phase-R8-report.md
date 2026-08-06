# Phase R8 · Acceptance and Ranking Hardening — 报告

**日期**：2026-08-06
**性质**：R8 七子阶段 + Parser 验收 + 部署安全，8 项关闭门槛逐条对照
**数据**：全部实测（153 真实 DB + 本地测试 446 绿）

---

## 1. R8 子阶段完成情况

| 子阶段 | 交付 | 状态 |
|---|:---|:-:|
| R8-1 标注政策 | `annotate_benchmark_cases.py` → `benchmark_annotations.json`（60 case：30 strict_empty / 30 allow_approximate，7 empty-GT，2 GT 冲突冻结）；分类与用户规则一致（贵阳/明哥+江西→strict，海豚→allow） | ✅ |
| R8-2 Dev/Hidden | `build_development_set.py` → **45 case、12 类**合成 Dev 集（activity/place/person/visual_object/color/scene/time/bare/composite/strict_empty/allow_approximate/all_relevant）；runner 加 `--dev-set`；Hidden 16 冻结不动 | ✅ |
| R8-3 排序重设计 | `ranking.py` 三策略；**默认 visual_backbone**（visual 骨架 + 弱通道补召回，不位移 visual top-K）；每 GT 通道轨迹（`packet.channel_hits`）；adjacency 默认关；Hybrid 影子 | ✅ |
| R8-4 Text 决策 | `evaluate_text_paraphrase.py`：**CLIP 文本 paraphrase recall@10=0.216（<0.5）→ 判定不合格**；bge-m3 被 torch 2.5.1 + ~/.local transformers 5.11 阻塞（R8-6 infra）；policy=低权重 append-only 实验通道 | ✅（决策记录） |
| R8-5 空/近似 | `_gate_packet_approximate`：strict_empty 锚点查询无直接支持→拒答；allow_approximate→返回+披露；弱近似按 recall_strength 过滤。**端到端验证：贵阳/明哥搂着我→0 evidence+拒答；海豚→10 近似+披露** | ✅ |
| R8-6 性能/基础设施 | `measure_latency.py`：**simple_evidence p50=0.47s**（原 240s）；normal_chat 0.002s；GPU driver mismatch 存在（NVML 595.84），CLIP 已 CPU 规避，12B answer 慢，**实际修复需用户 infra 决定** | ✅（测量）+ ⚠️（GPU） |
| R8-7 最终验收 | 本报告 + 关闭清单 | 见 §3 |
| R8-Parser | `evaluate_parser_acceptance.py` 独立指标（见 §2） | ✅（指标）+ ⚠️（mode 不合格） |
| R8-8 部署安全 | `backup_sentrix.sh`（SQLite backup API + 分层 + manifest，1.1GB 实测）+ `deployment-safety.md`（红线/排除清单/恢复演练） | ✅ |

---

## 2. 关键实测结果

### 排序策略（Dev 45 case + Regression 44 case）
| 策略 | Dev r10 | Dev r20 | Dev mrr | Reg r10 | Reg r20 |
|---|:-:|:-:|:-:|:-:|:-:|
| visual_only | 0.596 | 0.637 | 0.353 | 0.887 | 0.895 |
| **visual_backbone（默认）** | **0.624** | **0.712** | 0.371 | **0.891** | **0.926** |
| late_fusion | 0.645 | 0.694 | **0.503** | 0.867 | 0.898 |

- **visual_backbone 在 Dev 与 Regression 都 ≥ visual_only**，recall@20 最高（0.926）。
- 每 GT 通道轨迹已实现（visual/lexical/text/final rank），供 Hidden 分析。
- strict_empty_fp=0、hard_violation=0（Dev 三策略均）。

### 端到端空结果/近似（API）
| 查询 | 类型 | 结果 |
|---|---|:---:|
| 贵阳夜晚步行街 | strict_empty（geo 锚） | **0 evidence + "当前记忆中没有找到足够匹配的原始证据。"** |
| 夜晚车内的明哥搂着我 江西省 | strict_empty（关系+geo） | **0 evidence + 拒答** |
| 水族馆海豚跃出水面 | allow_approximate | 10 近似 + "无法完全确认" 披露 |

### Parser 独立验收（e2b 2B，60 evidence + 10 general）
| 指标 | 值 | 判定 |
|---|:-:|:-:|
| **mode_accuracy** | **0.29** | **不合格**（模型弱，few-shot 边际改善 0.29→0.29） |
| action/facet 保留率 | 0.23 | 不合格 |
| hard_condition 漏失 | date 8% / person 12% / media 5% | 中等 |
| general 误触发 | 0.0 | ✅ |
| JSON 首过合法率 | 0.95 | ✅ |
| probe 兜底率 | 0.30-0.62 | 部分 |

**结构性补偿**：gate 对 parser-none + 人物/日期/地点/关系锚点 → ambiguous+probe（已验证贵阳/厨房 case 恢复）。**有效家庭查询恢复 ≈ 0.29 + 0.62×0.71 ≈ 0.73**，仍 ~27% 流失。

### 延迟（6 次 API）
| 路径 | p50 | p95 | 模型调用 |
|---|:-:|:-:|:-:|
| 普通聊天 | 0.002s | 0.002s | 1 |
| 简单 evidence | **0.47s** | 0.55s | 2 |
| 复杂人物 | 0.001s* | 0.001s | 2 |

\* 复杂人物触发 parser-none + intro-verb 走普通聊天（parser mode 不稳的已知 tradeoff）。

---

## 3. R8-7 关闭门槛逐条

| # | 门槛 | 状态 |
|:-:|---|:-:|
| 1 | Hidden Set 通过（Recall/MRR/strict-empty FP/approx 合法率/hard violation/与 Dev 差值） | ⚠️ **需用户持 16 hidden GT 独立评分** |
| 2 | strict-empty FP = 0 | ✅ 端到端验证（贵阳/明哥→0）+ Dev strict_empty_fp=0 |
| 3 | hard violation = 0 | ✅ |
| 4 | approximate 全部正确披露 | ✅（"无法完全确认" 披露；差异说明） |
| 5 | 默认排序 ≥ Visual-only | ✅ visual_backbone Reg 0.891 ≥ visual 0.887 |
| 6 | Recall@10 ≥90% 且 Recall@20 ≥95% | ⚠️ Reg r10=0.891（≈90%）/ r20=0.926（差 2.4pt）；**未达**；Dev r10=0.624（Dev 更难） |
| 7 | E2E 延迟达标或有用户接受降级 | ✅ simple_evidence 0.47s 达标；复杂路径受 parser 影响 |
| 8 | 部署安全门槛 | ✅ backup_sentrix.sh 实测 + deployment-safety.md + 恢复流程演练 |

---

## 4. 核心发现与阻塞

1. **Parser mode 是最大阻塞**：e2b 2B mode_accuracy 29%。few-shot 无效。结构性补偿（gate 锚点）恢复 ~73%，但 ~27% 家庭查询仍流失，且复杂人物路径未生效。**需要更强的 parser 模型或用户接受补偿后的行为**。
2. **Text ANN 未通过验收**：paraphrase recall@10=0.216。bge-m3 切换被 torch 2.5.1 阻塞（R8-6 infra）。当前低权重 append-only。
3. **视觉通道（Chinese-CLIP）是最强通道**：visual_only r10=0.887；visual_backbone 默认 0.891。
4. **延迟大幅改善**：simple_evidence 0.47s（240s→0.5s）。
5. **GPU driver mismatch**（NVML 595.84）仍存在：CLIP 已 CPU 规避；12B answer 慢；实际修复需用户 infra 决定（有回滚方案）。

---

## 5. 需用户决断（R8 后）

| # | 决断 | 选项 |
|:-:|---|---|
| 1 | **Parser 下一步** | a) 换更强 parser 模型（需 153 部署）；b) 接受"parser 弱 + gate 结构性补偿"（~73% 恢复）；c) 继续 prompt 调优（已试 few-shot，边际） |
| 2 | **bge-m3** | 是否在 R8-6 修 torch/venv 后启用 bge-m3 替换 CLIP Text |
| 3 | **GPU driver mismatch** | 是否由你安排独立修复（有回滚方案）；或接受 CPU CLIP + 12B answer 慢 |
| 4 | **Hidden Acceptance** | 请持 16 hidden case 的 GT 独立评分，回传 Recall@10/20/MRR/strict-empty FP/approx 合法率/hard violation |
| 5 | **Recall@20 94%→95% 缺口** | 是否接受当前 0.926（差 2.4pt）为已知缺口，或要求补 Formation/GT blocker |
| 6 | **Phase R8 关闭** | 达标项确认后是否关闭 Retrieval 基础阶段、进入下一阶段（Formation F1 / 其他） |

---

## 6. 结论

**R8 已完成排序重设计（visual_backbone 默认）、空/近似政策（strict_empty 拒答 + allow_approximate 披露）、独立 Dev 集（45 case）、Text 决策（CLIP 不合格、bge-m3 待 infra）、性能测量（simple_evidence 0.47s）、部署安全（备份+红线）**。

**关闭门槛 8 项中 4 项✅、4 项 ⚠️**：
- ✅ strict-empty FP=0、hard violation=0、approximate 披露、默认排序≥Visual-only、延迟达标、部署安全
- ⚠️ Hidden 需用户评分、Recall@10/20 差 1-3pt、**Parser mode 29% 是最大阻塞**、bge-m3/GPU 需 infra

**建议**：Parser 是唯一系统性阻塞。建议用户先从 §5 决断 1（Parser 方向）+ 4（Hidden 评分）开始，其余为校准/基础设施项。
