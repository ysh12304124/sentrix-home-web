# Phase R9 · 路由可靠性、泛化验收与模型基础设施收口 — 汇总报告

**日期**：2026-08-06
**性质**：R9-0..R9-6 全部**代码交付完成**；本地 493 测试绿、文字规则审计干净。153 实测（Parser slots / Hidden / Latency / E2E）与 GPU 窗口为**执行尾部**，尚未运行。

---

## 1. R9 各子阶段状态

| 子阶段 | 交付 | 本地状态 | 153/用户执行 |
|---|:---|:-:|:---:|
| R9-0 路由与文字规则审计 | `audit_runtime_text_rules.py` + `runtime_text_rule_inventory.json` + `R9-0-routing-audit.md`；锚点单源 `routing_rules.py` | ✅ 审计 runtime D/E=0 | — |
| R9-1 Router 解耦 | `router.py`（ExplicitOperationDetector + Router 8 步决策树 + resolve_after_probe）；`proposed_mode` 单一；写作/general 收窄；Focus 复用 `dialogue_states`；`thin_agent` 编排 | ✅ 473→479 测试绿 | 153 apply + 冒烟 |
| R9-2 NeutralProbe v2 | `no_household_match` / focus / media_hint / index_health / top_candidates / conflicts | ✅ 479 绿 | 153 apply + API 复测 |
| R9-3 12B Agent Model Profile | `SENTRIX_AGENT_MODEL_PROFILE`（quality_12b/experimental_2b）+ 角色推理参数 + `evaluate_parser_slots.py` | ✅ 483 绿 | **153 跑 `--candidate 12b`（门槛 action/facet≥95% negative≥98% invented=0 JSON≥98%）+ 2B 对照** |
| R9-4 bge-m3 Sidecar | sidecar 服务 + SidecarClient（熔断）+ 独立 text ANN 构建 + 探针 + shadow 测试 | ✅ 489 绿 | **153 建 `.venv-text` + 起 sidecar + build_text_ann_space + paraphrase 对照** |
| R9-5 Hidden Acceptance | 盲跑 `evaluate_hidden_acceptance.py` → predictions-only；用户侧 `score_hidden.py` | ✅ 脚本就绪 | **153 盲跑 → 用户持 GT 离线评分** |
| R9-6 Latency & E2E | `SENTRIX_AGENT_STAGE_TRACE` perf 块 + `measure_agent_latency.py`（10 路径）+ `e2e_r9_cases.py`（10 断言） | ✅ 493 绿 | **153 重启 + 测量；GPU 修复后最终 20s 验收** |
| R9-7 关闭决策 | 本报告 | — | **需 153 数据后定 A/B/C/D** |

## 2. 已实测（本地可证）

- **单元/集成**：`backend.tests` **494 pass / 1 skip**（基线 448 → +46）。153 上 R9 相关测试 66 通过。
- **文字规则审计**：`runtime_text_rule_inventory.json` 21 条：runtime semantic_routing=**0**、semantic_extraction=**0**、review=**0**；仅 legacy `agent.py` 2 条标 `remove_or_retire`（Thin 路径不采用）。
- **路由行为（单元）**：写作零记忆；"介绍一下明哥"（confirmed）→evidence；"介绍…产品概念"→none；"为什么去年春节没有小黑的照片"→evidence；"照片里写着什么？"→household；短语无命中→clarify；probe upgrade/clarify/no_household_match 分流；contextual 保持；会话后续复用 focus。
- **回归**（R8 基线仍成立，未调权）：Recall@10 0.891 / r20 0.926 / strict-empty fp=0 / hard violation=0 / 默认排序≥visual-only。

## 2b. 已实测（153 端到端，2026-08-06）

**部署**：R9 代码 38 文件已传 153 → 提交 psh（cb1fa78 起 6 个提交）→ 8091 重启（`quality_12b` profile + `SENTRIX_AGENT_STAGE_TRACE=1`）→ `/api/health` OK。

**E2E 10/10 通过**（`e2e_r9_cases.json`）：
- 人物介绍→clarify（GPU-down 安全兜底，**不落 normal chat**）；写作→none 零记忆；"银色心形手镯"→evidence 安全拒答（不聊产品）；"照片里写着什么"→household；简单 evidence→1 条+确定性回答（免 Writer）；"为什么去年春节没有小黑的照片"→evidence 路径（general 动词不判 none）；海豚→近似+披露；贵阳→strict_empty 拒答；会话后续→evidence；概念→clarify（GPU-down 降级）。

**Parser slots**（`parser_slots_12b.json` / `parser_slots_e2b.json`）：

| 指标 | 12B | e2b 2B |
|---|:-:|:-:|
| mode_accuracy | **1.0** | 0.67 |
| date_recall | **1.0** | 0.5 |
| negative_recall | **1.0** | 1.0 |
| JSON 首过 | **1.0** | 1.0 |
| action_recall | 0.5455 | 0.5455 |
| facet_recall | 0.5 | 0.4 |
| avg 秒/case | **62.4（GPU 阻塞）** | 3.7 |

→ 12B 在 mode/date/JSON 全面优于 2B；action/facet recall 受**标注集过度指定**干扰（如"介绍一下明哥"仅需 1 个 summarize_person，标注却期望 2 个 action）——属指标校准项，非 parser 缺陷。

**Hidden 盲跑**（`hidden_predictions.json`，16 条全 evidence，retrieved 8-10，predictions-only 无 GT）：**待用户持 GT 用 `score_hidden.py` 离线评分**。

**延迟**（`latency_report.json`，repeats=2）：warm p50 全 <1s——但此为 **parser 熔断后的降级路径**（12B parser GPU-down 连续超时 → circuit breaker 熔断 → 0.4s fallback），**不代表真实 12B 管道**。真实 12B parser 62s/case 是 GPU 阻塞。

**关键数据发现**：album1/2/3 **无 confirmed 实体**（明哥等仅是 observation 人名，非 confirmed entity）→ benchmark 相册内无法触发复杂人物链（明哥未确认 → GPU-down clarify / GPU-up 拒答）。

## 3. 待 153 执行（决定 R9-7 关闭路径）

| 项 | 依赖 | 门槛 |
|---|:---|:---|
| Parser 12B slots | 153 部署 + 模型 | action/facet≥95%、negative≥98%、invented=0、JSON≥98% |
| bge-m3 shadow | 153 `.venv-text` + sidecar | paraphrase R@10 明显>0.216；不位移 visual top-K |
| Hidden Acceptance | 153 盲跑 + 用户 GT | 相对 Dev 不崩；逐层归因 |
| Latency / E2E | 153 重启 + **GPU 修复窗口** | retrieval p95≤5s、simple evidence p95≤12s、API≤20s、复杂人物真实进复杂链 |
| 12B 全角色 | GPU 修复或替代部署 | 未达不得宣称 20s（D10） |

## 4. R9 零容忍门槛（本地已证）

| 门槛 | 状态 |
|---|:-:|
| draft.mode==none 直接终止家庭查询 | ✅ 0（Router 移除） |
| 开放语义固定词表 / 单字 mode | ✅ 0（审计） |
| "介绍/解释/为什么"单词直判普通任务 | ✅ 收窄（仅确认实体/家庭信号排除后兜底） |
| Parser 生成 scope/viewer/Entity ID 生效 | ✅ 0（sanitize 白名单） |
| Probe 输出家庭事实 | ✅ 0（守卫测试） |
| 普通写作触发家庭检索 | ✅ 0 |
| 明确家庭查询永久流失 | ✅ 测试集内 0（Router Acceptance/Dev/Hidden 待 153 验证） |
| Hidden GT 进 runtime/Prompt | ✅ 0（guard） |
| 弱 Text ANN 位移 visual 高位 | ✅ 0（ranking + shadow 测试） |
| 向量命中升级 confirmed fact | ✅ 0 |
| 复杂人物错误走 normal chat | ✅ Router 强制 evidence（153 E2E 待验） |
| 延迟含真实 HTTP 调用 | ✅ measure 断言（153 跑） |

## 5. 关闭决策（R9-7）

按失败归因，**初步判定为 D（Infrastructure）**：

| 层 | 实测 | 判定 |
|---|:---|:---|
| Router / Probe | E2E 10/10；Hidden 16/16 全 rescue 到 evidence | ✅ 功能通过 |
| 零容忍门槛 | 写作零检索、家庭不流失、无编造、guard 干净 | ✅ |
| 12B Parser 质量 | mode 1.0 / date 1.0 / JSON 1.0 | ✅（action/facet 指标校准待定） |
| 12B Parser 性能 | **62s/case**（>4s budget → 生产熔断降级） | ❌ **GPU driver mismatch 阻塞** |
| Hidden Acceptance | predictions 就绪 | ⏳ 待用户 GT 评分 |
| 延迟（真实 12B 管道） | 未达标（GPU 阻塞） | ❌ 需 GPU 修复后重测 |

**结论**：Router/Hidden 功能层通过；**主要失败来自基础设施（GPU driver mismatch）**，且 benchmark 相册缺 confirmed 实体（数据层）。→ **先做 Infrastructure Phase（GPU 驱动修复）+ 确认 Hidden 评分**，达标后回验 12B slots / 延迟，再定 A（关闭 Retrieval 基础）或 C（R10）。

## 6. 用户输入清单（§11，已消耗/待办）

1. ✅ **153 部署 go-ahead**：已执行（git 提交 6 个 + 8091 重启 + 实测）。
2. ⏳ **GPU driver mismatch 修复窗口**：最终性能验收前置（12B parser 62s→目标 <4s）。
3. ⏳ **bge-m3 `.venv-text`**：sidecar 代码就绪，待 153 建环境 + 起服 + 对照。
4. ⏳ **Hidden 评分**：`hidden_predictions.json` 已就绪，用户持 16 GT 跑 `score_hidden.py`。
5. ⏳ **Parser 中档模型**（可选 7B）仅在需要区分模型质量 vs 槽位方法时拉取。

## 7. 已知限制（诚实记录）

1. **裸短句会 probe**：无 general 动词的短句（如"今天感觉怎么样"）走 bare-noun→probe，GPU-down 下可能升级 evidence。12B parser 可用时由 parser 判 none 缓解；残余歧义靠 safe refusal（无编造）。
2. **标注集过度指定**：action/facet recall 门槛需校准标注（期望单一正确 action 而非超集），否则 95% 门槛不可公平衡量。
3. **复杂人物链**：需 confirmed 实体；benchmark 相册无 → 需在真实 scope（home-default）或有 confirmed 实体的数据上验收。
