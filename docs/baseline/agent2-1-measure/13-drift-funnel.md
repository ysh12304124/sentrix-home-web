# Agent 2.1 P1 — Paraphrase Drift Funnel（四层 paired attribution）

日期：2026-08-20 ｜ 数据：22 父题 + 88 改写（12B `527f8f` / 0.8B `d8a8b4`）vs 父题原 run（12B `612fb3` / 0.8B `6d4919`）
> 纯测量，未改 Agent。

## 1. 无效样本解释（P1a）

12B 85/88、0.8B 86/88 有效，共 5 个无效（12B：q08-1/q08-2/q03-3；0.8B：q01-4/q26-01-2）。

全部 `reason = judge_inconsistent_score_reason`：**judge 模型给出的分数与自己的理由矛盾**，orchestrator 触发一致性重试后仍矛盾 → 判无效。**Agent 本身全部 complete**，无 runtime error。属 judge 侧一致性问题，非 agent 失败。

## 2. 四层 paired attribution（P1b）

对每个 parent + 4 改写提取四层信号：

| 层 | 信号 | 说明 |
| --- | --- | --- |
| Planner | `task_declaration.requirements` 的 evidence_type 集合 | 规划器声明要查哪些证据 |
| Search Request | 首次 `search_memories` 的 query + filters | agent 实际发起的检索请求 |
| Retrieval | `retrieval_recall` | 检索是否找回 GT 照片 |
| Evidence Availability | `answer_context.facts` 是否含关键事实 | 证据是否到 Writer |

对比 parent vs 改写，取**第一个显著分歧层**为"首漂移层"。

## 3. Drift Funnel（P1c，只看掉分的改写）——【修正版】

> 修正说明：v1 用 tool_trace 提取 search 参数得到空串，误报"Search 漂移=0"。改用 debug_trace 的 action JSON 提取真实 query/filters 后重算如下。

**12B**（19/88 掉分）：

| 风格 | 掉分/总 | planner | search | retrieval | evidence |
| --- | --- | --- | --- | --- | --- |
| syntax | 1/22 | 0 | 0 | 0 | 1 |
| lexicon | 4/22 | 3 | 1 | 0 | 0 |
| order | 6/22 | 3 | 2 | 0 | 1 |
| angle | 8/22 | 3 | **5** | 0 | 0 |
| **合计** | **19/88** | **9 (47%)** | **8 (42%)** | **0 (0%)** | **2 (11%)** |

## 4. 关键发现（修正版）

1. **Planner 与 Search 并列为主要漂移层**（47% + 42% = 89%）：换说法后，**Planner 声明不同需求集**（掉需求），且 **search_memories 构建了不同的 query/filters**（例：q26-q06 原题 query"上海青杉路顶呱呱炸鸡店 菜单"带 place → 改2/改4 变"顶呱呱 菜单"丢 place → judge 0）。
2. **Retrieval 本身零漂移**：给定 query，检索结果稳定。失败全在"语义→检索意图"这一跳。
3. **angle 最脆且 100% 在上游**（8/8 = planner 3 + search 5，下游 0）——换提问角度时，失败完全由"语义→检索意图"映射导致。
4. **Evidence 层 11%**：planner/search 对齐后下游仍错（视觉/OCR/写作），属 V1/V3 领域。

## 5. 对 P2（Canonical Retrieval Intent）的结论

- **数据强支持 P2**：89% 的 paraphrase 漂移发生在"语义→检索意图"（planner 需求 + search query/filters），Retrieval 自身稳定 → 治本点是**把不同说法映射到同一组 canonical retrieval_target + structured constraints**，让 search 不再依赖用户原话的措辞。
- **P2 能覆盖 ~89%** 的漂移（planner+search）；剩余 11%（evidence/visual/OCR）需 V1/V3 修复配合。
- 按你定的纪律：P2 先做 **shadow 对比**（legacy vs canonical query 的 retrieval recall/F1），数据说话再切 candidate；不与 0-result relaxation 同批。

## 6. 后续

- P2 shadow 实验：对 22 父题 + 88 改写，比较 legacy query vs canonical retrieval_target+constraints 的 recall/F1。
- 并行推进 W2.3（引用消解）、W3.2（OCR）。
- 全部完成后重跑 22+88+OCR subset+reference subset，报 AQ/angle AQ/retrieval/evidence/latency/extra-calls/regression。
- 最后重生成 Error Pareto，再决定是否进 V1。
