# Agent 2.1 P2 — Canonical Retrieval Intent 回归报告（candidate 评估）

日期：2026-08-20 ｜ 12B，canonical 开启（`SENTRIX_CANONICAL_SEARCH=1`）
run：38q canonical `192816-37d764` / paraphrase canonical `195129-6a1deb`，对比 baseline `612fb3`/`527f8f`

## 1. 38q 对比

| 指标 | baseline | canonical | Δ |
| --- | --- | --- | --- |
| AQ | 1.158 | 1.108 | **-0.050** |
| Core | 0.684 | 0.622 | -0.062 |
| Exact | 0.474 | 0.486 | +0.012 |
| Recall | 0.800 | 0.800 | 0 |
| **Agent 延迟** | 20.7s | **13.8s** | **-33%** |

## 2. Paraphrase 对比

| 指标 | baseline | canonical | Δ |
| --- | --- | --- | --- |
| AQ | 0.776 | 0.741 | -0.035 |
| angle 风格 AQ | 0.591 | 0.409 | **-0.182** |
| syntax AQ | 1.048 | 1.095 | +0.047 |
| order AQ | 0.714 | 0.810 | +0.096 |

逐父题：3 改善（q08 +0.5 / q26-q01 +0.5 / q47-q04 +0.75）、17 持平、2 退化（q26-q07 -1.0 / q47-q01 -1.08）。

## 3. 判定：不切 candidate

- **收益**：延迟 -33%（确定性元数据路径快于 hybrid ANN）。
- **代价**：质量略降（38q -0.05、para -0.035）；**angle 鲁棒性没改善反而更差**（-0.18）。
- **原因**：canonical 强制走元数据路径（空 query + filters），对需 OCR 关键词/语义召回的题（q26-q07、q47-q01）丢失 hybrid 优势。
- **结论（按纪律）**：数据不支持切 candidate。保持门控关闭。

## 4. Canonical v2 方向（不切 candidate 的前提下）

1. **融合而非替换**：canonical 结构化约束作为 filters 增强，仍保留 hybrid 检索。
2. **按题型门控**：仅明确"时间+地点"检索型题走 canonical；OCR/关键词型保留 legacy。
3. **angle 鲁棒性属 planner 层**（证据类型映射），非检索层。

## 5. 数据结论对计划的含义

- Canonical Retrieval Intent 的**延迟收益真实**（-33%），但当前实现质量不达标 → 迭代到 v2（融合）而非切换。
- P2 实现任务保持 pending；下一步转向 W3.2（OCR）与 W2.3（引用消解）。
