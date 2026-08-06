# R9-5 · Hidden Acceptance 只读评分 — 报告

**日期**：2026-08-06
**性质**：16 hidden case 盲跑（predictions-only）+ 用户侧离线评分器。Agent 实现期不读完整 hidden GT。
**测试**：本地 489 通过（无新增单元测试——评分流程依赖 153 DB + 用户 GT，离线执行）。

## 1. 交付

| 文件 | 说明 |
|---|---|
| `scripts/benchmarks/evaluate_hidden_acceptance.py`（新） | 读 `hidden_set_manifest.json` 16 条（query+category，无 GT）；对每条跑**生产路径**：12B QueryParser → Router.route → (ambiguous→NeutralProbe→resolve_after_probe) → EvidenceRetrievalKernel.retrieve；记录 route/reason/parser 槽位/retrieved_asset_ids/evidence_levels/recall_strengths/gaps/excluded/latency；输出 `hidden_predictions.json`（**不含 GT**） |
| `scripts/benchmarks/score_hidden.py`（新） | **用户侧**离线评分：读 predictions + 用户 GT（`--gt`，格式见 docstring），输出 §9.2 指标：Recall@1/5/10/20、MRR、strict-empty FP、approximate 合法率、all_relevant recall、Router 家庭→evidence 率 / general 误触发 / clarify 数 / 路由流失、Parser mode 准确率 |

## 2. 只读隔离

- `hidden_predictions.json` 只含预测（asset_ids、route、槽位），无 GT。
- 完整 GT 由用户持有；用 `score_hidden.py --gt <gt.json>` 离线评分。
- 若 Hidden 明显低于 Dev/Regression → **不调参**，按 parser/embedding/formation/ranking/GT/分布差异归因（R9 §9.3）。

## 3. 待执行（153 + 用户）

- [ ] 153：`evaluate_hidden_acceptance.py`（需 12B parser + DB + ANN）→ `hidden_predictions.json`
- [ ] 用户：准备 `hidden_gt.json` → `score_hidden.py` → `hidden_acceptance.json`
- [ ] 冻结结果，回传指标

## 4. 下一步

R9-6：stage trace + 全阶段延迟测量 + 复杂人物真实复杂链 + GPU 前后对照 + 端到端 case 验证（§10，11 条）。
