# Phase R1A · Retrieval-only runner + 基线

**日期**：2026-08-06
**状态**：工具层完成，本地 smoke 通过；153 真实数据基线待 R2 接通后填数（见下）

## 交付物

| 文件 | 作用 |
|---|---|
| `scripts/benchmarks/evaluate_retrieval_kernel.py` | Retrieval-only runner：60 case 不调回答大模型；Recall@1/5/10/20、MRR、Precision@5、all_relevant、empty-GT FP、hard violation、GT rank、通道消融参数（`--channels`，R2 前折叠到当前单 Kernel）、`--exclude-hidden` |
| `scripts/benchmarks/evaluate_parser_retrieval.py` | cached QuerySpec vs real Parser 双跑对比：约束指纹一致率、mode 一致率、Recall@10 delta；parser 不可用时降级只跑 cached 侧 |
| `scripts/benchmarks/split_hidden_set.py` | 60 case 分层抽样 16 个 Hidden（每类 ≥1），manifest **不含 GT 文件名/Asset ID** |
| `scripts/benchmarks/fixture.py` | 离线 smoke 用合成 fixture（不引用 benchmark GT） |
| `backend/tests/test_no_benchmark_runtime_dependency.py` | 隔离守护：扫 `backend/*.py`、`configs/retrieval/defaults.json`、`scripts/runtime|maintenance`，禁 benchmark query/GT 文件名/samples 路径 |
| `backend/tests/test_evidence_retrieval_benchmark.py`（扩展） | R1A runner 报告形状 / hidden 排除 / 聚合指标 3 个测试 |
| `configs/retrieval/defaults.json` | 双层配置默认层（进 git）；`data/configs/retrieval.local.json`（不进 git）留给部署覆盖 |

## 本地验证

```
unittest: backend.tests.test_no_benchmark_runtime_dependency → 3 ok
unittest: backend.tests.test_evidence_retrieval_benchmark → 7 ok
split_hidden_set → hidden=16 / regression=44（seed 20260806）
evaluate_retrieval_kernel --limit 5 --exclude-hidden → exit 0
```

## Hidden 划分（seed=20260806）

hidden_keys（16）：`album1-04,06,11,17 / album2-02,03,04,05,11,13,16,20 / album3-01,05,10,20`
regression_keys（44）：剩余全部。category_counts：empty_gt 7 / location 25 / object 8 / person 10 / time 10。

**注（D2 tradeoff）**：Hidden 从 60 case 划出，泛化力弱于全新 case；R7 报告按"划分型 Hidden"标注结论可信度。实现期间不读 hidden 的 GT 文件列表调 runtime 规则。

## 153 真实数据基线（待办，R2 后填数）

真实 60 case 的 Retrieval-only 基线与通道消融需在 153 真实 DB 上跑（本地无 `data/sentrix.db`）。执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmarks/evaluate_retrieval_kernel.py \
  --db data/sentrix.db --spec-source cached --exclude-hidden docs/baseline/hidden_set_manifest.json \
  --report docs/baseline/retrieval_baseline_R1A.json
```

> 说明：R1A 基线代表"当前单 Kernel"的原始水平（预期 Recall 低、FP 高，正是 Phase R 要修的）。R2 接通道后同 runner 复跑同一 Regression Set，比较提升。
