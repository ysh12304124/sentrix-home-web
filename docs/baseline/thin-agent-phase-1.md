# Phase 1 · Evidence Retrieval 基线记录

**采集时间**：2026-08-05
**报告 JSON**：`/tmp/phase1-baseline.json`
**运行方式**：`PYTHONPATH=. python scripts/benchmarks/evaluate_evidence_retrieval.py --report /tmp/phase1-baseline.json`

## Baseline 结果

| Configuration | Passed | Total |
|---|---|---|
| thin_agent_v1_on | 7 | 10 |
| thin_agent_v1_off | 4 | 10 |

## 逐 Case 详情（Thin Agent V1 ON）

| ID | Query | 通过 | 硬约束违反 | 缺失预期 | 观察到的 mode |
|---|---|---|---|---|---|
| B-01 | 2024 年 5 月厨房的照片 | PASS | — | — | evidence |
| B-02 | 厨房里做晚饭的照片 | PASS | — | — | evidence |
| B-03 | 浅黄色拼接毛绒睡衣自拍 | **FAIL** | — | pajamas_selfie.jpg | none |
| B-04 | 贵阳夜晚步行街 | **FAIL** | — | — | none |
| B-05 | 明哥的照片 | PASS | — | — | evidence |
| B-06 | 不要妈妈和视频 | **FAIL** | — | — | none |
| B-07 | 把厨房的所有相关照片都找出来 | PASS | — | — | evidence |
| B-08 | 请直接给我明哥的相关原图 | PASS | — | — | evidence |
| B-09 | 帮我写生日祝福 | PASS | — | — | none |
| B-10 | 今天很累，突然有点想小黑 | PASS | — | — | contextual |

## 逐 Case 详情（Thin Agent V1 OFF · 老 agent）

| ID | Query | 通过 | 硬约束违反 | 缺失预期 | 观察到的 mode |
|---|---|---|---|---|---|
| B-01 | 2024 年 5 月厨房的照片 | **FAIL** | kitchen_july_dinner.jpg | — | evidence |
| B-02 | 厨房里做晚饭的照片 | PASS | — | — | evidence |
| B-03 | 浅黄色拼接毛绒睡衣自拍 | **FAIL** | — | pajamas_selfie.jpg | none |
| B-04 | 贵阳夜晚步行街 | **FAIL** | — | — | none |
| B-05 | 明哥的照片 | PASS | — | — | evidence |
| B-06 | 不要妈妈和视频 | **FAIL** | — | — | none |
| B-07 | 把厨房的所有相关照片都找出来 | PASS | — | — | evidence |
| B-08 | 请直接给我明哥的相关原图 | **FAIL** | — | ming_kitchen.jpg, ming_outdoor.jpg | evidence |
| B-09 | 帮我写生日祝福 | PASS | — | — | none |
| B-10 | 今天很累，突然有点想小黑 | **FAIL** | — | — | none |

## 诊断

**Thin Agent V1 ON**（前一位 Agent 的骨架）:
- 修好了老 agent 的硬约束违反（B-01 月份过滤）与原图授权（B-08 image_results）
- 但 B-03/B-04/B-06 因关键词表未命中而误路由到 `none` 模式——精确印证了补充计划 §1.1 关于 MemoryGate 关键词表的诊断

**老 agent**:
- 老 agent 在硬过滤（时间/scope）上有明显漏洞（B-01 违反）
- 老 agent 不返回原图 assets（B-08 缺失）
- B-10 直接进 `none` 而非 `contextual`——老 agent 完全没有 contextual 模式概念

## Phase 2R 语义修复的目标增益（预期）

修复后 Thin Agent 应达到 10/10：
- B-03: 需要真实 QueryParser 解析开放语义"睡衣"、"毛绒"、"自拍" → 应进 evidence 模式并返回 pajamas_selfie
- B-04: 需要 QueryParser 识别"贵阳"是地名条件 → 应进 evidence 模式并返回空（因为无 Asset 匹配"贵阳"）
- B-06: 需要 QueryParser 识别"不要"作为语义否定 + 视频类型排除 → 应进 evidence 模式并返回排除后的相关 Asset

3 个 case 全都需要"模型开放语义理解"，不能靠关键词修补。

## 测试文件

- `backend/tests/test_evidence_retrieval_benchmark.py`：4 个结构测试全通过（在 CI 快速验证 benchmark 可跑）

## 退出条件

- ✅ `scripts/benchmarks/evaluate_evidence_retrieval.py` 可运行，输出可解析 JSON
- ✅ 10 个 case 全部有 baseline 数值
- ✅ 结构单元测试全绿
- ✅ 全量后端测试 `python -m unittest discover backend.tests` 相比 Phase 0 无回归

Phase 1 完成。下一步：Phase 2R-1 加入语义 red tests。
