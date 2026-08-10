# Phase B B2.2/B2.3 — Faithfulness Benchmark + Action Serialization Hardening

- 日期：2026-08-10
- 状态：✅ 完成（benchmark 达标；序列化 50%→13%，guided 对照已实测）

## B2.2 — Observation Faithfulness Benchmark（40 例，8105 gemma-12B）

固定工具 Observation + 用户问题 → 模型 final → FinalGuard + 规则评分。

| 指标 | 目标 | 实测 |
|---|---:|---:|
| Faithfulness 通过率 | >=95% | **100%（40/40）** |
| False Positive Fulfillment | 0 | **0** |
| False Negative / Omission | <=2% | **0** |
| Certainty Upgrade Error | 0 | **0** |
| Required Disclosure 失败 | <=2% | **0** |
| Raw JSON valid | - | 100% |

- 覆盖类别：exact count/exists/first-last/empty/candidate_only/partial/omit/all-has_more/group/conflicting
- 说明：模型在明确规则下高度诚实（所有 candidate_only 案例都输出"接近的候选，还不能完全确认"）；评分器初版 7 个"失败"经核查全部为评分正则缺口（模型表述诚实但用词不同），修正后 40/40
- 产物：`scripts/benchmarks/faithfulness_cases_v1.json` + `evaluate_faithfulness.py` + `/tmp/faithfulness_12b.json`

## B2.3 — Action Serialization Hardening（对照实测）

| 配置 | raw repair | complete/18 | avg | 备注 |
|---|---:|---:|---:|---|
| 基线（psh，free JSON） | **50%**（36 输出 18 修复，14 围栏） | 10/18 | 1.9s | A5 状态 |
| free JSON + 禁围栏提示 + 去重 | **13%**（46 输出 6 修复，围栏 0，截断 6） | **13/18** | 3.6s | 采纳为默认 |
| vLLM guided_json | 7.1%（56 输出 4 修复） | 11/18 | 7.1s | 修复低但 complete 下降 |
| guided + max_tokens 1500 | - | 10/18 | 10.4s | guided 解码慢且模型循环增加 |

- 采纳：**免费 JSON + 明确禁止 markdown 围栏 + 同轮相同 tool+arguments 去重**（`runtime.py`）
- 残留 13% 均为截断（unclosed string/brace），由既有修复链兜底
- vLLM `response_format=json_schema` 可用且修复率最低，但当前模型在约束解码下工具行为退化（重复调用/不 final），按计划"不强行迁移"；留作模型升级后重试
- 结论：**目标 <=5% 未完全达标**；50%→13% 为主链改进，剩余截断是模型序列化稳定性问题（12B 4-bit），修复链保证无用户可见坏输出

## 联动修改
- `runtime.py`：SYSTEM 禁 markdown 围栏 + 重复调用去重（`duplicate_tool_call` 拒绝）；`max_tokens 800→1500`（app.py + shadow 脚本）
- 去重使 s18 类"帮我看看照片"不再空转（`partial` + 明确 reason）
