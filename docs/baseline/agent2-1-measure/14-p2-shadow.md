# Agent 2.1 P2 — Canonical Retrieval Intent Shadow 报告

日期：2026-08-20 ｜ 数据：22 父题，canonical（时间+地点结构化约束）vs legacy（agent 实际 query）
> 纯 shadow 测量，未改 agent。

## 1. Shadow 设计

对 22 个父题，定义 canonical 结构化约束（时间前缀 + 地点别名集合），用**元数据检索路径**（captured_at + place/geocode label/district/city 匹配）查 scope `album_9ef2ac31b551`，度量 GT 图片召回；对比 legacy（agent 实际 run 的 `retrieval_recall`，取 4 个改写平均）。

地点用别名集合（如 清迈 → `[清迈, Chiang Mai, Hang Dong, 清迈夜间动物园]`），模拟 canonical 地点解析（等价于把"清迈夜间动物园"解析到存储的 Hang Dong district）。

## 2. 结果

| 统计 | 值 |
| --- | --- |
| **canonical ≥ legacy** | **22/22** |
| canonical 召回 1.00 的父题 | 22/22 |
| legacy 召回范围 | 0.00 ~ 1.00（q24-01=0.00、q26-03=0.00、q26-01=0.50、q03=0.25） |

逐题（节选）：
```
q03   GT1  canon1.00  legacy0.25
q24-01 GT3 canon1.00  legacy0.00
q26-01 GT2 canon1.00  legacy0.50
q26-03 GT2 canon1.00  legacy0.00
q47-04 GT4 canon1.00  legacy1.00（legacy已好）
```

## 3. 结论

1. **Canonical 结构化检索（时间+地点+别名）确定性胜出**：22/22 达到 1.00 召回，而 legacy 依赖 LLM 构建的 query/filters，随措辞波动（0.00~1.00）。
2. **地点别名解析是关键一环**：清迈 → Hang Dong 需要 alias 映射，说明 canonical intent 必须包含"规范化地点解析"（地名→存储值/实体）。
3. **与 P1 drift funnel 完全一致**：search 漂移（22/22 家族）正是 legacy 召回波动的直接原因；canonical 消除该漂移。
4. **P2 数据门通过**：按计划可进入 candidate 实现。

## 4. Candidate 实现要点（下一步）

- `answer_target + retrieval_target + structured constraints`：question →（确定性提取）→ 时间/地点/人物/关键词约束。
- search_memories 用 retrieval_target + constraints 构建检索，不再依赖 LLM 原始 query。
- 地点解析：地名 → 存储 place/geocode 别名/实体。
- 先 shadow 模式（legacy + canonical 双跑记录 recall），再切 candidate。
- 不与 0-result relaxation 同批。

## 5. 待报告矩阵（实现后）

AQ delta / angle AQ delta / retrieval delta / evidence availability delta / latency / extra model calls / regressions。
