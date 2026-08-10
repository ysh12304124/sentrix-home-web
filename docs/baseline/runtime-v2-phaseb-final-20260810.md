# Sentrix Agent Runtime v2 — Phase B Final Report

**日期：** 2026-08-10
**分支：** 本地 `psh-runtime-v2`（将合并进 153 `psh`）
**验证实例：** 8097/8098（tool_loop_shadow，模型 `gemma4-12b-it`@8105 4-bit，Chinese-CLIP 视觉，测试 DB `album2_e2b` 66 assets 只读）
**生产：** 8091 未动，默认 `pipeline`。

---

## 1. 执行总结

| 阶段 | 内容 | 结果 |
|---|---|---|
| B0 | 环境冻结 + Fresh Baseline | 前端 31/31；后端 692 测（7 挂 3 错为基线遗留）；修复 pipeline `recent_turns` 崩溃回归 |
| B1 | ANN 重建 + 检索就绪 | Chinese-CLIP 视觉全量 747 条；修复 `app.py` 未传 embedding_router（tool-loop 从未用 ANN）；recall@10 0.836 / MRR 0.764 |
| B2/B2.1 | search 真值合同 + Faithfulness Guard | `query_satisfaction/condition_summary/answerability` 确定性计算；omission/candidate/certainty/disclosure guard + 受控修正步 |
| B2.2/B2.3 | Faithfulness Benchmark + 序列化加固 | 40/40（100%）；repair 50%→13%（禁 markdown 围栏 + 去重）；guided_json 对照不采纳 |
| B3 | search→inspect→answer E2E | inspect_recall 1.0、不必要 inspect 0、10/11 complete（1 例 omission 被 guard 拦）；修复 tool_call_id 编号、sanitizer 剥离真值合同、handle 契约 |
| B3.1 | ResultSet 分页/持久化 | TTL 30min、全量稳定 handle、`get_result_page`、跨 turn 续接；unit 6/6、tool 10/10、model 3/3 |
| B3.2 | 原图交付 | 授权端点 `/api/assistant/result-set/{rid}/photo`（200/404/403 验证）；`get_original_photos` scope 校验修复 |
| B3.3/B3.4 | 前端交付 + 实时进度 | ResultSet 卡片（总数/剩余/下一页/缩略图）+ 原图；异步 turn（POST 进线程池 → GET 轮询），T2FP ~2-3s |
| B4 | 加固 | Emergency Renderer（预算/解析失败/guard 冲突→诚实摘要）；tool_readiness_matrix.json、agent_profile_manifest.json、structured_memory_coverage.json；canary telemetry 入库 |
| B5 | Canary 决断 | 结构化 canary 受控启动（见 §4）；图片 canary 暂缓（见 §5） |

## 2. L2 模型评审 Guard（新增分层）

- **L1 确定性规则**（FinalGuard）：零容忍项、权限、内部 ID、交付一致性，可证明、零成本。
- **L2 12B Judge**（judge.py）：L1 通过后对 `final + tool observations` 做语义级 faithful 判定，unfaithful 走一次修正步。
- 对照实验 18 例：bad_recall 90.9%、good_precision 100%、准确率 94.4%；漏检项均落在 L1 可证明范围内，分层并集覆盖全部反例。
- 运行期实证：si06“观察说没猫、回答猫是白色”、si05“观察多云、回答晴天”等编造被 L2 拦下并修正为忠实回答。

## 3. 关键实测指标（本阶段）

```text
Tool Selection（55 例）：schema 100% / primary 98.2% / unnecessary 0%
Faithfulness Benchmark：40/40
Search→Inspect E2E：inspect_recall 1.0、unnecessary inspect 0
Structured Canary：tool_selection 100%、count/first/last/exists/date/group 100%
ResultSet：pagination consistency 100%、原图交付 200/404/403 全对
Live Progress：T2FP ~2-3s（轮询 700ms）
Emergency Renderer：blocked/timeout 场景输出诚实摘要（无空回答）
```

## 4. 结构化事实 Canary 决断：**受控启动**

退出标准核对（§20.3）：

| 标准 | 结果 |
|---|---|
| Tool selection >= 95% | ✅ 100% |
| Structured fact accuracy >= 98% | ⚠️ 87.5%（8 例中 7 例；复合查询“照片+视频”模型编排不完整，被 guard 拦截，未出错） |
| Count / First / Last = 100% | ✅ |
| Safety critical errors = 0 | ✅ |
| Silent fallback = 0 | ✅ |
| Guard false-negative critical = 0 | ✅ |
| p95 within SLA | ✅ 7.6s |

**启动方式：** 独立实例（8098 类 `tool_loop_shadow`）+ 限定单意图结构化查询（count/exists/first/last/date/group by month），不做混合查询；每轮 telemetry（profile/tools/guard/latency/fallback）已入库；回滚 = 切回 `pipeline` 重启。

**本轮修复的真 bug：** `parse_time_expression` 只认 20xx 年份（1990 年返回 None 导致无过滤全量）；工具参数被模型包在 `arguments.schema` 时未展开；judge 输入缺 fact `value/operation` 导致 first/last 误报。

## 5. 图片类 Tool Canary 门槛逐项（§21）

| 门槛 | 状态 | 依据 |
|---|---|---|
| 1. ANN health = ready | ✅ | Chinese-CLIP 全量重建，manifest 可读 |
| 2. 图片检索不回退 | ✅ | recall@10 0.836 / MRR 0.764（优于 R8 记录） |
| 3. candidate_only 不再说成 full match | ✅ | L1 candidate_claimed_as_match + L2 judge，E2E 无通过 |
| 4. condition=unknown 必须披露 | ✅ | missing_disclosure guard + L2 |
| 5. false-positive fulfillment = 0 | ✅ | 空检索声称找到 = 0；unnecessary inspect = 0 |
| 6. search→inspect E2E 通过 | ✅ | inspect_recall 1.0；10/11 complete（1 例 omission 被 guard 拦 + emergency 摘要） |
| 7. ResultSet/分页/原图交付 E2E 通过 | ✅ | unit 6/6、tool 10/10、model 3/3、200/404/403 |
| 8. human helpful 达标 | ❌ 未测 | Phase B 未做人工体验评分 |

**决断：图片类 canary 暂不启动。** 结构性就绪已达标，但缺“体验级验证”（Human Helpful、Image Task Completion vs Canonical RX 对照）。此外 `search_memories` 的 place/person 语义覆盖仍 limited，语义质量子轨（§22）建议进入下一阶段。

## 6. 工具 Readiness 结论

| 工具 | 状态 | 说明 |
|---|---|---|
| query_memory_facts（count/exists/first/last/date/media/group） | ready（单意图） | 复合查询受限 |
| search_memories | visual ready / text limited | place/person 语义 limited |
| inspect_photo | ready | 低光/小物体计数有限 |
| get_original_photos / get_result_page | ready | scope+handle 双校验，TTL 30min |

## 7. 下一阶段建议

1. **结构化 canary 继续观察**（独立实例 + telemetry），先不收量。
2. **语义质量子轨**（§22）：_contains 失败类型、place/person 语义覆盖，作为图片 canary 的前置。
3. **模型优化子轨**（§23）：12B 对 filters 契约遵从不稳（复合查询、年份提取）是主要残留；constrained decoding / few-shot 可继续压 repair。
4. **Person/Core Memory/Write**：进入条件已记录（§24），Phase C 再评估。
