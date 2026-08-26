# Sentrix Agent2 P0–P5 完整执行与交付报告

日期：2026-08-24  
权威环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）  
模型：8100 当前服务的 `qwen3.5-0.8-lora-v2`，vLLM `max_num_seqs=12`

## 1. 总结

P0–P5 已完成一轮闭环：代码核对、P0 契约修复、P1 证据失败恢复、P2 通道和候选诊断、P3 记忆覆盖审计与结构化回填、P4 单变量基线评估、P5 严格 100QA 验证和报告交付均已执行。

当前可以确认的是：系统的“代码正确性、证据状态、轨迹完整性、服务健康性”明显改善；不能确认的是：整体 100QA 质量已经得到统计显著提升。最新 100QA 的主要瓶颈已从隐藏的代码契约问题，收敛为候选治理、视觉复核命中率和原始图像语义覆盖不足。

## 2. P0：代码级契约与轨迹一致性

### 已解决

- `search_memories` 空结果不再覆盖已有可用 ResultSet。
- `search_memories` preview 明确返回描述可用/缺失状态；缺失不再静默省略。
- `photo_N` 只保留在内部工具/调试链路；普通 final、writer、guard recovery 等最终回答路径统一清理。
- 运行时从 `debug_trace` 补全工具轨迹，保留真实 arguments、observation、`tool_call_id`、`step_id`、`parent_step_id`。
- 修复了 `backend/app.py` 中局部 `trace` 变量覆盖完整 execution trace 的问题；这是造成 8771 顶层轨迹为空的直接代码原因之一。
- 前端证据类型改用规范字符串映射，工具编号、模型步骤编号、证据类型和 judge 分数分开显示；不兼容证据明确显示“非当前需求证据”。
- runtime 注入的 recovery/completion 消息保留 API 的 `role=user` 兼容性，但轨迹标记为 `message_origin=system_recovery`。
- 时间表达式对未知相对词不再强行当作严格日期过滤。

### P0 验收

最新 P0 10QA smoke（run `20260824-212848-album3-14-qwen3.5-0.8-lora-v2-reuse-23b244`）中：工具步骤 16、execution steps 71、工具 observation 缺失 0、未绑定 0、`photo_N` 泄漏 0、描述缺失 0。服务重启后 153 Level-1 检索探测通过。

## 3. P1：证据完成门控与失败恢复

### 已解决

- “工具被调用”与“证据已获得”分离：OCR 只有实际文字/区域/结构化值才满足；视觉复核只有实际 observation 且非失败状态才满足。
- 失败或 partial 结果保留 requirement 的 failed/running 状态，不会把未确认回答放行。
- 增加 bounded recovery：视觉/OCR 失败时只在当前 preview 内选择尚未复核的下一张候选，不扩大搜索范围、不把失败结果当证据。
- 修复自动恢复路径的轨迹缺失：此前自动工具调用只更新 TaskState，现已统一写入 execution trace、TaskState、Agent2 ledger 和模型可见 observation。
- 新增回归测试验证 `photo_1` 失败后会切换 `photo_2`，且自动步骤带 `auto_resolution=true`。

### 运行验证

最新严格 100QA 没有自然触发 failed visual/OCR 分支，因此自动恢复由合成 runtime 测试验证；不能把“本轮自动恢复次数为 0”误读成恢复代码未覆盖。

## 4. P2：检索通道与候选治理

### 已解决/已确认

- 8101 BGE 文本 embedding health 和直接 embed 正常。
- 8091 线上检索保持 Qdrant 后端；最新 100QA 69 次采样均为 `visual_ann/text_ann ready, backend=qdrant`。
- 增加 health probe 短 TTL 缓存，避免批量 QA 重复健康请求造成延迟和假性不可用。
- `candidate_window` 诊断字段已接入，记录候选总量、可见窗口、rank、事件组分布及策略。

### 尚未解决

最新 100QA 仍有多道题返回 20 个跨事件候选，模型随后泛化作答或复核错误图片。根因是候选治理/视觉确认命中率，不是并发 12 的问题。下一轮应保持模型和 ANN 不变，单变量比较 `head-only`、`head+event-diversity`、`head+visual-query`，再决定是否引入 reranker。

## 5. P3：原始图像/视频记忆覆盖

### 已完成

运行只读审计并完成结构化回填：

- 资产：14,596；观察：17,201（图片 14,589、视频 7）。
- `detail_json` schema v1：17,201 / 17,201，覆盖率 100%。
- 已重建检索索引。

### 仍存在的信息缺口

结构化回填只能整理已有 raw/canonical 字段，不能凭空生成视觉事实。当前字段覆盖率：caption/activity 58.98%、place 93.52%、people 30.79%、objects 57.33%、clothing 11.72%、spatial_relations 13.41%、OCR 27.56%、transcript 0%；有索引词的观察占 56.09%。

因此，用户提出的“从原始图像/关键帧尽可能描述全部细节”仍是后续 VLM re-enrichment 任务，而不是本次结构化回填已解决的问题。

## 6. P4：单变量模型/检索验证

严格 100QA run：`20260824-213801-album3-max-qwen3.5-0.8-lora-v2-reuse-2f9bc0`。

| 指标 | 最新值 |
|---|---:|
| 完成 | 100/100 |
| retrieval recall mean | 0.618 |
| retrieval precision micro / recall micro / F1 | 0.108 / 0.468 / 0.175 |
| answer quality mean | 0.758 |
| exact accuracy | 0.303 |
| core accuracy | 0.455 |
| judge valid | 99/100 |
| JSON parse success | 0.887 |
| QA completion within steps | 0.24 |
| photo handle leak | 0 |
| unresolved trace binding | 0 |

与同模型此前基线（recall 0.616、quality 0.760、exact 0.330、core 0.430）相比，整体质量没有形成可宣称的显著提升：recall 基本不变，answer quality 持平，exact 有回落，core 略升。结论是本轮修复主要改善了可观测性和正确性边界，尚未改善端到端答案质量。

## 7. P5：最终验证和交付

### 153 验证结果

- 定向回归：98 tests OK。
- 关键文件 py_compile：通过。
- photobench 前端 `npm run build`：通过。
- 8091 安全重启：通过；Qdrant 346 collections、45,330 points，锁状态正常。
- 严格 100QA：100/100 完成，审计报告已保存为 `docs/reports/20260824-p5-100qa-audit.json`。

### 交付文件

- 本报告：`docs/reports/20260824-sentrix-p0-p5-final-delivery.md`
- P2 通道/候选诊断：`docs/reports/20260824-sentrix-p2-channel-candidate-diagnostic.md`
- 记忆覆盖审计：`docs/reports/20260824-memory-coverage-audit-after-backfill.json`
- 100QA 审计：`docs/reports/20260824-p5-100qa-audit.json`
- 既有 P0 契约计划和 P1 证据门控报告继续保留，作为变更依据和测试索引。

## 8. 下一阶段需要你确认的三个产品决策

1. 是否批准 P6 对全部图片/视频关键帧做 VLM 级重描述与增量回填？这会显著增加处理时间和 GPU 成本，但它是解决 caption/objects/clothing/spatial/OCR 缺口的直接路径。
2. 候选治理是否采用“服务端保留完整 ResultSet、模型只看到分阶段 bounded window”的方案？这是当前推荐方案；它既不丢失可分页结果，又避免一次把 20 张混杂图片塞给模型。
3. 8100 的 qwen3.5-0.8-lora-v2 是否继续作为固定基线，先完成候选窗口 A/B，再与其他模型比较？建议先固定模型，避免把候选问题和模型差异混在同一实验中。

