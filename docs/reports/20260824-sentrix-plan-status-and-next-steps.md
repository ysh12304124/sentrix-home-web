# Sentrix 修复计划阶段状态与下一步报告

日期：2026-08-24  
权威环境：153（asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web），分支 psh  
验证模型入口：127.0.0.1:8100/v1

## 一、总判断

最早计划的“代码核对 → 根因验证 → 分层修复 → 153 验证 → 100QA → 报告交付”已经完成第一轮闭环，但还没有完成最终质量闭环。

基础契约、记忆字段、ResultSet、Agent 门控、候选窗口和运行稳定性已经完成首轮修复；同一模型 profile 下的严格 100QA A/B、完整候选召回修复、OCR 失败后的候选升级、视觉/文本通道治理仍未完成。

当前结论是：根因已经基本定位，关键结构性问题已修复一批，质量提升仍需下一轮严格评测确认。

## 二、最早计划完成情况

| 计划项 | 状态 | 结论 |
|---|---|---|
| 本地与 153 代码核对 | 已完成 | 已确认不一致，后续以 153 为唯一权威。 |
| 图像/视频记忆细节补全 | 已实施，质量验证未完成 | 已增加 detail_json、扩大提示词字段、保留视频 frame_observations、接入检索索引；尚未量化信息覆盖率。 |
| ResultSet/分页/handle 契约 | 已完成首轮 | 已修复伪 ResultSet、错误分页、旧 handle 回退和内部 ID 暴露。 |
| 检索候选窗口治理 | 已完成首轮 A/B | 已改为相关性头部优先、事件多样性和视觉 cue 重排；固定 5 题 preview 命中由 1/5 提升到 4/5。 |
| Agent 证据门控 | 已完成首轮 | 已有视觉/OCR/检索门控、FinalGuard 和有限 recovery；动态重规划未完成。 |
| 100QA 验证 | 诊断完成，严格 A/B 未完成 | 已完成有效 qwen 100QA，但不能与历史 Gemma 直接比较；最新改动尚未重跑完整 100QA。 |
| 稳定性/OOM | 已完成安全修复 | 8091 worker 固定为 4；根因是 worker=16 与视觉/Embedding 资源扇出，不是并发 12 单一参数。 |
| 报告交付 | 阶段报告已完成 | 根因报告和本状态报告已同步到 153；最终质量报告待严格 A/B。 |

## 三、发现的问题

1. 图像/视频到记忆存在信息压缩：caption、对象/人物上限过低，视频代表帧过少，OCR、空间关系和不确定性没有稳定进入索引。
2. canonical 事件路径曾返回伪 ResultSet，导致后续 inspect 和原图交付无法续接。
3. 新 query/filter 曾被错误当成旧结果集分页，全局旧 handle 也可能跨结果集误取。
4. 完整候选集常达 20 张，正确图片可能排在第 3、10、13、14 位；原事件去重会把答案排除在前 6 张之外。
5. 某些问题在完整候选阶段就未召回正确图片，例如宜昌雕塑问题。
6. 模型曾生成用户问题中不存在的时间筛选，例如 2026-08-24。
7. 模型重排后仍可能按示例选择 photo_1，而不是当前最相关 preview。
8. OCR 推荐字段曾因扁平化结构未被 completion guard 读取。
9. 当前 Planner 只在开始时声明一次需求，后续主要靠模型自行决定和规则兜底，不是真正的动态重规划。
10. 当前 requirement 更偏向“工具调用过”，不完全等于“证据相关且足够”。
11. OCR 失败后仍可能回退到错误候选。
12. Qdrant 主后端健康，但部分 visual_ann 仍出现 SQLite fallback，不能视为视觉和文本 ANN 都可靠。

## 四、已经解决的问题

### 记忆与索引

- 增加 observation detail_json；
- 扩大图像/视频描述字段；
- 保留视频多关键帧观察；
- 将细节字段接入倒排和文本 ANN；
- 完成 album3 既有 observation 的无模型事实回填。

### ResultSet 与检索

- canonical 路径创建真实持久化 ResultSet；
- 新 query/filter 不再伪装成旧结果集翻页；
- preview/page 限制为 6；
- 模型不再看到内部 asset_ids；
- debug-only 保存完整候选与 preview 映射；
- 保留相关性头部后再做事件多样性；
- 增加受控视觉 cue 重排；
- 重排后 handle 被限制在当前可见 preview 范围内；
- 模型自造时间筛选由后端丢弃或改为用户原话中的显式时间。

### Agent 门控

- 扩展视觉意图关键词；
- 修复扁平化 recommended_resolution 导致 OCR 需求漏判；
- 增加视觉/OCR/检索 Completion Gate；
- 保留 FinalGuard 与 Faithfulness Judge；
- 固定问题已验证会实际调用 OCR。

### 稳定性与验证

- 153 启动脚本永久设置 SENTRIX_ASSISTANT_TURN_WORKERS=4；
- 8091/8771 已按安全配置运行；
- 核心守门回归：76 passed；
- Python 编译检查和 git diff --check 通过。

## 五、100QA 当前能说明什么

有效诊断 run：

20260824-143818-album3-max-qwen3.5-0.8-lora-v2-reuse-970a65

- 100/100 Agent 与 Judge 完成；
- retrieval macro recall：0.616；
- answer quality mean：0.76；
- exact accuracy：0.33；
- core accuracy：0.43；
- 完整召回 59 题，其中 AQ0 30 题，占 50.85%。

该 run 使用 qwen profile，历史基线使用 Gemma，配置和索引状态也不完全一致，因此只能作为诊断数据，不能宣称本轮代码已经提升最终 100QA 质量。

## 六、下一步计划

### P0：固定当前版本，重跑严格 100QA

固定同一 8100 profile、同一 Qdrant、同一 album3 scope、worker=4、QA/Judge concurrency 和评测提取契约。

新增统计：full-candidate recall、visible-window recall、inspect/OCR 触发率、OCR 成功率、AQ0、跨事件污染、preview 平均候选数、evidence sufficiency failure rate。

### P1：完成 Agent 证据闭环

把“工具调用过”升级为“证据相关、可用、覆盖问题要求”：

1. OCR 失败后自动升级下一候选；
2. inspect 结果与问题关键词不匹配时分页/扩展候选；
3. 允许 Agent2 根据新证据重新计算缺口；
4. 区分 candidate、supported、confirmed；
5. 证据不足时重新规划，而不是只追加提示。

### P1：完成视觉/文本通道差分

对固定问题分别运行 visual ANN、text ANN、fusion、Qdrant 和 SQLite fallback，定位宜昌雕塑等完整候选未命中问题，再决定是索引、Embedding、融合权重、query 归一化还是 reranker。

### P2：量化记忆覆盖率

对固定图片/视频样本比较原图事实、detail_json、检索摘要和 inspect/OCR 可恢复事实，确认信息已经进入记忆和索引后再大规模重建。

### P2：严格模型 A/B 后决定 reranker

同模型、同数据、同配置比较当前排序、cue 重排、多样性重排和 reranker，不在严格 A/B 之前继续扩大复杂度。

## 七、最终交付门槛

1. 同模型 100QA 可重复完成；
2. 没有 OOM；
3. full-candidate recall 不下降；
4. visible-window recall 提升；
5. OCR/inspect 失败后有可控升级路径；
6. AQ0 和跨事件污染有明确改善；
7. 核心回归测试通过；
8. 报告明确区分诊断结论和严格 A/B 结论。

