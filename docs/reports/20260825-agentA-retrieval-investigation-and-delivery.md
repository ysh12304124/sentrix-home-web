# AgentA 检索工具排查与 P0-P5 交付报告（2026-08-25）

## 结论

本轮确认“返回图片几乎完全不对”不是单一并发问题，而是检索链路与回答链路叠加造成的：地点元数据 JSON 未解析、显式地点/时间被错误放宽、事件锁定存在字段错误、候选窗口过浅，以及模型在候选中优先查看了错误图片。修复后，目标图片不再系统性丢失，最终 100QA 的真实候选召回均值从前一轮约 0.497 提升到 **0.750**；但回答层仍未达既定质量门，AQ=0.633、Exact=0.235、Core=0.398，Judge=0 的比例仍为 59/98。

因此当前状态是：检索基础设施和证据来源语义已完成一轮有效修复；“正确候选 → 正确事件 → 完整回答”的收敛仍是下一优先级，不能宣布整体质量达标。

## 已定位的根因

1. `assets.metadata_json` 在 `MetadataRetriever` 中按字典读取，但生产返回的是 JSON 字符串，地点预筛失效，导致整个相册进入 metadata 通道。
2. `_relaxed_retrieve` 在无严格命中时删除地点/时间条件，显式锚点被降级成了相似度提示，产生“赵州桥问题返回永年室内照片”等错配。
3. `_event_resolution_geo` 读取了未 SELECT 的 `summary` 字段，并把 `overlap_count` 写入了错误的事件解析函数，造成检索工具直接 `denied`。
4. 地点/时间检索的候选截断为 100，泛语义检索为 20；目标图片常在候选头之后，导致“检索库里有，但模型永远看不到”。
5. preview 为凑满 6 张图片，会用不满足场景线索的任意 `photo_1` 补位；模型随后对错误图片做 inspect。
6. 搜索 preview 中多个候选的人物字段被直接合并进回答；inspect 后仍可能把其他候选人物/地点带入最终答案。
7. Agent2 planner 只声明 `location_metadata`/`memory_asset` 时，authoritative final gate 会在视觉解析之前 block；模型即使拿到照片也可能没有机会 inspect。
8. `query_memory_facts` 的未解析人物过滤会退化为全相册统计，曾把“馆陶婚礼礼金”回答成 363。

## 已实施的修复

- `backend/retrieval/metadata.py`
  - 解析 `metadata_json` 字符串后再做 `reverse_geocode` 地点过滤。
- `backend/agent_runtime/canonical_intent.py`、`backend/geocoding.py`
  - 地点 token 采用问题中靠后的区县优先；增加赵州桥→赵县、三峡坝址→夷陵区的通用地理别名归一化。
- `backend/agent_runtime/tools.py`
  - 显式地点/时间不再在 relaxed retrieve 中丢弃。
  - 地点/时间锚定时阻断无直接地点证据的候选。
  - 修复事件解析 SQL/重叠计数；地点-only 事件锁定需满足独立语义重叠。
  - 候选窗口根据约束扩大到 500；预览仍限制在少量代表图。
  - 增加人数、婚礼、舞台、户外、夜间、水利等通用场景 cue 排序。
  - 明确场景/人物问题不再用任意图片填充 preview；人物/旅行/婚礼类问题推荐 inspect。
  - 未解析人物过滤直接返回 `unresolved_entity`，禁止全库 count。
- `backend/agent_runtime/runtime.py`
  - authoritative Agent2 也执行 `recommended_resolution`。
  - planner 未声明视觉 prerequisite 时，把 inspect 加入 gate 可用能力。
  - 模型忽略 inspect 时由代码对首个可见 handle 执行有界自动复核。
  - 保留检索候选、证据来源、最终展示三层集合。
- `backend/agent_runtime/tool_policy.py`、`result_set.py`
  - 保留服务端 source asset 映射和 preview asset 映射，但通过 `_model_visible_observation` 隐藏内部 ID，不泄漏给模型。
- `backend/agent_runtime/final_writer.py`
  - inspect 后只使用被 inspect 资产的人物/地点事实，避免候选污染。
  - 人物列表问题要求覆盖全部已确认人物，并明确未确认同行者。
  - 日期冲突、证据拒答、人物遗漏进入 deterministic completeness gate。
- `backend/tests/test_final_writer_evidence.py`、`backend/tests/test_retrieval_regressions.py`
  - 增加 JSON 元数据地点过滤、地点归一化、别名、人物完整性、候选污染回归测试。

## 153 验证结果

最终完整运行：`20260825-150835-album3-max-gemma4-12b-it-reuse-6c35db`，模型为 8100 上的 `gemma4-12b-it`，并发 12，100/100 完成。

| 指标 | 最终值 |
|---|---:|
| retrieval recall mean | 0.750 |
| retrieval recall=1 | 73/100 |
| answer quality mean | 0.633 |
| Exact | 0.235 |
| Core | 0.398 |
| Judge=0 | 59/98 |
| evidence source 非空 | 85/100 |
| selected delivery 非空 | 19/100 |

重点题：

- 002 婚礼年份：召回 1.0，回答 2017。
- 004 礼金：不再输出 363，返回无可确认结果。
- 052 赵州桥人物：召回 1.0，王建国、张晓莉均写入回答。
- 058 三峡坝址年份：召回 1.0，地点收敛到宜昌夷陵区并得到 2017 证据。
- 097 三人合影：目标 asset 进入候选并可被 inspect；直接回放已得到“我、明明、1 名未确认同行者”。
- 006/007/088：候选已包含 GT，但泛地点/泛语义场景仍可能选择错误代表图或无法从候选集合收敛到唯一事件，Judge 仍失败。

## 当前未完成项与下一优先级

P0 已从“目标图片丢失/工具拒绝”推进到“目标候选可追踪”。下一阶段不应继续盲目扩大召回，而应做事件级收敛和回答约束：

1. 对 `retrieved_candidates` 做事件级聚合与代表性排序，优先同一事件中满足人数/场景/时间的资产；泛“河北婚礼”“大型水利工程”不能把多个地点混成一个答案。
2. 对 `哪次旅行/哪次经历/具体地点/哪一年` 强制要求结构化时间、地点字段绑定到同一 source asset；仅有候选不允许 Writer 输出泛化句。
3. 将 `selected_delivery` 默认从 `evidence_sources` 生成 1–3 张，而不是依赖模型填写 `selected_image_handles`；用户端继续支持展开其余来源。
4. 建立“候选命中但回答错误”的专项 Judge 集合，目标是把直接证据充分题的 Judge=0 降到 20% 以下；当前尚未达到 AQ≥0.810、Exact≥0.320、Core≥0.490。

本报告对应的代码已同步到 153；本地定向测试 21 项、compileall 通过，前端 `vite build` 通过，153 服务已重启并使用最新代码。
