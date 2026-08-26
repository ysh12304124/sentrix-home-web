# Agent 2.1 W3.1 — OCR 专项调查报告

基线 run：`20260820-003839-agent2-1-ab` ｜ 对象：3 个 OCR 错误（q03/q24-02/q24-07）+ read_photo_text 延迟
> 纯分析，未改代码。核对了 read_photo_text 调用、answer_context 的 visible_text 原始输出、ocr_tool.py 配置。

## 1. 结论先行

| 维度 | 判定 |
| --- | --- |
| Provider | **默认是 VLM，不是设计里写的"PaddleOCR 小模型优先"**（`_ocr_provider()` 默认 `vlm`，`SENTRIX_OCR_PROVIDER` 未设时走 VLM） |
| Latency | VLM OCR 单次 **74~77s**（q26-01=76.98s、q24-02=74.19s）——正是延迟报告里的离群 |
| q24-02/q24-07 识别 | **全图 VLM 描述幻觉**（长春黄旗路 22048004），**切块 crop [tile_r0c0] 读对了**（长寿路 22048084/85）却在 context 里没被优先使用 |
| q03 沙雕文字 | **read_photo_text 根本没被调用**去读沙雕 → 工具选择缺口（不是 OCR 读错） |
| Fusion | 正确的 tile 结果与错误的全图结果同时进入 answer_context，无置信度/优先级区分 |

## 2. 详细证据

### 2.1 延迟：VLM OCR 是主成本
read_photo_text 共 8 次调用，其中 **2 次 74~77s**：
- `validation-album3-026-q01`：76.98s（顶呱呱创始年）
- `validation-album3-024-q02`：74.19s（店名/报警电话）

其余 6 次为 0.00s（说明走了缓存或未实际触达模型）。每次 VLM OCR = `[full_image]` 全图描述 + `[tile_r0c0]` 等切块 → 多次模型往返，单题耗时 74s+。

### 2.2 q24-02 / q24-07：全图幻觉，切块正确但没被用
同一张 江宁路单人留影照的 visible_text 同时包含：

```
[full_image] 蓝色长条形牌子：长春黄旗路派出所电话：22048004 22048005   ← 幻觉（长春/黄旗路/22048004）
[full_image] 红色大招牌：大兴 · 爱迪·油炸鸡                            ← 幻觉（真店名"大圣葱油拌面"）
[tile_r0c0]  长寿路派出所报警电话: 22048084 22048085 大 惠            ← 正确！但只是追加在尾部
```

- 正确答案 `22048084/85` **已经被切块 crop 读出**，却因为：
  - 全图 VLM 描述排在前、看起来更"权威"；切块结果无置信度标注、无优先级；
  - 最终 Writer 用了全图（错的）结果 → q24-02 答"大兴烧烤串儿"，q24-07 直接拒答（说看不出）。
- 归因：**recognition（VLM 全图幻觉）+ fusion（正确切块未胜出）**，不是"读不出来"。

### 2.3 q03：工具选择缺口
q03（沙雕主题名）的 tool_trace 只有 `search_memories` + `inspect_photo`，**没有调用 read_photo_text**。answer_context 里 visual_observation 是"照片中没有显示出沙雕上标注的文字内容"——是 inspect_photo 的 VLM 主观结论，不是 OCR。沙雕上的主题名从未被真正 OCR。
- 归因：**tool selection / planner**（该读文字的题没调用 OCR 工具），不是 OCR 质量。

### 2.4 Provider 配置：设计与实际不符
`backend/agent_runtime/ocr_tool.py`：
```python
def _ocr_provider() -> str:
    return os.getenv("SENTRIX_OCR_PROVIDER", "vlm").strip().lower() or "vlm"
```
- 设计说明"PaddleOCR 小模型优先，读不到回退 VLM"，但**默认值是 `vlm`**。small 引擎（`_get_small_engine()`）存在但未启用。
- 这解释了延迟（VLM 74s vs PaddleOCR CPU 秒级）与识别可靠性（PaddleOCR 对数字/招牌通常更准）。

## 3. 修复候选（按"先验证后改"原则，对应 W3.2）

| # | 候选 | 归属 | 预期 |
| --- | --- | --- | --- |
| A | 默认切到 small 优先 + 置信度门控回退 VLM | provider | 延迟 74s→秒级；数字识别更稳 |
| B | 切块 crop 结果优先于全图描述（或按区域置信度融合） | fusion | 修 q24-02/q24-07（正确 tile 已读出） |
| C | VLM 调用硬超时（~30s）+ 失败降级 | budget/latency | 压 76.98s 离群 |
| D | 工具选择：planner/能力矩阵识别"应 OCR 目标"时必调 read_photo_text | tool selection | 修 q03 类 |

## 4. 证据缺口 / 下一步验证
- 确认 run 是否真的走了 VLM（telemetry 未在 agent2_trace 中出现，需在 W1.2 重跑时把 OCR telemetry 纳入）。
- 候选 B 需要先复现"切块正确、全图错误"的原始 VLM 输出（raw_text 在 judge input 里，可提取复核）。
- 候选 A/D 都是 tool/capability 改动，属"可以改"授权范围，但按纪律先出 ROI 再立项。
