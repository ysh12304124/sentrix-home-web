# Agent 2.1 W5.1 — V1 视觉错误拆解报告

基线 run：`20260820-003839-agent2-1-ab` ｜ 对象：5 个 V1 错误（q08/q47-01/q47-03/q47-04/q47-07）
> 纯分析，未改代码。逐题核对了 inspect_photo 调用（asset_handle / question / VLM 返回）。

## 1. 结论先行

**V1-A（模型真看不见）≈ 0~1/5 → 明确不换 VLM。**

| 题 | 分类 | 证据 |
| --- | --- | --- |
| q08 明明上衣颜色 | **V1-E identity binding** | inspect photo_1，VLM 返回"照片中没有'明明'身份信息"——是 bind 不上人，不是看不出颜色 |
| q47-01 表演地点 | **V1-D 问题表述/证据源错** | 问 VLM"照片里有没有地点信息"——夜景表演照片本就没地点牌；答案应来自 EXIF/geocode（已给出 Hang Dong）+ event 数据 |
| q47-03 开场道具 | **V1-B 查错 asset** | inspect 的 asset_handle=`video_1`；离线 judge 明确"第四张图能看到火把/圆形道具" |
| q47-04 标志植物 | **V1-C 覆盖不足**（或 V1-A） | 只 inspect photo_1；离线 judge 称"四张图都能看到棕榈树" |
| q47-07 持火把照片 | **V1-C 覆盖不足 + V1-B** | 只 inspect photo_1；火把在第四张（gt=1，f1=0.4） |

计数：V1-A=0~1、V1-B=2、V1-C=2~3、V1-D=1、V1-E=1。

## 2. 共性根因：inspect_photo 默认只看预览第一张

`tools.py` 的 inspect_photo 描述：`asset_handle 可省略（默认用预览第一张）`。5 个 V1 案例里 **4 个（q08/q47-01/q47-04/q47-07）都只 inspect 了 photo_1**，q47-03 还 inspect 成了 video_1。

问题集中在：
1. **"哪一张 / 覆盖类"问题只查第一张**（q47-03/q47-07）：答案在第四张，Agent 查第一张找不到就拒答。→ inspect_photo 应支持**多候选 asset 遍历**（或对"哪张有 X"类问题强制遍历全部候选）。
2. **首张查不到就放弃**（q47-04）：judge 看到四张都有棕榈，Agent 只查 photo_1 说没有。→ 首张无果时应继续查其余候选。

## 3. 逐题详解

### q08（V1-E identity binding）
- inspect photo_1 + question "叫'明明'的人穿的上衣颜色？"
- VLM 返回"无法确定'明明'身份"。**颜色本身大概率可见**，卡在把"明明"绑定到图中人物。
- 修复方向：inspect_photo 的问题里带上已确认人物线索（如 face cluster/事件里的人物位置）；或与 get_person_memory 结合。

### q47-01（V1-D 问题表述/证据源）
- inspect photo_1 + "照片里是否有地点信息？" → 夜景表演照无地点牌，必然"没有"。
- 正确答案来自 **location_metadata（Hang Dong）+ 事件语义（清迈夜间动物园）**，不该靠 inspect_photo。
- 修复方向：Planner 对"在哪举办"类问题应优先 location_metadata/event 检索，而非视觉 inspect。

### q47-03 / q47-07（V1-B 查错 asset + V1-C 覆盖不足）
- q47-03 inspect 了 `video_1`（video handle），火把在第四张照片。
- q47-07 只 inspect photo_1，火把在第四张。
- 修复方向：inspect_photo 遍历全部候选 asset；"哪一张"类问题必须遍历。

### q47-04（V1-C 覆盖不足）
- 只 inspect photo_1；离线 judge（不同模型）在四张图都看到棕榈。可能 V1-A（gemma4 漏看）或覆盖不足。待 W1.2 重跑时用多 asset 遍历复测区分。

## 4. 对 W5.2 的决策建议

- **不换 VLM**（V1-A ≈ 0~1/5）。
- 低成本高价值：**inspect_photo 多候选遍历**（V1-B/C，涉及 4/5 案例）——tool 层改动，属"可以改"授权。
- identity binding（q08）与证据源选择（q47-01）属于 planner/人物绑定，另立小项。
- 建议在 W1.2 stage-timer 重跑时，顺带验证 inspect_photo 多 asset 后的效果（同一 run 复测 5 个 V1 案例）。
