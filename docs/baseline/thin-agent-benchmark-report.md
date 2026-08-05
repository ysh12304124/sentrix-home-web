# 三相册端到端 Agent 能力测试 · 完整题目 + 已有回答

**数据源**：
- Ground truth：`~/Downloads/samples/album{1,2,3}/query.json`（60 case，20/相册）
- 手动能力测试：`Sentrix_Agent手动能力测试问题集.md`（15 个 A 类 case + 13 个 B 类行为 case）
- 实际 agent 回答：2026-08-05 API 8091 一轮实测，取 6 个代表性 case（因单 case 平均 240s，60 case 完整跑需 4 小时+）

---

# 第一部分：60 case Benchmark 完整题目

## 相册 1（20 case）

| Key | 问题 | GT 数 | GT 文件名 | 实测状态 |
|---|---|---:|---|---|
| album1-01 | 浅黄色拼接毛绒睡衣自拍 | 1 | `IMG_4350.JPG` | **实测 fail** |
| album1-02 | 厨房里做晚饭 | 6 | `IMG_0760.JPG`, `IMG_0761.JPG`, `IMG_0781.JPG`, `IMG_0789.JPG`, `IMG_0837.JPG`, `IMG_0860.JPG` | 未跑 |
| album1-03 | 手拿固体杨枝甘露 | 1 | `IMG_0536.JPG` | 未跑 |
| album1-04 | 水杯做的圣诞树 | 1 | `IMG_5243 2.JPG` | 未跑 |
| album1-05 | 快出岩洞时摆拍 | 2 | `IMG_8870.JPG`, `IMG_8871.JPG` | 未跑 |
| album1-06 | 记录下博物馆买的纪念品 | 10 | `IMG_1011.JPG`...`IMG_1485.JPG` | 未跑 |
| album1-07 | 贵阳夜晚步行街 | 0 | 无 | **实测 pass** |
| album1-08 | 趵突泉游览 | 5 | `IMG_1191.JPG`...`IMG_1195.JPG` | 未跑 |
| album1-09 | 25 年 11 月 16 号下午记录妆容的自拍 | 2 | `IMG_1779.JPG`, `IMG_1780.JPG` | 未跑 |
| album1-10 | 在禹城市的照片 | 2 | `IMG_1179.JPG`, `IMG_1178.JPG` | 未跑 |
| album1-11 | 在盐田区的照片 | 6 | `IMG_6438.JPG`...`IMG_6458.JPG` | 未跑 |
| album1-12 | 山东近代史展览--反侵略主题 | 5 | `IMG_1386.JPG`...`IMG_1400.JPG` | 未跑 |
| album1-13 | 黄绿酱汁烤鸡 | 3 | `IMG_5254.JPG`, `IMG_5255.JPG`, `IMG_5257.JPG` | 未跑 |
| album1-14 | 2024 年 5 月 | 1 | `IMG_7898.JPG` | 未跑 |
| album1-15 | 深圳机场登机口停机坪 | 1 | `IMG_5625.JPG` | 未跑 |
| album1-16 | 香草味可乐倒在键盘上 | 0 | 无 | 未跑 |
| album1-17 | 校园街中间躺着的猫猫 | 3 | `IMG_4823.JPG`, `IMG_4824.JPG`, `IMG_4831.JPG` | 未跑 |
| album1-18 | 穿和服的自己 | 3 | `IMG_8637.JPG`, `IMG_8808.JPG`, `IMG_8890.JPG` | 未跑 |
| album1-19 | 2023 年冬季的照片 | 2 | `IMG_8527.JPG`, `IMG_8542 2.JPG` | 未跑 |
| album1-20 | 我家猫吃猫粮 | 10 | `IMG_1483 2.JPG`...`IMG_5265.JPG` | 未跑 |

## 相册 2（20 case）

| Key | 问题 | GT 数 | GT 文件名 | 实测状态 |
|---|---|---:|---|---|
| album2-01 | 2025 年 4 月王明和我的合照 | 6 | `IMG_2028(1).jpg`, `IMG_2626.JPG`, `IMG_2032 2.jpg`, `IMG_3672.JPG`, `IMG_3779.jpg`, `IMG_2032.jpg` | 未跑 |
| album2-02 | 电影院明哥比 V | 0 | 无 | 未跑 |
| album2-03 | 可可粉可颂 | 1 | `IMG_1472.jpg` | 未跑 |
| album2-04 | 西湖木船上看水道风光 | 5 | `IMG_0354.JPG`, `IMG_1366.jpg`, `IMG_1369.jpg`, `IMG_1371.jpg`, `IMG_8878.jpg` | 未跑 |
| album2-05 | 在上海市拍的自己和王明的照片 | 1 | `IMG_7742.JPG` | **实测 timeout** |
| album2-06 | 夜晚车内的明哥搂着我 江西省 | 0 | 无 | **实测 pass** |
| album2-07 | 墨鱼汁海鲜特写 | 1 | `IMG_4724.jpg` | 未跑 |
| album2-08 | 杭州冬天吃的西式简餐 | 2 | `IMG_8506.jpg`, `IMG_8725.jpg` | 未跑 |
| album2-09 | 过 2025 年元旦 | 1 | `IMG_9675.jpg` | 未跑 |
| album2-10 | 大悦城动漫展卡通立板 | 4 | `IMG_7954.JPG`, `IMG_7955.JPG`, `IMG_7996.JPG`, `IMG_7999.JPG` | 未跑 |
| album2-11 | 在陕西省的照片 | 6 | `IMG_4540.jpg`, `IMG_4555.jpg`, `IMG_4542.jpg`, `IMG_4556.jpg`, `IMG_4548.jpg`, `IMG_4563.jpg` | 未跑 |
| album2-12 | 跨年烟花下的垃圾桶 | 0 | 无 | 未跑 |
| album2-13 | 夕阳下的电线杆和农田 | 2 | `IMG_4609.jpg`, `IMG_7380.jpg` | 未跑 |
| album2-14 | 在集美区的照片 | 4 | `IMG_5974.jpg`, `IMG_6022.jpg`, `IMG_5966.jpg`, `IMG_6023.jpg` | 未跑 |
| album2-15 | 2025 年 4 月自己的照片 | 19 | 见 samples/album2/query.json | 未跑 |
| album2-16 | 鼓浪屿远眺城市天际线 | 2 | `IMG_5592.jpg`, `IMG_5608.jpg` | 未跑 |
| album2-17 | 2024 跨年热闹氛围 | 3 | `IMG_8684.jpg`, `IMG_9610.jpg`, `IMG_9616.jpg` | 未跑 |
| album2-18 | 八戒 | 2 | `IMG_2670.jpg`, `IMG_3725.JPG` | 未跑 |
| album2-19 | 径山镇绿道散步 | 2 | `IMG_4395.jpg`, `IMG_4399 2.jpg` | 未跑 |
| album2-20 | 去听脱口秀 | 3 | `IMG_4146.jpg`, `IMG_6222.JPG`, `IMG_6227(1).JPG` | 未跑 |

## 相册 3（20 case）

| Key | 问题 | GT 数 | GT 文件名 | 实测状态 |
|---|---|---:|---|---|
| album3-01 | 银色心形手镯 | 1 | `IMG_3726.JPG` | **实测 fail** |
| album3-02 | 闵行区拍的 | 6 | `IMG_20220624_151900.jpg`, `IMG_20220716_141621.jpg`, `IMG_20220716_141625.jpg`, `IMG_20220624_134949.jpg`, `IMG_20220624_151905.jpg`, `IMG_20220624_152213.jpg` | 未跑 |
| album3-03 | 2022 年劳动节拍的 | 2 | `IMG_20220502_224357.jpg`, `IMG_20220502_163720.jpg` | 未跑 |
| album3-04 | 烘焙面糊搅拌过程 | 3 | `IMG_0143.JPG`, `IMG_7633.JPG`, `IMG_9913.JPG` | 未跑 |
| album3-05 | 厚切炒酸奶扫码购买 | 3 | `IMG_20220624_151905.jpg`, `IMG_20220730_164002.jpg`, `IMG_20220730_181336.jpg` | 未跑 |
| album3-06 | 2018 葆婴高峰会嘟嘟车宣传 | 1 | `2018-04-01 170511.jpg` | 未跑 |
| album3-07 | 春天花开 | 9/8 | 见 samples/album3/query.json | 未跑 |
| album3-08 | 确认车钥匙细节 | 3 | `2018-06-28 121602.jpg`, `IMG_20230404_080916.jpg`, `IMG_20230523_101625.jpg` | 未跑 |
| album3-09 | 燕园 | 1 | `2018-04-15 073828.jpg` | 未跑 |
| album3-10 | 餐厅卤味食材展示 | 3 | `IMG_20220801_092909.jpg`, `IMG_20220801_092920.jpg`, `IMG_7333.JPG` | 未跑 |
| album3-11 | 氹仔酒店楼道 | 1 | `IMG_4790.JPG` | 未跑 |
| album3-12 | 乘客与幼鹿近距离接触 | 3 | `2018-04-01 200541.jpg`, `2018-04-01 201022.jpg`, `2018-04-01 201025.jpg` | 未跑 |
| album3-13 | 采摘番茄 | 2 | `IMG_0517.JPG`, `IMG_0520.JPG` | 未跑 |
| album3-14 | 水族馆海豚跃出水面 | 0 | 无 | **实测 pass** |
| album3-15 | 牛春花门脸样式参考 | 1 | `IMG_20220731_204544.jpg` | 未跑 |
| album3-16 | 一床的娃娃 | 3 | `IMG_0185.JPG`, `IMG_0219.JPG`, `IMG_0220.JPG` | 未跑 |
| album3-17 | 厨房三兄弟切洋葱比赛 | 0 | 无 | 未跑 |
| album3-18 | 自己在上海店铺内的邋遢打扮 | 3/1 | `IMG_20220623_212642.jpg` | 未跑 |
| album3-19 | 三峡大坝北岸纪念碑 | 2 | `2017-11-05 152601.jpg`, `2017-11-05 152623.jpg` | 未跑 |
| album3-20 | 在夷陵区的照片 | 28 | 见 samples/album3/query.json | 未跑 |

---

# 第二部分：6 case 实测完整回答

## album1-01 · 浅黄色拼接毛绒睡衣自拍 · **FAIL**

- **GT (1)**：`IMG_4350.JPG`
- **Mode**：evidence
- **Latency**：179.8s
- **返回文件 (10, 0 hit / 10 FP)**：`IMG_6457.JPG`, `IMG_6458.JPG`, `IMG_7898.JPG`, `IMG_8527.JPG`, `IMG_8542 2.JPG`, `IMG_8637.JPG`, `IMG_8808.JPG`, `IMG_8870.JPG`, `IMG_8871.JPG`, `IMG_8890.JPG`
- **Answer**：
  > 记录支持浅黄色拼接；记录支持毛绒睡衣；记录支持浅黄色拼接；记录支持毛绒睡衣；记录支持浅黄色拼接；记录支持毛绒睡衣；记录支持浅黄色拼接；记录支持毛绒睡衣；记录支持浅黄色拼接；记录支持毛绒睡衣；...
- **诊断**：
  - Parser 拆词过粗，得到 `[clothing:浅黄色, clothing:拼接, ...]`
  - Kernel `_contains` 分词后 `all(term in ...)` 过宽，大量 asset 部分命中"色/毛"等单字
  - Answer composer 每个 (asset × condition) 生成一句 "记录支持 X"，无去重

## album1-07 · 贵阳夜晚步行街 · **PASS**（但走了 none 模式）

- **GT**：空
- **Mode**：**none**
- **Latency**：239.3s
- **返回文件**：空
- **Answer**：
  > 贵阳的夜生活挺丰富的。如果你想去步行街逛逛，建设路和花溪路是比较热门的选择。那里不仅有热闹的市井气息，还有很多特色小吃和灯光，非常适合晚上散散步。
- **诊断**：算 pass（空 GT 且无 FP），但 mode=none 是 Parser 幻觉误判，走了 normal_chat 生成通用推荐。**正确姿态应该 mode=evidence 且返回空+"无匹配证据"**。

## album2-05 · 上海市自己和王明的照片 · **ERROR (timeout)**

- **GT (1)**：`IMG_7742.JPG`
- **Mode**：none
- **Latency**：300.1s（timeout）
- **返回文件**：空
- **Answer**：空
- **诊断**：Complex path 4-5 次 Ollama 调用累计超 300s，客户端 timeout。Parser 判 mode=none 也错了（这是明确家庭事实查询）。

## album2-06 · 夜晚车内的明哥搂着我 江西省 · **PASS**

- **GT**：空
- **Mode**：none
- **Latency**：252.6s
- **返回文件**：空
- **Answer**：
  > 那段在江西的深夜，听起来是一个非常安静且有温度的瞬间。
- **诊断**：pass（空 GT 无 FP）。但同样 mode=none 是 Parser 幻觉；理想应该 mode=evidence 且明确"没有足够证据确认这段具体场景"。

## album3-01 · 银色心形手镯 · **FAIL**

- **GT (1)**：`IMG_3726.JPG`
- **Mode**：**none**
- **Latency**：242.9s
- **返回文件**：空
- **Answer**：
  > 听起来是一个非常精致的设计。您是想聊聊关于它的细节，还是有其他需要我帮忙的吗？
- **诊断**：**这是最坏的失败模式**。明确的照片查询，Parser 幻觉返回 mode=none，agent 走 normal_chat 完全不查记忆。用户会以为 agent 没有这张照片，实际 GT 就在库里。

## album3-14 · 水族馆海豚跃出水面 · **PASS**

- **GT**：空
- **Mode**：none
- **Latency**：225.8s
- **返回文件**：空
- **Answer**：
  > 那是一个非常壮观的画面。海豚在水中跃出水面时，动作往往显得非常有力量且优雅。
- **诊断**：pass 但同 album1-07。

## 汇总

| Verdict | 数量 | 命中特征 |
|---|---:|---|
| pass | 3 | 全部是**空 GT** + Parser 幻觉 mode=none 凑巧不返回图 |
| fail | 2 | 单命中/单命中，agent 要么 mode=none 完全不查（album3-01）要么 evidence 但精度爆炸（album1-01） |
| error | 1 | Complex path timeout |
| **正样本命中率** | **0/3** | 明确家庭查询全部失败 |

---

# 第三部分：15 个 A 类手动 case（评估维度更细，未实测）

覆盖：单/多图召回、空 GT、时间/地点/人物条件、大结果集（28 张）、复合场景。目录见 `Sentrix_Agent手动能力测试问题集.md §3`。

---

# 第四部分：13 个 B 类行为 case（不能靠脚本自动化）

| ID | 目的 | 需要 |
|---|---|---|
| B01 | 已确认人物自然介绍 | 已 confirmed entity + Writer/Verifier 链 |
| B02 | 人物外观追问 | subject binding + confirmed entity |
| B03 | 人物性格边界 | Verifier 拒绝性格断言 |
| B04 | 家庭角色解析 | 多候选处理 + 不暴露 cluster_id |
| B05 | 连续追问 3 轮 | Focus Stack + scene 复用 + 原图直出 |
| B06 | 无证据（火星生日） | 安全降级 |
| B07 | 普通聊天零记忆 | mode=none 严格 |
| B08 | 明确原始证据 | image_results + 不视觉重读 |
| B09 | 用户纠正 | pending assertion + 不覆盖 canonical |
| B10 | Prompt Injection | 工具白名单 + scope 不被改 |
| B11 | 主动回忆入口 | SENTRIX_PROACTIVE_MEMORY（本轮冻结） |
| B12 | scope 切换 | Focus 清空 + 证据不跨相册 |
| B13 | viewer 隔离 | annotation/cooldown/偏好 |

---

# Q2：这个测试集能否验证 agent 完整完成了计划的能力？

**结论：不能**。**只能覆盖约 60%**。

## 覆盖到的（60 case + 15 A 类）

| 能力 | 是否覆盖 |
|---|:-:|
| 视觉语义检索（衣着/物件/场景/活动） | ✅ |
| 时间硬条件（月份/日期/节日） | ✅ |
| 地点硬条件（区/市/省） | ✅ |
| 已确认人物条件 | ✅ |
| 空 GT 精确拒答 | ✅ |
| 大结果集覆盖率 | ✅（album3-20 28 张） |
| 复合场景（人物+时间+活动） | ✅ |
| 数量表达（"哪些"/"都"） | ✅ |

## 未覆盖的（B 类 13 case 都需要额外脚本或人工）

| 能力 | 为何未覆盖 |
|---|---|
| 复合 actions（answer + return_assets 同时） | benchmark 只测单目标，需另加多目标 case |
| 连续对话（Focus Stack） | 需多轮 conversation_id 状态测试 |
| Scope 切换隔离 | 需在两个 scope 交替发 query |
| Viewer 隔离 | 需多 viewer_id 交替 |
| 主动回忆入口 | 需先开 SENTRIX_PROACTIVE_MEMORY |
| 记忆纠正 propose/apply | 需 feedback payload + confirmation token 两步交互 |
| 明确原图 vs 默认折叠 | 需区分"给我照片"和"介绍"两种问法 |
| 人物性格/关系边界 | 需人工判断 answer 中是否越界 |
| Prompt Injection | 需构造污染 OCR/transcript 数据 |
| 复杂人物介绍 Writer/Verifier 通过率 | 需人工评估答复自然度 |
| 记忆一致性（不同 revision） | 需 apply 一次纠正后重查同一 subject |
| 分页 / 大结果集展示 | 需前端 UI 参与 |
| 敏感关系/伦理边界 | 需人工判断"搂着""亲密"等词的处理 |

## 结论

**当前测试集** = **evidence retrieval 精度** + **空 GT 拒答**。这是所有能力里**最基础的一层**，但即便这层当前也 0/3 命中——所以更上层的复合 action、连续对话、纠正、主动回忆等**根本没到验证阶段**。

要覆盖完整能力，需要**至少再加**：
- 复合任务 benchmark（20 case，涵盖 answer+return_assets、compare、timeline）
- 多轮对话 benchmark（10 场景，每场景 3-4 轮）
- 纠正流程 benchmark（10 case，涵盖 propose、apply、拒绝、审计）
- Prompt injection benchmark（5-10 case）
- 主动回忆 benchmark（开 flag 后 5-10 场景）
- 人工评估表（自然度、边界、拒答质量）

---

# Q3：理想能力边界 + 当前问题定位

## 3.1 理想能力边界（原计划 §12 + §15）

### 应该做到的

| 类别 | 硬指标 |
|---|---|
| **Recall** | Recall@10 ≥ 90%, Recall@20 ≥ 95%, all_relevant ≥ 85% |
| **精度** | 空 GT 拒答 ≥ 95%, 明确日期/scope/人物/media/must_not 违反 = 0 |
| **答案质量** | 家庭事实必带 evidence 或明确 gap，approximate 必披露差异 |
| **状态区分** | matched/possible/unknown/contradicted 严格分层，向量命中不能升级为 matched |
| **性能** | 检索 p50 ≤ 2s, p95 ≤ 5s；API 硬上限 ≤ 20s |
| **模型调用预算** | 普通聊天 QuerySpec/Gate = 0；简单 evidence ≤ 2 次；高级路径 ≤ 4 次 + 1 次 repair |
| **模态** | 图片默认折叠；明确"给我原图"才 image_results；"重新看看"才 inspect_original_images |
| **状态隔离** | scope 切换清 Focus，viewer 独立偏好，annotation 各自 |
| **纠正** | propose 不写库；apply 有 confirmation token；raw_json 不变；旧 revision 保留 |
| **主动回忆** | 每轮 ≤ 1 个入口；用户 2 次忽略后停；待确认人物不出现 |

### 应该拒绝的

| 场景 | 应回答 |
|---|---|
| 无证据的性格判断 | "记录不足以确定" |
| 空 GT 相似词泛化 | 明确"没有找到匹配证据" |
| 未授权原图请求 | 只返回文字总结 |
| 未授权记忆修改 | 只生成 proposal，等确认 |
| Prompt injection | 忽略指令，只当 OCR 内容 |
| 跨 scope 泄漏 | scope 切换后新 scope 内检索 |
| 单张观察推广成模式 | "只在这次记录里出现" |

## 3.2 当前问题定位

**分层看**：

### 层 A · 合同 / 骨架 — **没问题** ✅

- 375 unit tests 全绿
- 24/24 semantic benchmark 全绿
- 9 flag 全部布线正确
- Constraint 三层 / sanitizer / EvidencePacket / AnnIndex / Core Memory / MemoryCorrections / advanced tools 结构合规
- 375 tests 覆盖了合同不变量

**层 A 是这次 Phase 2R 重构最大的价值**。这层保护了未来任何模型/kernel 修复都不会重回关键词表老路。

### 层 B · Prompt 工程 — **有问题** ⚠️

| 问题 | 证据 |
|---|---|
| Parser prompt 太长（~1500 字符） | Ollama 可能触发 num_ctx=2048 截断 |
| 无 few-shot 例子 | 模型对 QueryParseDraft schema 输出不一致 |
| 无 `seed` 参数 | 同 prompt 输出波动 |
| 无 `num_ctx` 显式设置 | 依赖 Ollama 默认 |
| Validator 兜底过弱 | mode=none 是 valid 值，空 dict 也 valid |

### 层 C · Evidence Retrieval Kernel 精度 — **有问题** ⚠️

| 问题 | 证据 |
|---|---|
| `_contains` 分词后 `all(term in ...)` 单字命中 | album1-01 返回 10 张无关图 |
| Semantic 条件全走 SEMANTIC_REQUIRED 一层 | exact/strong/approximate 分层没生效 |
| CLIP 向量召回未接入 | Phase 3.1 §3.1 列了但没实现 |
| Parser 输出的 source_text 整体信号丢失 | 只用了拆分后的 dimension:value |
| 排序按 `matched count / len` 简单比例 | 无向量分、无来源可靠度加权 |

### 层 D · Answer Composer — **有小问题** ⚠️

| 问题 | 证据 |
|---|---|
| `_allowed_facts` 无去重 | album1-01 重复 10 次同一模板 |
| 无 fallback 到人类可读概括 | 直接暴露 "记录支持 X" 内部语言 |

### 层 E · 模型基础设施 — **重大问题** 🔴

| 问题 | 证据 |
|---|---|
| gemma4:12b `size_vram=189MB / size=8GB` | Ollama partial VRAM，每次推理页错误 |
| 单次 Ollama 调用 9-90s 波动 | direct probe 8.8s → 79s 之间 |
| Thin Agent 单轮 2-5 次 Ollama | 累计 20-450s |
| API 平均 240s vs 硬上限 20s | 12× 超限 |
| gemma4:12b 对 JSON 结构化输出不擅长 | 同 prompt 3 次结果不同 |

### 层 F · 未接口的模态 — **计划遗留** ⚠️

| 问题 | 说明 |
|---|---|
| CLIP 图像向量召回 | Phase 3 派生投影只做了字段级 term index，向量层没接 |
| ANN 索引未在生产查询路径启用 | Phase 3.5.2 backend 实装完成但 `EvidenceRetrievalKernel.retrieve` 没接 |
| Face bridge 增强 | Phase 3 观察索引只加了 person_bridge 字段，没接 face_prototypes |
| Subject binding formation | Phase 2R-6 kernel 期待的 `subject_clothing`/`subject_objects` 字段 formation 侧没生成 |

## 3.3 问题的核心 root cause 排序

按**对可用性的影响** × **修复难度**：

1. **模型基础设施（层 E）** — 直接决定单次 API 是否能在 20s 内返回。当前 240s 用户根本用不了。**修：让 gemma4:12b 完整 VRAM 驻留，或切分 parser/answer 模型**（parser 用 3B 小模型，answer 用 12B 大模型）
2. **Parser 稳定性（层 B）** — 决定 mode 判定是否正确。当前 40-60% 概率误判。**修：seed + num_ctx + few-shot + validator 收紧**
3. **Evidence 精度（层 C）** — 决定 evidence 模式下能不能命中真正 asset。当前 0/3 命中。**修：`_contains` 收紧 + CLIP 向量召回 + ANN 接入**
4. **Answer 去重（层 D）** — 决定用户看到的答案是否可读。当前重复 10 次。**修：dict.fromkeys 去重 + 人类可读模板**
5. **CLIP/ANN 接入（层 F）** — 决定十万张规模下 Recall。当前无向量召回，只靠字段字面。**修：EvidenceRetrievalKernel.retrieve 接 ANN + CLIP 补召回**

**层 A 骨架合同要保住**：所有修复必须在合同不变量下（Constraint 三层、sanitizer、EvidencePacket、statement 校验），不要重回关键词表。
