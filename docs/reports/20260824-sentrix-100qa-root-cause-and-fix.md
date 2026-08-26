# Sentrix 100QA 根因核验与阶段交付报告

日期：2026-08-24  
执行环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）  
权威基线：分支 `psh`，HEAD `37bf86e722383d735771a9764e258553717aec50`。初始核对确认本地副本与 153 不一致，后续以 153 工作区为准；未用本地版本覆盖远端。

## 结论

报告指出的两个主要方向均得到代码证据支持：

1. 图像/视频关键帧到记忆的链路存在信息压缩：旧图像提示词将 caption 限制为 20 字、对象/人物等各最多 2 项，视频事件默认只保留一个代表帧，检索预览也不携带丰富细节。
2. 检索工具存在结果集语义和展示层问题：`get_result_page` 会把新的 query/filter 当成旧结果集翻页；旧的全局 handle 回退会造成跨结果集误取；一次返回过多图片会淹没关键证据。
3. canonical 事件路径曾返回未持久化的伪 ResultSet：结果总数能显示，但后续 `inspect_photo`/原图交付无法续接；同时该路径会丢失原始问题的视觉复核意图。

报告中关于“必须先修检索/上下文，再评估 reranker”的优先级判断成立。reranker 暂列 P2，只有在结果集契约、细节记忆和评测闭环稳定后才有意义。

## 已实施改动

- 记忆细节：为 observation 增加版本化 `detail_json`，保留 caption、活动、地点、人物、对象、服饰、情绪、空间关系、OCR、事实和不确定性；图像/视频提示词扩大了受控字段上限；视频事件保留 `frame_observations`。
- 检索索引：细节字段进入倒排索引和文本 ANN 构建输入。
- ResultSet 契约：模型可见预览最多 6 个且按事件分散；不再暴露内部 `asset_ids`；新 query/filter 必须重新搜索；翻页最多 6 个；inspect/ocr 不再回退到全局旧 handle。
- 候选窗口：保留检索排序头部后再做事件多样性，并对已有候选的受控视觉 cue 做轻量重排；不改变服务端完整 ResultSet，避免用硬截断换取表面 precision。
- canonical 事件检索现在创建并持久化真实 ResultSet，保留原始 query/user goal，确保视觉问题能触发 `inspect_photo`，并支持后续分页/原图交付。
- 模型可见 observation 移除内部 `asset_ids`；完整映射仍保留在服务端 ResultSet 和 debug-only 评测链路，避免模型把候选全集当作已查看证据。
- 扩展视觉意图词表，覆盖“雕塑/石雕/景观/建筑/广场/喷泉/设施/装置”等对象型问题。
- 评测与维护：新增 100QA 根因审计脚本和 observation 细节回填脚本。回填仅复用已有原始/规范化字段，不调用模型、不凭空生成事实。
- 兼容性：`_timed_chat` 仅向支持的 chat 函数传递扩展参数，避免旧测试/调用方因关键字参数失败。

## 153 上的验证证据

- 新增 ResultSet 契约测试：6 passed（包含隐藏 ID 与视觉意图保持）。
- 检索/工具/运行时回归集合：36 passed。
- pipeline：20 passed。
- 视频相关集合：7 passed。
- 检索索引集合：21 passed。
- 全量 backend：691 passed，19 skipped，7 failed（其中 1 项通过新增评测映射相关覆盖增加）。

全量失败已分类：4 个属于未触碰的 event/evidence 合并语义；2 个属于 153 原有 Agent2 guard 脚本与既有断言不一致（本轮只修复了其 chat 函数兼容性）；1 个是 153 基线已有的 OCR 内部指标协议缺口。本轮核心 ResultSet、细节记忆、索引和视频改动的针对性测试均通过。

## 100QA 基线与当前阻塞

审计对象：`services/photobench/results/20260824-091104-album3-max-gemma4-12b-it-reuse-a8b848/run.json`。

- 100 题中只有 40 题有有效 judge 分数，不能把 40 题均值当作完整 100QA 指标。
- 29 题达到完整图片召回，其中 16 题仍为 AQ0（完整召回但回答没有利用关键图片）。
- 基线完整召回题中的 AQ0 比例为 16/29 = 55.17%。
- 发现多轮新 query/filter 被错误路由为旧结果集翻页，这是高优先级根因。

153 的模型入口保持为 `127.0.0.1:8100/v1`，本轮所有在线探针均通过该入口；当前 Manager 状态显示 8100 实际 serving profile 可能随 Manager 切换，不以端口号臆断具体模型名。生产 8091 的健康检查确认 Qdrant 主后端在线（当前约 430 collections、76740 points、`degraded=false`），但最近的 `visual_ann` 记录仍出现 SQLite backend，说明当前是“Qdrant 主后端 + 视觉通道回退”的混合状态，不能把健康标志等同于所有 ANN 通道均在 Qdrant 上运行。

固定输入差分（直接调用检索内核，隔离 Agent 模型波动）得到：

- Qdrant 多检索器：视觉 ANN 与文本 ANN 各返回 20 个候选；对“宜昌滨水纪念广场截流石雕塑”等语义问题，融合结果仍为 20 张，且跨年份/跨事件噪声明显。
- SQLite fallback：视觉索引报告 `incompatible`、文本索引 `uninitialized`，在相同输入下出现 0 召回或仅元数据候选；不能作为质量等价 fallback。
- canonical 时间/地点问题可以收敛到 2～12 张正确事件候选，但旧实现的伪 ResultSet 会让后续 inspect 失效；该问题已在本轮修复。

完整 100QA 已按 153/8100 链路完成多次诊断性复测。第一次发现 PhotoBench 提取器因契约变化把所有返回图片记成 0，结果作废；第二次运行在第 19 题后触发主机全局 OOM，也不计入质量指标；随后降低 Sentrix turn worker 数后完成第三次运行。

修复后的 run：`20260824-134140-album3-max-gemma4-12b-it-reuse-5dcf50`。

- `run_valid=true`，100 题 Agent 完成；Judge 有效 96/100，另有 4 题 Judge 记录无效。
- 图片召回：完整召回 18/96；有效题宏平均 recall 约 0.195，微平均 recall 0.158、precision 0.070。
- 完整召回题中 AQ0 为 10/18 = 55.56%，与基线 16/29 = 55.17% 基本一致；说明仅修工具契约没有自动改善回答利用证据的能力。
- 工具调用：`search_memories` 136 次、`get_result_page` 16 次、`inspect_photo` 44 次、`read_photo_text` 13 次。部分任务预览出现 10~20 张同事件或无关 `faceid_*` 图片，直接验证了“返回过多、关键图被淹没”的问题。
- 运行使用 8092 验证配置、并发 4，基线使用 8091/不同并发与索引状态；两次 run 不能作为严格 A/B 提升结论。当前结果用于确认根因和评测闭环是否有效，不作为最终质量门槛。

本轮第三次完成 run：`20260824-143818-album3-max-qwen3.5-0.8-lora-v2-reuse-970a65`。

- `run_valid=true`，100/100 Agent 与 Judge 完成；8100 实际 serving profile 为 `qwen3.5-0.8-lora-v2`，不能与历史 Gemma run 做模型质量 A/B。
- retrieval：宏平均 recall `0.616`；micro precision `0.118`、recall `0.406`、F1 `0.183`。
- Judge：有效 100/100；answer quality mean `0.76`，exact accuracy `0.33`，core accuracy `0.43`。
- 完整召回 59 题，其中 AQ0 30 题，比例 `30/59=50.85%`；该比例只作为当前 profile 的诊断值。
- 工具：`search_memories` 98 次、`inspect_photo` 14 次、`read_photo_text` 5 次。评测 debug 候选集合平均 12.75 张、中位数 20 张、最大 20 张，说明“服务端候选全集过宽”仍是独立的 P1 检索问题，即使模型可见预览已限制为 5 张。

OOM 证据：内核日志记录 14:33:21 全局 OOM，分别杀掉 8091（RSS 约 48.5GB）和 8771（RSS 约 2.5GB）。这不是 `qa_concurrency=12` 或 vLLM `max_num_seqs=12` 单一参数故障；真正的放大路径是 8091 默认 16 个 turn worker 与每个 turn 的视觉/嵌入资源叠加，批量并发把进程级内存推高。复测将 `SENTRIX_ASSISTANT_TURN_WORKERS=4`、PhotoBench QA/Judge concurrency=4，8091 RSS 稳定在约 19GB，100 题完成且无 OOM。该 worker 上限已固化到 153 的 `scripts/runtime/start_sentrix_api_8091.sh`，后续重启不会恢复到 16。

## 下一阶段候选窗口 A/B（同一 8100 profile、固定 scope）

为区分“完整候选召回”与“模型实际可见证据窗口”，在同一 qwen profile 和同一 album3 scope 上重放 5 个固定视觉问题，并通过 debug-only side channel 分别统计 full candidate IDs 与 preview IDs：

- 旧事件多样性策略：完整候选命中 4/5，preview 命中 1/5；关键图常在第 3、10、13、14 位，被前 6 张窗口排除。
- 保留检索前 3 名再做事件多样性：preview 命中提升到 2/5，证明事件去重不能覆盖相关性头部。
- 增加受控视觉 cue 重排（布置/婚房/展架/祝福/文字/雕塑等，仅重排已有候选，不扩大全库）：最终 preview 命中 4/5，完整候选仍为 4/5；5 题中 4 题触发了 inspect_photo。该结果只证明窗口策略改善，不等同于完整 100QA 质量提升。

本阶段同步修复了三个契约问题：

- debug projection 同时记录 `debug_asset_ids`、`debug_preview_asset_ids` 和 preview handles，模型仍不可见内部 ID；
- 模型误传 `photo_1` 等不在当前 preview 的 handle 时，运行时改用当前可见窗口首个 handle，避免重排后 inspect 错图；
- 模型生成的时间筛选必须与用户原话/query 中的显式时间一致；没有显式时间时丢弃模型自造的当前日期筛选。固定问题曾观测到模型错误传入 `2026年8月24日`，该值现已被后端过滤。
- completion guard 现在同时读取扁平化 `recommended_resolution`，不再因字段落在 tool-result 顶层而漏掉 `read_photo_text`；固定文字问题已验证会实际调用 OCR。

仍未解决的剩余问题：有些问题虽命中正确 preview，模型只调用 inspect_photo 而没有继续 read_photo_text；另有宜昌雕塑问题 full candidate 本身就未命中 GT。因此下一步应分别做 OCR 路由守门和视觉通道/文本通道召回差分，不能继续单纯扩大 preview 数量。

PhotoBench 映射修复采用 debug-only side channel：只有 `include_debug=true` 的评测响应包含 handle 对应的内部 asset ID，模型可见的 `task_state.tool_results` 和预览仍不含内部 ID。相关提取测试 36 项通过。

本轮新增验证：

- ResultSet/运行时/检索回归集合：39 passed；加入视觉意图与隐藏内部 ID 后的守门集合：45 passed。
- 使用 8100 的临时 8095 API 端到端验证 canonical 雕塑问题：真实 ResultSet 已生成；模型可见 `asset_ids=null`；问题触发 `inspect_photo`，并成功续接 `rs_*` 结果集。
- 临时 8093/8094/8095 服务及 Qdrant 诊断副本已停止并删除；生产 8091 未停止。

## 数据与回滚

- 153 数据库在 album3 细节回填前已备份：`/tmp/sentrix-db-before-detail-20260824-132618.db`。
- album3 已将 69 条 observation 的已有字段回填到 `detail_json`；没有新增模型事实。
- 初始工作区未提交改动已保留，基线补丁位于 `/tmp/sentrix-baseline-20260824-130635/worktree.patch`。
- 当前所有测试和改动均在 153 执行；未将本地副本作为验证依据。

## 下一步闭环

1. 当前 8091 已按 worker=4 重启，8771 已按 QA/Judge concurrency=4 重启；启动脚本也已固化该安全上限。
2. 8100 当前是 `qwen3.5-0.8-lora-v2`，本轮 100QA 已完成但不能与历史 Gemma 结果作严格质量 A/B；待固定同一模型 profile 后再冻结一次对照。
3. 对“候选全集过宽”做单变量 A/B：比较服务端完整 ResultSet 与模型证据窗口，分别记录 full-candidate recall、visible-window recall、inspect 触发率、AQ0 和跨事件污染；先不直接截断完整 ResultSet，以免损失长尾召回。
4. 在 A/B 结果稳定后，再决定数据库回填、Qdrant/视觉索引重建、候选去重/多样性重排及是否引入 reranker。
