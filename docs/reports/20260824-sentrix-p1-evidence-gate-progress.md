# Sentrix P1 证据完成门控进度

日期：2026-08-24  
执行基线：`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`

## 本次落地

本次先落地 P1 的最小安全闭环：工具“被调用”不再等价于“证据已获得”。

- `read_photo_text` 只有在返回非空 `full_text`、`exact_values` 或 `text_regions` 时，才满足 OCR 需求。
- `inspect_photo` 只有在返回实际视觉 observation、且没有 blocked/失败状态时，才满足视觉需求。
- 兼容扁平化 TaskState 和旧的嵌套 replay trace；`status`、`reason`、`text_regions`、原始视觉 observation 现在会被保留到工具结果记录中。
- 既有 Agent2 planner 门控优先级不变；当 planner 没有映射需求时仍使用 legacy intent 兜底。
- `runtime._pending_resolution()` 现在按“是否产生可用证据”判断完成；失败工具不会被一次调用消费掉恢复机会。
- 确定性恢复只允许一次，并会跳过已经复核过的视觉候选；恢复预算耗尽后输出自然 partial，避免把未确认的 final 放行。

## 验证

在 153 上先运行新增测试确认旧实现会失败，再完成实现并运行通过：

- `test_completion_keeps_ocr_blocked_after_partial_tool_result`
- `test_completion_satisfies_ocr_only_when_text_was_returned`
- `test_completion_keeps_visual_blocked_after_empty_inspection`
- `test_completion_satisfies_visual_only_with_inspection_observation`

153 上回归结果：**91 tests OK**（既有门控、Agent2 shadow、runtime、ResultSet、planner 和 guard 测试集合）。

语法检查和 `git diff --check` 均通过；本次涉及的代码与测试文件已与本地副本逐字节核对一致。

代码验证后已按 153 安全 SOP 重启 8091；health 与 Qdrant Level-1 探测通过（430 collections、76740 points，未发现锁降级）。

## 尚未宣称完成的部分

这一步修复了“失败结果被误判为完成”以及恢复机会被失败调用消费的问题，但还没有实现完整的 `candidate → supported → confirmed` 重新规划闭环，也没有改变既有 retry 上限。下一步需要在 153 上用失败 OCR/inspect 的真实 trace 验证：

1. Agent2 requirement 是否进入 `partially_supported` 并触发新的取证决策；
2. 若模型再次输出 final，runtime 是否应选择下一个候选或重新调用 planner；
3. 失败次数、候选游标和预算约束如何避免无界循环。

Qdrant/SQLite 通道统一和 embedder 稳定性仍按 Wave 2 诊断报告后的 P1-B 方案推进，本次未改动。
