# Agent Runtime v2 — Phase C/C11 Tool Schema Ergonomics 完成与测试报告

- 日期：2026-08-10
- 范围：C11（工具参数安全默认值与遵从：operation/mode/handle/page 兜底、时间自动提取、schema 条件约束说明）
- 153 正式分支：`psh`（HEAD 含 `5a680a9` 合并）
- 本地工作分支：`psh-runtime-v2`（HEAD `5a680a9`）
- 验证基线：C10 报告提交 `d91af4a9`

## 1. 代码实现（commit `5a680a9`，+105/-7 行，仅 `tools.py` + 测试）

### 参数安全默认值（C11）
- **`query_memory_facts`**：`operation` 枚举校验（非法回退 `count`）；模型只填 `group_by` 未填 `operation` 时自动补 `operation=group`；`operation=group` 缺 `group_by` 默认 `month`。
- **`search_memories`**：`mode` 只接受 `best|all|representative`（非法回退 `best`）；模型把时间写进 `query` 文本而 `filters.time` 为空时，自动提取（`20xx年[月]`、`去年X月`、`去年/今年/前年/这两年/上个月/去年春天…`）并落入 `filters.time`（`_extract_time_from_query`）。
- **`inspect_photo`**：`asset_handle` 可省略，默认取当前结果集 preview 首个 handle（先解析当前结果集，再回退 last_handles）。
- **`get_result_page`**：非法 `page/page_size` 不再整单拒绝（`bad_page_args`），改为安全默认（page=1、page_size=6）。
- **`get_original_photos`**：原有行为（缺 handle 回退首个资产）保持不变。
- **schema 描述**：明确 `operation=group 必须填 group_by`、`search 时间必须填 filters.time（忘填会自动提取）`、`inspect handle 可省略（默认预览第一张）`。

## 2. 测试结果

### 后端（本地 + 153）
- 新增 `ToolSchemaDefaultsTests`（6 用例）：group_by 推断 operation、非法 operation 回退 count、group 缺省 month、query 时间提取（年份/相对/去年十月/无时间）、inspect handle 默认解析、分页非法参数兜底。
- 相关文件 73 个测试全过；本地全量与基线一致（仅环境相关失败：缺 hnswlib/GPU/vLLM）。

### 前端
- 无改动，37 项结构性测试保持通过。

## 3. 线上回归（153 生产 8091，C11 代码已重启生效）

复用 C10 Faithfulness 电池（5 个 P0 场景）作为冒烟：
```
completion: 5/5
guard recovery success: 4/4 (100%)
```
- q03 exists / 去年去过哪里 / 这两年吃过什么 / 去年春天去了哪里 全部 complete（含 guard 拦截-恢复闭环）。
- c8 爬山场景本轮模型先给 empty-ref 回答被拦后恢复完成（模型方差，非代码回归；无候选时"没有找到"为诚实回答）。

## 4. 部署状态（153）

- 分支：`psh`，工作树干净；8091/8097/8098/4174 全部运行 C11 代码，health 正常。

## 5. Phase C 剩余与收尾

- C16（Product Reality E2E + 人工评审）用户已确认不做。
- 本阶段（C1-C11 + C12/C13-C15）全部完成并上线；下一步建议：Person/Core Memory Tool、Semantic Evidence、或 Model optimization（见 Phase C DoD §24）。
