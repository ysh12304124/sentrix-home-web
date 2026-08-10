# Agent Runtime v2 — Phase C/C10 Faithfulness v2 回归与指标 完成与测试报告

- 日期：2026-08-10
- 范围：C10（P0 场景固化为正式回归；Guard 指标：FP/FN/Recovery Success；place/meal 编造校验补全）
- 153 正式分支：`psh`（HEAD 含 `7698901` 合并）
- 本地工作分支：`psh-runtime-v2`（HEAD `7698901`）
- 验证基线：C9 报告提交 `b7a8c743`

## 1. 代码实现（4 个提交：`2d36e0b` + 3 个修复）

### 新增：正式回归集 `backend/tests/test_phasec_faithfulness_regression.py`
- **exists 类（P0-3/q03）**：exists=true 诚实/hedge 放行、明确否认拦截；exists=false 断言存在拦截、否认放行。
- **地点聚合（P0-1）**：「去年去过哪里」答案形态——真实城市+覆盖披露放行、编造城市拦截、partial 检索必须披露、整体否认拦截。
- **meal 总结（P0-2）**：explicit_foods 答案放行、编造食物拦截。
- **candidate upgrade**：candidate_only→full match 升级必须拦截、带披露的候选回答放行。
- **Guard 指标电池**（12 个用例）：FP=0（<=2%）、FN=0、Recovery Success=100%（脚本化 runtime 恢复用例）。
- 附带 4 个真实格式回归（子弹列表 `- 杭州市：150条记录`、内联括号 `（共100条记录）`、观察场景行、覆盖披露共存）。

### Guard 补全（`final_guard.py`）
- **place/meal 列举编造校验**：`_check_group` 对 `group_by=place` 用「去过/去了…」动词段 + 子弹列表提取列举项，对 meal 用 `fact_rows/fact_value` 的 food 行校验「吃过/喝了…」列举项，编造即 `group_fabrication`。
- **`没/未` 提前返回不再跳过编造检查**：修复「还有 N 张没有可靠地点信息」的覆盖披露让编造城市漏网的缺陷。
- **规则 4 否定上下文**：`确认是/确定是…` 前 4 字符内有 `没/未/不/还/难/无法/不能` 时不视为条件升级（修复「还不能完全确认是爬山」误伤）。
- **place 级否认**：`没有去过任何地方/没去过任何地方` 纳入 omission 否认集（限定量词，避免「还有没去过的地方」误伤）。
- 列举提取器：支持子弹列表、内联括号、剥离结构短语（以下地方/以下地点/的地方包括 等）黑名单。

### 线上指标脚本 `scripts/benchmarks/run_phasec_faithfulness_regression.py`
- 打 8091 生产跑 5 个 P0 场景，输出 completion / guard recovery success / disclosure。

## 2. 测试结果

### 后端（153 psh 工作树，全量，echo venv + hnswlib）
```
762 passed | 3 failed | 4 skipped（769 项收集）
```
- 3 个失败仍是已知事件归并 provenance 语义（用户已拍板不改），非本轮引入。
- 本轮新增 23 个用例（回归集 19 + phasec 扩展 4）。

### 前端
- 无前端改动；37 项结构性测试保持通过。

## 3. 线上指标实测（153 生产 8091，C10 代码已重启生效）

| 场景 | 状态 | recovery | l1_codes | 说明 |
| --- | --- | --- | --- | --- |
| q03 exists 2023年5月 | complete | 0 | - | 直接正确回答 |
| p0_1 去年去过哪里 | complete | 1 | group_fabrication | 首次编造城市被 L1 拦截 → 重写为真实 6 城 |
| p0_2 这两年吃过什么 | complete | 1 | omission_conflict | 恢复后列出 explicit_foods（饮料7/咖啡4/茶4…） |
| c8 去年十月爬山有雪吗 | complete | 1 | missing_disclosure | 分层披露 + 复核观察 |
| p0_4 去年春天去了哪里 | complete | 1 | placeholder_leak | 占位符被拦 → 重写为杭州100条 + 场馆记录 |

```
completion: 5/5
guard recovery success: 4/4 (100%)   # 目标 >= 90%
candidate→full match upgrade: 0      # 目标 = 0
Guard FP / FN（确定性电池）: 0 / 0    # FP <= 2%，FN = 0
```

## 4. 部署状态（153）

- 分支：`psh`，工作树干净；8091/8097/8098/4174 全部运行 C10 代码，health 正常。

## 5. 后续建议

- C11：Tool Schema Ergonomics——参数安全默认值（search 时间自动落到 filters.time、inspect 缺 handle 用 preview 首个、page 强制 int≥1、operation 枚举兜底）。
- C16（Product Reality E2E + 人工评审）：用户已确认本轮不做。
