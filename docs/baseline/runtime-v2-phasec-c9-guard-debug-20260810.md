# Agent Runtime v2 — Phase C/C9 Guard/Debug UX 分层 完成与测试报告

- 日期：2026-08-10
- 范围：C9（普通用户只看到自然文案；管理员可看 L1/L2 规则码、恢复步数、最终状态）
- 153 正式分支：`psh`（HEAD `0bd224e3`，由 `9cfad48` 合并组成）
- 本地工作分支：`psh-runtime-v2`（HEAD `9cfad48`）
- 验证基线：C8 报告提交 `590ed079`

## 1. 代码实现（commit `9cfad48`，+93/-1 行）

### 后端：结构化 guard debug 数据

**`backend/agent_runtime/runtime.py`**
- 每次 `guard.check` 后记录结构化步骤：`{type: "guard", status: pass|fail, codes: [L1 规则码], attempt: N}`——L1 失败与恢复后的第二次检查各一条，恢复次数天然可见。

**`backend/app.py`**
- turn 结果新增 `guard_debug` 字段：
  ```json
  {
    "status": "complete|blocked_by_guard|...",
    "reason": "",
    "recovery_attempts": 1,
    "l1_codes": ["missing_disclosure"],
    "judge": [{"faithful": true, "problems": []}]
  }
  ```
- `retrieval_trace` 的 guard 步骤带 `detail={l1_codes, attempt}`，judge 步骤带 `detail={faithful, problems}`（L2）。

### 前端：两套展示明确分开
- 普通用户：只看到 `public_progress` 自然文案（「正在核对结果…」「正在整理回答…」），与 C2/C3 一致，**不出现任何规则码**。
- 管理员（URL `?debug=1` 或 localStorage `sentrix.adminDebug=1`）：新增 `Guard 校验明细` 折叠块（默认展开当 `blocked_by_guard`）——L1 规则码、L2 评审逐次 faithful/problems、恢复步数、最终状态。

## 2. 测试结果

### 后端（153 psh 工作树，全量，echo venv + hnswlib）
```
739 passed | 3 failed | 4 skipped（746 项收集）
```
- 3 个失败仍是已知事件归并 provenance 语义（用户已拍板不改），非本轮引入。
- 新增 `GuardDebugTraceTests.test_guard_recovery_steps_recorded`：脚本化 chat_fn 走完整 tool-loop——L1 拦截（`fact_exists_contradiction`）→ 恢复 → 第二次 guard 通过 → L2 faithful；断言 guard 步骤 status/codes/attempt、recovering 进度事件、恢复后回答为可信事实。

### 前端（本地）
```
37 tests | 0 failures
```
- 新增结构性断言：`guardDebug` 函数、`Guard 校验明细`、`l1_codes`、`L2 评审`、`恢复步数`、`adminDebug()` 门控。

## 3. 线上 QA 实测（153 生产 8091，C9 代码已重启生效）

| 问题 | 结果 |
| --- | --- |
| 2023年5月拍过照片吗？ | complete；`guard_debug`：recovery_attempts=0、l1_codes=[], judge=[{faithful:true}]；trace 一条 guard pass |
| 去年十月爬山的照片山上有雪吗？ | complete；`guard_debug`：recovery_attempts=1、l1_codes=["missing_disclosure"]；trace 两条 guard 步骤（attempt1 fail → attempt2 pass）；最终回答分层披露 + 复核观察 |

- Q2 同时验证了 C8 规则在真实轮次的拦截-恢复闭环，且 C9 让管理员能看到完整的 L1 代码与恢复步数，普通用户仍只见自然进度文案。

## 4. 部署状态（153）

- 分支：`psh`（`0bd224e3`），工作树干净。
- 8091/8097/8098/4174 全部重启到 C9 代码，health 正常。

## 5. 后续建议

- C10：Faithfulness v2 回归集——把 exists/place/meal/guard-recovery 固化为正式 regression 模块与指标（Guard Recovery Success、Hard fact wrong final=0、candidate_only→full match=0）。
- C11：Tool Schema ergonomics（参数遵从与安全默认值）。
