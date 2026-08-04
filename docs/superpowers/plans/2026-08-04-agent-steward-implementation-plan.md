# Agent 家庭记忆助手实施计划

## 目标

在不修改记忆生成管线的前提下，完成 Agent 核心契约、分层记忆调用、连续对话、必有证据展示、原始证据直出、反馈治理和验收调优。

## 架构

`backend/agent.py` 负责意图、计划校验、工具执行、状态和证据排序；`backend/app.py` 只负责 assistant API 边界；`src/api.js` 与 `src/app.js` 负责响应消费和证据交互；现有 MemoryStore 作为记忆读取权威来源。记忆生成、模型适配和重建保持不变。

## 技术栈

Python `unittest`、FastAPI 现有 assistant 路由、Node test runner、原生浏览器 JavaScript。正式后端提交只在 153 `psh`。

## B：Agent 核心契约

1. 在 `backend/tests/test_agent.py` 增加失败测试：记忆回答必须返回 `memory_used`、非空证据或明确 `query_gap`。
2. 运行 `PYTHONPATH=. .venv/bin/python -m unittest backend.tests.test_agent.AgentEvidenceContractTests`，确认因契约未实现而失败。
3. 在 `backend/agent.py` 统一 memory/chat/feedback/clarify 响应字段，禁止无证据事实回答。
4. 重跑新增测试和原有 Agent 测试，确认通过。
5. 增加失败测试：普通聊天不能执行记忆读取；记忆计划只能使用白名单工具；非法模型计划必须走确定性 fallback。
6. 实现计划校验、工具白名单和 `scope_id` 传递，不修改记忆生成模块。
7. 增加失败测试：连续追问继承当前人物/事件，但切换 `scope_id` 后清除旧状态。
8. 实现有界对话状态和澄清状态恢复。
9. 增加失败测试：向量候选没有文本/事件锚点时不能成为证据。
10. 实现证据排序和证据层级绑定，保持现有检索 API 兼容。
11. 运行完整 Python Agent 测试和 Node 基线，记录结果。

## A：完整产品场景

1. 为人物介绍、时间线、比较、推荐、澄清和原始证据请求分别增加失败测试夹具。
2. 实现每条场景路径的最小工具组合：实体介绍优先语义层，时间线优先事件层，图片请求进入原始证据层。
3. 增加失败测试：所有记忆响应都有折叠证据入口；明确原始证据请求返回媒体 URL/Asset，而不是只返回证据摘要。
4. 更新 `backend/app.py` 和 `src/api.js` 的响应适配，保留旧字段并补充证据状态。
5. 更新 `src/app.js` 与 `src/styles.css`：记忆回答始终显示证据入口；原始证据请求直接展示原图/媒体；普通聊天不渲染记忆证据区。
6. 增加反馈测试：没有明确目标或确认时不写入；有目标并确认时使用现有审计入口。
7. 完成一条纵向流程回归：人物介绍 -> 连续追问 -> 事件 -> 原始照片。
8. 运行 Python、Node、语法检查和 `git diff --check`。

## C：问题集和调优

1. 在 Agent 测试夹具中建立普通聊天、事实查询、追问、时间线、比较、推荐、澄清、无证据、原始证据和反馈问题集。
2. 增加结构化断言：普通聊天读取次数、证据覆盖、原始媒体直出、scope 隔离和未经确认写入。
3. 以当前基线运行问题集，记录路由、回答、证据和图片结果，不修改记忆生成数据。
4. 只针对失败样本调优 Agent 计划、检索排序、回答模板和前端呈现。
5. 重跑完整项目验证命令：

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
git diff --check
```

6. 请求代码审查，修复 Critical/Important 问题。
7. 将已验证的 Agent 代码和文档安全传输到 153 `psh`，重新验证服务和回归测试后再正式提交。

## 完成标准

- 普通聊天不读取家庭记忆。
- 每个具体记忆回答都有可展开证据。
- 明确原始证据请求直接给出可打开媒体。
- 无证据时不编造事实。
- 计划、工具和状态受 scope 限制。
- 反馈不会未经确认修改事实。
- 记忆生成相关文件没有被修改；若接口不足，形成独立请求。
