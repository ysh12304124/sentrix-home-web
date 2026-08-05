# Agent 家庭记忆助手实施计划

## 目标

在不修改记忆生成管线的前提下，完成 Agent 核心契约、分层记忆调用、连续对话、必有证据展示、原始证据直出、反馈治理和验收调优。

## 架构

`backend/agent.py` 负责意图、计划校验、工具执行、状态和证据排序；`backend/app.py` 只负责 assistant API 边界；`src/api.js` 与 `src/app.js` 负责响应消费和证据交互；现有 MemoryStore 作为记忆读取权威来源。记忆生成、模型适配和重建保持不变。

## 技术栈

Python `unittest`、FastAPI 现有 assistant 路由、Node test runner、原生浏览器 JavaScript。正式后端提交只在 153 `psh`。

## 执行拆解与检查点

任务按 `C0 → B → A → C1 → D` 推进；每个任务必须先有失败测试或可复现基线，再实现、验证，并在 153 `psh` 复跑。以下台账是本计划的交付门，不是实现建议。

### C0：真实旧记忆前置基线

- [x] 只读复制旧备份，确认已确认人物、semantic claims、person patterns、事件、观察、资产和三条人物证据链均可读取。
- [x] 固化 `scripts/benchmarks/evaluate_agent_replay.py`，缺表、缺列、无已确认人物时阻断回放。
- [x] 检查点：原始数据库字节不变化；“明哥”已确认且不是 pending cluster；报告可 JSON 序列化。

### B0：Agent-owned Annotation Store

- [x] 增加版本迁移、checksum、单事务初始化、关闭开关、幂等 assertion、orphaned 引用、preference 和 cooldown upsert。
- [x] 检查点：只新增 Agent 表，不写 canonical 表；迁移失败不阻断 Memory Kernel；`test_agent_annotations.py` 全部通过。

### B1：内部契约与读取强度

- [x] 独立 `ClaimExtractor` 扫描完整回答和 follow-up；Writer claim 仅为候选。
- [x] 建立 `NarrativeContextPacket`、`Canonical Evidence Bundle`、逐 claim `ClaimVerification`、一次局部 Repair 和 fallback。
- [x] 固化 `none / probe / ambient / targeted / forensic` 关系；普通聊天不注入家庭事实。
- [x] 检查点：模型不能降级 memory 计划、不能引入未知工具、Scene narrative 不能单独证明事实。

### A1-A2：自然叙事、Scene 与 Focus Stack

- [x] 人物介绍通过 Context Packet 归纳身份、关系、重复活动、外观和未知边界；已确认人物不退化为 cluster 澄清。
- [x] Scene 使用时间范围、观察、资产、参与者、source revision 和 confidence，并限制 12 observations、6 assets、8 participants。
- [x] Focus Stack 限制人物/Event/topic 数量，每轮衰减 `0.75`，低于 `0.2` 删除，切换 scope 清空。
- [x] 检查点：真实“明哥”人物介绍不再只是事件列表；Writer、Extractor、Verifier、Repairer 路径有单测。

### A3：API 与 claim-level evidence

- [x] API 稳定返回 `claims`、`claim_verifications`、`claim_verification_status`、`repair_count`、`evidence_bundles`、`claim_evidence_index` 和 `segments`，保留旧字段。
- [x] 前端用 `segments + claim_id` 渲染，不使用跨 Python/JavaScript 的 offset；依据区默认折叠，逐句映射到 Event/Observation/Asset。
- [x] 明确原始证据请求直接展示媒体；普通聊天不展示记忆证据区。
- [x] 检查点：Python Agent/API、Node、`node --check`、`compileall`、`git diff --check` 通过。

### C1：真实问题集与回放

- [x] 固化人物介绍、衣着、性格边界、关系边界、偏好边界、追问、原始照片、角色歧义、无证据和 scope 切换问题集。
- [x] `evaluate_agent_c1.py` 每次复制数据库，记录回答、claims、evidence bundles、逐 claim verification、repair 次数、图片数量和耗时；关键 deterministic 基线重复 3 次。
- [x] 检查点：153 旧备份 `3 repeats × 10 cases`，`failures = 0`；人物维度问题不再被路由为闲聊或无关事件列表。
- [ ] 实际 Gamma Writer 的 3 次自然回答回放仍需在可接受的模型运行配置下完成；当前 12B 本机单次调用过慢，未将未完成结果计入通过。

### D：适度主动回忆

- [x] `SENTRIX_PROACTIVE_MEMORY` 默认关闭；开启后普通聊天只读轻量 Event index，返回 `probe`，不读取具体 Observation/Asset。
- [x] 实现硬敏感门、viewer 级开关、归一化主动评分、单入口、Scene cooldown、重复惩罚、连续忽略降级和接受后具体证据回放。
- [x] 反馈通过 Agent-owned Annotation Store 持久化，不修改 canonical facts；前端支持查看、暂不查看和关闭主动回忆。
- [x] 检查点：fixture 覆盖 flag、probe、敏感内容、cooldown、ignore streak、acceptance、viewer 隔离；D 默认关闭，未通过独立灰度前不对生产用户主动开放。

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
