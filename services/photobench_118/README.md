# PhotoBench 评测服务

独立维护的 PhotoBench + Sentrix 端到端评测服务，包含本地编排器、Vue 3 前端、引用数据集、历史结果与日志。

## 目录

```text
backend/                 Python 编排器与 HTTP API
frontend/                Vue 3 源码
frontend/dist/           前端生产构建
config/                  vLLM Manager 目标配置
data/                    评测相册、身份图和 QA（本地数据，不纳入 Git）
results/                 历史评测结果（本地数据，不纳入 Git）
logs/                    服务日志（不纳入 Git）
scripts/                 本地启动脚本
```

## 默认开发模式

```bash
./scripts/start.sh
```

- Vue 3 开发前端：<http://127.0.0.1:5173/>，监听 `0.0.0.0`、支持热更新和局域网访问。
- Python API：<http://127.0.0.1:8771/api/>。
- Vite 会把 `/api/*` 代理到 8771，前端代码不需要区分开发和生产 API 地址。
- 8771 仍可访问最后一次构建的生产前端，但日常开发请打开 5173。
- 局域网设备使用 `http://<本机局域网IP>:5173/` 访问；macOS 防火墙需允许 Node.js 接收入站连接。

停止服务：

```bash
./scripts/stop.sh
```

## 生产构建

```bash
cd frontend
npm install
npm run build
```

构建完成后，生产页面由 Python 后端从 `frontend/dist/` 提供，可通过 <http://127.0.0.1:8771/> 访问。

默认依赖：100 服务器 Sentrix 后端、153 服务器 vLLM Manager，以及本机 Judge 服务。Judge 默认使用编排器所在机器自动探测的局域网 IPv4（例如 `http://192.168.1.65:1234`），不使用 `localhost`；具体地址仍可通过 `BENCH_SENTRIX_URL`、`BENCH_JUDGE_URL`、`BENCH_VLLM_API_URL` 等环境变量覆盖。

## 结果 API

逐题结果按“运行元数据 → 分页摘要 → 单题详情”分层返回，前端不会一次加载完整 run：

- `GET /api/runs`：全部 run 的轻量列表和汇总。
- `GET /api/runs/{run_id}`：单个 run 的阶段、指标和逐题总数，不包含 `items`。
- `GET /api/runs/{run_id}/items?page=1&page_size=20`：逐题分页摘要；支持 `search`、`score`、`task_type`、`angle`、`difficulty`、`answerability`、`agent_status`、`primary` 服务端筛选。
- `GET /api/runs/{run_id}/items/{index}`：用户展开题目时按需返回完整回答、图片、Judge、模型调用和工具调用详情。
- `GET /api/runs/{run_id}/judge-prompt`：读取该 run 保存的 Judge System Prompt。
- `GET/POST /api/runs/{run_id}/reviews`：读取或保存独立人工复核记录；数据写入 run 目录的 `reviews.json`，不修改 Agent 原始结果。

新结果会保存 Sentrix 现有 Agent 响应中的逐轮模型性能、工具调用、Guard/结束原因、Agent 状态和 QA 分类字段。评测 UI 仅展示工具名称、状态、调用关系和耗时，不依赖或呈现具体后端实现。历史结果缺少 QA 分类时，只读关联本地 QA 数据集补充展示；缺少真实运行指标时继续明确标记为未记录，不从服务日志推测或补造。

## 数据来源与扩展边界

| 信息 | 当前来源 | 是否需要 Sentrix 新接口 |
|---|---|---|
| 主 Agent 每轮 TTFT、总时延、输入/输出 token、token/s、上下文预算 | `/api/assistant/turn` 的 `model_call_metrics` | 否 |
| 工具名称、状态、总耗时及所属模型轮次 | `tool_trace` 与 `retrieval_trace` 的步骤顺序 | 否，由编排器绑定；历史单模型轮次可安全推断，多轮缺轨迹时保持未绑定 |
| 工具内部实现字段 | 可由 Sentrix 响应兼容保留 | 评测 UI 不展示，不作为评测契约 |
| Agent 状态、结束原因、Guard 恢复与回答证据约束 | `tool_loop_status`、`termination_reason`、`guard_debug`、`answer_grounding` | 否 |
| QA 分类 | QA 数据集；历史结果按 `qa_id` 只读关联 | 否 |
| R/V/O/T/S/G/J 初步分层归因 | 基于已保存召回、工具、回答、Guard 和 Judge 字段确定性派生 | 否；与 153 完全同口径仍需迁移其完整规则 |
| 人工复核判定和备注 | 评测服务 `reviews.json` 与 review API | 否，不写回 Sentrix |
| GPU/模型运行时起止快照 | Manager `/state`、`/gpu-stats`、`/process-memory` | 否；新 run 已保存，CPU/系统内存仍待数据源 |
| 数据集与 QA 完整性 | 本地 manifest、QA、照片和身份图 SHA-256 | 否；新 run 创建时保存 |

新 run 还会原样保存 Sentrix 返回的 `retrieval_trace` 为 `execution_trace`，用于追溯模型与工具执行顺序。历史 run 若没有该字段，只能展示当时真实保存的逐轮性能与工具字段，不能恢复每轮完整 messages 或模型原始输出。

运行日志只用于定位请求失败、服务异常等问题，不作为结果字段的数据源。评测指标必须来自结构化响应、Manager/GPU 采样接口或数据集，不从文本日志反向猜测。
