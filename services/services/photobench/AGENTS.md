# PhotoBench 评测服务

- 默认使用简体中文，修改保持精炼、可验证。
- 服务入口为 `backend/benchmark_orchestrator.py`，Vue 3 源码位于 `frontend/`，生产构建位于 `frontend/dist/`。
- 本地运行数据位于 `data/`，历史结果位于 `results/`，日志位于 `logs/`；不得删除历史结果或伪造缺失指标。
- 受管多模型公平对比必须通过模型 Manager 批量冷切换，不得手工切模型；复用单个外部 OpenAI-compatible 端点时允许不配置 Manager，生命周期和硬件遥测明确记为 `not_applicable`，不得因此判定测评失败。
- 每条 QA 的多轮主 Agent 调用必须保留 TTFT、总时延、输入/输出 token、token/s 和流式状态。
- 本仓库内文件引用使用相对路径；跨工作区交接记录由外部 Codex 工作区维护，不在仓库内建立绝对路径依赖。
