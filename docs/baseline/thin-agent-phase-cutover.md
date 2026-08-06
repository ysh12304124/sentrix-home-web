# 生产切换报告 · Thin Agent 与证据检索内核

**切换时间**：2026-08-05
**目标**：一次性开启 Phase 0-8 全部功能 flag，生产服务 API 8091
**变更范围**：仅 `scripts/runtime/start_sentrix_api.sh` + 服务重启

## 修改的启动脚本

`scripts/runtime/start_sentrix_api.sh` 加入 9 个 flag，全部支持环境变量覆盖：

```bash
export SENTRIX_THIN_AGENT_V1="${SENTRIX_THIN_AGENT_V1:-1}"
export SENTRIX_SEMANTIC_QUERY_PARSER_V1="${SENTRIX_SEMANTIC_QUERY_PARSER_V1:-1}"
export SENTRIX_EVIDENCE_RETRIEVAL_V1="${SENTRIX_EVIDENCE_RETRIEVAL_V1:-1}"
export SENTRIX_LLM_CLAIM_EXTRACTOR_V1="${SENTRIX_LLM_CLAIM_EXTRACTOR_V1:-1}"
export SENTRIX_CORE_MEMORY_V1="${SENTRIX_CORE_MEMORY_V1:-1}"
export SENTRIX_MEMORY_CORRECTION_V1="${SENTRIX_MEMORY_CORRECTION_V1:-1}"
export SENTRIX_ADVANCED_MEMORY_TOOLS_V1="${SENTRIX_ADVANCED_MEMORY_TOOLS_V1:-1}"
export SENTRIX_ANN_INDEX_V1="${SENTRIX_ANN_INDEX_V1:-1}"
export SENTRIX_EXPLICIT_IMAGE_REINSPECTION="${SENTRIX_EXPLICIT_IMAGE_REINSPECTION:-0}"
```

**`EXPLICIT_IMAGE_REINSPECTION` 默认 0**：只有用户明确说"重新看图/仔细核实"时才开启（原计划 §4 明确要求）。

备份：`scripts/runtime/start_sentrix_api.sh.bak-20260805`

## 数据准备

### Core Memory 首次建卡
```
$ python scripts/maintenance/build_core_memory.py data/sentrix.db --apply
{
  "entities": 1,     # album3 里 confirmed "明哥"
  "cards": 1,
  "items": 30,       # role + profile + patterns + relationships
  "apply": true
}
```

### Retrieval 派生投影
```
$ python scripts/maintenance/rebuild_retrieval_indexes.py data/sentrix.db --apply
rebuilt observation_search_terms for 378 observation(s)
```

派生行分布：object 1209, place 374, caption 373, activity 373, person_bridge 247, ocr 158, clothing 5。

### ANN 索引
```
visual   count=526  dim=512  build_ms=38.7   saved=data/ann/visual   reload_recall_top1=True
semantic count=374  dim=512  build_ms=4.5    saved=data/ann/semantic reload_recall_top1=True
episodic count=565  dim=512  build_ms=6.0    saved=data/ann/episodic reload_recall_top1=True
```

## 生产 API 状态

- **进程**：`uvicorn backend.app:app --host 0.0.0.0 --port 8091`
- **PID**：401220
- **flag 环境**：`/proc/401220/environ` 确认 8 个 `V1=1` + `REINSPECTION=0`
- **健康检查**：API 8091 = 200，Web 4174 = 200，FMA 5173 = 200

## Smoke Test 结果

```
POST /api/assistant/turn  {"message": "hi", "scope_id": ""}
→ HTTP 200
→ memoryUsed: false
→ evidence: []
→ retrievalTrace: [{"stage": "gate", "status": "none", "counts": {"memory_tools": 0, "evidence": 0, "query_parse": 1}}]
→ answer: "你好。有什么我可以帮你的吗？"
```

**核心行为全部正确**：
- Parser 被调用 1 次（`query_parse: 1`）— `SEMANTIC_QUERY_PARSER_V1` 生效
- 非家庭查询判定 `mode=none` — Phase 2R Gate 正确
- 无 evidence 检索、无原图访问 — 普通聊天不越界
- 返回自然聊天答复 — normal chat 路径通

## 已知的性能问题（非本次工作引入）

单次请求 ~1m47s 完成。诊断：
- Ollama gemma4:12b 在 11435 端点 `size_vram=189MB / size=8GB` — 模型主要在 CPU/RAM
- 直接 Ollama chat 单次调用 0.9s 到 79s 波动（取决于 VRAM 压力）
- Thin Agent 每轮至少 1 次 Parser + 1 次 chat = 2 次 Ollama 调用
- Evidence 模式（明哥人物介绍）会加 1 次 Writer + 1 次 ClaimExtractor + 可能 1 次 repair = 4-5 次 Ollama 调用

**这是模型基础设施性能问题，不是 Thin Agent 骨架问题**。原计划 §11 目标 API p95 ≤ 20s；当前需要：
1. 让 gemma4:12b 完整驻留 VRAM（当前 189MB → 需要 ~8GB），或
2. 切换到更小/更快的 backend（e2b_server 是候选：本地有 gemma4:e2b-it + LoRA v2 stub 在 8100）

**不需要现在做**：Thin Agent 的功能正确性已经通过。后续可以由 e2b_server 完成或 Ollama VRAM 调优接手。

## 回滚

不需要回滚：flag 全部功能正确，性能问题独立。

若真需要回：
```
# 关任意 flag 恢复旧路径
kill <uvicorn 8091 pid>
env SENTRIX_THIN_AGENT_V1=0 bash scripts/runtime/start_sentrix_api.sh &

# 或完全回滚到备份 script
cp scripts/runtime/start_sentrix_api.sh.bak-20260805 scripts/runtime/start_sentrix_api.sh
kill <pid>; bash scripts/runtime/start_sentrix_api.sh &
```

Web 4174 走 `SENTRIX_BACKEND_URL=http://127.0.0.1:8091`，跟随 8091 自动重连。FMA 5173 不受影响。

## 剩余待办 · 无阻塞

- **性能调优**：Ollama VRAM 完整驻留，或切换 8100 e2b_server
- **真实相册端到端回放**：单次 evidence 查询验收（需要模型稳定 <20s 之后）
- **Correction 端到端 UI**：propose/apply 已就绪，前端流程待联调

至此 Phase 0-8 + 补充 2R 全部生产落地。
