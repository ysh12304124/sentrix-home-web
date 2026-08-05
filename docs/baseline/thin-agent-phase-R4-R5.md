# Phase R4 + R5 · Gate + Neutral Probe · 模型路由/统一 deadline/调用预算

**日期**：2026-08-06
**状态**：代码 + 单测完成（428 tests 全绿）；153 模型延迟验收待跑

## R4 交付物

### `memory_gate.py` 重写（P0-6）
- `GateDecision` 加 `allow_probe` + `proposed_mode` 别名
- **删除长度 >6 repair 与 confidence 单点路由**（模型自报置信度只进 trace）
- 综合判定：fast-path（写作/翻译/"不用查" → none 不 probe；feedback/selected_entity → evidence）→ parser evidence/contextual 直通 → parser none 时：
  - `_has_household_signal(draft)`（facets 家庭维度 / semantic_conditions / negative_conditions）→ **ambiguous + probe**
  - `_explicit_general_task(message)`（写作/解释/为什么/介绍一下…概念 等一般任务结构）→ none
  - 其余（裸名词短语）→ **ambiguous + probe**
- **已知 tradeoff（记录在案）**：parser 幻觉 none 且消息以一般介绍动词开头的家庭人物查询不进 probe；该 case 的真正修复在 R5 parser prompt / 校验。

### `retrieval/probes.py` 实装（P0-7/P0-8）
- Neutral Probe 用 **raw text** 中性查询（不依赖错误 QuerySpec，不构造未确认的 hard semantic constraints）
- 强候选 = 多信号：多通道一致性（≥`minimum_channels_agreement` 且共享 asset）、lexical exact 短语、per-space 校准阈值（`configs/retrieval/defaults.json probe.per_space`）
- **只输出 upgrade/clarify/none 路由信号，不产出家庭事实**

### `evidence_retrieval.py` `probe()` + `thin_agent._ambiguous_path`
- kernel.probe(raw_text, scope) → 共享 retriever + probe 预算
- ambiguous → probe upgrade → 补 semantic condition + 重新正式检索；clarify → 询问"找照片还是聊别的"；none → normal chat

### `query_parser.py` 校验收紧
- `mode=evidence` 但 `actions=[]` → repair（P0-6 结构不一致，不用长度）

## R5 交付物

### `model_clients.py::GammaClient`
- `parse_model`/`answer_model`/`verify_model`/`parse_backend`/`parse_base_url`；`chat(role=...)`
- 显式构造参数始终生效；env 默认只在 `SENTRIX_MODEL_SPLIT_V1=1` 时启用
- parser role 可指向 153 2B（D6：`SENTRIX_PARSE_BACKEND=e2b` + `SENTRIX_PARSE_BASE_URL`），**不修改 153 e2b_server 实现**
- 主 Answer 保持 gemma4:12b（D4）

### 角色接线
| 调用点 | role |
|---|---|
| QueryParser | parser |
| thin_agent._normal_chat | answer |
| ComplexAnswerBuilder Writer | answer |
| LLMClaimExtractor | verify |

### `model_routing.py`
- `ModelSpec`/`resolve_specs`（env 解析）
- `RequestDeadline`（统一 20s；parser 4s / retrieval 5s / answer 7s / overhead 2s；`phase_available=min(remaining, budget)`）—— **P0-11 不等 30s httpx timeout**
- `CircuitBreaker`（按 role，threshold 3，60s 半开）
- `ModelRouter.chat(role, prompt, fallback)`——deadline + breaker 包一层

### `scripts/maintenance/probe_model_health.py`
- cold/warm latency + parser JSON 合法率 → `docs/baseline/model_health_*.json`；**合成 probe 文本（无 benchmark 查询，guard 守护）**

### `start_sentrix_api.sh` 补齐
- 补 9 个 Phase 0-8/2R flag（本地落后于 153）+ R5 flag（`SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1` 等默认 on、`SENTRIX_MODEL_SPLIT_V1` 默认 off）
- **交付时只追加，不覆盖 153 已含 flag 的版本**

## 测试
`test_gate_probe`（11：writing/no-length/none-bare→probe/household-facet→probe/evidence-stays/confidence-not-routing/多通道 upgrade/单通道 clarify/无命中 clarify/exact phrase/无事实）、`test_model_budget`（10：role 解析/endpoint/deadline/breaker/router fallback/role 转发）

## 本地验证
```
unittest discover backend.tests → 428 OK (skipped=1)
```

## 153 验收（待跑）
```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/probe_model_health.py --report docs/baseline/model_health_20260806.json
# 若 parser 延迟/JSON 合法率不达标，按 D6 切 SENTRIX_PARSE_BACKEND=e2b + SENTRIX_PARSE_BASE_URL=http://127.0.0.1:8100
```
