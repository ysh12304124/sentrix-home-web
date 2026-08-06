# R9-3 · 12B Agent Model Profile — 报告

**日期**：2026-08-06
**性质**：12B 默认 Profile 基础设施 + Parser 槽位验收脚本（bake-off 主目标从"选默认 Parser"改为"验证 12B 达门槛"）。
**测试**：本地 483 通过（R9-2 479 + 新增 4）。

## 1. 交付

| 文件 | 改动 |
|---|---|
| `backend/model_routing.py` | `ROLES` 增 `claim`/`repair`；`resolve_specs` 支持 `SENTRIX_AGENT_MODEL_PROFILE`（quality_12b 默认 / experimental_2b→2B parser via e2b 8100），显式 per-role env 优先；`SENTRIX_CLAIM_MODEL`/`SENTRIX_REPAIR_MODEL` |
| `backend/model_clients.py` | `ROLE_INFERENCE` 参数表（parser/answer/writer/verify/claim/repair 的 temperature/think/num_ctx/num_predict）；`GammaClient.__init__` profile 逻辑 + claim/repair 模型；`_endpoint_for` 支持 claim/repair；`chat()` 按 role 下发推理参数 |
| `scripts/benchmarks/evaluate_parser_slots.py`（新） | 15 条**合成**槽位标注（不用 benchmark 原句）；`--candidate e2b/12b/7b` 切换；指标：action/facet/semantic/negative/date/media recall、invented hard、JSON 首过、延迟、mode_accuracy（次要）；输出 `parser_slots_{candidate}.json` |
| `backend/tests/test_parser_slots.py`（新） | 标注集覆盖、不引用 benchmark 原句、candidate env、指标谓词 |

## 2. 12B 角色参数（ROLE_INFERENCE）

| 角色 | temperature | think | num_predict | num_ctx |
|---|:-:|:-:|:-:|:-:|
| parser | 0 | false | 512 | 4096 |
| answer / writer | 0.3 | false | 800 | 8192 |
| verify / claim | 0 | false | 512 | 4096 |
| repair | 0 | false | 512 | 4096 |

keep_alive 沿用 -1（Ollama 常驻）。`role=None`（非 Thin 路径，如图像分析）保持旧行为（temp 0，无 num_ctx/predict）。

## 3. Profile 解析

```text
SENTRIX_AGENT_MODEL_PROFILE=quality_12b (默认)
  PARSER=gemma4:12b (ollama_local 11434) / ANSWER=12b / VERIFY=12b / CLAIM=12b / REPAIR=12b
SENTRIX_AGENT_MODEL_PROFILE=experimental_2b
  PARSER=gemma-4-e2b-it+lora-v2 (e2b 8100) / ANSWER/VERIFY/CLAIM=12b / REPAIR=2b
显式 SENTRIX_PARSE_MODEL / SENTRIX_PARSE_BACKEND / SENTRIX_ANSWER_MODEL ... 始终优先
```

**12B Parser 超时/失败 → 空 draft → Router（确定性硬条件→Probe→evidence/clarify），不自动切 2B。**

## 4. 待 153 实测（本阶段代码已就绪）

- [ ] `evaluate_parser_slots.py --candidate 12b`（12B 门槛：action/facet≥95%、negative≥98%、invented=0、JSON≥98%）
- [ ] `--candidate e2b`（2B 对照，记录不作门槛）
- [ ] 153 `start_sentrix_api_8091.sh` 切 `SENTRIX_AGENT_MODEL_PROFILE=quality_12b`（移除 2B parser env 强制）

## 5. 下一步

R9-4：bge-m3 Text ANN Shadow **Sidecar**（独立 venv + sidecar 服务 + 独立 text ANN + shadow 对照）。
