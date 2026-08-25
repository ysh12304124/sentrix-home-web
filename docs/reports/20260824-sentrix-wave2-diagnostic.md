# Sentrix Agent2 Wave 2 只读诊断报告

日期：2026-08-24  
基线主机：`asus@192.168.0.153`  
范围：只读诊断；本报告没有修改运行代码，也没有重启服务。

## 1. 100QA 严格对比

对比的两个 run 使用相同的 qwen3.5 profile、相同 scope、相同 QA 数据 hash 和相同 Judge 配置：

| 指标 | 基线 `20260824-143818` | 修复后 `20260824-173228` | 变化 |
|---|---:|---:|---:|
| retrieval recall mean | 0.616 | 0.641 | +0.025 |
| answer quality mean | 0.76 | 0.78 | +0.02 |
| exact accuracy | 0.33 | 0.34 | +0.01 |
| core accuracy | 0.43 | 0.44 | +0.01 |
| retrieval precision micro | 0.118 | 0.114 | -0.004 |
| retrieval recall micro | 0.406 | 0.500 | +0.094 |
| within-step completion | 0.87 | 0.87 | 0 |
| agent loop calls mean | 2.93 | 3.04 | +0.11 |

新 run：100/100 完成，100/100 Judge 完成。结果说明 Wave 0/1 没有造成回归，并出现小幅正向变化；但单次 benchmark 的最终质量提升仍不足以宣称是稳定收益，下一轮应保留同一配置做重复 A/B。

## 2. `embedder_unavailable` 根因

### 2.1 sidecar 本身正常

153 当前 8091 环境明确设置：

- `SENTRIX_TEXT_EMBEDDER=bge`
- `SENTRIX_VECTOR_BACKEND=qdrant`
- `SENTRIX_QDRANT_PATH=/home/asus/Github/Sentrix-Home-Web/data/qdrant`

8101 健康检查返回：`{"status":"ok","model":"BAAI/bge-m3","dimension":1024}`。`/tmp/bge_sidecar.log` 只有 ready 日志，没有 timeout、异常或熔断记录。

新 100QA 的 68 次实际检索中，`text_ann` 全部为 `status=ready`，每次都有 text embedding latency；trace 中没有 `embedder_unavailable`。因此本轮没有证据表明 BGE sidecar 或熔断器是主要根因。

### 2.2 benchmark scope 的 Qdrant collection 不完整

8091 `/api/health` 显示 Qdrant 可用、430 个 collection、76740 个 point，但最近检索的 `active_backend` 是 `sqlite`，`degraded_since` 从 17:33:17 开始，最近 visual/text 搜索全部记录为 SQLite。

对 SQLite 中 benchmark scope `album_cba01be9502b` 的向量组合和 Qdrant `meta.json` 做了只读匹配，发现以下 3 个应有 collection 不存在：

| space | model | dimension | SQLite 行数 | Qdrant collection |
|---|---|---:|---:|---|
| semantic | BAAI/bge-m3 | 1024 | 535 | 缺失 |
| episodic | BAAI/bge-m3 | 1024 | 535 | 缺失 |
| visual | chinese-clip-ViT-L-14 | 768 | 363 | 缺失 |

同一 scope 的旧 `ViT-B-32` 等向量 collection 存在。当前代码在 `MemoryStore._search_vectors_impl()` 中，如果 Qdrant 没有匹配 collection 或返回空结果，就直接落到 SQLite 全表余弦扫描；该路径没有把“collection 缺失”单独记录为错误，所以外部只看到 `backend=sqlite`，看不到具体原因。

结论：本次 100QA 的视觉/文本 ANN 并没有真正使用 Qdrant，而是对 benchmark scope 的 SQLite 向量做了回退扫描。它不是 sidecar 不可用，也不是外部进程抢占 Qdrant 锁；当前锁由 8091 自己持有。

## 3. 视觉/文本通道实际状态

新 run 的 68 次检索通道统计：

| 通道 | 调用次数 | 状态 | backend | 平均候选数 | 平均耗时 |
|---|---:|---|---|---:|---:|
| visual_ann | 68 | ready | sqlite | 20 | 4191.6 ms |
| text_ann | 68 | ready | sqlite | 20 | 381.4 ms |
| metadata | 68 | ok | — | 29.19 | 296.6 ms |

visual_ann 平均耗时明显高于 text_ann，且两个 ANN 通道都走 SQLite。这解释了“视觉/文本通道差异”目前无法公平比较：当前比较的是不同 embedding 生成耗时加上同一个 SQLite fallback，而不是 Qdrant 与 SQLite 的等价通道。

## 4. “一次性放弃”与证据闭环诊断

### 4.1 Wave 1 已解决的部分

Wave 1 能在第一次 final 前识别一部分缺证据情况，并追加检索、inspect 或 OCR 提示；flattened `recommended_resolution` 也已能触发 OCR 路由。

### 4.2 尚未解决的闭环缺口

新 run 的 Agent2 trace：

- `candidate_closure`：97 个；
- `candidate_partial`：14 个；
- 14 个 partial 中，11 个是 8500 `/tokenize-current` 返回 502，1 个是 context budget exceeded，1 个是 parse failure，1 个是 guard recovery exhausted。它们不应与证据检索失败混为一谈，属于模型服务/预算可靠性问题。
- 有 32 个最终 trace 仍有 `open` requirement，其中 19 个仍被标记为 `candidate_closure` 并正常结束。

这说明当前门控仍是“最多提示一次”，不是“证据状态闭环”。`runtime.py` 当前 `max_completion_retries=1`；Completion Gate 在一次提示后，如果模型再次输出 final，就不会无限阻断。更关键的是，CompletionState 主要以“工具是否调用过”判断满足：

- `read_photo_text` 返回 `status=partial, reason=ocr_failed, full_text=''` 后，只要工具名出现在 `tools_called` 中，传统 OCR requirement 就可能被标记为 satisfied；
- inspect 返回了不相关或低覆盖观察时，没有“问题要求与观察覆盖是否匹配”的判定；
- 没有“当前候选失败 → 自动升级下一候选/分页 → 重新计算缺口”的状态转换。

### 4.3 OCR 直接证据

新 run 中实际执行的 10 次 `read_photo_text` 全部返回：

```text
status=partial
reason=ocr_failed
full_text=""
_model_call_metrics=[]
```

其中多数调用耗时接近 0 秒，且没有 OCR 模型调用指标。代码层面 `_ocr_single_asset()` 只实现 PaddleOCR small 路径，失败后直接返回 `None`；`_read_photo_text_impl()` 随后返回 partial，没有真正的候选升级或可靠的第二通道兜底。独立只读复现表明 153 的 PaddleOCR 包和缓存模型可加载，因此当前更像是 8091 worker 中的运行时路径/初始化/结果集绑定问题，而不是模型完全不可用；这需要后续 P1 以 worker 内 telemetry 和单 worker 复现继续定位。

## 5. P1 方案建议（本轮不实现）

### P1-A：证据闭环

建议新增独立的 evidence coverage 判定，不再使用“调用过工具”作为满足条件：

1. `candidate`：工具返回了候选，但尚未证明候选与要求相关；
2. `supported`：观察/文字结果覆盖了要求的一部分；
3. `confirmed`：结果满足要求，且有 provenance、候选句柄和问题属性绑定；
4. `failed`：工具失败、空结果、OCR partial 或观察不匹配；
5. `failed` 时由 Agent2 重新计算缺口，优先换下一个 preview handle，再考虑分页或重新检索；
6. 只有 `confirmed` 或明确的“不足以确认”终态才允许 closure；
7. 对 `ocr_failed`、空 inspect、candidate-only 结果，不应仅因为调用发生过就消除 gate。

实现前应先补 trace 字段：`requested_evidence_type`、`covered_evidence_type`、`coverage_status`、`candidate_handle`、`failure_reason`，否则无法判断重试是否真的改善了证据。

### P1-B：检索通道

优先顺序建议为：

1. 先为 `album_cba01be9502b` 补齐 BGE semantic/episodic 和 Chinese-CLIP visual 的 Qdrant collection，并核对 point 数与 SQLite；
2. 在检索 telemetry 中区分 `qdrant_hit`、`qdrant_no_collection`、`qdrant_empty`、`qdrant_stale`、`sqlite_fallback_error`；
3. 固定同一 query 集比较 Qdrant、SQLite、HNSW、fusion 四条路径，再决定索引、embedding、融合权重或 reranker 的改动；
4. 在严格 A/B 前不继续扩大 preview 截断或叠加新的排序复杂度。

### P1-C：OCR

先增加 worker 级别的 OCR telemetry，至少记录：结果集解析、asset path、small OCR 初始化、检测框数量、重试次数、最终失败原因和耗时。确认失败点后再决定是修复 worker 初始化、保留 PaddleOCR、还是增加明确的视觉模型 fallback；不要直接把 OCR 失败当成“没有文字”。

## 6. 交付结论

1. Wave 0/1 已落地，严格同配置 100QA 有小幅正向变化，但还不足以宣称稳定提升。
2. Wave 2 的 `embedder_unavailable` 诊断结论：BGE sidecar 正常，benchmark scope 的新模型 Qdrant collection 缺失，导致 visual/text ANN 全部 SQLite fallback。
3. Wave 1 没有完成完整 evidence closure；“工具调用过即满足”和固定一次 completion retry 仍会让 open requirement 的回答提前 closure。
4. 本轮不修改 P1 代码。下一步应先确认是否允许补齐 Qdrant collection/telemetry，然后按 P1-A 证据闭环、P1-C OCR worker 诊断、P1-B 通道 A/B 的顺序拆分任务，避免文件冲突。

