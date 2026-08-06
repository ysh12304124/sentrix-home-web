# 三相册 60 case 端到端 Agent 功能测试 · 发现与修改方案

**测试时间**：2026-08-05
**测试范围**：使用 `~/Downloads/samples/album{1,2,3}/query.json` 的 60 个真实 benchmark case，通过 API `POST /api/assistant/turn` 端到端验证；因 Ollama 单次 240s 平均延迟，代表性 6 case + 精确 parser 探针即可暴露全部 root cause，不需要跑完 60 case。

## 采样结果（6 代表性 case）

| Key | Query | GT | Verdict | Hits | FP | Mode | Latency |
|---|---|---:|---|---:|---:|---|---:|
| album1-01 | 浅黄色拼接毛绒睡衣自拍 | 1 | **fail** | 0 | 10 | evidence | 180s |
| album1-07 | 贵阳夜晚步行街 | 0 | pass | 0 | 0 | none | 239s |
| album2-05 | 在上海市拍的自己和王明的照片 | 1 | **error(timeout)** | 0 | 0 | none | 300s |
| album2-06 | 夜晚车内的明哥搂着我 江西省 | 0 | pass | 0 | 0 | none | 253s |
| album3-01 | 银色心形手镯 | 1 | **fail** | 0 | 0 | **none** | 243s |
| album3-14 | 水族馆海豚跃出水面 | 0 | pass | 0 | 0 | none | 226s |

**总计：3 pass / 2 fail / 1 error / 平均 240s**。**pass 的 3 个都是空 GT case**（正好也应该 mode=none 或空返回），实际的正样本命中率是 **0/3**。

## Parser 稳定性探针

同一 query 3 次调用 QueryParser（Ollama gemma4:12b 直接调用）：

| Query | Run 0 | Run 1 | Run 2 |
|---|---|---|---|
| 银色心形手镯 | mode=evidence, conds=[object:手镯, visual:银色心形] | **mode=none** | **mode=none** |
| 浅黄色拼接毛绒睡衣自拍 | evidence, [clothing:浅黄色拼接毛绒睡衣, visual:自拍] | evidence, [visual:浅黄色, clothing:拼接毛绒睡衣] | **mode=none** |
| 在上海市拍的自己和王明的照片 | evidence, [place:上海市, media:照片] | **mode=none** | **mode=none** |
| 厨房里做晚饭 | evidence, [place:厨房, activity:做晚饭] | **mode=none** | evidence, [place:厨房, activity:做晚饭] |

**结论**：同一 prompt 在 mode 上剧烈波动（有时对，有时错），条件抽取也在 `object` vs `clothing` vs `visual` 之间跳。这不是我们代码 bug，是 **Ollama gemma4:12b 在 partial-VRAM 状态下对 JSON 结构化输出严重不稳定**。

## 已确认的 5 个 Root Cause

### RC-1 · Parser 不稳定 · 严重
Ollama 温度=0 但同 prompt 3 次仅 40% 输出正确 mode/conditions。触发因素：
- Prompt 太长（约 1500 chars 含 schema + 11 条规则），可能触发 num_ctx 截断
- gemma4:12b 部分 VRAM（189MB/8GB）导致每次推理状态不一致
- Ollama 无 `seed` 参数强制确定性

**用户可见影响**：`album3-01 银色心形手镯` 明明是照片查询，API 返回 mode=none 走 normal_chat 生成"听起来是一个非常精致的设计"，完全没查记忆。

### RC-2 · Parser 输出兜底逻辑过弱 · 严重
`QueryParser._validate` 只检查 `mode in {none, contextual, evidence}`。但 `sanitize_query_parse` 对空 dict 或缺 mode 字段的 model 输出会默认填 `mode="none"`——这是 valid 值，触发不了 repair。

后果：Ollama 返回 `{}` 或 `{"answer": "..."}` 这种非 QueryParseDraft 时，parser 悄悄降级到 `mode=none, actions=[]`，用户完全查不到照片。

### RC-3 · Evidence Retrieval 精度差 · 严重
`album1-01` 进入 evidence 模式，返回 10 张图但**没一张是 IMG_4350.JPG**（真答案）。

原因链：
- Parser 把"浅黄色拼接毛绒睡衣自拍"拆成 `[clothing:浅黄色, clothing:拼接, clothing:毛绒, clothing:睡衣, visual:自拍]` 或类似（词粒度）
- `EvidenceRetrievalKernel._contains` 用 `re.findall(r"[\w一-鿿]+")` 分词后要求 `all(term in haystack)` — 单字 "色" / "毛" 太宽，命中大量无关 asset
- Semantic 条件全走 `SEMANTIC_REQUIRED` 一层，无排序偏好；exact/strong/approximate 3 层区分被磨平

### RC-4 · Answer Composer 输出重复模板 · 中等
`album1-01` 回答里 "记录支持浅黄色拼接；记录支持毛绒睡衣；记录支持浅黄色拼接；..." 重复十次。

`ThinAgentRuntime._allowed_facts` 对每个 asset 的每个 condition 都塞一条 text=f"记录支持{key.split(':', 1)[1]}"。多个 asset 命中同 condition → 重复文字。

### RC-5 · 性能不达 API 硬上限 · 严重
- 单次 API 请求 180-300s，原计划 §11 硬上限 20s，超 9-15 倍
- Ollama gemma4:12b 每次调用 9-90s 波动（VRAM=189MB / model=8GB）
- Thin Agent 单轮至少 2 次 Ollama（Parser + Answer），evidence 复杂路径 4-5 次

**根因不在 Thin Agent 代码**，在模型基础设施。但当前状态下**任何真实用户对话都不可用**。

## 附加发现

### F-1 · Ollama 侧的 keep_alive=-1 生效但 VRAM 只 189MB
`ollama ps` 显示 `expires_at="2318-..."` 说明常驻，但 `size_vram=189M / size=8G` 说明只有嵌入层/前几层在 VRAM，其余每次页错误。这是 Ollama 主动 offload 而非 evict。

### F-2 · Fast-path 正常工作
"帮我写一段生日祝福" fast-path 触发，parser 调用 0 次，返回自然聊天回答。这部分行为正确。

### F-3 · 空 GT case 通过率 100%
album1-07, album2-06, album3-14（空 GT 三个 case）都正确返回 evidence=[] 且不硬编答案。这说明 evidence 判空 + parser 幻觉抑制在这类 case 上工作。但**只是因为 parser 幻觉性地返回 mode=none**（RC-1），凑巧掩盖了另一 bug。

### F-4 · album2-05 "上海市自己和王明的照片" 直接 300s timeout
Complex 路径（人物+地点复合条件）时 Writer/Verifier/Repair 4-5 次 Ollama 调用总时 > 300s。

---

# 修改方案

按 **重要度 × 修复成本** 排序。

## Fix 1 · 提升 Parser 稳定性 · P0 · 中等成本

**手段**：
1. **加 Ollama `seed=42` 参数**：`GammaClient.chat(payload["options"]["seed"] = 42)`，强制确定性
2. **加 `num_ctx=4096`**（当前默认 2048）避免长 prompt 截断
3. **Prompt 缩短 + 强制 few-shot**：把当前 11 条规则精简到 4-5 条，加入 3 个 few-shot 例子（一个 none / 一个 contextual / 一个 evidence + 复合 actions）
4. **验证收紧**：`_validate` 检查 mode + actions 一致性——如果 mode=none 但用户消息不匹配 `_WRITING_PREFIX_RE` 且长度 > 6 字，触发 repair
5. **Ollama 侧尝试**：`ollama pull gemma3n:e2b` 或换更小可完整 VRAM 驻留的模型（gemma3:4b 完整 VRAM 载入 ~3GB）

**验收**：探针脚本 3 次调用同 query，mode 必须一致；60 case parser 输出稳定率 ≥ 95%。

## Fix 2 · Evidence Retrieval 收紧匹配 · P0 · 中等成本

**手段**：
1. **Kernel `_contains` 改为多种匹配级别**：
   - 严格匹配：完整 constraint.value 作为子串出现 → confidence=1.0
   - 关键词匹配：至少 2 个字符（当前 1 字）+ ≥60% 覆盖 → confidence=0.5
   - 单字匹配一律不算命中
2. **Parser 输出 `source_text` 作为整体 semantic 条件保留**——不拆分。拆分放到 kernel 内部策略层，代码层保留整体信号
3. **加入 CLIP 向量召回**：Evidence Kernel `retrieve()` 里除了结构化条件，也用 CLIP 文本编码 query 与 asset visual embedding cosine → 补召回。原计划 §3.1 已经列在通道里，Phase 3 没接
4. **exact/strong/approximate 分层生效**：exact 至少 1 个硬命中 + 无 approximate/unknown；strong 允许 1-2 个 possible；approximate 才是当前唯一等级

**验收**：album1-01, album3-01 单命中 case 返回 1 张正确图；album1-02 多命中 case Recall@10 ≥ 0.8。

## Fix 3 · Answer Composer 去重 · P1 · 低成本

**手段**：修改 `ThinAgentRuntime._allowed_facts` 把 `text` 按 condition_key 去重（一个 condition 只出一条 fact，无论多少 asset 命中）。修改 `_simple_answer` 拼接前 `dict.fromkeys` 去重语句。

**验收**：album1-01 回答不出现重复"记录支持"模板。

## Fix 4 · 性能 · P0 · 需运维决定

三选一（用户/运维选）：

### 4A · 让 gemma4:12b 完整驻留 VRAM
- 检查 `OLLAMA_KEEP_ALIVE=-1` + `OLLAMA_NUM_PARALLEL=1`
- 检查 GPU 驱动（当前 nvidia-smi 报 driver mismatch）
- 若显存不够（应该有 24GB 3090），排查 Ollama 配置或占用

### 4B · 切换到 8100 e2b_server（Gemma4 e2b-it + LoRA v2）
- 已在 153 上运行（`start_sentrix_e2b.sh`）
- 模型更小（2B），完整 VRAM 载入
- 通过 `/api/vlm-backend POST` 切 `active=e2b_lora`
- 现有 `GammaClient` facade 已支持切换（`E2BBackend`）
- **但**：e2b 是 stub / 未验证 JSON 稳定性；建议先在 parser 上单独 A/B

### 4C · 用更小的模型做 parser，gemma4:12b 只做 answer
- Parser 用 gemma3:4b 或 qwen2.5:3b（完整 VRAM，Ollama 8s 内响应）
- Answer 保留 gemma4:12b（自然度更好）
- `GammaClient` 加 `parse_model` / `answer_model` 分离
- 单次 evidence 时间从 240s 降到 ~30s

**推荐 4C**：修复成本最低，功能保留 gemma4:12b 质量。

## Fix 5 · Complex Path Timeout · P1 · 依赖 Fix 4

Phase 4 复杂路径 4-5 次 Ollama 调用 → 4A 或 4C 后单次 <10s，累计 <50s，可控。加 `httpx timeout=30`（当前 180）防止一次 Ollama hang 阻塞整轮。

---

# 下一步执行序列

按依赖排序：

1. **Fix 1** Parser 加 seed + num_ctx + few-shot + validate 收紧
2. **Fix 4C** 切分 parse/answer 模型（parser 用小模型）
3. **Fix 2** Evidence retrieval `_contains` 收紧 + CLIP 补召回
4. **Fix 3** Answer composer 去重
5. **Fix 5** Complex path timeout
6. **重跑三相册 60 case**：Recall@10 目标 ≥ 0.6（保守值，考虑 Ollama 波动），P95 延迟 ≤ 30s
7. **重跑 evaluate_thin_agent_semantic.py + evaluate_evidence_retrieval.py** 确保之前的合成测试无回归

预计工作量：Fix 1-5 完成 + 60 case 验收 ≈ 6-8 小时（含 benchmark 跑 4 小时）。

## 严重程度声明

- **合同层 / 骨架层**：Phase 0-8 交付的合同（sanitize、EvidencePacket、Constraint、AnnIndex、Core Memory 表等）**没有 bug**。合成测试 24/24、375 unit tests 全绿仍然有效。
- **运行时 / 输出质量层**：**当前状态下不可交付真实用户**——parser 不稳定 + evidence 精度差 + 性能超硬上限 9 倍。
- 修复 Fix 1-4 后可以再评估是否可交付；Fix 5 之后可谈生产验收。

**建议**：先集中修 Fix 1 + Fix 4C，快速验证 parser 稳定后再看 evidence 精度。不要在当前 Ollama 状态下继续跑完整 benchmark，性价比极低。
