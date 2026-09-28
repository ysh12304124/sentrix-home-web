# Sentrix P2 检索通道与候选治理诊断

日期：2026-08-24  
权威环境：153（`asus@192.168.0.153:/home/asus/Github/Sentrix-Home-Web`）

## 结论

P2 的通道稳定性问题已定位并处理；候选过多导致的质量问题已被观测、结构化记录，但尚未宣称解决。当前不应直接更换模型或盲目降低召回数量，应先用现有 `candidate_window` 诊断字段做单变量候选策略 A/B。

## 通道验证

- `8101 /health`：`BAAI/bge-m3`、1024 维，正常。
- 直接 `/embed`：153 本机成功。
- 8091 重启后的 Level-1 探测：Qdrant 346 collections、45,330 points，锁由服务进程持有，无降级。
- 最新 100QA 的 69 次检索采样：`visual_ann` 与 `text_ann` 均为 `ready / backend=qdrant`，未出现新的 `embedder_unavailable`。
- 外部执行 `sync_qdrant_vectors.py --benchmark` 无法打开嵌入式 Qdrant 锁，这是服务持锁的预期互斥，不是线上检索降级；线上服务内检索正常。
- `bge_text` 增加健康探针短 TTL 缓存，避免批量 QA 中每次请求重复等待 1.5 秒健康检查。

## 候选窗口诊断

`search_memories` 已输出有界诊断信息：候选总数、可见候选数、可见 rank、事件组数量、最大事件组及策略；模型仍只看到 bounded preview，服务端保留完整 ResultSet。这样解决了“模型看到内部全量 ID/无法区分展示窗口”的契约问题，但没有替模型完成最终候选选择。

最新 100QA 的根因抽样显示，仍有多道题返回 20 个跨事件候选，模型随后用泛化元数据或错误复核图作答。典型表现是：检索召回存在，但关键图片没有进入有效复核链；这与并发（当前 vLLM `max_num_seqs=12`）无关，属于候选治理/视觉确认策略问题。

## 下一步单变量实验

1. 保持模型、相册、并发和 ANN 不变，仅改变候选窗口策略：`head-only`、`head+event-diversity`、`head+visual-query`。
2. 每种策略记录 `candidate_window`、首个 inspect handle、GT 覆盖率、inspect 后回答质量。
3. 只有在候选策略出现稳定收益后，再考虑 reranker 或模型切换；否则会把召回问题和模型问题混在一起。

