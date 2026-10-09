# 两段 HippoVlog 长视频故障归因旁路

本目录保存 2026-10-09 在 200 服务器上完成 80/80 道问答的原版脚本。它复用既有运行 `20260928-163006/svd_esvd` 的视频抽帧、事件与 WorldMM 记忆产物，独立重建 WorldMM 检索索引并运行 QA。**它不是从 RAW VIDEO 重新抽帧构建记忆的一次新运行。**

测试视频为 `6Z_qEtbmK34.mp4` 和 `Ei7hTKr8Ins.mp4`，各 40 道题。有效结果为 46/80，分别 24/40 和 22/40。失败题中 3 例经原帧、10 秒记忆与实际 Top-K 对照确认为检索漏召回；20 组 eSVD 语义冲突仅为待核实候选。QA 原始题目没有金标准证据时间戳，其他失败题不能据此指定 VSD、YOLO、eSVD、Event 或 Memory 为首次丢失位置。

## 在 200 上复现

脚本内的 `SERVICE`、`SOURCE_RUN` 和 `DATASET` 固定为当次测试的只读输入路径。请先确认路径存在，再将本目录脚本复制到独立实验目录运行，不在在线服务的工作目录里生成 `results/`：

```bash
cd /home/sscy/evaluations/hint-sd-hippovlog-20261009
/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog/.venv/bin/python build_trace.py
/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog/.venv/bin/python qa_trace.py --output results
/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog/.venv/bin/python make_report.py
```

`qa_trace.py` 按 `answers.partial.jsonl` 断点续跑。文字 embedding 使用已有 Qwen3-Embedding-4B 权重并放在 CPU；Qwen3-VL 与视觉检索在 GPU。Top-K 保持 episodic 3、semantic 10、visual 3。初次全 GPU 运行的 OOM 记录不计入成绩。

输出含逐原帧采样决策、VSD/YOLO/eSVD/Event/Memory provenance、逐次检索、QA 资源采样、异常列表和带原始帧的 HTML 报告。原始数据、模型、在线服务和生产数据库只读；大型输出、权重、缓存均不纳入 Git。历史上游耗时使用原日志，历史显存和各子阶段缺失的资源值不估算。

当次工程与文件哈希见 `BASELINE.md`。
