# M3 家庭场景：独立视频测评

## 固定版本与样本

算法基线为 `yxw2` 的 `10a5874318b1eb6bcfb085abf2dc5fa77ad9a5a2`；本适配只增加数据准备与测评文件，不改关键帧、检索或记忆算法。

`selection.lock.json` 锁定官方代码提交、数据集版本、标注 SHA256、原片 SHA256 和选片清单。8 段完整视频，121 道官方题，未用改写增加题量：

| 场景 | 视频 | 官方题 |
|---|---|---:|
| 厨房 | kitchen_05 / kitchen_12 / kitchen_15 | 15 / 20 / 14 |
| 客厅 | living_room_10 / living_room_17 / living_room_20 | 12 / 15 / 15 |
| 书房 | study_07 / study_17 | 13 / 17 |

只测这些视频。使用独立相册 `m3-family-1gb`、独立数据库和新的 benchmark scope，不合并原家庭相册、不复用其记忆。题集为 `m3-official-121q`。

## 数据与协议边界

- [M3-Agent 官方仓库](https://github.com/ByteDance-Seed/m3-agent)，[M3-Bench 数据集](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)。数据许可为 [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。代码许可与数据许可不同。
- 原片约 3.376 GB。统一 H.264 960×540、15 fps、AAC 64 kbit/s，保留完整时长与音轨，目标约 1 GB（十进制）。不截取视频片段。实际码率、字节数、时长、校验值由转码验证清单记录。
- 降分辨率与帧率可能影响细节和短暂动作；不同算法必须使用同一组转码视频，不能将原片与转码片直接当算法对照。
- 官方 `timestamp` 是提问时刻，`before_clip` 控制官方流程能访问的历史；它们**不是证据发生时刻**。
- 当前适配协议是 `full_video_album_adaptation`：完整视频建立记忆，然后问答。不能把结果称为官方 M3 因果协议得分。
- 问题仅前置中性视频名称上下文，保留英文原题 `source_question`、原始答案及完整 `source_annotation`。不伪造中文翻译、日期、GPS 或片段证据标注。
- GT 为视频级归属；关键帧通过父视频映射。命中视频不能证明定位到正确片段，应结合回答与证据核验解释。
- 答案与 reasoning 只在离线评测文件内。导入元数据仅提供中性来源 ID/标题，不导入问答或推理链。
- 官方有 21 题带 Cross-Modal Reasoning 标签，但无此标签也不能推断为纯视觉题。yxw2 视频关键帧链路未接入视频语音转写；保留音轨不意味着已理解音频。不把能力不足的题改成不可回答题。
- 121 题只来自 8 段视频。汇报逐题分数与逐视频分数；小幅提升需要更多独立视频验证。

## 准备方法

先按锁文件的固定 URL 下载原片，取得固定版本的 `data/robot.json` 官方标注；脚本验证 SHA256。示例路径均由运行者传入：

```bash
python services/photobench/scripts/transcode_m3_family.py \
  --source ./m3-downloads --output ./m3-encoded \
  --lock services/photobench/datasets/m3-family/selection.lock.json

python services/photobench/scripts/prepare_m3_family.py \
  --annotations ./m3-agent/data/annotations/robot.json \
  --lock services/photobench/datasets/m3-family/selection.lock.json \
  --encoded-dir ./m3-encoded \
  --validation ./m3-encoded/encoding-validation.json \
  --output services/photobench/data/m3-family-1gb
```

转码脚本需要 FFmpeg、ffprobe 与可用的 h264_nvenc。拒绝覆盖已有正式输出。准备脚本检查标注/视频校验、音轨、时长差（≤0.25 秒），以临时目录构建后原子发布，拒绝覆盖现有相册。

启动独立 yxw2 后端及 PhotoBench，选择该相册与 121 题，使用单模型完整流程（导入→记忆构建→QA→Judge）。外部单模型端点无需 Manager；Manager 生命周期/遥测记为不适用。测试完成前不能把数据准备验证当问答成绩。

测试：`python -m unittest discover -s services/photobench/tests -p test_m3_family_adapter.py -v`。

## 本轮运行设置

200 上使用独立 yxw2 服务与当前本地 `qwen3.5-4b-lora` 端点。初轮 Judge 使用同一本地模型，属于自评，不能替代独立 Judge 或人工复核。开启证据核验；保留原始回答、工具轨迹、时延及 token 记录。
