# WorldMM-a 关键帧与语义提取包

这是从 `WorldMM-a` 整理出的独立交接包。它不包含视频数据集、模型权重或已有运行结果；安装脚本会在需要时创建本地虚拟环境，并可下载默认 YOLO 权重。

## 1. 安装

在本目录执行：

```bash
bash install.sh
```

默认会安装核心依赖，并把默认权重下载到：

```text
models/keyframe/yolo11n.pt
models/keyframe/yolo11n-pose.pt
```

只安装依赖、不下载权重：

```bash
DOWNLOAD_MODELS=0 bash install.sh
```

启用可选手部分析：

```bash
WITH_HANDS=1 bash install.sh
```

安装脚本不会读取或复制任何数据集。GPU 机器应使用与本机 CUDA 匹配的 PyTorch；安装结束时脚本会打印 `CUDA available` 检查结果。

## 2. 输入视频与运行接口

唯一必需输入是视频文件，接口入口是本目录的 `run.sh`。视频可以放在任意位置，不需要复制到工程目录：

```bash
./run.sh \
  --video /path/to/video.mp4 \
  --video-id demo_video \
  --output ./output/demo_video \
  --device 0 \
  --yolo-model ./models/keyframe/yolo11n.pt \
  --pose-model ./models/keyframe/yolo11n-pose.pt
```

参数说明：

```text
--video                 输入视频路径，必填
--output                本次任务输出目录，必填
--video-id              稳定的视频 ID；不传则使用文件名
--device                GPU 使用 0/1 等，CPU 使用 cpu
--width                 处理宽度，默认 640；<=0 保留原始尺寸
--sample-fps            候选扫描帧率，默认 10
--analysis-fps          DMD 分析帧率，默认 5
--yolo-model            本地目标检测权重
--pose-model            本地姿态权重
--world-model           可选 YOLO-World 权重
--world-classes         YOLO-World 类别，逗号分隔或文本/JSON 文件
--enable-hands          启用可选 MediaPipe 手部分析
--emotion-model         可选本地表情分类模型目录
--speech-json           可选 ASR 段落 JSON
--subtitle-srt          可选字幕文件
--disable-semantics     只抽帧，不加载 YOLO/姿态语义模型
--allow-model-download  允许运行时从模型名下载缺失权重
--max-scan-frames       烟雾测试限制帧数；0 表示完整视频
```

查看完整接口：

```bash
./run.sh --help
```

## 3. 关键帧、语义和 Memory 输出

例如 `--output ./output/demo_video` 后，输出为：

```text
output/demo_video/
├── job.json
├── memory.json
├── semantic/
│   ├── manifest.json
│   ├── frames.json
│   ├── rejected_frames.json
│   ├── view_transitions.json
│   ├── tracks.json
│   ├── events.json
│   ├── events_flat.json
│   └── event_hierarchy.json
├── research/
│   ├── technical_curves.csv
│   ├── research_metrics.json
│   └── summary_keyframes.json
└── keyframe_package/
    ├── original/
    ├── labeled_tracks/
    ├── keyframes.json
    └── worldmm_visual_records.jsonl
```

最常用文件：

- `keyframe_package/original/`：保存的关键帧图片；
- `keyframe_package/labeled_tracks/`：带检测框、轨迹和动作标签的图片；
- `keyframe_package/keyframes.json`：关键帧索引、时间戳、质量分数和语义；
- `semantic/frames.json`：完整关键帧语义证据；
- `semantic/tracks.json`：跨帧目标轨迹；
- `semantic/events.json`：Scene → Event → Object/Action/Expression 层级事件；
- `keyframe_package/worldmm_visual_records.jsonl`：可直接对接 WorldMM Visual Memory 的逐行记录；
- `memory.json`：本任务的统一入口和文件索引。

该包负责“关键帧 + 语义 + 事件 + WorldMM Visual Records”构建；三层 Memory 的最终写入和 QA 仍由上层 WorldMM 流程负责。

## 7. KATNA 入口（推荐速度测试）

本交接包同时包含 KATNA + YOLO/Pose 入口：

```bash
./run_katna.sh \
  --video /path/to/video.mp4 \
  --video-id demo_video \
  --output ./output/demo_katna \
  --katna-engine gpu \
  --device 0 \
  --yolo-model ./models/keyframe/yolo11n.pt \
  --pose-model ./models/keyframe/yolo11n-pose.pt
```

KATNA 流程为 LUV + Hanning 局部极值、无限关键帧质量筛选、YOLO/Pose 语义分析。保存原图时采用一次顺序解码，避免对 H.264 视频逐帧随机 seek 导致长视频重复 GOP 解码。默认检测输入宽度为 640，可通过 `--semantic-width` 调整。

`install.sh` 默认安装 CUDA 12.6 PyTorch；如果机器使用其他 CUDA 轮子，可设置 `TORCH_INDEX_URL`。包内已经提供 `yolo11n.pt` 和 `yolo11n-pose.pt`，不会因为模型缺失触发运行时下载。

## 4. GPU、CPU 与快速测试

GPU：

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh \
  --video /path/to/video.mp4 \
  --output ./output/demo_video \
  --device 0 \
  --yolo-model ./models/keyframe/yolo11n.pt \
  --pose-model ./models/keyframe/yolo11n-pose.pt
```

CPU：

```bash
./run.sh --video /path/to/video.mp4 --output ./output/demo_cpu --device cpu
```

无语义烟雾测试：

```bash
./run.sh --video /path/to/video.mp4 --output ./output/smoke \
  --disable-semantics --max-scan-frames 300
```

如果模型文件不在本地，可显式使用 `--allow-model-download`；生产运行建议传入固定的本地权重路径，避免模型版本变化。

## 5. 算法说明与降级行为

流程包含候选帧调度、清晰度/抖动门控、信息增量选帧、DMD 内容边界、场景切分、YOLO 目标检测、YOLO Pose 动作规则、跨帧轨迹和事件聚合。对容易误判的目标，代码对 `person/chair/cup/bottle/book/cell phone/remote` 使用更严格的保存置信度；低置信度检测不会写入关键帧语义。

缺少 YOLO、Pose、手部或表情模型时，任务会记录组件状态并安全降级，仍然输出质量抽帧和 JSON 文件，不会伪造语义结果。运行日志和实际加载的组件见 `semantic/manifest.json`。

## 6. 文件接口位置

```text
run.sh                              同事实际调用的命令行入口
run_katna.sh                        KATNA + YOLO/Pose 入口
worldmm_keyframe_pipeline.py       算法实现和 argparse 接口
katna/run_katna_yolo_single.py     KATNA 单视频入口
katna/extract_keyframes.py         KATNA 扫描和顺序保存实现
install.sh                          独立安装脚本
requirements-core.txt               核心依赖
requirements-hands.txt              可选手部依赖
models/keyframe/                   权重放置目录（压缩包内为空）
```
