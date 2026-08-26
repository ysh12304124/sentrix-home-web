# 视频关键帧记忆构建方案

## 1. 实测结论

本次对两个 HippoVlog 视频使用同一条链路：

`10 FPS YOLO 批量初筛 → 事件合并 → 高运动片段 Katna → NVDEC 目标帧解码 → 全分辨率 WebP → 每事件一张记忆图`

| 视频 | 时长 | 事件/记忆帧 | 总耗时 | 折算每小时 | WebP 大小 | 40 QA |
|---|---:|---:|---:|---:|---:|---:|
| BpVmNB3eKdM | 1357.59 秒 | 204 | 156.995 秒 | 416.3 秒 | 22.72 MiB | 31/40（77.5%）|
| Ei7hTKr8Ins | 69.60 秒 | 9 | 11.087 秒 | 573.5 秒* | 1.87 MiB | 30/40（75.0%）|

\* 短视频的每小时折算会被模型初始化和固定启动开销放大，不能与长视频线性比较。

QA 使用的是项目现有的 40 题结构化测试集。失败题主要是测试集内置的 2024 年历史照片以及“明哥/乐乐”实体题，与这两个视频的真实标注范围不匹配，不是关键帧生成失败。

## 2. 端到端数据流

```text
上传 H.264/H.265
       │
       ├─ ffprobe：分辨率、FPS、帧数、时长、编码格式
       │
       ├─ 10 FPS 粗采样
       │    └─ YOLO detector，batch=16，输入宽度=640
       │         └─ 目标标签、帧间变化、清晰度、时间戳
       │
       ├─ 事件切分与合并
       │    ├─ 目标/标签签名变化切段
       │    ├─ 单段最长约 20 秒
       │    └─ 相同主标签、相邻段最多合并 12 秒
       │
       ├─ 片段运动分数
       │    └─ 帧间灰度差 P90，取分数最高的 25% 片段
       │
       ├─ 高运动片段进入 Katna
       │    ├─ NVDEC/CUDA 解码
       │    ├─ 384×216、10 FPS、LUV 差分
       │    ├─ Hanning 平滑局部极大值
       │    └─ 亮度/熵/Laplacian/Tenengrad 清晰度门控
       │
       ├─ 稳定片段直接选择粗采样中最清晰的一帧
       │
       ├─ NVDEC 按代表帧时间点并行解码原分辨率
       │    └─ 1920×1080 BGR → WebP quality=80
       │
       ├─ frame_map.json + semantic.json
       │    └─ 源帧号、源时间、事件、对象、动作、替代帧数
       │
       └─ Sentrix 记忆层
            ├─ 每个事件只保留一张 WebP 代表图
            ├─ 长时间未变化帧由事件语义替代
            ├─ Chinese-CLIP 图像向量
            └─ Qwen3-VL/VLLM 视觉描述与 QA
```

## 3. 各模块职责与关键参数

### 3.1 视频探测与时间索引

调用 `ffprobe` 获取 `width/height/fps/frame_count/duration/codec`。所有后续结果同时保存 `source_frame_index` 和 `source_timestamp_sec`，因此 WebP 代表帧可以准确回跳原视频，不依赖压缩后视频的帧号。

### 3.2 YOLO 初筛

当前实测使用 Ultralytics YOLO detector，10 FPS、batch=16、640 宽输入、GPU 0。采样步长使用 `ceil(source_fps / 10)`，避免 29.97 FPS 被错误采成 12 FPS。

YOLO 初筛不保存 JPEG，也不把所有帧送入视觉语言模型。每个采样点只保留内存中的检测标签、帧间变化和清晰度统计。稳定画面直接使用粗采样的清晰帧，减少后续 Katna 和 VLM 工作量。

### 3.3 事件切分、合并与语义替代

检测标签签名发生变化或片段超过 20 秒时切段；相邻片段主标签相同且合并后不超过 12 秒时合并。最终每个事件只输出一张代表图，事件内被省略的帧数量记录为 `substituted_sample_count`，由事件时间区间、对象和动作语义替代。

本次结果中：

- Bp：13,359 个粗采样点被事件语义替代。
- Ei：571 个粗采样点被事件语义替代。

这解决了“事件太多”和“长时间静止画面重复送模型”的问题，同时保留事件时间范围和语义信息。

### 3.4 Katna 清晰度筛选

Katna 只处理运动分数位于最高 25% 的片段，不再对整段视频做完整质量扫描。处理分辨率为 384×216，扫描速率 10 FPS。候选选择包含：

- LUV 帧差 + Hanning 平滑 + 局部极大值。
- 亮度范围：10～90。
- 熵范围：1～10。
- 384 宽图像 Laplacian 方差至少 150。
- Tenengrad 作为二级锐度排序。

如果一个稳定片段没有 Katna 候选，回退到 YOLO 粗采样中 Laplacian 最清晰的帧；因此不会因为质量门控导致事件没有代表图。

### 3.5 NVDEC 与 WebP

最终代表帧使用 FFmpeg CUDA/NVDEC 按时间点解码，4 个并行 worker，保持原始 1920×1080 分辨率。解码后的单帧直接编码为 WebP quality=80，不生成 JPEG，不保存整段原始视频帧图像。

WebP 既用于页面显示，也用于视觉向量和 VLM 输入；页面不需要从 JPEG 或原视频重新抽帧。`frame_map.json` 负责 WebP 与原视频时间点的映射。

## 4. 硬件、软件与调用

### 硬件

- 服务器：192.168.0.200。
- GPU：NVIDIA GeForce RTX 3090，24 GiB。
- NVIDIA 驱动：535.230.02。
- GPU 0：NVDEC、YOLO、CLIP、Qwen3-VL 共用；生产环境应通过任务队列避免视频解码和 VLM 同时争抢显存。

### 软件版本

- Python 3.10.20。
- PyTorch 2.8.0+cu128，CUDA 12.8。
- Ultralytics 8.4.60。
- OpenCV 4.13.0，NumPy 2.2.6。
- FFmpeg 4.2.7，使用 `-hwaccel cuda -hwaccel_output_format cuda`。
- YOLO 权重：`yolo11n.pt`。
- 姿态权重：`yolo11n-pose.pt`，用于代表帧/事件级 Pose 扩展。
- 视觉向量：Chinese-CLIP，生产配置为 `CLIP_DEVICE=cuda:0`。
- 视觉描述：200 服务器当前配置为 Qwen3-VL 4B 本地 CUDA 推理；如切换 VLLM，使用 OpenAI-compatible `/v1` 接口即可，不改变上层事件和证据协议。

### 主要程序调用

```text
ffprobe
  → tools/video_keyframe/katna/run_yolo_prefilter_event_webp.py
      → Ultralytics YOLO batch predict
      → Katna gpu_katna_candidates
          → ffmpeg NVDEC + hwdownload + LUV/Laplacian/Tenengrad
      → ffmpeg NVDEC -ss 目标时间点
      → cv2.imencode(.webp)
      → frame_map.json / semantic.json
  → tools/video_keyframe/import_event_package.py
      → backend.video.processor
      → Chinese-CLIP embedding
      → Qwen3-VL/VLLM
      → SQLite events/assets/memory_vectors
  → /api/assistant/turn
      → structured retrieval + evidence answer
```

## 5. 两个视频的详细耗时

### BpVmNB3eKdM

| 模块 | 耗时 | 占总耗时 | 结果 |
|---|---:|---:|---:|
| YOLO 10 FPS 批量初筛 | 71.056 秒 | 45.3% | 13,563 个采样点，batch=16 |
| 高运动片段 Katna | 61.148 秒 | 38.9% | 51 个不稳定片段、52 个窗口 |
| 目标帧 GPU 解码 | 23.325 秒 | — | 204 张全分辨率目标帧 |
| WebP/包写入阶段 | 23.538 秒 | 15.0% | 204 张，22.72 MiB |
| 完整关键帧链路 | **156.995 秒** | 100% | 204 个事件 |

`gpu_target_decode_sec` 是包写入阶段中的子阶段，不应与 `package_write` 再相加。

### Ei7hTKr8Ins

| 模块 | 耗时 | 占总耗时 | 结果 |
|---|---:|---:|---:|
| YOLO 10 FPS 批量初筛 | 5.467 秒 | 49.3% | 580 个采样点，batch=16 |
| 高运动片段 Katna | 2.483 秒 | 22.4% | 3 个不稳定片段、3 个窗口 |
| 目标帧 GPU 解码 | 1.931 秒 | — | 9 张全分辨率目标帧 |
| WebP/包写入阶段 | 1.940 秒 | 17.5% | 9 张，1.87 MiB |
| 完整关键帧链路 | **11.087 秒** | 100% | 9 个事件 |

## 6. 当前瓶颈与解决方案

### 瓶颈一：视频解码与 CPU-GPU 拷贝

旧方案在清晰度精修和 WebP 写入阶段重复使用 `cv2.VideoCapture`，会重复顺序读取和重复 resize。当前方案已经删除全分辨率精修，最终帧改为 NVDEC 按时间点并行解码；Katna 只把 384 宽窗口传回 CPU。

当前粗采样阶段仍使用低分辨率 `cv2.VideoCapture` 顺序读取，这是为了稳定获得 10 FPS 时间轴；它不再重复做全分辨率清晰度扫描。下一步若需要继续压缩 71 秒左右的 YOLO 阶段，应将粗采样也改成 FFmpeg CUDA `fps/select` 输出，或按 8～12 秒窗口使用并行 `-ss` 解码。

### 瓶颈二：Katna 对整段视频计算过慢

高运动分位数筛选把 Katna 限制在不稳定片段；稳定片段直接选 YOLO 粗采样的锐度最大帧。Bp 的 Katna 为 61.148 秒，Ei 为 2.483 秒；不会再为长时间静止背景计算全部局部极大值。

### 瓶颈三：模糊帧进入记忆

不能只按颜色方差选帧。当前用 Laplacian 和 Tenengrad 衡量焦点清晰度，用 LUV 帧差定位动作，再加亮度/熵门控。运动片段即使动作信息重要，也优先选择动作发生前后最清晰的帧，避免把运动拖影送入 VLM。

### 瓶颈四：事件过多、VLM 调用次数过多

通过相同标签片段合并、12 秒合并上限、稳定片段语义替代和“每事件一张图”，把长视频的代表帧固定在事件数规模，而不是 10 FPS 采样点规模。VLM/CLIP 只处理代表图；事件内其余帧保留时间范围和检测语义，不再逐帧调用模型。

### 瓶颈五：视觉向量与图像格式脱节

WebP 是唯一的记忆图像来源：页面显示、Chinese-CLIP 向量、VLM 输入和证据回跳都使用同一张 WebP。这样避免 JPEG 临时文件、重复压缩和“页面看到的图”与“向量使用的图”不一致。

### 瓶颈六：模型显存互相抢占

建议生产调度分为两个阶段：

1. 视频阶段独占 GPU：NVDEC、YOLO、Katna 目标解码。
2. 记忆阶段批量处理 WebP：CLIP embedding、Pose/YOLO-World、Qwen3-VL/VLLM。

如果必须并发，应固定 CUDA stream、限制 VLLM `max_num_seqs`，并给视频任务预留 NVDEC/显存水位；不能让每个代表帧重新加载模型。

## 7. 生产验收标准

- 不生成 JPEG；代表图格式为 WebP，分辨率保持源视频分辨率。
- `frame_map.json` 中每个事件有源帧号和源时间。
- 每个事件只有一张送入 VLM 的代表图。
- 任何事件都不能因 Katna 质量门控而丢失代表帧。
- 页面回放时间点与原视频误差小于一个源帧。
- 结构化 QA、视觉语义 QA 和向量检索都能回到同一事件证据。
- 长视频耗时主要由 YOLO 初筛和代表帧视觉记忆构建组成，不再由重复的全分辨率 OpenCV 扫描组成。
