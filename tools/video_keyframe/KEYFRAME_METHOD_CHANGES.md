# WorldMM-a 关键帧方法对比与修改位置

## 对比基线

基线是 `WorldMM-a-keyframe-package.zip` 中的单文件
`worldmm_keyframe_pipeline.py`。原压缩包没有 KATNA 独立入口、NVDEC 适配器、
WebP 输出器、模型权重和阶段耗时统计，因此原方法的耗时不能从压缩包直接得到；
本文件按代码路径对比，并列出当前方法的实际长视频测试数据。

## 修改位置

| 文件 | 修改内容 |
|---|---|
| `katna/run_yolo_prefilter_event_webp.py` | 新增 10 FPS YOLO 批量初筛、事件合并、不稳定片段筛选、目标窗口 Katna、代表帧 NVDEC 解码和 WebP 写出。 |
| `katna/run_katna_yolo_single.py` | 新增 NVDEC 目标窗口适配；保留 LUV 差分 + Hanning 局部极值，支持 `scan_fps`。 |
| `katna/extract_keyframes.py` | 提取候选帧、Hanning、亮度和熵门控等 Katna 原语。 |
| `run_hybrid.sh` | 新方法独立启动入口，自动绑定本包 pipeline、Katna 目录和模型权重。 |
| `requirements-core.txt` | 新方法增加 `scikit-image`，用于局部熵质量门控。 |

`worldmm_keyframe_pipeline.py` 的语义模型定义和事件/轨迹数据结构保持兼容；新入口
复用其 `SemanticAnalyzer`，因此 YOLO/Pose 标签格式不变。

## 算法差异

### 原方法

1. `_scan_and_select()` 顺序读取视频，以候选帧率扫描。
2. 合格候选帧立即运行 YOLO、Pose 和可选语义分析。
3. 保留帧写 JPEG 原图和带框 JPEG。
4. `run()` 后续又调用 `analyze_dmd_boundaries()` 独立扫描视频。
5. 关键帧由时间间隔、视觉新颖度和信息增益控制，没有“每事件一张 WebP”输出。

### 新方法

1. 先以 10 FPS 低分辨率画面进行 YOLO batch GPU 初筛（默认 batch 16），得到
   对象标签、场景签名和相邻帧变化量。
2. 同标签短片段合并，默认最长 12 秒；静态区间不进入 Katna，用事件语义替代。
3. 仅将变化量第 75 百分位以上的片段扩展为目标时间窗，在窗口内以 10 FPS 运行
   LUV + Hanning 局部极值 Katna。
4. 亮度 10～90、局部熵 1～10，并增加 384 缩略图 Laplacian 方差 >=150 和
   Tenengrad 排序，主动排除运动模糊；不再使用 160 张硬限制。
5. 仅对最终代表帧按时间戳调用 FFmpeg CUDA/NVDEC，保持原分辨率；删除全分辨率
   精修和 WebP 写出阶段重复的 `cv2.VideoCapture`。
6. NVDEC 输出直接编码 WebP，不生成 JPEG；同一张 WebP 用于页面证据和 VLM 输入。
7. 每个事件只保留一张视觉代表帧，中间样本由事件时间范围、对象、动作和场景
   语义替代并写入 `semantic.json`。

## 运行方式与输出

```bash
./run_hybrid.sh \
  --video /path/to/video.mp4 \
  --video-id demo_video \
  --output ./output/demo_video \
  --device 0
```

```text
output/demo_video/
├── webp/event_00001.webp ...
├── semantic.json       # 事件范围和语义替代
├── frame_map.json      # 原帧/时间戳到 WebP 的映射
└── stats.json          # 阶段耗时、数量、编码字节数
```

## 实测数据

| 视频 | 时长 | YOLO 初筛 | Katna 目标窗 | NVDEC+WebP/语义写出 | 总耗时 | 事件/代表帧 |
|---|---:|---:|---:|---:|---:|---:|
| `BpVmNB3eKdM` | 1357.6 s | 71.1 s | 61.1 s | 23.5 s | 157.0 s | 204 / 204 |
| `Ei7hTKr8Ins` | 1548.7 s | 80.8 s | 68.9 s | 27.5 s | 178.4 s | 222 / 222 |

约为 **6.9～7.0 秒/小时视频** 的端到端速度。Katna 实际只处理 51/56 个不稳定
片段，其余约 1.34 万/1.53 万粗采样点由事件语义替代。

原始压缩包没有阶段计时字段，因此上表不能解释为“原方法固定耗时”；它是当前
实现的可复现实测结果。

## 注意

- `ffmpeg` 必须支持 CUDA/NVDEC；不可用时应显式报错，不应静默退回全量 CPU 扫描。
- 10 FPS 是初筛采样率，不是最终输出帧率；提高 `--scan-fps` 会增加 YOLO 初筛耗时。
- 本包不删除原视频；原视频生命周期由上层导入流程决定。
