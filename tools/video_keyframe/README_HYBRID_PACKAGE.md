# WorldMM-a Hybrid Keyframe Package v2

本包只负责关键帧提取、清晰度筛选、事件语义和 WebP 证据生成，不包含视频数据集、
Memory/QA 业务代码或运行结果。

## 安装

```bash
DOWNLOAD_MODELS=0 bash install.sh
```

压缩包内已放置 `models/keyframe/yolo11n.pt` 和 `yolo11n-pose.pt`；若需重新准备
模型可使用 `DOWNLOAD_MODELS=1 bash install.sh`。

## 运行新方法

```bash
./run_hybrid.sh \
  --video /path/to/video.mp4 \
  --video-id demo_video \
  --output ./output/demo_video \
  --device 0
```

常用调参：

```text
--scan-fps 10                    YOLO 初筛采样率
--yolo-batch-size 16             GPU 批量大小
--katna-scan-fps 10              目标窗口 Katna 采样率
--katna-unstable-percentile 75   只对高变化片段做 Katna
--merge-max-sec 12               同标签事件合并上限
--webp-quality 80                WebP 质量
--target-decode-workers 4        最终帧 NVDEC 并行数
```

输出 `webp/`、`semantic.json`、`frame_map.json` 和 `stats.json`。每个事件默认一个
代表 WebP；没有视觉代表帧的稳定区间仍保留在事件语义中。

## 基线与差异

- 原方法入口：`run.sh` → `worldmm_keyframe_pipeline.py`。
- 新方法入口：`run_hybrid.sh` → `katna/run_yolo_prefilter_event_webp.py`。
- 详细的修改文件、算法差异、实测耗时和限制见
  `KEYFRAME_METHOD_CHANGES.md`。
