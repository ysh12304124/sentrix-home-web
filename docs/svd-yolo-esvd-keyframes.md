# SVD / YOLO / eSVD 视频关键帧

该实现把视频处理替换为内容自适应的两阶段关键帧流程，照片处理、记忆构建和 QA 链路保持不变。

## 流程

1. 以配置帧率采样视频，用局部低秩 SVD 折叠近重复帧。
2. 只对第一阶段保留的代表帧运行 CUDA YOLO，生成物体、人物与布局变化时间线。
3. 在各语义事件内部运行第二阶段 eSVD，并根据低秩残差、子空间变化、谱变化和时间覆盖生成候选帧。
4. 结合检测目标、人物组合、画面结构和光流，对连续目标保留全局最佳帧，对人物进入、离开及高信息动作保留独立状态。
5. 事件级 VLM 从临时证据图中选择最少的非重复代表帧，最终只持久化 WebP，并沿用现有记忆构建和 QA 流程。

事件数量和代表帧数量由视频内容变化决定，不按视频时长设置固定总量。

## 启用

```bash
export SENTRIX_VIDEO_KEYFRAME_ALGORITHM=svd_lowrank
export SENTRIX_VIDEO_DEVICE=0
export SENTRIX_VIDEO_PYTHON=/path/to/python
# 可选；默认读取仓库根目录的 svd_config.yaml
export SENTRIX_SVD_CONFIG_PATH=/path/to/svd_config.yaml
```

YOLO 是生产路径的必需阶段。默认模型路径沿用
`tools/video_keyframe/models/keyframe/yolo11n.pt`，也可通过
`SENTRIX_VIDEO_YOLO_MODEL` 覆盖。若 YOLO 不可用，流程会明确失败，不会静默退化为纯视觉结果。

未设置 `SENTRIX_VIDEO_KEYFRAME_ALGORITHM` 时，`yxw2` 仍使用原有
`hybrid_webp` 路径。也可以在单个视频资产的 `metadata_json.keyframe_algorithm`
中指定 `svd_lowrank`。

## 主要文件

- `backend/video/svd_keyframe.py`：两阶段 SVD/eSVD、内容自适应分段和候选选择。
- `backend/video/yolo_event_worker.py`：第一阶段代表帧的稀疏 CUDA YOLO 扫描。
- `backend/video/processor.py`：事件级语义选择、WebP 持久化和记忆链路接入。
- `svd_config.yaml`：默认算法参数。

当前版本是已经完成 PhotoBench 对比评测的候选版本。合并为默认生产算法前，仍建议在不同长度、不同镜头运动和不同人物密度的视频集上做扩展回归。
