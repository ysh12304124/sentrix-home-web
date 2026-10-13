# yxw2 同步 main 与 200 服务器验证记录

日期：2026-10-13（北京时间）。

## 合并范围

- 原 yxw2：`25a0f3fe965373258c0643f7d6d2f38de77d8026`。
- 纳入 main：`6ca19e0958571efa8c2fc238db0403fe2d999e6a`。
- 合并前 main 独有 107 个提交，yxw2 独有 4 个提交；采用保留双方历史的 merge。
- 保留 SVD → 稀疏 YOLO 事件划分 → 事件内 eSVD、事件级 VLM 选帧、WebP 入库，以及 HippoVlog 服务。
- 纳入主线图记忆、检索、视频来源信息、运行适配和 PhotoBench 更新。

冲突处理保留主线原视频预览与 yxw2 事件上下文；同时保留主线 VLM 图片上限、上下文预算和空描述恢复逻辑，以及 yxw2 语义状态/状态转换索引映射。

SVD 入库路径同步传递主线新增的 `capture_time_source`、`location_source`、`location_provenance`、`source_media_id` 和 `source_media_title`，记录事件视觉耗时及状态转换。全局配置和单资产 `keyframe_algorithm=svd_lowrank` 都可进入该路径。

## 真实抽帧回归

在 200 服务器的隔离目录执行，使用 RTX 3090、同一 Python 环境与 YOLO 权重。两侧均设 `SENTRIX_VIDEO_DEVICE=0`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`，逐段顺序运行。未修改既有服务部署。

比较原 yxw2 与合并结果。10 段视频两侧全部成功，YOLO 状态均为 `complete`，均记录事件内第二次 SVD 折叠。事件边界、关键帧时间、选中索引、YOLO 观察记录与 JPEG 文件 SHA-256 全部一致，共 15 个事件、15 张提取候选帧。

| 视频 | 事件/候选帧 | 原 yxw2 秒 | 合并后秒 | 合并后进程树 RSS 峰值 MiB |
|---|---:|---:|---:|---:|
| video-001 | 1 / 1 | 3.400 | 3.151 | 1519.8 |
| video-003 | 1 / 1 | 3.130 | 3.348 | 1516.6 |
| video-005 | 1 / 1 | 3.115 | 3.171 | 1356.7 |
| video-006 | 1 / 1 | 3.199 | 3.152 | 1376.5 |
| video-007 | 2 / 2 | 3.561 | 3.779 | 1497.8 |
| video-012 | 2 / 2 | 3.357 | 3.180 | 1364.4 |
| video-016 | 1 / 1 | 3.396 | 3.132 | 1369.4 |
| video-021 | 2 / 2 | 3.123 | 3.361 | 1408.7 |
| video-025 | 2 / 2 | 3.365 | 3.610 | 1365.5 |
| video-026 | 2 / 2 | 3.171 | 3.128 | 1372.7 |
| 样本均值 | — | 3.282 | 3.301 | 1414.8 |

原 yxw2 的进程树 RSS 峰值样本均值为 1420.9 MiB。RSS 按约 0.2 秒采样并汇总父子进程，可能重复计算共享页，不能当作显存或整机独占内存。本轮单次配对用于排查合并退化，不能据微小耗时差异判断性能优劣。

此处的真实运行范围是提取阶段；不含事件 VLM、WebP 入库、嵌入、图构建和全量 QA。入库接口由下述受控测试验证。

## 自动检查与已知限制

在相同 200 验证环境下：

- 视频、SVD 记忆接口、模型客户端、人物预览、图检索与来源契约的定向集合：93 通过、8 失败。
- 这 8 项在未修改的 main 同环境全部复现，涉及模型默认 token/图片尺寸、旧 HTTP mock、计时 mock 和缺少 `get_asset` 的人物测试夹具。相应 main 集合为 34 通过、8 失败。
- PhotoBench suite control / retrieval ranking：30 通过、3 失败；main 同集合也是 30 通过、同样 3 失败，涉及模型管理器配置与当前模型选择假设。
- Mac 前端 `npm test`：48 通过、4 失败；main 同环境结果完全一致，均为既有 UI 源码断言。
- Python 编译检查通过；本次冲突处理和新增文件的 diff 空白检查通过。主线自带的文档尾空行和生成 HTML 的 CRLF 没有改写。

新增回归覆盖：图片上限后语义状态及转换索引回映射；上下文压缩后来源索引一致；单资产 SVD 路由覆盖全局 hybrid 配置；两个 YOLO 事件独立入库、实际 WebP 文件生成及来源信息继承。

因此，本轮没有发现合并新增的测试失败，但项目现有测试集并非全绿，也未重新执行真实 VLM 的完整记忆/QA 评测。

关键代码与回归入口：

- [SVD 提取器](../backend/video/svd_keyframe.py)
- [视频处理器](../backend/video/processor.py)
- [VLM 图片与状态映射测试](../backend/tests/test_video_event_image_limit.py)
- [SVD 记忆接口测试](../backend/tests/test_svd_memory_integration.py)
