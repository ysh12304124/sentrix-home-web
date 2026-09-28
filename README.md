# Sentrix Home

Sentrix Home is a local-first household memory system. It turns original media
into evidence-backed episodic, semantic, and visual memory. The canonical
product definition, architecture, data contracts, acceptance results, and
current work queue are maintained in [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md).

## Repository Layout

- `backend/`: FastAPI application, SQLite memory store, media pipeline, model
  adapters, identity clustering, and Python regression tests.
- `src/`: browser application code and styles.
- `scripts/runtime/`: project-local runtime utilities.
- `scripts/maintenance/`: destructive or long-running maintenance commands.
- `scripts/benchmarks/`: controlled clustering evaluation tools.
- `scripts/fixtures/`: reproducible public test-data and metadata generators.
- `test/`: Node-based frontend and repository-layout regression tests.
- `docs/`: live project memory and approved design/implementation records.

## 运行

三台机器（**153** 主机 / **118** AGX Orin / **46** Orin NX）跑**同一份代码**。
默认不需要任何启动参数，机器差异分两层处理：

| 层 | 位置 | 负责 |
|---|---|---|
| **能力探测** | `backend/platform_profile.py` | CUDA 是否**真能用**（实跑 matmul + conv2d 验证，不只看 `torch.cuda.is_available()`）、ffmpeg 编译了哪些硬件解码器、内存多大、装没装 qdrant-client |
| **平台参数** | `scripts/deploy/platform-profiles.sh` | 探测不出来的那部分 —— 「这台机器上我们选择这么做」 |

```bash
./start.sh            # 起 API + Web（默认 8090 / 4174）
./start.sh -r         # 重启
```

启动日志会打印这台机器**实际生效**的能力摘要。改完配置先看这几行，比翻代码快：

```
INFO sentrix.platform_profile: platform_profile jetson = True
INFO sentrix.platform_profile: platform_profile cuda_usable = True
INFO sentrix.platform_profile: platform_profile memory_total_mib = 15655
INFO sentrix.platform_profile: platform_profile ffmpeg_hw_decoders = ['h264_nvv4l2dec', ...]
INFO sentrix.platform_profile: platform_profile clip_device = cuda
INFO sentrix.platform_profile: platform_profile video_device = 0
INFO sentrix.platform_profile: platform_profile face_embedding_mode = legacy
INFO sentrix.platform_profile: platform_profile vector_backend = qdrant
INFO sentrix.platform_profile: platform_profile pipeline_workers = 2
```

### 三台的参数差异

平台按 `/proc/device-tree/model` 自动识别，下表是各自的默认值：

| 参数 | 153（host） | 118（orin-agx） | 46（orin-nx） |
|---|---|---|---|
| `SENTRIX_PLATFORM` | `host` | `orin-agx` | `orin-nx` |
| `CLIP_DEVICE` | **`cpu`** | `cuda` | `cuda` |
| `FACE_EMBEDDING_MODE` | `adaface` | `legacy`（buffalo_l） | `legacy`（buffalo_l） |
| `PIPELINE_MAX_WORKERS` | 12 | 8 | 2 |
| 推理后端 | vLLM | llama.cpp | llama.cpp |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` | `qdrant` |
| `VIDEO_KEYFRAME_ALGORITHM` | `hybrid_webp` | `hybrid_webp` | `hybrid_webp` |

**为什么 153 的 CLIP 走 CPU**：它的 GPU 驱动与 NVML 不匹配，会让 CUDA caching
allocator 崩。注意 CUDA 实跑自检**是能过的** —— 所以这一条探测不出来，只能显式写在
平台层。（118 的提交 `15227b8` 记着「原 cpu 是照搬 153 的驱动问题」，反过来说明
这条只对 153 成立，Orin 上不该照抄。）

**为什么 Orin 的人脸用 buffalo_l 而不是 AdaFace**：AdaFace 是 700MB fp32 PyTorch
权重，16GB 统一内存里它是压垮内存的那一根稻草；buffalo_l 是 174MB ONNX。注意
`face_embeddings.py` 的契约**禁止跨模型静默回退**（否则存进库的向量与实际模型
不符），所以这里必须显式选，不能靠自动降级。

### 覆盖与排障

平台层只写**默认值**：`.env`、命令行、以及 `SENTRIX_PLATFORM` 都能覆盖它。

```bash
SENTRIX_PLATFORM=orin-nx ./start.sh        # 强制按某个平台起（排障用）
SENTRIX_PIPELINE_MAX_WORKERS=4 ./start.sh  # 单项覆盖
```

### 向量后端与索引

`qdrant` 与 `hnswlib` **都只是派生索引**，权威数据始终是 `sentrix.db` 的
`memory_vectors` 表（见 `backend/qdrant_memory.py` 的说明）。三台默认都用
qdrant（embedded 模式，`QdrantClient(path=...)`，不需要独立服务端）。

两者的**生效方式不同**，这条差异会直接影响运维：

| 后端 | 写完向量之后 |
|---|---|
| **qdrant** | 立即生效（`upsert_vector` 双写） |
| **hnswlib** | 必须显式重建 `data/ann/*.hnsw`，否则检索读到的还是旧图 |

收尾钩子按后端自动决定要不要重建（`qdrant` 时跳过）。检索侧 qdrant 是**镜像**：
查不到结果会继续回退到 HNSW，不会因为一次索引抖动就静默返回空。

手工重建（仅在 hnswlib 后端需要）：

```bash
.venv/bin/python scripts/maintenance/rebuild_ann_indices.py \
    --db data/sentrix.db --ann-dir data/ann --apply
```

> **不要**加 `--visual-embedder chinese_clip`：那一支会绕开数据库、给**全库**图片
> 重新跑一遍 chinese-clip，而收尾时刚算过完全相同的向量。实测 465 张图要 20 分钟
> + 2.3GB 内存，且与向量回填串在同一线程里 —— 这是 46 上一次 OOM 的直接来源。

### 推理服务

VLM 与 BGE 嵌入服务由各自的启动脚本负责（153 是 vLLM，Orin 是 llama.cpp）：

```bash
scripts/runtime/start_sentrix_api.sh    # API（端口由 SENTRIX_API_PORT 决定）
scripts/runtime/start_sentrix_web.sh    # Web 网关
```

模型权重路径按 `$HOME` 推导（`~/benchmarks/retinaface/...`、仓库内
`models/AdaFace/...`），不再写死某个用户名；用 `RETINAFACE_MODEL_PATH` /
`ADAFACE_MODEL_PATH` 覆盖即可。

## Verify

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
npm test
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
```

Video imports require `ffprobe` on `PATH`. The repository vendors the supplied
WorldMM-a pipeline and its fixed YOLO/Pose weights under `tools/video_keyframe/`;
runtime tuning uses `SENTRIX_VIDEO_WIDTH`, `SENTRIX_VIDEO_SAMPLE_FPS`,
`SENTRIX_VIDEO_ANALYSIS_FPS`, `SENTRIX_VIDEO_DEVICE`, and
`SENTRIX_VIDEO_MAX_KEYFRAMES`. WorldMM's complete `memory_keyframes` remain
available in the derived output; Sentrix uses the package's recommended
`research/summary_keyframes.json` and imports at most the configured number of
representative frames (160 by default).

The maintenance rebuild is intentionally explicit because it replaces derived
memory data:

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py --root . --source /path/to/source-album
```
