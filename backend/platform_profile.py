"""运行时能力档案 —— 平台差异靠探测，不靠启动参数。

为什么要有这个模块
------------------
同一份代码要跑在三种机器上：153（x86 + 独立 GPU）、118（AGX Orin 64GB 统一内存）、
46（Orin NX 16GB 统一内存）。早期做法是每台机器各维护一份 .env，结果是三份配置
各自漂移、互相不知道对方改了什么，新增一台机器就要再配一遍。

真实事故：46 上 `.env` 少了一个 `SENTRIX_VIDEO_KEYFRAME_ALGORITHM`，视频处理就
**静默回退**到另一条（worldmm）链路，而所有针对 hybrid_webp 写的 NVDEC 修复
全部失效 —— 从日志上看不出任何异常。

这里的约定
----------
1. 探测给出**安全默认值**；环境变量仍然可以**覆盖**它（排障、做对照实验），
   但不再是**必填**。所以「从 153 拷代码到 46」不需要任何启动参数。
2. 所有探测用 `lru_cache` 缓存，可重复调用。
3. 探测失败一律回落到保守值，并 `logging.warning` 留痕 —— 不静默降级。

消费方式
--------
    from .platform_profile import profile
    device = profile.clip_device()      # env 有值就用 env，否则探测

注意：`profile` 是模块级单例，进程内只探测一次。
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("sentrix.platform_profile")

# ffmpeg 里 Jetson 硬件解码器的命名后缀；映射表必须按实测结果裁剪，
# 不能凭"Orin 支持 H.264/HEVC/VP9..."的资料硬写 —— 见 `ffmpeg_hw_decoders`。
_HW_DECODER_SUFFIX = "_nvv4l2dec"


def _override(name: str) -> str | None:
    """环境变量作为覆盖项；未设置或为空串时返回 None，交给探测。"""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@functools.lru_cache(maxsize=1)
def is_jetson() -> bool:
    """Jetson 判定：设备树里有 jetson 字样。x86 上该路径不存在，返回 False。"""
    try:
        model = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace")
    except OSError:
        return False
    return "jetson" in model.lower()


@functools.lru_cache(maxsize=1)
def _memory_mib() -> tuple[int, int]:
    """返回 (total, available) MiB。读不到时返回 (0, 0)。"""
    total = available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return total, available


@functools.lru_cache(maxsize=1)
def cuda_usable() -> bool:
    """torch 说 cuda 可用**不算数**，必须真跑一次计算。

    为什么不能只信 `torch.cuda.is_available()`：46 上出现过 cuBLAS 被 conda 的
    pip 版覆盖、`cublasCreate_v2` 返回 `CUBLAS_STATUS_ALLOC_FAILED` 的情况 ——
    此时 `is_available()` 仍为 True，但**任何一次视觉推理都会让整个进程崩掉**。
    只有实跑一个小卷积才能真正确认 cuBLAS/cuDNN 能建出 handle。
    """
    override = _override("SENTRIX_CUDA_USABLE")
    if override is not None:
        return override.lower() in {"1", "true", "yes", "on"}
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # 覆盖 cuBLAS(matmul) 与 cuDNN(conv2d) 两条路径，这两条在 Jetson 上
        # 走的是不同的库加载链路，只测其中一条会漏判。
        a = torch.randn(8, 8, device="cuda")
        (a @ a).sum().item()
        x = torch.randn(1, 3, 32, 32, device="cuda")
        torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
        torch.cuda.synchronize()
        return True
    except Exception as error:  # noqa: BLE001 - 任何异常都意味着 CUDA 不可用
        log.warning(
            "CUDA 自检失败，将回落 CPU：%s。常见原因：LD_LIBRARY_PATH 里 cuBLAS/cuDNN "
            "被其它发行版覆盖（Jetson 上应让 JetPack 路径优先）。", error,
        )
        return False


@functools.lru_cache(maxsize=1)
def ffmpeg_hw_decoders() -> frozenset[str]:
    """实测这台机器的 ffmpeg 有哪些 nvv4l2 硬件解码器。

    为什么必须实测：映射表里写了 `av1_nvv4l2dec`，但 46 与 118 的 ffmpeg（4.4.2）
    实际只编译了 h264/hevc/mpeg2/mpeg4/vp8/vp9 六个 —— 不存在的解码器会让每个
    av1 视频先走一次必然失败的 NVDEC、再回退 CPU，白白浪费一轮。
    """
    if shutil.which("ffmpeg") is None:
        log.warning("找不到 ffmpeg，视频链路将完全依赖 CPU 软解")
        return frozenset()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("探测 ffmpeg 解码器失败：%s", error)
        return frozenset()
    decoders = set()
    for line in result.stdout.splitlines():
        for token in line.split():
            if token.endswith(_HW_DECODER_SUFFIX):
                decoders.add(token)
    return frozenset(decoders)


class PlatformProfile:
    """一台机器的能力快照。所有方法都返回**安全默认值**，env 有值时以 env 为准。"""

    # ---------- 计算设备 ----------

    def is_jetson(self) -> bool:
        """是否 Jetson（统一内存平台）。"""
        return is_jetson()

    def clip_device(self) -> str:
        """视觉编码设备。env `CLIP_DEVICE` 优先；否则按 CUDA 自检结果。"""
        return _override("CLIP_DEVICE") or ("cuda" if cuda_usable() else "cpu")

    def adaface_device(self) -> str:
        return _override("ADAFACE_DEVICE") or ("cuda" if cuda_usable() else "cpu")

    def video_device(self) -> str:
        """视频关键帧抽帧设备。

        历史坑：同一个 `SENTRIX_VIDEO_DEVICE` 在不同文件里的默认值不一致
        （`hybrid_keyframe`/`processor`/`worldmm_adapter` 默认 `cpu`，而同文件的
        `--device` 参数默认 `0`），等于一半进程按 CPU、一半按 GPU 0 起。这里统一。
        """
        override = _override("SENTRIX_VIDEO_DEVICE")
        if override:
            return override
        return "0" if cuda_usable() else "cpu"

    def face_providers(self) -> str:
        """ONNX Runtime provider 链。带 CPU 兜底，避免 CUDA 抖动直接让整批失败。"""
        override = _override("FACE_PROVIDERS")
        if override:
            return override
        return "CUDAExecutionProvider,CPUExecutionProvider" if cuda_usable() else "CPUExecutionProvider"

    def retinaface_providers(self) -> str:
        override = _override("RETINAFACE_PROVIDERS")
        if override:
            return override
        return self.face_providers()

    # ---------- 规模 ----------

    def pipeline_workers(self) -> int:
        """导入管线并发。

        由内存推算而不是写死：16GB 的 46 上 4 并发会把系统压进 swap（实测可用内存
        最低 342MB、swap 用到 98.5%），64GB 的 118 则可以到 8。
        env `SENTRIX_PIPELINE_MAX_WORKERS` 仍然优先。
        """
        override = _override("SENTRIX_PIPELINE_MAX_WORKERS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                log.warning("SENTRIX_PIPELINE_MAX_WORKERS=%r 不是整数，改为探测", override)
        total, _ = _memory_mib()
        if total <= 0:
            return 2
        return max(1, min(8, total // 6))  # 16GB→2、32GB→5、48GB+→8

    def service_parallel(self) -> int:
        """VLM 服务端槽位数。必须 ≥ 管线并发，否则请求挤在单槽排队，等于没提速。"""
        override = _override("SENTRIX_SERVICE_MAX_SEQS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                log.warning("SENTRIX_SERVICE_MAX_SEQS=%r 不是整数，改为探测", override)
        return self.pipeline_workers()

    def event_summary_workers(self) -> int:
        override = _override("SENTRIX_EVENT_SUMMARY_MAX_WORKERS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return self.pipeline_workers()

    # ---------- 存储与模型 ----------

    def vector_backend(self) -> str:
        """向量后端。

        默认 sqlite。**不**去探测仓库里有没有 `data/qdrant` 目录：那个目录在
        153 上一直存在，一旦把它当作信号，任何没设 `SENTRIX_VECTOR_BACKEND`
        的进程（包括单元测试）都会被导进 qdrant 分支并失败。
        需要 qdrant 就显式设置后端与路径；选错时启动日志里的
        `platform_profile vector_backend = ...` 会立刻暴露。
        """
        override = _override("SENTRIX_VECTOR_BACKEND")
        if override:
            return override.lower()
        return "sqlite"

    def face_embedding_mode(self) -> str:
        """人脸识别模型。

        默认 `legacy`（InsightFace buffalo_l / w600k_r50，174MB ONNX）而不是 AdaFace，
        原因：AdaFace 是 700MB fp32 PyTorch 权重，且在 153 上本身就跑 `ADAFACE_DEVICE=cpu`；
        对 16GB 的 46 来说这 700MB 是压垮内存的那一根稻草。

        注意 `backend/face_embeddings.py` 的硬约束：**不允许跨模型静默回退**，
        因为那会让存进库的向量与实际模型不符。所以这里只决定"用哪个"，
        绝不在运行时中途切换。
        """
        override = _override("FACE_EMBEDDING_MODE")
        if override:
            return override.lower()
        return "legacy"

    def face_embedding_is_adaface(self) -> bool:
        return self.face_embedding_mode() == "adaface"

    def video_keyframe_algorithm(self) -> str:
        """视频关键帧算法。

        默认 `hybrid_webp` 而不是 `worldmm`。这不是口味问题，是实测结论：

        - `worldmm` 产出的场景摘要是**占位符**（"视频场景 2" / "视频场景 · 5.9s~10.0s"），
          VLM 根本没有真正总结；
        - 针对 NVDEC 并发上限、编解码器映射、CPU 回退做的全部修复**只在 hybrid_webp 生效**。

        默认值选错会让这些修复静默失效 —— 46 上就因为 `.env` 少写这一个变量，
        整轮视频都跑在 worldmm 上，日志里看不出任何异常。
        """
        override = _override("SENTRIX_VIDEO_KEYFRAME_ALGORITHM")
        if override:
            return override.lower()
        return "hybrid_webp"

    # ---------- 汇总（用于启动时打日志） ----------

    def summary(self) -> dict:
        total, available = _memory_mib()
        return {
            "jetson": is_jetson(),
            "cuda_usable": cuda_usable(),
            "memory_total_mib": total,
            "memory_available_mib": available,
            "ffmpeg_hw_decoders": sorted(ffmpeg_hw_decoders()),
            "clip_device": self.clip_device(),
            "video_device": self.video_device(),
            "face_embedding_mode": self.face_embedding_mode(),
            "vector_backend": self.vector_backend(),
            "pipeline_workers": self.pipeline_workers(),
            "service_parallel": self.service_parallel(),
        }

    def log_summary(self) -> None:
        """启动时调用一次。

        为什么必须留痕：46 上曾经因为一个变量缺失而静默换掉了整条视频链路，
        排查花了很久。把实际生效的能力打出来，这类问题一眼可见。
        """
        if not log.handlers:
            # 后端没有配置 logging，`sentrix.*` 的 INFO 会被 Python 的 lastResort
            # 处理器（只收 WARNING 以上）丢掉 —— 实测启动日志里一行都看不到。
            # 这条摘要存在的意义就是"必须可见"，所以给本模块挂一个专属 handler；
            # propagate=False 避免将来 root logger 配好后重复输出。
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False
        for key, value in self.summary().items():
            log.info("platform_profile %s = %s", key, value)


profile = PlatformProfile()
