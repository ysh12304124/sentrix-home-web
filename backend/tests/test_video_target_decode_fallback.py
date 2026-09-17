import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

KATNA = Path(__file__).resolve().parents[2] / "tools" / "video_keyframe" / "katna"

# 不能写 `from tools.video_keyframe...` —— 仓库根的 tools.py 会遮蔽 tools/ 目录。
# 按文件路径加载，并注册进 sys.modules，让被测脚本里的 `import decode_strategy`
# 复用同一个模块实例。
_STRATEGY_SPEC = importlib.util.spec_from_file_location("decode_strategy", KATNA / "decode_strategy.py")
decode_strategy = importlib.util.module_from_spec(_STRATEGY_SPEC)
sys.modules["decode_strategy"] = decode_strategy
_STRATEGY_SPEC.loader.exec_module(decode_strategy)
DecodeStrategy = decode_strategy.DecodeStrategy

SCRIPT = KATNA / "run_yolo_prefilter_event_webp.py"
SPEC = importlib.util.spec_from_file_location("video_target_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

# 模拟一个"硬件解码可用"的策略；具体是 nvv4l2dec 还是 cuda hwaccel 由机器决定，
# 这里只关心"先试硬件、失败退回 CPU"这条路径。
_HW = DecodeStrategy(name="hw-test", input_args=("-hwaccel", "cuda"), vf_prefix="hwdownload,format=nv12,", concurrency=4)
_CPU = DecodeStrategy(name="cpu", input_args=(), vf_prefix="", concurrency=4)


def _result(returncode, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def test_target_decode_falls_back_to_cpu_after_hardware_failure():
    pixels = np.arange(12, dtype=np.uint8).tobytes()
    with patch.object(MODULE, "detect_codec", return_value="h264"), \
         patch.object(MODULE, "select_strategy", return_value=_HW), \
         patch.object(
             MODULE.subprocess,
             "run",
             side_effect=[_result(1, stderr=b"CUDA_ERROR_OUT_OF_MEMORY"), _result(0, stdout=pixels)],
         ) as run:
        frame_index, frame, backend = MODULE._decode_one_target("video.mp4", 24, 24, 2, 2)

    assert frame_index == 24
    assert frame.shape == (2, 2, 3)
    assert backend == "cpu"
    assert "-hwaccel" in run.call_args_list[0].args[0]
    assert "-hwaccel" not in run.call_args_list[1].args[0]


def test_target_decode_reports_both_failures():
    with patch.object(MODULE, "detect_codec", return_value="h264"), \
         patch.object(MODULE, "select_strategy", return_value=_HW), \
         patch.object(
             MODULE.subprocess,
             "run",
             side_effect=[_result(1, stderr=b"nvdec oom"), _result(1, stderr=b"invalid input")],
         ):
        with pytest.raises(RuntimeError, match="NVDEC .* CPU fallback"):
            MODULE._decode_one_target("video.mp4", 0, 24, 2, 2)


def test_target_decode_uses_cpu_directly_when_no_hardware_available():
    pixels = np.arange(12, dtype=np.uint8).tobytes()
    with patch.object(MODULE, "detect_codec", return_value="av1"), \
         patch.object(MODULE, "select_strategy", return_value=_CPU), \
         patch.object(MODULE.subprocess, "run", return_value=_result(0, stdout=pixels)) as run:
        _, _, backend = MODULE._decode_one_target("video.mp4", 0, 24, 2, 2)

    assert backend == "cpu"
    assert "-hwaccel" not in run.call_args_list[0].args[0]


@pytest.mark.parametrize(
    ("frame_index", "frame_count", "expected"),
    [(-1, 241, 0), (0, 241, 0), (240, 241, 240), (252, 241, 240)],
)
def test_target_index_is_clamped_to_real_frame_range(frame_index, frame_count, expected):
    assert MODULE._clamp_target_index(frame_index, frame_count) == expected
