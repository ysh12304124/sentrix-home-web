import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "video_keyframe"
    / "katna"
    / "run_yolo_prefilter_event_webp.py"
)
SPEC = importlib.util.spec_from_file_location("video_target_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(returncode, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def test_target_decode_falls_back_to_cpu_after_nvdec_failure():
    pixels = np.arange(12, dtype=np.uint8).tobytes()
    with patch.object(
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
    with patch.object(
        MODULE.subprocess,
        "run",
        side_effect=[_result(1, stderr=b"nvdec oom"), _result(1, stderr=b"invalid input")],
    ):
        with pytest.raises(RuntimeError, match="NVDEC .* CPU fallback"):
            MODULE._decode_one_target("video.mp4", 0, 24, 2, 2)


@pytest.mark.parametrize(
    ("frame_index", "frame_count", "expected"),
    [(-1, 241, 0), (0, 241, 0), (240, 241, 240), (252, 241, 240)],
)
def test_target_index_is_clamped_to_real_frame_range(frame_index, frame_count, expected):
    assert MODULE._clamp_target_index(frame_index, frame_count) == expected
