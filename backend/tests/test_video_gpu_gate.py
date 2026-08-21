import os
from unittest.mock import patch

import pytest

from backend.video.hybrid_keyframe import check_video_gpu_capacity


class TestVideoGpuCapacityGate:
    def test_cpu_device_skips_check(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "cpu"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib") as probe:
                check_video_gpu_capacity()
        probe.assert_not_called()

    def test_auto_device_skips_check(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "auto"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib") as probe:
                check_video_gpu_capacity()
        probe.assert_not_called()

    def test_sufficient_gpu_memory_passes(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "0", "SENTRIX_VIDEO_GPU_MIN_FREE_MIB": "4096"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib", return_value=11264):
                check_video_gpu_capacity()  # should not raise

    def test_insufficient_gpu_memory_raises(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "0", "SENTRIX_VIDEO_GPU_MIN_FREE_MIB": "4096"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib", return_value=1024):
                with pytest.raises(RuntimeError, match="insufficient GPU memory"):
                    check_video_gpu_capacity()

    def test_unqueryable_gpu_skips_check(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "0"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib", return_value=None):
                check_video_gpu_capacity()  # cannot query; do not block

    def test_run_checks_capacity_before_command(self):
        with patch.dict(os.environ, {"SENTRIX_VIDEO_DEVICE": "0", "SENTRIX_VIDEO_GPU_MIN_FREE_MIB": "4096"}):
            with patch("backend.video.hybrid_keyframe._gpu_free_memory_mib", return_value=1024):
                with pytest.raises(RuntimeError, match="insufficient GPU memory"):
                    from backend.video.hybrid_keyframe import run as run_hybrid_keyframes
                    run_hybrid_keyframes("/tmp/nonexistent.mov", "/tmp/out", "video-1")
