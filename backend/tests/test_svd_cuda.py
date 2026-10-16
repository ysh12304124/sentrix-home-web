import os
import numpy as np
import pytest
from backend.video.svd_keyframe import SVDLowRankKeyframeExtractor


def require_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


def test_cuda_split_and_permutation_statistics_match_cpu(monkeypatch):
    require_cuda()
    x = np.random.default_rng(42).normal(size=(500, 8))
    x[250:] += 3
    e = SVDLowRankKeyframeExtractor()
    monkeypatch.setenv('SENTRIX_SVD_DEVICE', 'cpu')
    cpu = e._best_split_gain(x)
    null_cpu = e._null_split_gains(x, np.random.default_rng(2), 16)
    monkeypatch.setenv('SENTRIX_SVD_DEVICE', 'cuda')
    gpu = e._best_split_gain(x)
    null_gpu = e._null_split_gains(x, np.random.default_rng(2), 16)
    assert cpu[1] == gpu[1] == 250
    np.testing.assert_allclose(cpu, gpu, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(null_cpu, null_gpu, rtol=1e-8, atol=1e-8)


def test_cuda_svd_preserves_reconstruction_and_rank(monkeypatch):
    require_cuda()
    x = np.random.default_rng(7).normal(size=(2307, 80)).astype(np.float32)
    e = SVDLowRankKeyframeExtractor()
    monkeypatch.setenv('SENTRIX_SVD_DEVICE', 'cpu')
    cpu = e.low_rank_decomposition(x)
    monkeypatch.setenv('SENTRIX_SVD_DEVICE', 'cuda')
    u, s, vt = e._svd(x)
    assert np.linalg.norm(x - (u * s) @ vt) / np.linalg.norm(x) < 1e-5
    assert e.estimate_rank(s)[0] == cpu[3]
    np.testing.assert_allclose(s, cpu[1], rtol=1e-5, atol=1e-5)
