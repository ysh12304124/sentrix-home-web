#!/usr/bin/env python3
"""Small Katna primitives used by the targeted keyframe pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    frame_index: int
    timestamp: float
    image: np.ndarray
    relevance: float = 0.0
    query_index: int = -1
    redundancy: float = 0.0


def hanning_smooth(values: np.ndarray, window_len: int = 20) -> np.ndarray:
    if values.size < window_len:
        return values
    reflected = np.r_[2 * values[0] - values[window_len:1:-1], values,
                       2 * values[-1] - values[-1:-window_len:-1]]
    window = np.hanning(window_len)
    smoothed = np.convolve(window / window.sum(), reflected, mode="same")
    return smoothed[window_len - 1 : -window_len + 1]


def brightness(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean() * 100.0 / 255.0)


def entropy_score(image: np.ndarray) -> float:
    from skimage.filters.rank import entropy
    from skimage.morphology import disk

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(entropy(gray, disk(5)).mean())
