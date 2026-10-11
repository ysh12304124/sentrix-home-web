import unittest
from unittest.mock import patch

import numpy as np

from backend.face_detector import RetinaFaceTiledDetector


class RetinaFaceTiledDetectorTests(unittest.TestCase):
    def test_fine_pass_is_fallback_when_coarse_pass_finds_nothing(self):
        detector = RetinaFaceTiledDetector()
        small_face = {
            "bbox": [140.0, 120.0, 174.0, 158.0], "confidence": 0.4,
            "landmarks": [],
        }
        with patch.object(detector, "_load"), patch.object(
            detector, "_detect_tiled", side_effect=[[], [small_face]]
        ) as tiled:
            result = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

        self.assertEqual(tiled.call_count, 2)
        self.assertEqual(tiled.call_args_list[0].kwargs["tile"], 1024)
        self.assertEqual(tiled.call_args_list[1].kwargs["tile"], 640)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], small_face["bbox"])

    def test_fine_pass_is_skipped_when_coarse_detects_faces(self):
        detector = RetinaFaceTiledDetector()
        coarse_face = {
            "bbox": [10.0, 10.0, 90.0, 90.0], "confidence": 0.9,
            "landmarks": [],
        }
        duplicate = {
            "bbox": [11.0, 11.0, 89.0, 89.0], "confidence": 0.8,
            "landmarks": [],
        }
        small_face = {
            "bbox": [140.0, 120.0, 174.0, 158.0], "confidence": 0.4,
            "landmarks": [],
        }
        with patch.object(detector, "_load"), patch.object(
            detector, "_detect_tiled", side_effect=[[coarse_face], [duplicate, small_face]]
        ) as tiled:
            result = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

        self.assertEqual(tiled.call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], coarse_face["bbox"])


if __name__ == "__main__":
    unittest.main()
