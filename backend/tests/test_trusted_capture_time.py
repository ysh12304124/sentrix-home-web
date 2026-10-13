import unittest
from types import SimpleNamespace

from backend.evidence_retrieval import EvidenceRetrievalKernel
from backend.query_contracts import Constraint, HARD
from backend.retrieval.temporal import trusted_captured_at


class _ParentStore:
    def __init__(self, parent):
        self.parent = parent

    def get_asset(self, _asset_id):
        return self.parent


class TrustedCaptureTimeTests(unittest.TestCase):
    def test_filesystem_mtime_is_not_a_capture_time(self):
        asset = {
            "id": "frame-1", "captured_at": "2026-09-18T06:26:30+00:00",
            "metadata_json": {"capture_time_source": "file_mtime_fallback"},
        }
        self.assertIsNone(trusted_captured_at(asset))

    def test_legacy_keyframe_reads_capture_time_source_from_parent_video(self):
        frame = {
            "id": "frame-1", "parent_asset_id": "video-1",
            "derived_kind": "video_keyframe", "captured_at": "2026-09-18T06:26:30+00:00",
        }
        parent = {"metadata_json": {"video_metadata": {"creation_source": "file_mtime_fallback"}}}
        self.assertIsNone(trusted_captured_at(frame, store=_ParentStore(parent)))

    def test_untrusted_time_is_unknown_not_a_hard_contradiction(self):
        asset = {
            "id": "frame-1", "captured_at": "2026-09-18T06:26:30+00:00",
            "metadata_json": {"capture_time_source": "file_mtime_fallback"},
        }
        constraint = Constraint("time", "2017年", HARD, "asset_metadata")
        kernel = EvidenceRetrievalKernel(store=None)
        spec = SimpleNamespace(constraints=[constraint])
        result = kernel._evaluate(asset, {}, spec)
        self.assertFalse(result["excluded"])
        self.assertEqual(result["item"]["condition_results"][constraint.key]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
