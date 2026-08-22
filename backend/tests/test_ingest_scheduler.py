import contextlib
import threading
import unittest
from unittest.mock import patch

from backend import app as ingest_app


class _FakeStore:
    def __init__(self):
        self.failures = {}
        self.cleaned = []

    def cleanup_asset_derivatives(self, asset_id):
        self.cleaned.append(asset_id)

    def update_asset(self, asset_id, status, metadata):
        self.failures[asset_id] = {"status": status, **metadata}


class _FakePipeline:
    def __init__(self, events, semantic_started):
        self.events = events
        self.semantic_started = semantic_started

    def commit_fast_image(self, asset_id, prepared):
        if asset_id == "asset-2":
            if not self.semantic_started.wait(timeout=2):
                raise AssertionError("semantic stage did not start after the first fast commit")
        self.events.append(("fast-commit", asset_id))

    def commit_semantic_image(self, asset_id, prepared, summarize_event=False):
        self.events.append(("semantic-commit", asset_id, summarize_event))


class IngestSchedulerTests(unittest.TestCase):
    def test_semantic_inference_starts_before_all_fast_commits(self):
        events = []
        semantic_started = threading.Event()
        store = _FakeStore()
        pipeline = _FakePipeline(events, semantic_started)

        def prepare(asset_id, stage):
            events.append((f"{stage}-start", asset_id))
            if stage == "semantic":
                semantic_started.set()
            return {"asset_id": asset_id}

        with patch.object(ingest_app, "_prepare_asset_stage", side_effect=prepare), \
                patch.object(ingest_app, "db_write_guard", lambda label: contextlib.nullcontext()):
            ingest_app._process_image_stages(
                ["asset-1", "asset-2", "asset-3", "asset-4"],
                store,
                pipeline,
                {"effective_workers": 2},
            )

        semantic_start_index = next(index for index, item in enumerate(events) if item[0] == "semantic-start")
        fast_commit_indexes = [index for index, item in enumerate(events) if item[0] == "fast-commit"]
        self.assertLess(semantic_start_index, max(fast_commit_indexes))
        self.assertEqual(
            [item[1] for item in events if item[0] == "fast-commit"],
            ["asset-1", "asset-2", "asset-3", "asset-4"],
        )
        self.assertEqual(
            [item[1] for item in events if item[0] == "semantic-commit"],
            ["asset-1", "asset-2", "asset-3", "asset-4"],
        )
        self.assertTrue(all(item[2] is False for item in events if item[0] == "semantic-commit"))

    def test_stage_failures_are_isolated_and_marked(self):
        store = _FakeStore()
        pipeline = _FakePipeline([], threading.Event())

        def prepare(asset_id, stage):
            if asset_id == "asset-2" and stage == "fast":
                raise RuntimeError("face failed")
            if asset_id == "asset-3" and stage == "semantic":
                raise RuntimeError("vlm failed")
            return {"asset_id": asset_id}

        with patch.object(ingest_app, "_prepare_asset_stage", side_effect=prepare), \
                patch.object(ingest_app, "db_write_guard", lambda label: contextlib.nullcontext()):
            ingest_app._process_image_stages(
                ["asset-1", "asset-2", "asset-3"],
                store,
                pipeline,
                {"effective_workers": 2},
            )

        self.assertEqual(store.cleaned, ["asset-2"])
        self.assertEqual(store.failures["asset-2"]["failed_stage"], "fast")
        self.assertEqual(store.failures["asset-3"]["failed_stage"], "semantic")


if __name__ == "__main__":
    unittest.main()
