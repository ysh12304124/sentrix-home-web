import importlib.util
from io import BytesIO
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("benchmark_orchestrator.py")
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_photo_import", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PhotoImportTests(unittest.TestCase):
    def test_request_json_includes_remote_error_detail(self):
        error = urllib.error.HTTPError(
            "http://sentrix.test/api/memory-spaces",
            500,
            "Internal Server Error",
            {},
            BytesIO(b'{"detail":"database is locked"}'),
        )
        with patch.object(MODULE.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500.*database is locked"):
                MODULE.request_json("http://sentrix.test/api/memory-spaces")

    def test_photo_upload_uses_bounded_concurrency_and_keeps_result_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(3):
                path = root / f"photo-{index}.jpg"
                path.write_bytes(f"photo-{index}".encode())
                paths.append(path)
            video = root / "clip.mp4"
            video.write_bytes(b"video-bytes")
            paths.append(video)

            run = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
            run.album_dir = root
            run.manifest = {
                "photos": [path.name for path in paths[:3]],
                "videos": [video.name],
            }
            run.sentrix_url = "http://sentrix.test"
            run.state = {"scope_id": "scope-1", "phases": {}, "items": []}
            run.run_id = "test-run"
            run.results_root = root
            run.lock = threading.RLock()
            run._cancel = threading.Event()
            run._phase_started_perf = {}
            run.persist = lambda wait=False: None
            run._record_phase = lambda *args, **kwargs: None
            run._cancel_remote_batch = lambda *args, **kwargs: None
            phase_done = {}
            run._phase_done = lambda phase, extra=None: phase_done.update(extra or {})

            barrier = threading.Barrier(2)
            active = 0
            max_active = 0
            active_lock = threading.Lock()
            call_order = []

            def fake_upload(url, fields, files, timeout):
                nonlocal active, max_active
                name = files[0][1]
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                    call_order.append(name)
                barrier.wait(timeout=2)
                time.sleep(0.005)
                with active_lock:
                    active -= 1
                return {"items": [{"accepted": True, "fileName": name}]}

            with patch.dict(os.environ, {
                "PHOTOBENCH_IMPORT_CHUNK_SIZE": "1",
                "PHOTOBENCH_IMPORT_UPLOAD_WORKERS": "2",
            }), patch.object(MODULE, "upload_files", side_effect=fake_upload), \
                    patch.object(MODULE, "request_json", return_value={}):
                run._phase_photo_import()

            self.assertEqual(max_active, 2)
            self.assertEqual(phase_done["accepted_count"], 4)
            self.assertEqual(phase_done["total_photos"], 3)
            self.assertEqual(phase_done["total_videos"], 1)
            self.assertEqual(phase_done["total_media"], 4)
            self.assertEqual(phase_done["upload_workers"], 2)
            self.assertEqual(phase_done["chunk_count"], 4)
            self.assertEqual(set(call_order), {path.name for path in paths})

    def test_album_media_entries_keeps_photos_then_videos(self):
        self.assertEqual(
            MODULE.album_media_entries({
                "photos": ["photos/a.jpg", "photos/b.jpg"],
                "videos": ["videos/c.mp4", "photos/a.jpg"],
            }),
            ["photos/a.jpg", "photos/b.jpg", "videos/c.mp4"],
        )

    def test_photo_upload_records_failed_chunk_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            run = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
            run.album_dir = root
            run.manifest = {"photos": [first.name, second.name], "videos": []}
            run.sentrix_url = "http://sentrix.test"
            run.state = {"scope_id": "scope-1", "phases": {}, "items": []}
            run.run_id = "test-run"
            run.results_root = root
            run.lock = threading.RLock()
            run._cancel = threading.Event()
            run._phase_started_perf = {}
            run.persist = lambda wait=False: None
            run._record_phase = lambda *args, **kwargs: None
            partial = {}
            run._phase_partial = lambda phase, extra=None: partial.update(extra or {})

            def fake_upload(url, fields, files, timeout):
                name = files[0][1]
                if name == first.name:
                    raise TimeoutError("upload timeout")
                return {"items": [{"accepted": True, "fileName": name}]}

            with patch.dict(os.environ, {
                "PHOTOBENCH_IMPORT_CHUNK_SIZE": "1",
                "PHOTOBENCH_IMPORT_UPLOAD_WORKERS": "1",
                "PHOTOBENCH_IMPORT_MAX_ATTEMPTS": "1",
            }), patch.object(MODULE, "upload_files", side_effect=fake_upload), \
                    patch.object(MODULE, "request_json", return_value={}) as request:
                run._phase_photo_import()

            self.assertEqual(partial["accepted_count"], 1)
            self.assertEqual(partial["failed_count"], 1)
            self.assertEqual(partial["error_details"][0]["sample_id"], first.name)
            self.assertIn("upload timeout", partial["error_details"][0]["reason"])
            self.assertEqual(request.call_count, 1)

    def test_photo_upload_retries_transient_item_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "retry.jpg"
            photo.write_bytes(b"retry")

            run = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
            run.album_dir = root
            run.manifest = {"photos": [photo.name], "videos": []}
            run.sentrix_url = "http://sentrix.test"
            run.state = {"scope_id": "scope-1", "phases": {}, "items": []}
            run.run_id = "test-run"
            run.results_root = root
            run.lock = threading.RLock()
            run._cancel = threading.Event()
            run._phase_started_perf = {}
            run.persist = lambda wait=False: None
            run._record_phase = lambda *args, **kwargs: None
            phase_done = {}
            run._phase_done = lambda phase, extra=None: phase_done.update(extra or {})

            responses = [
                {"items": [{
                    "accepted": False,
                    "fileName": photo.name,
                    "status": "failed",
                    "error": "database is locked",
                }]},
                {"items": [{"accepted": True, "fileName": photo.name, "status": "queued"}]},
            ]
            with patch.dict(os.environ, {
                "PHOTOBENCH_IMPORT_CHUNK_SIZE": "1",
                "PHOTOBENCH_IMPORT_UPLOAD_WORKERS": "1",
                "PHOTOBENCH_IMPORT_MAX_ATTEMPTS": "3",
            }), patch.object(MODULE, "upload_files", side_effect=responses) as upload, \
                    patch.object(MODULE.time, "sleep"), \
                    patch.object(MODULE, "request_json", return_value={}):
                run._phase_photo_import()

            self.assertEqual(upload.call_count, 2)
            self.assertEqual(phase_done["accepted_count"], 1)
            self.assertEqual(phase_done["failed_count"], 0)
            self.assertEqual(phase_done["retried_file_count"], 1)
            self.assertEqual(phase_done["max_upload_attempts"], 3)

    def test_processing_batch_failure_becomes_partial_without_polling_forever(self):
        run = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        run.sentrix_url = "http://sentrix.test"
        run.use_cloud_model = True
        run.state = {
            "scope_id": "scope-1",
            "batch_id": "batch-1",
            "import_accepted_count": 2,
            "phases": {},
        }
        run.lock = threading.RLock()
        run._cancel = threading.Event()
        run._phase_started_perf = {}
        run.persist = lambda wait=False: None
        partial = {}
        run._phase_partial = lambda phase, extra=None: partial.update(extra or {})

        assets = {"assets": [
            {"id": "asset-ok", "file_name": "ok.jpg", "status": "processed", "metadata_json": {}},
            {"id": "asset-stuck", "file_name": "stuck.jpg", "status": "queued", "metadata_json": {}},
        ]}
        batch = {
            "batch": {"status": "complete"},
            "pipeline_metrics": {"status": "failed", "error": "worker crashed"},
        }

        def fake_request(url, *args, **kwargs):
            return batch if "ingest-batches" in url else assets

        with patch.object(MODULE, "request_json", side_effect=fake_request):
            run._phase_processing()

        self.assertEqual(partial["processed_photo_count"], 1)
        self.assertEqual(partial["skipped_asset_count"], 1)
        self.assertEqual(partial["error"], "worker crashed")
        self.assertEqual(partial["progress"]["pending"], 0)


if __name__ == "__main__":
    unittest.main()
