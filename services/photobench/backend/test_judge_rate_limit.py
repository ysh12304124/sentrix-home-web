import importlib.util
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("benchmark_orchestrator.py")
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_judge_rate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RecordingCancel:
    def __init__(self):
        self.waits = []

    def wait(self, seconds):
        self.waits.append(seconds)
        return False


class JudgeRateLimitTests(unittest.TestCase):
    def make_run(self):
        run = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        run.judge_url = "http://judge.test/v1"
        run.judge_api_key = ""
        run._cancel = RecordingCancel()
        run._judge_rate_lock = threading.Lock()
        run._judge_next_request_at = 0.0
        return run

    def test_default_judge_concurrency_is_bounded(self):
        with patch.dict(MODULE.os.environ, {}, clear=True):
            self.assertEqual(MODULE.BenchmarkRun._resolve_judge_concurrency(12), 8)
            self.assertEqual(MODULE.BenchmarkRun._resolve_judge_concurrency(1), 1)

    def test_retry_after_and_jitter_are_honored(self):
        http_error = urllib.error.HTTPError(
            "http://judge.test", 429, "Too Many Requests", {"Retry-After": "17"}, None,
        )
        wrapped = RuntimeError("HTTP 429")
        wrapped.__cause__ = http_error
        with patch.object(MODULE.random, "uniform", return_value=1.25):
            self.assertEqual(MODULE.BenchmarkRun._judge_retry_delay(wrapped, 1), 18.25)

    def test_judge_request_retries_then_succeeds(self):
        run = self.make_run()
        with patch.object(MODULE, "JUDGE_RETRY_ATTEMPTS", 3), \
                patch.object(MODULE, "JUDGE_RETRY_BACKOFF_SECONDS", 2.0), \
                patch.object(MODULE, "JUDGE_RETRY_BACKOFF_MAX_SECONDS", 10.0), \
                patch.object(MODULE, "JUDGE_REQUEST_INTERVAL_SECONDS", 0.0), \
                patch.object(MODULE.random, "uniform", return_value=0.0), \
                patch.object(MODULE, "request_json", side_effect=[
                    RuntimeError("HTTP 429 Too Many Requests"),
                    RuntimeError("HTTP 429 Too Many Requests"),
                    {"choices": []},
                ]) as request:
            response, attempts = run._judge_request({"model": "judge"})
        self.assertEqual(response, {"choices": []})
        self.assertEqual(attempts, 3)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(run._cancel.waits, [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
