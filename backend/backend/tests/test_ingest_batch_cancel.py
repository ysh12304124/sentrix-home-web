import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore


class IngestBatchCancelTests(unittest.TestCase):
    def test_cancelled_batch_is_not_reopened_by_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "test.db")
            try:
                store.create_ingest_batch("batch-test", "scope-test")
                cancelled = store.cancel_ingest_batch("batch-test", "unit-test")
                completed = store.complete_ingest_batch("batch-test")
            finally:
                store.close()

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["metadata_json"]["cancel_source"], "unit-test")
        self.assertTrue(cancelled["metadata_json"]["cancel_requested_at"])
        self.assertEqual(completed["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
