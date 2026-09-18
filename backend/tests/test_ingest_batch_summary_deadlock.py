import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore


class IngestBatchSummaryDeadlockTests(unittest.TestCase):
    def test_stuck_summarizing_batch_can_be_recovered(self):
        """收尾途中进程退出后，批次不能永远卡在 summarizing。

        complete_ingest_batch 原本保留 'summarizing'，而 claim_ingest_batch_summary
        要求 status=='complete'（且只做 complete→summarizing 单向迁移）、
        finish_ingest_batch 要求 status=='summarizing' —— 三者构成单向死结：
        批次既认领不了也复位不了，其后的 scope_finalize 与 ANN 索引重建钩子永不执行，
        表现为相册建好了但检索全空，且没有任何告警。
        """
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "test.db")
            try:
                store.create_ingest_batch("batch-stuck", "scope-test")
                store.complete_ingest_batch("batch-stuck")
                self.assertTrue(store.claim_ingest_batch_summary("batch-stuck"))
                self.assertEqual(store.get_ingest_batch("batch-stuck")["status"], "summarizing")

                # 模拟进程在 summarize 途中退出后的恢复路径：复位 -> 重新认领
                self.assertEqual(store.complete_ingest_batch("batch-stuck")["status"], "complete")
                self.assertTrue(store.claim_ingest_batch_summary("batch-stuck"))

                # 正常收尾路径不受影响
                self.assertEqual(store.finish_ingest_batch("batch-stuck")["status"], "completed")
                # 已完成的批次不会被复位回 complete
                self.assertEqual(store.complete_ingest_batch("batch-stuck")["status"], "completed")
            finally:
                store.close()

    def test_cancelled_batch_is_still_never_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "test.db")
            try:
                store.create_ingest_batch("batch-cancelled", "scope-test")
                store.cancel_ingest_batch("batch-cancelled", "unit-test")
                self.assertEqual(
                    store.complete_ingest_batch("batch-cancelled")["status"], "cancelled")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
