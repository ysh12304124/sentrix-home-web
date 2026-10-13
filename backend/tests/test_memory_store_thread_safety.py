import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.db import MemoryStore


class MemoryStoreThreadSafetyTests(unittest.TestCase):
    def test_concurrent_entity_reads_use_isolated_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.db"))
            try:
                with ThreadPoolExecutor(max_workers=12) as executor:
                    results = list(executor.map(lambda _: store.list_entities(), range(60)))
                self.assertEqual([[] for _ in results], results)
            finally:
                store.close()

    def test_memory_database_remains_shared_across_worker_threads(self):
        store = MemoryStore(":memory:")
        try:
            store.create_memory_space("home-default", "默认家庭")
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: store.list_memory_spaces(), range(24)))
            self.assertEqual(24, len(results))
            for spaces in results:
                self.assertEqual("home-default", spaces[0]["id"])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
