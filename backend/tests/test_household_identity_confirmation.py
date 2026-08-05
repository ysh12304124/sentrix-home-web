import json
import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from scripts.benchmarks.confirm_household_benchmark_identities import run


class HouseholdIdentityConfirmationTests(unittest.TestCase):
    def test_confirms_single_person_seed_then_resolves_multi_person_by_elimination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "memory.db"
            store = MemoryStore(str(database))
            store.create_memory_space("album1", "album1")
            embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
            for file_name, identities in {"solo-a.jpg": ["a"], "pair.jpg": ["a", "b"]}.items():
                asset = store.create_asset(file_name, file_name, "image", str(root / file_name), scope_id="album1")
                observation = store.add_observation(asset["id"], {"scope_id": "album1"}, scope_id="album1")
                for identity in identities:
                    store.add_face_instance(asset["id"], observation["id"], {"embedding": embeddings[identity], "quality": 0.9, "confidence": 0.95})
            store.close()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"spaces": [{
                "scope_id": "album1",
                "evaluation": {
                    "face_id_to_nicknames": {"1": ["我"], "2": ["朋友"]},
                    "image_to_face_ids": {"solo-a.jpg": ["1"], "pair.jpg": ["1", "2"]},
                },
            }]}), encoding="utf-8")

            preview = run(manifest, database)
            applied = run(manifest, database, apply=True)

            self.assertEqual(preview["summary"]["eligible_confirmations"], 2)
            self.assertEqual(applied["summary"]["applied_confirmations"], 2)
            verified = MemoryStore(str(database))
            try:
                self.assertEqual({item["canonical_name"] for item in verified.list_entities(status="confirmed", scope_id="album1")}, {"我", "朋友"})
            finally:
                verified.close()


if __name__ == "__main__":
    unittest.main()
