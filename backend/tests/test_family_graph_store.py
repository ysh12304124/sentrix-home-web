import tempfile
import unittest

from backend.db import MemoryStore


class FamilyGraphStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.father = self.store.create_entity("人物 A", "person", "pending", scope_id="album-a")
        self.daughter = self.store.create_entity("人物 B", "person", "pending", scope_id="album-a")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_model_relationship_writes_bidirectional_effective_edges(self):
        rows = self.store.set_family_relationship(
            "album-a", self.father["id"], "父亲", self.daughter["id"], "女儿",
            source="model", confidence=0.92, evidence_refs=["obs_1"],
        )

        self.assertEqual({row["predicate"] for row in rows}, {"父亲", "女儿"})
        effective = self.store.list_effective_family_relationships("album-a")
        self.assertEqual({row["predicate"] for row in effective}, {"父亲", "女儿"})
        self.assertTrue(all(row["source"] == "model" for row in effective))

    def test_user_relationship_override_survives_later_model_result(self):
        self.store.set_family_relationship(
            "album-a", self.father["id"], "父亲", self.daughter["id"], "女儿",
            source="model", confidence=0.92, evidence_refs=["obs_1"],
        )
        self.store.set_family_relationship(
            "album-a", self.father["id"], "叔叔", self.daughter["id"], "侄女",
            source="user_override", confidence=1.0, evidence_refs=["obs_2"],
        )
        self.store.set_family_relationship(
            "album-a", self.father["id"], "父亲", self.daughter["id"], "女儿",
            source="model", confidence=0.99, evidence_refs=["obs_3"],
        )

        effective = self.store.list_effective_family_relationships("album-a")
        self.assertEqual({row["predicate"] for row in effective}, {"叔叔", "侄女"})
        self.assertTrue(all(row["source"] == "user_override" for row in effective))
        self.assertTrue(all(row["locked"] for row in effective))

    def test_membership_override_blocks_later_model_result(self):
        self.store.set_family_membership(
            "album-a", self.father["id"], "family", source="model", confidence=0.84,
        )
        self.store.set_family_membership(
            "album-a", self.father["id"], "friend", source="user_override", confidence=1.0,
        )
        retained = self.store.set_family_membership(
            "album-a", self.father["id"], "friend", source="model", confidence=0.95,
        )

        self.assertEqual(retained["membership"], "friend")
        self.assertEqual(retained["source"], "user_override")
        self.assertTrue(retained["locked"])
