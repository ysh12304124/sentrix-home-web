import tempfile
import unittest

from backend.db import MemoryStore
from scripts.benchmarks.evaluate_person_insights import (
    freeze_answers,
    hide_answers,
    input_leaks_answers,
    restore_answers,
)


class PersonInsightEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.store.create_memory_space("album-b", "相册 B")
        self.answers = []
        for index in range(7):
            person = self.store.create_entity(
                f"人物{index + 1}", "person", "confirmed",
                family_role=("母亲" if index == 0 else None), scope_id="album-a",
            )
            cluster = self.store.create_face_cluster(
                [float(index + 1), 0, 0], confidence=0.9, scope_id="album-a"
            )
            self.store.connection.execute(
                "UPDATE face_clusters SET entity_id = ? WHERE id = ?",
                (person["id"], cluster["id"]),
            )
            self.answers.append({
                "person_id": person["id"], "canonical_name": f"人物{index + 1}",
                "family_role": "母亲" if index == 0 else None,
                "entity_status": "confirmed", "cluster_status": cluster["status"],
            })
        self.store.connection.commit()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_freeze_answers_reads_confirmed_people(self):
        frozen = freeze_answers(self.store, "album-a")
        self.assertEqual(len(frozen), 7)
        self.assertEqual({item["canonical_name"] for item in frozen},
                         {f"人物{i + 1}" for i in range(7)})
        self.assertNotIn("album-b", [item["person_id"] for item in frozen])

    def test_hide_and_restore_roundtrip(self):
        hide_answers(self.store, self.answers)
        person = self.store.get_entity(self.answers[0]["person_id"])
        self.assertEqual(person["status"], "pending")
        self.assertIsNone(person["family_role"])
        restore_answers(self.store, self.answers)
        person = self.store.get_entity(self.answers[0]["person_id"])
        self.assertEqual(person["canonical_name"], "人物1")
        self.assertEqual(person["family_role"], "母亲")
        self.assertEqual(person["status"], "confirmed")

    def test_input_payload_does_not_leak_frozen_answers(self):
        hide_answers(self.store, self.answers)
        graph_payload = {
            "people": ["P01", "P02"],
            "events": [{"id": "e1", "place": "家中餐厅", "activity": "聚会", "date": "2026-01-01"}],
            "cooccurrence": {"P01-P02": 3},
            "moments": [{"person": "P01", "action_text": "抱着孩子", "event_id": "e1"}],
            "devices": {},
        }
        self.assertFalse(input_leaks_answers(graph_payload, self.answers))
        portrait_pack = {
            "person": {"id": "p1", "display_name": "核心人物 1", "confirmed_name": None,
                       "confirmed_role": None},
            "moments": [{"kind": "person_moment", "id": "m1", "action_text": "张罗饭桌"}],
            "confirmed_relationships": [], "suggested_relationships": [],
        }
        self.assertFalse(input_leaks_answers(portrait_pack, self.answers))

    def test_input_leaks_are_detected(self):
        leaky = {"people": ["P01"], "notes": "人物3 是家里的母亲"}
        self.assertTrue(input_leaks_answers(leaky, self.answers))
        leaky_name = {"events": [{"activity": "人物7 生日"}]}
        self.assertTrue(input_leaks_answers(leaky_name, self.answers))


if __name__ == "__main__":
    unittest.main()
