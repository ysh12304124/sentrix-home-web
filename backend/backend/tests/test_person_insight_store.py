import tempfile
import unittest

from backend.db import MemoryStore


class PersonInsightStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/sentrix.db"
        self.store = MemoryStore(self.db_path)
        self.store.create_memory_space("album-a", "相册 A")
        self.store.create_memory_space("album-b", "相册 B")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def make_face_fixture(self, scope_id, asset_id="a1", embedding=(1, 0, 0)):
        asset = self.store.create_asset(
            asset_id, "one.jpg", "image", "/tmp/one.jpg", "image/jpeg", scope_id=scope_id
        )
        obs = self.store.add_observation(
            asset_id, {"caption": "家人合影", "people": [], "captured_at": "2026-01-01T10:00:00"},
            scope_id=scope_id,
        )
        face = self.store.add_face_instance(
            asset_id, obs["id"],
            {"bbox": [1, 2, 3, 4], "confidence": 0.95, "quality": 0.9, "embedding": list(embedding)},
        )
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id=scope_id)
        event = self.store.merge_observation_into_event(obs)
        return {
            "person_id": person["id"],
            "cluster_id": face["cluster_id"],
            "event_id": event["id"],
            "observation_id": obs["id"],
            "asset_id": asset["id"],
            "face_instance_id": face["id"],
        }

    def test_five_new_tables_exist(self):
        tables = {
            row["name"]
            for row in self.store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in (
            "person_insight_runs",
            "person_moments",
            "person_role_hypotheses",
            "relationship_hypotheses",
            "person_portrait_revisions",
        ):
            self.assertIn(expected, tables)

    def test_person_identity_states_are_independent(self):
        person = self.store.create_entity(
            "待确认人物簇", "person", "pending", scope_id="album-a"
        )
        updated = self.store.update_person_identity_state(
            person["id"], role="母亲", role_state="confirmed"
        )
        self.assertEqual(updated["identity_state"], "clustered")
        self.assertEqual(updated["role_state"], "confirmed")
        self.assertEqual(updated["name_state"], "anonymous")
        self.assertEqual(updated["family_role"], "母亲")

    def test_person_moment_cannot_cross_memory_spaces(self):
        fixture = self.make_face_fixture("album-a")
        other = self.store.create_entity(
            "另一个相册的人", "person", "pending", scope_id="album-b"
        )
        with self.assertRaisesRegex(ValueError, "same memory space"):
            self.store.upsert_person_moment({
                **fixture,
                "person_id": other["id"],
                "action_text": "抱着孩子",
                "prompt_version": "person-moment-v1",
            })

    def test_person_moment_json_fields_roundtrip(self):
        fixture = self.make_face_fixture("album-a")
        moment = self.store.upsert_person_moment({
            **fixture,
            "action_text": "抱着孩子",
            "interaction_target_ids": [fixture["person_id"]],
            "interaction_text": "和孩子互动",
            "participation_style": "照顾",
            "visible_affect": "微笑",
            "social_role_cues": ["照顾"],
            "narrative_note": "反复出现",
            "confidence": 0.8,
            "model_name": "gemma4-12b-it",
            "prompt_version": "person-moment-v1",
        })
        self.assertEqual(moment["action_text"], "抱着孩子")
        self.assertEqual(moment["interaction_target_ids"], [fixture["person_id"]])
        self.assertEqual(moment["social_role_cues"], ["照顾"])
        self.assertEqual(moment["person_id"], fixture["person_id"])
        self.assertEqual(moment["event_id"], fixture["event_id"])
        self.assertEqual(moment["asset_id"], fixture["asset_id"])
        self.assertEqual(moment["observation_id"], fixture["observation_id"])
        self.assertEqual(moment["face_instance_id"], fixture["face_instance_id"])
        self.assertEqual(moment["cluster_id"], fixture["cluster_id"])
        listed = self.store.list_person_moments(person_id=fixture["person_id"], scope_id="album-a")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["interaction_target_ids"], [fixture["person_id"]])

    def test_hypotheses_are_scope_isolated(self):
        run_a = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        run_b = self.store.create_person_insight_run("album-b", {"max_core_people": 10})
        person_a = self.store.create_entity("甲", "person", "pending", scope_id="album-a")
        person_b = self.store.create_entity("乙", "person", "pending", scope_id="album-b")
        self.store.replace_role_hypotheses("album-a", run_a["id"], [{
            "person_id": person_a["id"], "role": "父亲", "rank": 1, "confidence": 0.9,
            "prompt_version": "person-graph-v1",
        }])
        self.store.replace_role_hypotheses("album-b", run_b["id"], [{
            "person_id": person_b["id"], "role": "母亲", "rank": 1, "confidence": 0.9,
            "prompt_version": "person-graph-v1",
        }])
        only_a = self.store.list_role_hypotheses(scope_id="album-a")
        self.assertEqual(len(only_a), 1)
        self.assertEqual(only_a[0]["person_id"], person_a["id"])
        self.assertEqual(only_a[0]["role"], "父亲")
        # Replacing album-a must not touch album-b hypotheses.
        self.store.replace_role_hypotheses("album-a", run_a["id"], [{
            "person_id": person_a["id"], "role": "配偶", "rank": 1, "confidence": 0.6,
            "prompt_version": "person-graph-v2",
        }])
        still_b = self.store.list_role_hypotheses(scope_id="album-b")
        self.assertEqual(len(still_b), 1)
        self.assertEqual(still_b[0]["role"], "母亲")

    def test_role_hypothesis_rejects_cross_scope_person(self):
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        outsider = self.store.create_entity("外人", "person", "pending", scope_id="album-b")
        with self.assertRaisesRegex(ValueError, "same memory space"):
            self.store.replace_role_hypotheses("album-a", run["id"], [{
                "person_id": outsider["id"], "role": "父亲", "rank": 1,
                "prompt_version": "person-graph-v1",
            }])

    def test_relationship_requires_distinct_people_and_scope(self):
        run_a = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        person_a = self.store.create_entity("甲", "person", "pending", scope_id="album-a")
        person_b = self.store.create_entity("乙", "person", "pending", scope_id="album-a")
        self.store.replace_relationship_hypotheses("album-a", run_a["id"], [{
            "subject_person_id": person_a["id"], "predicate": "母亲",
            "object_person_id": person_b["id"], "inverse_predicate": "孩子",
            "confidence": 0.7, "prompt_version": "person-graph-v1",
        }])
        listed = self.store.list_relationship_hypotheses("album-a")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["subject_person_id"], person_a["id"])
        self.assertEqual(listed[0]["inverse_predicate"], "孩子")
        with self.assertRaisesRegex(ValueError, "same memory space"):
            self.store.replace_relationship_hypotheses("album-a", run_a["id"], [{
                "subject_person_id": person_a["id"], "predicate": "朋友",
                "object_person_id": self.store.create_entity(
                    "丙", "person", "pending", scope_id="album-b"
                )["id"],
                "inverse_predicate": "朋友", "prompt_version": "person-graph-v1",
            }])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.store.replace_relationship_hypotheses("album-a", run_a["id"], [{
                "subject_person_id": person_a["id"], "predicate": "朋友",
                "object_person_id": person_a["id"],
                "inverse_predicate": "朋友", "prompt_version": "person-graph-v1",
            }])

    def test_insight_run_status_and_retry(self):
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        self.assertEqual(run["status"], "queued")
        self.assertEqual(run["current_stage"], "queued")
        updated = self.store.update_person_insight_run(
            run["id"], status="running", stage="extract_moments",
            stats={"moments": 3},
        )
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["current_stage"], "extract_moments")
        self.assertEqual(updated["stats"], {"moments": 3})
        fetched = self.store.get_person_insight_run(run["id"])
        self.assertEqual(fetched["id"], run["id"])
        latest = self.store.latest_person_insight_run("album-a")
        self.assertEqual(latest["id"], run["id"])
        self.assertIsNone(self.store.latest_person_insight_run("album-b"))

    def test_portrait_versions_lock_and_feedback(self):
        person = self.store.create_entity("核心人物 2", "person", "pending", scope_id="album-a")
        v1 = self.store.create_portrait_revision(person["id"], {
            "portrait_text": "从照片看，他常常把大家聚在一起。",
            "themes": [{"title": "把大家聚在一起", "summary": "常见于聚会", "evidence_refs": []}],
            "evidence_refs": [],
            "trigger_type": "initial",
            "model_name": "gemma4-12b-it",
            "prompt_version": "person-portrait-v1",
        })
        self.assertEqual(v1["revision"], 1)
        self.assertEqual(v1["status"], "active")
        v2 = self.store.create_portrait_revision(person["id"], {
            "portrait_text": "第二个版本的画像。",
            "themes": [],
            "evidence_refs": [],
            "trigger_type": "role_change",
            "model_name": "gemma4-12b-it",
            "prompt_version": "person-portrait-v1",
        })
        self.assertEqual(v2["revision"], 2)
        self.assertEqual(v2["status"], "active")
        self.assertEqual(self.store.get_active_portrait(person["id"])["id"], v2["id"])
        # v1 must be superseded once a newer revision exists.
        superseded = [r for r in self.store.list_portrait_revisions(person["id"]) if r["id"] == v1["id"]]
        self.assertEqual(superseded[0]["status"], "superseded")
        locked = self.store.set_portrait_lock(v2["id"], True)
        self.assertEqual(locked["status"], "user_locked")
        # A locked portrait is not overwritten by a new generation.
        kept = self.store.create_portrait_revision(person["id"], {
            "portrait_text": "不应该落库的新版本。",
            "themes": [],
            "evidence_refs": [],
            "trigger_type": "retry",
            "model_name": "gemma4-12b-it",
            "prompt_version": "person-portrait-v1",
        })
        self.assertEqual(kept["id"], v2["id"])
        self.assertEqual(len(self.store.list_portrait_revisions(person["id"])), 2)
        feedback = self.store.set_portrait_feedback(v2["id"], "像他")
        self.assertEqual(feedback["user_feedback"], "像他")

    def test_legacy_confirmed_person_migration_is_idempotent(self):
        self.store.create_entity("爸爸", "person", "confirmed", family_role="父亲", scope_id="album-a")
        self.store.close()
        # Re-open triggers migration in _create_schema.
        self.store = MemoryStore(self.db_path)
        person = [
            p for p in self.store.list_entities(scope_id="album-a")
            if p["entity_type"] == "person"
        ][0]
        self.assertEqual(person["identity_state"], "stable")
        self.assertEqual(person["role_state"], "confirmed")
        self.assertEqual(person["name_state"], "confirmed")
        self.store.close()
        # Re-open again: migration must be idempotent and not raise.
        self.store = MemoryStore(self.db_path)
        person = [
            p for p in self.store.list_entities(scope_id="album-a")
            if p["entity_type"] == "person"
        ][0]
        self.assertEqual(person["identity_state"], "stable")
        self.assertEqual(person["role_state"], "confirmed")
        self.assertEqual(person["name_state"], "confirmed")


if __name__ == "__main__":
    unittest.main()
