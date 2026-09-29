import os
import tempfile
import unittest

# Point the app's global store at a throwaway database before importing it.
_APP_TMP = tempfile.mkdtemp(prefix="sentrix-api-test-")
os.environ["SENTRIX_DB_PATH"] = os.path.join(_APP_TMP, "sentrix.db")
os.environ["SENTRIX_DATA_DIR"] = os.path.join(_APP_TMP, "data")

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend.db import MemoryStore  # noqa: E402


def _make_hypothesis(store, person_id, role="母亲", rank=1, relative_to=None):
    run = store.create_person_insight_run("album-a", {"max_core_people": 10})
    rows = [{
        "person_id": person_id, "role": role, "rank": rank, "confidence": 0.72,
        "prompt_version": "person-graph-v1",
    }]
    if relative_to:
        rows[0]["relative_to_person_id"] = relative_to
    return store.replace_role_hypotheses("album-a", run["id"], rows)[0]


class RoleNameStateTransitionTests(unittest.TestCase):
    """DB-level state transitions: role, name and identity stay independent."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_confirm_role_keeps_name_anonymous(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"])
        updated = self.store.confirm_role_hypothesis(hyp["id"])
        self.assertEqual(updated["family_role"], "母亲")
        self.assertEqual(updated["role_state"], "confirmed")
        self.assertEqual(updated["name_state"], "anonymous")
        self.assertEqual(updated["canonical_name"], "待确认人物簇")

    def test_rename_keeps_role_unknown(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        renamed = self.store.rename_person(person["id"], "妈妈")
        entity = renamed["entity"]
        self.assertEqual(entity["canonical_name"], "妈妈")
        self.assertEqual(entity["name_state"], "confirmed")
        self.assertEqual(entity["role_state"], "unknown")

    def test_confirm_owner_role_sets_self(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"], role="本人")
        updated = self.store.confirm_role_hypothesis(hyp["id"], is_self=True)
        self.assertEqual(updated["family_role"], "本人")
        self.assertEqual(updated["role_state"], "confirmed")

    def test_confirm_unknown_role_does_not_invent_fact(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"], role="无法判断")
        updated = self.store.confirm_role_hypothesis(hyp["id"])
        self.assertEqual(updated["role_state"], "confirmed")
        self.assertIsNone(updated["family_role"])

    def test_reject_supersedes_hypothesis(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"])
        self.store.reject_role_hypothesis(hyp["id"])
        listed = self.store.list_role_hypotheses(person_id=person["id"], status="suggested")
        self.assertEqual(listed, [])

    def test_confirm_relationship_creates_edge_and_claims(self):
        person_a = self.store.create_entity("甲", "person", "confirmed", scope_id="album-a")
        person_b = self.store.create_entity("乙", "person", "confirmed", scope_id="album-a")
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        rel_hyp = self.store.replace_relationship_hypotheses("album-a", run["id"], [{
            "subject_person_id": person_a["id"], "predicate": "母亲",
            "object_person_id": person_b["id"], "inverse_predicate": "孩子",
            "confidence": 0.7, "prompt_version": "person-graph-v1",
        }])[0]
        before = self.store.count("relationships")
        self.store.confirm_relationship_hypothesis(rel_hyp["id"])
        self.assertGreater(self.store.count("relationships"), before)
        relationships = self.store.list_relationships(scope_id="album-a")
        edge = next(
            (r for r in relationships
             if r["subject_entity_id"] == person_a["id"] and r["predicate"] == "母亲"
             and r["object_entity_id"] == person_b["id"]),
            None,
        )
        self.assertIsNotNone(edge)
        self.assertEqual(edge["inverse_predicate"], "孩子")
        self.assertGreater(self.store.count("semantic_claims"), 0)

    def test_correction_supersedes_hypotheses_keeps_portraits(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"])
        portrait = self.store.create_portrait_revision(person["id"], {
            "portrait_text": "第一版画像。",
            "themes": [], "evidence_refs": [], "trigger_type": "initial",
            "model_name": "gemma4-12b-it", "prompt_version": "person-portrait-v1",
        })
        affected = self.store.supersede_conflicting_hypotheses(person["id"])
        self.store.mark_portraits_stale([person["id"], *affected])
        # The old portrait stays in history but is no longer active.
        listed = self.store.list_portrait_revisions(person["id"])
        self.assertTrue(any(item["id"] == portrait["id"] for item in listed))
        self.assertEqual(self.store.get_active_portrait(person["id"]), None)
        # The hypothesis is no longer suggested.
        self.assertEqual(
            self.store.list_role_hypotheses(person_id=person["id"], status="suggested"), []
        )


class PersonInsightApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        app_module.store = self.store
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.temp_dir.cleanup()

    def test_get_person_insights_returns_run_and_tiers(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        _make_hypothesis(self.store, person["id"])
        response = self.client.get("/api/person-insights", params={"scope_id": "album-a"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["scope_id"], "album-a")
        self.assertIsNotNone(payload["run"])
        for tier in ("core", "common", "incidental"):
            self.assertIn(tier, payload["tiers"])

    def test_role_decision_confirm(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"])
        response = self.client.post(
            f"/api/people/{person['id']}/role-decision",
            json={"hypothesis_id": hyp["id"], "decision": "confirm", "role": "母亲", "is_self": False},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["person"]["family_role"], "母亲")

    def test_role_decision_reject(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        hyp = _make_hypothesis(self.store, person["id"])
        response = self.client.post(
            f"/api/people/{person['id']}/role-decision",
            json={"hypothesis_id": hyp["id"], "decision": "reject"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_patch_name(self):
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        response = self.client.patch(
            f"/api/people/{person['id']}/name", json={"name": "妈妈"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get_entity(person["id"])["canonical_name"], "妈妈")

    def test_relationship_decision_confirm(self):
        person_a = self.store.create_entity("甲", "person", "confirmed", scope_id="album-a")
        person_b = self.store.create_entity("乙", "person", "confirmed", scope_id="album-a")
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        rel_hyp = self.store.replace_relationship_hypotheses("album-a", run["id"], [{
            "subject_person_id": person_a["id"], "predicate": "母亲",
            "object_person_id": person_b["id"], "inverse_predicate": "孩子",
            "confidence": 0.7, "prompt_version": "person-graph-v1",
        }])[0]
        response = self.client.post(
            f"/api/relationship-hypotheses/{rel_hyp['id']}/decision",
            json={"decision": "confirm"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        relationships = self.store.list_relationships(scope_id="album-a")
        self.assertTrue(any(
            r["subject_entity_id"] == person_a["id"] and r["predicate"] == "母亲"
            for r in relationships
        ))


if __name__ == "__main__":
    unittest.main()
