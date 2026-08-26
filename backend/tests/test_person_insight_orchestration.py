import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline
from backend.person_insights import PersonInsightService, STAGES


class FakeOrchestrationGamma:
    def __init__(self, fail_graph=False):
        self.model = "fake-model"
        self.fail_graph = fail_graph

    def analyze_person_moments(self, path, labels, context):
        return {"moments": [{
            "label": labels[0] if labels else "P1", "action_text": "抱着孩子",
            "interaction_labels": [], "interaction_text": "",
            "participation_style": "照顾", "visible_affect": "微笑",
            "social_role_cues": [], "narrative_note": "", "confidence": 0.8,
        }]}

    def infer_person_graph(self, paths, graph_payload, role="verify"):
        if self.fail_graph:
            raise RuntimeError("vlm down during graph inference")
        people = list(graph_payload.get("people") or [])
        if not people:
            return {"album_owner_candidates": [], "roles": [], "relationships": []}
        owner = people[0]
        roles = [{
            "person_ref": owner, "relative_to": owner,
            "candidates": [
                {"role": "母亲", "confidence": 0.7, "reason": "反复照顾"},
                {"role": "无法判断", "confidence": 0.1, "reason": ""},
            ],
        }]
        return {"album_owner_candidates": [], "roles": roles, "relationships": []}

    def write_person_portrait(self, pack, role="writer"):
        refs = [{"kind": "person_moment", "id": m["id"]} for m in (pack.get("moments") or [])]
        ref = refs[0] if refs else {"kind": "person_moment", "id": "none"}
        return {
            "portrait_text": (
                "从照片看，这位家庭成员常常把大家聚在一起，反复照顾孩子，也常张罗饭桌。"
                "他陪伴家人出游，把大家聚在一起，反复照顾孩子，也常张罗饭桌。"
                "从照片看，他可能是一位母亲，把大家聚在一起，反复照顾孩子。"
                "他让人安心，把大家聚在一起，反复照顾孩子，也常张罗饭桌，让家有了温度。"
            ),
            "themes": [
                {"title": "把大家聚在一起", "summary": "常见于聚会", "evidence_refs": [ref]},
                {"title": "照顾孩子", "summary": "反复出现", "evidence_refs": [ref]},
            ],
        }


class PersonInsightOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.person = self.store.create_entity(
            "妈妈", "person", "confirmed", family_role="母亲", scope_id="album-a"
        )
        path = f"{self.temp_dir.name}/a1.jpg"
        Image.new("RGB", (100, 80), "white").save(path)
        self.store.create_asset("a1", "a1.jpg", "image", path, "image/jpeg", scope_id="album-a")
        obs = self.store.add_observation(
            "a1", {"caption": "合影", "captured_at": "2026-01-01T10:00:00"}, scope_id="album-a"
        )
        self.face = self.store.add_face_instance(
            "a1", obs["id"],
            {"bbox": [10, 10, 20, 20], "confidence": 0.95, "quality": 0.9, "embedding": [1, 0, 0]},
        )
        self.store.merge_observation_into_event(obs)
        # Link the confirmed person to the face cluster so features rank it.
        self.store.connection.execute(
            "UPDATE face_clusters SET entity_id = ? WHERE id = ?",
            (self.person["id"], self.face["cluster_id"]),
        )
        self.store.connection.commit()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _run(self, gamma, trigger="manual"):
        config = {"max_core_people": 10, "trigger_type": trigger}
        run = self.store.create_person_insight_run("album-a", config)
        service = PersonInsightService(self.store, gamma)
        service.run(run["id"], "album-a", config)
        return run["id"]

    def test_run_completes_all_stages(self):
        run_id = self._run(FakeOrchestrationGamma())
        run = self.store.get_person_insight_run(run_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["current_stage"], "done")
        self.assertGreater(len(self.store.list_person_moments(scope_id="album-a")), 0)
        self.assertGreater(len(self.store.list_role_hypotheses(scope_id="album-a")), 0)
        self.assertIsNotNone(self.store.get_active_portrait(self.person["id"]))
        self.assertEqual(run["stats"]["event_watermark"],
                         self.store.connection.execute(
                             "SELECT COUNT(*) FROM events WHERE scope_id = 'album-a'"
                         ).fetchone()[0])

    def test_retry_resumes_from_failed_stage(self):
        run_id = self._run(FakeOrchestrationGamma(fail_graph=True))
        run = self.store.get_person_insight_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["current_stage"], "infer_graph")
        # Retry with a working gamma; it resumes at infer_graph and completes.
        retried = PersonInsightService(self.store, FakeOrchestrationGamma())
        retried.run(run_id, "album-a", {"max_core_people": 10})
        run = self.store.get_person_insight_run(run_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["current_stage"], "done")

    def test_scope_has_single_active_run(self):
        config = {"max_core_people": 10, "trigger_type": "manual"}
        first = self.store.create_person_insight_run("album-a", config)
        second = self.store.create_person_insight_run("album-a", config)
        service = PersonInsightService(self.store, FakeOrchestrationGamma())
        service.run(first["id"], "album-a", config)
        # A run that is already running must not be started a second time.
        self.store.update_person_insight_run(first["id"], status="running")
        with self.assertRaisesRegex(RuntimeError, "already running"):
            service.run(first["id"], "album-a", config)
        self.assertEqual(self.store.get_person_insight_run(second["id"])["status"], "queued")


class IngestAllowlistGatingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _pipeline(self):
        pipeline = IngestionPipeline(self.store, gamma=FakeOrchestrationGamma())
        return pipeline

    def _trigger(self, pipeline, scope_id):
        # Run the spawned thread synchronously so it finishes before teardown.
        real_start = threading.Thread.start

        def sync_start(thread):
            real_start(thread)
            thread.join()

        with patch.object(threading.Thread, "start", sync_start):
            return pipeline._maybe_trigger_person_insight(scope_id)

    def test_scope_outside_allowlist_does_not_trigger(self):
        pipeline = self._pipeline()
        with patch.dict(os.environ, {"SENTRIX_PERSON_INSIGHT_SCOPES": "album-b"}, clear=False):
            result = self._trigger(pipeline, "album-a")
        self.assertIsNone(result)
        self.assertIsNone(self.store.latest_person_insight_run("album-a"))

    def test_scope_in_allowlist_triggers_when_new_events(self):
        pipeline = self._pipeline()
        with patch.dict(os.environ, {"SENTRIX_PERSON_INSIGHT_SCOPES": "album-a"}, clear=False):
            run = self._trigger(pipeline, "album-a")
        self.assertIsNotNone(run)
        self.assertEqual(run["scope_id"], "album-a")

    def test_no_new_events_does_not_retrigger(self):
        pipeline = self._pipeline()
        with patch.dict(os.environ, {"SENTRIX_PERSON_INSIGHT_SCOPES": "album-a"}, clear=False):
            first = self._trigger(pipeline, "album-a")
            # Simulate the completed run's watermark being persisted.
            self.store.update_person_insight_run(first["id"], status="completed", stats={
                "event_watermark": self.store.connection.execute(
                    "SELECT COUNT(*) FROM events WHERE scope_id = 'album-a'"
                ).fetchone()[0],
            })
            second = self._trigger(pipeline, "album-a")
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
