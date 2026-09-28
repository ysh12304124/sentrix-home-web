import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.person_moments import PersonMomentExtractor, normalize_person_moments, render_numbered_preview


def _moment(label, action="动作", interaction=None, style="共同参与", affect="微笑",
            cues=None, confidence=0.8):
    return {
        "label": label, "action_text": action,
        "interaction_labels": interaction or [], "interaction_text": "",
        "participation_style": style, "visible_affect": affect,
        "social_role_cues": cues or [], "narrative_note": "", "confidence": confidence,
    }


class AnalyzePersonMomentsContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = f"{self.temp_dir.name}/preview.jpg"
        Image.new("RGB", (100, 80), "white").save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _client(self, response):
        client = GammaClient(backend="vllm")
        client.chat = lambda *args, **kwargs: response
        return client

    def test_accepts_only_preview_labels(self):
        client = self._client(json.dumps({"moments": [
            _moment("P1", action="抱着孩子"),
            _moment("P3", action="未知人物"),
        ]}))
        result = client.analyze_person_moments(self.image_path, ["P1", "P2"], {})
        self.assertEqual([m["label"] for m in result["moments"]], ["P1"])

    def test_drops_empty_actions(self):
        client = self._client(json.dumps({"moments": [
            _moment("P1", action=""),
        ]}))
        result = client.analyze_person_moments(self.image_path, ["P1"], {})
        self.assertEqual(result["moments"], [])

    def test_drops_sensitive_role_cues(self):
        client = self._client(json.dumps({"moments": [
            _moment("P1", cues=["照顾", "年收入很高", "信佛教", "汉族"]),
        ]}))
        result = client.analyze_person_moments(self.image_path, ["P1"], {})
        cues = result["moments"][0]["social_role_cues"]
        self.assertIn("照顾", cues)
        self.assertNotIn("年收入很高", cues)
        self.assertNotIn("信佛教", cues)
        self.assertNotIn("汉族", cues)

    def test_normalizes_unknown_participation_style(self):
        client = self._client(json.dumps({"moments": [
            _moment("P1", style="搭把手"),
        ]}))
        result = client.analyze_person_moments(self.image_path, ["P1"], {})
        self.assertEqual(result["moments"][0]["participation_style"], "无法判断")

    def test_normalize_rejects_sensitive_affect(self):
        normalized = normalize_person_moments({"moments": [
            _moment("P1", affect="这个人性格内向抑郁"),
        ]}, ["P1"])
        self.assertEqual(normalized[0]["visible_affect"], "")


class RenderNumberedPreviewTests(unittest.TestCase):
    def _make_image(self, name="preview.jpg", size=(100, 80)):
        path = f"{self.temp_dir.name}/{name}"
        Image.new("RGB", size, "white").save(path)
        return path

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_renders_preview_and_label_map(self):
        path = self._make_image()
        faces = [{
            "face_instance_id": "f1", "person_id": "p1", "cluster_id": "c1",
            "observation_id": "o1", "event_id": "e1", "bbox": [10, 10, 20, 20],
        }]
        preview, label_map = render_numbered_preview(path, faces)
        try:
            with Image.open(preview) as image:
                self.assertEqual(image.size, (100, 80))
            self.assertEqual(label_map["P1"]["face_instance_id"], "f1")
            self.assertEqual(label_map["P1"]["person_id"], "p1")
            self.assertEqual(label_map["P1"]["cluster_id"], "c1")
        finally:
            Path(preview).unlink(missing_ok=True)


class FakeGamma:
    def __init__(self, responses=None, fail_asset=None):
        self.responses = list(responses or [])
        self.fail_asset = fail_asset
        self.model = "fake-model"

    def analyze_person_moments(self, path, labels, context):
        if self.fail_asset and context.get("asset_id") == self.fail_asset:
            raise RuntimeError("vlm down")
        response = self.responses.pop(0) if self.responses else {"moments": []}
        if isinstance(response, dict) and "moments" not in response:
            response = {"moments": [response]}
        return response


class PersonMomentExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.store.create_memory_space("album-b", "相册 B")
        self.run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _photo_asset(self, scope_id, asset_id, path):
        self.store.create_asset(
            asset_id, f"{asset_id}.jpg", "image", path, "image/jpeg", scope_id=scope_id
        )
        obs = self.store.add_observation(
            asset_id, {"caption": "合影", "captured_at": "2026-01-01T10:00:00"},
            scope_id=scope_id,
        )
        face = self.store.add_face_instance(
            asset_id, obs["id"],
            {"bbox": [10, 10, 20, 20], "confidence": 0.95, "quality": 0.9, "embedding": [1, 0, 0]},
        )
        event = self.store.merge_observation_into_event(obs)
        return asset_id, obs["id"], face["id"], face["cluster_id"], event["id"]

    def _selection(self, asset_id, obs_id, face_id, cluster_id, event_id, person_id):
        return {
            "asset_id": asset_id, "face_instance_id": face_id, "person_id": person_id,
            "cluster_id": cluster_id, "observation_id": obs_id, "event_id": event_id,
        }

    def test_extractor_binds_evidence_and_skips_unknown_targets(self):
        path = f"{self.temp_dir.name}/a1.jpg"
        Image.new("RGB", (100, 80), "white").save(path)
        asset_id, obs_id, face_id, cluster_id, event_id = self._photo_asset("album-a", "a1", path)
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")

        gamma = FakeGamma([{"moments": [
            _moment("P1", action="抱着孩子", interaction=["P1", "P9"]),
        ]}])
        extractor = PersonMomentExtractor(self.store, gamma)
        result = extractor.extract("album-a", self.run["id"], [
            self._selection(asset_id, obs_id, face_id, cluster_id, event_id, person["id"]),
        ])
        self.assertEqual(result["moments"], 1)
        moments = self.store.list_person_moments(scope_id="album-a")
        self.assertEqual(len(moments), 1)
        moment = moments[0]
        self.assertEqual(moment["person_id"], person["id"])
        self.assertEqual(moment["asset_id"], asset_id)
        self.assertEqual(moment["event_id"], event_id)
        self.assertEqual(moment["face_instance_id"], face_id)
        # P1 targets self; P9 is not a preview label so it is dropped.
        self.assertEqual(moment["interaction_target_ids"], [person["id"]])
        self.assertEqual(moment["action_text"], "抱着孩子")

    def test_extractor_records_failure_and_continues(self):
        path_a = f"{self.temp_dir.name}/a1.jpg"
        path_b = f"{self.temp_dir.name}/b1.jpg"
        Image.new("RGB", (100, 80), "white").save(path_a)
        Image.new("RGB", (100, 80), "black").save(path_b)
        a_asset, a_obs, a_face, a_cluster, a_event = self._photo_asset("album-a", "a1", path_a)
        b_asset, b_obs, b_face, b_cluster, b_event = self._photo_asset("album-a", "b1", path_b)
        person_a = self.store.create_entity("人物甲", "person", "pending", scope_id="album-a")
        person_b = self.store.create_entity("人物乙", "person", "pending", scope_id="album-a")

        gamma = FakeGamma([_moment("P1", action="失败照片"), _moment("P1", action="正常照片")],
                          fail_asset=a_asset)
        extractor = PersonMomentExtractor(self.store, gamma)
        result = extractor.extract("album-a", self.run["id"], [
            self._selection(a_asset, a_obs, a_face, a_cluster, a_event, person_a["id"]),
            self._selection(b_asset, b_obs, b_face, b_cluster, b_event, person_b["id"]),
        ])
        self.assertEqual(result["failures"], 1)
        moments = self.store.list_person_moments(scope_id="album-a")
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["person_id"], person_b["id"])
        run = self.store.get_person_insight_run(self.run["id"])
        self.assertGreaterEqual(run["stats"].get("moment_failures", 0), 1)


if __name__ == "__main__":
    unittest.main()
