import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.person_portraits import (
    compile_portrait_evidence,
    deterministic_portrait,
    validate_portrait,
)


def _valid_portrait(pack, hedge=True):
    refs = [{"kind": "person_moment", "id": moment["id"]} for moment in pack["moments"]]
    if hedge:
        role_line = "从照片看，他可能是一位母亲。"
        opening = "从照片看，"
    else:
        role_line = "他是这个家庭里的母亲。"
        opening = ""
    text = (
        opening + "这位家庭成员常常把大家聚在一起，反复照顾孩子，也常张罗饭桌。"
        "他陪伴家人出游，把大家聚在一起，反复照顾孩子，也常张罗饭桌。"
        + role_line +
        "他让人安心，把大家聚在一起，反复照顾孩子，也常张罗饭桌，让家有了温度。"
    )
    return {
        "portrait_text": text,
        "themes": [
            {"title": "把大家聚在一起", "summary": "常见于家庭聚会", "evidence_refs": refs[:1]},
            {"title": "照顾孩子", "summary": "反复出现的互动", "evidence_refs": refs[1:2]},
        ],
    }


class PortraitEvidencePackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.person, self.other = self._seed()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _photo(self, asset_id, path, captured_at):
        self.store.create_asset(
            asset_id, f"{asset_id}.jpg", "image", path, "image/jpeg", scope_id="album-a"
        )
        obs = self.store.add_observation(
            asset_id, {"caption": "合影", "captured_at": captured_at}, scope_id="album-a"
        )
        face = self.store.add_face_instance(
            asset_id, obs["id"],
            {"bbox": [10, 10, 20, 20], "confidence": 0.95, "quality": 0.9, "embedding": [1, 0, 0]},
        )
        event = self.store.merge_observation_into_event(obs)
        return obs["id"], face["id"], face["cluster_id"], event["id"]

    def _seed(self):
        base = {
            "person_id": None,
            "event_id": None,
            "observation_id": None,
            "asset_id": None,
            "face_instance_id": None,
            "cluster_id": None,
            "prompt_version": "person-moment-v1",
        }
        moments = []
        for index in range(3):
            asset_id = f"a{index + 1}"
            path = f"{self.temp_dir.name}/{asset_id}.jpg"
            Image.new("RGB", (100, 80), "white").save(path)
            obs_id, face_id, cluster_id, event_id = self._photo(
                asset_id, path, f"2026-01-0{index + 1}T10:00:00"
            )
            payload = dict(base)
            payload.update({
                "person_id": self.person["id"] if hasattr(self, "person") else "placeholder",
                "asset_id": asset_id,
                "observation_id": obs_id,
                "face_instance_id": face_id,
                "cluster_id": cluster_id,
                "event_id": event_id,
                "action_text": "抱着孩子" if index < 2 else "张罗饭桌",
                "confidence": 0.8 if index < 2 else 0.7,
            })
            moments.append(payload)
        person = self.store.create_entity("待确认人物簇", "person", "pending", scope_id="album-a")
        other = self.store.create_entity("另一位", "person", "confirmed", scope_id="album-a")
        for moment in moments:
            moment["person_id"] = person["id"]
            self.store.upsert_person_moment(moment)
        self.store.create_relationship(person["id"], "配偶", other["id"], confidence=0.8, status="active")
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        self.store.replace_relationship_hypotheses("album-a", run["id"], [{
            "subject_person_id": person["id"], "predicate": "朋友",
            "object_person_id": other["id"], "inverse_predicate": "朋友",
            "confidence": 0.6, "prompt_version": "person-graph-v1",
        }])
        return person, other

    def test_evidence_pack_layers_moments_and_relationships(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        self.assertEqual(pack["person"]["id"], self.person["id"])
        self.assertEqual(len(pack["confirmed_relationships"]), 1)
        self.assertEqual(len(pack["suggested_relationships"]), 1)
        # Near-duplicate moments collapse to one; distinct action stays.
        actions = [m["action_text"] for m in pack["moments"]]
        self.assertEqual(actions.count("抱着孩子"), 1)
        self.assertIn("张罗饭桌", actions)
        # Every moment carries a traceable evidence envelope.
        for moment in pack["moments"]:
            for key in ("kind", "id", "asset_id", "observation_id", "event_id"):
                self.assertIn(key, moment)

    def test_validate_accepts_valid_portrait(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        ok, errors = validate_portrait(pack, _valid_portrait(pack))
        self.assertTrue(ok, errors)

    def test_validate_rejects_short_portrait(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        ok, errors = validate_portrait(pack, {"portrait_text": "太短", "themes": []})
        self.assertFalse(ok)
        self.assertTrue(any("80" in error for error in errors))

    def test_validate_rejects_sensitive_portrait(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        portrait = _valid_portrait(pack)
        portrait["portrait_text"] = portrait["portrait_text"] + "他年收入很高，是家里经济支柱。"
        ok, errors = validate_portrait(pack, portrait)
        self.assertFalse(ok)
        self.assertTrue(any("sensitive" in error for error in errors))

    def test_validate_requires_hedging_for_unconfirmed_role(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        ok, errors = validate_portrait(pack, _valid_portrait(pack, hedge=False))
        self.assertFalse(ok)
        self.assertTrue(any("hedg" in error.lower() for error in errors))

    def test_validate_rejects_theme_without_evidence(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        portrait = _valid_portrait(pack)
        portrait["themes"].append({"title": "无证据主题", "summary": "缺少引用", "evidence_refs": []})
        ok, errors = validate_portrait(pack, portrait)
        self.assertFalse(ok)
        self.assertTrue(any("evidence_refs" in error for error in errors))

    def test_deterministic_portrait_uses_hedged_language(self):
        pack = compile_portrait_evidence(self.store, self.person["id"])
        portrait = deterministic_portrait(pack)
        self.assertIn("从照片看", portrait["portrait_text"])
        self.assertNotIn("总是", portrait["portrait_text"])
        self.assertNotIn("年收入", portrait["portrait_text"])


class PortraitWriterContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image = f"{self.temp_dir.name}/p.jpg"
        Image.new("RGB", (64, 64), "white").save(self.image)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_portrait_uses_writer_role(self):
        client = GammaClient(backend="vllm")
        captured = {}

        def fake_chat(prompt, images=None, vision_options=None, json_mode=True, role=None):
            captured["role"] = role
            return json.dumps({
                "portrait_text": "从照片看，这位家庭成员常常把大家聚在一起，反复照顾孩子，也常张罗饭桌。",
                "themes": [{
                    "title": "把大家聚在一起", "summary": "常见于聚会",
                    "evidence_refs": [{"kind": "person_moment", "id": "moment_x"}],
                }],
            })

        with patch.object(client, "chat", side_effect=fake_chat):
            result = client.write_person_portrait({"person": {"id": "p1"}, "moments": []})
        self.assertEqual(captured["role"], "writer")
        self.assertIn("portrait_text", result)
        self.assertEqual(result["themes"][0]["evidence_refs"][0]["id"], "moment_x")


if __name__ == "__main__":
    unittest.main()
