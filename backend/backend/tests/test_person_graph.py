import json
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.person_graph import (
    INVERSE_RELATION,
    ROLE_OPTIONS,
    apply_relationship_threshold,
    ensure_unknown_fallback,
    find_graph_violations,
    normalize_person_graph,
    normalize_relationship,
    sanitize_reason,
)


class PersonGraphConstraintTests(unittest.TestCase):
    def test_inverse_relation_table(self):
        self.assertEqual(INVERSE_RELATION["父亲"], "孩子")
        self.assertEqual(INVERSE_RELATION["母亲"], "孩子")
        self.assertEqual(INVERSE_RELATION["孩子"], "父母")
        self.assertEqual(INVERSE_RELATION["祖父母"], "孙辈")
        self.assertEqual(INVERSE_RELATION["孙辈"], "祖父母")
        self.assertEqual(INVERSE_RELATION["配偶"], "配偶")
        self.assertEqual(INVERSE_RELATION["朋友"], "朋友")
        self.assertEqual(INVERSE_RELATION["老师"], "学生")

    def test_role_options_cover_family_and_social(self):
        for expected in ("本人", "父亲", "母亲", "配偶", "孩子", "朋友", "同事", "老师", "无法判断"):
            self.assertIn(expected, ROLE_OPTIONS)

    def test_every_person_gets_unknown_fallback(self):
        candidates = [{"role": "母亲", "confidence": 0.7, "reason": "照顾"}]
        result = ensure_unknown_fallback(candidates)
        roles = [c["role"] for c in result]
        self.assertIn("无法判断", roles)
        self.assertGreaterEqual(len(result), 2)

    def test_same_person_cannot_be_parent_and_child(self):
        relationships = [
            {"subject_ref": "P01", "predicate": "父亲", "object_ref": "P02",
             "inverse_predicate": "孩子", "confidence": 0.7},
            {"subject_ref": "P01", "predicate": "孩子", "object_ref": "P02",
             "inverse_predicate": "父母", "confidence": 0.6},
        ]
        violations = find_graph_violations(relationships)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["kind"], "parent_and_child")

    def test_symmetric_relationship_normalized(self):
        rel = normalize_relationship({
            "subject_ref": "P02", "predicate": "朋友", "object_ref": "P01",
            "inverse_predicate": "朋友",
        })
        self.assertEqual(rel["subject_ref"], "P01")
        self.assertEqual(rel["object_ref"], "P02")

    def test_weak_evidence_relationship_downgraded(self):
        strong = {
            "subject_ref": "P01", "predicate": "母亲", "object_ref": "P02",
            "inverse_predicate": "孩子", "confidence": 0.7,
            "evidence_event_ids": ["e1", "e2"], "evidence_moment_ids": [],
        }
        weak = {
            "subject_ref": "P01", "predicate": "同事", "object_ref": "P03",
            "inverse_predicate": "同事", "confidence": 0.5,
            "evidence_event_ids": ["e1"], "evidence_moment_ids": [],
        }
        result = apply_relationship_threshold([strong, weak])
        self.assertEqual(result[0]["predicate"], "母亲")
        self.assertEqual(result[1]["predicate"], "无法判断")

    def test_sanitize_reason_removes_age_gender_generation(self):
        cleaned = sanitize_reason("多次照顾孩子，看起来像四十岁男性长辈")
        self.assertNotIn("四十岁", cleaned)
        self.assertNotIn("男性", cleaned)
        self.assertNotIn("长辈", cleaned)

    def test_normalize_graph_strips_model_attributions(self):
        parsed = {
            "album_owner_candidates": [{
                "person_ref": "P01", "confidence": 0.71,
                "reason": "长期贯穿多类记录，像三十岁男性",
            }],
            "roles": [{
                "person_ref": "P02", "relative_to": "P01",
                "candidates": [
                    {"role": "母亲", "confidence": 0.72, "reason": "多次照顾孩子，像是女性长辈"},
                    {"role": "无法判断", "confidence": 0.10, "reason": ""},
                ],
            }],
            "relationships": [{
                "subject_ref": "P02", "predicate": "母亲", "object_ref": "P01",
                "inverse_predicate": "孩子", "confidence": 0.68,
                "reason": "跨多个事件反复出现照顾互动",
            }],
        }
        result = normalize_person_graph(parsed, people=["P01", "P02"])
        self.assertNotIn("男性", result["album_owner_candidates"][0]["reason"])
        self.assertNotIn("女性", result["roles"][0]["candidates"][0]["reason"])
        self.assertEqual(result["roles"][0]["candidates"][0]["role"], "母亲")
        self.assertEqual(result["relationships"][0]["inverse_predicate"], "孩子")


class GraphHypothesisIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_hypotheses_do_not_touch_formal_tables(self):
        person_a = self.store.create_entity("甲", "person", "confirmed", family_role="父亲", scope_id="album-a")
        person_b = self.store.create_entity("乙", "person", "confirmed", scope_id="album-a")
        self.store.create_relationship(person_a["id"], "父亲", person_b["id"], confidence=0.9, status="active")
        before = (
            self.store.count("relationships"),
            self.store.count("semantic_claims"),
            self.store.get_entity(person_a["id"])["family_role"],
        )
        run = self.store.create_person_insight_run("album-a", {"max_core_people": 10})
        self.store.replace_role_hypotheses("album-a", run["id"], [{
            "person_id": person_a["id"], "role": "父亲", "rank": 1, "confidence": 0.9,
            "relative_to_person_id": person_b["id"], "prompt_version": "person-graph-v1",
        }])
        self.store.replace_relationship_hypotheses("album-a", run["id"], [{
            "subject_person_id": person_a["id"], "predicate": "父亲",
            "object_person_id": person_b["id"], "inverse_predicate": "孩子",
            "confidence": 0.7, "prompt_version": "person-graph-v1",
        }])
        after = (
            self.store.count("relationships"),
            self.store.count("semantic_claims"),
            self.store.get_entity(person_a["id"])["family_role"],
        )
        self.assertEqual(after, before)


class InferPersonGraphContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image = f"{self.temp_dir.name}/p.jpg"
        Image.new("RGB", (64, 64), "white").save(self.image)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_infer_graph_uses_verify_role_and_multimodal_images(self):
        client = GammaClient(backend="vllm")
        captured = {}

        def fake_chat(prompt, images=None, vision_options=None, json_mode=True, role=None):
            captured["role"] = role
            captured["images"] = images
            return json.dumps({
                "album_owner_candidates": [{"person_ref": "P01", "confidence": 0.71, "reason": "长期贯穿"}],
                "roles": [{"person_ref": "P02", "relative_to": "P01", "candidates": [
                    {"role": "母亲", "confidence": 0.72, "reason": "多次照顾"},
                    {"role": "无法判断", "confidence": 0.1, "reason": ""},
                ]}],
                "relationships": [],
            })

        with patch.object(client, "chat", side_effect=fake_chat):
            result = client.infer_person_graph(
                [self.image], {"people": ["P01", "P02"], "events": [], "cooccurrence": {}}
            )
        self.assertEqual(captured["role"], "verify")
        self.assertGreaterEqual(len(captured["images"]), 1)
        self.assertEqual(result["album_owner_candidates"][0]["person_ref"], "P01")
        self.assertEqual(result["roles"][0]["candidates"][0]["role"], "母亲")


if __name__ == "__main__":
    unittest.main()
