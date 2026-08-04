import unittest

from backend.agent_contracts import (
    build_evidence_bundle,
    build_text_segments,
    claim_evidence_index,
    extract_claims,
    merge_claim_candidates,
    repair_answer,
    resolve_memory_intensity,
    verify_claims,
)


class AgentContractV2Tests(unittest.TestCase):
    def test_claim_surface_uses_claim_ids_instead_of_cross_language_offsets(self):
        text = "从记录看，明哥参加过展览。"
        claims = [{"claim_id": "claim_1", "start": 5, "end": len(text), "text": "明哥参加过展览。"}]
        bundles = [{"claim_id": "claim_1", "canonical_evidence": [{"evidence_id": "event_1"}]}]
        verifications = [{"claim_id": "claim_1", "status": "reasonable_summary"}]

        segments = build_text_segments(text, claims, verifications)
        index = claim_evidence_index(claims, bundles, verifications)

        self.assertEqual([item["type"] for item in segments], ["text", "claim"])
        self.assertEqual(segments[-1]["claim_id"], "claim_1")
        self.assertEqual(index["claim_1"]["evidence_ids"], ["event_1"])

    def test_extractor_finds_writer_omitted_family_claim(self):
        text = "明哥经常参加展览。他应该很喜欢艺术。"
        writer_candidates = [{
            "claim_id": "writer_claim_1",
            "start": 0,
            "end": 11,
            "text": "明哥经常参加展览。",
            "intended_type": "derived_pattern",
            "candidate_evidence_ids": ["event_1", "event_2"],
        }]

        extraction = extract_claims(text)
        merged = merge_claim_candidates(text, writer_candidates)

        self.assertEqual([item["text"] for item in extraction["claims"]], [
            "明哥经常参加展览。", "他应该很喜欢艺术。",
        ])
        self.assertEqual(len(merged["claims"]), 2)
        self.assertEqual(merged["claims"][1]["text"], "他应该很喜欢艺术。")
        self.assertEqual(merged["claims"][1]["candidate_evidence_ids"], [])

    def test_extractor_checks_uncertainty_and_follow_up_fact(self):
        result = merge_claim_candidates(
            "目前关于他的性格，记录还不足以判断。",
            [],
            follow_up_text="这次是在展览馆，要不要看看照片？",
        )

        texts = [item["text"] for item in result["claims"]]
        self.assertIn("目前关于他的性格，记录还不足以判断。", texts)
        self.assertIn("这次是在展览馆，", texts)
        self.assertNotIn("要不要看看照片？", texts)
        self.assertEqual(result["uncovered_spans"], [])

    def test_evidence_bundle_requires_canonical_source(self):
        scene_only = build_evidence_bundle(
            {"claim_id": "claim_1", "candidate_evidence_ids": ["scene_1"]},
            [{"kind": "scene", "id": "scene_1", "narrative": "明哥经常看展"}],
            derived_context=[{"scene_id": "scene_1", "text": "明哥经常看展", "is_canonical": False}],
            scope_id="home-default", viewer_id="viewer-1",
        )
        anchored = build_evidence_bundle(
            {"claim_id": "claim_1", "candidate_evidence_ids": ["event_1", "obs_1"]},
            [
                {"kind": "event", "id": "event_1", "summary": "明哥在展览馆参观", "subject_ids": ["person-ming"], "scope_id": "home-default"},
                {"kind": "observation", "id": "obs_1", "caption": "明哥在照片中拍照", "person_id": "person-ming", "scope_id": "home-default"},
            ],
            derived_context=[{"scene_id": "scene_1", "text": "明哥经常看展", "is_canonical": False}],
            scope_id="home-default", viewer_id="viewer-1",
        )

        self.assertEqual(scene_only["canonical_evidence"], [])
        self.assertEqual(scene_only["derived_context"][0]["is_canonical"], False)
        self.assertEqual([item["evidence_id"] for item in anchored["canonical_evidence"]], ["event_1", "obs_1"])
        self.assertTrue(all(item["source_text"] for item in anchored["canonical_evidence"]))
        self.assertTrue(all(item["scope_id"] == "home-default" for item in anchored["canonical_evidence"]))

    def test_verifier_rejects_claim_without_explicit_canonical_anchor(self):
        bundle = build_evidence_bundle(
            {"claim_id": "claim_1", "candidate_evidence_ids": []},
            [{"kind": "event", "id": "event_1", "summary": "明哥参观展览", "scope_id": "home-default"}],
            derived_context=[{"scene_id": "scene_1", "text": "明哥经常看展", "is_canonical": False}],
            scope_id="home-default", viewer_id="viewer-1",
        )

        result = verify_claims(
            [{"claim_id": "claim_1", "text": "明哥很喜欢艺术", "claim_kind": "family_inference"}],
            [bundle], scope_id="home-default", viewer_id="viewer-1",
        )

        self.assertEqual(result[0]["status"], "unsupported")
        self.assertEqual(result[0]["supported_evidence_ids"], [])

    def test_repairer_only_replaces_one_failed_claim_span(self):
        text = "明哥经常参加展览。他应该很喜欢艺术。"
        claims = merge_claim_candidates(text, [{
            "claim_id": "writer_claim_1",
            "text": "明哥经常参加展览。",
            "candidate_evidence_ids": ["event_1"],
        }])["claims"]
        verifications = [
            {"claim_id": claims[0]["claim_id"], "status": "reasonable_summary"},
            {"claim_id": claims[1]["claim_id"], "status": "unsupported"},
        ]

        repaired = repair_answer(text, claims, verifications)

        self.assertEqual(repaired["repair_count"], 1)
        self.assertEqual(repaired["repaired_claim_ids"], [claims[1]["claim_id"]])
        self.assertIn("明哥经常参加展览。", repaired["text"])
        self.assertNotIn("他应该很喜欢艺术。", repaired["text"])

    def test_memory_intensity_distinguishes_probe_from_concrete_memory(self):
        self.assertEqual(resolve_memory_intensity("chat", proactive_enabled=False), "none")
        self.assertEqual(resolve_memory_intensity("chat", proactive_enabled=True), "probe")
        self.assertEqual(resolve_memory_intensity("memory", proactive_enabled=True), "targeted")
        self.assertEqual(resolve_memory_intensity("feedback", proactive_enabled=True), "forensic")


if __name__ == "__main__":
    unittest.main()
