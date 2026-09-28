import unittest

from backend.agent_runtime.final_writer import (
    build_final_context,
    evidence_answer_problems,
)


class FinalWriterEvidenceTests(unittest.TestCase):
    def test_identity_evidence_is_carried_into_context_and_refusal_is_flagged(self):
        task = {
            "tool_results": [{
                "tool": "inspect_photo",
                "asset_id": "asset_052",
                "inspect_text": "两位成年人站在桥边。",
                "photo_identities": [{"person_name": "王建国", "family_role": "父亲", "asset_id": "asset_052"}],
            }],
        }
        context = build_final_context("这张照片里是谁？", task)
        self.assertIn("王建国", context["confirmed_people"])
        self.assertIn("direct_evidence_refused", evidence_answer_problems("这张照片里是谁？", "现有照片里看不出来。", context))
        self.assertTrue(any("王建国" in str(f.get("value")) for f in context["facts"]))

    def test_date_fact_cannot_be_answered_by_generic_candidate_sentence(self):
        task = {"fact_operation": "date", "fact_value": "2017年11月5日", "tool_results": []}
        context = build_final_context("这是什么时候拍的？", task)
        self.assertIn("date_fact_missing", evidence_answer_problems("这是什么时候拍的？", "找到了一些照片。", context))

    def test_multi_person_inspection_preserves_unconfirmed_companion(self):
        task = {
            "tool_results": [{
                "tool": "inspect_photo",
                "asset_id": "asset_097",
                "inspect_text": "三人合影。",
                "photo_identities": [
                    {"person_name": "我", "asset_id": "asset_097"},
                    {"person_name": "明明", "asset_id": "asset_097"},
                ],
                "unconfirmed_people_count": 1,
            }],
        }
        context = build_final_context("这张合影里有哪些人？", task)
        self.assertIn("我", context["confirmed_people"])
        self.assertIn("明明", context["confirmed_people"])
        self.assertTrue(any("未确认身份" in str(f.get("value")) for f in context["facts"]))

    def test_inspection_scopes_identities_and_counts_unnamed_people(self):
        context = build_final_context(
            "这张三人合影里都有谁？",
            {"tool_results": [
                {"tool": "search_memories", "preview": [{"people": [
                    {"name": "乐乐", "identity_status": "confirmed"},
                ]}]},
                {"tool": "inspect_photo", "inspect_text": "照片中包含一个小孩和两个成年人。",
                 "photo_identities": [
                     {"person_name": "我", "identity_status": "confirmed"},
                     {"person_name": "明明", "identity_status": "confirmed"},
                 ]},
            ]},
        )
        self.assertNotIn("乐乐", context["confirmed_people"])
        self.assertTrue(any("未确认身份" in str(f.get("value")) for f in context["facts"]))

    def test_search_preview_people_are_answer_facts(self):
        context = build_final_context(
            "那次合影里都有谁？",
            {"tool_results": [{"tool": "search_memories", "preview": [
                {"handle": "photo_1", "people": [
                    {"name": "王建国", "family_role": "父亲", "identity_status": "confirmed"},
                    {"name": "张晓莉", "family_role": "母亲", "identity_status": "confirmed"},
                ]}
            ]}], "result_preview": ["photo_1"]},
        )
        self.assertEqual(set(context["confirmed_people"]), {"王建国", "张晓莉"})

    def test_person_answer_must_include_all_confirmed_people(self):
        context = build_final_context(
            "那次合影里都有谁？",
            {"tool_results": [{"tool": "search_memories", "preview": [
                {"handle": "photo_1", "people": [
                    {"name": "王建国", "identity_status": "confirmed"},
                    {"name": "张晓莉", "identity_status": "confirmed"},
                ]}
            ]}]},
        )
        self.assertIn("person_fact_missing", evidence_answer_problems(
            "那次合影里都有谁？", "里面有王建国。", context))

    def test_wrong_year_is_rejected_by_evidence_check(self):
        context = {"facts": [{"source": "facts", "value": "相关时间是 2017-11-05。"}], "facts_confirmed": True}
        problems = evidence_answer_problems("是哪一年？", "是在2018年。", context)
        self.assertIn("date_fact_conflict", problems)


if __name__ == "__main__":
    unittest.main()
