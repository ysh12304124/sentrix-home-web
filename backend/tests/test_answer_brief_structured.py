"""TFPE v2: build_structured_brief maps exact SQL facts into an AnswerBrief."""

import unittest

from backend.answer_brief import build_structured_brief
from backend.query_contracts import QueryParseDraft, QuerySpec
from backend.structured_memory import StructuredResult


def _draft(answer_type):
    return QueryParseDraft(answer_type=answer_type, structured={})


def _spec():
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general", constraints=[])


class StructuredBriefTests(unittest.TestCase):
    def test_count_brief(self):
        brief = build_structured_brief("去年拍了多少张照片", _spec(), _draft("count"),
                                       StructuredResult("count", 2, total=2), "structured_fact")
        self.assertEqual(brief.response_mode, "structured_fact")
        self.assertEqual(brief.user_goal, "answer_fact")
        self.assertEqual(len(brief.visible_assets), 0)
        self.assertFalse(brief.presentation.show_images)
        self.assertEqual(brief.facts[0].text, "符合条件的记录共 2 条")
        self.assertEqual(brief.facts[0].certainty, "confirmed")

    def test_exists_true_false(self):
        yes = build_structured_brief("2024年5月拍过照片吗", _spec(), _draft("exists"),
                                     StructuredResult("exists", True, total=1), "structured_fact")
        self.assertEqual(yes.facts[0].text, "存在符合条件的记录")
        no = build_structured_brief("2030年拍过照片吗", _spec(), _draft("exists"),
                                    StructuredResult("exists", False, total=0), "structured_fact")
        self.assertEqual(no.facts[0].text, "没有符合条件的记录")

    def test_last_occurrence_brief(self):
        brief = build_structured_brief("明哥最后一次出现是什么时候", _spec(), _draft("last_occurrence"),
                                       StructuredResult("last_occurrence", "2024-10-05T08:00:00", total=1),
                                       "entity_fact")
        self.assertEqual(brief.facts[0].text, "最晚的时间是 2024-10-05T08:00:00")

    def test_grouped_list_brief(self):
        result = StructuredResult("grouped_list", [{"group": "2024-10", "count": 2}],
                                  rows=[{"group": "2024-10", "count": 2}], total=2)
        brief = build_structured_brief("去年有哪些月份有照片", _spec(), _draft("grouped_list"),
                                       result, "aggregation")
        self.assertEqual(brief.response_mode, "aggregate_answer")
        self.assertEqual(brief.user_goal, "aggregate_memory")
        self.assertEqual(brief.facts[0].text, "2024年10月有 2 条")

    def test_writer_payload_hides_internals(self):
        brief = build_structured_brief("去年拍了多少张照片", _spec(), _draft("count"),
                                       StructuredResult("count", 2, total=2), "structured_fact")
        payload = brief.writer_payload()
        self.assertNotIn("structured_result", payload)
        self.assertNotIn("asset_id", str(payload))


if __name__ == "__main__":
    unittest.main()
