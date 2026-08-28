#!/usr/bin/env python3
"""Regression coverage for QA tag facets and filtering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.benchmark_orchestrator import OrchestratorRepository


class QaTagFilteringTest(unittest.TestCase):
    def test_run_items_expose_facets_and_filter_by_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = OrchestratorRepository(Path(temporary))
            repo.runs["tag-test"] = {
                "run_id": "tag-test",
                "status": "completed",
                "items": [
                    {"qa_id": "tag-q1", "question": "q1", "tags": ["media:video", "difficulty:easy"]},
                    {"qa_id": "tag-q2", "question": "q2", "tags": ["media:image", "difficulty:hard"]},
                    {
                        "qa_id": "tag-q3",
                        "question": "q3",
                        "task_type": "T2_fact_qa",
                        "question_type": "event_memory_qa",
                        "angle": "place",
                        "difficulty": "medium",
                        "answerability": "answerable",
                        "expected_action": "answer",
                    },
                ],
            }

            page = repo.get_run_items("tag-test", tag="media:video")

            self.assertEqual(page["total"], 1)
            self.assertEqual(page["items"][0]["qa_id"], "tag-q1")
            self.assertEqual(page["items"][0]["tags"], ["media:video", "difficulty:easy"])
            self.assertEqual(page["filters"]["tag"], "media:video")
            self.assertEqual(
                page["facets"]["tags"],
                [
                    "action:answer",
                    "angle:place",
                    "answerability:answerable",
                    "difficulty:easy",
                    "difficulty:hard",
                    "difficulty:medium",
                    "media:image",
                    "media:video",
                    "question:event_memory_qa",
                    "task:T2_fact_qa",
                ],
            )

            derived = repo.get_run_items("tag-test", tag="question:event_memory_qa")
            self.assertEqual(derived["total"], 1)
            self.assertEqual(derived["items"][0]["qa_id"], "tag-q3")
            self.assertIn("difficulty:medium", derived["items"][0]["tags"])


if __name__ == "__main__":
    unittest.main()
