"""TFPE v2: RetrievalStrategy planner — model hint + deterministic capability gate."""

import unittest

from backend.query_contracts import Constraint, HARD, SEMANTIC, QueryParseDraft, QuerySpec
from backend.retrieval_strategy import plan_retrieval_strategy


def _spec(*constraints, answer_target="general"):
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", answer_target,
                     constraints=list(constraints))


def _draft(answer_type="asset_set", strategy_hint="", structured=None):
    return QueryParseDraft(answer_type=answer_type, strategy_hint=strategy_hint,
                           structured=structured or {})


class RetrievalStrategyTests(unittest.TestCase):
    def test_count_with_time_is_structured(self):
        draft = _draft(answer_type="count", strategy_hint="aggregation",
                       structured={"time_range": {"start": "2024-01-01", "end": "2025-01-01"}})
        strategy = plan_retrieval_strategy(draft, _spec(Constraint("time", "去年", HARD, "asset_metadata")))
        self.assertEqual(strategy.strategy, "structured_fact")
        self.assertIn("visual_ann", strategy.skipped_channels)

    def test_visual_hint_overridden_when_pure_structured(self):
        # The model said visual, but a count with no picture-only dimension must
        # be answered exactly from the DB (zero-tolerance: 结构化可精确回答却用 ANN 估算=0).
        draft = _draft(answer_type="count", strategy_hint="visual_semantic")
        strategy = plan_retrieval_strategy(draft, _spec())
        self.assertEqual(strategy.strategy, "structured_fact")
        self.assertIn("model hint overridden", strategy.reason)

    def test_asset_set_defaults_to_retrieval(self):
        draft = _draft(answer_type="asset_set")
        strategy = plan_retrieval_strategy(draft, _spec())
        self.assertEqual(strategy.strategy, "hybrid")

    def test_count_with_clothing_downgrades_to_hybrid(self):
        # Picture-only dimension present: the structured executor cannot answer
        # clothing exactly, so the normal retrieval path owns the turn.
        draft = _draft(answer_type="count", strategy_hint="structured_fact")
        strategy = plan_retrieval_strategy(
            draft, _spec(Constraint("clothing", "黄色外套", SEMANTIC, "direct_or_possible")))
        self.assertEqual(strategy.strategy, "hybrid")

    def test_person_last_occurrence_is_entity_fact(self):
        spec = _spec(Constraint("person", "明哥", HARD, "confirmed_bridge"))
        spec.entity_ids = ["entity-1"]
        draft = _draft(answer_type="last_occurrence")
        strategy = plan_retrieval_strategy(draft, spec)
        self.assertEqual(strategy.strategy, "entity_fact")

    def test_model_silent_grouped_list_defaults_aggregation(self):
        draft = _draft(answer_type="grouped_list")
        strategy = plan_retrieval_strategy(draft, _spec())
        self.assertEqual(strategy.strategy, "aggregation")

    def test_model_silent_asset_set_defaults_hybrid(self):
        draft = _draft(answer_type="asset_set", strategy_hint="")
        strategy = plan_retrieval_strategy(draft, _spec())
        self.assertEqual(strategy.strategy, "hybrid")


if __name__ == "__main__":
    unittest.main()
