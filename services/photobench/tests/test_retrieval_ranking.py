import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "backend" / "benchmark_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_ranking", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RankingMetricTests(unittest.TestCase):
    def test_multi_positive_reports_recall_and_hit_separately(self):
        data = MODULE._ranking_metrics(["x", "a", "y", "b", "z", "c"], {"a", "b", "c"})

        # R@K is the share of the three positives inside the window, so the
        # rank-1 slot holding a non-positive costs two thirds of it.
        self.assertEqual(data["r_at_1"], 0.0)
        self.assertEqual(data["r_at_5"], 0.6667)
        self.assertEqual(data["r_at_10"], 1.0)
        # Hit@K only asks whether any positive made it.  At K=5 it reads 1.0
        # while recall reads 0.6667, which is why both are reported.
        self.assertEqual([data["hit_at_1"], data["hit_at_5"], data["hit_at_10"]], [0.0, 1.0, 1.0])
        self.assertEqual(data["mrr"], 0.5)
        self.assertEqual(data["average_precision"], 0.5)
        self.assertEqual(data["p_at_5"], 0.4)

    def test_single_positive_makes_recall_and_hit_agree(self):
        data = MODULE._ranking_metrics(["a", "x", "y"], {"a"})

        self.assertEqual(data["r_at_1"], 1.0)
        self.assertEqual(data["hit_at_1"], 1.0)
        self.assertEqual(data["mrr"], 1.0)
        self.assertEqual(data["average_precision"], 1.0)
        self.assertEqual(data["p_at_5"], 0.3333)

    def test_no_hit_scores_zero_without_going_negative(self):
        data = MODULE._ranking_metrics(["x", "y", "z"], {"a"})

        self.assertEqual([data["r_at_1"], data["r_at_5"], data["r_at_10"]], [0.0, 0.0, 0.0])
        self.assertEqual([data["hit_at_1"], data["hit_at_5"], data["hit_at_10"]], [0.0, 0.0, 0.0])
        self.assertEqual(data["mrr"], 0.0)
        self.assertEqual(data["average_precision"], 0.0)

    def test_positives_beyond_the_candidate_list_keep_their_honest_share(self):
        # Six positives, two candidates: the question still reports the share it
        # actually found.  The shortfall is counted in the run summary instead of
        # being folded into this number.
        data = MODULE._ranking_metrics(["a", "b"], {"a", "b", "c", "d", "e", "f"})

        self.assertEqual(data["r_at_5"], 0.3333)
        self.assertEqual(data["mrr"], 1.0)
        self.assertEqual(data["p_at_5"], 1.0)
        self.assertEqual(data["ranked_count"], 2)
        self.assertEqual(data["gt_count"], 6)

    def test_empty_candidate_list_scores_zero(self):
        data = MODULE._ranking_metrics([], {"a", "b"})

        self.assertEqual(data["r_at_5"], 0.0)
        self.assertEqual(data["p_at_5"], 0.0)
        self.assertEqual(data["ranked_count"], 0)

    def test_question_without_ground_truth_is_not_scored(self):
        self.assertEqual(MODULE._ranking_metrics(["a", "b"], set()), {})


class MeanRankingMetricTests(unittest.TestCase):
    @staticmethod
    def _perfect(count):
        """A question with `count` positives, all returned in rank order."""
        gt = {f"g{index}" for index in range(count)}
        return MODULE._ranking_metrics([f"g{index}" for index in range(count)], gt)

    def test_recall_at_k_skips_questions_whose_positives_exceed_k(self):
        # 25 positives cannot reach R@5 = 1 under any ranking, so it must not be
        # averaged in with a single-positive question at K=5.
        items = [{"image_ranking": self._perfect(25)}, {"image_ranking": self._perfect(1)}]

        data = MODULE._mean_ranking_metrics(items)

        self.assertEqual(data["r_at_5"], 1.0)
        self.assertEqual(data["r_at_5_question_count"], 1)
        self.assertEqual(data["r_at_1_question_count"], 1)
        self.assertEqual(data["r_at_10_question_count"], 1)
        self.assertEqual(data["question_count"], 2)
        self.assertEqual(data["multi_positive_count"], 1)

    def test_a_question_enters_the_recall_population_at_the_k_it_fits(self):
        # Seven positives: too many for R@5, eligible for R@10.
        items = [{"image_ranking": self._perfect(7)}]

        data = MODULE._mean_ranking_metrics(items)

        self.assertEqual(data["r_at_5_question_count"], 0)
        self.assertIsNone(data["r_at_5"])
        self.assertEqual(data["r_at_10_question_count"], 1)
        self.assertEqual(data["r_at_10"], 1.0)

    def test_map_and_mrr_keep_every_question_including_large_positive_sets(self):
        items = [{"image_ranking": self._perfect(25)}, {"image_ranking": self._perfect(1)}]

        data = MODULE._mean_ranking_metrics(items)

        # A perfect ranking yields AP = 1 for both, so the multi-positive
        # question is still represented where the metric is defined for it.
        self.assertEqual(data["average_precision"], 1.0)
        self.assertEqual(data["mrr"], 1.0)

    def test_counts_report_the_scored_shape(self):
        items = [
            {"image_ranking": self._perfect(3)},
            {"image_ranking": self._perfect(1)},
            {"no_ranking": True},
        ]

        data = MODULE._mean_ranking_metrics(items)

        self.assertEqual(data["question_count"], 2)
        self.assertEqual(data["multi_positive_count"], 1)
        # Both returned fewer than ten candidates, so both count as short.
        self.assertEqual(data["short_candidate_count"], 2)

    def test_counts_report_the_candidate_coverage_shortfall(self):
        two_ok = self._perfect(2)                                    # 2 candidates, 2 positives
        one_short = MODULE._ranking_metrics(["a"], {"a", "b"})       # 1 candidate, 2 positives
        none_at_all = MODULE._ranking_metrics([], {"a"})             # tool returned nothing
        items = [{"image_ranking": two_ok}, {"image_ranking": one_short},
                 {"image_ranking": none_at_all}]

        data = MODULE._mean_ranking_metrics(items)

        self.assertEqual(data["question_count"], 3)
        self.assertEqual(data["coverage_short_count"], 2)
        self.assertEqual(data["no_candidate_count"], 1)

    def test_no_scored_rows_yields_no_metrics(self):
        self.assertEqual(MODULE._mean_ranking_metrics([{"no_ranking": True}]), {})
        self.assertEqual(MODULE._mean_ranking_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
