"""Phase 12B-FC V2 — expected-role / model-match assertions."""

import unittest

from backend.validation.assertions import validate_turn


def _rec(role, actual="gemma4:12b", fallback=False, cache=False, call_id="c1"):
    return {"call_id": call_id, "role": role, "actual_model": actual,
            "fallback_used": fallback, "cache_hit": cache}


class ExpectedModelRoleTests(unittest.TestCase):
    def test_all_expected_roles_called(self):
        records = [_rec("parser"), _rec("answer")]
        v = validate_turn(records, ["parser", "answer"], required_model="gemma4:12b")
        self.assertTrue(v["passed"])
        self.assertTrue(v["all_expected_roles_called"])
        self.assertTrue(v["all_models_match"])
        self.assertFalse(v["degradation_used"])

    def test_missing_role_fails(self):
        records = [_rec("parser")]
        v = validate_turn(records, ["parser", "answer"], required_model="gemma4:12b")
        self.assertFalse(v["passed"])
        self.assertIn("missing_roles", str(v["issues"]))

    def test_model_mismatch_fails(self):
        records = [_rec("parser", actual="gemma-4-e2b-it+lora-v2")]
        v = validate_turn(records, ["parser"], required_model="gemma4:12b")
        self.assertFalse(v["all_models_match"])
        self.assertFalse(v["passed"])

    def test_degradation_fails(self):
        records = [_rec("parser", fallback=True)]
        v = validate_turn(records, ["parser"], required_model="gemma4:12b")
        self.assertTrue(v["degradation_used"])
        self.assertFalse(v["passed"])

    def test_parser_failed_marks_degradation(self):
        records = []
        v = validate_turn(records, ["parser"], required_model="gemma4:12b", parser_failed=True)
        self.assertTrue(v["degradation_used"])
        self.assertFalse(v["passed"])


if __name__ == "__main__":
    unittest.main()
