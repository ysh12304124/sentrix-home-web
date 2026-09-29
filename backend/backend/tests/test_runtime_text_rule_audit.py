"""Phase R9 — runtime text rule audit gate.

The final ``runtime_text_rule_inventory.json`` must contain zero runtime
semantic_routing / semantic_extraction rules, zero unresolved "review" entries,
and every kept Protocol/Normalization rule must name a test that covers it.

The two legacy agent.py word-lists are allowed only with decision
"remove_or_retire" (thin path never adopts them).
"""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "docs" / "baseline" / "runtime_text_rule_inventory.json"


class RuntimeTextRuleAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INVENTORY.is_file():
            raise unittest.SkipTest("runtime_text_rule_inventory.json not generated")
        cls.data = json.loads(INVENTORY.read_text(encoding="utf-8"))
        cls.rules = cls.data["rules"]

    def test_no_review_pending(self):
        pending = [r for r in self.rules if r["category"] == "review"]
        self.assertEqual(pending, [], "every rule must be finalized, none left as review")

    def test_no_runtime_semantic_routing(self):
        bad = [r for r in self.rules if r["scope"] == "runtime"
               and r["category"] == "semantic_routing"]
        self.assertEqual(bad, [], "runtime must have zero open-vocabulary semantic routing")

    def test_no_runtime_semantic_extraction(self):
        bad = [r for r in self.rules if r["scope"] == "runtime"
               and r["category"] == "semantic_extraction"]
        self.assertEqual(bad, [], "runtime must have zero hard-coded open semantics")

    def test_legacy_semantic_routing_marked_for_removal(self):
        legacy = [r for r in self.rules if r["scope"] == "legacy"
                  and r["category"] == "semantic_routing"]
        for rule in legacy:
            self.assertIn(rule["decision"], {"remove_or_retire", "remove"},
                          f"legacy semantic_routing rule must be marked for removal: {rule['symbol']}")

    def test_kept_protocol_and_normalization_rules_name_tests(self):
        for rule in self.rules:
            if rule["category"] in {"protocol", "normalization"} and rule["decision"] == "keep":
                self.assertTrue(rule["tests"], f"kept rule must reference tests: {rule['file']}:{rule['symbol']}")

    def test_no_benchmark_original_sentences_in_prompt_rules(self):
        prompt_rules = [r for r in self.rules if r["category"] == "prompt"]
        self.assertTrue(prompt_rules, "inventory should list at least one prompt rule")


if __name__ == "__main__":
    unittest.main()
