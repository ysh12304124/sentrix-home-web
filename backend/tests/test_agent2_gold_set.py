import json
import unittest
from pathlib import Path

from backend.agent_runtime.task_state import (
    EVIDENCE_TYPES,
    TASK_TERMINAL_STATES,
    UNMET_REASONS,
    EvidenceRequirement,
    TaskDeclaration,
    TaskState,
)


class Agent2GoldSetTests(unittest.TestCase):
    def setUp(self):
        # Resolve relative to repo root
        root = Path(__file__).resolve().parent.parent.parent
        self.gold_path = root / "scripts" / "benchmarks" / "agent2_gold_set.json"
        self.assertTrue(self.gold_path.exists(), f"agent2_gold_set.json must exist at {self.gold_path}")
        with open(self.gold_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_gold_set_schema_and_vocabulary_compliance(self):
        cases = self.data.get("cases", [])
        self.assertGreaterEqual(len(cases), 5, "Requirement Gold Set must contain representative cases")

        for case in cases:
            case_id = case["case_id"]
            self.assertTrue(case_id, "case_id is required")
            self.assertTrue(case.get("goal"), f"goal required for {case_id}")
            self.assertTrue(case.get("scope_id"), f"scope_id required for {case_id}")

            # Validate terminal outcome compliance
            term = case.get("expected_terminal_outcome")
            self.assertIn(term, TASK_TERMINAL_STATES, f"Invalid terminal outcome {term} in {case_id}")

            # Validate accepted evidence types
            for ev in case.get("accepted_evidence_types", []):
                self.assertIn(ev, EVIDENCE_TYPES, f"Invalid evidence type {ev} in {case_id}")

            # Validate unmet reasons
            for reason in case.get("possible_unmet_reasons", []):
                self.assertIn(reason, UNMET_REASONS, f"Invalid unmet reason {reason} in {case_id}")

            # Build TaskState and verify schema
            requirements = [
                EvidenceRequirement(
                    id=req["id"],
                    evidence_type=req["evidence_type"],
                    description=req.get("description", ""),
                    parent_id=req.get("parent_id", ""),
                )
                for req in case["minimal_requirements"]
            ]
            decl = TaskDeclaration(
                goal=case["goal"],
                scope_id=case["scope_id"],
                requirements=tuple(requirements),
            )
            state = TaskState.from_declaration(decl)
            state.set_terminal_outcome(term)
            self.assertEqual(len(state.requirements), len(requirements))


if __name__ == "__main__":
    unittest.main()
