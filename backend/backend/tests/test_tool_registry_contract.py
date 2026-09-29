import unittest

from backend.agent_runtime.evidence_contract import PUBLIC_EVIDENCE_TYPES
from backend.agent_runtime.tool_registry import list_tools
from backend.agent_runtime.tools import register_tools


class ToolRegistryContractTests(unittest.TestCase):
    def test_registered_tools_have_unique_complete_contracts(self):
        register_tools()
        specs = list_tools()
        self.assertEqual(len({spec.name for spec in specs}), len(specs))
        for spec in specs:
            self.assertTrue(spec.name)
            self.assertTrue(spec.description)
            self.assertIsInstance(spec.input_schema, dict)
            self.assertTrue(spec.required_inputs or spec.preconditions, spec.name)
            self.assertTrue(set(spec.produces_evidence_types) <= PUBLIC_EVIDENCE_TYPES)

    def test_identity_tool_declares_read_only_prerequisite(self):
        register_tools()
        spec = next(item for item in list_tools() if item.name == "inspect_photo")
        self.assertEqual(spec.read_write, "read")
        self.assertIn("asset_handle", spec.required_inputs)
        self.assertIn("asset_handle_in_current_preview", spec.preconditions)
        self.assertIn("memory_asset", spec.prerequisite_evidence_types)


if __name__ == "__main__":
    unittest.main()
