import unittest

from backend.agent_runtime.tool_registry import ToolSpec
from backend.agent_runtime.tool_registry import get_tool
from backend.agent_runtime.tools import register_tools


def _executor(arguments, *, context=None):
    return {"summary": "ok"}


class ToolCapabilityContractTests(unittest.TestCase):
    def test_registered_tools_expose_non_overlapping_evidence_capabilities(self):
        register_tools()

        self.assertTrue(get_tool("read_photo_text").can_satisfy("visible_text"))
        self.assertFalse(get_tool("read_photo_text").can_satisfy("visual_observation"))
        self.assertTrue(get_tool("inspect_photo").can_satisfy("visual_observation"))
        self.assertFalse(get_tool("inspect_photo").can_satisfy("visible_text"))
        self.assertTrue(get_tool("search_memories").can_satisfy("memory_asset"))

    def test_capability_declares_evidence_it_can_and_cannot_establish(self):
        spec = ToolSpec(
            name="read_photo_text",
            description="Read visible text.",
            input_schema={"asset_handle": ""},
            executor=_executor,
            produces_evidence=("visible_text",),
            cannot_establish=("confirmed_identity", "visual_observation"),
            budget_unit="image",
        )

        self.assertTrue(spec.can_satisfy("visible_text"))
        self.assertFalse(spec.can_satisfy("confirmed_identity"))
        self.assertIn("visible_text", spec.as_contract()["produces_evidence"])

    def test_rejects_overlapping_or_unknown_evidence_contracts(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            ToolSpec(
                name="bad",
                description="bad",
                input_schema={},
                executor=_executor,
                produces_evidence=("visible_text",),
                cannot_establish=("visible_text",),
            )

        with self.assertRaisesRegex(ValueError, "unsupported evidence type"):
            ToolSpec(
                name="bad",
                description="bad",
                input_schema={},
                executor=_executor,
                produces_evidence=("imagined_fact",),
            )


if __name__ == "__main__":
    unittest.main()
