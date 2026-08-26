import unittest

from backend.agent_runtime.evidence_ledger import EvidenceLedger
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState
from backend.agent_runtime.tool_registry import ToolSpec
from backend.agent_runtime.runtime import record_agent2_tool_evidence


def _execute(arguments, *, context=None):
    return {}


class Agent2EvidenceRecordingTests(unittest.TestCase):
    def test_records_matching_tool_evidence_and_satisfies_requirement(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read sign", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="read_photo_text", description="ocr", input_schema={},
                        executor=_execute, produces_evidence=("visible_text",))

        recorded = record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1", input_refs=("photo_1",),
            observation={"full_text": "招牌文字", "certainty": "supported"})

        self.assertTrue(recorded)
        self.assertEqual(task.requirement("text").status, "satisfied")
        self.assertEqual(ledger.entries[0].evidence_type, "visible_text")

    def test_ignores_incompatible_capability_without_changing_requirement(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read sign", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="inspect_photo", description="visual", input_schema={},
                        executor=_execute, produces_evidence=("visual_observation",))

        self.assertFalse(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1", input_refs=("photo_1",)))
        self.assertEqual(task.requirement("text").status, "open")
        self.assertEqual(ledger.entries, [])

    def test_ocr_failure_summary_is_not_visible_text_evidence(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read sign", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="read_photo_text", description="ocr", input_schema={},
                        executor=_execute, produces_evidence=("visible_text",))

        self.assertTrue(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="ocr_failed",
            observation={"summary": "这次没能可靠读出照片里的文字。",
                         "status": "partial", "reason": "ocr_failed"}))
        self.assertEqual(task.requirement("text").coverage_status, "failed")
        self.assertEqual(ledger.entries[0].failure_reason, "ocr_failed")

    def test_metadata_place_operation_does_not_consume_temporal_attempt(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="where and when", scope_id="album1",
            requirements=(
                EvidenceRequirement(id="date", evidence_type="temporal_metadata"),
                EvidenceRequirement(id="place", evidence_type="location_metadata"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="query_memory_metadata", description="metadata", input_schema={},
                        executor=_execute,
                        produces_evidence=("structured_fact", "temporal_metadata", "location_metadata"))

        self.assertTrue(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="metadata_place",
            observation={"metadata_operation": "place", "value": "上海",
                         "source_asset_ids": ["asset_1"]}))

        self.assertEqual(task.requirement("place").status, "satisfied")
        self.assertEqual(task.requirement("date").status, "open")
        self.assertEqual(task.requirement("date").attempt_count, 0)

    def test_negative_visual_observation_is_asset_bound_contradiction(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="find a boat", scope_id="album1",
            requirements=(EvidenceRequirement(
                id="scene", evidence_type="visual_observation", description="是否有船"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="inspect_photo", description="visual", input_schema={},
                        executor=_execute, produces_evidence=("visual_observation",))

        self.assertTrue(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="inspect_1", input_refs=("photo_1",),
            observation={"observation": "这张照片中没有船。",
                         "_source_asset_id": "asset_1", "certainty": "supported"}))
        self.assertEqual(task.requirement("scene").status, "contradicted")
        self.assertEqual(ledger.entries[0].certainty, "contradicted")
        self.assertEqual(ledger.entries[0].asset_id, "asset_1")

    def test_observation_can_bind_multiple_requirements_from_one_search(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="查找有日期和地点的照片", scope_id="album1",
            requirements=(
                EvidenceRequirement(id="asset", evidence_type="memory_asset"),
                EvidenceRequirement(id="place", evidence_type="location_metadata"),
                EvidenceRequirement(id="date", evidence_type="temporal_metadata"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="search_memories", description="search", input_schema={},
                        executor=_execute, produces_evidence=(
                            "memory_asset", "location_metadata", "temporal_metadata"))

        recorded = record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1",
            input_refs=("photo_1",), observation={
                "result_set_id": "result_1",
                "total": 1,
                "asset_ids": ["asset_1"],
                "preview": [{"handle": "photo_1", "place": "秦皇岛", "captured_at": "2019-07-22"}],
                "query": "沙雕合影", "evidence_status": "validated", "evidence_asset_ids": ["asset_1"],
                "certainty": "supported",
            })

        self.assertTrue(recorded)
        self.assertEqual({entry.evidence_type for entry in ledger.entries}, {
            "memory_asset", "location_metadata", "temporal_metadata",
        })
        self.assertEqual(task.requirement("asset").status, "satisfied")
        self.assertEqual(task.requirement("place").status, "satisfied")
        self.assertEqual(task.requirement("date").status, "satisfied")
        self.assertEqual(set(ledger.entries[0].requirement_refs), {"asset"})

    def test_search_records_confirmed_people_from_preview(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="identify people in a photo", scope_id="album1",
            requirements=(EvidenceRequirement(id="identity", evidence_type="photo_identity"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="search_memories", description="search", input_schema={},
                        executor=_execute, produces_evidence=("memory_asset", "photo_identity"))

        recorded = record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1", observation={
                "result_set_id": "result_1", "total": 1, "asset_ids": ["asset_1"],
                "preview": [{"handle": "photo_1", "people": [{
                    "name": "明明", "family_role": "孩子", "identity_status": "confirmed",
                }]}], "query": "合影", "evidence_status": "validated", "evidence_asset_ids": ["asset_1"], "certainty": "supported",
            })

        self.assertTrue(recorded)
        identity = next(entry for entry in ledger.entries if entry.evidence_type == "photo_identity")
        self.assertEqual(identity.extracted_value["person_name"], "明明")
        self.assertEqual(identity.asset_id, "asset_1")

    def test_search_summary_never_satisfies_visible_text(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read full menu", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="search_memories", description="search", input_schema={},
                        executor=_execute,
                        produces_evidence=("memory_asset", "location_metadata", "temporal_metadata"))

        self.assertTrue(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="search_1",
            observation={"result_set_id": "result_1", "total": 1,
                         "asset_ids": ["asset_1"],
                         "preview": [{"handle": "photo_1", "evidence_summary": "文字：菜单价格"}],
                         "evidence_status": "validated", "evidence_asset_ids": ["asset_1"]}))
        self.assertEqual(task.requirement("text").status, "open")
        self.assertEqual(task.requirement("text").attempt_count, 0)
        self.assertFalse(any(e.evidence_type == "visible_text" for e in ledger.entries))


if __name__ == "__main__":
    unittest.main()
