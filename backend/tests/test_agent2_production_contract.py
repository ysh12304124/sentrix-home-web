import sqlite3
import unittest
from types import SimpleNamespace

from backend.agent_runtime.evidence_contract import PUBLIC_EVIDENCE_TYPES
from backend.agent_runtime.evidence_ledger import EvidenceLedger
from backend.agent_runtime.jit_prompt import build_jit_system_prompt
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState
from backend.agent_runtime.tool_policy import ToolPolicy
from backend.agent_runtime.tool_registry import list_tools
from backend.agent_runtime.tools import _confirmed_photo_identities, register_tools
from backend.embeddings.router import EmbeddingRouter
from backend.agent_runtime.goal_planner import GoalPlanner
from backend.agent_runtime.runtime import AgentRuntime, _normalize_selected_image_handles


class Agent2ProductionContractTests(unittest.TestCase):
    def test_public_evidence_contract_is_exact(self):
        self.assertEqual(PUBLIC_EVIDENCE_TYPES, frozenset({
            "memory_asset", "location_metadata", "temporal_metadata",
            "confirmed_identity", "photo_identity", "visual_observation",
            "visible_text", "structured_fact", "user_statement",
        }))

    def test_registry_exposes_identity_and_no_internal_reference(self):
        register_tools()
        specs = {spec.name: spec for spec in list_tools()}
        self.assertIn("photo_identity", specs["inspect_photo"].produces_evidence)
        self.assertNotIn("memory_reference", {
            item for spec in specs.values() for item in spec.produces_evidence
        })

    def test_task_status_requires_all_required_evidence(self):
        state = TaskState.from_declaration(TaskDeclaration(
            goal="identity", scope_id="album1", requirements=(
                EvidenceRequirement("asset", "memory_asset"),
                EvidenceRequirement("identity", "photo_identity"),
            )))
        self.assertEqual(state.recompute_status(has_available_tools=True), "in_progress")
        state.mark_satisfied("asset", evidence_refs=("tool_1",))
        self.assertEqual(state.recompute_status(has_available_tools=True), "in_progress")
        state.mark_satisfied("identity", evidence_refs=("tool_2",))
        self.assertEqual(state.recompute_status(has_available_tools=False), "complete")

    def test_policy_rejects_visual_tool_without_current_preview(self):
        register_tools()
        spec = next(item for item in list_tools() if item.name == "inspect_photo")
        decision = ToolPolicy(scope_id="album1").execute(
            spec, {"asset_handle": "photo_1", "question": "看什么"},
            context={"task_state": {"result_preview": []}})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "asset_handle_not_in_current_preview")

    def test_selected_image_handles_are_bounded_and_must_be_current_preview(self):
        self.assertEqual(
            _normalize_selected_image_handles(
                ["photo_2", "photo_2", "photo_9", "photo_1", "photo_3", "photo_4", "photo_5"],
                ["photo_1", "photo_2", "photo_3", "photo_4", "photo_5"],
            ),
            ["photo_2", "photo_1", "photo_3", "photo_4", "photo_5"],
        )

    def test_confirmed_identity_query_is_read_only(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE face_instances (
                id TEXT, asset_id TEXT, observation_id TEXT, cluster_id TEXT,
                bbox_json TEXT, detection_confidence REAL, quality REAL
            );
            CREATE TABLE face_clusters (
                id TEXT, entity_id TEXT, status TEXT
            );
            CREATE TABLE entities (
                id TEXT, canonical_name TEXT, family_role TEXT,
                entity_type TEXT, status TEXT
            );
            CREATE TABLE entity_mentions (
                face_instance_id TEXT, entity_id TEXT, confidence REAL
            );
        """)
        conn.execute("INSERT INTO face_instances VALUES ('fi1','a1','o1','fc1','',.9,.8)")
        conn.execute("INSERT INTO face_clusters VALUES ('fc1','e1','confirmed')")
        conn.execute("INSERT INTO entities VALUES ('e1','乐乐','孩子','person','confirmed')")
        conn.execute("INSERT INTO entity_mentions VALUES ('fi1','e1',.95)")
        conn.commit()
        store = SimpleNamespace(connection=conn)
        before = conn.total_changes
        rows = _confirmed_photo_identities(store, "a1")
        self.assertEqual(rows[0]["person_name"], "乐乐")
        self.assertEqual(rows[0]["identity_status"], "confirmed")
        self.assertEqual(conn.total_changes, before)

    def test_jit_offers_search_as_identity_prerequisite(self):
        register_tools()
        state = TaskState.from_declaration(TaskDeclaration(
            goal="photo identity", scope_id="album1", requirements=(
                EvidenceRequirement("identity", "photo_identity"),
            )))
        prompt = build_jit_system_prompt(
            task_state=state, current_time_str="", tool_results=[],
            preview_handles=[], is_candidate=True)
        self.assertIn("search_memories", prompt)
        self.assertNotIn("inspect_photo: 复核", prompt)

    def test_embedding_status_reports_slots_without_throwing(self):
        router = EmbeddingRouter(visual=None, text=None)
        status = router.status()
        self.assertFalse(status["visual"]["available"])
        self.assertFalse(status["text"]["available"])

    def test_planner_normalization_deduplicates_evidence_types(self):
        payload = GoalPlanner._normalize_payload({
            "action": "declare",
            "requirements": [
                {"type": "count", "description": "数量"},
                {"type": "structured_fact", "description": "同一结构化事实"},
            ],
        }, scope_id="album1", default_goal="统计")
        self.assertEqual([r["evidence_type"] for r in payload["declaration"]["requirements"]],
                         ["structured_fact"])

    def test_authoritative_final_gate_blocks_before_task_completion(self):
        responses = iter((
            '{"action":"declare","declaration":{"goal":"看照片","scope_id":"album1",'
            '"requirements":[{"id":"visual","evidence_type":"visual_observation",'
            '"description":"照片内容"} ]}}',
            '{"action":"final","answer":"照片里有一个人","evidence_refs":[]}',
        ))
        runtime = AgentRuntime(
            chat_fn=lambda messages, **kwargs: next(responses),
            profile_name="goal_driven_candidate",
            scope_id="album1",
        )
        turn = runtime.run("照片里有几个人？")
        self.assertNotEqual(turn.status, "complete")
        self.assertIn(turn.agent2_trace["final_gate"]["status"], {"in_progress", "insufficient_evidence"})
        self.assertFalse(turn.agent2_trace["final_gate"].get("candidate_closure", False))


if __name__ == "__main__":
    unittest.main()
