import tempfile

from backend.db import MemoryStore


def test_family_analysis_run_tracks_text_only_evidence_and_completion():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")

        run = store.create_family_analysis_run(
            "album-a", {"trigger_type": "ingest", "input_mode": "semantic_text_only"}
        )
        assert run["status"] == "queued"
        assert run["config"]["input_mode"] == "semantic_text_only"

        updated = store.update_family_analysis_run(
            run["id"], status="completed", stage="portrait", stats={"people": 3}
        )
        assert updated["status"] == "completed"
        assert updated["current_stage"] == "portrait"
        assert updated["stats"] == {"people": 3}
        assert updated["completed_at"]
