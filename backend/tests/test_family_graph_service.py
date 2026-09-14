import tempfile

from backend.db import MemoryStore
from backend.family_graph_service import FamilyGraphService


def test_family_evidence_uses_persisted_semantic_text_not_asset_paths():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")
        store.create_asset("asset-a", "family.jpg", "image", "/private/family.jpg", "image/jpeg", scope_id="album-a")
        observation = store.add_observation("asset-a", {
            "caption": "一家人在生日聚会上合影",
            "activity": "庆祝生日",
            "canonical": {"scene": "家庭聚会"},
        }, scope_id="album-a")
        person = store.create_entity("人物一", "person", "pending", scope_id="album-a")
        store.connection.execute(
            "INSERT INTO entity_mentions(id, entity_id, observation_id, face_instance_id, confidence, created_at) VALUES (?, ?, ?, NULL, ?, ?)",
            ("mention-a", person["id"], observation["id"], 0.9, "2026-01-01T00:00:00"),
        )
        store.connection.commit()

        evidence = FamilyGraphService(store)._build_text_evidence("album-a")

        assert evidence["people"][0]["person_id"] == person["id"]
        assert any("一家人在生日聚会上合影" in item for item in evidence["people"][0]["descriptions"])
        assert "/private/family.jpg" not in str(evidence)
