import tempfile

from backend.db import MemoryStore
from backend.family_graph_service import FamilyGraphService


class FakeGamma:
    def __init__(self, person_a="person-a", person_b="person-b"):
        self.calls = []
        self.person_a = person_a
        self.person_b = person_b

    def chat(self, prompt, images=None, json_mode=True, role=None):
        self.calls.append({"prompt": prompt, "images": images, "json_mode": json_mode, "role": role})
        return '{"memberships": [{"person_id": "' + self.person_a + '", "membership": "family", "confidence": 0.91}], "relationships": [{"subject_entity_id": "' + self.person_a + '", "predicate": "父亲", "object_entity_id": "' + self.person_b + '", "inverse_predicate": "女儿", "confidence": 0.88}]}'


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


def test_run_persists_model_family_graph_without_sending_images():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")
        person_a = store.create_entity("人物甲", "person", "pending", scope_id="album-a")
        person_b = store.create_entity("人物乙", "person", "pending", scope_id="album-a")
        store.create_asset("asset-a", "family.jpg", "image", "/private/family.jpg", "image/jpeg", scope_id="album-a")
        observation = store.add_observation("asset-a", {"caption": "父女合影"}, scope_id="album-a")
        for index, person in enumerate((person_a, person_b)):
            store.connection.execute(
                "INSERT INTO entity_mentions(id, entity_id, observation_id, face_instance_id, confidence, created_at) VALUES (?, ?, ?, NULL, ?, ?)",
                (f"mention-{index}", person["id"], observation["id"], 0.9, "2026-01-01T00:00:00"),
            )
        store.connection.commit()
        run = store.create_family_analysis_run("album-a", {"input_mode": "semantic_text_only"})
        gamma = FakeGamma(person_a["id"], person_b["id"])

        result = FamilyGraphService(store, gamma).run(run["id"], "album-a")

        assert result["status"] == "completed", result.get("error")
        assert gamma.calls[0]["images"] is None
        membership = store.get_effective_family_membership("album-a", person_a["id"])
        assert membership["membership"] == "family"
        relations = store.list_effective_family_relationships("album-a")
        assert {(item["predicate"], item["inverse_predicate"]) for item in relations} == {
            ("父亲", "女儿"), ("女儿", "父亲")
        }
