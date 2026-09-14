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
        face = store.add_face_instance("asset-a", observation["id"], {
            "bbox": [1, 2, 30, 40], "confidence": 0.95, "quality": 0.9,
            "embedding": [1.0, 0.0, 0.0],
        })
        person_id = store.connection.execute(
            "SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],)
        ).fetchone()["entity_id"]

        evidence = FamilyGraphService(store)._build_text_evidence("album-a")

        assert evidence["people"][0]["person_id"] == person_id
        assert any("一家人在生日聚会上合影" in item for item in evidence["people"][0]["descriptions"])
        assert "/private/family.jpg" not in str(evidence)


def test_family_evidence_includes_pending_face_cluster_bindings_without_confirmation():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")
        store.create_asset("asset-a", "family.jpg", "image", "/private/family.jpg", "image/jpeg", scope_id="album-a")
        observation = store.add_observation("asset-a", {"caption": "一家人在客厅庆祝生日"}, scope_id="album-a")
        face = store.add_face_instance("asset-a", observation["id"], {
            "bbox": [1, 2, 30, 40], "confidence": 0.95, "quality": 0.9,
            "embedding": [1.0, 0.0, 0.0],
        })

        evidence = FamilyGraphService(store)._build_text_evidence("album-a")

        assert evidence["people"][0]["face_instance_ids"] == [face["id"]]
        assert evidence["people"][0]["descriptions"]


def test_run_persists_model_family_graph_without_sending_images():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")
        store.create_asset("asset-a", "family.jpg", "image", "/private/family.jpg", "image/jpeg", scope_id="album-a")
        observation = store.add_observation("asset-a", {"caption": "父女合影"}, scope_id="album-a")
        faces = [store.add_face_instance("asset-a", observation["id"], {
            "bbox": [index * 40, 2, index * 40 + 30, 40], "confidence": 0.95, "quality": 0.9,
            "embedding": embedding,
        }) for index, embedding in enumerate(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))]
        people = [store.connection.execute(
            "SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],)
        ).fetchone()["entity_id"] for face in faces]
        run = store.create_family_analysis_run("album-a", {"input_mode": "semantic_text_only"})
        gamma = FakeGamma(*people)

        result = FamilyGraphService(store, gamma).run(run["id"], "album-a")

        assert result["status"] == "completed", result.get("error")
        assert gamma.calls[0]["images"] is None
        membership = store.get_effective_family_membership("album-a", people[0])
        assert membership["membership"] == "family"
        relations = store.list_effective_family_relationships("album-a")
        assert {(item["predicate"], item["inverse_predicate"]) for item in relations} == {
            ("父亲", "女儿"), ("女儿", "父亲")
        }
