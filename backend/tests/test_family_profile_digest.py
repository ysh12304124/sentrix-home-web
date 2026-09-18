import tempfile

from backend.db import MemoryStore


def test_person_profile_digest_reads_effective_family_graph_not_legacy_relationships():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(f"{temp_dir}/sentrix.db")
        store.create_memory_space("album-a", "相册 A")
        father = store.create_entity("爸爸", "person", "pending", scope_id="album-a")
        daughter = store.create_entity("女儿", "person", "pending", scope_id="album-a")
        store.set_family_membership("album-a", father["id"], "family", source="model")
        store.set_family_relationship(
            "album-a", father["id"], "父亲", daughter["id"], "女儿", source="model"
        )

        digest = store.person_profile_digest(father["id"], "album-a")

        assert digest["membership"] == "family"
        assert digest["relationships"] == [{"predicate": "父亲", "other_name": "女儿", "source": "model"}]
