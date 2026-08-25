import sqlite3
import unittest
from types import SimpleNamespace

from backend.retrieval.entity import EntityRetriever


class EntityRetrieverTests(unittest.TestCase):
    def test_falls_back_when_derived_person_projection_is_empty(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE observation_search_terms (asset_id TEXT, field_type TEXT, normalized_value TEXT, scope_id TEXT)")
        conn.commit()
        store = SimpleNamespace(
            connection=conn,
            list_observations=lambda limit=100000: [{
                "asset_id": "asset-1", "scope_id": "album3-max", "people": [{"name": "明明"}],
            }],
        )
        self.assertEqual(
            EntityRetriever(store)._resolve_asset_ids_for_person("明明", "album3-max"),
            {"asset-1"},
        )


if __name__ == "__main__":
    unittest.main()
