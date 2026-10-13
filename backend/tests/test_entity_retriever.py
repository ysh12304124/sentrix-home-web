import sqlite3
import unittest
from types import SimpleNamespace

from backend.retrieval.entity import EntityRetriever
from backend.retrieval.base import HardFilterContext, RetrievalQuery
from backend.query_contracts import Constraint, SEMANTIC


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

    def test_uses_confirmed_face_cluster_membership(self):
        """Person retrieval must work even when observation people_json is sparse."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE observation_search_terms (asset_id TEXT, field_type TEXT, normalized_value TEXT, scope_id TEXT)")
        conn.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, scope_id TEXT)")
        conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, canonical_name TEXT, entity_type TEXT, status TEXT)")
        conn.execute("CREATE TABLE face_clusters (id TEXT PRIMARY KEY, entity_id TEXT, status TEXT)")
        conn.execute("CREATE TABLE face_instances (id TEXT PRIMARY KEY, asset_id TEXT, cluster_id TEXT)")
        conn.execute("INSERT INTO assets VALUES ('asset-face', 'album3-max')")
        conn.execute("INSERT INTO entities VALUES ('person-1', '明明', 'person', 'confirmed')")
        conn.execute("INSERT INTO face_clusters VALUES ('cluster-1', 'person-1', 'confirmed')")
        conn.execute("INSERT INTO face_instances VALUES ('face-1', 'asset-face', 'cluster-1')")
        conn.commit()
        store = SimpleNamespace(connection=conn, list_observations=lambda limit=100000: [])
        self.assertEqual(
            EntityRetriever(store)._resolve_asset_ids_for_person("明明", "album3-max"),
            {"asset-face"},
        )

    def test_cooccurrence_ranks_relation_candidate_first(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE observation_search_terms (asset_id TEXT, field_type TEXT, normalized_value TEXT, scope_id TEXT)")
        conn.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, scope_id TEXT)")
        conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, canonical_name TEXT, entity_type TEXT, status TEXT)")
        conn.execute("CREATE TABLE face_clusters (id TEXT PRIMARY KEY, entity_id TEXT, status TEXT)")
        conn.execute("CREATE TABLE face_instances (id TEXT PRIMARY KEY, asset_id TEXT, cluster_id TEXT)")
        conn.executemany("INSERT INTO assets VALUES (?, 'album3-max')", [('asset-one',), ('asset-both',)])
        conn.executemany("INSERT INTO entities VALUES (?, ?, 'person', 'confirmed')",
                         [('person-1', '明明'), ('person-2', '乐乐')])
        conn.executemany("INSERT INTO face_clusters VALUES (?, ?, 'confirmed')",
                         [('cluster-1', 'person-1'), ('cluster-2', 'person-2')])
        conn.executemany("INSERT INTO face_instances VALUES (?, ?, ?)",
                         [('face-one', 'asset-one', 'cluster-1'),
                          ('face-both-1', 'asset-both', 'cluster-1'),
                          ('face-both-2', 'asset-both', 'cluster-2')])
        conn.commit()
        store = SimpleNamespace(connection=conn, list_observations=lambda limit=100000: [])
        query = RetrievalQuery(
            whole_query="明明和乐乐合影",
            constraints=[Constraint("person", "明明", SEMANTIC, "direct_or_possible"),
                         Constraint("person", "乐乐", SEMANTIC, "direct_or_possible")],
        )
        hits = EntityRetriever(store).retrieve(query, HardFilterContext(scope_ids=("album3-max",)), 10)
        self.assertEqual([hit.asset_id for hit in hits], ['asset-both', 'asset-one'])
        self.assertGreater(hits[0].raw_score, hits[1].raw_score)


if __name__ == "__main__":
    unittest.main()
