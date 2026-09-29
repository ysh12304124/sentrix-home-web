import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.evaluate_agent_replay import inspect_database


class AgentReplayReadinessTests(unittest.TestCase):
    def _make_database(self, root):
        database = root / "memory.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE entities (
                id TEXT PRIMARY KEY, scope_id TEXT, entity_type TEXT,
                canonical_name TEXT, status TEXT, family_role TEXT
            );
            CREATE TABLE relationships (
                id TEXT PRIMARY KEY, scope_id TEXT, subject_entity_id TEXT,
                predicate TEXT, object_entity_id TEXT, status TEXT,
                evidence_ids_json TEXT
            );
            CREATE TABLE semantic_claims (
                id TEXT PRIMARY KEY, scope_id TEXT, person_id TEXT,
                dimension TEXT, value_text TEXT, supporting_event_ids_json TEXT,
                evidence_ids_json TEXT, status TEXT
            );
            CREATE TABLE person_patterns (
                id TEXT PRIMARY KEY, scope_id TEXT, person_id TEXT,
                pattern_type TEXT, value_text TEXT,
                supporting_event_ids_json TEXT, evidence_ids_json TEXT,
                status TEXT
            );
            CREATE TABLE semantic_profiles (
                id TEXT PRIMARY KEY, scope_id TEXT, person_id TEXT,
                summary_zh TEXT, evidence_ids_json TEXT
            );
            CREATE TABLE events (
                id TEXT PRIMARY KEY, scope_id TEXT, time_start TEXT,
                time_end TEXT, place TEXT, summary TEXT
            );
            CREATE TABLE observations (
                id TEXT PRIMARY KEY, scope_id TEXT, asset_id TEXT,
                captured_at TEXT, caption TEXT, transcript TEXT,
                people_json TEXT
            );
            CREATE TABLE assets (
                id TEXT PRIMARY KEY, scope_id TEXT, media_type TEXT,
                file_name TEXT, captured_at TEXT
            );
            CREATE TABLE person_event_memory (
                id TEXT PRIMARY KEY, scope_id TEXT, person_id TEXT,
                event_id TEXT, evidence_ids_json TEXT
            );
            CREATE TABLE entity_observations (
                entity_id TEXT, observation_id TEXT
            );
            CREATE TABLE event_entities (
                event_id TEXT, entity_id TEXT, evidence_ids_json TEXT
            );
            CREATE TABLE memory_vectors (
                id TEXT PRIMARY KEY, scope_id TEXT, source_type TEXT,
                source_id TEXT
            );
            INSERT INTO entities VALUES
                ('person-ming', 'album1', 'person', '明哥', 'confirmed', '孩子'),
                ('person-pending', 'album1', 'person', '候选人物', 'pending', NULL);
            INSERT INTO assets VALUES ('asset-1', 'album1', 'image', 'ming.jpg', '2025-04-30');
            INSERT INTO observations VALUES ('obs-1', 'album1', 'asset-1', '2025-04-30', '明哥在展览馆拍照', '', '["person-ming"]');
            INSERT INTO events VALUES ('event-1', 'album1', '2025-04-30', '2025-04-30', '展览馆', '明哥参观展示');
            INSERT INTO person_event_memory VALUES ('memory-1', 'album1', 'person-ming', 'event-1', '["event-1", "obs-1", "asset-1"]');
            INSERT INTO entity_observations VALUES ('person-ming', 'obs-1');
            INSERT INTO event_entities VALUES ('event-1', 'person-ming', '["event-1"]');
            INSERT INTO semantic_claims VALUES ('claim-1', 'album1', 'person-ming', 'activity', '参观展示', '["event-1"]', '["obs-1"]', 'active');
            INSERT INTO person_patterns VALUES ('pattern-1', 'album1', 'person-ming', 'activity', '看展', '["event-1", "event-2"]', '["claim-1"]', 'active');
            INSERT INTO semantic_profiles VALUES ('profile-1', 'album1', 'person-ming', '明哥已确认', '["claim-1"]');
            """
        )
        connection.commit()
        connection.close()
        return database

    def test_report_reads_only_and_reports_person_evidence_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._make_database(Path(directory))
            before = database.read_bytes()

            report = inspect_database(database, target_terms=("明哥",), sample_limit=30)

            self.assertTrue(report["readonly"])
            self.assertEqual(report["integrity_check"], "ok")
            self.assertTrue(report["ready"])
            self.assertEqual(report["counts"]["confirmed_people"], 1)
            self.assertEqual(report["target_entities"][0]["name"], "明哥")
            self.assertEqual(report["target_entities"][0]["claims"], 1)
            self.assertEqual(report["target_entities"][0]["patterns"], 1)
            self.assertEqual(report["target_entities"][0]["person_events"], 1)
            self.assertEqual(report["target_entities"][0]["observations"], 1)
            self.assertEqual(report["target_entities"][0]["assets"], 1)
            self.assertEqual(database.read_bytes(), before)

    def test_report_uses_person_event_evidence_when_entity_observation_link_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._make_database(Path(directory))
            connection = sqlite3.connect(database)
            connection.execute("DELETE FROM entity_observations")
            connection.commit()
            connection.close()

            report = inspect_database(database, target_terms=("明哥",), sample_limit=30)

            self.assertTrue(report["ready"])
            self.assertEqual(report["target_entities"][0]["observations"], 1)
            self.assertEqual(report["target_entities"][0]["assets"], 1)
            self.assertIn("person_event_memory.evidence_ids_json", report["target_entities"][0]["evidence_link_paths"])

    def test_report_blocks_database_with_missing_agent_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "empty.db"
            sqlite3.connect(database).close()

            report = inspect_database(database, target_terms=("明哥",), sample_limit=30)

            self.assertFalse(report["ready"])
            self.assertEqual(report["integrity_check"], "ok")
            self.assertIn("entities", report["missing_tables"])
            self.assertIn("semantic_claims", report["missing_tables"])
            self.assertTrue(report["warnings"])

    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._make_database(Path(directory))

            report = inspect_database(database)

            json.dumps(report, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
