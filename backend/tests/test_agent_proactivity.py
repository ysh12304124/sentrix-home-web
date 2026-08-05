import os
import tempfile
import unittest
from unittest.mock import patch

from backend.agent import MemoryAgent
from backend.db import MemoryStore
from backend.tests.test_agent import RefusingGamma


class AgentProactivityTests(unittest.TestCase):
    def _agent(self, directory):
        store = MemoryStore(f"{directory}/memory.db")
        person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
        asset = store.create_asset("proactive_asset", "proactive.jpg", "image", "/tmp/proactive.jpg")
        observation = store.add_observation(asset["id"], {"caption": "明哥在展览馆拍照"})
        event = store.merge_observation_into_event(observation)
        store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
        return store, event

    def test_probe_requires_feature_flag_and_does_not_read_concrete_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store, _ = self._agent(directory)
            with patch.dict(os.environ, {"SENTRIX_PROACTIVE_MEMORY": "1"}, clear=False):
                result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
                    "今天去了展览馆", "proactive-probe", scope_id="home-default", viewer_id="owner",
                )
            self.assertEqual(result["memory_intensity"], "probe")
            self.assertFalse(result["memory_used"])
            self.assertFalse(result["memory_actually_referenced"])
            self.assertTrue(result["proactivity_probe_performed"])
            self.assertTrue(result["proactivity_candidate_found"])
            self.assertTrue(result["proactive_recall"]["entry_text"])
            self.assertFalse(result["evidence"])

    def test_disabled_feature_and_sensitive_topic_never_offer_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            store, _ = self._agent(directory)
            disabled = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
                "今天去了展览馆", "proactive-off", scope_id="home-default", viewer_id="owner",
            )
            self.assertEqual(disabled["memory_intensity"], "none")
            self.assertFalse(disabled["proactivity_probe_performed"])

            with patch.dict(os.environ, {"SENTRIX_PROACTIVE_MEMORY": "1"}, clear=False):
                sensitive = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
                    "今天发生了家庭冲突", "proactive-sensitive", scope_id="home-default", viewer_id="owner",
                )
            self.assertFalse(sensitive["proactivity_probe_performed"])
            self.assertFalse(sensitive["proactivity_candidate_found"])

    def test_cooldown_ignore_streak_and_acceptance_are_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            store, event = self._agent(directory)
            with patch.dict(os.environ, {"SENTRIX_PROACTIVE_MEMORY": "1"}, clear=False):
                agent = MemoryAgent(store, gamma=RefusingGamma())
                first = agent.answer_turn("今天去了展览馆", "proactive-feedback", scope_id="home-default", viewer_id="owner")
                scene_key = first["proactive_recall"]["scene_key"]
                repeated = agent.answer_turn("今天去了展览馆", "proactive-feedback-2", scope_id="home-default", viewer_id="owner")
                self.assertFalse(repeated["proactivity_candidate_found"])
                agent.answer_turn(
                    "好的", "proactive-ignore-1", feedback={"proactivity_outcome": "ignored", "proactivity_scene_key": scene_key},
                    scope_id="home-default", viewer_id="owner",
                )
                agent.answer_turn(
                    "好的", "proactive-ignore-2", feedback={"proactivity_outcome": "ignored", "proactivity_scene_key": scene_key},
                    scope_id="home-default", viewer_id="owner",
                )
                preference = agent.annotation_store.get_preference("home-default", "owner")
                self.assertGreaterEqual(preference["ignore_streak"], 2)
                self.assertLessEqual(preference["level"], 1)
                opened = agent.answer_turn(
                    "看看这段回忆", "proactive-feedback", feedback={"proactivity_outcome": "accepted", "proactivity_scene_key": scene["event_id"] if isinstance(scene := first["proactive_recall"], dict) else event["id"]},
                    scope_id="home-default", viewer_id="owner",
                )
            self.assertTrue(opened["memory_used"])
            self.assertTrue(opened["evidence"])
