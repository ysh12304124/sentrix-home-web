#!/usr/bin/env python3
"""Regression coverage for selecting models from unmanaged endpoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.photobench.backend import benchmark_orchestrator as orchestrator


class ModelEndpointSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = orchestrator.OrchestratorRepository(Path(self.temporary.name) / "results")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @patch.object(orchestrator, "OpenAICompatibleInferenceProvider")
    def test_single_model_endpoint_is_selected_automatically(self, provider_cls) -> None:
        provider = provider_cls.return_value
        provider.list_models.return_value = {"models": ["model-a"]}
        provider.capabilities.return_value = {"provider": "openai_compatible"}

        result = self.repo.query_current_model("", "host:11434")

        self.assertEqual(result["served_model_name"], "model-a")
        self.assertFalse(result["selection_required"])

    @patch.object(orchestrator, "OpenAICompatibleInferenceProvider")
    def test_multiple_models_require_explicit_selection_without_error(self, provider_cls) -> None:
        provider = provider_cls.return_value
        provider.list_models.return_value = {"models": ["model-a", "model-b"]}
        provider.capabilities.return_value = {"provider": "openai_compatible"}

        result = self.repo.query_current_model("", "host:11434")

        self.assertIsNone(result["served_model_name"])
        self.assertTrue(result["selection_required"])
        self.assertEqual(result["served_models"], ["model-a", "model-b"])

    @patch.object(orchestrator, "OpenAICompatibleInferenceProvider")
    def test_explicit_model_must_exist(self, provider_cls) -> None:
        provider = provider_cls.return_value
        provider.list_models.return_value = {"models": ["model-a", "model-b"]}
        provider.capabilities.return_value = {}

        selected = self.repo.query_current_model("", "host:11434", "model-b")
        self.assertEqual(selected["served_model_name"], "model-b")
        with self.assertRaisesRegex(ValueError, "is not exposed"):
            self.repo.query_current_model("", "host:11434", "missing")

    @patch.object(orchestrator, "OpenAICompatibleInferenceProvider")
    @patch.object(orchestrator, "ManagerLifecycleProvider")
    def test_manager_state_must_match_endpoint_and_requested_model(
        self, manager_cls, provider_cls,
    ) -> None:
        manager_cls.return_value.state.return_value = {"served_model_name": "model-a", "profile": "a"}
        provider = provider_cls.return_value
        provider.list_models.return_value = {"models": ["model-a", "model-b"]}
        provider.capabilities.return_value = {}

        with self.assertRaisesRegex(ValueError, "cannot reuse requested model"):
            self.repo.query_current_model("manager:8500", "host:8100", "model-b")

        manager_cls.return_value.state.return_value = {"served_model_name": "stale", "profile": "a"}
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            self.repo.query_current_model("manager:8500", "host:8100")

    @patch.object(orchestrator, "OpenAICompatibleInferenceProvider")
    def test_post_probe_targets_selected_model(self, provider_cls) -> None:
        provider = provider_cls.return_value
        provider.list_models.return_value = {"models": ["model-a", "model-b"]}
        provider.capabilities.return_value = {}
        provider.chat.return_value.json.return_value = {"choices": [{"message": {"content": "OK"}}]}

        result = self.repo.test_model_endpoint("host:11434", "model-b")

        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "model-b")
        self.assertEqual(provider.chat.call_args.args[0]["model"], "model-b")

    def test_start_suite_revalidates_and_binds_selected_external_model(self) -> None:
        data_root = Path(self.temporary.name) / "data"
        album = data_root / "album"
        album.mkdir(parents=True)
        (album / "manifest.json").write_text(json.dumps({
            "album_id": "album",
            "album_name": "Album",
            "qa_sets": {"qa": "qa.jsonl"},
        }), encoding="utf-8")
        snapshot = {
            "model_id": "model-b",
            "served_model_name": "model-b",
            "selection_required": False,
            "model_base_url": "http://host:11434/v1",
        }
        self.repo.query_current_model = Mock(return_value=snapshot)

        with patch.object(orchestrator, "BENCHMARK_DATA_ROOT", data_root), \
                patch.object(orchestrator, "request_json") as request, \
                patch.object(orchestrator, "BenchmarkRun") as run_cls, \
                patch.object(orchestrator.threading, "Thread") as thread_cls:
            result = self.repo.start_suite({
                "album_id": "album",
                "qa_set": "qa",
                "models": [orchestrator.CURRENT_MODEL_SELECTION],
                "sentrix_url": "http://sentrix:8091",
                "model_base_url": "host:11434",
                "endpoint_model": "model-b",
            })

        self.repo.query_current_model.assert_called_once_with(
            "", "http://host:11434/v1", "model-b",
        )
        self.assertEqual(result["models"], ["model-b"])
        self.assertEqual(run_cls.call_args.kwargs["model_profile"], "model-b")
        request.assert_called_once_with(
            "http://sentrix:8091/api/model-profiles/bind-external-runtime",
            {"base_url": "http://host:11434/v1", "model": "model-b"},
            "POST", 30,
        )
        thread_cls.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
