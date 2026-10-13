import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agent_runtime import tools
from backend.agent_runtime.result_set import ResultSetStore
from backend.db import MemoryStore
from backend.model_clients import GammaClient


class SearchMetricsTests(unittest.TestCase):
    def test_slot_call_is_drained_into_tool_on_success_parse_failure_and_http_error(self):
        for response_kind in ('success', 'invalid_json', 'http_error'):
            with self.subTest(response_kind=response_kind), tempfile.TemporaryDirectory() as tmp:
                store = MemoryStore(str(Path(tmp) / 'test.db'))
                gamma = GammaClient(base_url='http://sentrix-vllm/v1', model='test-model')
                slots = {'time': {}, 'place': {}, 'person': [], 'event': {}, 'object': []}
                content = json.dumps(slots) if response_kind == 'success' else 'invalid json'
                with patch.dict(tools._RUNTIME, {
                    'store': store, 'gamma': gamma, 'result_sets': ResultSetStore(store),
                }, clear=True), patch('backend.model_clients.httpx.post') as post, patch.object(
                    tools, '_relaxed_retrieve', return_value=(tools._SlotRetrievalPacket([]), 0)
                ):
                    if response_kind == 'http_error':
                        import httpx
                        post.side_effect = httpx.ConnectError('offline')
                    else:
                        post.return_value.json.return_value = {
                            'choices': [{'message': {'content': content}}],
                            'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                        }
                    result = tools._search_memories(
                        {'query': 'trip'}, context={'scope_id': 'test'})
                    calls = result['_model_call_metrics']
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]['tool_subtask'], 'semantic_slots')
                    self.assertIsNotNone(calls[0]['total_ms'])
                    self.assertEqual(gamma.get_and_clear_call_metrics(), [])
                    if response_kind == 'http_error':
                        self.assertEqual(calls[0]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
