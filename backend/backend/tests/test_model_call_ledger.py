"""Phase 12B-FC V2 — ModelCallLedger tests."""

import unittest

from backend.validation import model_call_ledger as ledger


class ModelCallLedgerTests(unittest.TestCase):
    def tearDown(self):
        ledger.end_turn()

    def test_begin_end_turn_bracket_records(self):
        ledger.begin_turn()
        ledger.new_call("parser", "gemma4:12b", "http://x")
        ledger.record_response('{"mode":"evidence"}', actual_model="gemma4:12b",
                               endpoint="http://x", json_mode=True)
        ledger.finish_call()
        records = ledger.end_turn()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["role"], "parser")
        self.assertEqual(records[0]["actual_model"], "gemma4:12b")
        self.assertTrue(records[0]["json_valid"])
        self.assertFalse(records[0]["fallback_used"])
        self.assertIsNotNone(records[0]["latency_ms"])

    def test_record_without_active_is_noop(self):
        ledger.begin_turn()
        out = ledger.record_response("x", actual_model="gemma4:12b", endpoint="e", json_mode=False)
        self.assertIsNone(out)
        ledger.end_turn()

    def test_degraded_call_marked(self):
        ledger.begin_turn()
        rec = ledger.new_call("answer", "gemma4:12b", "http://x")
        ledger.record_response(None, actual_model=None, endpoint="http://x", json_mode=False,
                               fallback_used=True, error="timeout")
        ledger.finish_call()
        records = ledger.end_turn()
        self.assertTrue(records[0]["fallback_used"])
        self.assertEqual(records[0]["error"], "timeout")

    def test_invalid_json_marked(self):
        ledger.begin_turn()
        ledger.new_call("parser", "gemma4:12b", "http://x")
        ledger.record_response("not-json", actual_model="gemma4:12b", endpoint="http://x",
                               json_mode=True)
        ledger.finish_call()
        records = ledger.end_turn()
        self.assertFalse(records[0]["json_valid"])


if __name__ == "__main__":
    unittest.main()
