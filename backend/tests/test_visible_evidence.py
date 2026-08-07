"""RX-3 VisibleEvidence selection tests."""

import unittest

from backend.evidence_retrieval import EvidencePacket
from backend.visible_evidence import select_visible_assets


def _item(asset_id, level="approximate", conds=None, group=None, attributions=None):
    return {
        "asset_id": asset_id, "file_name": f"{asset_id}.jpg", "media_type": "image",
        "observation_ids": [f"obs-{asset_id}"], "evidence_ids": [asset_id],
        "condition_results": conds or {"semantic:查询": {"status": "unknown"}},
        "level": level, "score": 0.5, "captured_at": "2025-10-02",
        "near_duplicate_group": group, "attributions": attributions,
    }


def _packet(items):
    return EvidencePacket("q", "home", "general", assets=items)


class VisibleEvidenceTests(unittest.TestCase):
    def test_exact_then_approximate_capped_at_three(self):
        exact = [_item(f"e{i}", level="exact", conds={"image": {"status": "matched"}}) for i in range(5)]
        approx = [_item(f"a{i}", conds={"place:海边": {"status": "matched"}}) for i in range(5)]
        visible = select_visible_assets(_packet(exact + approx))
        self.assertEqual(len(visible), 6)  # 3 exact + 3 approximate
        self.assertEqual(visible[0].display_handle, "照片1")
        self.assertEqual(visible[0].result_level, "exact")

    def test_all_unknown_excluded_by_default(self):
        approx = [_item(f"a{i}", conds={"semantic:海豚": {"status": "unknown"}}) for i in range(5)]
        visible = select_visible_assets(_packet(approx))
        self.assertEqual(visible, [])

    def test_all_unknown_included_when_all_relevant(self):
        approx = [_item(f"a{i}", conds={"semantic:海豚": {"status": "unknown"}}) for i in range(4)]
        visible = select_visible_assets(_packet(approx), all_relevant=True)
        self.assertEqual(len(visible), 4)

    def test_near_duplicate_collapses_to_representative(self):
        items = [
            _item("a1", conds={"place:海边": {"status": "matched"}}, group="g1"),
            _item("a2", conds={"place:海边": {"status": "matched"}}, group="g1"),
            _item("b1", conds={"place:爬山": {"status": "matched"}}, group=None),
        ]
        visible = select_visible_assets(_packet(items))
        self.assertEqual(len(visible), 2)
        rep = next((v for v in visible if v.asset_id == "a1"), None)
        self.assertIsNotNone(rep)
        self.assertEqual(rep.near_duplicate_size, 2)

    def test_no_assets(self):
        self.assertEqual(select_visible_assets(_packet([])), [])

    def test_media_url_for_images_only(self):
        item = _item("a1", conds={"place:海边": {"status": "matched"}})
        item["media_type"] = "video"
        visible = select_visible_assets(_packet([item]))
        self.assertIsNone(visible[0].media_url)

    def test_supported_and_uncertain_aspects(self):
        item = _item("a1", conds={"time:2025年10月": {"status": "matched"},
                                  "activity:爬山": {"status": "unknown"}})
        visible = select_visible_assets(_packet([item]))
        self.assertIn("2025年10月", visible[0].supported_aspects)
        self.assertIn("爬山", visible[0].uncertain_aspects)


if __name__ == "__main__":
    unittest.main()
