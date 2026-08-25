import json
import unittest

from backend.retrieval.metadata import MetadataRetriever
from backend.retrieval.base import HardFilterContext, RetrievalQuery
from backend.agent_runtime.canonical_intent import extract_constraints


class _Store:
    def __init__(self, assets):
        self._assets = assets

    def list_assets(self, **kwargs):
        return list(self._assets)


class RetrievalRegressionTests(unittest.TestCase):
    def test_metadata_place_filter_parses_json_metadata(self):
        assets = [
            {"id": "target", "scope_id": "s", "media_type": "image",
             "metadata_json": json.dumps({"reverse_geocode": {"label": "河北省邯郸市馆陶县", "city": "邯郸市", "district": "馆陶县"}})},
            {"id": "other", "scope_id": "s", "media_type": "image",
             "metadata_json": json.dumps({"reverse_geocode": {"label": "河北省邯郸市永年区", "city": "邯郸市", "district": "永年区"}})},
        ]
        hits = MetadataRetriever(_Store(assets)).retrieve(
            RetrievalQuery("婚礼", []),
            HardFilterContext(scope_ids=("s",), place="馆陶"),
            limit=20,
        )
        self.assertEqual([hit.asset_id for hit in hits], ["target"])

    def test_canonical_place_prefers_trailing_district(self):
        class Store:
            class Conn:
                def execute(self, sql, params):
                    class Rows:
                        def fetchall(self):
                            return [{"label": "河北省邯郸市馆陶县", "dist": "馆陶县", "city": "邯郸市"}]
                    return Rows()
            connection = Conn()

        result = extract_constraints("我在邯郸馆陶记录的婚礼", Store(), "s")
        self.assertEqual(result["place"], "馆陶")

    def test_landmark_alias_maps_to_authoritative_district(self):
        class Store:
            class Conn:
                def execute(self, sql, params):
                    class Rows:
                        def fetchall(self):
                            return [{"label": "河北省石家庄市赵县", "dist": "赵县", "city": "石家庄市"}]
                    return Rows()
            connection = Conn()

        result = extract_constraints("赵州桥石桥上的合影", Store(), "s")
        self.assertEqual(result["place"], "赵县")


if __name__ == "__main__":
    unittest.main()
