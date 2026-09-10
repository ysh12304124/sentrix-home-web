import unittest

from backend.retrieval.base import HardFilterContext, RetrievalQuery
from backend.retrieval.lite_named_vectors import (
    LiteNamedVectorIndex,
    LiteNamedVectorRetriever,
)


class _Point:
    def __init__(self, point_id, asset_id, score, media_type="image"):
        self.id = point_id
        self.score = score
        self.payload = {
            "asset_id": asset_id,
            "media_type": media_type,
            "unit_type": "video_segment" if media_type == "video" else "photo",
            "segment_index": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
        }


class _Index:
    last_error = None

    def __init__(self, points):
        self.points = points
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.points


class _Router:
    def __init__(self):
        self.visual_calls = 0
        self.text_calls = 0

    def embed_visual(self, text):
        self.visual_calls += 1
        return [1.0, 0.0]

    def embed_text(self, text):
        self.text_calls += 1
        return [0.0, 1.0]


class _Store:
    def __init__(self):
        self.assets = {
            "native-a": {"id": "native-a", "scope_id": "native", "media_type": "video",
                         "metadata_json": {"legacy_asset_id": "legacy-a"}},
            "native-b": {"id": "native-b", "scope_id": "native", "media_type": "image",
                         "metadata_json": {"legacy_asset_id": "legacy-b"}},
        }

    def list_assets(self, limit=100000):
        return list(self.assets.values())

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)


class LiteNamedVectorRetrieverTests(unittest.TestCase):
    def test_maps_legacy_ids_and_deduplicates_video_segments(self):
        index = _Index([
            _Point("p1", "legacy-a", 0.9, "video"),
            _Point("p2", "legacy-a", 0.8, "video"),
            _Point("p3", "legacy-b", 0.7, "image"),
        ])
        retriever = LiteNamedVectorRetriever(
            _Store(), _Router(), name="relation", vector_name="relation",
            embedding_slot="text", index=index, legacy_scope_id="legacy-scope",
        )
        hits = retriever.retrieve(
            RetrievalQuery("wedding", []),
            HardFilterContext(scope_ids=("native",)),
            limit=10,
        )
        self.assertEqual([hit.asset_id for hit in hits], ["native-a", "native-b"])
        self.assertEqual(hits[0].metadata["legacy_unit"]["start_sec"], 0.0)
        self.assertEqual(index.calls[0]["route"], "relation")
        self.assertEqual(index.calls[0]["scope_id"], "legacy-scope")

    def test_media_filter_is_preserved_after_legacy_mapping(self):
        index = _Index([
            _Point("p1", "legacy-a", 0.9, "video"),
            _Point("p2", "legacy-b", 0.8, "image"),
        ])
        retriever = LiteNamedVectorRetriever(
            _Store(), _Router(), name="clip_image", vector_name="image",
            embedding_slot="visual", index=index, legacy_scope_id="legacy-scope",
        )
        hits = retriever.retrieve(
            RetrievalQuery("photo", []),
            HardFilterContext(scope_ids=("native",), media_types=("image",)),
            limit=10,
        )
        self.assertEqual([hit.asset_id for hit in hits], ["native-b"])
        self.assertEqual(index.calls[0]["media_types"], ("image",))

    def test_named_text_routes_share_the_same_query_embedding(self):
        index = LiteNamedVectorIndex(".", "unused")
        router = _Router()
        first = index.embed(router, "text", "same query")
        second = index.embed(router, "text", "same query")
        visual = index.embed(router, "visual", "same query")
        self.assertEqual(first, second)
        self.assertEqual(router.text_calls, 1)
        self.assertEqual(router.visual_calls, 1)
        self.assertNotEqual(first, visual)


if __name__ == "__main__":
    unittest.main()
