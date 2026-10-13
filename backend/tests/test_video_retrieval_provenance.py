from backend.agent_runtime.tools import (
    _add_video_parent_candidates,
    _explicit_media_filter,
)
from backend.retrieval.base import effective_media_type


class _Store:
    def __init__(self):
        self.assets = {
            "frame-1": {
                "id": "frame-1",
                "media_type": "image",
                "derived_kind": "video_keyframe_webp",
                "parent_asset_id": "video-1",
            },
            "video-1": {"id": "video-1", "media_type": "video"},
            "photo-1": {"id": "photo-1", "media_type": "image"},
        }

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)


def test_keyframe_is_video_for_retrieval_media_filter():
    assert effective_media_type({
        "media_type": "image",
        "derived_kind": "video_keyframe_webp",
    }) == "video"
    assert effective_media_type({"media_type": "image"}) == "image"


def test_explicit_media_filter_keeps_mixed_queries_unfiltered():
    assert _explicit_media_filter("找那段视频") == "video"
    assert _explicit_media_filter("找那张照片") == "image"
    assert _explicit_media_filter("比较视频和照片") is None


def test_keyframe_hit_adds_parent_video_before_candidate_cap():
    store = _Store()
    scores = {"frame-1": 1.0, "photo-1": 0.95}
    added = _add_video_parent_candidates(scores, list(scores), store)
    assert added == ["video-1"]
    assert scores["video-1"] == 0.94


def test_image_only_search_does_not_add_parent_video():
    store = _Store()
    scores = {"frame-1": 1.0}
    assert _add_video_parent_candidates(
        scores, list(scores), store, media_constraint="image") == []
    assert "video-1" not in scores
