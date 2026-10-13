from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from backend.video.contracts import VideoMetadata, WorldMMKeyframe, WorldMMResult, WorldMMScene
from backend.video.processor import VideoMemoryAdapter


def test_svd_events_keep_separate_memory_and_inherit_source_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTRIX_DATA_DIR", str(tmp_path))
    source = tmp_path / "candidate.jpg"
    Image.new("RGB", (32, 32), "blue").save(source)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"test")
    scenes = [
        WorldMMScene(f"event-{index}", index, index * 5, (index + 1) * 5, [
            WorldMMKeyframe(f"frame-{index}", str(source), index * 5 + 1, index * 50,
                            raw={"yolo_event_id": f"event-{index}"}),
        ])
        for index in range(2)
    ]
    store = MagicMock()
    store.update_asset.side_effect = lambda asset_id, status, metadata: {"status": status, **metadata}
    store.create_video_scene_event.side_effect = [{"id": "memory-0"}, {"id": "memory-1"}]
    gamma = MagicMock()
    gamma.analyze_video_event.side_effect = [
        {"caption": "饮水", "representative_indices": [0]},
        {"caption": "讲话", "representative_indices": [0]},
    ]
    pipeline = SimpleNamespace(store=store, gamma=gamma, geocoder=MagicMock(),
                               process=MagicMock(return_value={"status": "processed"}),
                               summarize_event=MagicMock())
    # A per-asset choice must override the global hybrid setting.
    adapter = VideoMemoryAdapter(keyframe_algorithm="hybrid_webp")
    adapter.svd.run = MagicMock(return_value=WorldMMResult({}, scenes, str(tmp_path)))
    asset = {"id": "video-1", "path": str(video), "scope_id": "scope-1",
             "captured_at": "2026-09-01T10:00:00+00:00", "captured_location": "公园",
             "metadata_json": {"keyframe_algorithm": "svd_lowrank",
                               "source_media_id": "media-1", "source_media_title": "家庭视频"}}
    with patch("backend.video.processor.probe_video_metadata", return_value=VideoMetadata(codec="h264")):
        result = adapter.process(asset, pipeline)

    assert result["status"] == "processed"
    assert result["video_scene_event_ids"] == ["memory-0", "memory-1"]
    assert result["persistent_keyframe_count"] == 2
    assert [call.kwargs["forced_event_id"] for call in pipeline.process.call_args_list] == ["memory-0", "memory-1"]
    for call in store.create_asset.call_args_list:
        assert call.args[4] == "image/webp"
        with Image.open(call.args[3]) as image:
            assert image.format == "WEBP"
        provenance = call.args[6]
        assert provenance["capture_time_source"] == "upload_metadata"
        assert provenance["location_source"] == "upload_metadata"
        assert provenance["location_provenance"] == "upload_metadata"
        assert provenance["source_media_id"] == "media-1"
    for call in store.create_video_scene_event.call_args_list:
        event = call.args[0]
        assert event["title"].startswith("家庭视频 · ")
        assert event["source_metadata"]["source_media_id"] == "media-1"
        assert event["source_metadata"]["capture_time_source"] == "upload_metadata"
