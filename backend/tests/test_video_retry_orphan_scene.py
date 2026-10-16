from backend.db import MemoryStore


def test_retry_cleans_scene_dependencies_without_observations(tmp_path):
    store = MemoryStore(str(tmp_path / 'memory.db'))
    try:
        store.create_asset('video', 'video.mp4', 'video', '/unused/video.mp4', 'video/mp4', 1, {})
        event = store.create_video_scene_event({'source_asset_id': 'video', 'source_scene_index': 0})
        store.connection.execute(
            "INSERT INTO event_revisions(id,event_id,field_name,source,created_at) VALUES (?,?,?,?,?)",
            ('revision', event['id'], 'summary', 'model', '2026-10-16T00:00:00Z'))
        store.connection.commit()
        store.cleanup_video_derivatives('video')
        assert store.get_asset('video') is not None
        assert store.get_event(event['id']) is None
        assert not store.connection.execute('PRAGMA foreign_key_check').fetchall()
    finally:
        store.close()
