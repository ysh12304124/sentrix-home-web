import math


def sort_video_scene_observations(observations, keyframe_assets=None):
    """Return video observations in source-timeline order.

    The event/observation join table has no ordering column, so SQLite may
    return observations in insertion order.  Prefer the timestamp carried by
    each derived keyframe and fall back to the event's keyframe list.
    """
    frame_order = {}
    for index, asset in enumerate(keyframe_assets or []):
        asset_id = asset.get("id")
        if asset_id:
            frame_order[asset_id] = (_source_timestamp(asset), index)

    def ordering_key(observation):
        asset = observation.get("asset") or {}
        asset_id = observation.get("asset_id") or asset.get("id")
        timestamp = _source_timestamp(asset)
        frame_index = len(frame_order)
        if timestamp is None and asset_id in frame_order:
            timestamp, frame_index = frame_order[asset_id]
        elif asset_id in frame_order:
            frame_index = frame_order[asset_id][1]
        if timestamp is not None:
            return (0, timestamp, frame_index, str(observation.get("id") or ""))
        return (
            1,
            math.inf,
            frame_index,
            str(observation.get("captured_at") or ""),
            str(observation.get("id") or ""),
        )

    return sorted(observations or [], key=ordering_key)


def _source_timestamp(asset):
    value = (asset or {}).get("source_timestamp_sec")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
