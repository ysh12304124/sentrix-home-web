import json
from unittest.mock import Mock, patch

from backend.model_clients import GammaClient, ModelError


def _client():
    client = object.__new__(GammaClient)
    client.backend = "openai"
    client._model_setting = "qwen3.5-4b"
    return client


def test_video_event_retries_at_reported_image_limit_and_maps_indices():
    client = _client()
    client.chat = Mock(side_effect=[
        ModelError("At most 2 image(s) may be provided in one prompt. (parameter=image)"),
        json.dumps({"caption": "ok", "representative_indices": [1]}),
    ])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        result = client.analyze_video_event([f"frame-{index}.webp" for index in range(5)])

    assert len(client.chat.call_args_list[0].args[1]) == 5
    assert len(client.chat.call_args_list[1].args[1]) == 2
    assert result["representative_indices"] == [4]
    assert result["video_event_evidence_indices"] == [0, 4]
    assert result["video_event_evidence_count"] == 2
    assert result["video_event_source_evidence_count"] == 5


def test_video_event_does_not_retry_unrelated_model_errors():
    client = _client()
    client.chat = Mock(side_effect=ModelError("service unavailable"))
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        try:
            client.analyze_video_event(["frame.webp"])
        except ModelError as error:
            assert "service unavailable" in str(error)
        else:
            raise AssertionError("ModelError was not raised")
    assert client.chat.call_count == 1


def test_video_event_retries_three_images_when_five_image_json_is_unparseable():
    client = _client()
    client.chat = Mock(side_effect=[
        "truncated non-json response",
        json.dumps({"caption": "ok", "activity": "walking", "representative_indices": [1]}),
    ])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        result = client.analyze_video_event([f"frame-{index}.webp" for index in range(5)])

    assert len(client.chat.call_args_list[0].args[1]) == 5
    assert len(client.chat.call_args_list[1].args[1]) == 3
    assert result["representative_indices"] == [2]
    assert result["video_event_evidence_indices"] == [0, 2, 4]
    assert result["video_event_evidence_count"] == 3
    assert result["video_event_source_evidence_count"] == 5
    assert result["video_event_fallback_reason"] == "unparseable_multi_image_response"
