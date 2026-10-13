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


def test_video_event_recovers_empty_two_image_result_from_one_frame():
    client = _client()
    client.chat = Mock(side_effect=[
        ModelError("At most 2 image(s) may be provided in one prompt. (parameter=image)"),
        json.dumps({"scene_type": "其他或不确定", "detail": {"visible_details": []}}),
        json.dumps({"caption": "一人在公园步行", "representative_indices": [0]}),
    ])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        result = client.analyze_video_event([f"frame-{index}.webp" for index in range(5)])

    assert [len(call.args[1]) for call in client.chat.call_args_list] == [5, 2, 1]
    assert result["caption"] == "一人在公园步行"
    assert result["representative_indices"] == [0]
    assert result["video_event_fallback_reason"] == "single_image_recovery"


def test_video_event_rejects_persistently_empty_observations():
    client = _client()
    empty = json.dumps({"scene_type": "其他或不确定", "detail": {"visible_details": []}})
    client.chat = Mock(side_effect=[empty, empty, empty])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        try:
            client.analyze_video_event(["frame-0.webp", "frame-1.webp"])
        except ModelError as error:
            assert "no observable content" in str(error)
        else:
            raise AssertionError("Empty video observations must not be indexed as processed")
    assert client.chat.call_count == 3


def test_video_event_maps_semantic_states_and_transitions_after_image_limit():
    client = _client()
    client.chat = Mock(side_effect=[
        ModelError("At most 2 image(s) may be provided in one prompt."),
        json.dumps({
            "caption": "饮水后开始讲话",
            "distinct_states": [
                {"activity_class": "饮水", "representative_index": 0},
                {"activity_class": "讲话", "representative_index": 1},
            ],
            "state_transitions": [{
                "from_activity_class": "饮水", "to_activity_class": "讲话",
                "from_index": 0, "to_index": 1,
            }],
        }),
    ])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        result = client.analyze_video_event([f"frame-{index}.webp" for index in range(5)])

    assert result["representative_indices"] == [0, 4]
    assert [state["representative_index"] for state in result["distinct_states"]] == [0, 4]
    assert result["state_transitions"][0]["from_index"] == 0
    assert result["state_transitions"][0]["to_index"] == 4


def test_video_event_context_compaction_keeps_source_index_consistent():
    client = _client()
    client.chat = Mock(side_effect=[
        ModelError("maximum context length exceeded"),
        json.dumps({"caption": "一人饮水", "representative_indices": [0]}),
    ])
    with patch.object(client, "_encode_core_image", return_value=("encoded", "image/webp")):
        result = client.analyze_video_event([f"frame-{index}.webp" for index in range(5)])

    assert [len(call.args[1]) for call in client.chat.call_args_list] == [5, 1]
    assert result["representative_indices"] == [0]
    assert result["video_event_evidence_indices"] == [0]
    assert result["state_transitions"] == []
    assert result["video_event_fallback_reason"] == "context_budget_compaction_retry"
