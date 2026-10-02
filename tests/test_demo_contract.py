import base64
import json

import numpy as np
import pytest

from demo.contract import (
    decode_chat_request,
    decode_transcribe_request,
    decode_transcribe_response,
    encode_chat_request,
    encode_transcribe_request,
    encode_transcribe_response,
)


# --- /transcribe ---

def test_transcribe_request_roundtrip():
    audio = np.linspace(-1, 1, 4000, dtype=np.float32)
    payload = encode_transcribe_request(audio, 16000)
    got_audio, sr = decode_transcribe_request(payload)
    assert sr == 16000
    assert np.allclose(got_audio, audio, atol=1e-4)


def test_transcribe_response_roundtrip():
    payload = encode_transcribe_response("what time is it")
    assert decode_transcribe_response(payload) == "what time is it"


def test_transcribe_response_defaults_to_empty_string():
    assert decode_transcribe_response({}) == ""


def test_transcribe_request_is_json_safe():
    payload = encode_transcribe_request(np.zeros(10, np.float32), 16000)
    json.dumps(payload)  # must not raise


# --- /chat ---

def _over_the_wire(payload):
    return decode_chat_request(json.loads(json.dumps(payload)))


def test_chat_request_round_trips_history_text_and_a_tool_result():
    request = _over_the_wire(encode_chat_request(
        [("hi", "hello")], "What's my name?",
        tool_result={"name": "recall", "result": {"memories": ["m"]}}))
    assert request.history == [("hi", "hello")]
    assert request.text == "What's my name?"
    assert request.tool_result == {"name": "recall", "result": {"memories": ["m"]}}
    assert request.image_jpeg is None


def test_chat_request_round_trips_an_image_and_its_note():
    # One picture is a list of one: the callers that attach a single frame
    # (demo/conversation.py) are unchanged by the day's answer carrying three.
    request = _over_the_wire(encode_chat_request(
        [], "What do you see?", image_jpeg=b"\xff\xd8JPEG",
        image_note="(Your camera, right now.)"))
    assert request.image_jpegs == (b"\xff\xd8JPEG",)
    assert request.image_jpeg == b"\xff\xd8JPEG"
    assert request.image_note == "(Your camera, right now.)"
    assert request.tool_result is None


def test_chat_request_round_trips_several_images_under_one_note():
    # A day's answer: up to three frames in ONE turn, one note for all of
    # them.
    request = _over_the_wire(encode_chat_request(
        [], "What did you see today?", image_jpeg=[b"ONE", b"TWO", b"THREE"],
        image_note="(These are your memories of today.)"))
    assert request.image_jpegs == (b"ONE", b"TWO", b"THREE")
    assert request.image_jpeg == b"ONE"
    assert request.image_note == "(These are your memories of today.)"


def test_a_request_with_no_pictures_carries_none():
    request = _over_the_wire(encode_chat_request([], "hi", image_jpeg=[]))
    assert request.image_jpegs == () and request.image_jpeg is None


@pytest.mark.parametrize("payload", [
    ["not", "an", "object"],
    {},
    {"text": "   "},
    {"text": "q", "history": [["only one side"]]},
    {"text": "q", "history": [[1, 2]]},
    {"text": "q", "tool_result": {"result": "no name"}},
    {"text": "q", "images_b64": "not a list"},
    {"text": "q", "images_b64": [base64.b64encode(b"J").decode(), 7]},
    {"text": "q", "images_b64": ["not base64 !!"]},
    {"text": "q", "tool_result": {"name": "recall"},
     "images_b64": [base64.b64encode(b"J").decode()]},
])
def test_decode_chat_request_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        decode_chat_request(payload)


# --- "send me the words, I will make the sound" (synthesis on the robot) ---

def test_a_chat_request_asks_for_sentences_only_when_told_to():
    plain = encode_chat_request([], "hello")
    assert "sentences" not in plain
    assert decode_chat_request(plain).sentences is False

    asked = encode_chat_request([], "hello", sentences=True)
    assert asked["sentences"] is True
    assert decode_chat_request(asked).sentences is True
