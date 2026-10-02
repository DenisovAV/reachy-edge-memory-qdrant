"""The language model engine (LiteRTLMEngine, in-process via the litert_lm
library), its open chat (LiteRTChat), and build_llm.

LiteRTLMEngine is tested entirely against a fake `litert_lm` module injected
into sys.modules — the real library needs a multi-GB model file on disk, and
CI has neither the file nor the point in loading it just to check that
events come out in the right shape.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from emulator import models
from emulator.engines import (
    MAX_IMAGES_PER_TURN,
    LiteRTLMEngine,
    build_llm,
)


class _FakeCPU:
    def __init__(self, thread_count=None):
        self.thread_count = thread_count


class _FakeGPU:
    def __init__(self):
        pass


class _FakeBackendNamespace:
    CPU = _FakeCPU
    GPU = _FakeGPU


class _FakeTool:
    """Stands in for litert_lm.Tool — a plain subclassable base, since the
    fake module doesn't need to enforce the real ABC."""


class _FakeContentPart:
    """Stands in for one litert_lm.Content.* value — just enough shape
    (kind + value) for a test to assert what LiteRTLMEngine attached,
    without modelling the real Content classes."""

    def __init__(self, kind: str, value) -> None:
        self.kind = kind
        self.value = value


class _FakeContentNamespace:
    @staticmethod
    def ImageBytes(data):
        return _FakeContentPart("image", data)

    @staticmethod
    def Text(text):
        return _FakeContentPart("text", text)


class _FakeContents:
    """Stands in for litert_lm.Contents — a plain ordered bag of parts."""

    def __init__(self, parts) -> None:
        self.parts = list(parts)

    @staticmethod
    def of(*parts):
        return _FakeContents(parts)


class _FakeMessage:
    """Stands in for litert_lm.Message — records the role and the Contents
    it was built with, so a test can assert the exact shape LiteRTLMEngine
    sends to send_message/send_message_async."""

    def __init__(self, role: str, contents: _FakeContents) -> None:
        self.role = role
        self.contents = contents

    @staticmethod
    def user(contents):
        return _FakeMessage("user", contents)

    @staticmethod
    def model(contents):
        return _FakeMessage("model", contents)


class _FakeConversation:
    token_count = 42

    def __init__(self, chunks=(), reply_response=None, **kwargs):
        self.kwargs = kwargs
        self._chunks = list(chunks)
        self._reply_response = reply_response
        self.closed = False
        self.sent_prompts = []

    def send_message(self, prompt):
        self.sent_prompts.append(prompt)
        return self._reply_response

    def send_message_async(self, prompt):
        self.sent_prompts.append(prompt)
        return iter(self._chunks)

    def close(self):
        self.closed = True


class _FakeEngine:
    """Records every conversation it creates so tests can inspect the
    kwargs LiteRTLMEngine passed through (system_message, tools,
    automatic_tool_calling, ...)."""

    fail_on_gpu = False

    def __init__(self, model_path, backend, max_num_images=None,
                vision_backend=None):
        if self.fail_on_gpu and isinstance(backend, _FakeGPU):
            raise RuntimeError("simulated GPU backend failure")
        self.model_path = model_path
        self.backend = backend
        self.max_num_images = max_num_images
        self.vision_backend = vision_backend
        self.conversations: list[_FakeConversation] = []
        self.closed = False

    def create_conversation(self, **kwargs):
        conv = _FakeConversation(
            chunks=self.next_chunks, reply_response=self.next_reply,
            **kwargs)
        self.conversations.append(conv)
        return conv

    def close(self):
        self.closed = True


@pytest.fixture
def fake_litert_lm(monkeypatch):
    """Installs a fake `litert_lm` module so LiteRTLMEngine's lazy `import
    litert_lm` picks it up, with no real library/model involved."""
    _FakeEngine.fail_on_gpu = False
    _FakeEngine.next_chunks = []
    _FakeEngine.next_reply = None
    module = types.ModuleType("litert_lm")
    module.Engine = _FakeEngine
    module.Backend = _FakeBackendNamespace
    module.Tool = _FakeTool
    module.Content = _FakeContentNamespace
    module.Contents = _FakeContents
    module.Message = _FakeMessage
    monkeypatch.setitem(sys.modules, "litert_lm", module)
    return module


def test_litertlm_engine_constructs_with_gpu_backend_by_default(fake_litert_lm):
    engine = LiteRTLMEngine("model.litertlm")
    assert isinstance(engine._engine.backend, _FakeGPU)


def test_litertlm_engine_constructs_with_vision_enabled(fake_litert_lm):
    # Without max_num_images/vision_backend, generation raises "Vision
    # executor should not be null" the moment an image is attached — this
    # must be set on construction, not deferred to the first image turn.
    engine = LiteRTLMEngine("model.litertlm")
    assert engine._engine.max_num_images == MAX_IMAGES_PER_TURN
    assert isinstance(engine._engine.vision_backend, _FakeGPU)


def test_litertlm_engine_falls_back_to_cpu_when_gpu_construction_fails(
        fake_litert_lm, caplog):
    _FakeEngine.fail_on_gpu = True
    with caplog.at_level("WARNING"):
        engine = LiteRTLMEngine("model.litertlm")
    assert isinstance(engine._engine.backend, _FakeCPU)
    assert engine._engine.backend.thread_count == 8
    # The vision backend must fall back to CPU right alongside the main one
    # — a GPU-only vision_backend on a CPU-only main engine would be an
    # inconsistent half-fallback.
    assert isinstance(engine._engine.vision_backend, _FakeCPU)
    assert engine._engine.vision_backend.thread_count == 8
    assert "falling back to CPU" in caplog.text


def test_litertlm_engine_reply_stream_yields_content_chunks_in_order(
        fake_litert_lm):
    _FakeEngine.next_chunks = [
        {"content": [{"type": "text", "text": "Hello"}]},
        {"content": [{"type": "text", "text": " there"}]},
    ]
    engine = LiteRTLMEngine("model.litertlm")
    assert list(engine.reply_stream("hi", system="s")) == [
        {"type": "content", "text": "Hello"},
        {"type": "content", "text": " there"},
    ]


def test_litertlm_engine_reply_stream_surfaces_tool_calls_not_executed(
        fake_litert_lm):
    # demo/serve.py needs the RAW call (e.g. move_head) — LiteRTLMEngine must
    # not let litert_lm auto-invoke it, only report it as an event.
    _FakeEngine.next_chunks = [
        {"tool_calls": [
            {"function": {"name": "move_head", "arguments": {"gesture": "nod"}}}]},
    ]
    engine = LiteRTLMEngine("model.litertlm")
    move_head_tool = {
        "type": "function",
        "function": {"name": "move_head", "parameters": {}},
    }
    events = list(engine.reply_stream("nod please", system="s", tools=[move_head_tool]))
    assert events == [
        {"type": "tool_call", "name": "move_head", "arguments": {"gesture": "nod"}},
    ]
    conv = engine._engine.conversations[-1]
    assert conv.kwargs["automatic_tool_calling"] is False
    registered_tools = conv.kwargs["tools"]
    assert len(registered_tools) == 1
    assert registered_tools[0].get_tool_description() == move_head_tool


def test_litertlm_engine_reply_stream_drops_and_logs_a_nameless_tool_call(
        fake_litert_lm, caplog):
    _FakeEngine.next_chunks = [{"tool_calls": [{"function": {"arguments": {}}}]}]
    engine = LiteRTLMEngine("model.litertlm")
    with caplog.at_level("WARNING"):
        events = list(engine.reply_stream("hi", system="s"))
    assert events == []
    assert "dropping a tool_call with no name" in caplog.text


def test_litertlm_engine_reply_stream_closes_the_conversation(fake_litert_lm):
    _FakeEngine.next_chunks = [{"content": [{"type": "text", "text": "hi"}]}]
    engine = LiteRTLMEngine("model.litertlm")
    list(engine.reply_stream("hi", system="s"))
    assert engine._engine.conversations[-1].closed


def test_litertlm_engine_reply_stream_attaches_an_image_as_a_message(
        fake_litert_lm):
    _FakeEngine.next_chunks = [{"content": [{"type": "text", "text": "a red mug"}]}]
    engine = LiteRTLMEngine("model.litertlm")
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg"
    list(engine.reply_stream("what did you see?", system="s", image=jpeg))
    sent = engine._engine.conversations[-1].sent_prompts[-1]
    assert isinstance(sent, _FakeMessage)
    kinds = [(part.kind, part.value) for part in sent.contents.parts]
    assert kinds == [("image", jpeg), ("text", "what did you see?")]


def test_litertlm_engine_attaches_several_images_in_one_message(fake_litert_lm):
    # A day's answer is up to three frames and they go in ONE turn: measured
    # on the Mac, gemma-4-E2B describes three that way in 2.1s.
    _FakeEngine.next_chunks = [{"content": [{"type": "text", "text": "three rooms"}]}]
    engine = LiteRTLMEngine("model.litertlm")
    jpegs = [b"ONE", b"TWO", b"THREE"]
    list(engine.reply_stream("what did you see today?", system="s", image=jpegs))
    sent = engine._engine.conversations[-1].sent_prompts[-1]
    assert [(part.kind, part.value) for part in sent.contents.parts] == [
        ("image", b"ONE"), ("image", b"TWO"), ("image", b"THREE"),
        ("text", "what did you see today?")]


def test_more_images_than_the_engine_holds_are_capped_and_said_out_loud(
        fake_litert_lm, caplog):
    # litert_lm refuses the whole turn past max_num_images ("Provided more
    # images than expected in the prompt"), so the extras go — loudly.
    _FakeEngine.next_chunks = [{"content": [{"type": "text", "text": "ok"}]}]
    engine = LiteRTLMEngine("model.litertlm")
    too_many = [bytes([n]) for n in range(MAX_IMAGES_PER_TURN + 2)]
    with caplog.at_level(logging.WARNING, logger="emulator.engines"):
        list(engine.reply_stream("what did you see?", system="s", image=too_many))
    sent = engine._engine.conversations[-1].sent_prompts[-1]
    images = [part.value for part in sent.contents.parts if part.kind == "image"]
    assert images == too_many[:MAX_IMAGES_PER_TURN]
    assert "dropping" in caplog.text


# --- build_llm ---

def test_build_llm_fetches_the_model_and_builds_the_engine(monkeypatch):
    calls = []
    monkeypatch.setattr("emulator.models.fetch", lambda spec: "resolved.litertlm")

    class _RecordingEngine:
        def __init__(self, model_path):
            calls.append(model_path)

    monkeypatch.setattr("emulator.engines.LiteRTLMEngine", _RecordingEngine)
    engine = build_llm(models.Model(repo="r", file="f"))
    assert isinstance(engine, _RecordingEngine)
    assert calls == ["resolved.litertlm"]


# --- LiteRTChat: one conversation kept open across turns ---

_RECALL_SCHEMA = {"type": "function", "function": {
    "name": "recall", "description": "d",
    "parameters": {"type": "object", "properties": {}}}}


def _broken_tool_call(message=("INVALID_ARGUMENT: Failed to parse tool calls "
                               "from code block: call:look{\nfull response: "
                               "<|tool_call>call:look{<tool_call|>")):
    raise RuntimeError(message)
    yield  # pragma: no cover — makes this a generator that fails on first use


def test_open_chat_seeds_the_history_as_user_and_model_turns(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    engine.open_chat("sys", [("hi", "hello")], [_RECALL_SCHEMA])
    conv = engine._engine.conversations[-1]
    assert conv.kwargs["system_message"] == "sys"
    assert [(m.role, m.contents.parts) for m in conv.kwargs["messages"]] == [
        ("user", ["hi"]), ("model", ["hello"])]
    assert conv.kwargs["automatic_tool_calling"] is False
    assert [t.get_tool_description()["function"]["name"]
            for t in conv.kwargs["tools"]] == ["recall"]


def test_open_chat_without_history_or_tools_passes_none(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    engine.open_chat("sys")
    conv = engine._engine.conversations[-1]
    assert conv.kwargs["messages"] is None and conv.kwargs["tools"] is None


def test_a_chat_stays_open_across_sends_until_closed(fake_litert_lm):
    _FakeEngine.next_chunks = [{"content": [{"type": "text", "text": "Hi"}]}]
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    assert list(chat.send("a")) == [{"type": "content", "text": "Hi"}]
    list(chat.send("b"))
    conv = engine._engine.conversations[-1]
    assert conv.sent_prompts == ["a", "b"] and not conv.closed
    assert chat.token_count == 42
    chat.close()
    assert conv.closed


def test_a_chat_answers_a_tool_call_with_a_tool_response_message(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    list(chat.send_tool_result("recall", {"memories": ["m"]}))
    assert engine._engine.conversations[-1].sent_prompts[-1] == {
        "role": "tool", "content": [{"type": "tool_response", "name": "recall",
                                     "response": {"memories": ["m"]}}]}


def test_a_chat_message_can_carry_a_picture(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    list(chat.send_with_image("what is this?", b"JPEG"))
    message = engine._engine.conversations[-1].sent_prompts[-1]
    assert message.role == "user"
    assert [(p.kind, p.value) for p in message.contents.parts] == [
        ("image", b"JPEG"), ("text", "what is this?")]


def test_a_chat_message_can_carry_several_pictures(fake_litert_lm):
    # The whole day in one message, not one message per frame.
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    list(chat.send_with_image("what did you see?", [b"ONE", b"TWO"]))
    message = engine._engine.conversations[-1].sent_prompts[-1]
    assert [(p.kind, p.value) for p in message.contents.parts] == [
        ("image", b"ONE"), ("image", b"TWO"), ("text", "what did you see?")]


def test_a_chat_caps_the_pictures_at_what_the_engine_holds(fake_litert_lm, caplog):
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    with caplog.at_level(logging.WARNING, logger="emulator.engines"):
        list(chat.send_with_image("what did you see?",
                                  [bytes([n]) for n in range(MAX_IMAGES_PER_TURN + 1)]))
    message = engine._engine.conversations[-1].sent_prompts[-1]
    images = [p.value for p in message.contents.parts if p.kind == "image"]
    assert len(images) == MAX_IMAGES_PER_TURN
    assert "dropping" in caplog.text


def test_a_chat_surfaces_tool_calls_as_events(fake_litert_lm):
    _FakeEngine.next_chunks = [{"tool_calls": [{"function": {
        "name": "recall", "arguments": {"query": "name"}}}]}]
    engine = LiteRTLMEngine("m.litertlm")
    assert list(engine.open_chat("sys").send("name?")) == [
        {"type": "tool_call", "name": "recall", "arguments": {"query": "name"}}]


def test_a_tool_call_litert_lm_cannot_parse_is_recovered_in_a_chat(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    chat._conversation.send_message_async = lambda message: _broken_tool_call()
    assert list(chat.send("look")) == [
        {"type": "tool_call", "name": "look", "arguments": {}}]
    assert chat.needs_rebuild


def test_a_tool_call_litert_lm_cannot_parse_is_recovered_in_reply_stream(
        fake_litert_lm, monkeypatch):
    monkeypatch.setattr(_FakeConversation, "send_message_async",
                        lambda self, message: _broken_tool_call(
                            "Failed to parse tool calls\nfull response: "
                            '<|tool_call>call:express{emotion:<|"|>happy<|"|>'))
    engine = LiteRTLMEngine("m.litertlm")
    assert list(engine.reply_stream("x", system="s", tools=[_RECALL_SCHEMA])) == [
        {"type": "tool_call", "name": "express", "arguments": {"emotion": "happy"}}]


def test_other_runtime_errors_still_raise(fake_litert_lm):
    engine = LiteRTLMEngine("m.litertlm")
    chat = engine.open_chat("sys")
    chat._conversation.send_message_async = lambda message: _broken_tool_call("engine died")
    with pytest.raises(RuntimeError, match="engine died"):
        list(chat.send("hi"))


def test_a_closed_chat_refuses_to_send(fake_litert_lm):
    chat = LiteRTLMEngine("m.litertlm").open_chat("sys")
    chat.close()
    with pytest.raises(RuntimeError):
        list(chat.send("hi"))


@pytest.mark.parametrize("raw,expected", [
    ('<|tool_call>call:look{<tool_call|>', {"name": "look", "arguments": {}}),
    ('<|tool_call>call:recall{query:<|"|>my name<|"|>}<tool_call|>',
     {"name": "recall", "arguments": {"query": "my name"}}),
    # No opening brace at all — seen in a routing run.
    ('<|tool_call>call:recall query:<|"|>what we talked about<|"|>}<tool_call|>',
     {"name": "recall", "arguments": {"query": "what we talked about"}}),
])
def test_broken_tool_calls_are_recovered_in_every_shape_seen(raw, expected):
    from emulator.engines import _recover_tool_call

    exc = RuntimeError("INVALID_ARGUMENT: Failed to parse tool calls from code "
                       f"block\nfull response: {raw}")
    assert _recover_tool_call(exc) == {"type": "tool_call", **expected}
