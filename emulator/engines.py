"""The language model, in this process, through the litert_lm library.

One engine (LiteRTLMEngine) and the chat it opens (LiteRTChat). Replies stream
as events: {"type": "content", "text": ...} for a piece of the reply, and
{"type": "tool_call", "name": ..., "arguments": {...}} when the model calls a
tool — which is surfaced, never executed here: the robot runs its tools
(demo/conversation.py).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from emulator import models

logger = logging.getLogger(__name__)

# CPU thread count for LiteRTLMEngine's GPU->CPU fallback, verified on the
# laptop for gemma-4-E2B. The litert_lm C API otherwise defaults to a single
# thread, a measured regression.
LITERTLM_CPU_THREADS = 8

# Pictures one turn may carry, and what the Engine is built for. A day's
# answer is at most three frames, and this
# leaves one spare. Measured on the Mac: with max_num_images=4, gemma-4-E2B
# describes three pictures sent as ONE message in 2.1s.
# It is a hard ceiling, not a hint — over it litert_lm refuses the turn
# ("Provided more images than expected in the prompt"), which is why anything
# above this cap is dropped rather than sent, see _jpegs_for_turn. Unused
# capacity costs
# nothing measurable: a text-only turn on a vision-enabled engine is the same
# 0.06s as one on an engine built without vision.
MAX_IMAGES_PER_TURN = 4

# Replies are one or two short spoken sentences; this caps a runaway one.
DEFAULT_MAX_OUTPUT_TOKENS = 64


def _text_from_content(content: list[dict[str, Any]] | None) -> str:
    """Concatenate the text parts of a litert_lm `content` list. A turn that
    ends on a tool call can legitimately have no content at all — that's
    coerced to "", not treated as a contract violation."""
    if not content:
        return ""
    return "".join(part.get("text", "") for part in content
                   if isinstance(part, dict) and part.get("type") == "text")


# gemma sometimes opens a tool call and never closes its arguments —
# `<|tool_call>call:look{<tool_call|>` — and litert_lm's parser then RAISES out
# of the stream ("Failed to parse tool calls from code block") instead of
# returning the call. Measured: once within ~40 routing questions, on `look`,
# the tool that takes no arguments; `dance` in the motion pass has the same
# shape. The call is plain in the raw text the error carries, so it is
# recovered from there rather than failing a turn the model got right.
# Also without the opening brace — `call:recall query:<|"|>...<|"|>}` came back
# in a routing run and failed the same way.
_BROKEN_TOOL_CALL = re.compile(
    r"call:(?P<name>[A-Za-z_]\w*)\s*\{?(?P<body>.*?)(?:\}|<tool_call\|>|$)", re.DOTALL)
_BROKEN_TOOL_ARG = re.compile(r'(?P<key>\w+):(?:<\|"\|>)?(?P<value>[^<}]*)')


def _recover_tool_call(exc: Exception) -> dict | None:
    """The tool_call event a parse failure was carrying, or None if `exc` is
    anything else."""
    message = str(exc)
    if "Failed to parse tool calls" not in message:
        return None
    match = _BROKEN_TOOL_CALL.search(message.split("full response:", 1)[-1])
    if not match:
        return None
    arguments = {arg.group("key"): arg.group("value").strip()
                 for arg in _BROKEN_TOOL_ARG.finditer(match.group("body"))
                 if arg.group("value").strip()}
    return {"type": "tool_call", "name": match.group("name"),
            "arguments": arguments}


def _jpegs_for_turn(image: bytes | Sequence[bytes] | None,
                    limit: int) -> tuple[bytes, ...]:
    """The pictures one turn carries, capped at what the engine was built for.

    A bare `bytes` is ONE picture, never a sequence of ints.

    Past `limit` the extras are dropped with a warning instead of raising:
    litert_lm refuses the whole turn ("Provided more images than expected in
    the prompt"), and a day answered from the first few frames beats a turn
    that fails in front of the room.
    """
    if image is None:
        return ()
    if isinstance(image, (bytes, bytearray, memoryview)):
        return (bytes(image),)
    jpegs = tuple(image)
    if len(jpegs) > limit:
        logger.warning("a turn carried %d pictures but this engine holds %d: "
                       "dropping the last %d", len(jpegs), limit,
                       len(jpegs) - limit)
        return jpegs[:limit]
    return jpegs


def _user_message(litert_lm, text: str, jpegs: Sequence[bytes]) -> Any:
    """One user turn with its pictures in front of its words. This exact
    shape — Message.user(Contents.of(ImageBytes..., Text)) — is what the
    vision executor requires; a plain string with the picture described in
    words is not the same as attaching it. Several ImageBytes in one Contents
    is how a turn carries several pictures: measured on the Mac, gemma-4-E2B
    describes three that way in 2.1s."""
    parts = [litert_lm.Content.ImageBytes(jpeg) for jpeg in jpegs]
    parts.append(litert_lm.Content.Text(text))
    return litert_lm.Message.user(litert_lm.Contents.of(*parts))


def _events_from_chunk(chunk: dict[str, Any]) -> Iterator[dict]:
    """One streamed litert_lm chunk as reply_stream events (see the module
    docstring): every tool call it carries, then its text. No accumulation
    across chunks — litert_lm delivers each tool_calls list fully formed in
    one chunk, not as OpenAI-style incremental argument deltas."""
    for call in chunk.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            logger.warning(
                "reply_stream: dropping a tool_call with no name: %r", call)
            continue
        yield {"type": "tool_call", "name": name,
               "arguments": function.get("arguments") or {}}
    text = _text_from_content(chunk.get("content"))
    if text:
        yield {"type": "content", "text": text}


class LiteRTLMEngine:
    """The model in this process — no server, no HTTP round trip.

    `litert_lm` is imported inside __init__, not at module top: it is a heavy
    native dependency, and a test or a tool that never builds an engine
    should not pay for loading it.
    """

    def __init__(self, model_path: Path | str,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> None:
        import litert_lm

        self._litert_lm = litert_lm
        self._max_output_tokens = max_output_tokens
        model_path = str(model_path)

        # GPU is the fast path; construction is the only failure mode this
        # falls back on (a genuinely optional acceleration, not a silently
        # swallowed bug — see the class docstring). A failure DURING
        # generation on an already-constructed GPU engine is NOT retried
        # here: that would mean redoing an unknown amount of already-spent
        # work on a guess, and a real inference failure should surface
        # loudly rather than be masked by a silent, slower retry.
        #
        # Vision is enabled on the engine ALWAYS, not lazily on the first
        # image turn: without max_num_images/vision_backend, generation
        # raises "Vision executor should not be null" the moment an image is
        # attached, and building a second Engine on demand would mean
        # redoing the GPU->CPU fallback and doubling the model's memory
        # footprint for the life of that second instance. There's no
        # measured cost to doing it unconditionally either way — a
        # text-only turn on a vision-enabled engine is the same 0.06s as
        # one without — so always-on is strictly simpler with no downside.
        # max_num_images is the hard ceiling for the life of the engine (a
        # turn carrying more is refused outright), hence MAX_IMAGES_PER_TURN
        # rather than the 1 a single remembered frame needs.
        self._max_images = MAX_IMAGES_PER_TURN
        try:
            self._engine = litert_lm.Engine(
                model_path=model_path, backend=litert_lm.Backend.GPU(),
                max_num_images=MAX_IMAGES_PER_TURN,
                vision_backend=litert_lm.Backend.GPU())
        except (RuntimeError, OSError):
            logger.warning(
                "LiteRTLMEngine: GPU backend failed for %s, falling back to "
                "CPU (thread_count=%d)", model_path, LITERTLM_CPU_THREADS,
                exc_info=True)
            self._engine = litert_lm.Engine(
                model_path=model_path,
                backend=litert_lm.Backend.CPU(
                    thread_count=LITERTLM_CPU_THREADS),
                max_num_images=MAX_IMAGES_PER_TURN,
                vision_backend=litert_lm.Backend.CPU(
                    thread_count=LITERTLM_CPU_THREADS))

        # Wraps one OpenAI-style tool schema dict (demo/chat_session.py) as a
        # litert_lm Tool so it can be registered on a conversation. Defined
        # here, not at module scope, because it must subclass the real
        # litert_lm.Tool ABC — litert_lm.create_conversation only treats a
        # registered tool as a Tool if isinstance(t, interfaces.Tool) holds;
        # otherwise it tries to wrap it as a plain callable via
        # tool_from_function, which would fail on a schema dict.
        # execute() is never called: every conversation here sets
        # automatic_tool_calling=False, so litert_lm surfaces ToolCalls to
        # the caller instead of invoking them itself.
        class _SchemaTool(litert_lm.Tool):
            def __init__(self, schema: dict[str, Any]) -> None:
                self._schema = schema

            def get_tool_description(self) -> dict[str, Any]:
                return self._schema

            def execute(self, param: dict[str, Any]) -> Any:
                raise NotImplementedError(
                    "_SchemaTool.execute should never run — tool calls are "
                    "surfaced, not invoked")

        self._schema_tool_cls = _SchemaTool

    def _message_for(self, prompt: str,
                     image: bytes | Sequence[bytes] | None) -> Any:
        """A bare string for the text-only turn (unchanged, cheapest path —
        exactly what every existing caller already sends); a litert_lm
        Message when one picture is attached, or several (_user_message)."""
        jpegs = _jpegs_for_turn(image, self._max_images)
        if not jpegs:
            return prompt
        return _user_message(self._litert_lm, prompt, jpegs)

    def reply_stream(self, prompt: str, system: str,
                     tools: list[dict] | None = None,
                     image: bytes | Sequence[bytes] | None = None) -> Iterator[dict]:
        """One reply in a fresh conversation, as events (see the module
        docstring). For single questions outside the chat — reading a name
        out of an answer (demo/serve.py's /name), warming the model up.

        automatic_tool_calling=False: with the litert_lm default (True), a
        registered tool would be executed in-library and its result fed back
        to the model; here a call comes back to the caller instead.
        """
        engine_tools = ([self._schema_tool_cls(t) for t in tools]
                        if tools else None)
        conversation = self._engine.create_conversation(
            system_message=system,
            tools=engine_tools,
            automatic_tool_calling=False,
            max_output_tokens=self._max_output_tokens)
        try:
            for chunk in conversation.send_message_async(
                    self._message_for(prompt, image)):
                yield from _events_from_chunk(chunk)
        except RuntimeError as exc:
            recovered = _recover_tool_call(exc)
            if recovered is None:
                raise
            logger.warning("reply_stream: recovered a tool call litert_lm "
                           "could not parse: %r", recovered)
            yield recovered
        finally:
            conversation.close()

    def open_chat(self, system: str, history=(),
                  tools: list[dict] | None = None) -> "LiteRTChat":
        """A conversation that stays open across turns (see LiteRTChat),
        where reply_stream() opens a fresh one per call and closes it.

        `history` — (person, reply) pairs already said, oldest first — is
        replayed as ordinary user/model turns: that is how a chat is rebuilt
        from the robot's copy after older turns moved into its memory. Free
        here: litert_lm prefills a seeded history on the first send, not on
        creation (measured: token_count stays 0 until then). Tool calls are
        surfaced, never executed (automatic_tool_calling=False), exactly as in
        reply_stream — the robot runs them.
        """
        litert_lm = self._litert_lm
        messages = []
        for person, reply in history:
            messages.append(litert_lm.Message.user(litert_lm.Contents.of(person)))
            messages.append(litert_lm.Message.model(litert_lm.Contents.of(reply)))
        conversation = self._engine.create_conversation(
            system_message=system,
            messages=messages or None,
            tools=[self._schema_tool_cls(t) for t in tools] if tools else None,
            automatic_tool_calling=False,
            max_output_tokens=self._max_output_tokens)
        return LiteRTChat(litert_lm, conversation, self._max_images)


class LiteRTChat:
    """One litert_lm conversation kept open across turns — the chat pattern
    flutter_gemma's InferenceChat is built on.

    Kept open, the model holds every earlier turn in its KV cache: measured on
    gemma-4-E2B, each message after the first costs only its own tokens
    (0.11-0.14s) and "what's my name?" is answered from three turns back with
    no retrieval at all. The context is finite — 4096 tokens for this model,
    refused past that ("Input token ids are too long") — so its owner watches
    token_count and rebuilds the chat with fewer turns (demo/chat_session.py,
    demo/conversation.py).
    """

    def __init__(self, litert_lm, conversation,
                 max_images: int = MAX_IMAGES_PER_TURN) -> None:
        self._litert_lm = litert_lm
        self._conversation = conversation
        # What the Engine behind this conversation was built for: a turn
        # carrying more pictures than that is refused outright, so
        # send_with_image caps against it (_jpegs_for_turn).
        self._max_images = max_images
        # Set when a turn ended in a tool call recovered from a parse failure
        # (_recover_tool_call): what the conversation recorded of that turn is
        # unknown, so its owner rebuilds it rather than continue in it.
        self.needs_rebuild = False

    def send(self, text: str) -> Iterator[dict]:
        """The person's next message; reply_stream's event contract."""
        yield from self._stream(text)

    def send_tool_result(self, name: str, result: Any) -> Iterator[dict]:
        """Answer the tool call the previous message ended on; the model
        continues from it."""
        yield from self._stream({"role": "tool", "content": [
            {"type": "tool_response", "name": name, "response": result}]})

    def send_with_image(self, text: str,
                        jpegs: bytes | Sequence[bytes]) -> Iterator[dict]:
        """A message with a picture attached, or several — the shape
        _message_for sends. One JPEG and a list of one are the same thing, so
        a caller that shows a single frame is unchanged.

        Several go in ONE message, never one message each: that is what a
        day's answer is, and the model described three of them in 2.1s that
        way.

        Only in a chat opened WITHOUT tools: an image in a chat that carries
        tools is refused or answered with another tool call (measured, see
        demo/chat_session.py)."""
        yield from self._stream(_user_message(
            self._litert_lm, text,
            _jpegs_for_turn(jpegs, self._max_images)))

    @property
    def token_count(self) -> int:
        """Tokens the context holds: system prompt, tool schemas, every turn."""
        if self._conversation is None:
            raise RuntimeError("chat is closed")
        return self._conversation.token_count

    def close(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
            self._conversation = None

    def _stream(self, message) -> Iterator[dict]:
        if self._conversation is None:
            raise RuntimeError("chat is closed")
        try:
            for chunk in self._conversation.send_message_async(message):
                yield from _events_from_chunk(chunk)
        except RuntimeError as exc:
            recovered = _recover_tool_call(exc)
            if recovered is None:
                raise
            logger.warning("chat: recovered a tool call litert_lm could not "
                           "parse: %r", recovered)
            self.needs_rebuild = True
            yield recovered


def build_llm(spec: models.Model) -> "LiteRTLMEngine":
    """The engine for a catalog entry (emulator/models.py)."""
    return LiteRTLMEngine(models.fetch(spec))
