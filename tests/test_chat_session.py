"""demo/chat_session.py — the Mac's live chat, kept as a cache of the robot's
conversation: reused while it matches, rebuilt from the robot's copy when not,
and never shown a picture while it carries tools."""
import pytest

from demo.chat_session import CHAT_SYSTEM, CHAT_TOOLS, IMAGE_SYSTEM, ChatSession
from demo.contract import ChatRequest


class _Chat:
    def __init__(self, llm, system, history, tools):
        self.system, self.history, self.tools = system, history, tools
        self.sent = []
        self.closed = False
        self.token_count = 0
        self.needs_rebuild = False
        self._llm = llm

    def _reply(self, message):
        self.sent.append(message)
        self.token_count += 10
        return iter(self._llm.script.pop(0))

    def send(self, text):
        return self._reply(("text", text))

    def send_tool_result(self, name, result):
        return self._reply(("tool", name, result))

    def send_with_image(self, text, jpegs):
        return self._reply(("image", text, tuple(jpegs)))

    def close(self):
        self.closed = True


class _LLM:
    """Every open_chat call gets a _Chat; each send takes the next scripted
    reply from `script`, shared across chats in order."""

    def __init__(self, *script):
        self.script = list(script)
        self.chats = []

    def open_chat(self, system, history, tools):
        chat = _Chat(self, system, history, tools)
        self.chats.append(chat)
        return chat


def SAY(text):
    return [{"type": "content", "text": text}]


def CALL(name, **arguments):
    return [{"type": "tool_call", "name": name, "arguments": arguments}]


def _run(session, request):
    """What demo/serve.py's chat_respond does with a session: stream, then
    report how the turn ended."""
    events = list(session.turn(request))
    calls = [e for e in events if e["type"] == "tool_call"]
    text = "".join(e["text"] for e in events if e["type"] == "content")
    tool_call = ({"name": calls[0]["name"], "arguments": calls[0]["arguments"]}
                 if calls else None)
    session.finish(request, text, tool_call)
    return text, tool_call


def test_a_day_of_frames_gets_its_own_system_message():
    """One remembered frame is IMAGE_SYSTEM; several are the day being read
    back, and that needs DAY_SYSTEM — measured, the same words in a note let
    the robot answer "I saw a man looking intently in the first image"
   ."""
    from demo.chat_session import DAY_SYSTEM, IMAGE_SYSTEM

    llm = _LLM(SAY("I saw three things."), SAY("A bottle."))
    session = ChatSession(llm)
    list(session.turn(ChatRequest([], "What did you see today?",
                                  image_jpegs=(b"ONE", b"TWO", b"THREE"))))
    assert llm.chats[-1].system == DAY_SYSTEM
    assert llm.chats[-1].tools is None, "a picture never travels with tools"
    list(session.turn(ChatRequest([], "What is this?", image_jpegs=(b"ONE",))))
    assert llm.chats[-1].system == IMAGE_SYSTEM


def test_the_first_turn_opens_a_chat_with_the_robot_history_and_both_tools():
    llm = _LLM(SAY("Hello!"))
    _run(ChatSession(llm), ChatRequest(history=[("a", "b")], text="hi"))
    chat = llm.chats[0]
    assert (chat.system, chat.history, chat.tools) == (CHAT_SYSTEM, [("a", "b")],
                                                       CHAT_TOOLS)
    assert chat.sent == [("text", "hi")]


def test_a_matching_history_reuses_the_live_chat():
    llm = _LLM(SAY("Hello!"), SAY("Sasha."))
    session = ChatSession(llm)
    _run(session, ChatRequest([], "I'm Sasha"))
    _run(session, ChatRequest([("I'm Sasha", "Hello!")], "What's my name?"))
    assert len(llm.chats) == 1
    assert llm.chats[0].sent[-1] == ("text", "What's my name?")


def test_a_different_history_rebuilds_the_chat_from_the_robot_copy():
    # The robot moved "I'm Sasha" into Qdrant: its history no longer has it.
    llm = _LLM(SAY("Hello!"), SAY("Hm?"))
    session = ChatSession(llm)
    _run(session, ChatRequest([], "I'm Sasha"))
    _run(session, ChatRequest([], "What's my name?"))
    assert len(llm.chats) == 2
    assert llm.chats[0].closed
    assert llm.chats[1].history == []


def test_token_count_is_what_the_chat_holds_after_the_turn():
    llm = _LLM(SAY("Hello!"))
    session = ChatSession(llm)
    _run(session, ChatRequest([], "hi"))
    assert session.token_count == 10


def test_a_tool_result_continues_the_chat_that_asked_for_it():
    llm = _LLM(CALL("recall", query="name"), SAY("Your name is Sasha."), SAY("ok"))
    session = ChatSession(llm)
    _, call = _run(session, ChatRequest([], "What's my name?"))
    assert call == {"name": "recall", "arguments": {"query": "name"}}
    text, _ = _run(session, ChatRequest([], "What's my name?", tool_result={
        "name": "recall", "result": {"memories": ["m"]}}))
    assert text == "Your name is Sasha."
    assert llm.chats[0].sent[-1] == ("tool", "recall", {"memories": ["m"]})
    # The exchange is held now, so the next turn reuses the same chat.
    _run(session, ChatRequest([("What's my name?", "Your name is Sasha.")], "thanks"))
    assert len(llm.chats) == 1


def test_a_tool_result_for_a_lost_chat_re_asks_and_hands_over_the_result():
    llm = _LLM(CALL("recall", query="name"), SAY("Sasha."))
    session = ChatSession(llm)  # e.g. serve.py restarted between the requests
    text, _ = _run(session, ChatRequest([], "name?", tool_result={
        "name": "recall", "result": {"memories": ["m"]}}))
    assert text == "Sasha."
    assert llm.chats[0].sent == [("text", "name?"),
                                 ("tool", "recall", {"memories": ["m"]})]


def test_an_image_turn_runs_in_a_tool_free_side_chat_and_drops_the_main_one():
    llm = _LLM(CALL("camera"), SAY("A red mug."), SAY("Red."))
    session = ChatSession(llm)
    _run(session, ChatRequest([], "What do you see?"))
    text, _ = _run(session, ChatRequest([], "What do you see?", image_jpegs=(b"J",),
                                        image_note="(Your camera, right now.)"))
    main, side = llm.chats
    assert text == "A red mug."
    assert (side.system, side.tools) == (IMAGE_SYSTEM, None)
    assert side.sent == [("image", "(Your camera, right now.) What do you see?",
                          (b"J",))]
    assert main.closed and side.closed
    assert session.token_count is None
    # The next turn rebuilds the main chat from the robot's history — which
    # now includes the image turn's answer.
    _run(session, ChatRequest([("What do you see?", "A red mug.")], "What color?"))
    assert llm.chats[2].history == [("What do you see?", "A red mug.")]
    assert llm.chats[2].tools == CHAT_TOOLS


def test_a_days_frames_reach_the_side_chat_as_one_message():
    # All of them in ONE turn, under one
    # note that names no times and no people — not one message per frame.
    llm = _LLM(SAY("I saw a window, a desk and a bottle."))
    session = ChatSession(llm)
    note = "(These are your memories of today.)"
    text, _ = _run(session, ChatRequest(
        [], "What did you see today?", image_jpegs=(b"ONE", b"TWO", b"THREE"),
        image_note=note))
    assert text == "I saw a window, a desk and a bottle."
    side = llm.chats[0]
    assert side.tools is None  # a picture may not enter a chat that has tools
    assert side.sent == [("image", f"{note} What did you see today?",
                          (b"ONE", b"TWO", b"THREE"))]


def test_an_empty_reply_drops_the_chat():
    llm = _LLM(SAY(""), SAY("hi"))
    session = ChatSession(llm)
    _run(session, ChatRequest([], "hello?"))
    assert llm.chats[0].closed
    _run(session, ChatRequest([], "hello?"))
    assert len(llm.chats) == 2


def test_a_turn_that_fails_midway_drops_the_chat():
    def failing():
        yield {"type": "content", "text": "par"}
        raise RuntimeError("engine died")

    llm = _LLM(failing())
    session = ChatSession(llm)
    with pytest.raises(RuntimeError):
        list(session.turn(ChatRequest([], "hi")))
    assert llm.chats[0].closed


def test_abandoning_a_reply_drops_the_chat():
    llm = _LLM(SAY("a") + SAY("b"))
    session = ChatSession(llm)
    events = session.turn(ChatRequest([], "hi"))
    next(events)
    events.close()  # the robot hung up mid-reply
    assert llm.chats[0].closed


def test_a_recovered_parse_failure_forces_a_rebuild():
    llm = _LLM(CALL("camera"))
    session = ChatSession(llm)
    list(session.turn(ChatRequest([], "look")))
    llm.chats[0].needs_rebuild = True  # emulator/engines.py's recovery path
    session.finish(ChatRequest([], "look"), "", {"name": "camera", "arguments": {}})
    assert llm.chats[0].closed
    assert session.token_count is None
