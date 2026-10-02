"""demo/conversation.py — the robot's side of the chat: the window the model
holds in context, the move of its oldest part into memory, the tools, and
the request sequence of one turn."""
import base64

import pytest

from demo.contract import decode_chat_request
from demo.conversation import (DEFAULT_CONTEXT_BUDGET, LOOK_NOTE,
                               ConversationWindow, Recalled, chat_turn,
                               lookup, memory_note, recall,
                               recall_seen)
from emulator.memory import EXCHANGE_KIND


class _Memory:
    """TextMemory stand-in: records what is stored, scores in-context texts
    from a table, returns scripted stored hits."""

    def __init__(self, scores=None, stored_hits=(), fail=False):
        self.remembered = []
        self._scores = scores or {}
        self._stored_hits = list(stored_hits)
        self._fail = fail

    def remember(self, text, kind, meta=None):
        if self._fail:
            raise OSError("embed service down")
        self.remembered.append((text, kind, meta))

    def recall_exchanges(self, query, k=3):
        return list(self._stored_hits)

    def score_texts(self, query, texts):
        return [self._scores.get(text, 0.0) for text in texts]


def _window(memory=None, **kwargs):
    ticks = iter(range(100, 10_000))
    return ConversationWindow(memory, clock=lambda: next(ticks), **kwargs)


def _filled(window, n):
    for i in range(n):
        window.add(f"q{i}", f"a{i}")
    return window


# — the window —

def test_history_is_the_exchanges_in_order():
    window = _filled(_window(), 2)
    assert window.history == [("q0", "a0"), ("q1", "a1")]


def test_under_budget_nothing_moves():
    memory = _Memory()
    window = _filled(_window(memory, budget_tokens=500), 4)
    assert window.after_turn(499) == []
    assert memory.remembered == []
    assert len(window.history) == 4


def test_over_budget_the_oldest_half_moves_into_memory_as_exchanges():
    memory = _Memory()
    window = _filled(_window(memory, budget_tokens=500), 6)
    stored = window.after_turn(501)
    assert stored == ["Person: q0 — Reachy: a0", "Person: q1 — Reachy: a1",
                      "Person: q2 — Reachy: a2"]
    assert [kind for _, kind, _ in memory.remembered] == [EXCHANGE_KIND] * 3
    assert memory.remembered[0][2] == {"said_at": 100}
    assert window.history == [("q3", "a3"), ("q4", "a4"), ("q5", "a5")]


def test_an_unknown_token_count_moves_nothing():
    # An image turn runs in a side chat and reports no count.
    window = _filled(_window(_Memory(), budget_tokens=10), 4)
    assert window.after_turn(None) == []
    assert len(window.history) == 4


def test_a_single_exchange_is_never_evicted():
    window = _filled(_window(_Memory(), budget_tokens=10), 1)
    assert window.after_turn(10_000) == []
    assert len(window.history) == 1


def test_the_cap_evicts_even_without_a_token_count():
    memory = _Memory()
    window = _filled(_window(memory, max_exchanges=3), 4)
    assert window.after_turn(None) == ["Person: q0 — Reachy: a0",
                                       "Person: q1 — Reachy: a1"]
    assert len(window.history) == 2


def test_a_failed_store_keeps_the_exchanges_in_context():
    window = _filled(_window(_Memory(fail=True), budget_tokens=500), 4)
    assert window.after_turn(900) == []
    assert len(window.history) == 4


def test_a_failed_store_over_the_cap_drops_rather_than_grows():
    window = _filled(_window(_Memory(fail=True), max_exchanges=3), 4)
    assert window.after_turn(None) == []  # nothing was stored
    assert len(window.history) == 2


def test_without_memory_eviction_still_bounds_the_context():
    window = _filled(_window(None, budget_tokens=500), 4)
    assert window.after_turn(900) == []
    assert len(window.history) == 2


def test_flush_moves_everything_still_in_context():
    memory = _Memory()
    window = _filled(_window(memory), 2)
    assert len(window.flush()) == 2
    assert window.history == []
    assert len(memory.remembered) == 2


def test_search_finds_in_context_exchanges_over_the_gate():
    said = "Sasha: Hi, I'm Sasha. — Reachy: Nice to meet you, Sasha!"
    window = _window(_Memory(scores={said: 0.7}))
    window.add("Hi, I'm Sasha.", "Nice to meet you, Sasha!")
    window.add("Can you nod?", "Sure!")
    assert window.search("what's my name?", 0.62) == [
        {"text": said, "score": 0.7, "source": "context"}]


# — the recall tool —

class _Frames:
    """FrameMemory stand-in: `hits` answer a search, `looks` are the frames
    taken on request (newest first, as latest_looks returns them)."""

    def __init__(self, hits, looks=(), words=()):
        self.hits = hits
        self.looks = list(looks)
        self.words = list(words)      # what the frame-words search answers
        self.min_scores = []
        self.looks_asked = []
        self.word_queries = []

    def recall(self, query, min_score=None):
        self.min_scores.append(min_score)
        return list(self.hits)

    def recall_text(self, query, k=3, before=None):
        self.word_queries.append(query)
        return list(self.words)

    def latest_looks(self, directions=None, limit=4, before=None):
        self.looks_asked.append((directions, limit, before))
        return [look for look in self.looks
                if (directions is None or look.get("looked") in directions)
                and (before is None or look["ts"] < before)][:limit]










# — one turn —

class _Display:
    def __init__(self):
        self.events = []

    def on_tool_call(self, name, arguments):
        self.events.append(("tool", name, arguments))

    def on_memory_write(self, texts):
        self.events.append(("memory", texts))

    def on_recall(self, hits):
        self.events.append(("frames", hits))

    def on_look(self, jpeg):
        self.events.append(("look", jpeg))

    def on_speech_recall(self, hits):
        self.events.append(("speech", hits))

    def on_context(self, tokens, budget, exchanges):
        self.events.append(("context", tokens, budget, exchanges))

    def on_memory_count(self, frames, exchanges):
        self.events.append(("count", frames, exchanges))


class _Brain:
    """Answers each /chat request with the next scripted done event, and
    keeps every request it was sent, decoded."""

    def __init__(self, *dones):
        self._dones = list(dones)
        self.requests = []

    def __call__(self, payload):
        self.requests.append(decode_chat_request(payload))
        return self._dones.pop(0)


def _turn(heard, brain, *, window=None, recall_fn=None, recall_seen_fn=None,
          camera_jpeg=None,
          display=None, **kwargs):
    return chat_turn(heard, window=window if window is not None else _window(),
                     send=brain, recall_fn=recall_fn,
                     recall_seen_fn=recall_seen_fn, camera_jpeg=camera_jpeg,
                     display=display or _Display(), **kwargs)


def test_a_plain_reply_is_one_request_and_joins_the_window():
    window = _window()
    brain = _Brain({"reply": "Hello Sasha!", "token_count": 300})
    done = _turn("Hi, I'm Sasha.", brain, window=window)
    assert done["reply"] == "Hello Sasha!"
    assert [(r.history, r.text) for r in brain.requests] == [([], "Hi, I'm Sasha.")]
    assert window.history == [("Hi, I'm Sasha.", "Hello Sasha!")]


def test_the_window_rides_along_with_every_request():
    window = _filled(_window(), 1)
    brain = _Brain({"reply": "c", "token_count": 1})
    _turn("q", brain, window=window)
    assert brain.requests[0].history == [("q0", "a0")]


def test_look_answers_with_the_camera_picture():
    window, display = _window(), _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}, "reply": ""},
                   {"reply": "I see a mug.", "token_count": None})
    _turn("What do you see?", brain, window=window, display=display,
          camera_jpeg=lambda: b"JPEG")
    second = brain.requests[1]
    assert (second.text, second.image_jpeg, second.image_note) == (
        "What do you see?", b"JPEG", LOOK_NOTE)
    assert ("tool", "camera", {}) in display.events
    assert window.history == [], "a look stays out of the model's conversation"


def test_look_without_a_picture_tells_the_model_so():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "My camera is dark.", "token_count": 1})
    _turn("What do you see?", brain, camera_jpeg=lambda: None)
    result = brain.requests[1].tool_result
    assert result["name"] == "camera" and "error" in result["result"]






def test_recall_with_no_query_searches_what_was_said():
    queries = []
    brain = _Brain({"tool_call": {"name": "recall", "arguments": {}}},
                   {"reply": "ok", "token_count": 1})
    _turn("What did I show you?", brain,
          recall_fn=lambda query: (queries.append(query), Recalled([], [], []))[1])
    assert queries == ["What did I show you?"]






def test_a_model_that_keeps_calling_tools_is_cut_off():
    window = _window()
    brain = _Brain(*[{"tool_call": {"name": "camera", "arguments": {}}}] * 4)
    _turn("look", brain, window=window, camera_jpeg=lambda: b"J",
          max_tool_rounds=2)
    # Three rounds of tools, then ONE request that asks for words — and no
    # more, whatever the model does with it.
    assert len(brain.requests) == 4
    assert window.history == []


def test_a_turn_out_of_tool_rounds_still_says_something():
    """It used to return the tool call itself, whose reply is empty: the robot
    stood there saying nothing for the whole turn. A bounded chain is right;
    silence in front of a room is not."""
    from demo.conversation import NO_MORE_TOOLS

    window = _window()
    brain = _Brain(*([{"tool_call": {"name": "remember", "arguments": {"query": "x"}}}] * 3
                     + [{"reply": "We talked about Qdrant.", "token_count": 1}]))
    _turn("What did we talk about?", brain, window=window, max_tool_rounds=2,
          recall_fn=lambda query: Recalled([], [], recent=["Person: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None, pictures=True: [])
    assert NO_MORE_TOOLS in brain.requests[-1].tool_result["result"]["note"]
    assert window.history == [("What did we talk about?", "We talked about Qdrant.")]


def test_an_empty_reply_is_not_remembered():
    window = _window()
    _turn("hello?", _Brain({"reply": "", "token_count": 1}), window=window)
    assert window.history == []


def test_eviction_is_reported_to_the_dashboard():
    memory, display = _Memory(), _Display()
    window = _filled(_window(memory, budget_tokens=100), 2)
    _turn("q", _Brain({"reply": "r", "token_count": 200}), window=window,
          display=display)
    assert ("memory", ["Person: q0 — Reachy: a0"]) in display.events


def test_the_camera_tool_also_answers_to_its_old_name():
    brain = _Brain({"tool_call": {"name": "look", "arguments": {}}},
                   {"reply": "A wall.", "token_count": None})
    _turn("What do you see?", brain, camera_jpeg=lambda: b"JPEG")
    assert brain.requests[1].image_jpeg == b"JPEG"




















@pytest.mark.parametrize("heard", ["What do you see right now?", "Look at this.",
                                   "What am I holding?", "What's in front of you?"])
def test_a_camera_call_about_the_present_stays_a_camera_call(heard):
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "ok", "token_count": 1})
    _turn(heard, brain, camera_jpeg=lambda: b"NOW",
          recall_fn=lambda query: pytest.fail("recall must not run"))
    assert brain.requests[1].image_jpeg == b"NOW"


def test_the_window_exposes_the_budget_the_dashboard_draws_against():
    assert _window(budget_tokens=321).budget == 321


def test_a_turn_reports_how_full_the_context_is():
    display = _Display()
    window = _window(_Memory(), budget_tokens=500)
    _turn("hi", _Brain({"reply": "hello", "token_count": 310}), window=window,
          display=display)
    assert ("context", 310, 500, 1) in display.events


def test_an_image_turn_reports_no_token_count():
    # It ran in a side chat, which has its own context — the bar must stay
    # where it was rather than draw a drop that did not happen.
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "A wall.", "token_count": None})
    _turn("What do you see?", brain, display=display, camera_jpeg=lambda: b"J")
    assert ("context", None, DEFAULT_CONTEXT_BUDGET, 0) in display.events


def test_the_context_is_reported_after_the_eviction_it_caused():
    display, memory = _Display(), _Memory()
    window = _filled(_window(memory, budget_tokens=100), 2)
    _turn("q", _Brain({"reply": "r", "token_count": 200}), window=window,
          display=display)
    # The count is the context BEFORE the eviction, the window is after it.
    assert ("context", 200, 100, 2) in display.events


@pytest.mark.parametrize("heard", ["What do you see right now?", "Look at this.",
                                   "What am I holding?"])
def test_a_real_camera_question_still_gets_the_picture(heard):
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "A mug.", "token_count": 1})
    _turn(heard, brain, camera_jpeg=lambda: b"NOW")
    assert brain.requests[1].image_jpeg == b"NOW"


def test_an_answer_read_out_of_memory_does_not_go_back_into_memory():
    # Measured live: storing them built a loop — "I recall we were discussing
    # something emotional" was stored, and the next "what did we discuss?"
    # recalled that sentence instead of the thing actually discussed.
    memory = _Memory()
    window = _window(memory, budget_tokens=100)
    window.add("Hi, I'm Sasha.", "Hello Sasha!")
    brain = _Brain({"tool_call": {"name": "recall", "arguments": {"query": "x"}}},
                   {"reply": "I recall we discussed something emotional.",
                    "token_count": 200})
    _turn("What did we discuss?", brain, window=window,
          recall_fn=lambda query: Recalled([], [], []))
    stored = [text for text, _kind, _meta in memory.remembered]
    assert stored == ["Sasha: Hi, I'm Sasha. — Reachy: Hello Sasha!"]
    assert "emotional" not in " ".join(stored)


def test_a_derived_exchange_leaves_the_window_without_being_stored():
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.add("What did we discuss?", "Something emotional.", derived=True)
    window.add("And then?", "More of it.", derived=True)
    assert window.after_turn(500) == []
    assert memory.remembered == []
    assert len(window.history) == 1


def test_a_plain_answer_still_goes_into_memory():
    memory = _Memory()
    window = _window(memory, budget_tokens=100)
    window.add("I'm giving a talk.", "Exciting!")
    window.add("Tell me a joke.", "Why did the robot cross the road?")
    assert window.after_turn(200) == ["Person: I'm giving a talk. — Reachy: Exciting!"]






# --- who the person is, until there is face recognition ---

@pytest.mark.parametrize("heard,name", [
    ("Hi Reachy, I'm Sasha.", "Sasha"),
    ("my name is Anna", "Anna"),
    ("Call me Sasha, please.", "Sasha"),
    ("this is Bob", "Bob"),
    ("I am Sasha", "Sasha"),
])
def test_a_self_introduction_gives_the_speaker_a_name(heard, name):
    from demo.conversation import speaker_name

    assert speaker_name(heard) == name


@pytest.mark.parametrize("heard", [
    "I'm giving a talk at Vector Space Stream.",
    "I'm not sure about that.",
    "I am here to talk about memory.",
    "What did you see earlier?",
    "i'm sasha",  # lower case: a name has to look like one
])
def test_ordinary_sentences_are_not_read_as_a_name(heard):
    from demo.conversation import speaker_name

    assert speaker_name(heard) is None


def test_the_name_replaces_Person_in_what_is_stored():
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.add("Hello there.", "Hi!")
    window.add("I'm Sasha, by the way.", "Nice to meet you, Sasha!")
    window.add("Tell me a joke.", "Why did the robot cross the road?")
    # The earlier exchange is relabelled too: it has not reached memory yet,
    # and it was the same person.
    assert window.after_turn(500) == [
        "Sasha: Hello there. — Reachy: Hi!"]
    assert [text for text, _kind, _meta in memory.remembered] == [
        "Sasha: Hello there. — Reachy: Hi!"]



def test_a_face_names_what_nobody_was_named_for_yet():
    # The person was talking before the camera knew them: the same person.
    window = _window(_Memory())
    window.add("Hello there.", "Hi!")
    window.set_speaker("Sasha")
    window.add("Tell me a joke.", "Why did the robot cross the road?")
    assert [e.speaker for e in window._exchanges] == ["Sasha", "Sasha"]


def test_a_new_face_does_not_take_the_last_persons_words():
    # Alice talks, steps away, Bob steps in and is recognised: what Alice said
    # is still in the window, and it must reach memory as hers.
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.set_speaker("Bob")
    window.add("What's the weather?", "I can't see outside.")
    assert [e.speaker for e in window._exchanges] == ["Alice", "Bob"]
    window.flush()
    assert [text for text, _kind, _meta in memory.remembered] == [
        "Alice: My dog is called Rex. — Reachy: Lovely name!",
        "Bob: What's the weather? — Reachy: I can't see outside."]


def test_someone_new_names_only_their_own_words_when_they_say_who_they_are():
    # Alice was recognised; a stranger steps in and talks, then says "I'm Bob".
    # His words become Bob's; Alice's stay hers.
    window = _window(_Memory())
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.someone_new()
    window.add("Hello there.", "Hi!")
    window.add("I'm Bob, by the way.", "Nice to meet you, Bob!")
    assert [e.speaker for e in window._exchanges] == ["Alice", "Bob", "Bob"]


def test_a_known_face_does_not_take_a_strangers_words():
    # The stranger leaves unnamed; Alice comes back and is recognised. What
    # the stranger said is not hers.
    window = _window(_Memory())
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.someone_new()
    window.add("I like trains.", "Me too!")
    window.set_speaker("Alice")
    window.add("Where were we?", "Your dog, Rex.")
    assert [e.text.split(":")[0] for e in window._exchanges] == ["Alice", "Person", "Alice"]


def test_a_stranger_who_said_their_name_keeps_it_while_they_stay():
    window = _window(_Memory())
    window.someone_new()
    window.add("Hi, I'm Bob.", "Nice to meet you, Bob!")
    window.add("What can you do?", "I remember things.")
    assert [e.speaker for e in window._exchanges] == ["Bob", "Bob"]


def test_an_answered_name_question_names_only_that_strangers_words():
    window = _window(_Memory())
    window.someone_new()
    window.add("I like trains.", "Me too!")            # the first stranger
    window.someone_new()
    window.add("Hello there.", "Hi!")                 # another one
    window.introduce("Bob")
    assert [e.speaker for e in window._exchanges] == [None, "Bob"]

# --- two memories: `recall` answers with words, `recall_seen` with a frame ---

def _frame(score, ts=0.0):
    return {"jpeg_b64": base64.b64encode(b"FRAME").decode(), "ts": ts, "score": score}


def _call(name, query="q", **arguments):
    return {"tool_call": {"name": name,
                          "arguments": {"query": query, **arguments}}}


def test_recall_keeps_memory_and_the_live_context_apart():
    # What is still in the window was not remembered — the model is looking
    # at it. Only Qdrant hits count as memory, for the model and for the room.
    stored = {"text": "Person: I have a dog called Rex. — Reachy: Lovely!", "score": 0.65}
    said = "Sasha: Hi, I'm Sasha. — Reachy: Nice to meet you, Sasha!"
    memory = _Memory(scores={said: 0.7}, stored_hits=[stored])
    window = _window(memory)
    window.add("Hi, I'm Sasha.", "Nice to meet you, Sasha!")
    found = recall("my name", window=window, speech_memory=memory)
    assert found.memories == [stored["text"]]
    assert [hit["source"] for hit in found.speech_hits] == ["qdrant"]
    assert found.in_context == [said]


def test_a_memory_that_never_opened_is_not_an_empty_one():
    # The memory shard could not open at start: told "Nothing in your memory
    # about this" all run, the robot denied everything it was asked about.
    from demo.conversation import MEMORY_OFF_NOTE

    brain = _Brain(_call("remember", "my dog", about="said"),
                   {"reply": "My memory is off right now.", "token_count": 1})
    _turn("Do you remember my dog?", brain, recall_fn=None, recall_seen_fn=None)
    assert brain.requests[1].tool_result["result"] == {"note": MEMORY_OFF_NOTE}

    # A memory that is there and has nothing still says so.
    brain = _Brain(_call("remember", "my dog", about="said"),
                   {"reply": "You never told me.", "token_count": 1})
    _turn("Do you remember my dog?", brain,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [])
    assert brain.requests[1].tool_result["result"] == {
        "note": "Nothing in your memory about this."}


def test_with_one_memory_off_the_question_is_answered_by_the_one_it_needs():
    # Frame memory can fail to open while the words open (or the other way
    # round): a question about the half that is there, and empty, still
    # gets "nothing"; a question about the half that is off gets "off".
    from demo.conversation import MEMORY_OFF_NOTE

    nothing = {"note": "Nothing in your memory about this."}
    no_words = lambda query: Recalled([], [])
    no_frames = lambda query, direction=None, pictures=True: []
    for about, question, words, frames, expected in (
            ("said", "Do you remember my dog?", no_words, None, nothing),
            ("seen", "Did you see my dog earlier?", None, no_frames, nothing),
            ("said", "Do you remember my dog?", None, no_frames, {"note": MEMORY_OFF_NOTE}),
            ("seen", "Did you see my dog earlier?", no_words, None, {"note": MEMORY_OFF_NOTE}),
            ("anything", "Do you remember my dog?", no_words, None, {"note": MEMORY_OFF_NOTE})):
        brain = _Brain(_call("remember", "my dog", about=about),
                       {"reply": "...", "token_count": 1})
        _turn(question, brain, recall_fn=words, recall_seen_fn=frames,
              day_frames_fn=lambda: [])
        assert brain.requests[1].tool_result["result"] == expected, (about, words, frames)


def test_with_memory_off_a_general_question_is_still_a_general_one():
    from demo.conversation import GENERAL_QUESTION_NOTE

    brain = _Brain(_call("remember", "airplanes", about="anything"),
                   {"reply": "Wings make lift.", "token_count": 1})
    _turn("How do airplanes fly?", brain, recall_fn=None, recall_seen_fn=None)
    assert brain.requests[1].tool_result["result"] == {"found": [], "note": GENERAL_QUESTION_NOTE}


def test_with_memory_off_the_facts_it_was_taught_still_answer():
    # The knowledge base is its own shard: it opens when the memory does not.
    fact = {"text": "Qdrant Edge runs inside the robot's own process.", "score": 0.8}
    brain = _Brain(_call("remember", "qdrant edge", about="taught"),
                   {"reply": "It runs in my own process.", "token_count": 1})
    _turn("Do you remember what Qdrant Edge is?", brain, recall_fn=None,
          recall_seen_fn=None, knowledge_fn=lambda query: [fact])
    result = brain.requests[1].tool_result["result"]
    assert result["facts_you_were_taught"] == [fact["text"]]
    assert "note" not in result


def test_a_memory_that_cannot_be_read_says_so_rather_than_finding_nothing():
    # "Nothing in your memory" for a store that could not be read had the
    # robot deny what it remembered.
    from demo.conversation import MemoryUnavailable

    class Down:
        def recall_exchanges(self, query, k=3):
            raise OSError("down")

        def score_texts(self, query, texts):
            raise OSError("down")

    down = Down()
    window = ConversationWindow(down)
    window.add("a", "b")
    with pytest.raises(MemoryUnavailable):
        recall("x", window=window, speech_memory=down)


def test_the_model_is_told_the_memory_could_not_be_searched():
    from demo.conversation import MEMORY_UNAVAILABLE_NOTE, MemoryUnavailable

    def down(query):
        raise MemoryUnavailable("exchanges: OSError: down")

    brain = _Brain(_call("remember", query="my name", about="said"),
                   {"reply": "My memory is not answering right now.", "token_count": 1})
    memory = _Memory()
    window = _window(memory)
    _turn("What's my name?", brain, window=window, recall_fn=down)
    assert brain.requests[1].tool_result["result"] == {"note": MEMORY_UNAVAILABLE_NOTE}
    # Nothing came out of memory, so nothing would be copied back into it:
    # what the person said is kept.
    assert not window._exchanges[-1].derived


def test_recall_seen_takes_the_best_frames_whatever_they_scored():
    frames = _Frames([_frame(0.05)])
    assert recall_seen("what did you see?", frame_memory=frames) == [_frame(0.05)]
    assert frames.min_scores == [0.0]


def test_recall_seen_drops_frames_stored_during_this_turn():
    frames = _Frames([_frame(0.12, ts=50.0), _frame(0.10, ts=10.0)])
    found = recall_seen("what did you see?", frame_memory=frames, turn_started_at=40.0)
    assert [frame["ts"] for frame in found] == [10.0]


def test_recall_seen_reports_a_failing_store():
    from demo.conversation import MemoryUnavailable

    class Down:
        def recall_text(self, query, before=None):
            raise OSError("down")

    with pytest.raises(MemoryUnavailable):
        recall_seen("x", frame_memory=Down())
    assert recall_seen("x", frame_memory=None) == []


def test_memory_note_says_it_is_a_memory_and_how_old():
    # "(What you saw N days ago.)" still got "I see a room…" in 2 answers of 4;
    # saying plainly that it is a memory got 0 of 4.
    note = memory_note({"ts": 1000.0}, 1300.0)
    assert "MEMORY" in note and "5 minutes ago" in note and "past tense" in note
    assert "2 days ago" in memory_note({"ts": 0.0}, 2 * 86400.0)


def test_the_projector_shows_confident_frames_plainly_and_a_weak_one_marked():
    display = _Display()
    strong = _frame(0.12)
    _turn("What did you see?", _Brain(_call("recall_seen"), {"reply": "ok", "token_count": 1}),
          display=display, recall_seen_fn=lambda query, direction=None, pictures=True: [strong, _frame(0.05)])
    assert ("frames", [{**strong, "weak": False}]) in display.events
    display = _Display()
    weak = _frame(0.05)
    _turn("What did you see?", _Brain(_call("recall_seen"), {"reply": "ok", "token_count": 1}),
          display=display, recall_seen_fn=lambda query, direction=None, pictures=True: [weak])
    assert ("frames", [{**weak, "weak": True}]) in display.events


def test_a_turn_that_used_either_memory_is_not_stored_back():
    for tool, kwargs in (("recall", {"recall_fn": lambda query: Recalled([], [])}),
                         ("recall_seen", {"recall_seen_fn": lambda query, direction=None, pictures=True: []})):
        memory = _Memory()
        window = _window(memory, budget_tokens=10)
        window.add("Hi.", "Hello!")
        _turn("What happened?", _Brain(_call(tool), {"reply": "Something.", "token_count": 500}),
              window=window, **kwargs)
        assert [text for text, _k, _m in memory.remembered] == ["Person: Hi. — Reachy: Hello!"]


# — the knowledge base —

def test_lookup_reports_a_failing_knowledge_base():
    from demo.conversation import MemoryUnavailable

    class _Broken:
        def search(self, query, k=3, min_score=None):
            raise OSError("embed service down")

    with pytest.raises(MemoryUnavailable):
        lookup("what is Qdrant Edge?", knowledge=_Broken())
    assert lookup("anything", knowledge=None) == []


# — looking around —

@pytest.mark.parametrize("direction", ["left", "right"])
def test_a_camera_call_with_a_direction_turns_the_head_and_says_so(direction):
    looks, display = [], _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": direction}}},
                   {"reply": "A plant.", "token_count": None})
    _turn(f"Look to your {direction}. What do you see?", brain, display=display,
          camera_jpeg=lambda: pytest.fail("the picture from before the turn"),
          look_fn=lambda d: (looks.append(d), b"TURNED")[1])
    second = brain.requests[1]
    assert looks == [direction]
    assert (second.image_jpeg, second.image_note) == (
        b"TURNED", "(Your camera, right now.)")  # the side is kept with the frame, not said in the note
    assert ("tool", "camera", {"direction": direction}) in display.events


@pytest.mark.parametrize("arguments", [{}, {"direction": "ahead"}, {"direction": "up"}])
def test_a_camera_call_straight_ahead_looks_ahead(arguments):
    looks = []
    brain = _Brain({"tool_call": {"name": "camera", "arguments": arguments}},
                   {"reply": "A mug.", "token_count": None})
    _turn("What do you see?", brain, camera_jpeg=lambda: pytest.fail("the looker has it"),
          look_fn=lambda d: (looks.append(d), b"NOW")[1])
    assert looks == ["ahead"]
    assert (brain.requests[1].image_jpeg, brain.requests[1].image_note) == (b"NOW", LOOK_NOTE)


def test_a_head_that_could_not_turn_is_told_to_the_model():
    from demo.conversation import LookFailed

    def look(direction):
        raise LookFailed("your head could not turn to your left")

    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I can't turn my head right now.", "token_count": None})
    _turn("Look to your left.", brain, display=display, look_fn=look)
    second = brain.requests[1]
    assert second.image_jpeg is None
    assert second.tool_result["result"] == {"error": "your head could not turn to your left"}
    assert not [event for event in display.events if event[0] == "look"], \
        "no picture on the screen either"


def test_without_a_head_to_turn_the_camera_still_answers():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "A mug.", "token_count": None})
    _turn("Look to your left.", brain, camera_jpeg=lambda: b"NOW")
    assert brain.requests[1].image_jpeg == b"NOW"


def _look(where, ts, caption=None):
    return {**_frame(0.0, ts=ts), "looked": where, **({"caption": caption} if caption else {})}


def test_recall_seen_by_direction_is_the_last_look_that_way_before_this_turn():
    frames = _Frames([_frame(0.9)], looks=[_look("left", 50.0), _look("right", 40.0),
                                            _look("left", 30.0)])
    found = recall_seen("what was on your left?", frame_memory=frames,
                        turn_started_at=45.0, direction="left")
    assert [frame["ts"] for frame in found] == [30.0]
    assert frames.looks_asked == [(["left"], 1, 45.0)]
    assert frames.min_scores == [], "no search: the words say nothing SigLIP can match"


def test_recall_seen_in_general_is_the_described_looks():
    frames = _Frames([_frame(0.9)], looks=[_look("right", 40.0, "I see a door."),
                                            _look("ahead", 35.0),
                                            _look("left", 30.0, "I see a lamp.")])
    found = recall_seen("what did you see?", frame_memory=frames)
    assert [frame["caption"] for frame in found] == ["I see a door.", "I see a lamp."]
    assert frames.min_scores == []


def test_recall_seen_without_described_looks_searches_as_before():
    frames = _Frames([_frame(0.9)], looks=[_look("left", 30.0)])
    assert recall_seen("the mug", frame_memory=frames) == [_frame(0.9)]


def test_a_question_about_a_thing_searches_the_frames_own_words_first():
    """Measured on the robot's 371 stored frames: "did you see a bottle?",
    "what was on the table?", "a plant in a pot" — the picture search found 8
    right frames of 36, searching the words the frame already carries found
    32. The words go first; the picture is what answers when they find
    nothing."""
    bottle = _frame(0.8, ts=900.0)
    frames = _Frames([_frame(0.2)], looks=[_look("left", 30.0, "I see a lamp.")],
                     words=[bottle])
    assert recall_seen("did you see a bottle?", frame_memory=frames) == [bottle]
    assert frames.word_queries == ["did you see a bottle?"]
    # …and nothing else was asked: neither the looks nor the picture search.
    assert frames.min_scores == [] and frames.looks_asked == []


def test_a_look_line_keeps_the_first_sentence_without_i_see():
    from demo.conversation import _look_line

    frame = _look("ahead", 940.0, "I see a man with a mug. He is on the right side of the image.")
    assert _look_line(frame, 1000.0) == "In front of me, a minute ago: a man with a mug."
    assert _look_line({**frame, "caption": "A door"}, 1000.0) == "In front of me, a minute ago: A door"


def test_a_look_stays_out_of_the_conversation_the_model_sees():
    window = _window()
    window.add("Hi.", "Hello!")
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I see a lamp.", "token_count": None})
    _turn("Look to your left.", brain, window=window, look_fn=lambda d: b"J")
    assert window.history == [("Hi.", "Hello!")]


def test_the_last_look_one_way_is_shown_plainly():
    display = _Display()
    left = _look("left", 50.0)
    brain = _Brain({"tool_call": {"name": "recall_seen",
                                  "arguments": {"query": "left", "direction": "left"}}},
                   {"reply": "I saw a lamp.", "token_count": None})
    _turn("What was on your left?", brain, display=display,
          recall_seen_fn=lambda query, direction=None, pictures=True: [left])
    shown = [event for event in display.events if event[0] == "frames"][-1][1]
    assert [frame["weak"] for frame in shown] == [False]


# — who —

class _People:
    enabled = True

    def __init__(self, here=(), met=()):
        self._here = list(here)
        self._met = list(met)

    def faces_in(self, frame):
        return list(self._here)

    def met(self):
        return list(self._met)


class _SeenPeople:
    def __init__(self, seen):
        self.seen = seen
        self.asked = []

    def people_seen(self, before=None):
        self.asked.append(before)
        return list(self.seen)


def test_who_names_who_is_here_who_was_seen_and_who_was_met():
    from demo.conversation import who

    people = _People(here=[{"name": "Sasha"}, {"name": None}], met=["Sasha", "Robin"])
    frames = _SeenPeople([("Robin", 880.0)])
    result = who(people=people, frame=object(), frame_memory=frames,
                 turn_started_at=990.0, clock=lambda: 1000.0)
    assert result == {"in_front_of_you": ["Sasha"],
                      "people_you_have_not_met_in_front_of_you": 1,
                      "seen_earlier": ["Robin, 2 minutes ago"],
                      "you_last_saw_them": [],
                      "people_you_have_met": ["Sasha", "Robin"]}
    assert frames.asked == [990.0]


def test_who_with_nobody_there_or_no_face_models_says_so():
    from demo.conversation import who

    assert who(people=_People(), frame=object())["note"] == \
        "Nobody is in front of your camera right now."
    assert who() == {"faces_off": True, "note": "You cannot recognise faces right now."}


def test_who_without_a_picture_cannot_tell_who_is_there():
    # The camera gave no frame this turn: that is not an empty room.
    from demo.conversation import who

    result = who(people=_People(here=[{"name": "Sasha"}], met=["Sasha"]), frame=None)
    assert result["faces_off"] is True
    assert result["note"] == "You cannot recognise faces right now."


def test_a_look_line_says_who_was_there():
    from demo.conversation import _look_line

    frame = {**_look("ahead", 940.0, "I see a man with a mug."),
             "people": [{"name": "Sasha"}, {"name": None}]}
    assert _look_line(frame, 1000.0) == "In front of me, a minute ago: a man with a mug. (Sasha was there.)"


# — what a live run broke —

def test_a_look_is_neither_in_the_chat_nor_in_the_conversation_memory():
    """A look is stored once, as the frame's caption (demo/run_demo.py's
    Looker.caption). Written into the conversation as an exchange too, it
    had "what did we talk about?" answered "we talked about what I saw in the
    room" (live)."""
    memory = _Memory()
    window = _window(memory)
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I see a lamp.", "token_count": None})
    _turn("Look to your left.", brain, window=window, display=display, look_fn=lambda d: b"J")
    assert window.history == []
    assert memory.remembered == []
    assert not [e for e in display.events if e[0] == "memory"]


# — the body moves because the model asked for it —

def test_the_move_tool_moves_the_body():
    moves = []
    brain = _Brain({"tool_call": {"name": "move", "arguments": {"how": "happy"}}},
                   {"reply": "Yes! I'm happy.", "token_count": 1})
    _turn("Show me your emotions.", brain, move_fn=moves.append,
          camera_jpeg=lambda: pytest.fail("a request to move is not a picture"))
    second = brain.requests[1]
    assert moves == ["happy"]
    assert second.tool_result == {"name": "move", "result": {"moved": "happy"}}


def test_a_move_without_a_body_still_answers():
    brain = _Brain({"tool_call": {"name": "move", "arguments": {"how": "dance"}}},
                   {"reply": "Dancing!", "token_count": 1})
    _turn("Dance for me!", brain)
    assert brain.requests[1].tool_result["result"] == {"moved": "dance"}


def test_the_move_tool_is_offered_with_the_moves_the_body_has():
    from demo.chat_session import CHAT_TOOLS, MOVES
    from demo.robot_reachy import EMOTION_MOVES, GESTURE_POSES

    move = next(t for t in CHAT_TOOLS if t["function"]["name"] == "move")
    assert move["function"]["parameters"]["properties"]["how"]["enum"] == MOVES
    for how in MOVES:
        assert how in GESTURE_POSES or how in EMOTION_MOVES or how == "dance"


# — one memory tool: the search decides which store answers —

def _remember(brain, **kwargs):
    return _turn("What do you remember?", brain, **kwargs)


def test_words_beat_pictures():
    # Told about a database, the robot used to answer with a photo of the room.
    told = "Sasha: Qdrant is a database. — Reachy: Got it!"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}])
    display = _Display()
    brain = _Brain(_call("remember", "Qdrant"),
                   {"reply": "You said Qdrant is a database.", "token_count": 1})
    _turn("What did I tell you about Qdrant?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: found,
          knowledge_fn=lambda query: [{"text": "Qdrant is a vector database.",
                                       "score": 0.8, "source": "knowledge"}],
          recall_seen_fn=lambda query, direction=None, pictures=True: [_look("ahead", 10.0, "I see a room.")])
    result = brain.requests[1].tool_result["result"]
    assert brain.requests[1].image_jpeg is None
    assert result["facts_you_were_taught"] == ["Qdrant is a vector database."]
    assert result["the_person_told_you_before"] == [told]
    # The look comes along IN WORDS, in the same answer — one search, every
    # source that had something. What it must not do is arrive as a picture
    # instead of the sentence the person actually said. A fixed clock, not the
    # real one: the frame's ts=10.0 read against wall time made this "N days
    # ago" tick over and fail a day after it was written.
    assert result["you_looked_at"] == ["In front of me, 16 minutes ago: a room."]


def test_a_turned_look_is_labelled_with_who_is_in_THAT_picture():
    """Live: asked what was on its left, the robot described the window and
    then said "Sasha is in front of me now" — the note came from the frame the
    turn started with, taken while it was still facing Sasha."""
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I saw a window.", "token_count": 1})
    _turn("What is on your left?", brain,
          look_fn=lambda direction: b"LEFT",
          look_names_fn=lambda: [],           # nobody is in the turned picture
          names_fn=lambda: ["Sasha"])         # …but Sasha is in front of the robot
    note = brain.requests[1].image_note
    assert "Sasha" not in note
    assert note == "(Your camera, right now.)"  # the side stays with the frame


def test_a_look_that_does_have_someone_in_it_still_names_them():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "right"}}},
                   {"reply": "I saw Masha.", "token_count": 1})
    _turn("Look to your right.", brain,
          look_fn=lambda direction: b"RIGHT",
          look_names_fn=lambda: ["Masha"], names_fn=lambda: ["Sasha"])
    assert "Masha is in front of you" in brain.requests[1].image_note


def test_a_question_about_a_side_is_answered_with_that_picture():
    left = _look("left", 50.0)
    display = _Display()
    brain = _Brain({"tool_call": {"name": "remember",
                                  "arguments": {"query": "left", "direction": "left"}}},
                   {"reply": "I saw a lamp.", "token_count": None})
    asked = []
    _turn("What was on your left?", brain, display=display,
          recall_seen_fn=lambda query, direction=None, pictures=True: (asked.append(direction), [left])[1])
    assert asked == ["left"]
    assert brain.requests[1].image_jpeg == b"FRAME"
    assert brain.requests[1].image_note.startswith("(This is your MEMORY of what you saw to your left")


def test_with_nothing_said_about_it_the_looks_answer():
    looks = [_look("right", 940.0, "I see a door."), _look("left", 900.0, "I see a lamp.")]
    display = _Display()
    brain = _Brain(_call("remember", "anything"),
                   {"reply": "I saw a door and a lamp.", "token_count": 1})
    _turn("What did you see?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: looks)
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == ["On my right, a minute ago: a door.",
                                       "On my left, 2 minutes ago: a lamp."]


def test_a_recalled_picture_says_who_was_in_it():
    """demo/chat_session.py's IMAGE_SYSTEM promises "when the note names the
    people in it, call them by their name instead of describing them" — and
    this note named nobody, so a picture of someone the robot had met came
    back as "a man with blonde hair"."""
    from demo.conversation import memory_note

    frame = {"ts": 900.0, "looked": "left", "names": ["Sasha"]}
    note = memory_note(frame, 1000.0)
    assert "Sasha was in it." in note
    assert "to your left" in note and "2 minutes ago" in note
    # Two people, and the ones the robot could not name are left out.
    frame = {"ts": 900.0, "people": [{"name": "Sasha"}, {"name": "Masha"},
                                     {"name": None}]}
    assert "Sasha and Masha were in it." in memory_note(frame, 1000.0)
    assert "in it" not in memory_note({"ts": 900.0}, 1000.0)


def test_a_look_is_not_read_back_twice_in_one_answer():
    """A look is stored as the frame's caption AND as the exchange that
    produced it (window.remember_now) — the same sentence written two ways.
    One answer must carry it once."""
    caption = "I see a window with dark brown curtains on the left side."
    looks = [_look("left", 940.0, caption)]
    found = Recalled([], [], recent=[
        "Sasha: What is on your left? — Reachy: " + caption,
        "Sasha: Tell me a joke. — Reachy: Why did the robot go on vacation?"])
    # `about` left out, so `anything`: a `seen` answer leaves the exchanges
    # out anyway. The dedup matters on the path that carries both.
    brain = _Brain(_call("remember", "what do you remember about today?"),
                   {"reply": "I saw a window.", "token_count": 1})
    _turn("What do you remember about today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: looks)
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == [
        "On my left, a minute ago: a window with dark brown curtains on the left side."]
    assert result["you_talked_about"] == [
        "Sasha: Tell me a joke. — Reachy: Why did the robot go on vacation?"]


def test_an_undescribed_frame_comes_back_as_a_picture():
    frame = _frame(0.4, ts=900.0)
    brain = _Brain(_call("remember", "the mug"), {"reply": "I saw a mug.", "token_count": None})
    _turn("Do you remember the mug?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [frame])
    assert brain.requests[1].image_jpeg == b"FRAME"


def test_the_person_in_front_of_the_robot_is_not_a_memory_of_them():
    """Live: "I saw Sasha moments ago" — about the person it was talking to,
    from a frame stored seconds earlier."""
    from demo.conversation import who

    class _People:
        enabled = True

        def faces_in(self, frame):
            return [{"name": "Sasha", "box": [0.1, 0.1, 0.3, 0.3]}]

        def met(self):
            return ["Sasha", "Masha"]

    class _Frames:
        def people_seen(self, before=None):
            return [("Sasha", 990.0), ("Masha", 400.0)]

    result = who(people=_People(), frame=object(), frame_memory=_Frames(),
                 clock=lambda: 1000.0)
    assert result["in_front_of_you"] == ["Sasha"]
    assert result["seen_earlier"] == ["Masha, 10 minutes ago"]


def test_the_people_it_saw_answer_when_nothing_else_does():
    brain = _Brain(_call("remember", "who"), {"reply": "I saw Sasha.", "token_count": 1})
    _turn("Who did you see today?", brain,
          recall_fn=lambda query: Recalled([], []),
          who_fn=lambda: {"seen_earlier": ["Sasha, 2 minutes ago"]})
    assert brain.requests[1].tool_result["result"] == {
        "people_you_saw": ["Sasha, 2 minutes ago"]}


def test_nothing_at_all_brings_back_the_latest_conversation():
    found = Recalled([], [], recent=["Person: hi — Reachy: hello"])
    brain = _Brain(_call("remember", "anything"), {"reply": "We said hello.", "token_count": 1})
    _turn("What did we talk about?", brain, recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: [])
    result = brain.requests[1].tool_result["result"]
    assert result["you_talked_about"] == found.recent


# --- what the robot answered live, and what it must answer
# instead. A face sighting from who() used to be enough to fire the words
# branch on its own, and everything behind it — the conversation, the looks —
# never reached the model: "what did we talk about today?" came back "We
# talked about Sasha and what you were curious about", "what did you see
# today?" came back "I saw Sasha moments ago".


def test_a_sighting_does_not_hide_the_conversation_it_was_asked_about():
    found = Recalled([], [], recent=["Sasha: what is Qdrant Edge? — Reachy: a vector "
                                     "database that runs on the robot"])
    brain = _Brain(_call("remember", "what did we talk about today?"),
                   {"reply": "We talked about Qdrant Edge.", "token_count": 1})
    _turn("What did we talk about today?", brain,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          who_fn=lambda: {"seen_earlier": ["Sasha, moments ago"]})
    result = brain.requests[1].tool_result["result"]
    assert result["you_talked_about"] == found.recent
    # And the sighting does not ride along beside it: the question was about
    # the conversation, and "Sasha, moments ago" is not part of the answer.
    assert "people_you_saw" not in result


def test_a_sighting_does_not_hide_what_the_robot_looked_at(monkeypatch):
    looks = [_look("left", 940.0, "I see a window with dark brown curtains.")]
    brain = _Brain(_call("remember", "what did you see today?"),
                   {"reply": "I saw a window on my left.", "token_count": 1})
    _turn("What did you see today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: looks,
          who_fn=lambda: {"seen_earlier": ["Sasha, moments ago"]})
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == [
        "On my left, a minute ago: a window with dark brown curtains."]
    # The look line carries whoever was in the frame (see _look_line); a flat
    # list of names beside it added nothing, and the model answered WITH it.
    assert "people_you_saw" not in result


def test_a_question_with_no_subject_comes_back_as_the_days_pictures():
    """"What did you see today?" names nothing a search can use, and the
    answer is the frames themselves — a frame IS the memory, the labels only
    pick which one."""
    day = [_frame(0.0, ts=940.0), _frame(0.0, ts=700.0)]
    display = _Display()
    brain = _Brain(_call("remember", "what did you see today?", about="seen"),
                   {"reply": "A window, and a desk.", "token_count": 1})
    _turn("What did you see today?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], [], recent=["Sasha: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          day_frames_fn=lambda: day)
    request = brain.requests[1]
    assert request.image_jpegs == (b"FRAME", b"FRAME"), "both frames, one turn"
    assert request.tool_result is None, "pictures and a tool result cannot travel together"
    # No note: what to say about a day of frames is demo/chat_session.py's
    # DAY_SYSTEM, measured to work there and not in a note.
    assert not request.image_note
    # …and the room sees the same frames the model was given.
    assert [len(event[1]) for event in display.events if event[0] == "frames"] == [2]


def test_the_days_own_system_message_carries_no_times_sides_or_names():
    """Measured: given each frame's time and side the model stopped looking
    and recited the metadata — "something was in front of me 40 minutes ago"
    — and given a name it bound it to the wrong picture every way it was
    worded (names_note.py)."""
    from demo.chat_session import DAY_SYSTEM

    for word in ("minute", "ago", "left", "right", "Sasha"):
        assert word not in DAY_SYSTEM, word
    assert "I saw" in DAY_SYSTEM and "past tense" in DAY_SYSTEM


def test_a_day_with_no_frames_falls_through_to_the_words():
    day_asked = []
    brain = _Brain(_call("remember", "what did you see today?", about="seen"),
                   {"reply": "Nothing yet.", "token_count": 1})
    _turn("What did you see today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], [], recent=["Sasha: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          day_frames_fn=lambda: (day_asked.append(1), [])[1])
    assert day_asked == [1]
    assert brain.requests[1].image_jpegs == ()
    assert brain.requests[1].tool_result is not None


def test_a_question_about_the_person_asking_is_answered_from_the_faces():
    """"Do you remember me?" and "did you see me today?" carry no subject a
    search can use. Left to the other cases the robot described its afternoon,
    and "did you see me today?" came back "I do not have any memory of seeing
    you today" — with the person in front of it."""
    brain = _Brain(_call("remember", "do you remember me?", about="me"),
                   {"reply": "Of course, Sasha.", "token_count": 1})
    _turn("Do you remember me?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("nor a picture search"),
          day_frames_fn=lambda: pytest.fail("nor the day"),
          who_fn=lambda: {"in_front_of_you": ["Sasha"],
                          "you_met": ["Sasha, 2 hours ago"],
                          "you_last_saw_them": ["Sasha, 5 minutes ago"],
                          "seen_earlier": ["Masha, an hour ago"],
                          "people_you_have_met": ["Sasha", "Masha"]})
    result = brain.requests[1].tool_result["result"]
    assert result["in_front_of_you"] == ["Sasha"]
    assert result["you_met"] == ["Sasha, 2 hours ago"]
    # The one place a sighting of the person in front is the answer, not noise.
    assert result["you_last_saw_them"] == ["Sasha, 5 minutes ago"]
    assert "seen_earlier" not in result


def test_a_stranger_asking_if_they_are_remembered_is_told_the_truth():
    brain = _Brain(_call("remember", "have we met?", about="me"),
                   {"reply": "I don't think we have.", "token_count": 1})
    _turn("Have we met?", brain, who_fn=lambda: {"in_front_of_you": [],
                                                 "people_you_have_met": ["Masha"]})
    result = brain.requests[1].tool_result["result"]
    assert "not met" in result["note"]
    assert result["people_you_have_met"] == ["Masha"]


def test_a_robot_that_cannot_recognise_faces_does_not_call_anyone_a_stranger():
    # Faces off, or the face read failed: "Do you remember me?" came back "I
    # don't think we've met" — to someone the robot had met.
    for who_fn in (lambda: {"faces_off": True,
                            "note": "You cannot recognise faces right now."}, None):
        brain = _Brain(_call("remember", "do you remember me?", about="me"),
                       {"reply": "I can't tell right now.", "token_count": 1})
        _turn("Do you remember me?", brain, who_fn=who_fn)
        result = brain.requests[1].tool_result["result"]
        assert "not met" not in result["note"]
        assert "cannot recognise faces" in result["note"]
        assert "faces_off" not in result, "the flag is for the code, not the model"


def test_who_says_so_when_the_faces_cannot_be_read():
    from demo.conversation import who

    class _Unreadable(_People):
        def faces_in(self, frame):
            raise OSError("embed service down")

    result = who(people=_Unreadable(met=["Sasha"]), frame=object())
    assert result["faces_off"] is True
    assert result["note"] == "You cannot recognise faces right now."
    assert result["people_you_have_met"] == ["Sasha"]

    # Through the `me` answer, as the voice loop wires it: the model hears it
    # cannot tell, and still who it has met.
    brain = _Brain(_call("remember", "do you remember me?", about="me"),
                   {"reply": "I can't tell right now.", "token_count": 1})
    _turn("Do you remember me?", brain,
          who_fn=lambda: who(people=_Unreadable(met=["Sasha"]), frame=object()))
    answer = brain.requests[1].tool_result["result"]
    assert "cannot recognise faces" in answer["note"]
    assert answer["people_you_have_met"] == ["Sasha"]


def test_a_question_that_names_something_is_still_a_search():
    """The day is read back only when the question named nothing. "What did I
    tell you about Qdrant?" must not come back with this afternoon's frames."""
    told = "Sasha: Qdrant Edge runs on the robot — Reachy: got it"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}])
    brain = _Brain(_call("remember", "Qdrant"), {"reply": "You told me.", "token_count": 1})
    _turn("What did I tell you about Qdrant?", brain,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          day_frames_fn=lambda: pytest.fail("a question with a subject is a search"))
    assert brain.requests[1].tool_result["result"] == {
        "the_person_told_you_before": [told]}


def test_the_words_of_a_question_that_carries_no_subject():
    from demo.conversation import subject_of

    for question in ("What did you see today?", "What did we talk about?",
                     "Do you remember anything?", "What happened earlier?",
                     "What do you remember?"):
        assert subject_of(question) == set(), question
    assert subject_of("What did I tell you about Qdrant?") == {"qdrant"}
    assert subject_of("Do you remember the mug?") == {"mug"}


def test_a_non_visual_question_with_no_subject_never_grabs_a_random_frame():
    from demo.conversation import GENERAL_QUESTION_NOTE

    """Live: "how do you work?" reduces to no subject the same
    way "what did you see?" does — but recall_seen's own no-subject fallback
    used to answer it anyway, with whatever frame happened to be nearest,
    and the model summarised an unrelated caption."""
    grabbed = []

    def recall_seen_fn(query, direction=None, pictures=True):
        grabbed.append(query)
        return [_look("ahead", 900.0, "I see a room with a red curtain.")]

    brain = _Brain(_call("remember", "how do you work", about="anything"),
                   {"reply": "I am not sure.", "token_count": 1})
    _turn("How do you work?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []), recall_seen_fn=recall_seen_fn)
    assert grabbed == [], "a non-visual question must not search for a picture at all"
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": GENERAL_QUESTION_NOTE}


def test_a_bare_what_did_you_see_still_reaches_the_looks_even_with_a_vague_query():
    """The model's own `query` argument can be a placeholder ("anything")
    that carries no seeing word — the check falls back to what was actually
    HEARD, or this regresses to the same bug from the other direction."""
    looks = [_look("right", 940.0, "I see a door.")]
    brain = _Brain(_call("remember", "anything"),
                   {"reply": "I saw a door.", "token_count": 1})
    _turn("What have you seen?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: looks)
    assert brain.requests[1].tool_result["result"]["you_looked_at"] == [
        "On my right, a minute ago: a door."]


def test_a_fact_that_cleared_the_gate_on_a_bare_question_brings_no_pictures():
    """Live: "tell me how does your memory work?" came as
    `anything`, the fact answered at 0.74 — and the latest looks came along
    as "you_looked_at", so the screen showed the day's pictures under an
    answer about Qdrant Edge. "memory" is a word about the mechanism, not
    the past; and a bare question the knowledge base answers is about what
    the robot was taught, whatever `about` says."""
    display = _Display()
    brain = _Brain(_call("remember", "how does your memory work", about="anything"),
                   {"reply": "In Qdrant Edge shards.", "token_count": 1})
    _turn("Tell me how does your memory work?", brain, display=display,
          clock=lambda: 1000.0,
          knowledge_fn=lambda query: [
              {"text": "My memory lives in Qdrant Edge shards.", "score": 0.74,
               "source": "knowledge"}],
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("no pictures for a fact"))
    result = brain.requests[1].tool_result["result"]
    assert result == {"facts_you_were_taught": ["My memory lives in Qdrant Edge shards."]}
    assert not [e for e in display.events if e[0] == "frames" and e[1]]


def test_about_anything_is_answered_as_anything_whatever_the_words():
    """The words of a question never change the model's `about`. "How do you
    work?" marked `anything` is answered as `anything`: the knowledge base is
    searched, no picture search — it names nothing and says nothing about
    the past — and, with no fact over the gate, the latest exchanges."""
    asked = []

    def knowledge_fn(query):
        asked.append(query)
        return []

    found = Recalled([], [], recent=["Sasha: Look left. — Reachy: I see a curtain."])
    brain = _Brain(_call("remember", "how do you work", about="anything"),
                   {"reply": "I listen and remember.", "token_count": 1})
    _turn("How do you work?", brain, knowledge_fn=knowledge_fn,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("no frames"))
    assert asked == ["how do you work"]
    assert brain.requests[1].tool_result["result"] == {
        "you_talked_about": ["Sasha: Look left. — Reachy: I see a curtain."]}


def test_taught_is_answered_from_the_facts_and_keeps_the_exchanges_out():
    """Live: "how do you work?" missed its own fact, the tool came
    back with the last three exchanges instead, and the robot answered "I
    work by processing information" — no Qdrant. The phrasings stored with
    each fact find it now (demo/knowledge.py); and for `taught` the
    exchanges stay out — they were the noise it answered from."""
    found = Recalled([], [], recent=["Sasha: Look left. — Reachy: I see a curtain."])
    brain = _Brain(_call("remember", "how do you work", about="taught"),
                   {"reply": "I work with Qdrant Edge.", "token_count": 1})
    _turn("How do you work?", brain,
          knowledge_fn=lambda query: [{"text": "I work with Qdrant Edge.",
                                       "score": 0.93, "source": "knowledge"}],
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("no frames for taught"))
    assert brain.requests[1].tool_result["result"] == {
        "facts_you_were_taught": ["I work with Qdrant Edge."]}


def test_the_facts_are_searched_for_every_question_but_one_about_the_person():
    # The one memory tool searches every store, and the scores decide what
    # answers; only a question about the person asking goes to the faces.
    asked = []

    def knowledge_fn(query):
        asked.append(query)
        return []

    for about, question in (("seen", "Do you remember the mug?"),
                            ("said", "What did we talk about?"),
                            ("anything", "What do you remember?"),
                            ("taught", "How do you work?"),
                            ("me", "Do you remember me?")):
        brain = _Brain(_call("remember", question, about=about),
                       {"reply": "Nothing about that.", "token_count": 1})
        _turn(question, brain, knowledge_fn=knowledge_fn,
              recall_fn=lambda query: Recalled([], []),
              recall_seen_fn=lambda query, direction=None, pictures=True: [],
              day_frames_fn=lambda: [], who_fn=lambda: {})
    assert asked == ["Do you remember the mug?", "What did we talk about?",
                     "What do you remember?", "How do you work?"]


def test_a_camera_call_is_the_camera_whatever_tense_the_question_is_in():
    """The tool the model called is the tool that runs. "Did you see the
    pollution?" sent to `camera` used to be rewritten into `remember(seen)` on
    the tense of its words (a crutch, removed before publication): now the
    camera is looked through, and nothing in memory is read."""
    brain = _Brain(_call("camera", "did you see the pollution?", direction="ahead"),
                   {"reply": "I see no pollution.", "token_count": None})
    _turn("Did you see the pollution?", brain, clock=lambda: 1000.0,
          camera_jpeg=lambda: b"NOW",
          recall_fn=lambda query: pytest.fail("not memory"),
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("not memory"),
          day_frames_fn=lambda: pytest.fail("not the day"))
    request = brain.requests[1]
    assert (request.image_jpeg, request.image_note) == (b"NOW", LOOK_NOTE)


def test_the_models_about_is_answered_as_it_was_given():
    """`about` is the model's word, run as given — the question's words never
    move it. "Did you see me today?" marked `seen` reads the day back (it
    used to be forced to `me`); "did you see anything today?" marked
    `anything` does not (it used to be forced to `seen`)."""
    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    brain = _Brain(_call("remember", "did you see me today?", about="seen"),
                   {"reply": "I saw a chair and a lamp.", "token_count": 1})
    _turn("Did you see me today?", brain, clock=lambda: 1000.0,
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          day_frames_fn=lambda: day,
          who_fn=lambda: pytest.fail("not the faces: the model said `seen`"))
    assert len(brain.requests[1].image_jpegs) == 2
    brain = _Brain(_call("remember", "what did you see", about="anything"),
                   {"reply": "Nothing much.", "token_count": 1})
    _turn("Did you see anything today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [],
          day_frames_fn=lambda: pytest.fail("not the day: the model said `anything`"))
    assert brain.requests[1].image_jpegs == ()
    assert brain.requests[1].tool_result is not None


def test_a_question_that_says_nothing_about_seeing_gets_no_picture_guess():
    """Live: "tell me about the universe" — about=anything,
    subject "universe" — matched nothing by words, so the nearest-picture
    search ran and put a frame of the room on the screen as a guess under
    an answer about galaxies. That search exists for "the one I showed
    you"; a question with no seeing word in it gets the words only."""
    asked = []

    def recall_seen_fn(query, direction=None, pictures=True):
        asked.append(pictures)
        return []

    brain = _Brain(_call("remember", "universe", about="anything"),
                   {"reply": "The universe is vast.", "token_count": 1})
    _turn("Tell me about the universe.", brain, recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=recall_seen_fn)
    brain = _Brain(_call("remember", "the thing", about="anything"),
                   {"reply": "A mug.", "token_count": 1})
    _turn("What was the thing I showed you?", brain, recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=recall_seen_fn)
    brain = _Brain(_call("remember", "bottle", about="seen"),
                   {"reply": "A bottle.", "token_count": 1})
    _turn("Do you remember the bottle?", brain, recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=recall_seen_fn)
    assert asked == [False, True, True]


def test_recall_seen_without_pictures_stops_at_the_words():
    frames = _Frames([_frame(0.9)], looks=[])
    assert recall_seen("universe", frame_memory=frames, pictures=False) == []
    assert frames.min_scores == [], "no SigLIP search"


def test_a_picture_under_the_gate_is_not_sent_to_the_model_as_the_answer():
    """Live: "did you see the blue shell?" matched nothing by
    words, the nearest picture (a guess) went to the model, and the guess
    came back as a fact: "I do not see a blue shell in front of me right
    now". The screen shows the guess marked weak; the model is told there is
    nothing."""
    from demo.conversation import NEVER_SAW_NOTE

    display = _Display()
    weak = _frame(0.04, ts=900.0)
    brain = _Brain(_call("remember", "blue shell", about="seen"),
                   {"reply": "I did not see a blue shell.", "token_count": 1})
    _turn("Did you see the blue shell?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [weak])
    assert brain.requests[1].image_jpeg is None
    assert brain.requests[1].tool_result["result"] == {"note": NEVER_SAW_NOTE}
    frames = [hits for kind, *rest in display.events for hits in rest if kind == "frames"]
    assert frames[-1] == [{**weak, "weak": True}]


def test_a_seen_question_is_answered_from_frames_never_from_the_conversation():
    """Live: "did you see people?" and "did you see any TV?" came
    back with earlier exchanges — "Did you see the pollution? — I do not see
    any pollution in front of me" — and the model copied their tense: "I do
    not see any people in front of me right now". A question about what was
    SEEN is answered from frames (A, B, C); the conversation is case D. Here
    "did you see the table?" gets the frame of the table as a picture, and
    the exchange that would have matched is never asked for."""
    display = _Display()
    strong = _frame(0.4, ts=900.0)
    brain = _Brain(_call("remember", "table", about="seen"),
                   {"reply": "I saw a small table.", "token_count": 1})
    _turn("Did you see the table?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: pytest.fail("seen never reads the conversation"),
          recall_seen_fn=lambda query, direction=None, pictures=True: [strong])
    assert brain.requests[1].image_jpeg == b"FRAME"
    frames = [hits for kind, *rest in display.events for hits in rest if kind == "frames"]
    assert frames[-1] == [{**strong, "weak": False}]


def test_a_filler_word_is_not_the_subject_of_a_question():
    """Live: "Nice, what did you see today?" had the subject
    "nice", missed the day-frames case, and came back as four frames of the
    same person under a sentence copied from an old exchange."""
    from demo.conversation import subject_of

    assert subject_of("Nice, what did you see today?") == set()
    assert subject_of("Okay, so what did you see?") == set()
    assert subject_of("Nice, did you see the bottle?") == {"bottle"}
    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    brain = _Brain(_call("remember", "Nice, what did you see today?", about="seen"),
                   {"reply": "I saw a chair and a lamp.", "token_count": 1})
    _turn("Nice, what did you see today?", brain, clock=lambda: 1000.0,
          recall_seen_fn=lambda query, direction=None, pictures=True: pytest.fail("the day, not a search"),
          day_frames_fn=lambda: day)
    assert len(brain.requests[1].image_jpegs) == 2


def test_what_is_stored_back_is_decided_by_the_turn_never_by_the_words():
    """A reply built from what memory handed back is never written back
    (Exchange.derived) — and only that, decided by the turn, never by the
    question's words. A search that found nothing leaves an ordinary
    exchange, whatever the model marked it: "Did you notice any pollution?",
    marked `seen`, found nothing. What keeps "I did not notice any" from
    coming back as a memory of seeing is the reading side: a `seen` question
    never reads the conversation (test_a_seen_question_is_answered_from_
    frames_never_from_the_conversation)."""
    nothing = {"recall_fn": lambda query: Recalled([], []),
               "recall_seen_fn": lambda query, direction=None, pictures=True: []}
    window = _window(_Memory())
    brain = _Brain(_call("remember", "pollution", about="seen"),
                   {"reply": "I did not notice any.", "token_count": 1})
    _turn("Did you notice any pollution?", brain, window=window, **nothing)
    brain = _Brain(_call("remember", "me", about="me"),
                   {"reply": "Yes, Sasha!", "token_count": 1})
    _turn("Have you seen me before?", brain, window=window,
          who_fn=lambda: {"in_front_of_you": ["Sasha"]})
    # A general question memory had nothing on: the model's own answer.
    brain = _Brain(_call("remember", "airplanes", about="anything"),
                   {"reply": "Wings make lift.", "token_count": 1})
    _turn("How do airplanes fly?", brain, window=window, **nothing)
    brain = _Brain({"reply": "I do not see any pollution.", "token_count": 1})
    _turn("Did you see the pollution?", brain, window=window)
    assert [e.derived for e in window._exchanges] == [False, True, False, False]


def test_a_statement_the_model_searched_on_is_still_remembered():
    # Live: the model called remember(about=said) before answering "My dog
    # is called Rex.", found nothing, and the exchange was never stored —
    # what the person told the robot was lost.
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    brain = _Brain(_call("remember", "the name of the dog", about="said"),
                   {"reply": "Rex is a lovely name!", "token_count": 500})
    _turn("My dog is called Rex.", brain, window=window,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [])
    window.flush()
    assert any("Rex" in text for text, _kind, _meta in memory.remembered)


def test_a_taught_question_nothing_answers_is_a_general_one():
    # Live: "tell me about black holes" came as `taught`, was given the latest
    # exchanges instead, and the robot said it had no information "in my
    # memory".
    from demo.conversation import GENERAL_QUESTION_NOTE

    brain = _Brain(_call("remember", "black holes", about="taught"),
                   {"reply": "Black holes are regions of space.", "token_count": 1})
    _turn("Tell me about black holes.", brain, knowledge_fn=lambda query: [],
          recall_fn=lambda query: pytest.fail("not a memory question"))
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": GENERAL_QUESTION_NOTE}


def test_an_invented_tool_is_answered_as_handled_not_refused():
    brain = _Brain(_call("yes_dance"), {"reply": "Sure, dancing!", "token_count": 1})
    _turn("Dance for me!", brain, recall_fn=lambda query: pytest.fail("not a memory call"))
    assert brain.requests[1].tool_result["result"]["done"] is True


def test_the_model_is_offered_one_memory_tool_the_camera_and_the_body():
    from demo.chat_session import CHAT_TOOLS

    assert [tool["function"]["name"] for tool in CHAT_TOOLS] == ["remember", "camera", "move"]


def test_the_picture_the_camera_gives_the_model_goes_to_the_screen():
    """The camera tool's own picture goes to the display, under the tool line
    — the same bytes the model gets."""
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "ahead"}}},
                   {"reply": "I see a mug.", "token_count": None})
    _turn("What do you see?", brain, display=display, camera_jpeg=lambda: b"JPEG-NOW")
    assert ("look", b"JPEG-NOW") in display.events
    assert brain.requests[1].image_jpeg == b"JPEG-NOW"


def test_tell_me_about_is_a_request_not_a_question_about_the_past():
    # "Tell me about black holes" read "tell" as the past and got "Nothing in
    # your memory about this." — the model then refused a general question.
    from demo.conversation import GENERAL_QUESTION_NOTE

    brain = _Brain(_call("remember", "black holes", about="anything"),
                   {"reply": "Black holes are regions of space.", "token_count": 1})
    _turn("Tell me about black holes.", brain,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None, pictures=True: [])
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": GENERAL_QUESTION_NOTE}


