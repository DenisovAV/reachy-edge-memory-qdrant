"""Incremental sentence-parsing of the stream — no network involved.

This checks the exact reason streaming exists: the first complete sentence
is extracted immediately, and an unfinished tail is never lost.
"""
from demo.stream import extract_sentences, flush_tail


def test_extracts_complete_sentence():
    sents, tail = extract_sentences("Hello. I see")
    assert sents == ["Hello."]
    assert tail == " I see"


def test_no_complete_sentence_keeps_all_as_tail():
    sents, tail = extract_sentences("I can see a")
    assert sents == []
    assert tail == "I can see a"


def test_multiple_sentences_in_one_go():
    sents, tail = extract_sentences("Hello. Nice to see you! How are you")
    assert sents == ["Hello.", "Nice to see you!"]
    assert tail == " How are you"


def test_question_and_exclamation_end_sentences():
    sents, tail = extract_sentences("Who are you? A robot!")
    assert sents == ["Who are you?", "A robot!"]
    assert tail == ""


def test_incremental_accumulation():
    # Simulates a token stream: the buffer grows, sentences are pulled out as they're ready.
    buffer = ""
    spoken = []
    for token in ["He", "llo", ".", " I ", "see", " you", "."]:
        buffer += token
        sents, buffer = extract_sentences(buffer)
        spoken.extend(sents)
    assert spoken == ["Hello.", "I see you."]
    assert buffer.strip() == ""


def test_flush_tail_returns_leftover():
    # The model cut off without a period — the tail still needs to be spoken.
    assert flush_tail(" and a cup") == "and a cup"


def test_flush_tail_ignores_empty():
    assert flush_tail("   ") is None
    assert flush_tail("") is None
