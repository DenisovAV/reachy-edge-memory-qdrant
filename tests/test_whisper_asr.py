"""The hotword guard: Whisper handing back its own prompt instead of speech.

faster-whisper feeds `hotwords` to the decoder as preceding context, and on a
short utterance it can decode nothing from, the decoder continues that context
— it writes the list out. Live on stage, asked for a name, the
robot heard "Qdrant Edge, Reachy, Sasha, vector database, embeddings" and
answered "Sorry, I didn't catch your name"; the person had said "Sasha".

No model is loaded here: WhisperRecognizer imports faster_whisper lazily, and
these test the pure rule that decides what the transcript IS.
"""
from __future__ import annotations

from emulator.whisper_asr import HOTWORDS, is_hotword_echo


def test_the_list_coming_back_is_not_speech():
    assert is_hotword_echo(HOTWORDS)
    assert is_hotword_echo("Qdrant, Qdrant Edge, Reachy")
    # As it actually came back, with the decoder's own punctuation and a word
    # repeated (the shape that made the robot search its memory for Qdrant).
    assert is_hotword_echo("Qdrant Edge, Reachy, Reachy,")
    assert is_hotword_echo("Reachy. Qdrant.")


def test_one_hotword_on_its_own_is_something_a_person_says():
    """The demo is ABOUT Qdrant: "Qdrant Edge" as a whole utterance is a
    question to answer, not a prompt echo."""
    for said in ("Qdrant", "Qdrant Edge", "Reachy", "Qdrant Edge?"):
        assert not is_hotword_echo(said), said


def test_anything_with_a_word_of_its_own_is_speech():
    for said in ("Tell me about Qdrant", "Reachy, what is Qdrant Edge?",
                 "Sasha", "I'm Sasha", "Qdrant Edge is fast, Reachy"):
        assert not is_hotword_echo(said), said


def test_nothing_is_not_an_echo():
    assert not is_hotword_echo("")
    assert not is_hotword_echo("   ")


def test_the_name_is_not_in_the_prompt_any_more():
    """It bought nothing — measured, Whisper writes "Sasha", "I'm Sasha" and
    "My name is Sasha" correctly without it (16/18 either way) — and it cost the one thing that must never be fabricated:
    with "Sasha" in the list, an echo READS LIKE AN ANSWER to "what's your
    name?", and a wrong name attached to a face outlives the mistake."""
    assert "Sasha" not in HOTWORDS
    assert "Qdrant" in HOTWORDS and "Reachy" in HOTWORDS
