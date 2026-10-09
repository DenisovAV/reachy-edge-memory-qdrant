"""emulator/speech_detector.py — whether an utterance holds speech at all.

The rule is Silero's own default (speech from 0.5, ended by 100 ms under
0.35, counted from 250 ms); a chunk is 32 ms, so 250 ms is just under eight
chunks. The model itself runs only when it is in the Hugging Face cache.
"""
from __future__ import annotations

import numpy as np
import pytest

from emulator.speech_detector import holds_speech


def test_a_quarter_second_of_speech_is_speech():
    assert holds_speech([0.0] * 5 + [0.9] * 8 + [0.0] * 5)


def test_less_than_a_quarter_second_is_not():
    # The longest run any noise clip reached was 7 chunks of motor whir.
    assert not holds_speech([0.0] * 5 + [0.9] * 7 + [0.0] * 5)


def test_a_dip_shorter_than_the_silence_rule_does_not_split_the_speech():
    # Two quiet chunks (64 ms) inside a word are not the end of it.
    assert holds_speech([0.9] * 5 + [0.1] * 2 + [0.9] * 3)


def test_a_real_pause_ends_the_speech_and_what_is_left_is_too_short():
    assert not holds_speech([0.9] * 5 + [0.1] * 5 + [0.9] * 5)


def test_a_started_stretch_goes_on_while_it_stays_over_the_lower_line():
    assert holds_speech([0.6] + [0.4] * 8)


def test_nothing_over_the_line_is_never_speech():
    assert not holds_speech([0.45] * 50)
    assert not holds_speech([])


def test_the_first_stretch_of_speech_is_enough():
    # The detector is lazy: past the answer it reads nothing more, so a turn
    # does not wait on the rest of the utterance.
    read = []

    def probabilities():
        for prob in [0.9] * 8 + [0.0] * 100:
            read.append(prob)
            yield prob

    assert holds_speech(probabilities())
    assert len(read) == 8


def _silero():
    try:
        from huggingface_hub import hf_hub_download

        from emulator.models import get

        spec = get("silero-vad")
        return hf_hub_download(spec.repo, spec.file, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, or offline
        return None


@pytest.mark.skipif(_silero() is None, reason="Silero VAD is not cached")
def test_with_the_real_model_noise_and_silence_hold_no_speech():
    from emulator.speech_detector import SpeechDetector

    detector = SpeechDetector(_silero())
    rng = np.random.default_rng(0)
    assert not detector.holds_speech(rng.normal(0, 0.02, 32000).astype(np.float32))
    assert not detector.holds_speech(np.zeros(32000, np.float32))
