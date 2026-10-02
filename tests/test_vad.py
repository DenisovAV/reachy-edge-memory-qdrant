"""Voice detection logic: no microphone, synthetic chunks only.

Checks what actually drives the robot's behavior: when an utterance starts,
when it ends, and that the start of a word isn't clipped by the prebuffer.
"""

import numpy as np

from demo.vad import VoiceGate, calibrate_threshold, collect_utterance, rms


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0


def test_rms_grows_with_amplitude():
    quiet = rms(np.full(1000, 0.1, dtype=np.float32))
    loud = rms(np.full(1000, 0.5, dtype=np.float32))
    assert loud > quiet
    assert abs(loud - 0.5) < 1e-5


def test_gate_needs_sustained_loudness_to_start():
    gate = VoiceGate(threshold=0.1, onset_chunks=2)
    # A single loud chunk is not speech (a click, a knock).
    assert gate.push(0.5) is None
    assert gate.push(0.0) is None
    assert not gate.speaking


def test_gate_starts_after_onset_chunks():
    gate = VoiceGate(threshold=0.1, onset_chunks=2)
    assert gate.push(0.5) is None
    assert gate.push(0.5) == "start"
    assert gate.speaking


def test_gate_ends_after_sustained_silence():
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=3)
    gate.push(0.5)  # start
    assert gate.speaking
    assert gate.push(0.0) is None
    assert gate.push(0.0) is None
    assert gate.push(0.0) == "end"
    assert not gate.speaking


def test_gate_short_pause_does_not_end_utterance():
    # A pause between words shorter than hangover must not cut off the utterance.
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=5)
    gate.push(0.5)  # start
    gate.push(0.0)  # quiet
    gate.push(0.0)  # quiet
    assert gate.push(0.5) is None, "speech resumed — not the end"
    assert gate.speaking
    # the silence counter reset, need 5 quiet chunks in a row again to end
    for _ in range(4):
        assert gate.push(0.0) is None
    assert gate.push(0.0) == "end"


def _source(levels, chunk_len=10):
    """Stream of (chunk, rms) built from a list of loudness levels."""
    for lv in levels:
        yield np.full(chunk_len, lv, dtype=np.float32), lv


def test_collect_utterance_captures_speech_span():
    # min_chunks=1 on purpose: this covers the span-capture MECHANISM, and
    # the short synthetic utterance below would otherwise be rejected by the
    # too-brief-to-be-speech policy (MIN_UTTERANCE_CHUNKS), which has its own
    # tests. Keeping them separate means neither can silently stop testing.
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=2)
    # silence, speech (3 chunks), silence
    levels = [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
    audio = collect_utterance(_source(levels), gate, preroll=1, min_chunks=1)
    assert audio is not None
    assert len(audio) > 0


def test_collect_utterance_preroll_keeps_word_onset():
    # The prebuffer must capture a chunk BEFORE onset is detected.
    gate = VoiceGate(threshold=0.1, onset_chunks=2, hangover_chunks=2)
    # two quiet chunks (prebuffer), then loud ones — onset is detected on the
    # 2nd loud chunk, but the first loud chunk and one quiet chunk before it
    # must still land in the recording
    levels = [0.05, 0.05, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    # min_chunks=1: this is about the prebuffer, not the length policy.
    audio = collect_utterance(_source(levels, chunk_len=10), gate, preroll=2,
                              min_chunks=1)
    assert audio is not None
    # 2 prebuffer chunks + at least 2 loud chunks before the end — at least 4 chunks of 10
    assert len(audio) >= 40


def test_collect_utterance_caps_at_max_chunks():
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=100)
    levels = [0.5] * 200  # speaks without stopping
    audio = collect_utterance(_source(levels), gate, preroll=0, max_chunks=10)
    assert len(audio) == 100  # 10 chunks of 10 samples


def test_calibrate_threshold_scales_with_noise():
    quiet = calibrate_threshold(_source([0.001] * 10), seconds=1.0,
                                chunk_ms=100, multiplier=3.0, floor=0.0)
    noisy = calibrate_threshold(_source([0.05] * 10), seconds=1.0,
                                chunk_ms=100, multiplier=3.0, floor=0.0)
    assert noisy > quiet
    assert abs(noisy - 0.15) < 1e-4


def test_calibrate_threshold_respects_floor():
    # A perfectly silent microphone must not produce a zero threshold.
    thr = calibrate_threshold(_source([0.0] * 10), seconds=1.0,
                              chunk_ms=100, multiplier=3.0, floor=0.01)
    assert thr == 0.01


def test_a_brief_noise_is_not_answered():
    """A cough or a chair scrape clears the energy gate as easily as a word.

    On the first live voice run one such blip was transcribed as "You" and
    answered in full — the robot holding a conversation with the room's
    noise. A burst too short to be speech must be discarded, and the loop
    must keep listening rather than returning it as an utterance.
    """
    import numpy as np
    from demo.vad import MIN_UTTERANCE_CHUNKS, VoiceGate, collect_utterance

    loud = np.ones(1600, np.float32)
    quiet = np.zeros(1600, np.float32)

    def stream():
        # a two-chunk blip, then silence — nothing worth a turn
        yield from [(quiet, 0.001)] * 3
        yield from [(loud, 0.5)] * 2
        yield from [(quiet, 0.001)] * 10

    assert collect_utterance(stream(), VoiceGate(threshold=0.1)) is None
    assert MIN_UTTERANCE_CHUNKS > 2


def test_a_real_utterance_after_a_blip_is_still_captured():
    """Discarding a blip must not deafen the robot: the sentence that follows
    it has to be collected normally, not swallowed with the noise."""
    import numpy as np
    from demo.vad import VoiceGate, collect_utterance

    loud = np.ones(1600, np.float32)
    quiet = np.zeros(1600, np.float32)

    def stream():
        yield from [(quiet, 0.001)] * 3
        yield from [(loud, 0.5)] * 2           # blip — dropped
        yield from [(quiet, 0.001)] * 6
        yield from [(loud, 0.5)] * 12          # a real sentence
        yield from [(quiet, 0.001)] * 10

    audio = collect_utterance(stream(), VoiceGate(threshold=0.1))
    assert audio is not None, "the real sentence after a blip was lost"
    assert len(audio) >= 12 * 1600


def test_collect_utterance_gives_up_waiting_when_told_to_stop():
    """The dashboard's pause, pressed during silence: the wait ends with an
    EMPTY array (not None, which would end the loop), within a few chunks."""
    from demo.vad import VoiceGate, collect_utterance

    quiet = np.zeros(1600, np.float32)
    asked = []

    def stop():
        asked.append(True)
        return True

    stream = iter([(quiet, 0.0)] * 50)
    out = collect_utterance(stream, VoiceGate(threshold=0.5), stop=stop, stop_every=5)
    assert out is not None and out.size == 0
    assert len(asked) == 1
    assert len(list(stream)) == 45, "gave up after five chunks, not fifty"
    # mid-speech too: a room that never goes quiet must still pause
    loud = np.ones(1600, np.float32)
    stream = iter([(loud, 1.0)] * 50)
    out = collect_utterance(stream, VoiceGate(threshold=0.5), stop=lambda: True, stop_every=5)
    assert out is not None and out.size == 0
    assert len(list(stream)) == 45
    # no stop: the old contract — the stream runs out, None comes back
    assert collect_utterance(iter([(quiet, 0.0)] * 6), VoiceGate(threshold=0.5)) is None


def test_clicks_while_nobody_speaks_do_not_count_towards_an_utterance():
    """Single clicks never open the gate, but they used to be counted as
    loud chunks all the same — and a cough after enough of them passed the
    minimum length on their account."""
    import numpy as np
    from demo.vad import VoiceGate, collect_utterance

    loud = np.ones(1600, np.float32)
    quiet = np.zeros(1600, np.float32)

    def stream():
        for _ in range(6):                       # six lone clicks, spaced out
            yield (loud, 0.5)
            yield (quiet, 0.001)
        yield from [(loud, 0.5)] * 2            # a cough: opens the gate
        yield from [(quiet, 0.001)] * 10

    assert collect_utterance(stream(), VoiceGate(threshold=0.1)) is None
