"""Voice detection and continuous microphone listening.

The demo triggers on voice, not a button press: the robot listens the whole
time, catches an utterance, and responds. Two pitfalls, both solved here.

The first is separating speech from silence. A simple energy-based VAD:
compute the loudness (RMS) of short chunks and compare against a threshold.
The threshold is fixed (demo/run_demo.py's --speech-threshold, 0.010): one
calibrated from a second of the room at start came out anywhere from 0.010 to
0.114 on the robot. `calibrate_threshold` is still here for
--speech-threshold 0.

The second is not hearing itself. While the robot is talking, the microphone
picks up its own voice and would trigger on it. So we don't listen during a
reply, and afterward we discard the buffer that accumulated
(`MicSource.flush`, platform-specific implementation — see demo/platform/).

The pure logic (RMS, the `VoiceGate` state machine, utterance assembly) is
separated from the microphone device and is platform-independent: `MicStream`
(the laptop's, ffmpeg) and `RobotMicSource` (the robot's, over HTTP) live in
demo/platform/ and both hand back the same `chunks() -> (chunk_float32, rms)` contract that
is tested here without a microphone.
"""

from __future__ import annotations

import numpy as np


def rms(chunk: np.ndarray) -> float:
    """Root-mean-square loudness of a chunk. An empty chunk is zero."""
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))


class VoiceGate:
    """State machine: silence -> speech -> silence.

    Speech starts once loudness stays above the threshold for `onset_chunks`
    chunks in a row (so a single click doesn't trigger it). It ends once
    loudness stays below the threshold for `hangover_chunks` chunks (a pause
    within an utterance shorter than that doesn't cut it off).
    """

    def __init__(self, threshold: float, onset_chunks: int = 2,
                 hangover_chunks: int = 8) -> None:
        self.threshold = threshold
        self.onset_chunks = onset_chunks
        self.hangover_chunks = hangover_chunks
        self.speaking = False
        self._loud_run = 0
        self._quiet_run = 0

    def push(self, chunk_rms: float) -> str | None:
        """Feed in a chunk's loudness. Return 'start', 'end', or None."""
        loud = chunk_rms >= self.threshold
        if not self.speaking:
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.onset_chunks:
                self.speaking = True
                self._quiet_run = 0
                return "start"
            return None
        # already speaking — waiting for sustained silence
        self._quiet_run = self._quiet_run + 1 if not loud else 0
        if self._quiet_run >= self.hangover_chunks:
            self.speaking = False
            self._loud_run = 0
            return "end"
        return None


# Shortest burst of sound worth answering, in chunks (chunks are ~100 ms).
# A cough, a laugh from the audience or a chair scraping clears the energy
# gate as easily as a word does, and on the first live run one such blip was
# transcribed as "You" and answered in full — the robot talking to the room's
# noise. Real speech to a robot is rarely under half a second, so a burst
# shorter than this is discarded and the loop keeps listening.
MIN_UTTERANCE_CHUNKS = 5


def collect_utterance(chunk_rms_source, gate: VoiceGate,
                      preroll: int = 3, max_chunks: int = 100,
                      min_chunks: int = MIN_UTTERANCE_CHUNKS,
                      stop=None, stop_every: int = 5):
    """Collect a single utterance from a stream of (chunk, its RMS).

    `chunk_rms_source` is an iterator of (chunk, rms) pairs. Returns the
    concatenated utterance audio, or None if the stream ran out. `stop`, if
    given, is asked every `stop_every` chunks — mid-utterance too: in a room
    where people keep talking the gate never closes, every utterance runs to
    `max_chunks`, and a pause that waited for silence never came (live) —
    whether to give up; when it says so, an EMPTY array comes back —
    distinct from None, which ends the loop. This is what lets the
    dashboard's pause take effect while the robot waits for someone to speak
    (live: pressed during silence, the pause only landed after
    the next thing said had been heard and answered, because this call
    blocks until then). Keeps a
    small `preroll` prebuffer so the start of a word isn't cut off: onset is
    detected with a delay of onset_chunks, and without a prebuffer the first
    ~200 ms would be lost.

    A burst shorter than `min_chunks` is dropped and collection restarts —
    see MIN_UTTERANCE_CHUNKS. The preroll is NOT counted towards the length:
    it is padding around the sound, not evidence that anyone spoke.
    """
    ring: list[np.ndarray] = []
    captured: list[np.ndarray] = []
    # Chunks that were actually LOUD — not merely captured. The gate keeps
    # "speaking" through hangover_chunks of silence so a pause mid-sentence
    # does not end the turn, so counting captured chunks would score a
    # two-chunk cough as ten and let it straight through.
    loud = 0
    seen = 0
    for chunk, level in chunk_rms_source:
        seen += 1
        if stop is not None and seen % stop_every == 0 and stop():
            return np.zeros(0, dtype=chunk.dtype)
        event = gate.push(level)
        if not gate.speaking and event != "end":
            # silence before the utterance starts — spin the ring prebuffer
            ring.append(chunk)
            if len(ring) > preroll:
                ring.pop(0)
            continue
        if event == "start":
            captured = list(ring) + [chunk]
            ring = []
            # Counted from the run that opened the gate, never from before
            # it: clicks while nobody spoke used to add up, and a cough then
            # cleared the minimum on their account.
            loud = gate.onset_chunks
            continue
        if level >= gate.threshold:
            loud += 1
        captured.append(chunk)
        if event == "end" or len(captured) >= max_chunks:
            if loud < min_chunks:
                # Too brief to be speech — throw it away and keep listening
                # rather than spending a whole turn (ASR, LLM, TTS, the robot
                # speaking) on a noise.
                captured = []
                loud = 0
                continue
            return np.concatenate(captured) if captured else None
    return np.concatenate(captured) if captured and loud >= min_chunks else None


def calibrate_threshold(chunk_rms_source, seconds: float = 1.0,
                        chunk_ms: int = 100, multiplier: float = 3.0,
                        floor: float = 0.01) -> float:
    """Speech threshold from background noise: median silence loudness × multiplier.

    The `floor` lower bound guards against a perfectly silent microphone (a
    median of 0 would give a zero threshold and a permanent trigger).
    """
    n = max(1, int(seconds * 1000 / chunk_ms))
    levels = []
    for _, level in chunk_rms_source:
        levels.append(level)
        if len(levels) >= n:
            break
    baseline = float(np.median(levels)) if levels else 0.0
    return max(baseline * multiplier, floor)
