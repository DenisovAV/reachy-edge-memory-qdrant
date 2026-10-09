"""Whether an utterance holds speech at all: Silero VAD, before moonshine.

The voice loop starts an utterance on loudness (demo/vad.py), so a cough, a
fan or the robot's own motors start one too, and a recognizer always writes
something. moonshine-tiny wrote something for every one of 63 noise clips —
"You" for 45 of them, at a mean token log-probability of -0.05 to -0.15, as
sure of it as of a real word, so its own confidence cannot tell noise from
speech. A speech detector can: Whisper already runs this one on the laptop
(faster-whisper's vad_filter), and here it runs in front of moonshine.

Measured on 288 utterances (24 phrases, the robot's voice and five macOS
voices, clean and under noise) and 63 noise clips (white, pink, hum, motor
whir, clicks, breath, tones; 0.8-3 s, at the loudness that starts an
utterance): with Silero's own default rule, below, every utterance holds
speech — the shortest, "Thanks." and "Bye.", 9 chunks of it — and no noise
clip does, the longest run being 7 chunks of motor whir. ~0.09 ms a chunk on
the laptop, and it stops at the first stretch of speech.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 512        # 32 ms: the model's step at 16 kHz
CONTEXT = 64       # samples of the previous chunk the model sees with each one

# Silero's own defaults (silero-vad's get_speech_timestamps): speech starts
# at 0.5, ends below 0.35 held for 100 ms, and counts from 250 ms.
THRESHOLD = 0.5
NEG_THRESHOLD = THRESHOLD - 0.15
MIN_SILENCE_SAMPLES = SAMPLE_RATE * 100 // 1000
MIN_SPEECH_SAMPLES = SAMPLE_RATE * 250 // 1000


def holds_speech(probabilities) -> bool:
    """Silero's segment rule over per-chunk speech probabilities: whether
    any stretch of speech lasts longer than MIN_SPEECH_SAMPLES."""
    start = None
    silence_from = None
    end = 0
    for i, prob in enumerate(probabilities):
        end = (i + 1) * CHUNK
        if start is None:
            if prob >= THRESHOLD:
                start, silence_from = i * CHUNK, None
            continue
        if prob >= THRESHOLD:
            silence_from = None
        elif prob < NEG_THRESHOLD:
            if silence_from is None:
                silence_from = i * CHUNK
            if i * CHUNK - silence_from >= MIN_SILENCE_SAMPLES:
                if silence_from - start > MIN_SPEECH_SAMPLES:
                    return True
                start = silence_from = None
                continue
        if silence_from is None and end - start > MIN_SPEECH_SAMPLES:
            return True
    return start is not None and end - start > MIN_SPEECH_SAMPLES


class SpeechDetector:
    """Silero VAD on the onnxruntime CPU EP, one thread: `holds_speech(pcm)`
    for float32 PCM at 16 kHz."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=options,
                                             providers=["CPUExecutionProvider"])

    def probabilities(self, pcm: np.ndarray):
        """The speech probability of each 32 ms chunk, in order — lazily, so
        holds_speech stops at the first stretch of speech."""
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros(CONTEXT, np.float32)
        rate = np.array(SAMPLE_RATE, dtype=np.int64)
        for begin in range(0, len(audio) - CHUNK + 1, CHUNK):
            chunk = audio[begin:begin + CHUNK]
            prob, state = self._session.run(
                None, {"input": np.concatenate([context, chunk])[None],
                       "state": state, "sr": rate})
            yield float(prob[0, 0])
            context = chunk[-CONTEXT:]

    def holds_speech(self, pcm: np.ndarray) -> bool:
        return holds_speech(self.probabilities(pcm))
