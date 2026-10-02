"""Speech synthesis via Inflect-Nano-v2 on LiteRT.

A streaming synthesizer exported with a fixed chunk size, which is what lets
it run on LiteRT at all (a dynamic output shape would not convert). Measured
on the robot: 2.04 s of work for 3.84 s of speech, so it talks faster than it
speaks.

Everything it needs comes from one Hugging Face repo (emulator/models.py):
the two LiteRT graphs, the runtime (`say.py`) and the text frontend. The
frontend phonemizes with espeak-ng (GPL-3.0, loaded in-process), so any reply
text can be spoken with no phoneme dictionary.

`speak(text) -> np.ndarray` plus a `sample_rate` attribute is the whole
interface the rest of the demo uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


class InflectSynthesizer:
    def __init__(self, models_dir: Path, threads: int = 4) -> None:
        # say.py ships inside the model's own folder; imported from there.
        models_dir = Path(models_dir)
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))
        from say import InflectTTS

        self._tts = InflectTTS(models_dir=str(models_dir),
                               frontend_dir=str(models_dir / "frontend"),
                               precision="fp32", threads=threads)
        self.sample_rate = SAMPLE_RATE

    def speak(self, text: str) -> np.ndarray:
        """Text -> float32 samples. Empty text yields an empty array, not an error."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._tts.say(text), dtype=np.float32)
