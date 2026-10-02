"""InflectSynthesizer.speak on empty text.

An empty phrase must return an empty array via the short-circuit path, WITHOUT touching the model: the
`__init__` constructor loads the espeak frontend and LiteRT weights from
disk, which is heavy and not what needs checking here. The instance is
built without `__init__` (object.__new__), so any access to self._tts would
raise AttributeError — if speak() failed to short-circuit on empty text, the
test would catch that as an error rather than passing by accident.
"""

import numpy as np

from emulator.inflect_tts import SAMPLE_RATE, InflectSynthesizer


def _bare_synth() -> InflectSynthesizer:
    """Instance without loading the model — only speak() should ever run here."""
    return object.__new__(InflectSynthesizer)


def test_speak_empty_text_returns_empty_float32_array():
    synth = _bare_synth()
    out = synth.speak("")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.size == 0


def test_speak_whitespace_only_text_is_also_short_circuited():
    synth = _bare_synth()
    out = synth.speak("   \n\t  ")
    assert out.size == 0


def test_sample_rate_matches_measured_inflect_output():
    assert SAMPLE_RATE == 24000


def test_init_loads_the_runtime_from_the_models_own_folder(tmp_path, monkeypatch):
    # say.py ships inside the model's folder on the Hub; the synthesizer
    # imports it from there and points it at the folder's frontend.
    import sys
    import types

    seen = {}
    fake_say = types.ModuleType("say")
    fake_say.InflectTTS = lambda **kw: seen.update(kw) or types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "say", fake_say)
    monkeypatch.setattr(sys, "path", list(sys.path))

    synth = InflectSynthesizer(tmp_path)
    assert str(tmp_path) in sys.path
    assert seen["models_dir"] == str(tmp_path)
    assert seen["frontend_dir"] == str(tmp_path / "frontend")
    assert synth.sample_rate == SAMPLE_RATE
