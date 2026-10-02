"""RobotMicSource / RobotSpeakerPlayer: the robot's ears and mouth, as the
voice loop reaches them over HTTP. No real network anywhere here — RobotMicSource is driven through
its `stream_factory` seam (fake streaming responses), and RobotSpeakerPlayer
through a fake robot object satisfying the one method it calls.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

import demo.platform.robot_audio as ra
from demo.camera_service import AUDIO_PATH, DEFAULT_PORT

SAMPLES_PER_CHUNK = ra.SAMPLES_PER_CHUNK  # 1600 at the default 16 kHz/100 ms


def _stereo_raw(left: int, right: int, n: int = SAMPLES_PER_CHUNK) -> bytes:
    """n interleaved stereo S16_LE frames, both channels constant — enough
    to check the downmix arithmetic exactly rather than approximately."""
    stereo = np.empty(n * 2, dtype="<i2")
    stereo[0::2] = left
    stereo[1::2] = right
    return stereo.tobytes()


class FakeStream:
    """Fake streaming response: pre-baked headers, and a queue of items
    `read()` hands out one at a time — either exactly enough bytes for one
    `_read_exact` call, `b""` for a clean EOF, or an Exception to raise
    (simulating a read timeout / dropped connection)."""

    def __init__(self, items, headers=None):
        self._items = list(items)
        self.headers = headers or {
            "X-Sample-Rate": "16000", "X-Channels": "2",
            "X-Sample-Format": "S16_LE"}
        self.closed = False

    def read(self, n):
        if not self._items:
            return b""
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def _factory(streams):
    """stream_factory returning the given streams/exceptions in order, one
    per call — a fresh "connection attempt" each time RobotMicSource needs
    to (re)connect."""
    it = iter(streams)

    def factory():
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return factory


def _mic(streams, **kwargs) -> ra.RobotMicSource:
    kwargs.setdefault("initial_backoff", 0.001)
    kwargs.setdefault("max_backoff", 0.001)
    return ra.RobotMicSource("http://robot:9700/audio",
                             stream_factory=_factory(streams), **kwargs)


# --- mic_url ---

def test_mic_url_uses_camera_services_default_port_and_audio_path():
    assert ra.mic_url("reachy-mini.local") == \
        f"http://reachy-mini.local:{DEFAULT_PORT}{AUDIO_PATH}"


def test_mic_url_accepts_a_custom_port():
    assert ra.mic_url("10.0.0.9", 9999) == f"http://10.0.0.9:9999{AUDIO_PATH}"


# --- chunks(): downmix + RMS on the happy path ---

def test_chunks_downmixes_stereo_channels_by_averaging():
    # +0.5 on the left, -0.5 on the right -> mono averages to ~0.
    raw = _stereo_raw(16383, -16384)
    mic = _mic([FakeStream([raw])])
    chunk, level = next(mic.chunks())
    assert chunk.dtype == np.float32
    assert len(chunk) == SAMPLES_PER_CHUNK
    assert abs(float(chunk[0])) < 1e-3
    assert level == pytest.approx(0.0, abs=1e-3)


def test_chunks_reports_rms_of_a_constant_amplitude_chunk():
    raw = _stereo_raw(16384, 16384)  # both channels identical -> exact 0.5
    mic = _mic([FakeStream([raw])])
    _, level = next(mic.chunks())
    assert level == pytest.approx(0.5, abs=0.01)


def test_chunks_yields_multiple_chunks_from_one_connection():
    raw = _stereo_raw(0, 0)
    mic = _mic([FakeStream([raw, raw])])
    gen = mic.chunks()
    chunk1, _ = next(gen)
    chunk2, _ = next(gen)
    assert len(chunk1) == len(chunk2) == SAMPLES_PER_CHUNK


# --- stalls, drops, EOF: silence instead of ending or raising ---

def test_connect_failure_yields_silence_then_recovers(caplog):
    good = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([OSError("connection refused"), good])
    gen = mic.chunks()
    with caplog.at_level("WARNING", logger="demo.platform.robot_audio"):
        chunk, level = next(gen)
    assert level == 0.0
    assert np.all(chunk == 0.0)
    assert any("connection refused" in r.message for r in caplog.records)

    chunk2, _ = next(gen)  # backoff elapses, next connect attempt succeeds
    assert len(chunk2) == SAMPLES_PER_CHUNK


def test_read_failure_mid_stream_disconnects_and_yields_silence():
    first = FakeStream([TimeoutError("timed out")])
    second = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([first, second])
    gen = mic.chunks()
    chunk, level = next(gen)
    assert level == 0.0
    assert first.closed, "a failed read must disconnect before retrying"

    chunk2, _ = next(gen)
    assert len(chunk2) == SAMPLES_PER_CHUNK


def test_clean_eof_yields_silence_and_reconnects():
    first = FakeStream([b""])  # EOF before a single byte
    second = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([first, second])
    gen = mic.chunks()
    chunk, level = next(gen)
    assert level == 0.0
    assert first.closed

    chunk2, _ = next(gen)
    assert len(chunk2) == SAMPLES_PER_CHUNK


def test_a_stall_never_raises_out_of_chunks():
    # Whatever the underlying failure, chunks() must keep producing items —
    # a caller like collect_utterance/calibrate_threshold must never see an
    # exception propagate out of this generator.
    mic = _mic([OSError("refused"), OSError("still refused"),
               FakeStream([_stereo_raw(0, 0)])])
    gen = mic.chunks()
    for _ in range(3):
        next(gen)  # must not raise


def test_wrong_sample_rate_is_rejected_not_silently_misdecoded(caplog):
    # The whole demo pipeline hardcodes 16 kHz downstream; a mismatched
    # service must be a loud, specific failure, not silently misread bytes.
    bad = FakeStream([_stereo_raw(0, 0)],
                     headers={"X-Sample-Rate": "44100", "X-Channels": "2",
                              "X-Sample-Format": "S16_LE"})
    mic = _mic([bad])
    with caplog.at_level("WARNING", logger="demo.platform.robot_audio"):
        chunk, level = next(mic.chunks())
    assert level == 0.0
    assert any("44100" in r.message for r in caplog.records)


def test_wrong_sample_format_is_rejected():
    bad = FakeStream([_stereo_raw(0, 0)],
                     headers={"X-Sample-Rate": "16000", "X-Channels": "2",
                              "X-Sample-Format": "S24_3LE"})
    mic = _mic([bad])
    _, level = next(mic.chunks())
    assert level == 0.0


def test_mono_service_is_handled_via_the_channels_header():
    # channels=1: the downmix must be a no-op, not crash on reshape(-1, 1).
    mono_raw = np.zeros(SAMPLES_PER_CHUNK, dtype="<i2")
    mono_raw[:] = 8192
    stream = FakeStream([mono_raw.tobytes()],
                        headers={"X-Sample-Rate": "16000", "X-Channels": "1",
                                 "X-Sample-Format": "S16_LE"})
    mic = _mic([stream])
    chunk, level = next(mic.chunks())
    assert len(chunk) == SAMPLES_PER_CHUNK
    assert level == pytest.approx(0.25, abs=0.01)  # 8192/32768


# --- flush(): discard whatever's in flight, resume live on reconnect ---

def test_flush_closes_the_current_connection():
    stream = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([stream])
    next(mic.chunks())  # establish the connection
    mic.flush()
    assert stream.closed


def test_flush_forces_a_reconnect_on_the_next_chunk():
    first = FakeStream([_stereo_raw(0, 0)])
    second = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([first, second])
    gen = mic.chunks()
    next(gen)
    mic.flush()
    next(gen)  # must reconnect (a second factory item was consumed)
    assert first.closed


# --- close(): the generator stops producing, no more connects ---

def test_close_before_any_read_makes_chunks_empty():
    mic = _mic([])
    mic.close()
    assert list(mic.chunks()) == []


def test_close_disconnects_an_open_stream():
    stream = FakeStream([_stereo_raw(0, 0)])
    mic = _mic([stream])
    next(mic.chunks())
    mic.close()
    assert stream.closed


# --- RobotSpeakerPlayer: buffer -> upload+play -> block for real duration ---

class FakeSpeakerRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, float]] = []

    def play_sound_file(self, wav_bytes: bytes, duration_s: float) -> None:
        self.calls.append((wav_bytes, duration_s))


def test_player_close_is_a_noop_with_nothing_fed():
    robot = FakeSpeakerRobot()
    player = ra.RobotSpeakerPlayer(16000, robot)
    player.close()
    assert robot.calls == []


def test_player_skips_empty_feed():
    robot = FakeSpeakerRobot()
    player = ra.RobotSpeakerPlayer(16000, robot)
    player.feed(np.zeros(0, dtype=np.float32))
    player.close()
    assert robot.calls == []


def test_player_close_uploads_one_wav_of_all_fed_phrases():
    robot = FakeSpeakerRobot()
    player = ra.RobotSpeakerPlayer(16000, robot)
    player.feed(np.full(8000, 0.5, dtype=np.float32))
    player.feed(np.full(8000, -0.5, dtype=np.float32))
    player.close()

    assert len(robot.calls) == 1
    wav_bytes, duration_s = robot.calls[0]
    assert duration_s == pytest.approx(16000 / 16000)  # 16000 total samples

    with wave.open(io.BytesIO(wav_bytes)) as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == 16000
        assert fh.getnframes() == 16000


def test_player_close_resets_buffer_so_a_second_close_is_a_noop():
    robot = FakeSpeakerRobot()
    player = ra.RobotSpeakerPlayer(16000, robot)
    player.feed(np.ones(100, dtype=np.float32))
    player.close()
    player.close()
    assert len(robot.calls) == 1
