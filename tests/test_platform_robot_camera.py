"""RobotCameraSource: the VideoSource pulling JPEG frames from the
robot's own camera service, over HTTP.

Same fake-HTTP-layer approach as tests/test_detect_source.py: monkeypatch
`urllib.request.urlopen` so no test ever opens a socket, spawns a camera, or
needs the robot. Most tests use `_idle_source()` (the real background thread
is never started — a test-only constructor seam, see robot_camera.py) and
drive `_cycle()` by hand for an exact, deterministic fetch count, mirroring
how tests/test_detect_source.py drives RemoteDetectSource._cycle() directly
rather than racing a real thread. Two tests exercise the real background
thread (short poll_interval + the same `_wait_for` poll helper as
tests/test_platform_mac.py) to prove the threading/close() contract itself.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
from PIL import Image

from demo.platform.robot_camera import FRAME_PATH, DEFAULT_PORT, RobotCameraSource, frame_url


def _jpeg_bytes(color=(10, 20, 30), size=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


def _fail(*_a, **_k):
    raise OSError("robot camera down")


def _wait_for(predicate, timeout=2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


def _idle_source(url="http://x" + FRAME_PATH) -> RobotCameraSource:
    """A source with NO background thread running — every deterministic
    test below drives `_cycle()` by hand instead, so exactly as many
    fetches happen as the test asks for (no race against a real thread that
    would otherwise start polling the instant __init__ returns)."""
    return RobotCameraSource(url, _autostart=False)


# --- frame_url: the one place the URL is assembled ---

def test_frame_url_assembles_host_and_port():
    assert frame_url("192.168.1.50", 8100) == f"http://192.168.1.50:8100{FRAME_PATH}"


def test_frame_url_uses_default_port_when_omitted():
    assert frame_url("192.168.1.50") == f"http://192.168.1.50:{DEFAULT_PORT}{FRAME_PATH}"


# --- before the first frame ---

def test_latest_is_none_before_any_fetch():
    assert _idle_source().latest() is None


def test_latest_png_is_none_before_any_fetch():
    assert _idle_source().latest_png() is None


def test_alive_is_true_before_any_fetch_has_happened():
    # Parity with CameraStream: absence of a frame isn't
    # death — run_voice's startup warm-up wait only checks latest(), and the
    # health print only fires on a TRANSITION away from this initial True.
    assert _idle_source().alive is True


# --- one poll step (_cycle), driven deterministically ---

def test_cycle_decodes_a_successful_fetch_into_an_rgb_frame(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = _idle_source()
    source._cycle()
    frame = source.latest()
    assert frame.shape == (4, 4, 3)
    assert frame.dtype == np.uint8


def test_latest_is_a_copy_not_shared_state(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = _idle_source()
    source._cycle()
    frame = source.latest()
    frame[:] = 255
    frame2 = source.latest()
    assert not np.array_equal(frame, frame2), "latest() must return a copy, not shared state"


def test_latest_png_returns_real_png_bytes(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = _idle_source()
    source._cycle()
    png = source.latest_png()
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_failure_is_logged_and_counts_as_a_failure(monkeypatch, caplog):
    monkeypatch.setattr("urllib.request.urlopen", _fail)
    source = _idle_source()
    with caplog.at_level(logging.WARNING):
        source._cycle()
    assert source.latest() is None
    assert any("OSError" in r.message and "robot camera down" in r.message
               for r in caplog.records)


def test_bad_jpeg_is_logged_and_counts_as_a_failure(monkeypatch, caplog):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(b"not a jpeg"))
    source = _idle_source()
    with caplog.at_level(logging.WARNING):
        source._cycle()
    assert source.latest() is None
    assert any("bad JPEG" in r.message for r in caplog.records)


def test_alive_flips_false_after_consecutive_failures_then_recovers(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fail)
    source = _idle_source()
    for _ in range(4):
        source._cycle()
        assert source.alive is True
    source._cycle()  # 5th consecutive failure
    assert source.alive is False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source._cycle()
    assert source.alive is True


def test_a_single_failure_amid_successes_does_not_flip_alive(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = _idle_source()
    source._cycle()
    assert source.alive is True
    monkeypatch.setattr("urllib.request.urlopen", _fail)
    source._cycle()
    assert source.alive is True, "one dropped packet must not flip the health signal"


# --- real background thread: threading + close() ---

def test_background_thread_updates_latest_without_blocking_the_caller(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = RobotCameraSource("http://x" + FRAME_PATH, poll_interval=0.01)
    try:
        _wait_for(lambda: source.latest() is not None)
        frame = source.latest()
        assert frame.shape == (4, 4, 3)
    finally:
        source.close()


def test_close_terminates_the_background_thread(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResponse(_jpeg_bytes()))
    source = RobotCameraSource("http://x" + FRAME_PATH, poll_interval=0.01)
    _wait_for(lambda: source.latest() is not None)
    source.close()
    assert not source._thread.is_alive(), "close() must join the poll thread"


def test_a_frame_older_than_the_stale_limit_is_not_now(monkeypatch):
    # A camera that stopped must not keep handing its last picture to the
    # model as "your camera, right now".
    now = [100.0]
    source = RobotCameraSource("http://x" + FRAME_PATH, stale_after=3.0,
                               clock=lambda: now[0], _autostart=False)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda url, timeout=None: _FakeResponse(_jpeg_bytes()))
    source._cycle()
    assert source.latest() is not None
    monkeypatch.setattr("urllib.request.urlopen", _fail)
    now[0] = 102.0
    source._cycle()
    assert source.latest() is not None, "a short gap still counts as now"
    now[0] = 104.0
    source._cycle()
    assert source.latest() is None
