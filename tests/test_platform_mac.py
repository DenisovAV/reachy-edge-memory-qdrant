"""MacPlatform: ffmpeg avfoundation camera/mic, afplay, robot.

We never actually launch ffmpeg — `subprocess.Popen`/`subprocess.run` in
demo.platform.mac are replaced with fakes that hand back pre-baked bytes
matching the contract (rgb24 frame, s16le PCM). We only check that
MacPlatform assembles the right types and closes them correctly — the
ffmpeg protocol itself is already covered by the chunk/frame math in
test_platform_pcm.py.
"""

from __future__ import annotations

from demo.camera_service import DEFAULT_PORT, FRAME_PATH

import argparse
import io
import time

import numpy as np
import pytest

import demo.platform.mac as mac
from demo.robot_reachy import ConsoleRobot


class FakeProc:
    """Stub for subprocess.Popen: stdout is in-memory bytes, no real ffmpeg.

    `returncode` — if set, poll() returns it right away (simulates a process
    that already died on its own, without close()). `hang_first_wait` — the
    first wait() raises TimeoutExpired (simulates "didn't respond to
    terminate()"), forcing close() to fall back to kill()+another wait().
    """

    def __init__(self, stdout_data: bytes = b"", returncode: int | None = None,
                 hang_first_wait: bool = False) -> None:
        self.stdout = io.BytesIO(stdout_data)
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self._returncode = returncode
        self._hang_first_wait = hang_first_wait

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> None:
        self.wait_calls += 1
        if self._hang_first_wait and self.wait_calls == 1:
            raise mac.subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
        if self._returncode is None:
            self._returncode = 0

    def poll(self):
        return self._returncode

    @property
    def waited(self) -> bool:
        return self.wait_calls > 0


def _args(**overrides) -> argparse.Namespace:
    base = dict(video="0", audio="1", no_robot=True)
    base.update(overrides)
    return argparse.Namespace(**base)


def _wait_for(predicate, timeout=2.0) -> None:
    """Wait for the CameraStream background thread (reads a frame from
    memory — fast, but not instantaneous relative to the calling thread)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


def test_video_source_yields_a_frame_without_real_ffmpeg(monkeypatch):
    frame_bytes = bytes(640 * 480 * 3)  # one blank rgb24 frame by default
    fake = FakeProc(frame_bytes)
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    camera = platform.video_source()
    _wait_for(lambda: camera.latest() is not None)

    frame = camera.latest()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    png = camera.latest_png()
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_video_source_is_cached_singleton(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: FakeProc())
    platform = mac.MacPlatform(_args())
    assert platform.video_source() is platform.video_source()


def test_video_source_defaults_to_mac_camera_when_no_camera_flag(monkeypatch):
    # A bare Namespace without --camera (older callers, other tests) must
    # still get the Mac webcam, not raise.
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: FakeProc())
    platform = mac.MacPlatform(_args())
    assert isinstance(platform.video_source(), mac.CameraStream)


# --- --camera robot: the video source is a configuration choice, not a
# code fork — MacPlatform.video_source() picks the implementation, the
# voice loop/RemoteDetectSource/dashboard never know the difference. ---

class FakeRobotCameraSource:
    """Records the URL it was built with; no thread, no network."""

    def __init__(self, url: str) -> None:
        self.url = url

    def close(self) -> None:
        pass


def test_video_source_switches_to_robot_camera_when_selected(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_camera.RobotCameraSource",
                        FakeRobotCameraSource)
    platform = mac.MacPlatform(_args(camera="robot", robot_host="192.168.1.50"))
    camera = platform.video_source()
    assert isinstance(camera, FakeRobotCameraSource)
    assert camera.url == f"http://192.168.1.50:{DEFAULT_PORT}{FRAME_PATH}"


def test_video_source_robot_camera_uses_a_custom_port(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_camera.RobotCameraSource",
                        FakeRobotCameraSource)
    platform = mac.MacPlatform(
        _args(camera="robot", robot_host="10.0.0.9", robot_camera_port=9999))
    camera = platform.video_source()
    assert camera.url == f"http://10.0.0.9:9999{FRAME_PATH}"


def test_video_source_robot_camera_defaults_host_when_args_has_no_robot_host_attr(monkeypatch):
    # Same fallback as MacPlatform.robot(): a robot_host-less Namespace must
    # fall back to the mDNS default, not raise.
    monkeypatch.setattr("demo.platform.robot_camera.RobotCameraSource",
                        FakeRobotCameraSource)
    platform = mac.MacPlatform(_args(camera="robot"))
    camera = platform.video_source()
    assert camera.url == f"http://reachy-mini.local:{DEFAULT_PORT}{FRAME_PATH}"


def test_video_source_is_cached_singleton_for_robot_camera_too(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_camera.RobotCameraSource",
                        FakeRobotCameraSource)
    platform = mac.MacPlatform(_args(camera="robot"))
    assert platform.video_source() is platform.video_source()


def test_camera_marks_dead_when_reader_thread_exits(monkeypatch):
    # Empty data stream => raw_to_frame returns None immediately on the
    # first read => the reader thread instantly detects ffmpeg's "death"
    # and sets alive=False — without this, a stale frame would be
    # handed out by latest() silently and forever.
    fake = FakeProc(b"")
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    camera = platform.video_source()
    _wait_for(lambda: camera.alive is False)

    assert camera.alive is False
    assert camera.dead_since is not None


def test_camera_close_joins_reader_thread_and_closes_pipe(monkeypatch):
    fake = FakeProc(b"")  # empty stream — the reader thread exits quickly on its own
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    camera = platform.video_source()
    camera.close()

    assert not camera._thread.is_alive(), "close() must join the thread"
    assert camera._proc.stdout.closed, "close() must close the pipe"


def test_camera_close_reaps_child_after_kill(monkeypatch):
    # wait(timeout=2) times out (TimeoutExpired) => close() must kill the
    # process and call wait() AGAIN, otherwise the child is left as a
    # zombie.
    fake = FakeProc(b"", hang_first_wait=True)
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    camera = platform.video_source()
    camera.close()

    assert fake.killed
    assert fake.wait_calls == 2, "wait() must be called again after kill()"


def test_mic_source_yields_pcm_chunks_without_real_ffmpeg(monkeypatch):
    # chunk_ms=100 at 16kHz → 1600 samples → 3200 bytes of s16le per chunk.
    two_chunks = bytes(3200 * 2)
    monkeypatch.setattr(mac.subprocess, "Popen",
                        lambda *a, **k: FakeProc(two_chunks))

    platform = mac.MacPlatform(_args())
    mic = platform.mic_source()
    chunks = list(mic.chunks())
    assert len(chunks) == 2
    chunk, level = chunks[0]
    assert chunk.dtype == np.float32
    assert len(chunk) == 1600
    assert level == 0.0  # silence (zero bytes)


def test_mic_source_is_cached_singleton(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: FakeProc())
    platform = mac.MacPlatform(_args())
    assert platform.mic_source() is platform.mic_source()


def test_mic_chunks_logs_when_process_died(monkeypatch, caplog):
    # poll() is already not None from the start — simulates ffmpeg dying on
    # its own (not via close()); the generator must log this to distinguish
    # a dead mic from a normal stop.
    fake = FakeProc(b"", returncode=-9)
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    mic = platform.mic_source()
    with caplog.at_level("WARNING", logger="demo.platform.mac"):
        chunks = list(mic.chunks())

    assert chunks == []
    assert any("died" in r.message for r in caplog.records)


def test_mic_chunks_silent_on_empty_data_when_process_still_running(monkeypatch, caplog):
    # Data ran out, but the process is formally still alive (poll() is
    # None) — this is NOT the mic dying, no log needed (regression check
    # for existing behavior).
    fake = FakeProc(b"")
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    mic = platform.mic_source()
    with caplog.at_level("WARNING", logger="demo.platform.mac"):
        list(mic.chunks())

    assert not any("died" in r.message for r in caplog.records)


def test_mic_close_reaps_child_after_kill(monkeypatch):
    fake = FakeProc(b"", hang_first_wait=True)
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: fake)

    platform = mac.MacPlatform(_args())
    mic = platform.mic_source()
    mic.close()

    assert fake.killed
    assert fake.wait_calls == 2, "wait() must be called again after kill()"


def test_close_terminates_both_streams(monkeypatch):
    camera_proc = FakeProc()
    mic_proc = FakeProc()
    procs = iter([mic_proc, camera_proc])
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: next(procs))

    platform = mac.MacPlatform(_args())
    platform.mic_source()
    platform.video_source()
    platform.close()

    assert camera_proc.terminated and camera_proc.waited
    assert mic_proc.terminated and mic_proc.waited


def test_close_is_a_noop_when_nothing_was_opened(monkeypatch):
    # The platform may never have called video_source()/mic_source()
    # (e.g. an early server failure) — close() must not raise.
    platform = mac.MacPlatform(_args())
    platform.close()


def test_make_player_uses_injected_play(monkeypatch):
    calls = []
    monkeypatch.setattr("demo.audio_out._afplay",
                        lambda audio, sr: calls.append((len(audio), sr)))

    platform = mac.MacPlatform(_args())
    player = platform.make_player(16000)
    player.feed(np.zeros(800, dtype=np.float32))
    player.close()

    assert calls == [(800, 16000)]


# --- --mic robot: audio input is a configuration choice, not a code fork ---

class FakeRobotMicSource:
    """Records the URL it was built with; no socket, no thread."""

    def __init__(self, url: str) -> None:
        self.url = url

    def close(self) -> None:
        pass


def test_mic_source_defaults_to_mac_mic_when_no_mic_flag(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "Popen", lambda *a, **k: FakeProc())
    platform = mac.MacPlatform(_args())
    assert isinstance(platform.mic_source(), mac.MicStream)


def test_mic_source_switches_to_robot_mic_when_selected(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_audio.RobotMicSource",
                        FakeRobotMicSource)
    platform = mac.MacPlatform(_args(mic="robot", robot_host="192.168.1.50"))
    mic = platform.mic_source()
    assert isinstance(mic, FakeRobotMicSource)
    assert mic.url == f"http://192.168.1.50:{DEFAULT_PORT}/audio"


def test_mic_source_robot_mic_uses_the_camera_ports_flag():
    # The mic lives in the SAME service/process as the camera (one arecord,
    # one port) — no separate --robot-mic-port flag exists on purpose.
    from demo.platform.robot_audio import RobotMicSource

    platform = mac.MacPlatform(
        _args(mic="robot", robot_host="10.0.0.9", robot_camera_port=9999))
    mic = platform.mic_source()
    assert isinstance(mic, RobotMicSource)
    assert mic._url == "http://10.0.0.9:9999/audio"
    mic.close()


def test_mic_source_robot_mic_defaults_host_when_args_has_no_robot_host_attr(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_audio.RobotMicSource",
                        FakeRobotMicSource)
    platform = mac.MacPlatform(_args(mic="robot"))
    mic = platform.mic_source()
    assert mic.url == f"http://reachy-mini.local:{DEFAULT_PORT}/audio"


def test_mic_source_is_cached_singleton_for_robot_mic_too(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_audio.RobotMicSource",
                        FakeRobotMicSource)
    platform = mac.MacPlatform(_args(mic="robot"))
    assert platform.mic_source() is platform.mic_source()


# --- --speaker robot: audio output is a configuration choice too ---

class FakeRobotSpeakerPlayer:
    def __init__(self, sample_rate: int, robot) -> None:
        self.sample_rate = sample_rate
        self.robot = robot


def test_make_player_switches_to_robot_speaker_when_selected(monkeypatch):
    monkeypatch.setattr("demo.platform.robot_audio.RobotSpeakerPlayer",
                        FakeRobotSpeakerPlayer)
    platform = mac.MacPlatform(_args(speaker="robot", no_robot=False,
                                     robot_host="192.168.1.50"))
    player = platform.make_player(16000)
    assert isinstance(player, FakeRobotSpeakerPlayer)
    assert player.sample_rate == 16000
    assert player.robot._base == "http://192.168.1.50:8000/api"


def test_make_player_robot_speaker_reuses_platforms_cached_robot(monkeypatch):
    # make_player must talk to the SAME HttpReachyRobot run_voice already
    # holds (platform.robot()): one client for one physical robot, so a
    # robot that is not there is known to its motion and its sound at once.
    monkeypatch.setattr("demo.platform.robot_audio.RobotSpeakerPlayer",
                        FakeRobotSpeakerPlayer)
    platform = mac.MacPlatform(_args(speaker="robot", no_robot=False))
    robot = platform.robot()
    player = platform.make_player(16000)
    assert player.robot is robot


def test_make_player_robot_speaker_rejects_no_robot():
    # --no-robot gives a ConsoleRobot, which has no play_sound_file — this
    # must fail loudly and specifically here, not three calls deep.
    platform = mac.MacPlatform(_args(speaker="robot", no_robot=True))
    with pytest.raises(ValueError):
        platform.make_player(16000)


def test_robot_is_a_cached_singleton():
    platform = mac.MacPlatform(_args(no_robot=False))
    assert platform.robot() is platform.robot()


def test_no_robot_gives_console_robot():
    platform = mac.MacPlatform(_args(no_robot=True))
    robot = platform.robot()
    assert isinstance(robot, ConsoleRobot)


def test_robot_forwards_robot_host_to_http_robot():
    # --robot-host must reach the voice path (platform.robot()) — a
    # wrong/absent forward here means every call
    # silently targets the mDNS default and times out on a network where
    # mDNS fails (client isolation, no relay).
    platform = mac.MacPlatform(_args(no_robot=False, robot_host="192.168.1.50"))
    robot = platform.robot()
    assert robot._base == "http://192.168.1.50:8000/api"


def test_robot_defaults_host_when_args_has_no_robot_host_attr():
    # A bare Namespace without --robot-host (older callers, other tests)
    # must still fall back to the mDNS default, not raise.
    platform = mac.MacPlatform(_args(no_robot=False))
    robot = platform.robot()
    assert robot._base == "http://reachy-mini.local:8000/api"


# --- one-shot ffmpeg calls get a timeout ---

class _FakeRun:
    def __init__(self, stdout=b"", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_list_devices_passes_timeout(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls.update(kwargs)
        return _FakeRun(stderr="")

    monkeypatch.setattr(mac.subprocess, "run", fake_run)
    mac.list_devices()
    assert "timeout" in calls
