"""MacPlatform: where the voice loop's camera, mic, speaker and robot are.

Each is a command-line choice (`--camera`, `--mic`, `--speaker`,
`--robot-host`/`--no-robot`): the laptop's own camera and mic through ffmpeg
and afplay, or the robot's over HTTP. On the robot itself the loop runs this
same class with every choice pointed at the robot over loopback
(scripts/robot_service.sh). Pure transforms (device parser, PCM converter,
frame assembly, PNG) live in demo/platform/pcm.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

import numpy as np

from demo.platform import VideoSource
from demo.platform.pcm import (
    frame_to_png,
    parse_avfoundation_devices,
    pcm_bytes_to_float,
    raw_to_frame,
)
from demo.vad import rms

LOG = logging.getLogger(__name__)


def list_devices(timeout: float = 5.0) -> dict:
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, timeout=timeout)
    return parse_avfoundation_devices(proc.stderr)


class CameraStream:
    """Continuous stream of camera frames — the camera is always "looking".

    Like MicStream for audio: one ffmpeg process holds the camera, a
    background thread reads frames and keeps the latest one. This solves
    both the black first-frame problem (the camera is always warmed up) and
    device contention (a single holder), and the robot sees continuously
    instead of one snapshot at a time.

    Frames come in as raw rgb24 at a fixed size — so every frame is exactly
    width*height*3 bytes, and the stream is trivial to slice into frames
    without parsing JPEG.

    `alive`/`dead_since`: if ffmpeg dies (or the pipe closes), the reader
    thread exits — without this flag, `latest()` would keep silently
    returning the last saved frame FOREVER, and calling code couldn't tell
    "the camera is alive but the scene isn't changing" from "the camera
    dropped out". `alive` is set to False at exactly the moment the thread
    detects the source has died.
    """

    def __init__(self, video_index: str = "default", width: int = 640,
                 height: int = 480, fps: int = 30) -> None:
        # fps must EXACTLY match the camera's mode. The MacBook offers 15 and
        # 30, but ffmpeg is finicky about floats: "15.0 not supported", and
        # without -framerate it picks 29.97 and also fails. Exactly 30 is the
        # only value that opens.
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self.alive = True
        self.dead_since: float | None = None
        self._proc = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-framerate", str(fps),
             "-video_size", f"{width}x{height}", "-i", video_index,
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        got = 0
        while got < n:
            part = self._proc.stdout.read(n - got)
            if not part:
                return b""
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def _read_loop(self) -> None:
        while True:
            raw = self._read_exact(self._frame_bytes)
            frame = raw_to_frame(raw, self.width, self.height)
            if frame is None:
                # ffmpeg died or the pipe closed — no more frames coming.
                # Mark the stream dead so latest() doesn't silently keep
                # handing out a stale frame as if it were live, forever.
                self.alive = False
                self.dead_since = time.monotonic()
                LOG.warning(
                    "CameraStream: reader thread exiting (ffmpeg died or "
                    "pipe closed) — camera marked dead")
                return
            with self._lock:
                self._latest = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_png(self) -> bytes | None:
        frame = self.latest()
        return frame_to_png(frame) if frame is not None else None

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()  # reap the child — avoid a zombie process
        self._thread.join(timeout=2)
        if self._proc.stdout is not None:
            self._proc.stdout.close()


class MicStream:
    """Continuous PCM stream from the mic via ffmpeg.

    Reads in chunk_ms-sized pieces. `flush` discards whatever has accumulated
    (after the robot finishes speaking, so it doesn't hear its own voice).
    """

    def __init__(self, audio_index: str = "default", sample_rate: int = 16000,
                 chunk_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self._chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
        self._proc = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-i", f":{audio_index}",
             "-ar", str(sample_rate), "-ac", "1", "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def chunks(self):
        """Generator of (chunk_float32, rms) pairs while the stream is alive.

        An empty read means the stream ended — but for one of two different
        reasons: either the ffmpeg process died on its own (`poll()` is no
        longer None — a real failure, so we log it), or the stream is being
        closed normally from outside. Without the log, both cases would look
        identically silent.
        """
        while True:
            raw = self._proc.stdout.read(self._chunk_bytes)
            if not raw:
                if self._proc.poll() is not None:
                    LOG.warning(
                        "MicStream: ffmpeg process died (exit code %s) — "
                        "mic is dead, not an intentional stop",
                        self._proc.poll())
                return
            chunk = pcm_bytes_to_float(raw)
            yield chunk, rms(chunk)

    def flush(self) -> None:
        """Discard whatever has accumulated in the buffer (e.g. the robot's voice)."""
        fd = self._proc.stdout.fileno()
        os.set_blocking(fd, False)
        try:
            while True:
                data = self._proc.stdout.read(self._chunk_bytes)
                if not data:
                    break
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(fd, True)

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()  # reap the child — avoid a zombie process
        if self._proc.stdout is not None:
            self._proc.stdout.close()  # don't leak the pipe fd (CameraStream does this too)


class MacPlatform:
    """Camera, mic, player and robot, each chosen by a flag (see the module
    docstring)."""

    def __init__(self, args) -> None:
        self.args = args
        self._camera: CameraStream | None = None
        self._mic: MicStream | None = None
        # Cached like _camera/_mic above: make_player() (--speaker robot)
        # uses the SAME HttpReachyRobot instance run_voice already holds —
        # one client for one physical robot, its motion and its sound each
        # behind their own breaker (demo/robot_reachy.py).
        self._robot: object | None = None

    def video_source(self) -> VideoSource:
        # --camera {mac,robot}: the demo's hard requirement is that what
        # gets remembered is what the ROBOT saw, so this is a configuration
        # choice, not a code fork — the voice loop, RemoteDetectSource, and the
        # dashboard all consume whichever VideoSource comes back and must
        # not know the difference. Lazy import mirrors robot()/make_player()
        # below: the alternative backend's module is only pulled in once
        # actually selected. getattr guards a bare Namespace without
        # --camera (older callers, other tests), same as --robot-host below.
        if self._camera is None:
            if getattr(self.args, "camera", "mac") == "robot":
                from demo.platform.robot_camera import (DEFAULT_PORT,
                                                         RobotCameraSource,
                                                         frame_url)

                # Same host as --robot-host — the camera
                # service runs on the same CM4 as the Reachy daemon, just a
                # different port, so a plain IP on a conference network
                # fixes both the same way.
                host = getattr(self.args, "robot_host", "reachy-mini.local")
                port = getattr(self.args, "robot_camera_port", DEFAULT_PORT)
                self._camera = RobotCameraSource(frame_url(host, port))
            else:
                self._camera = CameraStream(self.args.video)
        return self._camera

    def mic_source(self):
        # --mic {mac,robot}: same configuration-not-code-fork shape as
        # --camera above — the requirement this exists for is that the
        # audience sees the ROBOT being talked to, not the Mac's mic
        # picking up the presenter instead.
        if self._mic is None:
            if getattr(self.args, "mic", "mac") == "robot":
                from demo.camera_service import DEFAULT_PORT
                from demo.platform.robot_audio import RobotMicSource, mic_url

                # Same host/port as --camera robot: the mic lives in the
                # SAME service/process on the robot as the camera ("one
                # process, one thing to start and stop" — see
                # demo/camera_service.py's module docstring), not a second
                # port, so --robot-camera-port covers both.
                host = getattr(self.args, "robot_host", "reachy-mini.local")
                port = getattr(self.args, "robot_camera_port", DEFAULT_PORT)
                self._mic = RobotMicSource(mic_url(host, port))
            else:
                self._mic = MicStream(self.args.audio, chunk_ms=100)
        return self._mic

    def make_player(self, sample_rate: int):
        # --speaker {mac,robot}: afplay locally, or the robot's own speaker
        # through the daemon's HTTP media API (the daemon holds the speaker
        # exclusively — see demo/platform/robot_audio.py's module docstring).
        if getattr(self.args, "speaker", "mac") == "robot":
            from demo.platform.robot_audio import RobotSpeakerPlayer
            from demo.robot_reachy import HttpReachyRobot

            robot = self.robot()
            if not isinstance(robot, HttpReachyRobot):
                # --no-robot gives a ConsoleRobot, which has no
                # play_sound_file — fail loudly and specifically HERE
                # rather than let an AttributeError surface three calls
                # later, inside RobotSpeakerPlayer.close(), as a generic
                # "[audio] skip" with no hint of why (demo/run_demo.py's
                # _FailureGuard still catches this either way, so the demo
                # doesn't crash — it just quietly loses sound until the
                # flag mismatch is fixed).
                raise ValueError(
                    "--speaker robot needs a real robot connection; drop "
                    "--no-robot or remove --speaker robot")
            return RobotSpeakerPlayer(sample_rate, robot)
        from demo.audio_out import StreamPlayer, _afplay

        return StreamPlayer(sample_rate, play=_afplay)

    def robot(self):
        if self._robot is None:
            from demo.robot_reachy import make_robot

            # getattr guards callers (tests) that build a bare Namespace
            # without --robot-host.
            host = getattr(self.args, "robot_host", "reachy-mini.local")
            self._robot = make_robot(use_robot=not self.args.no_robot, host=host)
        return self._robot

    def close(self) -> None:
        if self._mic is not None:
            self._mic.close()
        if self._camera is not None:
            self._camera.close()
