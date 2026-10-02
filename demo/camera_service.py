"""Camera service on the robot: its own CSI camera, served as JPEG over HTTP.

Reachy Mini has no ffmpeg (measured on the CM4: rpicam-jpeg/rpicam-vid,
gst-launch-1.0, v4l2-ctl only) and the camera is EXCLUSIVE — while the Pollen
daemon holds media, any rpicam process fails with "Pipeline handler in use by
another process". So this service, not the daemon, holds the camera for the
demo, and the Mac reads frames from here instead of its own webcam: what the
robot remembers has to be what it actually saw.

rpicam-jpeg costs 0.7-0.9s PER FRAME (measured) because it reinitialises the
sensor every call — far too slow for the ~4 fps detect loop. rpicam-vid
--codec mjpeg instead holds the sensor open and writes a continuous MJPEG
stream to stdout; this module keeps ONE such process alive, parses JPEG
frames out of the stream, and always serves only the newest one.

The microphone lives in this same process, for the same reason the camera
does: one thing to start, one thing to stop, one place that cannot leak a
capture process. Unlike the camera (exclusive, held by the Pollen daemon
whenever this service isn't running), the mic is FREE while the daemon runs
(measured: `arecord` captures alongside it with real signal) — so, unlike
CameraCapture, MicCapture never has to contend with the daemon for the
device, only with the hardware's own constraints (see build_mic_command).
A VAD/ASR consumer needs an unbroken, in-order STREAM of recent samples, not
one "latest" snapshot the way a video viewer does, so /audio fans captured
PCM chunks out to subscribers instead of exposing a single `latest()` value
— see MicCapture's docstring for how that keeps a slow or absent client from
growing memory without bound.

Shaped like the laptop's demo/detect_service.py (argparse --host/--port, quiet
logging on 2xx), with a localhost default and a loud warning when bound wider,
but it must run under the robot's plain system python 3.12, where
PIL/picamera2/numpy are NOT installed (numpy exists only inside the daemon's
separate venv) — so, unlike detect_service.py, this module is stdlib-only and
importable on the Mac purely for testing. The HTTP server is threaded
(ThreadingHTTPServer, not detect_service.py's single-threaded HTTPServer):
/audio is a long-lived connection that stays open for as long as the Mac is
listening, and a single-threaded server would leave /frame and /health
unanswered — silently starving the video half of the demo — for the entire time
someone is listening to the mic.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import select
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Protocol

# Vendored fallback, not a hard import: this file is COPIED to the robot on
# its own (the repo is not checked out there), so `demo` is not importable in
# that context — the real deployment would die at import. Verified live: the
# robot's python raised ModuleNotFoundError for `demo` while the Mac's test
# suite stayed green, which is exactly the gap a same-machine test cannot see.
try:
    from demo.http_util import ascii_reason
except ModuleNotFoundError:  # running as a standalone file on the robot
    def ascii_reason(text: str, limit: int = 200) -> str:
        """Same as demo/http_util.py's: ASCII only (http.server encodes the
        reason phrase as latin-1), and no control characters (a CR/LF there
        would split the response)."""
        ascii_text = text[:limit].encode("ascii", "replace").decode("ascii")
        return "".join(c if 0x20 <= ord(c) < 0x7f else " " for c in ascii_text)

LOG = logging.getLogger(__name__)

_SOI = b"\xff\xd8"  # JPEG Start Of Image
_EOI = b"\xff\xd9"  # JPEG End Of Image

# A "frame" that grows past this without an EOI is not a frame — either a
# corrupt/torn stream or garbage that happened to contain an SOI. Without a
# cap, that buffer would grow forever for as long as no EOI ever turns up,
# which is exactly the failure mode ("unbounded buffering") a demo running
# The service's address, imported by the Mac-side client
# (demo/platform/robot_camera.py) so the two halves cannot drift apart.
# 9700 sits beside the laptop's detect_service on 9600.
DEFAULT_PORT = 9700
FRAME_PATH = "/frame"
HEALTH_PATH = "/health"
AUDIO_PATH = "/audio"
MIC_HEALTH_PATH = "/mic/health"

# What we ask rpicam-vid for, and why it is 16:9. The robot's camera is an
# IMX708 wide, natively 4608x2592 — 16:9. Asked for 640x480 libcamera logs
# `Mode selection for 1728:1296`: it picks the full-sensor 2304x1296 mode and
# THROWS A QUARTER OF THE WIDTH AWAY to make 4:3. Asked for 960x540 it logs
# `Mode selection for 2304:1296` — the whole width, the same sensor mode. So
# the robot had been seeing 75% of what its camera sees, and no amount of
# layout work on the dashboard brings that back.
# Cost, measured over the 371 stored frames (median 39 KB each): 66 KB per
# frame, ~263 KB/s at the detect loop's 4 fps against 156 KB/s at 640x480.
# YOLO and SigLIP resize to their own inputs regardless, so what this buys is
# field of view, not recognition.
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540

# unattended can't afford.
MAX_FRAME_BYTES = 4 * 1024 * 1024

READ_CHUNK = 64 * 1024

# The camera or mic is considered dead (health signal false) once this long
# has passed with no new frame/chunk — long enough to absorb a slow
# individual read, short enough that "silently stopped producing" is caught
# within a couple of seconds, not minutes.
STALE_AFTER_S = 3.0

# A restart run must last at least this long before its own crash re-opens
# the backoff from the start again — otherwise one long healthy stretch
# followed by a single crash would restart instantly (good), but a
# *persistently* failing camera/mic (e.g. camera still held by the daemon,
# or a mic device string that doesn't exist) would only ever back off from
# its very first failure and never grow the wait.
BACKOFF_RESET_S = 30.0
INITIAL_BACKOFF_S = 0.5
MAX_BACKOFF_S = 10.0

# Reachy Mini Audio (USB 38fb:1001) — measured on hardware: 16 kHz S16_LE
# opens fine, but `-c 1` fails outright with "Channels count non available".
# It is STEREO-ONLY; mixing down to mono (the demo pipeline's format) is a
# downstream (Mac-side) concern, same principle as CameraCapture serving
# rpicam-vid's native MJPEG rather than re-encoding on the robot.
DEFAULT_MIC_DEVICE = "hw:0,0"
DEFAULT_MIC_RATE = 16000
DEFAULT_MIC_CHANNELS = 2
DEFAULT_MIC_SAMPLE_WIDTH = 2  # bytes per sample, S16_LE
DEFAULT_MIC_CHUNK_MS = 100    # matches MicStream's Mac-side chunk_ms default

# Bounded per-subscriber queue: at the default 100ms/chunk this is ~1.6s of
# audio before a slow subscriber starts losing the OLDEST queued chunk (see
# MicCapture._broadcast) rather than growing without bound — the same
# drop-oldest policy demo/display/web.py's SSE broadcaster uses for the same
# reason (a dead/slow client must cost bounded memory, not unbounded).
MIC_SUB_QUEUE_MAXSIZE = 16

# How often the /audio handler wakes up while the mic is quiet to check
# whether the client is still there (see _peer_closed). A silent room is a
# legitimate state — nothing ever gets BROADCAST to notice a disconnect
# through a failed write — so without this poll, a client that vanishes
# during silence would leak its handler thread and subscription for as long
# as the service runs, growing by one on every reconnect over an hours-long demo.
AUDIO_IDLE_POLL_S = 0.2


class MjpegFrameParser:
    """Extracts complete JPEG frames from a raw MJPEG byte stream.

    rpicam-vid --codec mjpeg writes frames back-to-back on stdout with no
    multipart boundary, so this just tracks SOI/EOI markers. `feed()` is fed
    arbitrarily-sized chunks (a `read()` can return a partial frame, several
    frames, or land mid-marker) and returns whatever complete frames that
    chunk completed, buffering the remainder for next time.
    """

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES) -> None:
        self._buf = bytearray()
        self._max = max_frame_bytes

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buf.find(_SOI)
            if start == -1:
                # No frame start anywhere in the buffer (pure inter-frame
                # garbage, or none has arrived yet). Keep a single trailing
                # 0xFF in case it's the first byte of a marker split across
                # reads; anything else is safe to drop.
                self._buf = self._buf[-1:] if self._buf[-1:] == b"\xff" else bytearray()
                break
            if start > 0:
                del self._buf[:start]  # discard garbage before the marker
            end = self._buf.find(_EOI, len(_SOI))
            if end == -1:
                if len(self._buf) > self._max:
                    # Runaway "frame" — this SOI never closes. Drop it and
                    # resync on whatever SOI comes next, rather than
                    # buffering an unbounded, never-completing frame.
                    LOG.warning(
                        "camera_service: dropping oversized/unterminated "
                        "frame candidate (>%d bytes) — resyncing", self._max)
                    del self._buf[:len(_SOI)]
                    continue
                break  # frame incomplete — wait for more data
            frame_len = end + len(_EOI)
            if frame_len > self._max:
                # An EOI eventually turned up, but only after a span larger
                # than any real frame at this resolution/quality — almost
                # certainly the "SOI" was a stray byte pair inside garbage,
                # not a real frame start. Drop it and resync on the next SOI
                # rather than emitting a bogus, oversized frame.
                LOG.warning(
                    "camera_service: dropping oversized frame candidate "
                    "(%d bytes) — resyncing", frame_len)
                del self._buf[:len(_SOI)]
                continue
            frames.append(bytes(self._buf[:frame_len]))
            del self._buf[:frame_len]
        return frames


class _Process(Protocol):
    """The slice of subprocess.Popen this module actually uses — narrow so
    tests can substitute a fake without spawning rpicam-vid or arecord.
    Shared by CameraCapture and MicCapture: both supervise exactly one
    long-lived subprocess the same way."""

    stdout: object | None
    stderr: object | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int | None: ...


def _terminate_process(proc: _Process, name: str, timeout: float = 5.0) -> None:
    """Stop a supervised subprocess for good, from CameraCapture.stop() or
    MicCapture.stop(). A process left running after the service exits would
    hold the camera/mic exclusively (camera) or just leak a file descriptor
    forever (mic) until someone SSHes into the robot to kill it by hand —
    the user can't always reach it physically.
    """
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        LOG.warning("camera_service: %s did not exit in time, killing", name)
        proc.kill()
        proc.wait()


def _spawn(cmd: list[str]) -> subprocess.Popen:
    """Start a capture process with its stderr in a temporary file, not a
    pipe. Nothing reads stderr until the process exits (_describe_exit), and
    an undrained pipe blocks the writer once the OS buffer fills (64 KiB) —
    the camera would freeze hours in, with no error anywhere. A file never
    blocks. Popen leaves `.stderr` None for a file it did not open itself, so
    the file is attached there by hand: without that, the one message that
    says why a capture died ("Pipeline handler in use by another process")
    was never read."""
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
    proc.stderr = log
    return proc


def _describe_exit(proc: _Process, name: str) -> str:
    """Best-effort diagnostics for why a supervised subprocess (rpicam-vid or
    arecord) stopped producing data. This is the ONLY place that turns
    "silently gone" into a message a human can act on — e.g. rpicam-vid's own
    "Pipeline handler in use by another process" when the daemon holds media,
    or arecord's "no such device" for a wrong --mic-device. Shared between
    CameraCapture and MicCapture so this handling can't drift between the
    video and audio capture paths.
    """
    stderr_tail = b""
    if proc.stderr is not None:
        try:
            # A file handle is positioned at the end after the child wrote to
            # it; a pipe has no seek. Rewind when we can.
            if hasattr(proc.stderr, "seek"):
                try:
                    proc.stderr.seek(0)
                except (OSError, ValueError):
                    pass
            stderr_tail = proc.stderr.read() or b""
        except (OSError, ValueError) as exc:
            # Diagnostics-only: losing the stderr tail must not stop
            # shutdown/restart, just be visible that it was lost.
            LOG.warning("camera_service: could not read %s stderr: %s",
                       name, exc)
        finally:
            proc.stderr.close()
    if proc.stdout is not None:
        proc.stdout.close()
    code = proc.poll()
    if code is None:
        # stdout hit EOF but the process hasn't reaped yet — give it a
        # moment so the reported code reflects the real exit, not "still running".
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        code = proc.poll()
    detail = stderr_tail.decode(errors="replace").strip()[-500:]
    return (f"{name} exited (code {code}): {detail}" if detail
            else f"{name} exited (code {code})")


def _peer_closed(sock: socket.socket) -> bool:
    """True once the client has sent a TCP FIN (a graceful close).

    Checked with a non-consuming MSG_PEEK, so it never steals a byte a real
    request would need — used only by the /audio handler between broadcasts
    to notice an abandoned connection that a silent mic would otherwise
    never surface (nothing is ever WRITTEN to trigger the usual
    BrokenPipeError path).
    """
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    try:
        return sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True  # a broken socket is as gone as a closed one


def build_capture_command(width: int, height: int, fps: int, quality: int) -> list[str]:
    """rpicam-vid invocation producing an MJPEG stream on stdout (`-o -`).

    `-t 0`: run until killed — any fixed --timeout would end the demo's
    camera on its own partway through. `--nopreview`: the robot has no
    display attached for rpicam to open a preview window on.
    """
    return [
        "rpicam-vid", "--codec", "mjpeg", "-t", "0", "-o", "-", "--nopreview",
        "--width", str(width), "--height", str(height),
        "--framerate", str(fps), "--quality", str(quality),
    ]


class CameraCapture:
    """Owns the long-lived rpicam-vid process; always exposes only the
    newest decoded JPEG frame.

    Only the newest frame is ever kept (no queue of pending frames), so a
    viewer that never calls `latest()` costs no extra memory — the exact
    "unbounded buffering for a slow consumer" failure mode this must avoid.

    `process_factory` is a zero-arg callable returning something shaped like
    subprocess.Popen (stdout/stderr pipes, poll/terminate/kill/wait) — real
    usage defaults to spawning rpicam-vid; tests inject a fake so no test
    ever spawns a real process or touches the camera.
    """

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 fps: int = 15, quality: int = 80,
                 process_factory: Callable[[], _Process] | None = None,
                 initial_backoff: float = INITIAL_BACKOFF_S,
                 max_backoff: float = MAX_BACKOFF_S) -> None:
        cmd = build_capture_command(width, height, fps, quality)
        self._process_factory = process_factory or (lambda: _spawn(cmd))
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._lock = threading.Lock()
        self._latest: bytes | None = None
        self._last_frame_time: float | None = None
        self._last_error: str | None = None
        self._restarts = 0
        self._proc: _Process | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._supervise, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the supervisor and kill the capture process. Must be called
        on shutdown: an rpicam-vid left running would hold the camera
        exclusively until someone SSHes into the robot to kill it by hand —
        the user can't always reach it physically."""
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            _terminate_process(proc, "rpicam-vid")
        self._thread.join(timeout=6)

    def latest(self) -> bytes | None:
        with self._lock:
            return self._latest

    def healthy(self) -> bool:
        with self._lock:
            last = self._last_frame_time
        return last is not None and (time.monotonic() - last) < STALE_AFTER_S

    def status(self) -> dict:
        """Everything an HTTP caller needs to judge the camera's health, in
        one snapshot — the failure that matters on stage is the camera going
        silently dead while /frame keeps answering with a stale image."""
        with self._lock:
            last = self._last_frame_time
            last_error = self._last_error
            restarts = self._restarts
        age = None if last is None else time.monotonic() - last
        return {
            "healthy": last is not None and age < STALE_AFTER_S,
            "frame_age_s": age,
            "restarts": restarts,
            "last_error": last_error,
        }

    # — capture loop —

    def _supervise(self) -> None:
        backoff = self._initial_backoff
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._run_once()
            except Exception as exc:  # noqa: BLE001 — top-level thread boundary, must not die silently
                LOG.exception("camera_service: capture loop crashed: %s",
                              type(exc).__name__)
                with self._lock:
                    self._last_error = f"capture loop crashed: {exc}"
            if self._stop.is_set():
                return
            if time.monotonic() - started > BACKOFF_RESET_S:
                backoff = self._initial_backoff  # a long healthy run forgives past failures
            self._restarts += 1
            with self._lock:
                last_error = self._last_error
            LOG.warning(
                "camera_service: rpicam-vid exited, restarting in %.1fs "
                "(attempt %d): %s", backoff, self._restarts, last_error)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    def _run_once(self) -> None:
        """Spawn rpicam-vid, read frames until it exits, record why. Split
        out of `_supervise` so tests can drive exactly one capture attempt
        deterministically, without threads or sleeps."""
        proc = self._process_factory()
        with self._lock:
            self._proc = proc
        if self._stop.is_set():
            # stop() may have already run (and found no process to
            # terminate) before this spawn finished — terminate immediately
            # rather than leaving an orphaned rpicam-vid holding the camera.
            proc.terminate()
        try:
            self._read_frames(proc.stdout)
        finally:
            self._collect_exit(proc)
            with self._lock:
                self._proc = None

    def _read_frames(self, stream) -> None:
        parser = MjpegFrameParser()
        while not self._stop.is_set():
            chunk = stream.read(READ_CHUNK)
            if not chunk:
                return  # EOF: rpicam-vid exited or the pipe closed
            for frame in parser.feed(chunk):
                with self._lock:
                    self._latest = frame
                    self._last_frame_time = time.monotonic()

    def _collect_exit(self, proc: _Process) -> None:
        """Diagnostics for why rpicam-vid stopped producing frames. This is
        the ONLY place that turns "camera silently gone" into a message a
        human can act on — see _describe_exit."""
        msg = _describe_exit(proc, "rpicam-vid")
        with self._lock:
            self._last_error = msg


def build_mic_command(device: str, rate: int, channels: int) -> list[str]:
    """arecord invocation for continuous raw capture.

    `-t raw`: no WAV header. A WAV header needs the total data length up
    front, which a run-until-killed capture can never supply — arecord's
    DEFAULT file type (with no `-t`) is "wav", and measured on hardware that
    put a spurious 44-byte header before the first sample: `arecord -D
    hw:0,0 -f S16_LE -r 16000 -c 2` (default type) produced 128044 bytes for
    2s of audio — exactly 128000 bytes of PCM (16000 Hz * 2 ch * 2 bytes *
    2s) plus that 44-byte header. `-t raw` means every byte on stdout is a
    sample, so this module never has to detect and skip a header.
    """
    return ["arecord", "-D", device, "-f", "S16_LE", "-r", str(rate),
            "-c", str(channels), "-t", "raw", "-"]


class MicCapture:
    """Owns the long-lived arecord process; fans out captured PCM chunks to
    any number of live HTTP subscribers.

    Unlike CameraCapture, which only ever needs to hand back the newest
    frame (a viewer that misses a few frames hasn't lost anything a human
    would notice), a VAD/ASR consumer needs an UNBROKEN, IN-ORDER stream of
    recent samples — dropping the middle of a word would corrupt every
    transcription after it. So instead of one shared "latest" value, each
    subscriber gets its own bounded queue.Queue that every captured chunk is
    pushed to (same drop-oldest-on-full pattern as demo/display/web.py's SSE
    broadcaster): a slow subscriber loses its OLDEST unread audio, not
    newest, and can never make this grow past MIC_SUB_QUEUE_MAXSIZE chunks.

    A brand-new subscriber's queue starts EMPTY — subscribe() never hands
    back anything captured before the call. That is what makes reconnecting
    safe: a client that was away for a minute cannot be handed a minute of
    stale backlog on reconnect, and with zero subscribers connected, chunks
    are broadcast to an empty list and never stored anywhere at all — an
    absent client costs no memory, exactly like an absent frame viewer.
    """

    def __init__(self, device: str = DEFAULT_MIC_DEVICE, rate: int = DEFAULT_MIC_RATE,
                 channels: int = DEFAULT_MIC_CHANNELS,
                 chunk_ms: int = DEFAULT_MIC_CHUNK_MS,
                 process_factory: Callable[[], _Process] | None = None,
                 initial_backoff: float = INITIAL_BACKOFF_S,
                 max_backoff: float = MAX_BACKOFF_S) -> None:
        cmd = build_mic_command(device, rate, channels)
        self.rate = rate
        self.channels = channels
        self.sample_width = DEFAULT_MIC_SAMPLE_WIDTH
        # Fixed-size reads (not "whatever's available", unlike the camera's
        # opportunistic READ_CHUNK) so every broadcast chunk is exactly
        # chunk_ms of audio, sample-aligned across both channels — a torn
        # read would shift stereo channels against each other for every
        # subscriber downstream, and this keeps chunk cadence predictable
        # for the health signal and for tests.
        self._chunk_bytes = (max(1, int(rate * chunk_ms / 1000))
                             * channels * self.sample_width)
        self._process_factory = process_factory or (lambda: _spawn(cmd))
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._lock = threading.Lock()
        self._last_chunk_time: float | None = None
        self._last_error: str | None = None
        self._restarts = 0
        self._proc: _Process | None = None
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._supervise, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the supervisor and kill the capture process. Must be called
        on shutdown: an arecord left running would leak the process (and,
        while it runs, the mic's file descriptor) until someone SSHes into
        the robot to kill it by hand — the user can't always reach it
        physically."""
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            _terminate_process(proc, "arecord")
        self._thread.join(timeout=6)

    def healthy(self) -> bool:
        with self._lock:
            last = self._last_chunk_time
        return last is not None and (time.monotonic() - last) < STALE_AFTER_S

    def status(self) -> dict:
        """Everything an HTTP caller needs to judge the mic's health, in one
        snapshot — mirrors CameraCapture.status()."""
        with self._lock:
            last = self._last_chunk_time
            last_error = self._last_error
            restarts = self._restarts
        age = None if last is None else time.monotonic() - last
        return {
            "healthy": last is not None and age < STALE_AFTER_S,
            "chunk_age_s": age,
            "restarts": restarts,
            "last_error": last_error,
        }

    # — subscriptions —

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=MIC_SUB_QUEUE_MAXSIZE)
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _broadcast(self, chunk: bytes) -> None:
        with self._subs_lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    # Slow subscriber — drop the OLDEST queued chunk and push
                    # the fresh one, rather than blocking the capture thread
                    # (which would stall every other subscriber too) or
                    # growing this queue without bound.
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(chunk)
                    except queue.Full:
                        pass

    # — capture loop —

    def _supervise(self) -> None:
        backoff = self._initial_backoff
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._run_once()
            except Exception as exc:  # noqa: BLE001 — top-level thread boundary, must not die silently
                LOG.exception("camera_service: mic capture loop crashed: %s",
                              type(exc).__name__)
                with self._lock:
                    self._last_error = f"capture loop crashed: {exc}"
            if self._stop.is_set():
                return
            if time.monotonic() - started > BACKOFF_RESET_S:
                backoff = self._initial_backoff  # a long healthy run forgives past failures
            self._restarts += 1
            with self._lock:
                last_error = self._last_error
            LOG.warning(
                "camera_service: arecord exited, restarting in %.1fs "
                "(attempt %d): %s", backoff, self._restarts, last_error)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    def _run_once(self) -> None:
        """Spawn arecord, read chunks until it exits, record why. Split out
        of `_supervise` so tests can drive exactly one capture attempt
        deterministically, without threads or sleeps."""
        proc = self._process_factory()
        with self._lock:
            self._proc = proc
        if self._stop.is_set():
            # stop() may have already run (and found no process to
            # terminate) before this spawn finished — terminate immediately
            # rather than leaving an orphaned arecord holding the mic.
            proc.terminate()
        try:
            self._read_chunks(proc.stdout)
        finally:
            self._collect_exit(proc)
            with self._lock:
                self._proc = None

    def _read_exact(self, stream, n: int) -> bytes:
        """Read exactly n bytes, or fewer only at EOF — a short read here
        would tear a chunk mid-sample and shift the two channels against
        each other for everyone downstream."""
        buf = bytearray()
        while len(buf) < n:
            part = stream.read(n - len(buf))
            if not part:
                break
            buf.extend(part)
        return bytes(buf)

    def _read_chunks(self, stream) -> None:
        while not self._stop.is_set():
            chunk = self._read_exact(stream, self._chunk_bytes)
            if len(chunk) < self._chunk_bytes:
                # EOF (arecord exited or the pipe closed), possibly after a
                # short final read. A partial chunk is not sample-aligned —
                # broadcasting it would shift channels for every subscriber,
                # so it's discarded rather than sent half-formed.
                return
            with self._lock:
                self._last_chunk_time = time.monotonic()
            self._broadcast(chunk)

    def _collect_exit(self, proc: _Process) -> None:
        """Diagnostics for why arecord stopped producing audio. This is the
        ONLY place that turns "mic silently gone" into a message a human can
        act on — see _describe_exit."""
        msg = _describe_exit(proc, "arecord")
        with self._lock:
            self._last_error = msg


def make_handler(camera: CameraCapture, mic: MicCapture | None = None):
    class Handler(BaseHTTPRequestHandler):
        # Served by a ThreadingHTTPServer (see main()), not a single-threaded
        # one: /audio is a long-lived connection that stays open for as long
        # as the Mac is listening, and a single-threaded server would leave
        # /frame and /health unanswered — silently starving the video half of
        # the demo — for that entire time. `timeout` still bounds how long a
        # hung/slow client can tie up its own thread waiting for the next
        # request on a keep-alive connection.
        timeout = 30

        def do_GET(self):
            if self.path == FRAME_PATH:
                self._frame()
            elif self.path == HEALTH_PATH:
                self._health()
            elif self.path == AUDIO_PATH:
                self._audio()
            elif self.path == MIC_HEALTH_PATH:
                self._mic_health()
            else:
                self.send_error(404)

        def _frame(self):
            frame = camera.latest()
            if frame is not None and not camera.healthy():
                # The camera STOPPED. Serving the last good frame with a 200
                # is the dangerous case on stage: the robot keeps "seeing" a
                # scene that is no longer there, stores it as a memory, and
                # the dashboard shows it to the audience — all while nothing
                # anywhere reports a fault. The client polls /frame, not
                # /health, so staleness has to surface HERE.
                detail = camera.status().get("last_error") or (
                    "camera stopped producing frames")
                LOG.warning("camera_service: /frame is stale: %s", detail)
                self.send_error(503, ascii_reason(detail))
                return
            if frame is None:
                # No frame has EVER arrived — most likely rpicam-vid can't
                # get the camera at all (daemon holding media). That must
                # read as a specific, actionable error, not a generic 500 or
                # a silently empty body.
                detail = camera.status()["last_error"] or (
                    "camera has not produced a frame yet")
                LOG.warning("camera_service: /frame requested with no frame "
                           "available: %s", detail)
                self.send_error(503, ascii_reason(detail))
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

        def _health(self):
            status = camera.status()
            body = json.dumps(status).encode()
            # 200 when healthy, 503 when not: a caller that only checks the
            # status code (a simple watchdog/curl -f) still gets the signal;
            # the JSON body is available either way for a caller that wants detail.
            self.send_response(200 if status["healthy"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _mic_health(self):
            if mic is None:
                # The service must still work with the camera alone if the
                # mic couldn't be constructed — a specific 404, not a crash,
                # so a caller can tell "no mic on this deployment" apart from
                # "mic is unhealthy".
                self.send_error(404, ascii_reason("microphone not configured on this service"))
                return
            status = mic.status()
            body = json.dumps(status).encode()
            self.send_response(200 if status["healthy"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _audio(self):
            if mic is None:
                self.send_error(404, ascii_reason("microphone not configured on this service"))
                return
            self.send_response(200)
            # application/octet-stream + explicit X- headers, rather than an
            # audio/* MIME type: "audio/L16" implies big-endian in its usual
            # (RTP) context, but arecord -f S16_LE is little-endian — the
            # format is spelled out so a client never has to guess.
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(mic.rate))
            self.send_header("X-Channels", str(mic.channels))
            self.send_header("X-Sample-Format", "S16_LE")
            self.end_headers()
            q = mic.subscribe()
            try:
                # No backlog is ever handed to a new connection — q starts
                # empty (see MicCapture docstring). q.get() is bounded, not
                # infinite: on a quiet mic nothing is ever WRITTEN, so a
                # vanished client would never hit the BrokenPipeError below —
                # the periodic wakeup is what notices it (see _peer_closed).
                while True:
                    try:
                        chunk = q.get(timeout=AUDIO_IDLE_POLL_S)
                    except queue.Empty:
                        if _peer_closed(self.connection):
                            return
                        continue
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected mid-write — not a server error
            finally:
                mic.unsubscribe(q)

        def log_request(self, code="-", size="-"):
            # Stay quiet on 2xx (normal /frame polling, /audio connecting);
            # 4xx/5xx go to the standard log — a backstop on top of the
            # targeted LOG.warning above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def parse_args(argv=None):
    """Kept out of main() so a test can execute the real defaults — the ones
    that matter here are the capture size, because scripts/robot_service.sh
    starts this service with no --width/--height at all, so whatever stands
    here is what the robot actually films with."""
    p = argparse.ArgumentParser(
        description="Camera+mic service on the robot: rpicam-vid MJPEG and "
                    "arecord PCM, both over HTTP")
    # Listens on localhost by default: the voice loop on the robot reads it
    # over loopback. scripts/robot_service.sh passes --host 0.0.0.0, so a
    # voice loop on the laptop (`--camera robot --mic robot`) can use it too.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    # 960x540, not 640x480: 4:3 costs a quarter of the sensor's width before
    # this service ever sees the frame (see DEFAULT_WIDTH/DEFAULT_HEIGHT).
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--mic-device", default=DEFAULT_MIC_DEVICE)
    p.add_argument("--mic-rate", type=int, default=DEFAULT_MIC_RATE)
    p.add_argument("--mic-channels", type=int, default=DEFAULT_MIC_CHANNELS)
    p.add_argument("--mic-chunk-ms", type=int, default=DEFAULT_MIC_CHUNK_MS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Its own setup, not demo/logs.py: this file runs on the robot alone.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    camera = CameraCapture(width=args.width, height=args.height,
                           fps=args.fps, quality=args.quality)
    mic = MicCapture(device=args.mic_device, rate=args.mic_rate,
                     channels=args.mic_channels, chunk_ms=args.mic_chunk_ms)
    # Bind BEFORE touching the camera/mic. Starting capture first means a
    # bind failure — a stale instance still holding the port is the likely
    # one — exits with rpicam-vid/arecord already spawned and the device left
    # occupied, with no process left alive to release it. The camera is
    # exclusive: the daemon and the Reachy app then cannot use it either, and
    # recovering needs an SSH session to a robot that may be in another room.
    # ThreadingHTTPServer, not HTTPServer: /audio holds its connection open
    # for as long as the Mac is listening, and a single-threaded server would
    # block /frame and /health for that whole time (see make_handler).
    server = ThreadingHTTPServer((args.host, args.port), make_handler(camera, mic))
    camera.start()
    mic.start()
    print(f"camera_service on {args.host}:{args.port} "
         f"({args.width}x{args.height}@{args.fps}fps, "
         f"mic {args.mic_device} {args.mic_rate}Hz x{args.mic_channels}ch)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: bound to non-loopback host {args.host!r} — this "
             f"unauthenticated service is reachable from the network.")
    try:
        server.serve_forever()
    finally:
        # Never leave rpicam-vid/arecord running after the service exits
        # (Ctrl+C included — KeyboardInterrupt unwinds through this finally)
        # — either would strand a device until someone SSHes in, and one
        # failing to stop cleanly must not skip stopping the other.
        try:
            camera.stop()
        finally:
            mic.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
