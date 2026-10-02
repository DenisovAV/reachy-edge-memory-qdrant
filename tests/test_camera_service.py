"""Tests for demo/camera_service.py: MJPEG parsing, the capture supervisor's
restart/backoff/shutdown behavior, mic capture/broadcast, and the HTTP layer
— all against fake processes/streams. Never spawns rpicam-vid or arecord,
never opens a socket to a real robot.
"""
from __future__ import annotations

import http.client
import json
import queue
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from demo.camera_service import (
    AUDIO_PATH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    FRAME_PATH,
    MIC_HEALTH_PATH,
    CameraCapture,
    MicCapture,
    MjpegFrameParser,
    build_capture_command,
    build_mic_command,
    make_handler,
)

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


# --- MjpegFrameParser: pure byte-stream parsing, no I/O ---

def test_single_frame_in_one_chunk():
    parser = MjpegFrameParser()
    frame = SOI + b"jpegdata" + EOI
    assert parser.feed(frame) == [frame]


def test_frame_split_across_multiple_reads():
    parser = MjpegFrameParser()
    frame = SOI + b"jpegdata" + EOI
    assert parser.feed(frame[:5]) == []
    assert parser.feed(frame[5:10]) == []
    assert parser.feed(frame[10:]) == [frame]


def test_marker_split_exactly_at_chunk_boundary():
    # The 2-byte EOI marker itself is split across two reads.
    parser = MjpegFrameParser()
    frame = SOI + b"jpegdata" + EOI
    assert parser.feed(frame[:-1]) == []
    assert parser.feed(frame[-1:]) == [frame]


def test_garbage_between_frames_is_discarded():
    parser = MjpegFrameParser()
    frame1 = SOI + b"first" + EOI
    frame2 = SOI + b"second" + EOI
    stream = frame1 + b"\x00\x00garbage\x00" + frame2
    assert parser.feed(stream) == [frame1, frame2]


def test_multiple_frames_in_one_chunk_all_returned_in_order():
    parser = MjpegFrameParser()
    frame1 = SOI + b"a" + EOI
    frame2 = SOI + b"bb" + EOI
    frame3 = SOI + b"ccc" + EOI
    assert parser.feed(frame1 + frame2 + frame3) == [frame1, frame2, frame3]


def test_truncated_tail_is_buffered_not_emitted():
    parser = MjpegFrameParser()
    frame1 = SOI + b"first" + EOI
    truncated = SOI + b"incomplete-no-eoi-yet"
    assert parser.feed(frame1 + truncated) == [frame1]
    # The truncated tail is still waiting for its EOI — nothing to emit, and
    # feeding more data (including a fresh SOI) doesn't crash.
    assert parser.feed(b"more-of-the-same") == []


def test_truncated_tail_completes_once_the_rest_arrives():
    parser = MjpegFrameParser()
    frame = SOI + b"payload" + EOI
    assert parser.feed(frame[:6]) == []
    assert parser.feed(frame[6:]) == [frame]


def test_oversized_unterminated_frame_is_dropped_and_resynced():
    # Cap tiny so a normal-looking frame with no EOI is treated as garbage;
    # the parser must recover once a real frame follows, not wedge forever.
    parser = MjpegFrameParser(max_frame_bytes=8)
    junk = SOI + b"x" * 20  # never closes, exceeds the cap
    frame = SOI + b"ok" + EOI
    assert parser.feed(junk + frame) == [frame]


# --- build_capture_command: pure function ---

def test_build_capture_command_includes_mjpeg_and_dimensions():
    cmd = build_capture_command(width=640, height=480, fps=15, quality=80)
    assert cmd[0] == "rpicam-vid"
    assert "--codec" in cmd and "mjpeg" in cmd
    assert "640" in cmd and "480" in cmd and "15" in cmd and "80" in cmd
    assert "-o" in cmd and "-" in cmd  # stdout


def test_the_camera_is_asked_for_16_9_not_4_3():
    """640x480 makes libcamera pick `Mode selection for 1728:1296` — the
    full-sensor 2304x1296 mode with a quarter of its WIDTH thrown away. 960x540
    logs `Mode selection for 2304:1296`: the same sensor mode, the whole
    width. The robot had been seeing 75% of what its camera sees."""
    assert (DEFAULT_WIDTH, DEFAULT_HEIGHT) == (960, 540)
    assert DEFAULT_WIDTH * 9 == DEFAULT_HEIGHT * 16


def test_the_command_line_and_the_capture_agree_on_the_size():
    """scripts/robot_service.sh starts this service with no --width/--height
    at all, so the CLI default is what the robot films with — and a capture
    class that disagreed would only show up on the robot."""
    import inspect

    from demo.camera_service import parse_args

    args = parse_args([])
    assert (args.width, args.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    signature = inspect.signature(CameraCapture.__init__)
    assert signature.parameters["width"].default == DEFAULT_WIDTH
    assert signature.parameters["height"].default == DEFAULT_HEIGHT
    cmd = build_capture_command(width=args.width, height=args.height,
                                fps=args.fps, quality=args.quality)
    assert "960" in cmd and "540" in cmd


# --- Fakes standing in for subprocess.Popen ---

class FakeStdout:
    """A pipe-like object: read() blocks until a value is queued, exactly
    like a real pipe blocks until data or EOF — never returns b"" unless the
    stream is actually done, so tests can model "alive, no data yet"."""

    def __init__(self, chunks):
        self._q: queue.Queue = queue.Queue()
        for c in chunks:
            self._q.put(c)

    def read(self, n=-1):
        return self._q.get()

    def push_eof(self):
        self._q.put(b"")

    def close(self):
        pass


class _StderrPipe:
    """stderr is read via a single, no-size .read() call in _collect_exit —
    exactly what a real closed pipe's .read() returns in one shot."""

    def __init__(self, data: bytes):
        self._data = data
        self._read = False

    def read(self, n=-1):
        if self._read:
            return b""
        self._read = True
        return self._data

    def close(self):
        pass


class FakeProcess:
    """Stand-in for subprocess.Popen — CameraCapture only uses
    stdout/stderr/poll/terminate/kill/wait."""

    def __init__(self, chunks, stderr=b"", poll_results=(0,)):
        self.stdout = FakeStdout(list(chunks))
        self.stderr = _StderrPipe(stderr)
        self._poll_results = list(poll_results)
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls: list[float | None] = []

    def poll(self):
        if len(self._poll_results) > 1:
            return self._poll_results.pop(0)
        return self._poll_results[0]

    def terminate(self):
        self.terminate_called = True
        self.stdout.push_eof()

    def kill(self):
        self.kill_called = True
        self.stdout.push_eof()

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self._poll_results[-1]


# --- CameraCapture._run_once: one capture attempt, no threads ---

def test_run_once_captures_frame_and_marks_healthy():
    jpeg = SOI + b"frame-data" + EOI
    proc = FakeProcess([jpeg, b""])
    cam = CameraCapture(process_factory=lambda: proc)
    cam._run_once()
    assert cam.latest() == jpeg
    assert cam.healthy()
    assert cam.status()["last_error"] == "rpicam-vid exited (code 0)"


def test_run_once_with_no_frames_and_clean_exit_leaves_latest_none():
    proc = FakeProcess([b""])
    cam = CameraCapture(process_factory=lambda: proc)
    cam._run_once()
    assert cam.latest() is None
    assert not cam.healthy()


def test_run_once_surfaces_camera_busy_error_from_stderr():
    # This is the exact failure the demo can hit live: the Pollen daemon
    # holds media, rpicam-vid can't acquire the camera at all.
    err = b"ERROR: *** Camera: Pipeline handler in use by another process ***\n"
    proc = FakeProcess([b""], stderr=err, poll_results=(1,))
    cam = CameraCapture(process_factory=lambda: proc)
    cam._run_once()
    assert cam.latest() is None
    status = cam.status()
    assert not status["healthy"]
    assert "Pipeline handler in use by another process" in status["last_error"]
    assert "code 1" in status["last_error"]


def test_run_once_waits_for_exit_code_when_poll_is_none_right_after_eof():
    proc = FakeProcess([b""], poll_results=(None, 2))
    cam = CameraCapture(process_factory=lambda: proc)
    cam._run_once()
    assert proc.wait_calls == [2]
    assert "code 2" in cam.status()["last_error"]


# --- health signal: goes false once frames stop arriving ---

def test_healthy_is_false_before_any_frame():
    cam = CameraCapture(process_factory=lambda: FakeProcess([]))
    assert not cam.healthy()
    status = cam.status()
    assert status["healthy"] is False
    assert status["frame_age_s"] is None


def test_healthy_goes_false_after_frames_go_stale(monkeypatch):
    import demo.camera_service as camera_service

    monkeypatch.setattr(camera_service, "STALE_AFTER_S", 0.05)
    cam = CameraCapture(process_factory=lambda: FakeProcess([]))
    with cam._lock:
        cam._latest = b"jpeg-bytes"
        cam._last_frame_time = time.monotonic()
    assert cam.healthy()
    time.sleep(0.1)
    assert not cam.healthy()
    assert cam.status()["healthy"] is False


# --- supervisor: start()/stop(), restart-with-backoff, no orphaned process ---

def _wait_until(predicate, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_supervisor_restarts_with_backoff_and_logs_each_attempt(caplog):
    attempts = []

    def factory():
        attempts.append(1)
        return FakeProcess([b""], stderr=b"camera busy", poll_results=(1,))

    cam = CameraCapture(process_factory=factory, initial_backoff=0.01,
                        max_backoff=0.02)
    with caplog.at_level("WARNING", logger="demo.camera_service"):
        cam.start()
        assert _wait_until(lambda: len(attempts) >= 3)
        cam.stop()

    assert len(attempts) >= 3
    assert cam.status()["restarts"] >= 2
    assert any("restarting in" in r.message for r in caplog.records)
    assert not cam._thread.is_alive()


def test_stop_terminates_a_running_process_and_joins_the_thread():
    proc = FakeProcess([])  # alive, no frames yet — read() blocks
    cam = CameraCapture(process_factory=lambda: proc)
    cam.start()
    assert _wait_until(lambda: cam._proc is not None)
    cam.stop()
    assert proc.terminate_called
    assert not cam._thread.is_alive()


def test_stop_before_spawn_finishes_does_not_orphan_the_process():
    # Regression for the race where stop() runs, sees no process yet (still
    # spawning), and _run_once's spawn only lands afterward — that process
    # must still get terminated, not left holding the camera forever.
    proc = FakeProcess([])
    spawned = threading.Event()

    def slow_factory():
        time.sleep(0.05)
        spawned.set()
        return proc

    cam = CameraCapture(process_factory=slow_factory)
    cam.start()
    cam.stop()  # races ahead of slow_factory() returning
    assert spawned.is_set()
    assert proc.terminate_called
    assert not cam._thread.is_alive()


# --- HTTP layer: fake camera, real sockets, no rpicam/robot involved ---

class FakeCamera:
    def __init__(self, frame=None, status=None):
        self._frame = frame
        self._status = status if status is not None else {
            "healthy": True, "frame_age_s": 0.1, "restarts": 0,
            "last_error": None,
        }

    def latest(self):
        return self._frame

    def healthy(self):
        # Mirrors CameraCapture: /frame refuses to serve a frame from a
        # camera that has stopped, so the fake has to answer this too.
        return bool(self._status.get("healthy", True))

    def status(self):
        return self._status


class _RunningServer:
    def __init__(self, camera, mic=None):
        # ThreadingHTTPServer, matching main(): a test that opens /audio and
        # keeps it open must not stall a concurrent /frame or /health request
        # on the same server the way a single-threaded HTTPServer would.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(camera, mic))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def address(self):
        return self.server.server_address

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def running_server():
    servers = []

    def make(camera, mic=None):
        srv = _RunningServer(camera, mic)
        servers.append(srv)
        return srv

    yield make
    for srv in servers:
        srv.close()


def test_frame_returns_jpeg_bytes_when_available(running_server):
    jpeg = SOI + b"some-jpeg-bytes" + EOI
    srv = running_server(FakeCamera(frame=jpeg))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/frame")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "image/jpeg"
    assert resp.read() == jpeg
    conn.close()


def test_frame_returns_503_with_specific_error_when_never_captured(running_server):
    status = {"healthy": False, "frame_age_s": None, "restarts": 3,
             "last_error": "rpicam-vid exited (code 1): Pipeline handler in "
                           "use by another process"}
    srv = running_server(FakeCamera(frame=None, status=status))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/frame")
    resp = conn.getresponse()
    assert resp.status == 503
    assert "Pipeline handler" in resp.reason
    resp.read()
    conn.close()


def test_health_returns_200_and_json_when_healthy(running_server):
    srv = running_server(FakeCamera(status={
        "healthy": True, "frame_age_s": 0.2, "restarts": 0, "last_error": None}))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["healthy"] is True
    conn.close()


def test_health_returns_503_when_unhealthy(running_server):
    srv = running_server(FakeCamera(status={
        "healthy": False, "frame_age_s": 30.0, "restarts": 5,
        "last_error": "rpicam-vid exited (code 1)"}))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 503
    body = json.loads(resp.read())
    assert body["healthy"] is False
    assert body["restarts"] == 5
    conn.close()


def test_unknown_path_returns_404(running_server):
    srv = running_server(FakeCamera())
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/nope")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_frame_endpoint_refuses_to_serve_a_stale_frame():
    """A camera that died must not keep answering 200 with its last picture.

    That is the failure that matters on stage: the robot goes on "seeing" a
    scene that is no longer in front of it, stores it as a memory, and the
    dashboard shows it to the audience — with nothing reporting a fault. The
    Mac client polls /frame, never /health, so staleness has to be visible on
    this endpoint.
    """
    import http.client

    class StaleCamera:
        def latest(self):
            return b"\xff\xd8stale\xff\xd9"

        def healthy(self):
            return False

        def status(self):
            return {"last_error": "camera stopped producing frames"}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(StaleCamera()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", FRAME_PATH)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 503, "a stale frame must not be served as 200"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


# --- build_mic_command: pure function ---

def test_build_mic_command_uses_the_measured_working_format():
    # Measured on hardware: 16 kHz S16_LE opens fine, but -c 1 fails with
    # "Channels count non available" — the mic is stereo-only.
    cmd = build_mic_command("hw:0,0", rate=16000, channels=2)
    assert cmd[0] == "arecord"
    assert "-D" in cmd and "hw:0,0" in cmd
    assert "-f" in cmd and "S16_LE" in cmd
    assert "-r" in cmd and "16000" in cmd
    assert "-c" in cmd and "2" in cmd
    # -t raw: no WAV header, which would need a length a live capture can't
    # supply and would misalign every downstream chunk read from stdout.
    assert "-t" in cmd and "raw" in cmd
    assert cmd[-1] == "-"  # stdout


# --- MicCapture._read_exact: exact, sample-aligned reads ---

class _PartialReadStream:
    """A stream that hands back fewer bytes than requested, like a real pipe
    can — _read_exact must loop until it has exactly what was asked for."""

    def __init__(self, pieces):
        self._pieces = list(pieces)

    def read(self, n=-1):
        return self._pieces.pop(0) if self._pieces else b""


def test_read_exact_reassembles_a_chunk_split_across_partial_reads():
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    stream = _PartialReadStream([b"ab", b"cd", b"ef"])
    assert cam._read_exact(stream, 6) == b"abcdef"


def test_read_exact_returns_short_buffer_at_eof():
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    stream = _PartialReadStream([b"ab", b""])  # EOF after 2 of 6 bytes
    assert cam._read_exact(stream, 6) == b"ab"


# --- MicCapture._run_once: one capture attempt, no threads ---

def test_mic_run_once_captures_chunk_and_marks_healthy():
    cam = MicCapture(rate=1000, channels=1, chunk_ms=20,
                     process_factory=lambda: FakeProcess([]))
    chunk = b"\x01\x02" * (cam._chunk_bytes // 2)
    assert len(chunk) == cam._chunk_bytes
    proc = FakeProcess([chunk, b""])
    cam._process_factory = lambda: proc
    cam._run_once()
    assert cam.healthy()
    assert cam.status()["last_error"] == "arecord exited (code 0)"


def test_mic_run_once_broadcasts_captured_chunk_to_a_subscriber():
    cam = MicCapture(rate=1000, channels=1, chunk_ms=20,
                     process_factory=lambda: FakeProcess([]))
    chunk = b"\x03\x04" * (cam._chunk_bytes // 2)
    proc = FakeProcess([chunk, b""])
    cam._process_factory = lambda: proc
    q = cam.subscribe()
    cam._run_once()
    assert q.get_nowait() == chunk


def test_mic_run_once_with_no_chunks_and_clean_exit_leaves_unhealthy():
    cam = MicCapture(process_factory=lambda: FakeProcess([b""]))
    cam._run_once()
    assert not cam.healthy()


def test_mic_run_once_with_short_final_read_is_not_broadcast():
    # A short read right before EOF is not sample-aligned (a torn chunk would
    # shift the channels against each other downstream) — it must be
    # discarded, not handed to a subscriber as if it were a full chunk.
    cam = MicCapture(rate=1000, channels=1, chunk_ms=20,
                     process_factory=lambda: FakeProcess([]))
    short = b"\x00" * (cam._chunk_bytes - 1)
    proc = FakeProcess([short, b""])
    cam._process_factory = lambda: proc
    q = cam.subscribe()
    cam._run_once()
    assert not cam.healthy()
    assert q.empty()


def test_mic_run_once_surfaces_device_error_from_stderr():
    # The exact failure a wrong --mic-device or a hardware hiccup produces.
    err = b"arecord: main:832: audio open error: No such device\n"
    proc = FakeProcess([b""], stderr=err, poll_results=(1,))
    cam = MicCapture(process_factory=lambda: proc)
    cam._run_once()
    assert not cam.healthy()
    status = cam.status()
    assert "No such device" in status["last_error"]
    assert "code 1" in status["last_error"]
    assert status["last_error"].startswith("arecord exited")


# --- health signal: goes false once chunks stop arriving ---

def test_mic_healthy_is_false_before_any_chunk():
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    assert not cam.healthy()
    status = cam.status()
    assert status["healthy"] is False
    assert status["chunk_age_s"] is None


def test_mic_healthy_goes_false_after_chunks_go_stale(monkeypatch):
    import demo.camera_service as camera_service

    monkeypatch.setattr(camera_service, "STALE_AFTER_S", 0.05)
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    with cam._lock:
        cam._last_chunk_time = time.monotonic()
    assert cam.healthy()
    time.sleep(0.1)
    assert not cam.healthy()
    assert cam.status()["healthy"] is False


# --- bounded broadcast: slow/absent subscribers cost bounded memory ---

def test_mic_broadcast_drops_oldest_chunk_when_a_subscriber_queue_is_full(monkeypatch):
    import demo.camera_service as camera_service

    monkeypatch.setattr(camera_service, "MIC_SUB_QUEUE_MAXSIZE", 2)
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    q = cam.subscribe()
    cam._broadcast(b"1")
    cam._broadcast(b"2")
    cam._broadcast(b"3")  # queue was full at 2 — oldest ("1") must be dropped
    assert q.qsize() == 2
    assert q.get_nowait() == b"2"
    assert q.get_nowait() == b"3"


def test_mic_broadcast_with_no_subscribers_stores_nothing():
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    for i in range(50):
        cam._broadcast(str(i).encode())  # must not raise, must not accumulate anywhere
    assert cam._subs == []


def test_mic_reconnecting_subscriber_never_receives_backlog():
    # The failure this guards: "hearing a question from a minute ago is
    # worse than hearing nothing" — a client that (re)connects must only
    # ever receive audio broadcast AFTER it subscribed.
    cam = MicCapture(process_factory=lambda: FakeProcess([]))
    cam._broadcast(b"stale-before-anyone-was-listening")
    q = cam.subscribe()
    assert q.empty()
    cam._broadcast(b"fresh")
    assert q.get_nowait() == b"fresh"


# --- supervisor: start()/stop(), restart-with-backoff, no orphaned process ---

def test_mic_supervisor_restarts_with_backoff_and_logs_each_attempt(caplog):
    attempts = []

    def factory():
        attempts.append(1)
        return FakeProcess([b""], stderr=b"no such device", poll_results=(1,))

    cam = MicCapture(process_factory=factory, initial_backoff=0.01,
                     max_backoff=0.02)
    with caplog.at_level("WARNING", logger="demo.camera_service"):
        cam.start()
        assert _wait_until(lambda: len(attempts) >= 3)
        cam.stop()

    assert len(attempts) >= 3
    assert cam.status()["restarts"] >= 2
    assert any("restarting in" in r.message for r in caplog.records)
    assert not cam._thread.is_alive()


def test_mic_stop_terminates_a_running_process_and_joins_the_thread():
    proc = FakeProcess([])  # alive, no chunks yet — read() blocks
    cam = MicCapture(process_factory=lambda: proc)
    cam.start()
    assert _wait_until(lambda: cam._proc is not None)
    cam.stop()
    assert proc.terminate_called
    assert not cam._thread.is_alive()


def test_mic_stop_before_spawn_finishes_does_not_orphan_the_process():
    # Same race as CameraCapture: stop() runs, sees no process yet (still
    # spawning), and _run_once's spawn only lands afterward — that arecord
    # must still get terminated, not left holding the mic's file descriptor
    # forever.
    proc = FakeProcess([])
    spawned = threading.Event()

    def slow_factory():
        time.sleep(0.05)
        spawned.set()
        return proc

    cam = MicCapture(process_factory=slow_factory)
    cam.start()
    cam.stop()  # races ahead of slow_factory() returning
    assert spawned.is_set()
    assert proc.terminate_called
    assert not cam._thread.is_alive()


def test_stop_terminates_both_camera_and_mic_processes():
    # main()'s shutdown path: a dead camera or mic must not stop the other
    # from being torn down, and neither process may be left orphaned.
    cam_proc = FakeProcess([])
    mic_proc = FakeProcess([])
    camera = CameraCapture(process_factory=lambda: cam_proc)
    mic = MicCapture(process_factory=lambda: mic_proc)
    camera.start()
    mic.start()
    assert _wait_until(lambda: camera._proc is not None and mic._proc is not None)
    camera.stop()
    mic.stop()
    assert cam_proc.terminate_called
    assert mic_proc.terminate_called
    assert not camera._thread.is_alive()
    assert not mic._thread.is_alive()


# --- HTTP layer: /audio and /mic/health, fake camera, real sockets ---

class FakeMic:
    def __init__(self, rate=16000, channels=2, status=None):
        self.rate = rate
        self.channels = channels
        self._status = status if status is not None else {
            "healthy": True, "chunk_age_s": 0.1, "restarts": 0,
            "last_error": None,
        }
        self._subs: list[queue.Queue] = []

    def status(self):
        return self._status

    def subscribe(self):
        q: queue.Queue = queue.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subs:
            self._subs.remove(q)


def test_mic_health_returns_404_when_mic_not_configured(running_server):
    srv = running_server(FakeCamera())  # mic=None (the default)
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", MIC_HEALTH_PATH)
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_audio_returns_404_when_mic_not_configured(running_server):
    # The service must still work with the camera alone if the mic couldn't
    # be constructed — a specific 404, not a hang or a crash.
    srv = running_server(FakeCamera())  # mic=None
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", AUDIO_PATH)
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_mic_health_returns_200_and_json_when_healthy(running_server):
    srv = running_server(FakeCamera(), FakeMic(status={
        "healthy": True, "chunk_age_s": 0.2, "restarts": 0, "last_error": None}))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", MIC_HEALTH_PATH)
    resp = conn.getresponse()
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["healthy"] is True
    conn.close()


def test_mic_health_returns_503_when_unhealthy(running_server):
    srv = running_server(FakeCamera(), FakeMic(status={
        "healthy": False, "chunk_age_s": 30.0, "restarts": 5,
        "last_error": "arecord exited (code 1)"}))
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", MIC_HEALTH_PATH)
    resp = conn.getresponse()
    assert resp.status == 503
    body = json.loads(resp.read())
    assert body["healthy"] is False
    assert body["restarts"] == 5
    conn.close()


def test_audio_declares_format_headers_and_streams_broadcast_chunks(running_server):
    mic = MicCapture(rate=16000, channels=2, chunk_ms=100,
                     process_factory=lambda: FakeProcess([]))
    srv = running_server(FakeCamera(), mic)
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", AUDIO_PATH)
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "application/octet-stream"
    assert resp.getheader("X-Sample-Rate") == "16000"
    assert resp.getheader("X-Channels") == "2"
    assert resp.getheader("X-Sample-Format") == "S16_LE"

    # subscribe() runs inside the handler's thread — wait for it to land
    # before broadcasting, rather than racing it.
    assert _wait_until(lambda: len(mic._subs) == 1)
    mic._broadcast(b"chunk-one")
    mic._broadcast(b"chunk-two")
    assert resp.read(len(b"chunk-one")) == b"chunk-one"
    assert resp.read(len(b"chunk-two")) == b"chunk-two"
    conn.close()


def test_audio_unsubscribes_the_queue_once_the_client_disconnects(running_server):
    mic = MicCapture(rate=16000, channels=2, chunk_ms=100,
                     process_factory=lambda: FakeProcess([]))
    srv = running_server(FakeCamera(), mic)
    host, port = srv.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", AUDIO_PATH)
    conn.getresponse()
    assert _wait_until(lambda: len(mic._subs) == 1)
    conn.close()
    assert _wait_until(lambda: len(mic._subs) == 0)


def test_frame_endpoint_still_responds_while_audio_stream_is_open(running_server):
    # Regression for a single-threaded HTTPServer: /audio holds its
    # connection open for as long as the Mac is listening, so /frame must
    # still be servable concurrently — a dead/slow mic listener must not
    # starve the video half of the demo.
    jpeg = SOI + b"still-picture" + EOI
    mic = MicCapture(rate=16000, channels=2, chunk_ms=100,
                     process_factory=lambda: FakeProcess([]))
    srv = running_server(FakeCamera(frame=jpeg), mic)
    host, port = srv.address

    audio_conn = http.client.HTTPConnection(host, port, timeout=5)
    audio_conn.request("GET", AUDIO_PATH)
    audio_resp = audio_conn.getresponse()
    assert audio_resp.status == 200
    try:
        frame_conn = http.client.HTTPConnection(host, port, timeout=5)
        frame_conn.request("GET", FRAME_PATH)
        frame_resp = frame_conn.getresponse()
        assert frame_resp.status == 200
        assert frame_resp.read() == jpeg
        frame_conn.close()
    finally:
        audio_conn.close()


def test_a_dead_capture_process_says_why():
    # Popen leaves .stderr None for a file it did not open; the one message
    # that explains a dead camera was never read before _spawn attached it.
    from demo.camera_service import _describe_exit, _spawn

    proc = _spawn(["sh", "-c", "echo 'Pipeline handler in use' >&2; exit 3"])
    proc.wait()
    assert _describe_exit(proc, "rpicam-vid") == (
        "rpicam-vid exited (code 3): Pipeline handler in use")


def test_the_standalone_reason_phrase_drops_control_characters():
    import demo.camera_service as service

    assert "\r" not in service.ascii_reason("bad\r\nX-Injected: 1")
