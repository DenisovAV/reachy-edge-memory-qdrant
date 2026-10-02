"""RobotCameraSource: a VideoSource pulling frames from the robot's OWN
camera (demo/camera_service.py) over HTTP. What gets remembered has to be
what the ROBOT saw, not the laptop's webcam.

Follows the exact client shape already established by
demo/detect_source.py::RemoteDetectSource: a background thread polls the
service on its own schedule and keeps the newest decoded frame behind a
lock, so `latest()` never blocks the voice loop on the network, and a
`_cycle()` method is split out of the poll loop so tests can drive it
directly instead of racing real threads and sleeps.

`alive` mirrors CameraStream (demo/platform/mac.py): the SAME health signal
`run_voice` already reads via `getattr(camera, "alive", True)` — a robot
camera outage is reported through that one existing print, not a second
mechanism. Only `frame_url()` assembles the service's host, port and path.
"""
from __future__ import annotations

import io
import http.client
import logging
import threading
import time
import urllib.request

import numpy as np
from PIL import Image

from demo.platform.pcm import frame_to_png

LOG = logging.getLogger(__name__)

# Overridable via --robot-camera-port. Imported from the service rather than
# restated here: this client and demo/camera_service.py once kept separate
# copies of these constants and silently disagreed (8100/frame.jpg vs
# 9700/frame), each half's tests asserting against its own copy — a green
# suite over a pipeline that 404'd on every request. One definition, no
# drift. camera_service is stdlib-only, so importing it costs nothing.
from demo.camera_service import DEFAULT_PORT, FRAME_PATH

# Consecutive failed GETs before `alive` flips False — same threshold and
# same reasoning as RemoteDetectSource.MAX_CONSECUTIVE_FAILURES: a single
# dropped packet on a conference network shouldn't flip the health signal,
# only a sustained outage should.
MAX_CONSECUTIVE_FAILURES = 5

# Bounded wait for a single GET so a hung/dead service can't stall the
# poll loop forever — mirrors RemoteDetectSource's urlopen(timeout=5), shorter
# here since a frame fetch has much less work to do than a detect pass.
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_POLL_INTERVAL_S = 0.1

# How old the last frame may be before it no longer counts as "now". Past
# this, latest() answers None: a camera that stopped must not keep handing
# the model its last picture as "your camera, right now".
STALE_AFTER_S = 3.0


def frame_url(host: str, port: int = DEFAULT_PORT) -> str:
    """The one place that assembles the camera service's frame URL."""
    return f"http://{host}:{port}{FRAME_PATH}"


class RobotCameraSource:
    """VideoSource pulling the newest JPEG frame from the robot's camera
    service — a drop-in for CameraStream (Mac webcam): the voice loop,
    RemoteDetectSource, and the dashboard all consume `latest()`/`latest_png()`/
    `close()` and must not know which one they were handed.
    """

    def __init__(self, url: str, poll_interval: float = DEFAULT_POLL_INTERVAL_S,
                 timeout: float = DEFAULT_TIMEOUT_S, *,
                 stale_after: float = STALE_AFTER_S, clock=time.monotonic,
                 _autostart: bool = True) -> None:
        self._url = url
        self._interval = poll_interval
        self._timeout = timeout
        self._stale_after = stale_after
        self._clock = clock
        self._latest: np.ndarray | None = None
        self._latest_at = float("-inf")
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        # _autostart=False is a test-only seam: it lets a test drive _cycle()
        # by hand for an exact, deterministic failure/success count, without
        # racing a real thread that would otherwise start polling the
        # instant __init__ returns. Every real caller wants the thread
        # running immediately, same as CameraStream (Mac).
        if _autostart:
            self._thread.start()

    @property
    def alive(self) -> bool:
        # Parity with CameraStream.alive: True
        # until a sustained run of failures, never a separate "healthy()"
        # call — run_voice's existing camera-health print already reads
        # this exact attribute via getattr(camera, "alive", True).
        return self._consecutive_failures < MAX_CONSECUTIVE_FAILURES

    def _fetch_jpeg(self) -> bytes | None:
        """GET the newest frame. None means a failure (service down/slow/
        unreachable) — logged here since the poll loop just counts it."""
        try:
            return urllib.request.urlopen(self._url, timeout=self._timeout).read()
        # HTTPException (e.g. IncompleteRead, when the body is shorter than
        # Content-Length) does NOT inherit OSError and urllib does not wrap
        # it — uncaught here it kills the polling thread, and the demo runs
        # blind for the rest of the talk.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            # Log the transition, not every failure: the poll loop runs at
            # ~10 Hz and a refused connection fails in milliseconds, so an
            # unguarded warning here buries the real output under ten lines a
            # second for as long as the service is down. Same shape as
            # _FailureGuard in demo/run_demo.py.
            if self._consecutive_failures == 0:
                LOG.warning("robot camera GET failed: %s: %s (further "
                           "failures silent until it recovers)",
                           type(exc).__name__, exc)
            return None

    def _decode(self, jpeg: bytes) -> np.ndarray | None:
        """JPEG bytes -> RGB frame. None means a corrupt/partial response —
        as real a failure as the GET itself not answering."""
        try:
            return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        except (OSError, ValueError) as exc:
            if self._consecutive_failures == 0:
                LOG.warning("robot camera: bad JPEG: %s: %s (further failures "
                           "silent until it recovers)",
                           type(exc).__name__, exc)
            return None

    def _cycle(self) -> None:
        """One poll step: GET -> decode -> snapshot. Split out of
        `_poll_loop` so tests can drive it deterministically, without
        threads or sleeps (mirrors RemoteDetectSource._cycle)."""
        jpeg = self._fetch_jpeg()
        frame = self._decode(jpeg) if jpeg is not None else None
        if frame is None:
            self._consecutive_failures += 1
            return
        self._consecutive_failures = 0
        with self._lock:
            self._latest = frame
            self._latest_at = self._clock()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._cycle()
            # wait() (not sleep) so close() wakes the loop immediately
            # between cycles instead of after a full interval.
            self._stop.wait(self._interval)

    def latest(self) -> np.ndarray | None:
        # None before the first successful fetch, and again once the last
        # frame is older than stale_after (see STALE_AFTER_S).
        with self._lock:
            if (self._latest is None
                    or self._clock() - self._latest_at > self._stale_after):
                return None
            return self._latest.copy()

    def latest_png(self) -> bytes | None:
        frame = self.latest()
        return frame_to_png(frame) if frame is not None else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
