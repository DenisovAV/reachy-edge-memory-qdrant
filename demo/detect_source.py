"""Detection source: camera frame -> the laptop's detect service -> boxes.

The screenless core: sends camera frames to the laptop's detect service
(~4 fps, only boxes come back) and holds a snapshot (latest frame +
detections) behind a lock. It knows nothing about the browser, SSE, or HTTP
serving — those responsibilities moved to demo/display/. The voice loop
always reads detections from here, regardless of whether a display is
attached: `WebDashboard.on_detections` is just one of the listeners
(`add_listener`), and `NullSink.on_detections` is a silent one.
"""
from __future__ import annotations

import io
import json
import logging
import threading
import urllib.request
from typing import Callable

import numpy as np
from PIL import Image

from demo.detections import detections_to_dicts

LOG = logging.getLogger(__name__)

# After how many consecutive failed POSTs to detect_service the source is
# considered unhealthy (healthy() == False). A single glitch (a brief network
# blip) shouldn't trip the signal — what matters is a sustained run
# of failures.
MAX_CONSECUTIVE_FAILURES = 5


def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class RemoteDetectSource:
    """Sends camera frames to the laptop's detect service, holds the latest snapshot.

    `latest()` always returns a list (even after an error — an empty one), so
    consumers (the voice loop, the display) never have to handle None. The
    distinction between "error" (None) and "empty scene" ([]) is tracked only
    internally, for `healthy()`.
    """

    def __init__(self, camera, detect_url: str, fps: int = 4) -> None:
        self.camera = camera
        self.detect_url = detect_url
        self._interval = 1.0 / fps
        self._latest_frame: np.ndarray | None = None
        self._latest_dets: list[dict] = []
        self._latest_faces: list[dict] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[list[dict]], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0

    # — detection —
    def start(self) -> None:
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Longer than _post_detect's urlopen(timeout=5) so an in-flight
            # POST finishes and the thread is actually reaped, rather than
            # stop() returning while a straggling cycle is still running.
            self._thread.join(timeout=6)

    def latest(self) -> tuple[np.ndarray | None, list[dict]]:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return frame, list(self._latest_dets)

    def faces(self) -> list[dict]:
        """Faces from the last detect cycle, largest first."""
        with self._lock:
            return list(self._latest_faces)

    def add_listener(self, fn: Callable[[list[dict]], None]) -> None:
        """fn(detections) is called on every detect cycle with fresh boxes."""
        self._listeners.append(fn)

    def healthy(self) -> bool:
        return self._consecutive_failures < MAX_CONSECUTIVE_FAILURES

    def _post_detect(self, frame: np.ndarray) -> list[dict] | None:
        """POST a frame to detect_service. None means a failure (the service
        unreachable or failing), [] means the service responded but found
        nothing (empty scene)."""
        try:
            req = urllib.request.Request(
                self.detect_url, data=frame_to_jpeg(frame),
                headers={"Content-Type": "image/jpeg"})
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
            with self._lock:
                # Faces ride along with the boxes: the frame is already there,
                # and the head tracks them between turns (demo/run_demo.py's
                # FaceTracker).
                self._latest_faces = list(resp.get("faces", []))
            return resp.get("detections", [])
        except Exception as exc:  # noqa: BLE001 — the service may drop out, don't kill the loop
            LOG.warning("detect_service POST failed: %s: %s",
                        type(exc).__name__, exc)
            return None

    def _cycle(self) -> None:
        """One loop step: frame -> POST -> snapshot -> listeners.

        Pulled out of `_detect_loop` so tests can drive the steps
        deterministically, without threads or sleeps.
        """
        frame = self.camera.latest()
        if frame is None:
            # The camera has nothing current: nor does this. A snapshot kept
            # from before would be handed to the model as what is in front of
            # the robot now.
            with self._lock:
                self._latest_frame = None
                self._latest_dets = []
                self._latest_faces = []
            return
        dets = self._post_detect(frame)
        if dets is None:
            self._consecutive_failures += 1
            dets_out: list[dict] = []
        else:
            self._consecutive_failures = 0
            dets_out = dets
        with self._lock:
            self._latest_frame = frame
            self._latest_dets = dets_out
        for fn in self._listeners:
            try:
                fn(dets_out)
            except Exception as exc:  # noqa: BLE001 — a bad listener must not kill the loop
                LOG.warning("detect listener failed: %s: %s",
                            type(exc).__name__, exc)

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._cycle()
            except Exception as exc:  # noqa: BLE001 — one bad cycle must not kill the loop
                # Log WITH the stack: unlike a POST failure (an expected outage
                # that counts toward healthy()), an exception here is an
                # unexpected bug and needs a traceback to localize.
                LOG.exception("detect cycle failed: %s", type(exc).__name__)
            # wait() (not sleep) so stop() wakes the loop immediately between
            # cycles instead of after a full interval.
            self._stop.wait(self._interval)


class LocalDetectSource(RemoteDetectSource):
    """The same loop, with the models in this process instead of over HTTP.

    Everything that makes the remote source what it is — the cadence, the
    listeners the dashboard hangs off, `latest()`, `faces()`, `healthy()` —
    is inherited untouched; only the step that turns one frame into boxes
    changes. That is the spec's requirement for a local detector: the same
    boxes at the same cadence — and they come from the same code: the Mac's
    service (demo/detect_service.py) runs this same emulator/detector.Detector,
    only behind HTTP.

    Faces ride along on the same cycle for the same reason they do remotely:
    the head tracker (demo/run_demo.py's FaceTracker) is fed from them four
    times a second, and a robot that only looked for a face once a turn would
    stop following the person it is talking to.
    """

    def __init__(self, camera, detector, faces=None, fps: int = 4) -> None:
        """`faces` is any callable frame -> [{"box", "score"}, …] (demo/
        run_demo.py passes one over emulator/face.py's FaceReader), or None
        when this run is not looking for faces."""
        super().__init__(camera, detect_url="(in this process)", fps=fps)
        self._detector = detector
        self._faces = faces

    def _post_detect(self, frame: np.ndarray) -> list[dict] | None:
        """The local stand-in for the POST: no network, same contract —
        None on a failure, [] for an empty scene."""
        try:
            dets = detections_to_dicts(self._detector.detect(frame))
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the loop
            LOG.warning("local detect failed: %s: %s", type(exc).__name__, exc)
            return None
        try:
            found_faces = self._faces(frame) if self._faces else []
        except Exception as exc:  # noqa: BLE001 — the objects stand without them
            LOG.warning("local face boxes failed: %s: %s", type(exc).__name__, exc)
            found_faces = []
        with self._lock:
            self._latest_faces = list(found_faces)
        return dets
