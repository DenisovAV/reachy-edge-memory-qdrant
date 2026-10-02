"""Detection service on the laptop: a JPEG frame in, object and face boxes out.

The robot's detect loop (demo/detect_source.py) posts a frame here about four
times a second. The detector runs on the laptop's CPU cores, about 20 ms a
frame: YOLO26n's end-to-end head does not compile for the GPU
(emulator/detector.py). Faces ride along: YuNet boxes from the same frame keep
the robot's head on the person.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

from demo.detections import detections_to_dicts
from demo.http_util import ascii_reason
from emulator import models
from emulator.detector import Detector

# Decompression-bomb guard: demo frames are under 1 Mpx.
Image.MAX_IMAGE_PIXELS = 10_000_000

LOG = logging.getLogger(__name__)

MAX_BODY = 4 * 1024 * 1024  # overflow guard: /detect request body size limit


class Objects:
    """The detector on the frames the robot posts."""

    def __init__(self, model_path: str, score_threshold: float = 0.3) -> None:
        self._detector = Detector(model_path, threads=4,
                                  score_threshold=score_threshold)

    def detect(self, jpeg: bytes) -> list[dict]:
        frame = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return detections_to_dicts(self._detector.detect(frame))


class Faces:
    """YuNet on the frames the detector already receives.

    Boxes only, no identity: this runs four times a second to keep the robot's
    head on the person's face (demo/run_demo.py's FaceTracker), while WHO that
    person is stays a once-per-turn question answered elsewhere
    (demo/people.py). Optional — a missing model disables it and says so once,
    rather than failing every frame."""

    def __init__(self) -> None:
        self._reader = None
        try:
            from emulator.face import FaceReader

            self._reader = FaceReader(identities=False)
        except Exception as exc:  # noqa: BLE001 — faces are an extra here
            LOG.warning("detect_service: no face detection (%s: %s)",
                        type(exc).__name__, exc)

    def detect(self, jpeg: bytes) -> list[dict]:
        if self._reader is None:
            return []
        frame = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"),
                           dtype=np.uint8)
        return [{"box": face.box, "score": face.score}
                for face in self._reader.read(frame, embed=False)]


def make_handler(detector: Objects, faces=None):
    class Handler(BaseHTTPRequestHandler):
        # Single-threaded HTTPServer (one worker) — the timeout stops a
        # hung/slow client from permanently hogging the only socket.
        timeout = 30

        def do_POST(self):
            if self.path != "/detect":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError as exc:
                # A present-but-non-numeric header (Content-Length: abc) would
                # otherwise raise out of do_POST — an unhandled traceback to
                # stderr and a dropped connection instead of a clean 400.
                LOG.warning("bad Content-Length: %s: %s",
                            type(exc).__name__, exc)
                self.send_error(400, "bad Content-Length")
                return
            if length <= 0 or length > MAX_BODY:
                # Overflow guard: body out of bounds (non-ASCII in the HTTP
                # status line's reason phrase breaks http.server's latin-1
                # encoder — the message itself must be ASCII, comments needn't be)
                self.send_error(413, "request body out of bounds")
                return
            jpeg = self.rfile.read(length)
            try:
                t0 = time.perf_counter()
                dets = detector.detect(jpeg)
                found_faces = faces.detect(jpeg) if faces is not None else []
                ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # noqa: BLE001 — the service must stay up
                # Broad boundary → log WITH the stack (exc_info) so an
                # unexpected bug is localizable, not just a one-line type+message.
                LOG.exception("detect failed: %s", type(exc).__name__)
                # Raw exception text in the 500 body is handy for debugging the
                # demo, not for a prod API (could leak an internal path/trace).
                # ascii_reason: the reason phrase is latin-1-encoded by
                # http.server, so a non-ASCII char in the message must not crash send_error.
                self.send_error(500, ascii_reason(str(exc)))
                return
            body = json.dumps({"detections": dets, "faces": found_faces,
                               "ms": ms}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_request(self, code="-", size="-"):
            # Stay quiet on 2xx (normal demo traffic); 4xx/5xx go to the
            # standard log (log_error -> log_message), a backstop on top of LOG.error above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Object and face detection service")
    # Bound wide by default: the robot reaches it across the network. See the
    # README's security section — this service has no authentication.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9600)
    p.add_argument("--model", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    from demo.logs import setup_logging

    setup_logging()
    args = parse_args(argv)
    detector = Objects(args.model or str(models.fetch(models.DETECTOR)))
    server = HTTPServer((args.host, args.port), make_handler(detector, Faces()))
    print(f"detect_service on {args.host}:{args.port}: {models.DETECTOR} "
          "on the CPU", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
