"""demo/detect_service.py: the HTTP handler, with a stand-in detector, and the
detector behind it — a JPEG in, JSON-ready dicts out."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import HTTPServer

import pytest

from demo.detect_service import MAX_BODY, make_handler, parse_args


def test_objects_decodes_the_jpeg_as_rgb_and_answers_in_json_dicts(monkeypatch):
    # The robot reads these as JSON: a Detection object here would drop every
    # answer, and the robot would see nothing at all.
    import io

    from PIL import Image

    import demo.detect_service as detect_service
    from emulator.detector import Detection

    seen = {}

    class FakeDetector:
        def __init__(self, path, threads=4, score_threshold=0.3):
            seen.update(threads=threads, threshold=score_threshold)

        def detect(self, frame):
            seen["frame"] = frame
            return [Detection(41, 0.8, (0.1, 0.2, 0.3, 0.4))]

    monkeypatch.setattr(detect_service, "Detector", FakeDetector)
    jpeg = io.BytesIO()
    Image.new("RGB", (64, 48), (255, 0, 0)).save(jpeg, "JPEG")
    out = detect_service.Objects("model.tflite",
                                 score_threshold=0.4).detect(jpeg.getvalue())
    assert json.loads(json.dumps(out)) == [
        {"label": "cup", "score": 0.8, "box": [0.1, 0.2, 0.3, 0.4]}]
    frame = seen["frame"]
    assert frame.shape == (48, 64, 3)
    assert frame[..., 0].mean() > 200 and frame[..., 2].mean() < 50, "red stays first"
    assert seen["threads"] > 1, "CompiledModel would otherwise run on one core"
    assert seen["threshold"] == 0.4


def test_parse_args_host_defaults_to_all_interfaces():
    # The robot reaches this across the network (README.md, Security).
    assert parse_args([]).host == "0.0.0.0"


# --- HTTP handler: a fake detector standing in for a real Objects ---

class FakeDetector:
    """Stand-in for Objects — make_handler doesn't check the type, only calls .detect()."""

    def __init__(self, dets=None):
        self._dets = dets if dets is not None else [
            {"label": "cup", "score": 0.8, "box": [0.1, 0.1, 0.3, 0.3]}]

    def detect(self, jpeg: bytes) -> list[dict]:
        return self._dets


class BrokenDetector:
    """Simulates an internal inference failure — checks the mapping to 500."""

    def detect(self, jpeg: bytes) -> list[dict]:
        raise RuntimeError("model exploded")


class _RunningServer:
    """Runs a single-threaded HTTPServer (like detect_service.main()) on a
    background thread — tested via real HTTP requests rather than a
    hand-assembled Handler."""

    def __init__(self, detector):
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(detector))
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
    srv = _RunningServer(FakeDetector())
    try:
        yield srv
    finally:
        srv.close()


def test_post_to_unknown_path_returns_404(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/wrong", body=b"junk")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_zero_length_body_returns_413(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/detect", body=b"")
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_oversized_body_returns_413(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=10)
    # Content-Length lies about the body size — the server rejects based on
    # the header BEFORE reading the body, so we don't actually need to send MAX_BODY+1 bytes.
    conn.request("POST", "/detect", body=b"",
                headers={"Content-Length": str(MAX_BODY + 1)})
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_valid_jpeg_body_returns_detections(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    conn.request("POST", "/detect", body=jpeg,
                 headers={"Content-Type": "image/jpeg"})
    resp = conn.getresponse()
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["detections"] == [
        {"label": "cup", "score": 0.8, "box": [0.1, 0.1, 0.3, 0.3]}]
    assert "ms" in body
    conn.close()


def test_detector_exception_returns_500_and_is_logged(caplog):
    srv = _RunningServer(BrokenDetector())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        with caplog.at_level("ERROR", logger="demo.detect_service"):
            conn.request("POST", "/detect", body=b"junk-jpeg-bytes")
            resp = conn.getresponse()
            resp.read()
        conn.close()
        assert resp.status == 500
        assert any("RuntimeError" in record.message
                   for record in caplog.records)
    finally:
        srv.close()
