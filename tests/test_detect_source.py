import logging

import numpy as np

from demo.detect_source import RemoteDetectSource


class FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


FRAME = np.zeros((8, 8, 3), np.uint8)


def test_latest_returns_frame_and_detections_snapshot():
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    dets = [{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]
    source._post_detect = lambda frame: dets
    source._cycle()
    frame, out = source.latest()
    assert np.array_equal(frame, FRAME)
    assert out == dets


def test_latest_is_a_snapshot_copy_not_shared_state():
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: []
    source._cycle()
    frame, _ = source.latest()
    frame[:] = 255
    frame2, _ = source.latest()
    assert not np.array_equal(frame2, frame), "latest() returns a copy, not a shared buffer"


def test_no_frame_yet_returns_none_and_empty_list():
    source = RemoteDetectSource(FakeCamera(None), "http://x/detect")
    frame, dets = source.latest()
    assert frame is None
    assert dets == []


def test_listener_fires_each_cycle_with_fresh_detections():
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    calls = []
    source.add_listener(calls.append)
    responses = iter([[{"label": "person"}], [{"label": "cup"}]])
    source._post_detect = lambda frame: next(responses)
    source._cycle()
    source._cycle()
    assert calls == [[{"label": "person"}], [{"label": "cup"}]]


def test_cycle_skips_when_camera_has_no_frame_yet():
    source = RemoteDetectSource(FakeCamera(None), "http://x/detect")
    calls = []
    source.add_listener(calls.append)
    source._cycle()
    assert calls == []


def test_post_detect_failure_is_logged(monkeypatch, caplog):
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")

    def boom(*a, **k):
        raise OSError("service down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with caplog.at_level(logging.WARNING):
        result = source._post_detect(FRAME)
    assert result is None
    assert any("OSError" in r.message and "service down" in r.message
                for r in caplog.records)


def test_none_vs_empty_list_distinction_for_healthy():
    """None (failure) bumps the failure counter; [] (empty scene) resets it."""
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    source._cycle()
    source._cycle()
    assert source._consecutive_failures == 2
    source._post_detect = lambda frame: []
    source._cycle()
    assert source._consecutive_failures == 0
    # latest() always hands the consumer a list, never None, either way
    _, dets = source.latest()
    assert dets == []


def test_healthy_flips_false_after_n_consecutive_failures():
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    for _ in range(4):
        source._cycle()
        assert source.healthy() is True
    source._cycle()   # 5th consecutive failure
    assert source.healthy() is False


def test_healthy_recovers_after_a_success():
    source = RemoteDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    for _ in range(5):
        source._cycle()
    assert source.healthy() is False
    source._post_detect = lambda frame: [{"label": "cup"}]
    source._cycle()
    assert source.healthy() is True


def test_faces_ride_back_with_the_boxes(monkeypatch):
    # The frame is already on its way to the Mac for object detection; face
    # boxes come back with it, and the robot's head follows them between
    # turns (demo/run_demo.py's FaceTracker).
    import json

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    body = json.dumps({"detections": [{"label": "person"}],
                       "faces": [{"box": [0.3, 0.2, 0.6, 0.7], "score": 0.9}],
                       "ms": 12}).encode()
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Response(body))
    source = RemoteDetectSource(FakeCamera(FRAME), "http://mac:9600/detect")
    assert source.faces() == []
    assert source._post_detect(FRAME) == [{"label": "person"}]
    assert source.faces() == [{"box": [0.3, 0.2, 0.6, 0.7], "score": 0.9}]
    assert source.faces() is not source.faces(), "a snapshot, not shared state"


def test_a_service_without_face_detection_just_has_no_faces(monkeypatch):
    import json

    class _Response:
        def read(self):
            return json.dumps({"detections": [], "ms": 3}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Response())
    source = RemoteDetectSource(FakeCamera(FRAME), "http://mac:9600/detect")
    source._post_detect(FRAME)
    assert source.faces() == []


# --- the same loop with the models in this process
# ---

class FakeDetector:
    """Stands in for emulator/detector.py's Detector: frames in, Detections out."""

    def __init__(self, detections):
        self._detections = detections
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self._detections


def test_local_source_produces_the_same_shape_of_boxes_as_the_remote_one():
    from demo.detect_source import LocalDetectSource
    from emulator.detector import Detection

    # 41 is COCO's "cup" in the label table demo/detections.py names.
    detector = FakeDetector([Detection(41, 0.8, (0.4, 0.4, 0.6, 0.6))])
    source = LocalDetectSource(FakeCamera(FRAME), detector)
    source._cycle()
    _, out = source.latest()
    assert out == [{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]
    assert detector.calls == 1


def test_local_source_finds_faces_on_every_cycle_like_the_remote_one():
    # The head tracker is fed from faces() four times a second; a robot that
    # looked for a face once a turn would stop following the person.
    from demo.detect_source import LocalDetectSource

    seen = []

    def faces(frame):
        seen.append(frame)
        return [{"box": [0.1, 0.1, 0.2, 0.2], "score": 0.9}]

    source = LocalDetectSource(FakeCamera(FRAME), FakeDetector([]), faces=faces)
    source._cycle()
    source._cycle()
    assert len(seen) == 2
    assert source.faces() == [{"box": [0.1, 0.1, 0.2, 0.2], "score": 0.9}]


def test_faces_that_fail_leave_the_objects_found_with_them():
    # The robot's detector asks the laptop for face boxes; a laptop without
    # them must not blind the robot to everything else in the frame.
    from demo.detect_source import LocalDetectSource
    from emulator.detector import Detection

    def faces(frame):
        raise OSError("HTTP Error 503: faces off")

    heard = []
    source = LocalDetectSource(FakeCamera(FRAME),
                               FakeDetector([Detection(41, 0.8, (0.4, 0.4, 0.6, 0.6))]),
                               faces=faces)
    source.add_listener(heard.append)
    source._cycle()
    assert heard == [[{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]]
    assert source.faces() == []


def test_local_source_notifies_the_same_listeners():
    from demo.detect_source import LocalDetectSource

    heard = []
    source = LocalDetectSource(FakeCamera(FRAME), FakeDetector([]))
    source.add_listener(heard.append)
    source._cycle()
    assert heard == [[]], "the dashboard hangs off these listeners"


def test_a_failing_local_detector_does_not_kill_the_loop():
    from demo.detect_source import LocalDetectSource

    class Broken:
        def detect(self, frame):
            raise RuntimeError("model went away")

    source = LocalDetectSource(FakeCamera(FRAME), Broken())
    source._cycle()
    _, out = source.latest()
    assert out == [], "a bad frame is an empty scene, not a crash"
    assert source.healthy() is True, "one failure is not unhealthy yet"


def test_a_camera_with_nothing_current_empties_the_snapshot():
    # Otherwise the last frame before the camera died is what the model gets
    # as the picture of "now".
    camera = FakeCamera(np.zeros((4, 4, 3), np.uint8))
    source = RemoteDetectSource(camera, "http://x/detect")
    source._post_detect = lambda frame: [{"label": "cup", "score": 0.9,
                                          "box": [0, 0, 1, 1]}]
    source._cycle()
    assert source.latest()[0] is not None
    camera._frame = None
    source._cycle()
    frame, detections = source.latest()
    assert frame is None and detections == []
