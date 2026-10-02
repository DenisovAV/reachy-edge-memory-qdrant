import json

import pytest

from demo.robot_reachy import (GESTURE_POSES, LOOK_POSES, ConsoleRobot,
                               gaze_to_head_pose)


def test_centre_gaze_is_zero_angles():
    pose = gaze_to_head_pose(0.0, 0.0)
    assert abs(pose["yaw"]) < 1e-6
    assert abs(pose["pitch"]) < 1e-6


def test_a_point_on_the_right_of_the_image_turns_the_head_right():
    # Positive yaw is the robot's left (measured on the robot): something on
    # the right of the picture needs a negative yaw to face it.
    pose = gaze_to_head_pose(1.0, 0.0, amplitude_deg=20.0)
    assert pose["yaw"] == -20.0
    assert abs(pose["pitch"]) < 1e-6


def test_gaze_is_clamped():
    # Gaze outside [-1,1] must not produce an angle beyond the amplitude.
    pose = gaze_to_head_pose(5.0, -5.0, amplitude_deg=20.0)
    assert pose["yaw"] == -20.0
    assert pose["pitch"] == -20.0


def test_left_is_positive_yaw_for_gestures_and_looks():
    assert GESTURE_POSES["look_left"][0][0] > 0 > GESTURE_POSES["look_right"][0][0]
    assert LOOK_POSES["left"][0] > 0 > LOOK_POSES["right"][0]
    assert LOOK_POSES["ahead"] == (0, 0)


def test_console_robot_records_a_look():
    robot = ConsoleRobot()
    robot.look("left")
    assert robot.events[-1] == ("look", {"direction": "left"})


def test_console_robot_records_gesture():
    robot = ConsoleRobot()
    robot.gesture("shake")
    assert robot.events[-1] == ("gesture", {"name": "shake"})


def test_console_robot_ignores_unknown_gesture():
    # An unknown gesture must not crash the demo: record the event, no motion.
    robot = ConsoleRobot()
    robot.gesture("moonwalk")
    assert robot.events[-1] == ("gesture", {"name": "moonwalk"})


def test_gesture_poses_return_head_to_centre():
    # Every gesture ends at (0, 0), otherwise the head stays twisted.
    for name, poses in GESTURE_POSES.items():
        assert poses[-1] == (0, 0), f"{name} does not return the head to center"


# --- HttpReachyRobot: host, timeout, failure cooldown ---
# No real network in these tests: urllib.request.urlopen is monkeypatched.

class _FakeResponse:
    def __init__(self, body: bytes = b"{}"):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_reachy_robot_accepts_plain_ip_host():
    # Conference/guest networks routinely break mDNS (reachy-mini.local) —
    # a plain IP must work just as well.
    from demo.robot_reachy import HttpReachyRobot

    robot = HttpReachyRobot(host="192.168.1.42")
    assert robot._base == "http://192.168.1.42:8000/api"


def test_make_robot_passes_host_through_to_http_robot():
    from demo.robot_reachy import HttpReachyRobot, make_robot

    robot = make_robot(use_robot=True, host="10.0.0.9")
    assert isinstance(robot, HttpReachyRobot)
    assert robot._base == "http://10.0.0.9:8000/api"


def test_http_reachy_robot_default_timeout_is_short(monkeypatch):
    # These are local-network calls; the old 8s default turned a dropped
    # robot into ~8s of dead air PER CALL.
    from demo.robot_reachy import DEFAULT_TIMEOUT_S, HttpReachyRobot

    assert DEFAULT_TIMEOUT_S <= 2.0

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    HttpReachyRobot(host="10.0.0.9").look_at(0.0, 0.0)
    assert captured["timeout"] == DEFAULT_TIMEOUT_S


def test_http_reachy_robot_trips_cooldown_after_consecutive_failures(monkeypatch):
    # After N failures in a row, further calls must not touch the network at
    # all — they should raise immediately instead of paying another timeout.
    from demo.robot_reachy import (
        CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, HttpReachyRobot)

    calls = {"n": 0}

    def failing_urlopen(req, timeout=None):
        calls["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    robot = HttpReachyRobot(host="10.0.0.9")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        try:
            robot.look_at(0.0, 0.0)
        except OSError:
            pass
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    with pytest.raises(TimeoutError):
        robot.look_at(0.0, 0.0)
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, (
        "a call made during the cooldown window must not hit the network")


def test_http_reachy_robot_cooldown_recovers_automatically(monkeypatch):
    from demo.robot_reachy import (
        CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, FAILURE_COOLDOWN_S,
        HttpReachyRobot)

    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])

    def failing_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    robot = HttpReachyRobot(host="10.0.0.9")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        try:
            robot.look_at(0.0, 0.0)
        except OSError:
            pass

    with pytest.raises(TimeoutError):
        robot.look_at(0.0, 0.0)  # still inside the cooldown window

    # The robot comes back and the cooldown window elapses: the next call
    # must try the network again, not stay tripped for the rest of the demo.
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse())
    clock["t"] += FAILURE_COOLDOWN_S + 0.1
    robot.look_at(0.0, 0.0)  # must not raise
    assert robot._motion.failures == 0


# --- play_sound_file: upload + play on the robot's own speaker ---
# No real network: urllib.request.urlopen is monkeypatched, same as above.

def test_multipart_form_contains_field_filename_and_content_type():
    from demo.robot_reachy import _multipart_form

    body, content_type = _multipart_form("file", "demo_reply.wav", b"RIFF...",
                                         content_type="audio/wav")
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1]
    assert boundary.encode() in body
    assert b'name="file"' in body
    assert b'filename="demo_reply.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"RIFF..." in body


def test_upload_sound_posts_multipart_to_sounds_upload(monkeypatch):
    from demo.robot_reachy import HttpReachyRobot

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["content_type"] = req.get_header("Content-type")
        captured["body"] = req.data
        return _FakeResponse(b'{"status": "ok", "path": "/tmp/reachy_mini_sounds/demo_reply.wav"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    robot = HttpReachyRobot(host="10.0.0.9")
    path = robot._upload_sound(b"fake-wav-bytes")

    assert captured["url"] == "http://10.0.0.9:8000/api/media/sounds/upload"
    assert captured["method"] == "POST"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b"fake-wav-bytes" in captured["body"]
    assert path == "/tmp/reachy_mini_sounds/demo_reply.wav"


def test_play_sound_file_uploads_then_plays_then_blocks_for_duration(monkeypatch):
    from demo.robot_reachy import PLAYBACK_MARGIN_S, HttpReachyRobot

    requests = []

    def fake_urlopen(req, timeout=None):
        requests.append(req)
        if req.full_url.endswith("/media/sounds/upload"):
            return _FakeResponse(b'{"status": "ok", "path": "/tmp/reachy_mini_sounds/demo_reply.wav"}')
        return _FakeResponse(b'{"status": "ok"}')

    sleeps = []
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    robot = HttpReachyRobot(host="10.0.0.9")
    robot.play_sound_file(b"fake-wav-bytes", duration_s=2.0)

    assert len(requests) == 2
    assert requests[0].full_url == "http://10.0.0.9:8000/api/media/sounds/upload"
    assert requests[1].full_url == "http://10.0.0.9:8000/api/media/play_sound"
    play_body = json.loads(requests[1].data)
    assert play_body == {"file": "/tmp/reachy_mini_sounds/demo_reply.wav"}
    # CRITICAL TIMING: POST /media/play_sound returns the instant the daemon
    # ACCEPTS the request, not when the speaker goes quiet — play_sound_file
    # must itself block for the clip's own duration (plus the margin) so the
    # caller's close() keeps StreamPlayer's "blocks until actually done"
    # contract.
    assert sleeps == [2.0 + PLAYBACK_MARGIN_S]


def test_the_speaker_has_its_own_failure_cooldown(monkeypatch):
    # A run of failed uploads trips the sound's cooldown, and leaves the
    # head free to move — the two kinds of call do not share one counter.
    from demo.robot_reachy import (
        CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, HttpReachyRobot)

    def urlopen(req, timeout=None):
        if req.full_url.endswith("/media/sounds/upload"):
            raise OSError("connection refused")
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    robot = HttpReachyRobot(host="10.0.0.9")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            robot._upload_sound(b"fake-wav-bytes")
    with pytest.raises(TimeoutError):
        robot._upload_sound(b"fake-wav-bytes")
    robot.look_at(0.0, 0.0)  # must not raise

def test_playback_wait_survives_a_lost_response(monkeypatch):
    """The robot must not start listening while it is still talking.

    The daemon begins playing the moment it ACCEPTS the sound and answers
    afterwards, so a dropped or late response does not stop the audio. If the
    wait were skipped on that error, the voice loop's mic flush would land at
    t=0 of a multi-second reply, the robot would hear its own voice, transcribe
    it, and answer itself in front of the audience.
    """
    import demo.robot_reachy as rr

    slept = []
    monkeypatch.setattr(rr.time, "sleep", lambda s: slept.append(s))

    robot = rr.HttpReachyRobot(host="test-robot")
    monkeypatch.setattr(robot, "_upload_sound", lambda wav: "/sounds/reply.wav")

    def post_that_loses_the_response(path, body=None, breaker=None):
        raise OSError("response lost after the daemon already accepted it")

    monkeypatch.setattr(robot, "_post", post_that_loses_the_response)

    with pytest.raises(OSError):
        robot.play_sound_file(b"RIFFfake", duration_s=4.0)

    assert slept, "the playback wait was skipped — the robot will hear itself"
    assert slept[0] >= 4.0


def test_sound_upload_gets_a_longer_timeout_than_control_calls():
    """A reply's WAV is hundreds of kilobytes and a socket timeout bounds the
    whole transfer, so the upload cannot share the sub-second budget tuned for
    tiny JSON control POSTs — a timeout there costs the entire spoken turn."""
    import demo.robot_reachy as rr

    assert rr.HttpReachyRobot.UPLOAD_TIMEOUT_S > rr.DEFAULT_TIMEOUT_S * 5


def test_http_reachy_robot_looks_and_holds_with_one_goto(monkeypatch):
    import math

    from demo.robot_reachy import HttpReachyRobot

    posted = []

    def fake_urlopen(req, timeout=None):
        posted.append((req.full_url, json.loads(req.data)))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    HttpReachyRobot(host="10.0.0.9").look("left")
    assert len(posted) == 1
    url, body = posted[0]
    assert url.endswith("/api/move/goto")
    assert body["head_pose"]["yaw"] == pytest.approx(math.radians(LOOK_POSES["left"][0]))


def test_every_emotion_is_gentle_and_ends_at_rest():
    from demo.robot_reachy import EMOTION_MOVES, emotion_moves

    for name, steps in EMOTION_MOVES.items():
        assert steps[-1][:4] == (0, 0, 0, 0.0), f"{name} does not come back to rest"
        for yaw, pitch, roll, antennas, duration in steps:
            assert max(abs(yaw), abs(pitch), abs(roll)) <= 20, f"{name} swings too far"
            assert 0.2 <= duration <= 1.0 and abs(antennas) <= 1.0
    assert emotion_moves("love") == EMOTION_MOVES["happy"], "an alias still moves"
    assert emotion_moves("whatever-the-model-said") == EMOTION_MOVES["happy"]


def test_an_emotion_is_played_as_goto_poses_not_a_recorded_move(monkeypatch):
    from demo.robot_reachy import EMOTION_MOVES, HttpReachyRobot

    posted = []

    def fake_urlopen(req, timeout=None):
        posted.append(req.full_url)
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("demo.robot_reachy.time.sleep", lambda s: None)
    HttpReachyRobot(host="10.0.0.9").emotion("happy")
    assert len(posted) == len(EMOTION_MOVES["happy"])
    assert all(url.endswith("/api/move/goto") for url in posted)


def test_a_refused_call_is_not_an_outage(monkeypatch):
    # An HTTP error is the daemon answering. Counted as a failure, a few
    # refused head turns tripped the breaker the speaker shares, and the
    # reply that followed was skipped.
    import io
    import urllib.error

    from demo.robot_reachy import (
        CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, HttpReachyRobot)

    def refusing(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable",
                                     {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", refusing)
    robot = HttpReachyRobot(host="10.0.0.9")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN + 2):
        with pytest.raises(urllib.error.HTTPError):
            robot.look_at(0.0, 0.0)
    assert robot._motion.failures == 0
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse())
    robot.look_at(0.0, 0.0)  # not in a cooldown


def test_head_movements_timing_out_do_not_silence_the_reply(monkeypatch):
    # A busy daemon times out three head turns in a row: the reply that
    # follows must still reach the speaker, not fail on the movements'
    # cooldown — speech never waits on motion.
    import socket

    from demo.robot_reachy import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, HttpReachyRobot

    urls = []

    def urlopen(req, timeout=None):
        urls.append(req.full_url)
        if req.full_url.endswith("/move/goto"):
            raise socket.timeout("timed out")
        if req.full_url.endswith("/media/sounds/upload"):
            return _FakeResponse(b'{"path": "/tmp/reachy_mini_sounds/demo_reply.wav"}')
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("demo.robot_reachy.time.sleep", lambda s: None)
    robot = HttpReachyRobot(host="10.0.0.9")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            robot.look_at(0.0, 0.0)
    robot.play_sound_file(b"RIFF", 0.1)
    assert urls[-1].endswith("/media/play_sound")


def _clocked_robot(monkeypatch, urlopen):
    from demo.robot_reachy import HttpReachyRobot

    clock = {"t": 100.0}
    monkeypatch.setattr("demo.robot_reachy.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("demo.robot_reachy.time.sleep", lambda s: None)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return HttpReachyRobot(host="10.0.0.9"), clock


def test_the_speakers_cooldown_ends_and_it_counts_again_from_nothing(monkeypatch):
    import socket

    from demo.robot_reachy import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, FAILURE_COOLDOWN_S

    up = {"ok": False}

    def urlopen(req, timeout=None):
        if not up["ok"]:
            raise socket.timeout("timed out")
        return _FakeResponse(b'{"path": "/tmp/reachy_mini_sounds/demo_reply.wav"}')

    robot, clock = _clocked_robot(monkeypatch, urlopen)
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            robot._upload_sound(b"RIFF")
    with pytest.raises(TimeoutError, match="cooldown"):
        robot._upload_sound(b"RIFF")
    clock["t"] += FAILURE_COOLDOWN_S + 0.1
    up["ok"] = True
    robot._upload_sound(b"RIFF")          # through, and the count starts over
    up["ok"] = False
    with pytest.raises(OSError) as exc:
        robot._upload_sound(b"RIFF")
    assert "cooldown" not in str(exc.value), "one failure after a success is no outage"


def test_a_call_that_fails_after_its_whole_timeout_still_opens_the_breaker(monkeypatch):
    # The cooldown counts from the failure: an upload that waited its 15 s
    # timeout used to set a deadline already in the past.
    import socket

    from demo.robot_reachy import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    calls = []

    def slow_then_fails(req, timeout=None):
        calls.append(req.full_url)
        clock["t"] += 15.0
        raise socket.timeout("timed out")

    robot, clock = _clocked_robot(monkeypatch, slow_then_fails)
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            robot._upload_sound(b"RIFF")
    with pytest.raises(TimeoutError, match="cooldown"):
        robot._upload_sound(b"RIFF")
    assert len(calls) == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN


def test_a_robot_that_is_not_there_opens_both_breakers(monkeypatch):
    # Connection refused on the head's calls: the robot is gone, so the
    # reply's upload fails at once instead of waiting out its own timeout.
    import urllib.error

    from demo.robot_reachy import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    calls = []

    def refused(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    robot, _clock = _clocked_robot(monkeypatch, refused)
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            robot.look_at(0.0, 0.0)
    with pytest.raises(TimeoutError, match="cooldown"):
        robot._upload_sound(b"RIFF")
    assert not any(url.endswith("/media/sounds/upload") for url in calls)
