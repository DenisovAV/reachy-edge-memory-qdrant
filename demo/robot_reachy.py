"""The robot's body: the Reachy Mini daemon's HTTP API, or a console stub.

HttpReachyRobot drives the robot over the daemon's plain HTTP API
(:8000/api) — the same API the MuJoCo simulator's daemon serves, so one class
covers the real robot and the emulator. Not the reachy_mini SDK client: its
websocket disconnect hangs on teardown (__del__ -> ws close -> thread join),
so a clean exit blocks. HTTP sidesteps that and exposes the whole repertoire:
head and antennas via /move/goto, recorded emotions and dances via
/move/play/... . ConsoleRobot prints instead, for a run with no robot at all
(--no-robot).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from urllib.parse import quote


def gaze_to_head_pose(x: float, y: float,
                      amplitude_deg: float = 20.0) -> dict:
    """Gaze [-1, 1] -> head angles in degrees, for a point in the camera image:
    positive x is the image's right, positive y its bottom.

    Positive yaw turns the head to the robot's LEFT, which is the image's left.
    Measured on the robot (/move/goto, then the camera's own frame):
    yaw +20° moved the scene 142 px right in a 640 px image, -20° moved it
    204 px left. This used to return +yaw for a point on the right, so the head
    turned AWAY from the face it was tracking."""
    cx = max(-1.0, min(1.0, float(x)))
    cy = max(-1.0, min(1.0, float(y)))
    return {"yaw": -cx * amplitude_deg, "pitch": cy * amplitude_deg}


# Head gestures as sequences of (yaw, pitch) poses in degrees. Positive yaw is
# the robot's left (see gaze_to_head_pose). shake swings yaw, nod swings
# pitch, look_* turns and returns to center. The final pose (0, 0) brings the
# head back to straight so the gesture doesn't leave it twisted.
GESTURE_POSES: dict[str, list[tuple[float, float]]] = {
    "shake": [(-18, 0), (18, 0), (-18, 0), (18, 0), (0, 0)],
    "nod": [(0, 15), (0, -8), (0, 15), (0, 0)],
    "look_left": [(25, 0), (25, 0), (0, 0)],
    "look_right": [(-25, 0), (-25, 0), (0, 0)],
    "look_up": [(0, -20), (0, -20), (0, 0)],
    "look_down": [(0, 20), (0, 20), (0, 0)],
}

# Where the head points for the `camera` tool's direction (demo/conversation.py)
# — held, unlike a look_* gesture, while the robot describes what is there.
# 45°, not the gestures' 25°: on the robot, a 20° turn still shared most of the
# picture with straight ahead, and 45° showed a different part of the room.
LOOK_POSES: dict[str, tuple[float, float]] = {
    "ahead": (0, 0),
    "left": (45, 0),
    "right": (-45, 0),
}

# Emotions as our own gentle poses, not the recorded library moves. Live
# a recorded emotion threw the head into a pose the kinematics
# refused — "Collision detected or head pose not achievable" — and it stayed
# there until the robot was power-cycled; the play API has no speed or
# amplitude to turn down. Each step is (yaw, pitch, roll in degrees, antennas
# in radians, seconds), small and slow, and every emotion ends back at rest.
EMOTION_MOVES: dict[str, list[tuple[float, float, float, float, float]]] = {
    "happy":     [(0, -8, 0, 0.6, 0.3), (0, 6, 0, 0.1, 0.3),
                  (0, -8, 0, 0.6, 0.3), (0, 0, 0, 0.0, 0.35)],
    "excited":   [(-8, -8, 0, 0.8, 0.22), (8, -6, 0, 0.3, 0.22),
                  (-6, -8, 0, 0.8, 0.22), (0, 0, 0, 0.0, 0.3)],
    "curious":   [(10, -5, 12, 0.4, 0.45), (10, -5, 12, 0.4, 0.3),
                  (0, 0, 0, 0.0, 0.45)],
    "surprised": [(0, -14, 0, 0.9, 0.2), (0, -12, 0, 0.9, 0.35),
                  (0, 0, 0, 0.0, 0.4)],
    "confused":  [(-10, 3, 8, 0.2, 0.4), (10, 3, -8, 0.2, 0.4),
                  (0, 0, 0, 0.0, 0.4)],
    "sad":       [(0, 12, 0, -0.5, 0.6), (0, 14, 0, -0.6, 0.6),
                  (0, 0, 0, 0.0, 0.6)],
    "scared":    [(0, 8, 0, -0.7, 0.25), (-8, 8, 0, -0.7, 0.3),
                  (0, 0, 0, 0.0, 0.4)],
    "proud":     [(0, -10, 0, 0.7, 0.4), (0, -10, 0, 0.7, 0.3),
                  (0, 0, 0, 0.0, 0.4)],
}
# Anything the model asks for that is not above is played as the closest one.
EMOTION_FALLBACK = {"love": "happy", "welcoming": "happy", "sleepy": "sad",
                    "yes": "happy", "no": "confused", "cheerful": "happy"}


def emotion_moves(name: str) -> list[tuple[float, float, float, float, float]]:
    """The poses for an emotion, or the closest one's — never nothing, so a
    name the model invents still moves the body."""
    key = (name or "").strip().lower()
    return EMOTION_MOVES.get(key) or EMOTION_MOVES[EMOTION_FALLBACK.get(key, "happy")]


# The recorded-move libraries the daemon plays (full HF repo ids — the daemon
# passes them straight to huggingface_hub, so the org prefix is required; the
# short name 404s). ~85 emotions + 19 dances, played by name via
# /move/play/recorded-move-dataset.
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"

class ConsoleRobot:
    """Prints what the robot would do. For --no-robot."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def look_at(self, x: float, y: float) -> None:
        pose = gaze_to_head_pose(x, y)
        self.events.append(("look_at", pose))
        print(f"  [robot] looks: yaw {pose['yaw']:+.0f}°, "
              f"pitch {pose['pitch']:+.0f}°")

    def gesture(self, name: str) -> None:
        self.events.append(("gesture", {"name": name}))
        poses = GESTURE_POSES.get(name)
        if poses is None:
            return
        print(f"  [robot] gesture {name}: " +
              " ".join(f"({y:+.0f},{p:+.0f})" for y, p in poses))

    def look(self, direction: str) -> None:
        yaw, pitch = LOOK_POSES.get(direction, LOOK_POSES["ahead"])
        self.events.append(("look", {"direction": direction}))
        print(f"  [robot] looks {direction}: yaw {yaw:+.0f}°, pitch {pitch:+.0f}°")

    def emotion(self, name: str) -> None:
        self.events.append(("emotion", {"name": name}))
        steps = emotion_moves(name)
        print(f"  [robot] emotion {name}: " +
              " ".join(f"({y:+.0f},{p:+.0f},{r:+.0f})" for y, p, r, _a, _d in steps))

    def dance(self, name: str = "yeah_nod") -> None:
        self.events.append(("dance", {"name": name}))
        print(f"  [robot] dance {name}")

    def wake_up(self) -> None:
        self.events.append(("wake_up", {}))
        print("  [robot] wake_up")

    def sleep(self) -> None:
        self.events.append(("sleep", {}))
        print("  [robot] sleep")


# Local-network HTTP calls to the daemon — the old 8s default turned a
# dropped/unreachable robot into ~8s of dead air PER CALL on stage (look_at +
# a 5-pose gesture could add up to tens of seconds); these are calls to a box
# on the same LAN, they should fail fast.
DEFAULT_TIMEOUT_S = 1.5

# Circuit breaker for a sustained outage: once this many calls have failed in
# a row, stop paying a network timeout on every subsequent call — the
# failure is already established — and short-circuit for FAILURE_COOLDOWN_S
# instead. The window is short on purpose: it should recover automatically
# the moment the robot comes back, not stay tripped for the rest of the demo.
CONSECUTIVE_FAILURES_BEFORE_COOLDOWN = 3
FAILURE_COOLDOWN_S = 5.0


class _Breaker:
    """One run of failures and the cooldown it opened."""

    def __init__(self) -> None:
        self.failures = 0
        self.until = 0.0  # time.monotonic() deadline; 0 == not tripped

    def failed(self) -> None:
        self.failures += 1
        if self.failures >= CONSECUTIVE_FAILURES_BEFORE_COOLDOWN:
            # From the failure, not from the call: a call that took its whole
            # timeout (15 s for a reply's upload) used to set a cooldown that
            # had already passed, so the breaker never opened at all.
            self.until = time.monotonic() + FAILURE_COOLDOWN_S


# Sound uploads on the daemon land at /tmp/reachy_mini_sounds/<filename> and
# are simply OVERWRITTEN when the name repeats (measured by reading the
# daemon's own routers/media.py on the robot,
# /venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/daemon/app/
# routers/media.py) — reusing one fixed name means a multi-hour, unattended
# demo doesn't leave a fresh orphaned WAV file on the robot's /tmp on every
# single turn.
SOUND_UPLOAD_FILENAME = "demo_reply.wav"

# POST /media/play_sound returns the instant the daemon ACCEPTS the request
# (measured from the same file: play_sound() calls backend.play_sound(file) and
# returns {"status": "ok"} immediately — there is no endpoint that reports when
# playback actually finishes). Playing through afplay on the laptop blocks for
# the sound's real duration; an HTTP round trip alone tells the caller nothing
# about when the speaker goes quiet. play_sound_file below closes that gap
# itself by sleeping for the clip's own duration plus this margin (covering the
# request's own network latency, plus whatever startup delay the daemon's audio
# backend has before sound is actually audible) before returning — that is what
# makes RobotSpeakerPlayer.close() (demo/platform/robot_audio.py) actually
# block until the robot is done talking, the same guarantee StreamPlayer
# already gets for free from afplay's subprocess call.
PLAYBACK_MARGIN_S = 0.5


def _multipart_form(field: str, filename: str, content: bytes,
                    content_type: str = "audio/wav") -> tuple[bytes, str]:
    """Hand-build a one-file multipart/form-data body for
    /media/sounds/upload (a FastAPI UploadFile route) — the stdlib has no
    client-side multipart encoder, and this is the only request in the
    codebase that needs one.
    """
    boundary = "----reachy-mini-audio-boundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; '
        f'filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


class HttpReachyRobot:
    """Real Reachy Mini driven over the daemon's HTTP API (:8000/api).

    No reachy_mini SDK client — see the module docstring for why (the SDK's ws
    disconnect hangs on teardown). Every call is a short urllib POST;
    /move/goto and /move/play/... return a move uuid immediately (async on the
    daemon), so a multi-pose gesture sleeps between poses to let each finish.
    Angles go to the daemon in RADIANS (XYZRPYPose); GESTURE_POSES/gaze are in
    degrees, so they're converted.

    `host` takes a plain IP just as well as a hostname — conference/guest
    networks routinely break mDNS (reachy-mini.local) and client-to-client
    traffic (see demo/run_demo.py's --robot-host), so the daemon should be
    reachable by IP there instead.

    A run of CONSECUTIVE_FAILURES_BEFORE_COOLDOWN failures opens a
    FAILURE_COOLDOWN_S breaker (see module constants): calls made while it's
    open raise immediately without touching the network, so a dead robot
    costs one timeout, not one timeout per event for the whole outage. Motion
    and sound have a breaker each: a busy daemon timing out three head turns
    in a row used to open the one breaker the reply's upload shared, and the
    reply was silenced — speech must never wait on motion. The
    caller (demo/run_demo.py's _FailureGuard) still only LOGS the unhealthy/
    recovered transition once — this class is what keeps the per-call cost
    down for the rest of that outage.
    """

    def __init__(self, host: str = "reachy-mini.local", port: int = 8000,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._base = f"http://{host}:{port}/api"
        self._timeout = timeout
        self._motion = _Breaker()
        self._sound = _Breaker()

    def _call(self, build_request, timeout: float | None = None,
              breaker: _Breaker | None = None) -> dict:
        """Run one urllib POST through a failure-cooldown breaker — the
        motion one unless `breaker` says otherwise.

        `build_request` is a zero-arg callable returning a fresh
        `urllib.request.Request` — freshness matters because a Request's
        file-like `data` can only be sent once, and both JSON calls
        (`_post`) and the multipart sound upload (`_upload_sound`) share this
        one bookkeeping, each against the breaker of its kind.

        `timeout` overrides the instance's short control-call budget for a
        call that legitimately takes longer — the sound upload (see
        UPLOAD_TIMEOUT_S).
        """
        breaker = breaker or self._motion
        now = time.monotonic()
        if now < breaker.until:
            raise TimeoutError(
                f"{self._base}: skipping call, {breaker.until - now:.1f}s "
                "left in failure cooldown")
        try:
            with urllib.request.urlopen(
                    build_request(),
                    timeout=self._timeout if timeout is None else timeout) as resp:
                result = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError:
            # The daemon ANSWERED — it refused this one call (a pose out of
            # range, a move while another plays). The robot is there: this
            # must not trip the breaker, or a few refused head turns would
            # silence the reply that shares it.
            breaker.failures = 0
            breaker.until = 0.0
            raise
        except urllib.error.URLError:
            # The request could not even be made (connection refused, no
            # route, DNS): the robot is not there, for motion and sound alike
            # — one breaker opening for both is what keeps a gone robot from
            # costing a reply its whole upload timeout, every turn.
            self._motion.failed()
            self._sound.failed()
            raise
        except OSError:
            # Connected, but no answer in time: the daemon is busy with this
            # kind of call (a head move behind a recorded one). Only its own
            # breaker — a slow head must never silence the reply.
            breaker.failed()
            raise
        breaker.failures = 0
        breaker.until = 0.0
        return result

    def _post(self, path: str, body: dict | None = None,
              breaker: _Breaker | None = None) -> dict:
        def build() -> urllib.request.Request:
            return urllib.request.Request(
                self._base + path, data=json.dumps(body or {}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")

        return self._call(build, breaker=breaker)

    # A reply's WAV is hundreds of kilobytes (24 kHz speech: ~240 KB for five
    # seconds), and a socket timeout bounds the WHOLE transfer, not the gaps
    # in it. DEFAULT_TIMEOUT_S is 1.5s because it was tuned for tiny JSON
    # control POSTs, which would demand a sustained multi-megabit link to the
    # CM4 with no stall — fine on a clean LAN, marginal on the conference
    # WiFi this project has repeatedly had to fall back from. A timeout here
    # costs the whole spoken turn, so the upload gets its own budget.
    UPLOAD_TIMEOUT_S = 15.0

    def _upload_sound(self, wav_bytes: bytes) -> str:
        """Upload a WAV to the daemon's temp sound directory; returns the
        absolute path play_sound_file needs to hand to /media/play_sound
        (see SOUND_UPLOAD_FILENAME for why the name is fixed)."""
        payload, content_type = _multipart_form(
            "file", SOUND_UPLOAD_FILENAME, wav_bytes)

        def build() -> urllib.request.Request:
            return urllib.request.Request(
                self._base + "/media/sounds/upload", data=payload,
                headers={"Content-Type": content_type}, method="POST")

        result = self._call(build, timeout=self.UPLOAD_TIMEOUT_S, breaker=self._sound)
        return result["path"]

    def play_sound_file(self, wav_bytes: bytes, duration_s: float) -> None:
        """Upload + play a WAV clip on the robot's own speaker, then block
        for the clip's real duration (see PLAYBACK_MARGIN_S above for why).
        `duration_s` is the clip's own length in seconds — passed in by the
        caller (demo/platform/robot_audio.py's RobotSpeakerPlayer), which
        already has the sample count and rate for free rather than this
        method re-deriving it from the WAV bytes it just built.
        """
        path = self._upload_sound(wav_bytes)
        try:
            self._post("/media/play_sound", {"file": path}, breaker=self._sound)
        finally:
            # Wait even if the POST raised. The daemon starts playing the
            # moment it ACCEPTS the request and answers afterwards, so a lost
            # or late response does not stop the sound — while skipping the
            # wait reopens the microphone at t=0 of a multi-second reply. The
            # robot then hears its own voice, transcribes it, and answers
            # itself in front of the audience: the exact failure this wait
            # exists to prevent. Waiting for a sound that never played only
            # costs a beat of silence.
            time.sleep(duration_s + PLAYBACK_MARGIN_S)

    def _goto(self, *, head=None, antennas=None,
              duration: float = 0.4) -> None:
        body: dict = {"duration": duration}
        if head is not None:
            body["head_pose"] = head          # {"yaw":rad,"pitch":rad,...}
        if antennas is not None:
            body["antennas"] = list(antennas)  # [left, right]
        self._post("/move/goto", body)

    def _play(self, dataset: str, move: str) -> None:
        self._post("/move/play/recorded-move-dataset/"
                   f"{quote(dataset, safe='')}/{quote(move, safe='')}")

    # — the interface the demo already drives —
    def look_at(self, x: float, y: float) -> None:
        pose = gaze_to_head_pose(x, y)
        # Slower than the 0.5 s it was: a head that takes its time looks
        # attentive, and the tracker above sends a new target every 0.9 s.
        self._goto(head={"yaw": math.radians(pose["yaw"]),
                         "pitch": math.radians(pose["pitch"])}, duration=0.8)

    def gesture(self, name: str) -> None:
        poses = GESTURE_POSES.get(name)
        if poses is None:
            return
        for i, (yaw, pitch) in enumerate(poses):
            self._goto(head={"yaw": math.radians(yaw),
                             "pitch": math.radians(pitch)}, duration=0.25)
            if i < len(poses) - 1:
                time.sleep(0.25)  # /goto is async — let each pose land

    def look(self, direction: str) -> None:
        yaw, pitch = LOOK_POSES.get(direction, LOOK_POSES["ahead"])
        self._goto(head={"yaw": math.radians(yaw), "pitch": math.radians(pitch)},
                   duration=0.6)

    # — the `move` tool's repertoire (demo/conversation.py) —
    def emotion(self, name: str) -> None:
        for yaw, pitch, roll, antennas, duration in emotion_moves(name):
            self._goto(head={"yaw": math.radians(yaw), "pitch": math.radians(pitch),
                             "roll": math.radians(roll)},
                       antennas=[antennas, -antennas], duration=duration)
            time.sleep(duration)

    def dance(self, name: str = "yeah_nod") -> None:
        self._play(DANCES_DATASET, name)

    def wake_up(self) -> None:
        self._post("/move/play/wake_up")

    def sleep(self) -> None:
        self._post("/move/play/goto_sleep")


def make_robot(use_robot: bool, host: str = "reachy-mini.local"):
    """The robot (or the simulator) over HTTP, or a console stub (--no-robot)."""
    return HttpReachyRobot(host=host) if use_robot else ConsoleRobot()
