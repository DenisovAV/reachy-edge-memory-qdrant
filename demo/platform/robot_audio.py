"""RobotMicSource / RobotSpeakerPlayer: HTTP clients for the robot's own
ears and mouth — the two pieces that make the ROBOT the thing the audience
sees being talked TO and hears talking BACK, instead of the Mac.

RobotMicSource is a MicSource (see demo/platform/__init__.py) pulling
continuous PCM off demo/camera_service.py's /audio endpoint instead of the
Mac's own ffmpeg mic (demo/platform/mac.py's MicStream) — same
(chunk_float32, its RMS) contract, same 100 ms chunking (imported from
camera_service so this and the server can't drift on chunk cadence the way
demo/platform/robot_camera.py's own docstring warns client/service already
DID once, on port and path). The hardware is STEREO-ONLY (measured:
`arecord -D hw:0,0 ... -c 1` fails with "Channels count non available"),
but the demo pipeline is mono throughout (VAD, ASR) — see
`_downmix_to_mono` for why that conversion happens HERE and not on the
robot.

RobotSpeakerPlayer is a Player (same module) sending the buffered reply to
the robot's OWN speaker via demo/robot_reachy.py's HttpReachyRobot, because
the daemon holds the speaker exclusively (measured: `speaker-test` fails
with "Device or resource busy" while the daemon runs) — POST
/api/media/play_sound is the only path in. See its docstring, and
HttpReachyRobot.play_sound_file's, for the CRITICAL TIMING fix: that POST
returns the instant the daemon ACCEPTS the sound, not when it finishes
playing, which would otherwise make the voice loop's post-reply
mic.flush() fire while the robot is still mid-sentence.
"""
from __future__ import annotations

import http.client
import io
import logging
import threading
import urllib.request
import wave

import numpy as np

from demo.camera_service import (AUDIO_PATH, DEFAULT_MIC_CHANNELS,
                                 DEFAULT_MIC_CHUNK_MS, DEFAULT_MIC_RATE,
                                 DEFAULT_MIC_SAMPLE_WIDTH, DEFAULT_PORT)
from demo.platform.pcm import pcm_bytes_to_float
from demo.vad import rms

LOG = logging.getLogger(__name__)

# 100 ms per chunk, taken from camera_service.py's own default rather than
# restated as a second literal 100: demo/vad.py's calibrate_threshold()
# (called with ITS OWN chunk_ms=100 default from demo/run_demo.py) assumes
# every item off chunks() covers chunk_ms milliseconds, and
# demo/platform/mac.py's MicStream hard-codes the same 100 ms — all three
# have to agree for the VAD's timing math (not just its threshold) to stay
# correct.
SAMPLES_PER_CHUNK = max(1, int(DEFAULT_MIC_RATE * DEFAULT_MIC_CHUNK_MS / 1000))

# Bounded wait for a single read() on the open /audio stream — long enough
# to absorb ordinary jitter (a live realtime capture normally delivers
# ~100 ms of bytes every ~100 ms), short enough that a genuine stall reads
# as one within about a second rather than blocking the voice loop
# indefinitely ("must not hang forever waiting for a chunk that never
# comes").
READ_TIMEOUT_S = 1.0

# Reconnect backoff after a stall/EOF/connect failure — same shape as
# demo/camera_service.py's own CameraCapture/MicCapture (INITIAL_BACKOFF_S/
# MAX_BACKOFF_S): grows so a sustained outage doesn't hammer the service,
# resets the moment a read actually succeeds.
INITIAL_BACKOFF_S = 0.5
MAX_BACKOFF_S = 5.0


def mic_url(host: str, port: int = DEFAULT_PORT) -> str:
    """The one place that assembles the robot mic service's URL."""
    return f"http://{host}:{port}{AUDIO_PATH}"


def _downmix_to_mono(raw: bytes, channels: int) -> np.ndarray:
    """Interleaved S16_LE bytes, N channels -> mono float32 in [-1, 1].

    Averaging all channels, not "just take channel 0": either mic capsule
    can be the quiet one depending on which side of the robot the presenter
    stands. Done HERE, in the voice loop, not in the capture service:
    camera_service.py must stay stdlib-only (measured — no numpy/PIL on the
    robot's system python outside the daemon's own venv), while the voice
    loop already depends on numpy for every other signal-processing step
    (RMS, VAD).
    """
    flat = pcm_bytes_to_float(raw)  # interleaved ch0,ch1,...,ch0,ch1,... float32
    usable = len(flat) - (len(flat) % channels)
    frames = flat[:usable].reshape(-1, channels)
    return frames.mean(axis=1).astype(np.float32)


def _read_exact(resp, n: int) -> bytes | None:
    """Read exactly n bytes off an open streaming response, or None on a
    clean EOF before n bytes arrived. May raise (OSError/HTTPException) on a
    read timeout or dropped connection — the caller decides what that means.
    Mirrors CameraStream._read_exact (demo/platform/mac.py) and
    MicCapture._read_exact (demo/camera_service.py) — the same "assemble a
    fixed-size chunk out of however the OS hands back bytes" shape, just
    tapping the robot's HTTP stream instead of a local pipe.
    """
    parts = []
    got = 0
    while got < n:
        part = resp.read(n - got)
        if not part:
            return None
        parts.append(part)
        got += len(part)
    return b"".join(parts)


class RobotMicSource:
    """MicSource pulling continuous PCM from the robot's own mic service
    (demo/camera_service.py's /audio) — a drop-in for demo/platform/mac.py's
    MicStream: the voice loop's VAD (demo/vad.py) consumes chunks()/flush()/
    close() and must not know which one it was handed.

    `stream_factory` is a zero-arg callable returning an open, readable
    response object (`.read(n)`, `.close()`, `.headers.get(name)`) — real
    usage opens `urllib.request.urlopen(url, timeout=READ_TIMEOUT_S)`; tests
    inject a fake so no test ever opens a socket (mirrors
    camera_service.CameraCapture's own `process_factory` seam).

    Network audio can stall or drop mid-stream. `chunks()` never raises and
    never runs out on its own over that: a stall/EOF/connect failure yields
    a SILENT chunk (rms 0.0) instead — collect_utterance/calibrate_threshold
    (demo/vad.py) must never read a network hiccup as speech, and
    demo/run_demo.py's run_voice treats a chunks() generator that ends
    (StopIteration -> collect_utterance returns None) as "the mic died,
    stop the whole voice loop" (see MicStream's own docstring for that same
    behaviour on the Mac) — a transient gap must not trigger that either.
    """

    def __init__(self, url: str, stream_factory=None,
                 read_timeout: float = READ_TIMEOUT_S,
                 initial_backoff: float = INITIAL_BACKOFF_S,
                 max_backoff: float = MAX_BACKOFF_S) -> None:
        self._url = url
        self._stream_factory = stream_factory or (
            lambda: urllib.request.urlopen(url, timeout=read_timeout))
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._resp = None
        self._channels = DEFAULT_MIC_CHANNELS
        self._bytes_per_chunk = (SAMPLES_PER_CHUNK * DEFAULT_MIC_CHANNELS
                                 * DEFAULT_MIC_SAMPLE_WIDTH)
        self._consecutive_failures = 0
        self._closed = False
        self._stop_event = threading.Event()

    def _connect(self) -> None:
        """Open a fresh /audio connection and read its format headers.

        The service reports X-Sample-Rate/X-Channels/X-Sample-Format
        (demo/camera_service.py's _audio handler) rather than this client
        assuming its own defaults blindly — a mismatch is validated and
        raised, not silently misdecoded: the whole demo pipeline downstream
        hardcodes 16 kHz (see demo/run_demo.py), so audio at any other rate
        would corrupt VAD timing and ASR without a resampling step this
        client deliberately doesn't take on.
        """
        resp = self._stream_factory()
        rate = int(resp.headers.get("X-Sample-Rate", DEFAULT_MIC_RATE))
        sample_format = resp.headers.get("X-Sample-Format", "S16_LE")
        if rate != DEFAULT_MIC_RATE or sample_format != "S16_LE":
            try:
                resp.close()
            except (OSError, ValueError):
                pass
            raise ValueError(
                f"robot mic: service is {rate} Hz {sample_format!r}, this "
                f"client only decodes {DEFAULT_MIC_RATE} Hz S16_LE (what the "
                "rest of the demo pipeline assumes) — fix --mic-rate on the "
                "robot's camera_service rather than resample here")
        self._channels = int(resp.headers.get("X-Channels", DEFAULT_MIC_CHANNELS))
        self._bytes_per_chunk = (SAMPLES_PER_CHUNK * self._channels
                                 * DEFAULT_MIC_SAMPLE_WIDTH)
        self._resp = resp

    def _disconnect(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except (OSError, ValueError):
                pass  # already gone — nothing left to clean up
        self._resp = None

    def _fail(self, exc: Exception | None) -> None:
        # Log the transition only, not every failure: a dropped connection
        # fails in milliseconds and chunks() retries every read_timeout, so
        # an unguarded warning here would bury real output the instant the
        # service goes down. Same shape as RobotCameraSource's own
        # _consecutive_failures-gated logging (demo/platform/robot_camera.py).
        if self._consecutive_failures == 0:
            if exc is None:
                LOG.warning(
                    "robot mic: stream ended unexpectedly — reconnecting "
                    "(yielding silence to the VAD meanwhile, further "
                    "failures silent until it recovers)")
            else:
                LOG.warning(
                    "robot mic: %s: %s — reconnecting (yielding silence to "
                    "the VAD meanwhile, further failures silent until it "
                    "recovers)", type(exc).__name__, exc)
        self._consecutive_failures += 1

    def _succeed(self) -> None:
        if self._consecutive_failures > 0:
            LOG.info("robot mic: recovered after %d failed attempt(s)",
                     self._consecutive_failures)
        self._consecutive_failures = 0

    def _fetch_chunk(self) -> bytes | None:
        """One attempt at BYTES_PER_CHUNK of raw PCM: (re)connect if needed,
        then read. None means ANY failure (connect refused, read timeout,
        dropped connection, clean EOF) — chunks() decides what that means
        for the VAD, this method's job stops at "got data or didn't"."""
        if self._resp is None:
            try:
                self._connect()
            except (OSError, ValueError, http.client.HTTPException) as exc:
                self._fail(exc)
                return None
        try:
            raw = _read_exact(self._resp, self._bytes_per_chunk)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self._fail(exc)
            self._disconnect()
            return None
        if raw is None:
            self._fail(None)
            self._disconnect()
            return None
        self._succeed()
        return raw

    def chunks(self):
        """Generator of (mono float32 chunk, its RMS), forever until
        close() — see the class docstring for why a stall/drop yields
        silence rather than ending the generator or raising."""
        backoff = self._initial_backoff
        silence = np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32)
        while not self._closed:
            raw = self._fetch_chunk()
            if raw is None:
                yield silence, 0.0
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue
            backoff = self._initial_backoff
            chunk = _downmix_to_mono(raw, self._channels)
            yield chunk, rms(chunk)

    def flush(self) -> None:
        """Drop whatever's in flight; the next chunks() pull reconnects.

        demo/platform/mac.py's MicStream.flush() drains its ffmpeg pipe's OS
        buffer non-blockingly to discard the robot's own reply from
        whatever accumulated while nobody was reading. Here, closing and
        reopening the connection discards it EITHER way, regardless of how
        much the service itself buffers server-side: any bytes already in
        flight on the old socket are simply dropped when it closes, and a
        fresh GET to /audio starts from a brand-new, EMPTY subscriber queue
        (demo/camera_service.py's MicCapture.subscribe() docstring: "a
        client that was away for a minute cannot be handed a minute of
        stale backlog on reconnect") — exactly the discard flush() needs,
        achieved the same way a fresh GET to /frame always returns the
        newest frame rather than a backlog.
        """
        self._disconnect()

    def close(self) -> None:
        self._closed = True
        self._stop_event.set()
        self._disconnect()


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1, 1] mono audio -> an in-memory 16-bit PCM WAV file.

    Same encoding demo/audio_out.py's play_wav writes to a temp file for
    afplay; this one stays in memory because the destination is an HTTP
    upload (HttpReachyRobot.play_sound_file), not a local file afplay opens
    directly.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return buf.getvalue()


class RobotSpeakerPlayer:
    """Player sending the buffered reply to the robot's OWN speaker, via
    HttpReachyRobot.play_sound_file (demo/robot_reachy.py) — the daemon
    holds the speaker directly (measured: `speaker-test` fails with "Device
    or resource busy" while it runs), so that HTTP call is the only path in.

    Same buffer-then-one-clip contract as demo/audio_out.py's StreamPlayer:
    feed() accumulates phrases, close() concatenates and plays them as a
    single clip. The one thing close() ALSO has to match is StreamPlayer's
    "blocks until the sound is actually done" behaviour (afplay's
    subprocess.run() really does block for the clip's duration) — see
    HttpReachyRobot.play_sound_file's docstring for why that needs a
    deliberate wait here: POST /media/play_sound returns the instant the
    daemon ACCEPTS the request, not when the speaker goes quiet, and
    demo/run_demo.py's run_voice calls mic.flush() right after close()
    specifically so the mic (which the daemon does NOT hold — measured, it
    stays free) doesn't hear the tail of the robot's own reply.
    """

    def __init__(self, sample_rate: int, robot) -> None:
        self.sample_rate = int(sample_rate)
        self._robot = robot
        self._chunks: list[np.ndarray] = []

    def feed(self, samples: np.ndarray) -> None:
        """Add a phrase to the response buffer (empty ones are skipped)."""
        data = np.asarray(samples, dtype=np.float32)
        if data.size:
            self._chunks.append(data)

    def close(self) -> None:
        """Concatenate the buffered phrases, upload+play them as one clip on
        the robot's speaker, and block until the daemon should be done."""
        if not self._chunks:
            return
        audio = np.concatenate(self._chunks)
        self._chunks = []
        wav_bytes = _wav_bytes(audio, self.sample_rate)
        duration_s = len(audio) / self.sample_rate
        self._robot.play_sound_file(wav_bytes, duration_s)
