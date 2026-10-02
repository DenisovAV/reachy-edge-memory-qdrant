"""Platform I/O seam: camera, mic, speaker, and robot behind protocols.

The voice loop (`demo/run_demo.py::run_voice`) reads a camera and a mic,
plays a reply and moves a robot without knowing where each one is: the
laptop's own camera and mic through ffmpeg, or the robot's through its
camera service (demo/camera_service.py); afplay on the laptop, or the
robot's speaker through its daemon. demo/platform/mac.py picks each one
from the command line.

Platform-independent pure logic (RMS, `VoiceGate`, `collect_utterance`,
`calibrate_threshold`) stays in `demo/vad.py`; pure frame and PCM
conversions live in `demo/platform/pcm.py`.
"""
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class VideoSource(Protocol):
    """Camera frame source: `CameraStream` (mac.py) or `RobotCameraSource`."""

    def latest(self) -> np.ndarray | None: ...

    def latest_png(self) -> bytes | None: ...

    def close(self) -> None: ...


class MicSource(Protocol):
    """Mic audio source: `MicStream` (mac.py) or `RobotMicSource`.

    `chunks()` — a generator of (float32 chunk, its RMS) pairs: the contract
    `collect_utterance`/`calibrate_threshold` in `demo/vad.py` consume.
    """

    def chunks(self) -> Iterator[tuple[np.ndarray, float]]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class Player(Protocol):
    """Reply player: `StreamPlayer` (demo/audio_out.py) or `RobotSpeakerPlayer`."""

    def feed(self, samples: np.ndarray) -> None: ...

    def close(self) -> None: ...


class Robot(Protocol):
    """The robot's body — HttpReachyRobot or ConsoleRobot
    (demo/robot_reachy.py)."""

    def look_at(self, x: float, y: float) -> None: ...

    def gesture(self, name: str) -> None: ...

    def look(self, direction: str) -> None:
        """Turn the head "left", "right" or "ahead" and hold it there."""
        ...

    def emotion(self, name: str) -> None:
        """Play one of the `move` tool's emotions."""
        ...

    def dance(self, name: str = "yeah_nod") -> None: ...


class Platform(Protocol):
    """Platform backend: bundles camera/mic/player/robot under one node."""

    def video_source(self) -> VideoSource: ...

    def mic_source(self) -> MicSource: ...

    def make_player(self, sample_rate: int) -> Player: ...

    def robot(self) -> Robot: ...

    def close(self) -> None: ...
