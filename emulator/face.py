"""Who is in front of the robot: detect a face, and turn it into an identity
vector.

The models run on the laptop (demo/embed_service.py) or, with `--on-robot
faces`, on the robot itself. What comes out — a 512-d vector per face — is
stored on the robot and matched against the people it has met in its own
Qdrant shard (emulator/face_memory.py). The robot's shard is the only place a
face, a vector or a name lives; the side that computes forgets.

The pipeline separated people cleanly on a small gallery — genuine ~0.49
against impostor ~0.00 cosine, 0% EER:

  frame -> YuNet (box + 5 landmarks) -> similarity warp onto the ArcFace
  template -> HSFace (Vec2Face iResNet50) -> 512-d unit vector

The warp is not optional: the embedder was trained on faces in that exact
geometry, and skipping alignment costs about seven accuracy points. HSFace
(Vec2Face, MIT-licensed weights) was chosen because the stronger embedders
(ArcFace/InsightFace, EdgeFace) carry research-only or CC-BY-NC weights; it
was itself trained from data derived from WebFace, so check its terms before
any use beyond research. `scripts/convert_hsface.py` makes the LiteRT file.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


# The canonical InsightFace 5-point template for a 112x112 aligned crop. The
# order matches YuNet's own landmark order (right eye, left eye, nose, right
# mouth corner, left mouth corner), so nothing needs reordering.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

# YuNet's cost scales with pixels, so a bigger frame is scaled down to this
# first (the robot films 960x540).
MAX_SIDE = 640

# A face smaller than this fraction of the frame is too small to identify
# reliably — measured as the floor where recognition starts to wander.
MIN_FACE_FRACTION = 0.02


@dataclasses.dataclass(frozen=True)
class Face:
    """One detected face, in frame fractions so the dashboard and the robot
    can use the box without knowing the resolution."""

    box: list[float]           # [x1, y1, x2, y2], 0..1
    score: float
    embedding: list[float] | None = None

    @property
    def fraction(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class FaceReader:
    """Detector plus embedder, loaded once and shared across requests.

    `identities=False` is a boxes-only reader — the detect loop, which keeps
    the head on a face and never asks who it is — and needs no embedder.
    With identities, the embedder must be there at construction: a missing
    file fails the start, not the first face. It is still LOADED on the first
    embedding — 175 MB that only a turn needs.

    Not thread-safe: OpenCV's detector keeps per-call state. Each thread that
    reads faces builds its own reader, or calls under a lock."""

    def __init__(self, detector_model: Path | None = None,
                 embedder_model: Path | None = None,
                 score_threshold: float = 0.6, *,
                 identities: bool = True) -> None:
        import threading

        import cv2

        from emulator import models

        self._cv2 = cv2
        self._lock = threading.Lock()
        detector_model = Path(detector_model or models.fetch("yunet"))
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), score_threshold=score_threshold)
        self._embedder_model = (
            Path(embedder_model or models.fetch("hsface")) if identities else None)
        self._embedder = None

    def read(self, frame_rgb: "np.ndarray", *, embed: bool = True) -> list[Face]:
        """Every face in the frame, largest first; with its identity vector
        unless `embed` is off (the dashboard only needs boxes)."""
        with self._lock:
            return self._read(frame_rgb, embed)

    def _read(self, frame_rgb: "np.ndarray", embed: bool) -> list[Face]:
        cv2 = self._cv2
        bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8),
                           cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        scale = MAX_SIDE / max(height, width) if max(height, width) > MAX_SIDE else 1.0
        small = (cv2.resize(bgr, (round(width * scale), round(height * scale)))
                 if scale < 1.0 else bgr)
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, detected = self._detector.detect(small)
        if detected is None:
            return []
        faces = []
        for row in detected:
            x, y, w, h = (row[:4] / scale)
            landmarks = (row[4:14].reshape(5, 2) / scale).astype(np.float32)
            box = [float(x / width), float(y / height),
                   float((x + w) / width), float((y + h) / height)]
            face = Face(box=[min(max(v, 0.0), 1.0) for v in box],
                        score=float(row[14]))
            if (embed and self._embedder_model is not None
                    and face.fraction >= MIN_FACE_FRACTION):
                vector = self._embed(bgr, landmarks)
                if vector is not None:
                    face = dataclasses.replace(face, embedding=vector)
            faces.append(face)
        faces.sort(key=lambda f: f.fraction, reverse=True)
        return faces

    def _load_embedder(self):
        if self._embedder is None:
            from ai_edge_litert.interpreter import Interpreter

            self._embedder = Interpreter(model_path=str(self._embedder_model))
            self._embedder.allocate_tensors()
            self._input = self._embedder.get_input_details()[0]
            self._output = self._embedder.get_output_details()[0]
        return self._embedder

    def _embed(self, bgr: "np.ndarray", landmarks: "np.ndarray") -> list[float] | None:
        cv2 = self._cv2
        self._load_embedder()
        warp, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE,
                                              method=cv2.LMEDS)
        if warp is None:
            return None
        aligned = cv2.warpAffine(bgr, warp, (112, 112), borderValue=0)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # [-1, 1], NCHW — the ArcFace input convention the weights expect.
        tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        self._embedder.set_tensor(self._input["index"], tensor)
        self._embedder.invoke()
        vector = self._embedder.get_tensor(self._output["index"])[0]
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        return [float(v) for v in vector]
