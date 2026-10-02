"""Pure detection conversions: detector objects <-> JSON dicts.

The dict format is what travels between the detector and the robot: class
name (not int), score, xyxy box in frame fractions. Nothing here imports a
model runtime — the voice loop on the robot uses these without one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emulator.detector import Detection

# All 80 COCO classes the detector can emit, named for the robot. The long
# names are shortened (glove, stoplight, table, …): they end up in what the
# robot says and shows, where "baseball glove" and "dining table" read worse.
COCO_NAMES = (
    "person", "bicycle", "car", "motorbike", "airplane", "bus", "train",
    "truck", "boat", "stoplight", "hydrant", "stop sign", "meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "ball", "kite", "bat", "glove", "skateboard",
    "surfboard", "racket", "bottle", "glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "plant", "bed",
    "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "phone",
    "microwave", "oven", "toaster", "sink", "fridge", "book", "clock", "vase",
    "scissors", "teddy bear", "drier", "brush",
)


def label_name(label: int) -> str:
    return COCO_NAMES[label] if 0 <= label < len(COCO_NAMES) else f"object{label}"


def detections_to_dicts(detections: list[Detection]) -> list[dict]:
    return [
        {"label": label_name(d.label), "score": round(float(d.score), 3),
         "box": [round(float(v), 3) for v in d.box]}
        for d in detections
    ]


def gaze_from_dicts(detections: list[dict]) -> tuple[float, float]:
    if not detections:
        return (0.0, 0.0)
    best = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = best["box"]
    return ((x1 + x2) - 1.0, (y1 + y2) - 1.0)
