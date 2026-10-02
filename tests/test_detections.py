from emulator.detector import Detection
from demo.detections import detections_to_dicts, gaze_from_dicts


def test_detections_to_dicts_uses_names_and_xyxy():
    got = detections_to_dicts([Detection(label=0, score=0.912,
                                          box=(0.1, 0.2, 0.3, 0.4))])
    assert got == [{"label": "person", "score": 0.912,
                    "box": [0.1, 0.2, 0.3, 0.4]}]


def test_gaze_from_dicts_picks_most_confident_center():
    dets = [{"label": "person", "score": 0.4, "box": [0.0, 0.0, 0.2, 0.2]},
            {"label": "cup", "score": 0.9, "box": [0.4, 0.4, 0.6, 0.6]}]
    # center (0.5, 0.5) -> (0.5+0.5)-1, (0.5+0.5)-1 = (0.0, 0.0)
    assert gaze_from_dicts(dets) == (0.0, 0.0)


def test_gaze_from_dicts_empty_is_forward():
    assert gaze_from_dicts([]) == (0.0, 0.0)
