"""emulator/face_memory.py — who the robot has met, and the two thresholds
that decide whether it says so."""
import time

import numpy as np
import pytest

from emulator.face_memory import (MATCH_MIN_SCORE, NEW_PERSON_MAX_SCORE,
                                  FaceMemory, Match, person_id)


def _vector(*weights) -> np.ndarray:
    """A unit vector in a 512-d space, built from the first few axes so a test
    can say "almost the same face" or "a different one" exactly."""
    vector = np.zeros(512, np.float32)
    vector[:len(weights)] = weights
    return vector / np.linalg.norm(vector)


SASHA = _vector(1.0, 0.0, 0.0)
SASHA_TURNED = _vector(1.0, 0.35, 0.0)      # same person, head turned
ANNA = _vector(0.0, 1.0, 0.0)               # someone else entirely


def test_an_empty_memory_knows_nobody():
    memory = FaceMemory()
    match = memory.recognize(SASHA)
    assert match.name is None and match.is_new
    assert memory.people() == [] and memory.count() == 0


def test_a_face_is_recognised_after_being_met():
    memory = FaceMemory()
    assert memory.enroll("Sasha", [SASHA, SASHA_TURNED]) == 2
    match = memory.recognize(SASHA_TURNED)
    assert match.known and match.name == "Sasha"
    assert match.score >= MATCH_MIN_SCORE
    assert memory.people() == ["Sasha"] and memory.count() == 1  # one point per person


def test_a_different_face_is_not_that_person():
    memory = FaceMemory()
    memory.enroll("Sasha", [SASHA])
    match = memory.recognize(ANNA)
    assert not match.known
    assert match.is_new, "nothing like anyone we know — safe to meet them"


def test_a_borderline_face_is_neither_greeted_nor_enrolled():
    # The band between the two thresholds: greeting a stranger by someone
    # else's name is the mistake the room notices, so the robot says nothing.
    match = Match("Sasha", (MATCH_MIN_SCORE + NEW_PERSON_MAX_SCORE) / 2)
    assert not match.known
    assert not match.is_new


def test_several_people_stay_apart():
    memory = FaceMemory()
    memory.enroll("Sasha", [SASHA])
    memory.enroll("Anna", [ANNA])
    assert memory.recognize(SASHA).name == "Sasha"
    assert memory.recognize(ANNA).name == "Anna"
    assert memory.people() == ["Anna", "Sasha"]


def test_faces_survive_a_restart(tmp_path):
    path = str(tmp_path / "faces")
    memory = FaceMemory(path)
    memory.enroll("Sasha", [SASHA, SASHA_TURNED])
    memory.close()
    # The shard is on the robot's disk: a restarted robot still knows them.
    again = FaceMemory(path)
    assert again.recognize(SASHA).name == "Sasha"
    assert again.people() == ["Sasha"]
    again.close()


def test_an_empty_embedding_is_not_stored():
    memory = FaceMemory()
    assert memory.enroll("Nobody", [[]]) == 0
    assert memory.count() == 0


def test_one_point_per_person_that_gathers_every_shot():
    memory = FaceMemory()
    memory.enroll("Sasha", [SASHA])
    memory.enroll("Sasha", [SASHA_TURNED])
    assert memory.count() == 1
    assert memory.recognize(SASHA_TURNED).name == "Sasha"


def test_forgetting_a_person_removes_every_shot_of_them():
    memory = FaceMemory()
    memory.enroll("Sasha", [SASHA, SASHA_TURNED])
    memory.forget("Sasha")
    assert memory.people() == [] and memory.count() == 0
    assert memory.recognize(SASHA).name is None


# — met_at: the field that answers "have we met?" and "when did we meet?" —
# `ts` cannot: enroll refreshes it with every new pose, so it means "last
# taught".


def test_meeting_someone_records_when_it_happened():
    memory = FaceMemory()
    before = time.time()
    memory.enroll("Sasha", [SASHA])
    assert memory.people_met() == [("Sasha", pytest.approx(before, abs=5.0))]


def test_a_second_pose_of_the_same_face_does_not_move_when_we_met(monkeypatch):
    # The live case this exists for: a returning person is enrolled again at a
    # new angle, and "when did we meet?" must still answer the first time.
    memory = FaceMemory()
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    memory.enroll("Sasha", [SASHA])
    monkeypatch.setattr(time, "time", lambda: 9000.0)
    memory.enroll("Sasha", [SASHA_TURNED])
    assert memory.count() == 1
    assert memory.people_met() == [("Sasha", 1000.0)]


def test_each_person_keeps_their_own_meeting_newest_first(monkeypatch):
    memory = FaceMemory()
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    memory.enroll("Anna", [ANNA])
    monkeypatch.setattr(time, "time", lambda: 2000.0)
    memory.enroll("Sasha", [SASHA])
    assert memory.people_met() == [("Sasha", 2000.0), ("Anna", 1000.0)]
    assert memory.people() == ["Anna", "Sasha"], "by name, as everything else reads it"


def test_a_person_stored_before_met_at_existed_falls_back_to_ts():
    from emulator.edge_store import FACE

    memory = FaceMemory()
    # The robot's own people shard was written before this field: its points
    # carry only ts. Write one the old way and read it back.
    memory._store.add(person_id("Sasha"), {FACE: [SASHA.tolist()]},
                      {"name": "Sasha", "kind": "person", "ts": 1000.0, "shots": 1})
    assert memory.people_met() == [("Sasha", 1000.0)]


def test_teaching_an_old_point_a_pose_keeps_its_fallback(monkeypatch):
    from emulator.edge_store import FACE

    memory = FaceMemory()
    memory._store.add(person_id("Sasha"), {FACE: [SASHA.tolist()]},
                      {"name": "Sasha", "kind": "person", "ts": 1000.0, "shots": 1})
    monkeypatch.setattr(time, "time", lambda: 9000.0)
    memory.enroll("Sasha", [SASHA_TURNED])
    assert memory.people_met() == [("Sasha", 1000.0)], "not stamped with today"


def test_a_forgotten_person_is_no_longer_someone_we_have_met():
    memory = FaceMemory()
    memory.enroll("Sasha", [SASHA])
    memory.forget("Sasha")
    assert memory.people_met() == []


def test_when_we_met_survives_a_restart(tmp_path, monkeypatch):
    # The robot is restarted between two meetings — a re-enrolment after that
    # reads met_at back off the disk rather than starting the clock again.
    path = str(tmp_path / "faces")
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    memory = FaceMemory(path)
    memory.enroll("Sasha", [SASHA])
    memory.close()
    monkeypatch.setattr(time, "time", lambda: 9000.0)
    again = FaceMemory(path)
    again.enroll("Sasha", [SASHA_TURNED])
    assert again.people_met() == [("Sasha", 1000.0)]
    again.close()


def test_a_persons_shots_are_capped_keeping_the_enrolment():
    # The people shard outlives the run: without a cap, someone seen every
    # day grows without end, and every match compares against all of it.
    from emulator.face_memory import (ENROLMENT_SHOTS_KEPT, MAX_SHOTS_PER_PERSON,
                                      FaceMemory, person_id)

    memory = FaceMemory(None)
    first = [_unit(i) for i in range(ENROLMENT_SHOTS_KEPT)]
    memory.enroll("Sasha", first)
    for round_ in range(10):
        memory.enroll("Sasha", [_unit(100 + round_ * 5 + i) for i in range(5)])
    kept = memory._store.vectors(person_id("Sasha"))["face"]
    assert len(kept) == MAX_SHOTS_PER_PERSON
    assert np.allclose(np.asarray(kept[:ENROLMENT_SHOTS_KEPT]), np.asarray(first),
                       atol=1e-5)
    memory.close()


def _unit(seed: int) -> list[float]:
    vector = np.random.default_rng(seed).random(512).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def test_a_face_is_never_enrolled_under_no_name():
    # A nameless point would leave every face nearest it unnamed for good.
    import pytest

    from emulator.face_memory import FaceMemory

    memory = FaceMemory(None, size=3)
    for name in (None, ""):
        with pytest.raises(ValueError):
            memory.enroll(name, [[1.0, 0.0, 0.0]])
    assert memory.people() == []
