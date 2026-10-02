"""The people this robot has met, on the robot.

A face vector is as personal as data gets, so this is the part of the demo that
makes the privacy claim concrete: the Mac computes a vector from a frame
(emulator/face.py) and forgets it; the vector, the name and the match all live
in a Qdrant Edge shard on the robot's own disk, next to the conversation and
the frames.

Recognition uses two thresholds rather than one, because the two mistakes are
not equal. Greeting a stranger by someone else's name is the failure everyone
in the room notices; asking a returning person for their name again is a small
awkwardness. So:

    score >= MATCH_MIN_SCORE      -> this is that person, greet them
    score >= NEW_PERSON_MAX_SCORE -> too close to call: say nothing, ask nothing
    below                         -> nobody we know: safe to meet them

The numbers come from measurements on real faces: the same person
scored around 0.49 and different people around 0.00, with 0% EER on a small
gallery. Enrollment stores several shots of one person, since one pose does not
survive someone turning their head.

One point per person, their shots together in a `face` multivector compared by
the closest (MaxSim): a search returns the person rather than one of their
shots, a new pose is added to that point, and "forget me" is one point. For a
single query face, MaxSim is the best cosine against any stored shot — the
same number the thresholds above were set on.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

# Cosine against a stored shot. 0.35 sits well below the genuine cluster (0.49)
# and far above the impostor one (0.00) — calibrate live if the camera or the
# lighting changes.
MATCH_MIN_SCORE = 0.35

# Between this and MATCH_MIN_SCORE the answer is "unsure": the robot neither
# greets by name nor offers to meet someone it may already know.
NEW_PERSON_MAX_SCORE = 0.25

KIND = "person"


@dataclasses.dataclass(frozen=True)
class Match:
    """Who the robot thinks it is looking at."""

    name: str | None
    score: float

    @property
    def known(self) -> bool:
        return self.name is not None and self.score >= MATCH_MIN_SCORE

    @property
    def is_new(self) -> bool:
        """Nobody close enough to be a person we have met — safe to enroll."""
        return self.score < NEW_PERSON_MAX_SCORE


# How many shots of one face a person's point holds, and how many of the
# first — the enrolment — are always among them.
MAX_SHOTS_PER_PERSON = 20
ENROLMENT_SHOTS_KEPT = 5


class FaceMemory:
    """The people this robot has met: the `people` shard."""

    def __init__(self, path: str | None = None, *, size: int = 512) -> None:
        from emulator.edge_store import FACE, EdgeStore, Vectors

        self._store = EdgeStore(path, vectors={FACE: Vectors(size, multivector=True)})

    def recognize(self, embedding) -> Match:
        from emulator.edge_store import FACE

        hits = self._store.search([_as_list(embedding)], 1, using=FACE)
        if not hits:
            return Match(None, 0.0)
        best = hits[0]
        return Match(best.get("name"), float(best.get("score", 0.0)))

    def enroll(self, name: str, embeddings) -> int:
        """Add shots of a person's face to their point, making it if this is
        someone new; returns how many were added. Several, not one: a single
        pose stops matching the moment someone turns their head."""
        from emulator.edge_store import FACE

        if not name:
            # A face kept under no name is a point nobody can be told apart
            # from: every face nearest to it would go unnamed for good.
            raise ValueError("a face is enrolled under a name")
        shots = [vector for vector in (_as_list(e) for e in embeddings) if vector]
        if not shots:
            return 0
        now = time.time()
        point_id = person_id(name)
        stored = self._payload(point_id)
        known = [list(shot) for shot in self._store.vectors(point_id).get(FACE, [])]
        # add() replaces the whole point, so met_at is carried over by hand.
        # The two times mean different things: `ts` moves every time a pose is
        # taught ("last taught"), `met_at` never moves again, which is the only
        # thing that can answer "when did we meet?" for a person who has been
        # re-enrolled at a new angle since.
        # Capped, or a person the robot sees every day grows without end —
        # and MaxSim compares a face against every shot. The enrolment shots
        # are kept (the pose they were named in), the rest are the latest.
        vectors = known + shots
        if len(vectors) > MAX_SHOTS_PER_PERSON:
            vectors = (vectors[:ENROLMENT_SHOTS_KEPT]
                       + vectors[-(MAX_SHOTS_PER_PERSON - ENROLMENT_SHOTS_KEPT):])
        self._store.add(point_id, {FACE: vectors},
                        {"name": name, "kind": KIND, "ts": now,
                         "met_at": _met_at(stored, now),
                         "shots": len(vectors)})
        return len(shots)

    def people(self) -> list[str]:
        """Everyone this robot has met, by name."""
        names = {payload.get("name")
                 for _id, payload in self._store.iter_points(fields=["name"])}
        return sorted(name for name in names if name)

    def people_met(self) -> list[tuple[str, float]]:
        """Everyone this robot has met and when it met them, newest first.

        What "do you remember me?" / "have we met?" / "when did we meet?" are
        answered from: those questions carry no subject a frame search can use,
        and answered from the day's pictures the robot said "I do not have any
        memory of seeing you today" while the person stood in front of it.
        """
        met = [(payload["name"], _met_at(payload))
               for _id, payload in self._store.iter_points(
                   fields=["name", "met_at", "ts"])
               if payload.get("name")]
        return sorted(met, key=lambda person: person[1], reverse=True)

    def count(self) -> int:
        """People met — one point each (see enroll)."""
        return self._store.count()

    def _payload(self, point_id: str) -> dict:
        """What is stored about a person, or {} if this is someone new."""
        rows = self._store.get([point_id])
        return rows[0][1] if rows else {}

    def forget(self, name: str) -> None:
        """Remove a person, every shot of their face with them."""
        from emulator.edge_store import match_value

        self._store.delete(match_value("name", name))

    def close(self) -> None:
        self._store.close()


def person_id(name: str) -> str:
    """The same point for the same name, every time."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reachy-person:{name}"))


def _met_at(payload: dict, default: float = 0.0) -> float:
    """When the robot first met the person this payload belongs to.

    A point written before met_at existed carries only `ts`, so that is read
    instead: "last taught" is the closest to "met" such a point holds.
    """
    for field in ("met_at", "ts"):
        value = payload.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _as_list(embedding) -> list[float]:
    return [float(value) for value in embedding]
