"""Where each model of the demo runs: on the robot, or on the laptop.

One option decides it — `--on-robot`, a comma list of model families — and
everything not named runs on the laptop over HTTP. The language model is not
a family: it is 2.5 GB against the robot's ~3 GB of free memory, and its
context cache needs more on top, so it always runs on the laptop, and asking
for it here fails at start rather than at the first reply.

Everything here reads `args` with getattr: many tests build a bare
SimpleNamespace with only the two or three fields they care about, and a
placement question must answer for those too.
"""

from __future__ import annotations

from typing import Iterable

# The families a run can move, in the order they are reported at start.
FAMILIES = ("asr", "tts", "detector", "faces", "embedder")

# What each family is called in the start report.
LABELS = {
    "asr": "speech recognition",
    "tts": "speech synthesis",
    "detector": "object detector",
    "faces": "face models",
    "embedder": "memory embeddings",
}

# Asked for often enough to be worth a real message instead of "unknown".
LLM_NAMES = ("llm", "gemma", "chat", "language-model", "language_model")

ROBOT = "robot"
MAC = "mac"


def parse(value: str | None) -> set[str]:
    """The `--on-robot` list as a set of families (empty when not given)."""
    if not value:
        return set()
    named = {part.strip().lower() for part in value.split(",") if part.strip()}
    for name in sorted(named):
        if name in LLM_NAMES:
            raise SystemExit(
                f"--on-robot: the language model cannot run on the robot "
                f"(gemma-4-E2B is 2.5 GB plus its context cache, against ~3 GB "
                f"free there with the Pollen daemon, the camera and the memory "
                f"already in it); drop \"{name}\" from the list")
        if name not in FAMILIES:
            raise SystemExit(
                f"--on-robot: no model family called \"{name}\"; "
                f"choose from {', '.join(FAMILIES)}")
    return named


def resolve(args) -> dict[str, str]:
    """{family: "robot" | "mac"} for this run."""
    named = parse(getattr(args, "on_robot", None))
    return {family: ROBOT if family in named else MAC for family in FAMILIES}


def where(args, family: str) -> str:
    """Where one family runs. Reads args every time: callers hold no state."""
    return resolve(args)[family]


def on_robot(args, family: str) -> bool:
    return where(args, family) == ROBOT


def report(args, *, models: dict[str, str] | None = None,
           addresses: dict[str, str] | None = None,
           off: Iterable[str] = ()) -> list[str]:
    """One line per family for the start log: where it runs, which model, and
    for the Mac ones the address it will be called on — or that this run has
    it `off` (faces without their models), not where it would have run.

    A run is verified from this and nothing else — which is why the model's
    name is on the line too: recognition is not the same model on both sides
    (whisper on the Mac, moonshine on the robot), and a log that said only
    "robot" would hide that.
    """
    placements = resolve(args)
    models = models or {}
    addresses = addresses or {}
    lines = []
    for family in FAMILIES:
        if family in off:
            lines.append(f"  {LABELS[family]}: off")
            continue
        place = placements[family]
        line = f"  {LABELS[family]}: {'robot' if place == ROBOT else 'Mac'}"
        model = models.get(family)
        if model:
            line += f" ({model})"
        if place == MAC and addresses.get(family):
            line += f" at {addresses[family]}"
        lines.append(line)
    return lines


def missing(family: str, what: str, cause: str) -> SystemExit:
    """The one way a local model is allowed to fail: at start, named.

    Never a silent fall back to the Mac — the whole point of a placement is
    that the log says where a model ran, and a fallback would make that a lie.
    """
    return SystemExit(
        f"{LABELS.get(family, family)} is placed on the robot but cannot be "
        f"loaded: {what} ({cause})")


def families_on_robot(args) -> Iterable[str]:
    placements = resolve(args)
    return tuple(family for family in FAMILIES
                 if placements[family] == ROBOT)
