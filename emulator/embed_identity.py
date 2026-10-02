"""Which embedding models wrote a memory — recorded, and checked on open.

A vector only means something next to vectors from the same model. The demo
can now embed on the robot or on the Mac (demo/placement.py), and the shard
outlives the choice: what the robot remembered while the Mac was embedding is
read back later by whichever side is embedding then. If those are not the same
model, nothing matches and the robot simply "forgets" — silently, with a full
shard and a working search.

So the shard records who wrote it, and every open compares. The identity has
to survive being computed on two machines, which rules out paths (a laptop
cache and the robot's are in different places) and, in practice, content
digests: sha256 over SigLIP's two ONNX files costs ~18 s on the robot's CM4,
every start. What is both cheap and exact is the registry's own
content-addressed revision — the snapshot id Hugging Face resolved the repo
to — plus the repo name.

The identity is the EMBEDDING machine's: `current()` when the robot embeds,
the laptop's embed service's /health answer when the laptop does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Written beside the shard, not inside it: Qdrant Edge owns that directory's
# contents, and this has to be readable without opening the shard at all.
RECORD = "embedders.json"


def _cache_root() -> Path:
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_revision(repo: str) -> str | None:
    """The snapshot id this machine has for `repo`, or None if it has none.

    Read from the cache layout rather than the network: a start must not
    depend on Hugging Face being reachable, least of all during a talk.
    """
    folder = _cache_root() / ("models--" + repo.replace("/", "--"))
    ref = folder / "refs" / "main"
    try:
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            if revision:
                return revision
        snapshots = sorted((folder / "snapshots").iterdir(),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    return snapshots[0].name if snapshots else None


def current() -> dict:
    """What this machine would embed with, right now."""
    from emulator.frame_memory import MODEL_REPO as SIGLIP_REPO
    from emulator.memory import DEFAULT_MODEL as BGE_MODEL

    return {
        "image": {"repo": SIGLIP_REPO, "revision": hf_revision(SIGLIP_REPO)},
        "text": {"repo": BGE_MODEL, "revision": hf_revision(BGE_MODEL)},
    }


def agree(stored: dict, identity: dict) -> bool:
    """Whether two identities could have written the same vectors: the same
    vectors, from the same repos, at the same revision wherever BOTH sides
    know one. A revision only one side knows — a model one machine has not
    cached yet — is nothing to disagree with."""
    if set(stored) != set(identity):
        return False
    for vector, theirs in stored.items():
        mine = identity[vector]
        if theirs.get("repo") != mine.get("repo"):
            return False
        if (theirs.get("revision") and mine.get("revision")
                and theirs["revision"] != mine["revision"]):
            return False
    return True


def describe(identity: dict) -> str:
    return ", ".join(
        f"{vector}: {spec.get('repo')}@{(spec.get('revision') or 'unknown')[:12]}"
        for vector, spec in sorted(identity.items()))


def check(memory_path: str | Path | None, identity: dict | None = None) -> None:
    """Compare what wrote this shard with what is about to write it.

    First open records and returns; a later one that disagrees raises, naming
    both sides. An unwritable record is not fatal — the memory still works,
    it just cannot be checked next time.
    """
    if not memory_path:
        return
    identity = current() if identity is None else identity
    record = Path(memory_path).parent / RECORD
    try:
        stored = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None
    if stored is not None:
        if not agree(stored, identity):
            raise SystemExit(
                "this memory was written with different embedding models, so "
                "nothing in it would be found again:\n"
                f"  in the shard: {describe(stored)}\n"
                f"  about to run: {describe(identity)}\n"
                "Use the same models on both sides, or start a fresh memory "
                "directory.")
        if stored == identity:
            return
        # Agreeing, but one side knows a revision the record lacked: keep the
        # fuller one, so a later open is compared against it.
        identity = {vector: {**identity[vector],
                             **{k: v for k, v in stored[vector].items() if v}}
                    for vector in identity}
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(identity, indent=2, sort_keys=True),
                          encoding="utf-8")
    except OSError:
        pass


def of_service(health_url: str, timeout: float = 5.0) -> dict:
    """The embedders the laptop's embed service runs, from its /health
    (demo/embed_service.py) — the identity to check when it embeds."""
    import urllib.request

    with urllib.request.urlopen(health_url, timeout=timeout) as resp:
        return json.loads(resp.read())["embedders"]
