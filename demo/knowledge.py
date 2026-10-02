"""The robot's knowledge base: what it knows about Qdrant, shipped as a snapshot.

A third store, and a different kind from the other two. Memory is what the
robot lived through — exchanges (emulator/memory.py) and frames
(emulator/frame_memory.py) — and it grows on the robot. Knowledge is prepared
before the robot ever runs: the facts in demo/qdrant_facts.txt are embedded on
the laptop, written into a Qdrant Edge shard and packed as a snapshot; the
robot restores that snapshot when it starts, with `EdgeShard.unpack_snapshot` —
the same call Qdrant Edge uses for a shard snapshot downloaded from a Qdrant
server — and searches it in-process. Nothing the robot hears is ever written
into it.

A fact is one point, and the questions it answers are its vectors: several
phrasings as a multivector, compared with MaxSim — the way a person's face
shots are stored (emulator/face_memory.py). A short question is compared with
equally short questions, not with the long sentence that answers them: "How
do you work?" has no word with a subject in it, and against its fact it
scored 0.62, under the gate; against its phrasings, 0.93.

    uv run python -m demo.knowledge build     # after editing the facts
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTS_PATH = HERE / "qdrant_facts.txt"
# Next to the code, so scripts/robot_service.sh's rsync of demo/ carries it to
# the robot with everything else.
SNAPSHOT_PATH = HERE / "qdrant_knowledge.snapshot"

# The bge cosine a question needs, against the nearest phrasing of a fact, to
# count as an answer. Measured on 32 questions worded unlike any phrasing in
# the file, and on 24 that are not about the facts at all — about the past,
# the person, small talk: every question that is not about the person asking
# searches here too (demo/conversation.py). The facts' questions scored 0.729
# at the lowest but one ("can you introduce yourself?", 0.677); the others
# 0.713 at the highest ("how are you today?"). Midway.
KNOWLEDGE_MIN_SCORE = 0.72

BLOCK = 512
# Holes are found a filesystem page at a time.
PAGE = 4096


def read_facts(path: str | os.PathLike = FACTS_PATH) -> list[tuple[tuple[str, ...], str]]:
    """(questions, fact) per fact. A line "question | fact" starts a fact; a
    line "question |", with nothing after the bar, is one more way to ask the
    fact above it; a line with no bar is a fact with no question. Blank lines
    and # comments skipped."""
    facts: list[tuple[list[str], str]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        question, bar, fact = line.partition("|")
        question, fact = question.strip(), fact.strip()
        if not bar:
            facts.append(([], question))
        elif fact:
            facts.append(([question] if question else [], fact))
        elif facts and question:
            facts[-1][0].append(question)
        else:
            raise ValueError(f"{path}:{number}: a phrasing needs a fact above it")
    return [(tuple(questions), fact) for questions, fact in facts]


def build_snapshot(facts, snapshot_path: str | os.PathLike, embedder) -> int:
    """One point per fact, its phrasings as a multivector (document side —
    they are what is stored), the fact as the payload a search returns; packed
    at `snapshot_path`. A fact with no question is its own only vector.
    Returns how many facts went in."""
    from emulator.edge_store import TEXT, EdgeStore, Vectors

    facts = [(tuple(questions) or (fact,), fact) for questions, fact in facts]
    texts = [text for questions, _fact in facts for text in questions]
    flat = [[float(x) for x in vector] for vector in embedder.embed(texts)]
    with tempfile.TemporaryDirectory(prefix="knowledge-") as work:
        shard_dir = os.path.join(work, "shard")
        store = EdgeStore(shard_dir, vectors={
            TEXT: Vectors(len(flat[0]), multivector=True)})
        start = 0
        for questions, fact in facts:
            vectors = flat[start:start + len(questions)]
            start += len(questions)
            store.add(store.new_id(), {TEXT: vectors},
                      {"kind": "fact", "text": fact, "questions": list(questions)})
        store.close()
        write_sparse_tar(shard_dir, snapshot_path)
    return len(facts)


def restore(snapshot_path: str | os.PathLike, target_dir: str | os.PathLike) -> None:
    """Unpack the snapshot into `target_dir`, replacing what an earlier start
    restored there — the directory only ever holds a copy of the snapshot."""
    from qdrant_edge import EdgeShard

    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    EdgeShard.unpack_snapshot(str(snapshot_path), str(target_dir))


class KnowledgeBase:
    """A restored knowledge shard, searched by meaning. `embedder` has the
    fastembed shape TextMemory uses (demo/embed_client.py's
    RemoteBgeEmbedder on the robot)."""

    def __init__(self, path: str | os.PathLike, embedder, *,
                 min_score: float = KNOWLEDGE_MIN_SCORE) -> None:
        from emulator.edge_store import TEXT, EdgeStore

        if not os.path.exists(os.path.join(path, "wal")):
            # EdgeStore would quietly create an empty shard here instead.
            raise FileNotFoundError(f"no restored knowledge shard at {path}")
        self._embedder = embedder
        self._min_score = min_score
        self._store = EdgeStore(str(path), vectors={TEXT: 1})  # loaded, never created here

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Facts whose nearest phrasing is closest to `query`, best first,
        those under the gate dropped. The query is a multivector of one:
        MaxSim then scores each fact by its closest phrasing."""
        vector = next(iter(self._embedder.query_embed([query])))
        from emulator.edge_store import TEXT

        hits = self._store.search([[float(x) for x in vector]], k, using=TEXT)
        return [{**hit, "source": "knowledge"} for hit in hits
                if hit["score"] >= self._min_score]

    def count(self) -> int:
        return self._store.count()

    def close(self) -> None:
        self._store.close()


def open_knowledge(snapshot_path: str | os.PathLike, target_dir: str | os.PathLike,
                   embedder) -> KnowledgeBase:
    restore(snapshot_path, target_dir)
    return KnowledgeBase(target_dir, embedder)


def is_stale(facts_path=FACTS_PATH, snapshot_path=SNAPSHOT_PATH) -> bool:
    """True when the snapshot is missing or older than the facts it holds."""
    return (not os.path.exists(snapshot_path)
            or os.path.getmtime(snapshot_path) < os.path.getmtime(facts_path))


# — the snapshot file —

def write_sparse_tar(src_dir: str | os.PathLike, archive_path: str | os.PathLike) -> None:
    """Pack a shard directory as a tar whose files are GNU sparse entries.

    An Edge shard preallocates its files — the WAL, the vector chunks, the
    payload pages — so a shard of sixty-odd facts holds over a hundred
    megabytes of mostly zeros while using well under one on disk. A plain tar
    stores every zero (136 MB, measured); Python's tarfile cannot write sparse
    entries, and bsdtar's sparse pax entries are not read by Edge (restored,
    the segment failed to load). Edge's unpack reads old GNU sparse entries
    (lib/common/common/src/tar_unpack.rs allows GNUSparse), so this writes
    those: the data pages, and where the holes are.
    """
    src_dir = os.fspath(src_dir)
    with open(archive_path, "wb") as out:
        for root, dirs, files in os.walk(src_dir):
            dirs.sort()
            rel_root = os.path.relpath(root, src_dir)
            if rel_root != ".":
                out.write(_header(rel_root + "/", b"5", 0, mode=0o755,
                                  mtime=int(os.path.getmtime(root))))
            for name in sorted(files):
                path = os.path.join(root, name)
                rel = os.path.normpath(os.path.join(rel_root, name))
                _write_sparse_file(out, path, rel)
        out.write(bytes(2 * BLOCK))


def _write_sparse_file(out, path: str, name: str) -> None:
    size = os.path.getsize(path)
    regions = _data_regions(path, size)
    data_size = sum(length for _offset, length in regions)
    out.write(_header(name, b"S", data_size, mode=0o644,
                      mtime=int(os.path.getmtime(path)),
                      real_size=size, regions=regions))
    rest = regions[4:]
    while rest:
        block = bytearray(BLOCK)
        for i, (offset, length) in enumerate(rest[:21]):
            block[24 * i:24 * i + 24] = _octal(offset, 12) + _octal(length, 12)
        rest = rest[21:]
        block[504] = 1 if rest else 0
        out.write(block)
    with open(path, "rb") as f:
        for offset, length in regions:
            f.seek(offset)
            out.write(f.read(length))
    out.write(bytes(-data_size % BLOCK))


def _data_regions(path: str, size: int) -> list[tuple[int, int]]:
    """(offset, length) of the parts of the file that are not all zeros,
    each but the last a whole number of pages (the reader requires 512-byte
    alignment)."""
    regions: list[tuple[int, int]] = []
    with open(path, "rb") as f:
        for offset in range(0, size, PAGE):
            page = f.read(PAGE)
            if not page.strip(b"\0"):
                continue
            if regions and sum(regions[-1]) == offset:
                regions[-1] = (regions[-1][0], regions[-1][1] + len(page))
            else:
                regions.append((offset, len(page)))
    if not regions or sum(regions[-1]) < size:
        # A trailing hole is written as an empty region at the end of the
        # file: Edge's reader requires the regions to reach the file's size.
        regions.append((size, 0))
    return regions


def _octal(value: int, width: int) -> bytes:
    return b"%0*o\0" % (width - 1, value)


def _header(name: str, typeflag: bytes, size: int, *, mode: int, mtime: int,
            real_size: int | None = None,
            regions: list[tuple[int, int]] = ()) -> bytes:
    raw = name.encode()
    if len(raw) >= 100:
        raise ValueError(f"path too long for a tar header: {name}")
    header = bytearray(BLOCK)
    header[0:len(raw)] = raw
    header[100:108] = _octal(mode, 8)
    header[108:116] = _octal(0, 8)
    header[116:124] = _octal(0, 8)
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(mtime, 12)
    header[148:156] = b" " * 8
    header[156:157] = typeflag
    header[257:265] = b"ustar  \0"
    if real_size is not None:
        for i, (offset, length) in enumerate(regions[:4]):
            at = 386 + 24 * i
            header[at:at + 24] = _octal(offset, 12) + _octal(length, 12)
        header[482] = 1 if len(regions) > 4 else 0
        header[483:495] = _octal(real_size, 12)
    header[148:156] = b"%06o\0 " % sum(header)
    return bytes(header)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("build",))
    p.add_argument("--facts", default=str(FACTS_PATH))
    p.add_argument("--out", default=str(SNAPSHOT_PATH))
    args = p.parse_args(argv)
    from emulator.memory import DEFAULT_MODEL, _embedder

    count = build_snapshot(read_facts(args.facts), args.out, _embedder(DEFAULT_MODEL))
    print(f"{count} facts -> {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
