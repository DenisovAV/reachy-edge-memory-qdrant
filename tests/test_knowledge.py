"""demo/knowledge.py: facts shipped as a Qdrant Edge snapshot, restored and
searched on the robot."""
import os
import tarfile

import pytest

from demo import knowledge as kb


class _Words:
    """A stand-in embedder: one dimension per word it knows, so a query
    matches the fact that shares its word."""
    VOCAB = ("rust", "edge", "license", "founded")

    def _vector(self, text):
        low = text.lower()
        return [1.0 if word in low else 0.0 for word in self.VOCAB] + [0.1]

    def embed(self, texts):
        return [self._vector(text) for text in texts]

    query_embed = embed


FACTS = [((), "Qdrant is written in Rust."),
         ((), "Qdrant Edge runs inside the application."),
         ((), "Qdrant uses the Apache license."),
         ((), "Qdrant was founded in 2021.")]


def test_facts_are_read_with_every_way_of_asking_them(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("# a comment\n\nWhat is it? | First fact.\n"
                    "What's that? |\n  Tell me about it.  |\n   \n"
                    "  Second fact.  \nWho? | Third fact.\n")
    assert kb.read_facts(path) == [
        (("What is it?", "What's that?", "Tell me about it."), "First fact."),
        ((), "Second fact."),
        (("Who?",), "Third fact.")]


def test_a_phrasing_with_no_fact_above_it_is_an_error(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("What is it? |\n")
    with pytest.raises(ValueError):
        kb.read_facts(path)


def test_a_fact_is_one_point_with_its_phrasings_as_a_multivector(tmp_path):
    snapshot = tmp_path / "facts.snapshot"
    kb.build_snapshot([(("Who founded it?", "When was it founded?"),
                        "Two people, in 2021.")], snapshot, _Words())
    base = kb.open_knowledge(snapshot, tmp_path / "knowledge", _Words())
    try:
        assert base.count() == 1
        [(point_id, payload)] = list(base._store.iter_points())
        assert len(base._store.vectors(point_id)["text"]) == 2
        assert payload["questions"] == ["Who founded it?", "When was it founded?"]
    finally:
        base.close()


def test_a_fact_is_found_by_its_closest_phrasing_and_only_the_fact_returns(tmp_path):
    # The second phrasing carries the word; the fact itself does not.
    snapshot = tmp_path / "facts.snapshot"
    kb.build_snapshot([(("Who started it?", "When was it founded?"),
                        "Two people, in 2021."),
                       ((), "Qdrant is written in Rust.")], snapshot, _Words())
    base = kb.open_knowledge(snapshot, tmp_path / "knowledge", _Words())
    try:
        hits = base.search("founded", k=1)
        assert [hit["text"] for hit in hits] == ["Two people, in 2021."]
    finally:
        base.close()


def test_the_sparse_tar_round_trips_holes_data_and_empty_files(tmp_path):
    src = tmp_path / "src"
    (src / "segments" / "one").mkdir(parents=True)
    holes = bytearray(3 * kb.PAGE + 100)
    holes[kb.PAGE + 7] = 1               # data between two holes, and a hole at the end
    (src / "segments" / "one" / "chunk.mmap").write_bytes(bytes(holes))
    (src / "zeros.dat").write_bytes(bytes(5 * kb.PAGE))
    (src / "empty").write_bytes(b"")
    many = bytearray(12 * kb.PAGE)       # more regions than the header holds
    for page in range(0, 12, 2):
        many[page * kb.PAGE] = 9
    (src / "many.dat").write_bytes(bytes(many))
    (src / "edge_config.json").write_text('{"vectors": {}}')

    archive = tmp_path / "shard.snapshot"
    kb.write_sparse_tar(src, archive)
    out = tmp_path / "out"
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="tar")

    for path in src.rglob("*"):
        if path.is_file():
            assert (out / path.relative_to(src)).read_bytes() == path.read_bytes(), path
    assert os.path.getsize(archive) < len(holes) + len(many)


def test_a_built_snapshot_restores_into_a_searchable_edge_shard(tmp_path):
    snapshot = tmp_path / "facts.snapshot"
    assert kb.build_snapshot(FACTS, snapshot, _Words()) == 4

    base = kb.open_knowledge(snapshot, tmp_path / "knowledge", _Words())
    try:
        assert base.count() == 4
        hits = base.search("What language, Rust?", k=1)
        assert [hit["text"] for hit in hits] == ["Qdrant is written in Rust."]
        assert hits[0]["source"] == "knowledge"
    finally:
        base.close()


def test_restoring_again_replaces_the_earlier_copy(tmp_path):
    snapshot = tmp_path / "facts.snapshot"
    kb.build_snapshot(FACTS, snapshot, _Words())
    kb.open_knowledge(snapshot, tmp_path / "knowledge", _Words()).close()
    base = kb.open_knowledge(snapshot, tmp_path / "knowledge", _Words())
    try:
        assert base.count() == 4
    finally:
        base.close()


def test_facts_under_the_gate_are_not_answers(tmp_path):
    snapshot = tmp_path / "facts.snapshot"
    kb.build_snapshot(FACTS, snapshot, _Words())
    kb.restore(snapshot, tmp_path / "knowledge")
    base = kb.KnowledgeBase(tmp_path / "knowledge", _Words(), min_score=0.5)
    try:
        assert base.search("tell me a joke") == []
    finally:
        base.close()


def test_a_missing_shard_is_an_error_not_an_empty_knowledge_base(tmp_path):
    with pytest.raises(FileNotFoundError):
        kb.KnowledgeBase(tmp_path / "nothing", _Words())
    assert not (tmp_path / "nothing").exists()


def test_the_snapshot_is_stale_when_missing_or_older_than_its_facts(tmp_path):
    facts = tmp_path / "facts.txt"
    facts.write_text("A fact.\n")
    snapshot = tmp_path / "facts.snapshot"
    assert kb.is_stale(facts, snapshot)
    snapshot.write_bytes(b"x")
    os.utime(facts, (1000, 1000))
    os.utime(snapshot, (2000, 2000))
    assert not kb.is_stale(facts, snapshot)
    os.utime(facts, (3000, 3000))
    assert kb.is_stale(facts, snapshot)


def test_the_shipped_snapshot_holds_exactly_the_shipped_facts(tmp_path):
    """A fact edited without `python -m demo.knowledge build` would reach the
    robot's facts file but never its snapshot."""
    kb.restore(kb.SNAPSHOT_PATH, tmp_path / "knowledge")
    base = kb.KnowledgeBase(tmp_path / "knowledge", _Words())
    try:
        stored = sorted(payload["text"] for _id, payload in base._store.iter_points())
    finally:
        base.close()
    assert stored == sorted(fact for _questions, fact in kb.read_facts())


def test_every_shipped_fact_can_be_asked_more_than_one_way():
    # The multivector is the point: a fact with one phrasing is a fact that
    # only one way of asking can find.
    single = [fact for questions, fact in kb.read_facts() if len(questions) < 2]
    assert single == []
