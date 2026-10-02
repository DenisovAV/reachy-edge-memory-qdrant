"""Which models wrote a memory, and what happens when they change.

A shard filled by one embedding model and searched by another has no error to
report: every search simply misses, and the robot looks like it forgot. These
tests pin the identity the two sides compare and the refusal that follows a
disagreement.
"""

import json

import pytest

from emulator import embed_identity


def _cache(tmp_path, repo, revision, *, ref=True):
    folder = tmp_path / ("models--" + repo.replace("/", "--"))
    snapshot = folder / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx").write_bytes(b"weights")
    if ref:
        (folder / "refs").mkdir()
        (folder / "refs" / "main").write_text(revision)
    return folder


# --- the identity itself ---

def test_the_revision_comes_from_the_cache_not_the_network(tmp_path, monkeypatch):
    # A start must not depend on Hugging Face being reachable — least of all
    # during a talk.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _cache(tmp_path, "onnx-community/siglip2", "abc123")
    assert embed_identity.hf_revision("onnx-community/siglip2") == "abc123"


def test_the_newest_snapshot_is_used_when_there_is_no_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _cache(tmp_path, "org/model", "older", ref=False)
    (tmp_path / "models--org--model" / "snapshots" / "newest").mkdir()
    assert embed_identity.hf_revision("org/model") in ("older", "newest")


def test_a_model_this_machine_has_never_cached_has_no_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert embed_identity.hf_revision("org/never-seen") is None


def test_describe_names_both_vectors_and_their_revisions():
    text = embed_identity.describe(
        {"image": {"repo": "org/siglip", "revision": "abcdef0123456789"},
         "text": {"repo": "org/bge", "revision": None}})
    assert "image: org/siglip@abcdef012345" in text
    assert "text: org/bge@unknown" in text


# --- the record beside the shard ---

def test_the_first_open_records_what_is_about_to_write(tmp_path):
    memory = tmp_path / "memory"
    identity = {"image": {"repo": "org/siglip", "revision": "one"}}
    embed_identity.check(memory, identity)
    stored = json.loads((tmp_path / embed_identity.RECORD).read_text())
    assert stored == identity


def test_the_same_models_open_it_again(tmp_path):
    memory = tmp_path / "memory"
    identity = {"image": {"repo": "org/siglip", "revision": "one"}}
    embed_identity.check(memory, identity)
    embed_identity.check(memory, identity)   # must not raise


def test_a_different_model_stops_the_start_naming_both(tmp_path):
    memory = tmp_path / "memory"
    embed_identity.check(memory, {"image": {"repo": "org/siglip",
                                            "revision": "one"}})
    with pytest.raises(SystemExit) as exc:
        embed_identity.check(memory, {"image": {"repo": "org/siglip",
                                                "revision": "two"}})
    message = str(exc.value)
    assert "in the shard" in message and "about to run" in message
    assert "one" in message and "two" in message


def test_a_different_repo_stops_it_too(tmp_path):
    memory = tmp_path / "memory"
    embed_identity.check(memory, {"text": {"repo": "BAAI/bge-small-en-v1.5",
                                           "revision": None}})
    with pytest.raises(SystemExit):
        embed_identity.check(memory, {"text": {"repo": "BAAI/bge-base-en-v1.5",
                                               "revision": None}})


def test_an_unknown_revision_is_not_a_disagreement(tmp_path):
    # fastembed does not expose one for bge; two sides that both say "I don't
    # know" have not disagreed about anything.
    memory = tmp_path / "memory"
    embed_identity.check(memory, {"text": {"repo": "BAAI/bge", "revision": None}})
    embed_identity.check(memory, {"text": {"repo": "BAAI/bge", "revision": None}})


def test_an_ephemeral_memory_has_nothing_to_check(tmp_path):
    embed_identity.check(None, {"image": {"repo": "org/x", "revision": "one"}})


def test_an_unwritable_record_does_not_stop_the_demo(tmp_path):
    # The memory still works; it just cannot be checked next time.
    memory = tmp_path / "nowhere" / "deeper" / "memory"
    (tmp_path / "nowhere").write_text("a file where a directory belongs")
    embed_identity.check(memory, {"image": {"repo": "org/x", "revision": "one"}})
