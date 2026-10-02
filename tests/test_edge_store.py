"""emulator/edge_store.py — the shards the robot's memory lives in."""
from emulator.edge_store import (FACE, IMAGE, TEXT, EdgeStore, Vectors, all_of,
                                 match_value)


def test_a_shard_with_named_vectors_keeps_points_that_carry_only_some():
    store = EdgeStore(None, vectors={TEXT: 3, IMAGE: 2}, tenant_field="kind")
    said, seen = store.new_id(), store.new_id()
    store.add(said, {TEXT: [1, 0, 0]}, {"kind": "exchange"})
    store.add(seen, {IMAGE: [1, 0]}, {"kind": "frame"})
    assert [hit["kind"] for hit in store.search([1, 0, 0], 5, using=TEXT)] == ["exchange"]
    assert [hit["kind"] for hit in store.search([1, 0], 5, using=IMAGE)] == ["frame"]
    assert store.count(match_value("kind", "frame")) == 1
    assert (store.size(TEXT), store.size(IMAGE)) == (3, 2)


def test_a_vector_can_be_added_to_a_point_later():
    store = EdgeStore(None, vectors={TEXT: 3, IMAGE: 2})
    frame = store.new_id()
    store.add(frame, {IMAGE: [1, 0]}, {"kind": "frame"})
    store.set_vector(frame, TEXT, [0, 1, 0])
    assert set(store.vectors(frame)) == {IMAGE, TEXT}
    assert store.search([0, 1, 0], 1, using=TEXT)[0]["kind"] == "frame"


def test_ids_come_back_as_the_strings_they_were_given():
    store = EdgeStore(None, vectors={TEXT: 3})
    point = store.new_id()
    store.add(point, {TEXT: [1, 0, 0]}, {"n": 1})
    assert [point_id for point_id, _ in store.iter_points()] == [point]
    assert store.get([point]) == [(point, {"n": 1})]


def test_a_multivector_matches_on_its_closest_member():
    store = EdgeStore(None, vectors={FACE: Vectors(2, multivector=True)})
    store.add(store.new_id(), {FACE: [[1, 0], [0.6, 0.8]]}, {"name": "Sasha"})
    store.add(store.new_id(), {FACE: [[0, 1]]}, {"name": "Robin"})
    hits = store.search([[0.6, 0.8]], 2, using=FACE)
    assert hits[0]["name"] == "Sasha" and round(hits[0]["score"], 3) == 1.0


def test_filters_combine_and_a_delete_by_filter_removes_only_those():
    store = EdgeStore(None, vectors={TEXT: 3})
    for kind, side in (("frame", "left"), ("frame", "right"), ("exchange", None)):
        store.add(store.new_id(), {TEXT: [1, 0, 0]}, {"kind": kind, "looked": side})
    left = all_of(match_value("kind", "frame"), match_value("looked", "left"), None)
    assert store.count(left) == 1
    store.delete(match_value("kind", "frame"))
    assert store.count() == 1
    assert all_of(None) is None


def test_a_shard_on_disk_is_loaded_with_what_it_holds(tmp_path):
    path = str(tmp_path / "memory")
    first = EdgeStore(path, vectors={TEXT: 3})
    first.add(first.new_id(), {TEXT: [1, 0, 0]}, {"kind": "exchange"})
    first.close()
    again = EdgeStore(path, vectors={TEXT: 999})  # a loaded shard keeps its own
    assert again.count() == 1 and again.size(TEXT) == 3
