"""Three slot files on disk: written whole or not at all, listed by header."""

import json
import os

import pytest

from src import save
from src.app import saves


def document(name="Solar Year 1.0", version=save.VERSION):
    return {"version": version, "name": name, "saved_at": "2026-09-12T14:03",
            "summary": {"scenario_label": "Solar", "days": 365.25, "bodies": 9},
            "world": {"time": 365.25, "next_id": 20, "bodies": []},
            "session": {}}


@pytest.fixture
def store(tmp_path):
    return saves.SlotStore(str(tmp_path / "saves"))


def test_there_are_exactly_three_slots(store):
    assert saves.SLOTS == 3
    assert store.list() == [None, None, None]
    with pytest.raises(ValueError):
        store.path(0)
    with pytest.raises(ValueError):
        store.path(4)


def test_the_directory_is_created_on_first_write(store):
    assert not os.path.isdir(store.directory)
    store.write(2, document())
    assert os.path.isdir(store.directory)
    assert os.path.basename(store.path(2)) == "slot2.json"
    assert os.path.exists(store.path(2))


def test_a_write_leaves_no_temporary_file_behind(store):
    store.write(1, document())
    assert sorted(os.listdir(store.directory)) == ["slot1.json"]


def test_listing_reads_the_header(store):
    store.write(3, document(name="Mars trip"))
    listed = store.list()
    assert listed[0] is None and listed[1] is None
    info = listed[2]
    assert info.slot == 3 and info.name == "Mars trip"
    assert info.saved_at == "2026-09-12T14:03"
    assert info.summary["bodies"] == 9
    assert info.readable


def test_a_corrupt_file_lists_as_unreadable_and_does_not_raise(store):
    os.makedirs(store.directory)
    with open(store.path(1), "w") as f:
        f.write("{not json")
    info = store.list()[0]
    assert info is not None and not info.readable
    assert info.name == saves.UNREADABLE_NAME
    with pytest.raises(ValueError):
        store.read(1)


def test_a_wrong_version_lists_as_unreadable(store):
    store.write(1, document(version=99))
    assert not store.list()[0].readable
    with pytest.raises(save.SaveFormatError):
        store.read(1)


def test_rename_changes_only_the_name(store):
    store.write(1, document())
    store.rename(1, "Kaboom")
    data = store.read(1)
    expected = document(name="Kaboom")
    assert data == expected
    assert store.list()[0].name == "Kaboom"


def test_read_returns_what_was_written(store):
    doc = document()
    store.write(2, doc)
    assert store.read(2) == json.loads(json.dumps(doc))


def test_writable_creates_and_reports(store):
    assert store.writable()
    assert os.path.isdir(store.directory)


def test_writable_is_false_where_it_cannot_create(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    store = saves.SlotStore(str(blocker / "saves"))
    assert not store.writable()


def test_default_directory_is_beside_run_command():
    root = saves.default_directory()
    assert os.path.basename(root) == "saves"
    assert os.path.exists(os.path.join(os.path.dirname(root), "run.command"))
