"""The Reading desk: page state in, clicks out. See
docs/superpowers/specs/2026-09-29-reading-desk-design.md."""

import json
import sqlite3
import time

from conftest import seeded_db

from attestation import desk
from attestation.rank import get_user


def add_item(conn, title="a paper", url="https://example.org/a", summary="about it"):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash) VALUES (NULL, ?, ?, ?, ?)",
        (title, url, summary, f"hash-{title}"),
    )
    conn.commit()
    return cur.lastrowid


def page_state_file(path, value, revision=1):
    """The exact table private_pages.py's page_state() creates
    (hermes-starters/shared-skills/agentmarkit-links/scripts/private_pages.py,
    page_state): `state(id INTEGER PRIMARY KEY, revision INTEGER, value TEXT)`,
    one row at id=1 holding the JSON blob. Copied, not invented, so this test
    reads the format the Worker really writes."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, revision INTEGER, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT, result TEXT)"
    )
    conn.execute("INSERT OR REPLACE INTO state VALUES (1,?,?)", (revision, json.dumps(value)))
    conn.commit()
    conn.close()
    return path


def verdict(useful, at="2026-09-29T14:00:00Z"):
    return {"useful": useful, "at": at}


def clicks(conn, user_id):
    return {
        r["item_id"]: (r["useful"], r["source"])
        for r in conn.execute(
            "SELECT item_id, useful, source FROM clicks WHERE user_id = ?", (user_id,)
        )
    }


def test_import_records_each_verdict_as_a_ui_click(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a, b = add_item(conn, "a"), add_item(conn, "b")
    state = {"v": 1, "verdicts": {str(a): verdict(True), str(b): verdict(False)}}

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=2, unchanged=0, skipped=0)
    assert clicks(conn, uid) == {a: (1, "ui"), b: (0, "ui")}


def test_import_is_idempotent(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {"v": 1, "verdicts": {str(a): verdict(True)}}

    desk.import_verdicts(conn, uid, state)
    again = desk.import_verdicts(conn, uid, state)

    assert again == desk.Imported(recorded=0, unchanged=1, skipped=0)
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 1


def test_a_changed_mind_overwrites(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(True)}})

    got = desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(False)}})

    assert got.recorded == 1
    assert clicks(conn, uid) == {a: (0, "ui")}


def test_malformed_and_unknown_entries_are_skipped_not_raised(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {
        "v": 1,
        "verdicts": {
            "999999": verdict(True),  # no such item
            "abc": verdict(True),  # not an id
            str(a): {"useful": "yes", "at": "x"},  # not a bool
            "-3": verdict(True),
        },
    }

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=0, unchanged=0, skipped=4)
    assert clicks(conn, uid) == {}


def test_no_verdicts_key_imports_nothing(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    assert desk.import_verdicts(conn, uid, {}) == desk.Imported()
    assert desk.import_verdicts(conn, uid, {"v": 1, "verdicts": []}) == desk.Imported()


def test_read_state_reads_the_worker_format(tmp_path):
    value = {"v": 1, "verdicts": {"7": verdict(True)}}
    path = page_state_file(tmp_path / "state.sqlite", value)
    assert desk.read_state(path) == value


def test_read_state_missing_file_or_wrong_version_is_empty(tmp_path):
    assert desk.read_state(tmp_path / "nope.sqlite") == {}
    path = page_state_file(tmp_path / "s.sqlite", {"v": 2, "verdicts": {}})
    assert desk.read_state(path) == {}
    garbage = tmp_path / "g.sqlite"
    garbage.write_bytes(b"not a database at all" * 100)
    assert desk.read_state(garbage) == {}


def test_read_state_never_writes(tmp_path):
    path = page_state_file(tmp_path / "state.sqlite", {"v": 1, "verdicts": {}})
    before = path.read_bytes()
    desk.read_state(path)
    assert path.read_bytes() == before
    assert not (tmp_path / "state.sqlite-journal").exists()


def test_read_state_returns_promptly_while_the_worker_holds_a_write_lock(tmp_path):
    """private_pages.py saves inside BEGIN IMMEDIATE. A reader in rollback-journal
    mode can still read during a RESERVED lock; this pins that the import does
    not hang or raise while a save is in flight."""
    value = {"v": 1, "verdicts": {"7": verdict(True)}}
    path = page_state_file(tmp_path / "state.sqlite", value)
    writer = sqlite3.connect(path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        start = time.monotonic()
        got = desk.read_state(path)
        assert time.monotonic() - start < 6
        assert got in (value, {})
    finally:
        writer.rollback()
        writer.close()


def test_import_pending_is_none_when_unconfigured(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    monkeypatch.delenv("ATTEST_DESK_STATE", raising=False)
    monkeypatch.delenv("ATTEST_DESK_USER", raising=False)
    assert desk.import_pending(conn) is None
    monkeypatch.setenv("ATTEST_DESK_STATE", str(tmp_path / "s.sqlite"))
    assert desk.import_pending(conn) is None  # user still unset


def test_import_pending_imports_for_the_desk_persona(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "Researcher")  # case folds, like get_user

    got = desk.import_pending(conn)

    assert got == desk.Imported(recorded=1)
    assert clicks(conn, uid) == {a: (1, "ui")}


def test_import_pending_unknown_persona_imports_nothing(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "nobody")

    assert desk.import_pending(conn) is None
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 0
    assert get_user(conn, "nobody") is None, "import must never autocreate a persona"


def test_import_pending_swallows_a_database_error(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {"1": verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(desk, "import_verdicts", boom)
    assert desk.import_pending(conn) is None
