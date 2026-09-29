"""The Reading desk: page state in, clicks out. See
docs/superpowers/specs/2026-09-29-reading-desk-design.md."""

import json
import re
import shutil
import sqlite3
import subprocess
import time

import pytest
from conftest import seeded_db

from attestation import desk
from attestation.rank import get_user, record_click


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


# --- rendering ----------------------------------------------------------------

HOST_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "font-src data:; img-src data:; connect-src 'none'; form-action 'none'; "
    "base-uri 'none'; frame-src 'none'; object-src 'none'; worker-src 'none'"
)


def ranked_db(tmp_path, embedder, n=3, **item):
    """A seeded DB with n embedded items, ranked for 'researcher'."""
    conn = seeded_db(tmp_path / "t.db")
    ids = []
    for i in range(n):
        title = item.get("title", f"paper {i}")
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash) VALUES (NULL,?,?,?,?)",
            (
                title,
                item.get("url", f"https://example.org/{i}"),
                item.get("summary", "abstract"),
                f"h{i}",
            ),
        )
        vec = embedder.embed_document(title, "abstract")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vec.tobytes()),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return conn, get_user(conn, "researcher")["id"], ids


def test_page_is_self_contained(tmp_path, fake_embedder):
    conn, uid, _ = ranked_db(tmp_path, fake_embedder)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()
    assert "http:" not in html
    assert "fetch(" not in html
    item_urls = {f"https://example.org/{i}" for i in range(3)}
    assert set(re.findall(r"https://[^\"'\s<\\]+", html)) <= item_urls
    assert len(html.encode()) < 8 * 1024 * 1024


def test_page_has_both_tabs_and_every_item(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert 'data-tab="triage"' in html and 'data-tab="read"' in html
    assert sorted(i["id"] for i in payload["items"]) == sorted(ids)
    assert all(i["summary"] == "abstract" for i in payload["items"])


def test_empty_ranking_renders_the_waiting_state(tmp_path, fake_embedder):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    html = desk.render_desk(conn, fake_embedder, uid)
    assert "No papers yet" in html


def test_caveat_shown_while_the_classifier_is_off(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    assert payload["caveat"]  # zero clicks: classifier inactive
    assert payload["rated"] == 0


def test_rated_counts_human_verdicts_only(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder, n=4)
    record_click(conn, uid, ids[0], True, source="ui")
    record_click(conn, uid, ids[1], False, source="agent")
    record_click(conn, uid, ids[2], True, source="simulated")
    assert desk.desk_payload(conn, fake_embedder, uid)["rated"] == 2


def test_hostile_title_cannot_close_the_data_script(tmp_path, fake_embedder):
    title = "</script><script>alert(1)</script>" + "x" * 10_000_000
    conn, uid, _ = ranked_db(tmp_path, fake_embedder, n=1, title=title)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert html.count("</script>") == html.count("<script")  # only our own tags close
    assert len(html.encode()) < 200_000


@pytest.mark.parametrize("url", ["http://example.org/x", "javascript:alert(1)", None, ""])
def test_a_non_https_url_is_not_a_link(tmp_path, fake_embedder, url):
    conn, uid, _ = ranked_db(tmp_path, fake_embedder, n=1, url=url)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    assert payload["items"][0]["url"] is None


node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
def test_page_logic_prunes_oldest_first_and_merges_newest_wins():
    script = (
        desk.DESK_LOGIC_JS
        + """
const v = {};
for (let i = 0; i < 1000; i++)
  v[String(i)] = {useful: true, at: new Date(Date.UTC(2026, 0, 1, 0, 0, i)).toISOString()};
const kept = prune(v, 28000);
const size = JSON.stringify({v: 1, verdicts: kept}).length;
const keys = Object.keys(kept).map(Number);
const mine = {"1": {useful: true, at: "2026-09-29T10:00:00Z"},
              "2": {useful: false, at: "2026-09-29T09:00:00Z"}};
const theirs = {"2": {useful: true, at: "2026-09-29T11:00:00Z"},
                "3": {useful: true, at: "2026-09-29T08:00:00Z"}};
const merged = merge(mine, theirs);
console.log(JSON.stringify({size, min: Math.min(...keys), max: Math.max(...keys), merged}));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["size"] <= 28000
    assert got["max"] == 999 and got["min"] > 0  # the OLDEST went, the newest stayed
    assert got["merged"] == {
        "1": {"useful": True, "at": "2026-09-29T10:00:00Z"},
        "2": {"useful": True, "at": "2026-09-29T11:00:00Z"},
        "3": {"useful": True, "at": "2026-09-29T08:00:00Z"},
    }


# --- publishing -----------------------------------------------------------------


def test_publish_command_survives_dotenv_and_shlex(tmp_path, monkeypatch):
    """Review focus 1: the value AgentMarkit writes into .env must reach
    subprocess as the argv it means, quotes and all."""
    from dotenv import dotenv_values

    env = tmp_path / ".env"
    env.write_text(
        "ATTEST_DESK_PUBLISH=python3 ~/.hermes/skills/x/private_pages.py register"
        ' --id reading-desk --title "Reading desk" --html ~/.hermes/workspace/d.html\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    argv = desk.publish_argv(dotenv_values(env)["ATTEST_DESK_PUBLISH"])
    assert argv == [
        "python3",
        f"{tmp_path}/.hermes/skills/x/private_pages.py",
        "register",
        "--id",
        "reading-desk",
        "--title",
        "Reading desk",
        "--html",
        f"{tmp_path}/.hermes/workspace/d.html",
    ]


def test_output_path_follows_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    assert desk.desk_output_path() == tmp_path / "hh" / "workspace" / "research-desk" / "desk.html"


@pytest.mark.skipif(node is None, reason="node not installed")
def test_page_counts_only_verdicts_for_papers_it_shows():
    """Seen in the browser: a verdict saved for a paper that has since left
    the list was counted as '1 on this page' on a page showing no verdict."""
    script = (
        desk.DESK_LOGIC_JS
        + """
const items = [{id: 1}, {id: 2}];
const verdicts = {"2": {useful: true, at: "x"}, "999": {useful: false, at: "y"}};
console.log(judgedHere(items, verdicts));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


# --- import before ranking --------------------------------------------------------


def test_feed_ask_imports_page_verdicts_before_ranking(tmp_path, fake_embedder, monkeypatch):
    """Drives feed.ask on a real FastMCP server: the 0.2.0 regression shipped
    because a test checked the dict before the tool, not the tool."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.db import get_db
    from attestation.mcp import _shared, register_all

    db = tmp_path / "t.db"
    conn = seeded_db(db)
    uid = get_user(conn, "researcher")["id"]
    ids = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, ?, 's', ?)",
            (f"paper {i}", f"https://example.org/{i}", f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, fake_embedder.embed_document(f"paper {i}", "s").tobytes()),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    state = page_state_file(
        tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(ids[0]): verdict(False)}}
    )
    monkeypatch.setenv("ATTEST_DB", str(db))
    monkeypatch.setenv("ATTEST_DESK_STATE", str(state))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")
    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)

    server = FastMCP("t")
    register_all(server)

    async def call():
        result = await server.call_tool(
            "feed.ask", {"user": "researcher", "question": "What should I read today?"}
        )
        return result[1] if isinstance(result, tuple) else result

    structured = asyncio.run(call())

    assert structured.get("ok") is True, structured
    row = (
        get_db(db)
        .execute(
            "SELECT useful, source FROM clicks WHERE user_id = ? AND item_id = ?", (uid, ids[0])
        )
        .fetchone()
    )
    assert row is not None and (row["useful"], row["source"]) == (0, "ui")
    # clicked items are excluded from the unread list, so the judged paper is gone
    refs = [r.get("item_id") for r in structured.get("refs", [])]
    assert refs and ids[0] not in refs, structured
