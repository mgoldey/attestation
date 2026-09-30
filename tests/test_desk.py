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


def desk_db(path):
    """seeded_db whose personas have a cutoff early in 2026, so the fixed
    verdict times below postdate it -- the order a real page produces:
    persona first, verdicts after (see users.feedback_since)."""
    conn = seeded_db(path)
    conn.execute("UPDATE users SET feedback_since = '2026-01-01 00:00:00'")
    conn.commit()
    return conn


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
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a, b = add_item(conn, "a"), add_item(conn, "b")
    state = {"v": 1, "verdicts": {str(a): verdict(True), str(b): verdict(False)}}

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=2, unchanged=0, skipped=0)
    assert clicks(conn, uid) == {a: (1, "ui"), b: (0, "ui")}


def test_import_is_idempotent(tmp_path):
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {"v": 1, "verdicts": {str(a): verdict(True)}}

    desk.import_verdicts(conn, uid, state)
    again = desk.import_verdicts(conn, uid, state)

    assert again == desk.Imported(recorded=0, unchanged=1, skipped=0)
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 1


def test_a_changed_mind_overwrites(tmp_path):
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(True)}})

    # A change of mind on the page is stamped when the reader taps, which is
    # after the first verdict was imported.
    later = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5))
    got = desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(False, at=later)}})

    assert got.recorded == 1
    assert clicks(conn, uid) == {a: (0, "ui")}


def test_malformed_and_unknown_entries_are_skipped_not_raised(tmp_path):
    conn = desk_db(tmp_path / "t.db")
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
    conn = desk_db(tmp_path / "t.db")
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
    conn = desk_db(tmp_path / "t.db")
    monkeypatch.delenv("ATTEST_DESK_STATE", raising=False)
    monkeypatch.delenv("ATTEST_DESK_USER", raising=False)
    assert desk.import_pending(conn) is None
    monkeypatch.setenv("ATTEST_DESK_STATE", str(tmp_path / "s.sqlite"))
    assert desk.import_pending(conn) is None  # user still unset


def test_import_pending_imports_for_the_desk_persona(tmp_path, monkeypatch):
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "Researcher")  # case folds, like get_user

    got = desk.import_pending(conn)

    assert got == desk.Imported(recorded=1)
    assert clicks(conn, uid) == {a: (1, "ui")}


def test_import_pending_unknown_persona_imports_nothing(tmp_path, monkeypatch):
    conn = desk_db(tmp_path / "t.db")
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "nobody")

    assert desk.import_pending(conn) is None
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 0
    assert get_user(conn, "nobody") is None, "import must never autocreate a persona"


def test_import_pending_swallows_a_database_error(tmp_path, monkeypatch):
    conn = desk_db(tmp_path / "t.db")
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
    conn = desk_db(tmp_path / "t.db")
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
    # The SVG namespace is a name the DOM API needs, not a request.
    assert "http:" not in html.replace("http://www.w3.org/2000/svg", "")
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
    conn = desk_db(tmp_path / "t.db")
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
    conn = desk_db(db)
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


# --- redesign: age, source marks, views -------------------------------------------


def test_payload_carries_each_papers_publication_time_as_utc(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder, n=2)
    conn.execute("UPDATE items SET published = '2026-09-29T17:06:45' WHERE id = ?", (ids[0],))
    conn.execute("UPDATE items SET published = '2026-09-28 08:00:00' WHERE id = ?", (ids[1],))
    conn.commit()
    got = {i["id"]: i["published"] for i in desk.desk_payload(conn, fake_embedder, uid)["items"]}
    assert got == {ids[0]: "2026-09-29T17:06:45Z", ids[1]: "2026-09-28T08:00:00Z"}


@pytest.mark.skipif(node is None, reason="node not installed")
def test_page_says_how_old_each_paper_is():
    script = (
        desk.DESK_LOGIC_JS
        + """
const now = Date.parse("2026-09-29T18:00:00Z");
console.log(JSON.stringify([
  ago("2026-09-29T17:59:30Z", now), ago("2026-09-29T17:15:00Z", now),
  ago("2026-09-29T09:00:00Z", now), ago("2026-09-26T18:00:00Z", now),
  ago("2026-08-01T00:00:00Z", now), ago("", now), ago("garbage", now)]));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == ["now", "45m", "9h", "3d", "Aug 1", "", ""]


@pytest.mark.skipif(node is None, reason="node not installed")
def test_every_source_gets_a_stable_mark():
    """The page cannot load favicons, so a source is recognised by a coloured
    initial. Same source, same colour, every build."""
    script = (
        desk.DESK_LOGIC_JS
        + """
console.log(JSON.stringify([mark("arXiv cs.LG"), mark("arXiv cs.LG"), mark("Nature"), mark("")]));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    a, b, nature, blank = json.loads(out.stdout)
    assert a == b
    assert a["letter"] == "A" and nature["letter"] == "N" and blank["letter"] == "?"
    assert 0 <= a["hue"] < 360 and a["hue"] != nature["hue"]


def test_views_are_named_for_what_they_show(tmp_path, fake_embedder):
    conn, uid, _ = ranked_db(tmp_path, fake_embedder)
    html = desk.render_desk(conn, fake_embedder, uid)
    assert ">Titles</button>" in html and ">Abstracts</button>" in html
    assert 'id="progress"' in html


def test_the_paper_grid_column_can_shrink(tmp_path, fake_embedder):
    """Seen in the browser: with a one-line meta row, a `1fr` column took the
    row's min-content width and the page scrolled sideways on a phone."""
    conn, uid, _ = ranked_db(tmp_path, fake_embedder)
    assert "grid-template-columns:28px minmax(0,1fr)" in desk.render_desk(conn, fake_embedder, uid)


# --- final review fixes ---------------------------------------------------------


def test_a_newer_chat_verdict_is_not_overwritten_by_an_older_page_verdict(tmp_path):
    """Review finding: the page said Useful at 10:00, chat said not useful at
    10:10, and every later import rewrote the row back to Useful. The last
    verdict must win, whichever surface gave it."""
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    page = {"v": 1, "verdicts": {str(a): verdict(True, at="2026-09-29T10:00:00Z")}}
    desk.import_verdicts(conn, uid, page)
    record_click(conn, uid, a, False, source="agent")
    conn.execute("UPDATE clicks SET clicked_at = '2026-09-29 10:10:00' WHERE item_id = ?", (a,))
    conn.commit()

    got = desk.import_verdicts(conn, uid, page)

    assert got == desk.Imported(recorded=0, unchanged=1, skipped=0)
    assert clicks(conn, uid) == {a: (0, "agent")}


def test_a_newer_page_verdict_still_overwrites_an_older_chat_verdict(tmp_path):
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    record_click(conn, uid, a, False, source="agent")
    conn.execute("UPDATE clicks SET clicked_at = '2026-09-29 10:00:00' WHERE item_id = ?", (a,))
    conn.commit()

    page = {"v": 1, "verdicts": {str(a): verdict(True, at="2026-09-29T10:10:00Z")}}
    got = desk.import_verdicts(conn, uid, page)

    assert got.recorded == 1
    assert clicks(conn, uid) == {a: (1, "ui")}


@pytest.mark.skipif(node is None, reason="node not installed")
def test_a_failed_save_points_at_the_hosts_retry_not_at_tapping_again():
    """Review finding: the page host keeps a failed save pending and refuses
    every new one until its own Retry runs, so "tap again" could never work."""
    script = (
        desk.DESK_LOGIC_JS
        + """
console.log(JSON.stringify([
  failureText({ok: false, error: "This page changed elsewhere."}),
  failureText({ok: false}), failureText(undefined),
  failureText(new Error("The page did not answer."))]));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    msgs = json.loads(out.stdout)
    assert msgs[0].startswith("This page changed elsewhere.")
    assert msgs[3].startswith("The page did not answer.")
    assert all("Retry at the top" in m for m in msgs)
    assert not any("again" in m.lower() for m in msgs)


# --- code review fixes ------------------------------------------------------------


def test_an_id_too_large_for_sqlite_is_skipped_not_raised(tmp_path):
    """Code review: an all-digit key past int64 raised OverflowError out of
    the SELECT, and import_pending runs in front of every ranking."""
    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {"v": 1, "verdicts": {"9" * 20: verdict(True), str(a): verdict(True)}}

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=1, unchanged=0, skipped=1)


def test_a_state_value_that_is_not_text_reads_as_no_state(tmp_path):
    """Code review: a NULL or numeric `value` made json.loads raise TypeError."""
    for value in (None, 7):
        path = tmp_path / f"s-{value}.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, revision INTEGER, value)")
        conn.execute("INSERT INTO state VALUES (1, 1, ?)", (value,))
        conn.commit()
        conn.close()
        assert desk.read_state(path) == {}


def test_import_pending_survives_a_hostile_state_file(tmp_path, monkeypatch):
    conn = desk_db(tmp_path / "t.db")
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {"9" * 20: verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")
    assert desk.import_pending(conn) == desk.Imported(skipped=1)


def test_a_persona_reset_is_not_undone_by_the_next_import(tmp_path):
    """Code review: feed.persona_reset deleted the clicks and the next ranking
    re-imported every verdict still in the page's state, restoring them."""
    from attestation.personas import purge_feedback

    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    page = {"v": 1, "verdicts": {str(a): verdict(True, at="2026-09-29T10:00:00Z")}}
    desk.import_verdicts(conn, uid, page)
    purge_feedback(conn, uid)
    conn.commit()

    got = desk.import_verdicts(conn, uid, page)

    assert got == desk.Imported(recorded=0, unchanged=0, skipped=1)
    assert clicks(conn, uid) == {}


def test_a_verdict_given_after_a_reset_still_imports(tmp_path):
    from attestation.personas import purge_feedback

    conn = desk_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    purge_feedback(conn, uid)
    conn.commit()
    later = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5))

    got = desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(True, at=later)}})

    assert got.recorded == 1


def test_a_recreated_persona_does_not_inherit_the_old_pages_verdicts(tmp_path):
    from attestation.personas import purge_feedback
    from attestation.rank import create_user

    conn = desk_db(tmp_path / "t.db")
    old = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    page = {"v": 1, "verdicts": {str(a): verdict(True, at="2026-09-29T10:00:00Z")}}
    purge_feedback(conn, old, delete_user=True)
    conn.commit()
    create_user(conn, "researcher", "machine learning")
    new = get_user(conn, "researcher")["id"]

    assert desk.import_verdicts(conn, new, page).recorded == 0


def test_migration_012_keeps_existing_personas_page_verdicts(tmp_path):
    """An upgraded database has no reset times on record, so its existing
    personas import every page verdict, exactly as before the migration."""
    from attestation.db import SCHEMA_VERSION, get_db

    db = tmp_path / "t.db"
    conn = seeded_db(db)
    conn.execute("ALTER TABLE users DROP COLUMN feedback_since")
    conn.execute("PRAGMA user_version = 11")
    conn.commit()
    conn.close()

    conn = get_db(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION >= 12
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    assert "feedback_since" in cols
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    page = {"v": 1, "verdicts": {str(a): verdict(True, at="2020-01-01T00:00:00Z")}}
    assert desk.import_verdicts(conn, uid, page).recorded == 1
