"""The reader's bibliography: fold, saves, tombstones, stable keys, render, cite.save."""

import pytest

from attestation import bibliography
from attestation.db import get_db
from attestation.library import ReferenceRecord, upsert
from attestation.mcp import citation
from attestation.personas import merge, purge_feedback
from attestation.rank import create_user, record_click


def _item(conn, n, *, doi=None, arxiv=None, title=None):
    conn.execute("INSERT OR IGNORE INTO feeds(id, url, title) VALUES (1, 'http://f', 'f')")
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash, published,"
        " doi, arxiv_id) VALUES (1, ?, ?, ?, 's', ?, '2026-09-01', ?, ?)",
        (f"g{n}", title or f"Paper number {n}", f"https://x/{n}", f"h{n}", doi, arxiv),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _engage(conn, user_id, item_id, kind):
    conn.execute(
        "INSERT OR IGNORE INTO engagement(user_id, item_id, kind) VALUES (?, ?, ?)",
        (user_id, item_id, kind),
    )


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "t.db"))
    c = get_db(tmp_path / "t.db")
    yield c
    c.close()


def _live(conn, user_id):
    return {(e["title"], e["reason"]) for e in bibliography.entries(conn, user_id)}


def test_fold_adds_human_engagement_with_the_strongest_reason(conn):
    u = create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    b = _item(conn, 2, doi="10.1/b")
    c = _item(conn, 3, arxiv="2609.00003")
    record_click(conn, u, a, True, source="ui")
    _engage(conn, u, a, "read")  # rated AND read: useful wins
    _engage(conn, u, b, "explain")
    _engage(conn, u, c, "read")
    assert bibliography.fold(conn, u) == 3
    assert _live(conn, u) == {
        ("Paper number 1", "useful"),
        ("Paper number 2", "explained"),
        ("Paper number 3", "read"),
    }
    assert bibliography.fold(conn, u) == 0  # idempotent


def test_fold_ignores_machine_clicks_and_items_without_ids(conn):
    u = create_user(conn, "reader", "x")
    sim = _item(conn, 1, arxiv="2609.00001")
    boot = _item(conn, 2, arxiv="2609.00002")
    hn = _item(conn, 3)  # no DOI, no arXiv id: not a paper
    record_click(conn, u, sim, True, source="simulated")
    record_click(conn, u, boot, True, source="bootstrap")
    record_click(conn, u, hn, True, source="agent")
    _engage(conn, u, hn, "read")
    assert bibliography.fold(conn, u) == 0
    assert bibliography.entries(conn, u) == []


def test_removal_sticks_against_engagement_but_not_against_a_save(conn):
    u = create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    _engage(conn, u, a, "read")
    bibliography.fold(conn, u)
    ref = bibliography.reference_for_item(conn, a)
    bibliography.remove(conn, u, ref)
    _engage(conn, u, a, "explain")  # a new implicit signal after removal
    bibliography.fold(conn, u)
    assert bibliography.entries(conn, u) == []
    entry = bibliography.save(conn, u, ref)
    assert entry["reason"] == "saved"
    assert _live(conn, u) == {("Paper number 1", "saved")}


def test_not_useful_tombstones_implicit_entries_but_never_a_save(conn):
    u = create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    b = _item(conn, 2, arxiv="2609.00002")
    _engage(conn, u, a, "read")
    _engage(conn, u, b, "read")
    bibliography.fold(conn, u)
    bibliography.save(conn, u, bibliography.reference_for_item(conn, b))
    record_click(conn, u, a, False, source="agent")
    record_click(conn, u, b, False, source="agent")
    bibliography.fold(conn, u)
    assert _live(conn, u) == {("Paper number 2", "saved")}


def test_a_cite_key_survives_removing_its_twin(conn):
    u = create_user(conn, "reader", "x")
    # Same family/year/first word: bibtex_key collides, so the second is suffixed.
    a = _item(conn, 1, arxiv="2609.00001", title="Equivariant networks A")
    b = _item(conn, 2, arxiv="2609.00002", title="Equivariant networks B")
    ka = bibliography.save(conn, u, bibliography.reference_for_item(conn, a))["key"]
    kb = bibliography.save(conn, u, bibliography.reference_for_item(conn, b))["key"]
    assert ka != kb and kb.startswith(ka)
    bibliography.remove(conn, u, bibliography.reference_for_item(conn, a))
    assert [e["key"] for e in bibliography.entries(conn, u)] == [kb]
    again = bibliography.save(conn, u, bibliography.reference_for_item(conn, a))
    assert again["key"] == ka  # a tombstone keeps its key


def test_save_takes_a_library_reference_that_is_not_a_feed_item(conn):
    u = create_user(conn, "reader", "x")
    rid, _ = upsert(
        conn,
        ReferenceRecord(
            source="research:pubmed",
            source_key="1",
            title="A paper only research found",
            doi="10.9/r",
            year=2026,
            authors=["Curie, Marie"],
        ),
    )
    entry = bibliography.save(conn, u, rid)
    assert entry["key"] == "curie2026paper"
    assert "@" in bibliography.render(conn, u)


def test_render_is_deterministic_and_says_it_is_generated(conn):
    u = create_user(conn, "reader", "x")
    for n in range(3):
        _engage(conn, u, _item(conn, n, arxiv=f"2609.0000{n}"), "read")
    bibliography.fold(conn, u)
    text = bibliography.render(conn, u)
    assert text == bibliography.render(conn, u)
    assert text.startswith("%") and "overwritten" in text.splitlines()[0] + text.splitlines()[1]
    assert text.count("@misc{") == 3


def test_write_all_is_inert_without_ATTEST_BIB_OUT(conn, monkeypatch, tmp_path):
    monkeypatch.delenv("ATTEST_BIB_OUT", raising=False)
    u = create_user(conn, "reader", "x")
    _engage(conn, u, _item(conn, 1, arxiv="2609.00001"), "read")
    assert bibliography.write_all(conn) == {}
    out = tmp_path / "bibout"
    monkeypatch.setenv("ATTEST_BIB_OUT", str(out))
    assert bibliography.write_all(conn) == {"reader": 1}
    assert (out / "reader.bib").read_text().count("@misc{") == 1


def test_delete_drops_entries_reset_keeps_them_merge_moves_them(conn):
    keep = create_user(conn, "keep", "x")
    drop = create_user(conn, "drop", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    b = _item(conn, 2, arxiv="2609.00002")
    bibliography.save(conn, keep, bibliography.reference_for_item(conn, a))
    bibliography.save(conn, drop, bibliography.reference_for_item(conn, a))
    bibliography.save(conn, drop, bibliography.reference_for_item(conn, b))
    purge_feedback(conn, keep)  # reset
    assert len(bibliography.entries(conn, keep)) == 1
    merge(conn, into="keep", drop=["drop"])
    conn.commit()
    assert len(bibliography.entries(conn, keep)) == 2
    other = create_user(conn, "other", "x")
    bibliography.save(conn, other, bibliography.reference_for_item(conn, a))
    purge_feedback(conn, other, delete_user=True)
    conn.commit()
    assert (
        conn.execute("SELECT count(*) FROM bibliography WHERE user_id = ?", (other,)).fetchone()[0]
        == 0
    )


def test_cite_save_tool_saves_renders_and_refuses(conn, monkeypatch, tmp_path):
    out = tmp_path / "bibout"
    monkeypatch.setenv("ATTEST_BIB_OUT", str(out))
    create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    conn.commit()
    got = citation._save("reader", str(a))
    assert got["ok"] is True
    assert got["entry"]["reason"] == "saved" and got["entry"]["bibtex"].startswith("@")
    assert got["entries"] == 1 and got["file"] == "reader.bib"
    assert "@misc{" in (out / "reader.bib").read_text()
    gone = citation._save("reader", str(a), remove=True)
    assert gone["ok"] is True and gone["removed"] is True and gone["entries"] == 0
    assert citation._save("reader", "10.0/nope")["ok"] is False
    assert citation._save("nobody", str(a))["ok"] is False


def test_a_useful_verdict_after_a_removal_revives_it_an_older_one_does_not(conn):
    u = create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    record_click(conn, u, a, True, source="ui")
    bibliography.fold(conn, u)
    ref = bibliography.reference_for_item(conn, a)
    bibliography.remove(conn, u, ref)
    bibliography.fold(conn, u)  # the old verdict predates the removal
    assert bibliography.entries(conn, u) == []
    conn.execute(
        "UPDATE clicks SET clicked_at = datetime('now', '+1 minute') WHERE user_id = ?", (u,)
    )
    bibliography.fold(conn, u)  # the reader changed their mind
    assert _live(conn, u) == {("Paper number 1", "useful")}


@pytest.mark.parametrize(
    "question",
    ["save this", "add it to my bib", "I'll cite this one", "put that in my bibliography"],
)
def test_feed_ask_routes_a_save_to_the_bibliography(question):
    from attestation.mcp.routing import route_feed

    assert route_feed(question).tool == "cite.save"


@pytest.mark.parametrize(
    "question",
    ["add https://example.org/rss", "subscribe me to Nature", "save me the trouble"],
)
def test_save_phrases_do_not_steal_feed_or_idiom_questions(question):
    from attestation.mcp.routing import route_feed

    assert route_feed(question).tool != "cite.save"


def test_feed_ask_saves_and_removes_an_item_on_the_deployed_surface(conn, monkeypatch, tmp_path):
    """The hosted feed session sees feed.ask only: a save has to complete there."""
    from attestation.mcp.ask import _feed_ask

    monkeypatch.setenv("ATTEST_BIB_OUT", str(tmp_path / "out"))
    create_user(conn, "reader", "x")
    a = _item(conn, 1, arxiv="2609.00001")
    conn.commit()
    got = _feed_ask("reader", "save this to my bib", item_id=a)
    assert got["ok"] is True and got["tool_used"] == "cite.save"
    assert "saved to the bibliography" in got["answer"]
    gone = _feed_ask("reader", "take it out of my bib", item_id=a)
    assert gone["ok"] is True and "removed from the bibliography" in gone["answer"]
    assert _feed_ask("reader", "save this")["ok"] is False  # which item?


def test_asking_about_the_bib_is_not_a_save():
    from attestation.mcp.routing import route_feed

    assert route_feed("is this in my bib?").tool != "cite.save"
