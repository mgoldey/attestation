"""The library dashboard: one JSON file a host page reads (schema 1).

The schema is a CONTRACT: AgentMarkit's Files tab is built against it, so
`validate` below is the contract written as code (no jsonschema dependency), and
every produced document, including the empty one, must pass it. Fields may be
added; a test here that starts failing because a field was renamed, removed or
retyped is the test doing its job.
"""

import json
import os
import re
import stat
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from attestation import bibliography, dashboard, library
from attestation.cli import main
from attestation.db import get_db
from attestation.library import ReferenceRecord
from attestation.rank import create_user

NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)  # a Wednesday, ISO week 2026-W40
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# --- the contract, as code ---------------------------------------------------------------------


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _opt_str(v, cap):
    return v is None or (isinstance(v, str) and 0 < len(v) <= cap)


def validate(doc: dict) -> None:
    """Raise AssertionError unless `doc` satisfies the v1 contract."""
    for key in (
        "schema",
        "generated_at",
        "persona",
        "totals",
        "last_refresh",
        "ingested",
        "by_source",
        "failures",
        "references",
        "references_truncated",
        "references_limit",
    ):
        assert key in doc, f"missing {key}"
    assert doc["schema"] == 1
    assert ISO.match(doc["generated_at"]) and isinstance(doc["persona"], str)
    for key in ("items", "references", "tagged", "embedded", "with_fulltext", "saved"):
        assert _is_int(doc["totals"][key]) and doc["totals"][key] >= 0, key
    assert doc["last_refresh"] is None or ISO.match(doc["last_refresh"])
    weeks = doc["ingested"]
    assert len(weeks) == 26
    for w in weeks:
        assert re.match(r"^\d{4}-W\d{2}$", w["week"]) and ISO.match(w["start"] + "T00:00:00Z")
        assert date.fromisoformat(w["start"]).weekday() == 0, "weeks start on Monday"
        assert _is_int(w["items"]) and _is_int(w["references"]) and w["items"] >= 0
    starts = [w["start"] for w in weeks]
    assert starts == sorted(starts) and len(set(starts)) == 26, "oldest first, no gaps"
    assert all(
        date.fromisoformat(b) - date.fromisoformat(a) == timedelta(weeks=1)
        for a, b in zip(starts, starts[1:], strict=False)
    )
    assert len(doc["by_source"]) <= 20
    for s in doc["by_source"]:
        assert isinstance(s["source"], str) and s["source"]
        assert s["kind"] in ("feed", "library", "zotero", "bib") and _is_int(s["items"])
    counts = [s["items"] for s in doc["by_source"]]
    assert counts == sorted(counts, reverse=True)
    f = doc["failures"]
    assert set(f) >= {"fetch", "tag", "embed", "since"}
    for key in ("fetch", "tag", "embed"):
        assert f[key] is None or _is_int(f[key])
    assert f["since"] is None or ISO.match(f["since"])
    refs = doc["references"]
    assert isinstance(doc["references_truncated"], bool) and _is_int(doc["references_limit"])
    assert len(refs) <= doc["references_limit"]
    added = [r["added"] for r in refs if r["added"]]
    assert added == sorted(added, reverse=True), "newest added first"
    for r in refs:
        assert isinstance(r["key"], str) and r["key"]
        assert isinstance(r["title"], str) and len(r["title"]) <= 300
        assert isinstance(r["authors"], list) and len(r["authors"]) <= 8
        assert all(isinstance(a, str) and 0 < len(a) <= 120 for a in r["authors"])
        assert r["year"] is None or _is_int(r["year"])
        assert _opt_str(r["venue"], 200) and _opt_str(r["doi"], 200) and _opt_str(r["arxiv"], 200)
        assert r["url"] is None or (
            isinstance(r["url"], str)
            and r["url"].startswith(("http://", "https://"))
            and len(r["url"]) <= 500
        )
        assert isinstance(r["tags"], list) and len(r["tags"]) <= 12
        assert all(isinstance(t, str) and 0 < len(t) <= 60 for t in r["tags"])
        assert isinstance(r["sources"], list) and len(r["sources"]) <= 5
        assert r["added"] is None or ISO.match(r["added"])
        assert isinstance(r["saved"], bool) and isinstance(r["fulltext"], bool)
    n = doc["totals"]["references"]
    assert len(refs) == min(n, doc["references_limit"])
    assert doc["references_truncated"] == (n > doc["references_limit"])
    text = json.dumps(doc, ensure_ascii=False)
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f]", text), "no control characters"


# --- fixtures ----------------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "t.db"))
    c = get_db(tmp_path / "t.db")
    yield c
    c.close()


def vec(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(256).astype(np.float32)
    return (v / np.linalg.norm(v)).tobytes()


def add_item(conn, n, *, published, feed=1, tags=(), embed=False, title=None, url=None):
    conn.execute(
        "INSERT OR IGNORE INTO feeds(id, url, title) VALUES (?, ?, ?)",
        (feed, f"http://feed{feed}", f"Feed {feed}"),
    )
    cur = conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, published, content_hash)"
        " VALUES (?, ?, ?, ?, 's', ?, ?)",
        (feed, f"g{n}", title or f"Item {n}", url or f"https://x/{n}", published, f"h{n}"),
    )
    item_id = cur.lastrowid
    for t in tags:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))
    if tags:
        conn.execute(
            "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'm')",
            (item_id,),
        )
    if embed:
        conn.execute("INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (item_id, vec(n)))
    conn.commit()
    return item_id


def add_ref(conn, source="bibtex:refs.bib", key="k", **kw):
    kw.setdefault("title", f"Paper {key}")
    kw.setdefault("doi", f"10.1/{key}")
    rid, _ = library.upsert(conn, ReferenceRecord(source=source, source_key=key, bib_key=key, **kw))
    return rid


def set_first_seen(conn, rid, when):
    conn.execute('UPDATE "references" SET first_seen = ? WHERE id = ?', (when, rid))
    conn.commit()


def build(conn, **kw):
    kw.setdefault("now", NOW)
    return dashboard.build(conn, **kw)


# --- the empty state matters: it is what a fresh Research Desk shows ---------------------------


def test_an_empty_database_is_a_valid_file_of_zeros_and_empty_lists(conn):
    doc = build(conn)
    validate(doc)
    assert doc["totals"] == {
        "items": 0,
        "references": 0,
        "tagged": 0,
        "embedded": 0,
        "with_fulltext": 0,
        "saved": 0,
        "items_tagged": 0,
        "items_embedded": 0,
        "references_tagged": 0,
        "references_embedded": 0,
    }
    assert doc["references"] == [] and doc["by_source"] == []
    assert doc["last_refresh"] is None and doc["references_truncated"] is False
    assert all(w["items"] == 0 and w["references"] == 0 for w in doc["ingested"])
    assert doc["failures"] == {"fetch": None, "tag": None, "embed": None, "since": None}
    assert doc["persona"] == "owner" and doc["references_limit"] == 5000


# --- totals ------------------------------------------------------------------------------------


def test_totals_count_what_the_database_holds(conn):
    owner = create_user(conn, "owner", "graph networks")
    add_item(conn, 1, published="2026-09-29T08:00:00", tags=["gnn", "chem"], embed=True)
    add_item(conn, 2, published="2026-09-29T09:00:00", tags=["gnn"])
    add_item(conn, 3, published="2026-09-28T09:00:00")
    r1 = add_ref(conn, key="a", title="Alpha")
    r2 = add_ref(conn, key="b", title="Beta", source="zotero")
    r3 = add_ref(conn, key="c", title="Gamma")
    conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, 'molecules')", (r1,))
    conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, 'gnn')", (r1,))
    conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, 'gnn')", (r2,))
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)", (r1, vec(101)))
    conn.execute(
        "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at)"
        " VALUES (?, 'full text', 'arxiv-pdf', '2026-09-01')",
        (r1,),
    )
    conn.execute(  # tried and found nothing: NOT full text
        "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at)"
        " VALUES (?, NULL, 'none', '2026-09-01')",
        (r2,),
    )
    conn.commit()
    bibliography.save(conn, owner, r1)
    bibliography.save(conn, owner, r3)
    bibliography.remove(conn, owner, r3)  # removed entries do not count
    doc = build(conn)
    validate(doc)
    t = doc["totals"]
    assert (t["items"], t["references"]) == (3, 3)
    assert (t["items_tagged"], t["references_tagged"], t["tagged"]) == (2, 2, 4)
    assert (t["items_embedded"], t["references_embedded"], t["embedded"]) == (1, 1, 2)
    assert (t["with_fulltext"], t["saved"]) == (1, 1)
    assert doc["backlog"] == {
        "items_untagged": 1,
        "references_untagged": 1,
        "references_unembedded": 2,
    }


def test_saved_marks_the_named_persona_only(conn):
    a, b = create_user(conn, "owner", "x"), create_user(conn, "other", "y")
    r1, r2 = add_ref(conn, key="a"), add_ref(conn, key="b")
    bibliography.save(conn, a, r1)
    bibliography.save(conn, b, r2)
    owner = build(conn)
    other = build(conn, persona="other")
    assert {r["key"]: r["saved"] for r in owner["references"]} == {"a": True, "b": False}
    assert {r["key"]: r["saved"] for r in other["references"]} == {"a": False, "b": True}
    assert owner["totals"]["saved"] == other["totals"]["saved"] == 1
    ghost = build(conn, persona="nobody-yet")
    validate(ghost)
    assert ghost["totals"]["saved"] == 0 and not any(r["saved"] for r in ghost["references"])
    assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 2, "never autocreates"


# --- weeks -------------------------------------------------------------------------------------


def test_the_last_26_iso_weeks_oldest_first_are_zero_filled(conn):
    add_item(conn, 1, published="2026-09-28T01:00:00")  # Monday of W40
    add_item(conn, 2, published="2026-09-30T11:00:00")  # W40
    add_item(conn, 3, published="2026-09-21T23:59:59")  # Sunday of W39
    add_item(conn, 4, published="2026-04-06T00:00:00")  # first Monday of the window: W15
    add_item(conn, 5, published="2026-04-05T23:00:00")  # a week too old: dropped
    add_item(conn, 6, published="2026-10-05T00:00:00")  # in the future: dropped
    doc = build(conn)
    validate(doc)
    weeks = doc["ingested"]
    assert weeks[0]["week"] == "2026-W15" and weeks[0]["start"] == "2026-04-06"
    assert weeks[-1]["week"] == "2026-W40" and weeks[-1]["start"] == "2026-09-28"
    by_week = {w["week"]: w["items"] for w in weeks}
    assert (by_week["2026-W40"], by_week["2026-W39"], by_week["2026-W15"]) == (2, 1, 1)
    assert sum(by_week.values()) == 4
    assert sum(1 for w in weeks if w["items"] == 0) == 23, "empty weeks are present as zeros"


def test_a_year_boundary_names_weeks_by_their_iso_year(conn):
    doc = build(conn, now=datetime(2027, 1, 2, tzinfo=UTC))  # Saturday, ISO week 2026-W53
    weeks = [w["week"] for w in doc["ingested"]]
    assert weeks[-1] == "2026-W53" and "2026-W52" in weeks and "2025-W..." not in weeks
    validate(doc)


def test_references_are_bucketed_by_when_the_library_first_saw_them(conn):
    r1, r2 = add_ref(conn, key="a"), add_ref(conn, key="b")
    set_first_seen(conn, r1, "2026-09-29T10:00:00+00:00")
    set_first_seen(conn, r2, "2026-09-22T10:00:00+00:00")
    by_week = {w["week"]: w["references"] for w in build(conn)["ingested"]}
    assert (by_week["2026-W40"], by_week["2026-W39"]) == (1, 1)


def test_all_three_stored_timestamp_shapes_are_read(conn):
    add_item(conn, 1, published="2026-09-29 08:00:00")  # datetime('now')
    add_item(conn, 2, published="2026-09-29T08:00:00")  # a feed's own date
    add_item(conn, 3, published="2026-09-29")  # a bare date
    add_item(conn, 4, published="garbage")  # unreadable: skipped, not fatal
    doc = build(conn)
    validate(doc)
    assert {w["week"]: w["items"] for w in doc["ingested"]}["2026-W40"] == 3


# --- last refresh and failures: never invented -------------------------------------------------


def test_last_refresh_is_the_latest_feed_fetch_and_null_when_none(conn):
    assert build(conn)["last_refresh"] is None
    conn.execute("INSERT INTO feeds(id, url, last_fetched) VALUES (1, 'u1', '2026-09-30 10:15:00')")
    conn.execute("INSERT INTO feeds(id, url, last_fetched) VALUES (2, 'u2', '2026-09-29 08:00:00')")
    conn.execute("INSERT INTO feeds(id, url) VALUES (3, 'u3')")
    conn.commit()
    doc = build(conn)
    assert doc["last_refresh"] == "2026-09-30T10:15:00Z"
    assert doc["failures"] == {"fetch": None, "tag": None, "embed": None, "since": None}


# --- by_source ---------------------------------------------------------------------------------


def test_by_source_names_feeds_bibs_zotero_and_library_enrichers_top_20(conn):
    for n in range(25):
        add_item(conn, n, published="2026-09-29T08:00:00", feed=n + 1)
    for _ in range(3):  # feed 1 gets more items than the rest
        add_item(conn, 100 + _, published="2026-09-29T08:00:00", feed=1)
    add_ref(conn, key="a", source="bibtex:library.bib")
    add_ref(conn, key="b", source="bibtex:library.bib")
    add_ref(conn, key="c", source="zotero")
    add_ref(conn, key="d", source="crossref")
    doc = build(conn)
    validate(doc)
    rows = doc["by_source"]
    assert len(rows) == 20
    assert rows[0] == {"source": "Feed 1", "kind": "feed", "items": 4}
    bib = [r for r in rows if r["kind"] == "bib"]
    assert bib == [{"source": "bibtex:library.bib", "kind": "bib", "items": 2}]
    assert {r["kind"] for r in rows} <= {"feed", "bib", "zotero", "library"}


def test_by_source_kinds(conn):
    add_ref(conn, key="a", source="bibtex:library.bib")
    add_ref(conn, key="b", source="zotero")
    add_ref(conn, key="c", source="crossref")
    add_ref(conn, key="d", source="research:arxiv")
    add_item(conn, 1, published="2026-09-29T08:00:00")
    kinds = {r["source"]: r["kind"] for r in build(conn)["by_source"]}
    assert kinds == {
        "bibtex:library.bib": "bib",
        "zotero": "zotero",
        "crossref": "library",
        "research:arxiv": "library",
        "Feed 1": "feed",
    }


# --- references --------------------------------------------------------------------------------


def test_references_are_newest_first_with_a_stable_tiebreak_and_truncate(conn):
    ids = [add_ref(conn, key=f"k{i}") for i in range(7)]
    for i, rid in enumerate(ids):
        set_first_seen(conn, rid, f"2026-09-{10 + i:02d}T00:00:00+00:00")
    set_first_seen(conn, ids[0], "2026-09-16T00:00:00+00:00")  # ties with ids[6]
    doc = build(conn, limit=4)
    validate(doc)
    assert [r["key"] for r in doc["references"]] == ["k6", "k0", "k5", "k4"]
    assert doc["references_truncated"] is True and doc["references_limit"] == 4
    full = build(conn, limit=7)
    assert full["references_truncated"] is False and len(full["references"]) == 7
    zero = build(conn, limit=0)
    validate(zero)
    assert zero["references"] == [] and zero["references_truncated"] is True


def test_a_reference_carries_its_fields_tags_sources_and_flags(conn):
    owner = create_user(conn, "owner", "x")
    rid = add_ref(
        conn,
        key="schnet",
        title="SchNet: a continuous-filter convolutional neural network",
        authors=[f"Author {i}, A." for i in range(12)],
        year=2017,
        venue="NeurIPS",
        arxiv_id="1706.08566",
        url="https://arxiv.org/abs/1706.08566",
    )
    add_ref(conn, key="schnet", source="zotero", title="SchNet", doi="10.1/schnet")
    for t in [f"tag-{i:02d}" for i in range(15)]:
        conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, ?)", (rid, t))
    conn.execute(
        "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at)"
        " VALUES (?, 't', 'arxiv-pdf', '2026-09-01')",
        (rid,),
    )
    conn.commit()
    bibliography.save(conn, owner, rid)
    (ref,) = build(conn)["references"]
    assert ref["key"] == "schnet" and ref["year"] == 2017 and ref["venue"] == "NeurIPS"
    assert ref["doi"] == "10.1/schnet" and ref["arxiv"] == "1706.08566"
    assert ref["url"] == "https://arxiv.org/abs/1706.08566"
    assert len(ref["authors"]) == 8 and ref["authors"][0] == "Author 0, A."
    assert ref["tags"] == [f"tag-{i:02d}" for i in range(12)], "sorted, capped at 12"
    assert ref["sources"] == ["bibtex:refs.bib", "zotero"]
    assert ref["saved"] is True and ref["fulltext"] is True and ISO.match(ref["added"])


def test_a_bare_reference_has_nulls_and_empty_lists_not_missing_keys(conn):
    add_ref(conn, key="bare", title="Only a title", doi=None, arxiv_id=None)
    (ref,) = build(conn)["references"]
    assert ref["year"] is None and ref["venue"] is None and ref["doi"] is None
    assert ref["arxiv"] is None and ref["url"] is None
    assert ref["authors"] == [] and ref["tags"] == [] and ref["saved"] is False


# --- hostile text ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox",
        "file:///etc/passwd",
        "//evil.example/x",
        "/relative/path",
        "https://user:pw@evil.example/",
        "https://",
        "http://ex ample.com/",
        "https://example.com/\x00",
        "https://example.com:notaport/",
        "x" * 600,
        "https://example.com/" + "a" * 600,
        "",
        "   ",
    ],
)
def test_a_url_that_is_not_plain_http_is_dropped_to_null(url):
    assert dashboard.safe_url(url) is None


def test_http_urls_pass_untouched():
    for url in ("http://a.example/p?q=1#f", "https://arxiv.org/abs/2609.00001"):
        assert dashboard.safe_url(url) == url


def test_hostile_titles_stay_literal_text_and_everything_is_capped(conn):
    add_ref(
        conn,
        key="x1",
        title="<script>alert('x')</script> <img src=x onerror=1>",
        authors=["<b>Eve</b>", "A" * 500, "tab\tand\nnewline", "\x00\x1b[31mred", ""],
        venue="V" * 900,
        url="javascript:alert(1)",
    )
    add_ref(conn, key="x2", title="T" * 10_000, doi="10.1/x2")
    add_ref(conn, key="x3", title="control\x00chars\x07here‮and​format", doi="10.1/x3")
    doc = build(conn)
    validate(doc)
    by_key = {r["key"]: r for r in doc["references"]}
    first = by_key["x1"]
    assert first["title"] == "<script>alert('x')</script> <img src=x onerror=1>"
    assert first["url"] is None
    assert first["authors"][0] == "<b>Eve</b>"
    assert len(first["authors"][1]) == 120 and first["authors"][1].endswith("…")
    assert first["authors"][2] == "tab and newline"
    assert first["authors"][3] == "[31mred" or first["authors"][3].startswith("[31m")
    assert "" not in first["authors"]
    assert len(first["venue"]) == 200
    assert len(by_key["x2"]["title"]) == 300 and by_key["x2"]["title"].endswith("…")
    assert by_key["x3"]["title"] == "control chars here and format"


def test_hostile_tags_sources_and_feed_names_are_cleaned(conn):
    rid = add_ref(conn, key="x", source="bibtex:evil\x00name\n.bib")
    conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, ?)", (rid, "T" * 200))
    conn.execute("INSERT INTO reference_tags(reference_id, tag) VALUES (?, ?)", (rid, "\x01\x02"))
    conn.execute(
        "INSERT INTO feeds(id, url, title) VALUES (1, 'http://f', ?)", ("Feed\x07 <b>x</b>",)
    )
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, published, content_hash)"
        " VALUES (1, 'g', 't', 'http://x', 's', '2026-09-29T00:00:00', 'h')"
    )
    conn.commit()
    doc = build(conn)
    validate(doc)
    (ref,) = doc["references"]
    assert [len(t) for t in ref["tags"]] == [60] and ref["sources"] == ["bibtex:evil name .bib"]
    assert doc["by_source"][0]["source"] in ("Feed <b>x</b>", "bibtex:evil name .bib")


def test_clean_collapses_whitespace_and_leaves_ordinary_unicode(conn):
    assert dashboard.clean("  Schütt, K.\t \n— π  ", 100) == "Schütt, K. — π"
    assert dashboard.clean(None, 10) == "" and dashboard.clean(12345, 10) == "12345"


def test_a_malformed_authors_column_does_not_break_the_export(conn):
    rid = add_ref(conn, key="bad")
    conn.execute('UPDATE "references" SET authors = ? WHERE id = ?', ("{not json", rid))
    conn.commit()
    (ref,) = build(conn)["references"]
    assert ref["authors"] == []


# --- determinism, atomic write, size -----------------------------------------------------------


def seeded(conn):
    owner = create_user(conn, "owner", "x")
    for i in range(12):
        add_item(conn, i, published=f"2026-09-{1 + i:02d}T08:00:00", feed=1 + i % 3, tags=["t"])
        rid = add_ref(conn, key=f"r{i}", year=2000 + i, authors=["A, B."])
        set_first_seen(conn, rid, f"2026-09-{1 + i:02d}T00:00:00+00:00")
    bibliography.save(conn, owner, 1)


def test_the_same_database_gives_the_same_bytes_apart_from_generated_at(conn):
    seeded(conn)
    first, second = build(conn), build(conn, now=NOW + timedelta(hours=3))
    assert first["generated_at"] != second["generated_at"]
    first["generated_at"] = second["generated_at"] = "X"
    assert dashboard.render(first) == dashboard.render(second)
    text = dashboard.render(first)
    assert json.loads(text) == first and text.endswith("\n")
    top_level = list(json.loads(text))
    assert top_level == sorted(top_level), "sorted keys"


def test_write_is_atomic_private_and_idempotent(tmp_path, conn):
    seeded(conn)
    out = tmp_path / "deep" / "library" / "library.json"
    dashboard.write(out, build(conn))
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert not [p for p in out.parent.iterdir() if p.name != "library.json"], "no temp left"
    once = out.read_bytes()
    dashboard.write(out, build(conn))
    twice = out.read_bytes()
    assert json.loads(once)["totals"] == json.loads(twice)["totals"]
    validate(json.loads(twice))


def test_a_failed_write_leaves_the_old_file_untouched_and_no_temp_behind(
    tmp_path, conn, monkeypatch
):
    seeded(conn)
    out = tmp_path / "library.json"
    dashboard.write(out, build(conn))
    before = out.read_bytes()

    def explode(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        dashboard.write(out, build(conn, limit=1))
    monkeypatch.undo()
    assert out.read_bytes() == before, "a reader still sees the previous complete file"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".library.json.")] == []


def test_a_failure_while_serialising_writes_nothing(tmp_path, conn):
    out = tmp_path / "library.json"
    doc = build(conn)
    doc["generated_at"] = object()  # not JSON
    with pytest.raises(TypeError):
        dashboard.write(out, doc)
    assert list(tmp_path.glob("*library.json*")) == [] or not out.exists()
    assert not out.exists()


def test_a_file_of_5000_references_is_well_under_3_mb(conn):
    conn.executemany(
        'INSERT INTO "references"(identity, doi, title, authors, year, venue, url, first_seen,'
        " updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"doi:10.1/{i}",
                f"10.1/{i}",
                f"A realistic title about graph neural networks for molecular property {i}",
                json.dumps(["Smith, Jane", "Doe, John", "Nguyen, An", "Garcia, Maria"]),
                2015 + i % 11,
                "Journal of Chemical Information and Modeling",
                f"https://doi.org/10.1/{i}",
                f"2026-0{1 + i % 9}-1{i % 9}T00:00:00+00:00",
                "2026-09-01",
            )
            for i in range(5000)
        ],
    )
    conn.commit()
    doc = build(conn)
    validate(doc)
    size = len(dashboard.render(doc).encode())
    assert len(doc["references"]) == 5000 and size < 3 * 1024 * 1024, size


# --- the CLI and the path ----------------------------------------------------------------------


def test_the_default_path_follows_the_env_then_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTEST_LIBRARY_DASHBOARD", str(tmp_path / "x" / "lib.json"))
    assert dashboard.default_path() == tmp_path / "x" / "lib.json"
    monkeypatch.delenv("ATTEST_LIBRARY_DASHBOARD")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "box" / ".hermes"))
    assert dashboard.default_path() == (
        tmp_path / "box" / ".hermes" / "workspace/research-desk/library/library.json"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "box" / "agent-home"))
    assert dashboard.default_path().parent.parent.parent.parent == tmp_path / "box" / ".hermes"
    monkeypatch.delenv("HERMES_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    assert dashboard.default_path() == (
        tmp_path / "me" / ".hermes" / "workspace/research-desk/library/library.json"
    )


def test_cli_writes_the_file_to_out_and_prints_one_line(tmp_path, conn, capsys):
    seeded(conn)
    out = tmp_path / "o" / "library.json"
    assert main(["library", "dashboard", "--out", str(out), "--limit", "5"]) == 0
    line = capsys.readouterr().out
    assert str(out) in line and "12 items, 12 references (5 listed, truncated)" in line
    doc = json.loads(out.read_text())
    validate(doc)
    assert doc["references_limit"] == 5 and doc["references_truncated"] is True


def test_cli_on_an_empty_database_writes_a_valid_empty_file_to_the_default_path(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "box" / ".hermes"))
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "fresh.db"))
    assert main(["library", "dashboard"]) == 0
    out = tmp_path / "box" / ".hermes" / "workspace/research-desk/library/library.json"
    validate(json.loads(out.read_text()))
    assert "0 items, 0 references" in capsys.readouterr().out


def test_cli_persona_defaults_to_the_desk_user_then_owner(tmp_path, conn, monkeypatch):
    out = tmp_path / "library.json"
    monkeypatch.setenv("ATTEST_DESK_USER", "deskuser")
    main(["library", "dashboard", "--out", str(out)])
    assert json.loads(out.read_text())["persona"] == "deskuser"
    main(["library", "dashboard", "--out", str(out), "--persona", "someone"])
    assert json.loads(out.read_text())["persona"] == "someone"
    monkeypatch.delenv("ATTEST_DESK_USER")
    main(["library", "dashboard", "--out", str(out)])
    assert json.loads(out.read_text())["persona"] == "owner"


def test_cli_rejects_a_negative_limit_and_writes_nothing(tmp_path, conn, capsys):
    out = tmp_path / "library.json"
    assert main(["library", "dashboard", "--out", str(out), "--limit", "-1"]) != 0
    assert not out.exists()


def test_cli_is_read_only_on_the_database(tmp_path, conn):
    seeded(conn)
    conn.commit()
    before = conn.total_changes
    snapshot = [tuple(r) for r in conn.execute("SELECT * FROM bibliography")]
    main(["library", "dashboard", "--out", str(tmp_path / "library.json")])
    assert [tuple(r) for r in conn.execute("SELECT * FROM bibliography")] == snapshot
    assert conn.total_changes == before


def test_the_command_is_in_the_help_table_and_the_reference(capsys):
    from attestation.cli import HELP

    assert "library.dashboard" in HELP
    with pytest.raises(SystemExit):
        main(["library", "dashboard", "--help"])
    assert "--out" in capsys.readouterr().out


def test_a_dashboard_failure_is_a_nonzero_exit_the_refresh_treats_as_degraded(tmp_path, conn):
    """The refresh script (test_install.py) runs this as a degraded step; here the
    command itself must fail loudly rather than write a half file."""
    seeded(conn)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    assert main(["library", "dashboard", "--out", str(blocker / "library.json")]) != 0
    assert blocker.read_text() == "a file where a directory is needed"


# --- review findings: size bound, snapshot, bytes, profile path, bad bytes ----------------------


def _insert_maxed_refs(conn, n, filler, authors=8, tags=12):
    """Rows at every cap, with `filler` as the text: emoji and quote-heavy are the
    worst cases for a character cap (4 bytes each; every quote doubles in JSON)."""
    rows = []
    for i in range(n):
        rows.append(
            (
                f"doi:10.1/{i}",
                f"10.1/{i}",
                filler * 400,
                json.dumps([filler * 200 for _ in range(authors)], ensure_ascii=False),
                2020,
                filler * 400,
                "https://example.org/" + "a" * 400,
                f"2026-09-{1 + i % 28:02d}T00:00:00+00:00",
                "2026-09-01",
            )
        )
    conn.executemany(
        'INSERT INTO "references"(identity, doi, title, authors, year, venue, url, first_seen,'
        " updated) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    for rid in range(1, n + 1):
        for t in range(tags):
            conn.execute("INSERT INTO reference_tags VALUES (?, ?)", (rid, f"{filler * 60}{t}"))
    conn.commit()


@pytest.mark.parametrize("filler", ["\U0001f9ea", '"', "x"])
def test_the_file_is_bounded_whatever_the_text_is(conn, tmp_path, filler):
    """5000 rows at the character caps were 22.9 MB ASCII, 43 MB quote-heavy and
    84 MB emoji against a consumer that refuses more than 8 MB."""
    _insert_maxed_refs(conn, 2500, filler)
    doc = build(conn)
    validate(doc)
    size = len(dashboard.render(doc).encode())
    assert size <= dashboard.MAX_BYTES == 6 * 1024 * 1024, size
    assert doc["totals"]["references"] == 2500, "totals stay honest"
    out = tmp_path / "library.json"
    dashboard.write(out, doc)
    assert out.stat().st_size <= dashboard.MAX_BYTES


def test_when_rows_are_dropped_for_size_it_is_the_oldest_and_truncated_is_true(conn):
    _insert_maxed_refs(conn, 2500, "\U0001f9ea")
    doc = build(conn)
    kept = doc["references"]
    assert 0 < len(kept) < 2500 and doc["references_truncated"] is True
    assert doc["references_limit"] == len(kept)
    newest_first = [r["added"] for r in kept]
    assert newest_first == sorted(newest_first, reverse=True)
    all_added = [
        _iso
        for (_iso,) in conn.execute('SELECT first_seen FROM "references" ORDER BY first_seen DESC')
    ]
    assert kept[0]["added"][:10] == all_added[0][:10]


def test_text_is_capped_in_bytes_as_well_as_characters():
    text = dashboard.clean("\U0001f9ea" * 400, 300)
    assert len(text.encode()) <= 2 * 300 and len(text) <= 300
    assert dashboard.clean("a" * 400, 300) == "a" * 299 + "…"


def test_limit_is_clamped_to_the_maximum(conn):
    doc = build(conn, limit=10**9)
    assert doc["references_limit"] == dashboard.MAX_LIMIT == 5000


def test_consumer_aligned_caps(conn):
    add_ref(conn, key="k", source="bibtex:" + "s" * 300, arxiv_id="1" * 90)
    doc = build(conn)
    (ref,) = doc["references"]
    assert max(map(len, ref["sources"])) <= 120 and len(ref["arxiv"]) <= 40
    assert len(build(conn, persona="p" * 100)["persona"]) <= 40


def test_a_non_utf8_byte_in_a_column_does_not_break_the_export(conn):
    rid = add_ref(conn, key="bad", title="placeholder")
    conn.execute(
        'UPDATE "references" SET title = CAST(? AS TEXT) WHERE id = ?', (b"caf\xe9 \xff title", rid)
    )
    conn.execute('UPDATE "references" SET venue = CAST(? AS TEXT) WHERE id = ?', (b"V\xc3", rid))
    conn.commit()
    doc = build(conn)
    validate(doc)
    assert "caf" in doc["references"][0]["title"] and "�" in doc["references"][0]["title"]
    assert conn.text_factory is str, "the connection's text factory is restored"


def test_build_reads_one_snapshot(conn):
    """Counts and rows come from one read transaction, so a refresh landing
    mid-export cannot make `references` disagree with `totals.references`."""
    seeded(conn)
    seen = []
    real = dashboard._references

    def spy(c, *a, **k):
        seen.append(c.in_transaction)
        return real(c, *a, **k)

    dashboard._references, saved = spy, dashboard._references
    try:
        build(conn)
    finally:
        dashboard._references = saved
    assert seen == [True]
    assert not conn.in_transaction, "and it is closed again"


def test_backlog_and_tagged_use_one_definition_and_never_go_negative(conn):
    i1 = add_item(conn, 1, published="2026-09-29T08:00:00", tags=["a"])
    add_item(conn, 2, published="2026-09-29T08:00:00")
    conn.execute("DELETE FROM item_features WHERE item_id = ?", (i1,))  # tags without features
    conn.commit()
    doc = build(conn)
    assert doc["totals"]["items_tagged"] == 1 and doc["backlog"]["items_untagged"] == 1


def test_the_default_path_for_a_profile_home_is_the_roots_workspace(tmp_path, monkeypatch):
    """HERMES_HOME=<root>/profiles/<name> must not write under profiles/.hermes."""
    monkeypatch.delenv("ATTEST_LIBRARY_DASHBOARD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "work"))
    assert dashboard.default_path() == (
        tmp_path / ".hermes" / "workspace/research-desk/library/library.json"
    )


def test_out_pointing_at_a_directory_is_a_clear_error_not_a_traceback(tmp_path, conn, capsys):
    rc = main(["library", "dashboard", "--out", str(tmp_path)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "directory" in err and "Traceback" not in err


def test_an_unwritable_destination_is_a_clear_error(tmp_path, conn, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert main(["library", "dashboard", "--out", str(blocker / "library.json")]) != 0
    assert "Traceback" not in capsys.readouterr().err


def test_clean_survives_tiny_limits_and_cuts_by_bytes_before_characters():
    assert dashboard.clean("abcdef", 1) == "a" or len(dashboard.clean("abcdef", 1)) <= 1
    assert len(dashboard.clean("\U0001f9ea" * 5, 2).encode()) <= 4
    # 250 characters, 1000 bytes: under the character cap, over the byte cap
    cut = dashboard.clean("\U0001f9ea" * 250, 300)
    assert len(cut.encode()) <= 600 and cut.endswith("\u2026")
