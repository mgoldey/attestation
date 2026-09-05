"""Regression tests from review round 2 (2026-09-05): five lenses over the
round-1 delta. Each finding is the failing test the lens named, against the
live path (a sync, not a fixture) wherever the lens caught round 1 doing
otherwise.
"""

import httpx
import pytest

from attestation import library, library_readers
from attestation.db import get_db
from attestation.library import ReferenceRecord, sync, upsert


def _store_with(conn, **kw):
    kw.setdefault("source", "bibtex:a.bib")
    kw.setdefault("source_key", kw.get("bib_key", "k"))
    return upsert(conn, ReferenceRecord(**kw))[0]


def _transport(status: int, body: bytes = b"", headers: dict | None = None):
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(status, content=body, headers=headers or {}, request=request)

    return httpx.MockTransport(handler), calls


def _n_refs(conn) -> int:
    return conn.execute('SELECT count(*) FROM "references"').fetchone()[0]


# ---------------------------------------------------------------------------
# the title join persists, in both orders
# ---------------------------------------------------------------------------


def test_a_title_only_bib_entry_survives_a_second_sync(tmp_path):
    """Round 1 let a DOI record upgrade a title-only row; the way back did not
    exist, so every re-sync of the .bib entry added a duplicate row."""
    conn = get_db(tmp_path / "t.db")
    p = tmp_path / "a.bib"
    p.write_text(
        "@article{schnet,\n  title = {SchNet: A continuous-filter CNN},\n  year = {2017},\n}\n"
    )
    conn.execute(
        "INSERT INTO items(id, title, content_hash, doi, published) VALUES"
        " (9, 'SchNet: A continuous-filter CNN', 'h', '10.5555/x', '2017-06-01')"
    )
    conn.commit()
    readers = [library_readers.BibtexRecords([p]), library_readers.FeedRecords(conn)]
    sync(conn, readers)
    r2 = sync(conn, readers)
    assert _n_refs(conn) == 1
    assert r2.sources["bibtex"]["unchanged"] == 1 and r2.sources["bibtex"]["added"] == 0
    assert r2.conflicts == 0  # an earlier pass's conflicts are not re-reported


def test_a_title_only_record_meets_a_doi_row_that_came_first(tmp_path):
    """The live database is feed-first: the DOI row exists before any .bib."""
    conn = get_db(tmp_path / "t.db")
    a = _store_with(
        conn, source="feed", source_key="1", title="SchNet: a CNN", year=2017, doi="10.5/x"
    )
    b = _store_with(conn, bib_key="schnet", title="SchNet: A CNN", year=2017)
    assert a == b
    row = conn.execute('SELECT identity, bib_key, title_key FROM "references"').fetchone()
    assert tuple(row) == ("doi:10.5/x", "schnet", "schnet a cnn")


def test_different_years_stay_different_papers(tmp_path):
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, bib_key="a", title="Introduction", year=2019)
    b = _store_with(conn, bib_key="b", title="Introduction", year=2021)
    assert a != b


def test_a_title_stub_resolves_to_the_row_that_gained_a_doi(tmp_path):
    """MACE cites SphereNet by title; SphereNet is in the library under its
    arXiv id. `related` lands the edge, and a .bib `cites` title form is
    normalised like any identity."""
    conn = get_db(tmp_path / "t.db")
    s = _store_with(
        conn,
        bib_key="sphere",
        title="Spherical Message Passing for 3D Graph Networks",
        arxiv_id="2102.05013",
    )
    _store_with(
        conn,
        bib_key="mace",
        title="MACE",
        cites=library_readers._bib_cites("Title:Spherical Message-Passing for 3D Graph Networks:-"),
    )
    rel = library.related(conn, "mace")
    assert [(n.identity, n.in_library, n.id) for n in rel.cites] == [
        ("title:spherical message passing for 3d graph networks:-", True, s)
    ]


# ---------------------------------------------------------------------------
# identifiers: LaTeX, collisions in columns, preprint venues
# ---------------------------------------------------------------------------


def test_latex_fold_does_not_merge_distinct_titles():
    a = library.normalise_title("$\\alpha$-synuclein aggregation kinetics")
    b = library.normalise_title("$\\beta$-synuclein aggregation kinetics")
    assert a != b and a == "alpha synuclein aggregation kinetics"
    assert library.normalise_title("Fran\\c{c}ois \\& Schr\\o der") == "francois schroder"
    assert library.normalise_title("\\emph{Ab initio} {M}ethods") == "ab initio methods"


def test_an_id_held_in_another_rows_column_is_refused_too(tmp_path):
    conn = get_db(tmp_path / "t.db")
    b = _store_with(
        conn, source="feed", source_key="1", title="P", doi="10.1/d", arxiv_id="2106.02347"
    )
    a = _store_with(conn, title="P preprint", year=2021)
    conn.commit()

    class S2:
        name, network = "s2", True
        errors: list[str] = []

        def records(self, conn, limit):
            yield ReferenceRecord(
                source="s2",
                source_key="title:p preprint:2021",
                title="P preprint",
                arxiv_id="2106.02347",
                fetched_at="d",
            )

    sync(conn, [S2()])
    ids = [
        r["id"]
        for r in conn.execute('SELECT id FROM "references" WHERE arxiv_id = ?', ("2106.02347",))
    ]
    assert ids == [b]
    raw = conn.execute("SELECT raw FROM reference_sources WHERE source = 's2'").fetchone()[0]
    assert '"identity"' in raw and a != b


def test_a_preprint_venue_does_not_move_the_year_backwards():
    merged, conflicts = library.merge(
        {"title": "P", "year": 2022, "venue": None},
        {"title": "P", "year": 2021, "venue": "arXiv:2101.00001 [cs.LG]"},
    )
    assert merged["year"] == 2022 and conflicts["year"] == {"kept": 2022, "offered": 2021}


# ---------------------------------------------------------------------------
# an armed sync finishes, and marks tried only what it actually answered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [b"<html>cloudflare</html>", b"[]", b'"str"', b""])
def test_a_non_json_200_neither_marks_tried_nor_caches(tmp_path, monkeypatch, body):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", doi="10.1/a")
    conn.commit()
    transport, calls = _transport(200, body)
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    for _ in range(2):
        report = sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
        assert report.sources["crossref"]["failed"] == 1
    assert len(calls) == 2
    assert not list((tmp_path / "c").glob("*.json"))
    tried = conn.execute(
        "SELECT count(*) FROM reference_sources WHERE source='crossref'"
    ).fetchone()
    assert tried[0] == 0


def test_s2_non_json_200_is_forgotten_too(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", doi="10.1/a")
    conn.commit()
    transport, calls = _transport(200, b"<html>rate limited</html>")
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(library_readers, "_sleep", lambda s: None)
    for _ in range(2):
        sync(conn, [library_readers.S2Enricher(cache_dir=tmp_path / "c")])
    assert len(calls) == 2 and not list((tmp_path / "c").glob("*.json"))


def test_an_empty_crossref_message_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", doi="10.1/x")
    conn.commit()
    transport, calls = _transport(200, b'{"message": {}}')
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    for _ in range(2):
        report = sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
    assert report.sources["crossref"] == {
        "seen": 0,
        "added": 0,
        "merged": 0,
        "unchanged": 0,
        "enriched": 0,
        "failed": 0,
    }
    assert len(calls) == 1  # a parsed-but-empty answer is a definite miss: tried once


def test_an_arxiv_entity_payload_is_forgotten_not_a_crash(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", arxiv_id="1706.08566")
    conn.commit()
    body = b'<!DOCTYPE feed [<!ENTITY x "y">]><feed xmlns="http://www.w3.org/2005/Atom">&x;</feed>'
    transport, calls = _transport(200, body)
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    report = sync(conn, [library_readers.ArxivEnricher(cache_dir=tmp_path / "c")])
    assert report.sources["arxiv"]["failed"] == 1 and not list((tmp_path / "c").glob("*.json"))


def test_a_malformed_id_is_marked_tried_alone_and_never_put_on_a_url(tmp_path, monkeypatch):
    """One `#frag` in an arXiv id truncated the batch query and marked 49
    other rows answered (review round 2)."""
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="bad", title="Bad", arxiv_id="1706.08566#f")
    good = _store_with(conn, bib_key="good", title="Good", arxiv_id="2106.02347")
    conn.commit()
    transport, calls = _transport(404)
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    sync(conn, [library_readers.ArxivEnricher(cache_dir=tmp_path / "c")])
    assert len(calls) == 1 and "id_list=2106.02347&" in calls[0] and "#" not in calls[0]
    tried = {
        r[0]
        for r in conn.execute("SELECT reference_id FROM reference_sources WHERE source='arxiv'")
    }
    assert good in tried and len(tried) == 2  # the bad row is tried too, on its own


def test_a_doi_with_a_query_or_dot_segment_is_refused(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="q", title="Q", doi="10.1/a?rows=1000")
    _store_with(conn, bib_key="d", title="D", doi="10.1/a/../b")
    conn.commit()
    transport, calls = _transport(200, b'{"message": {"title": ["T"]}}')
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
    assert calls == []


def test_5xx_backs_off_twice_bounded(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(library_readers, "_sleep", slept.append)
    transport, calls = _transport(503, headers={"Retry-After": "1e12"})
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    resp, err = library_readers._fetch_with_backoff("https://api.crossref.org/works/x")
    assert resp is None and err == "HTTP 503" and len(calls) == 3 and slept == [60.0, 60.0]


def test_two_bib_files_with_one_name_keep_separate_provenance(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d, venue in (("a", "J1"), ("b", "J2")):
        (tmp_path / d / "refs.bib").write_text(
            f"@article{{k1,\n  title = {{T}},\n  journal = {{{venue}}},\n  doi = {{10.1/t}},\n}}\n"
        )
    conn = get_db(tmp_path / "t.db")
    reader = library_readers.BibtexRecords(
        [tmp_path / "a" / "refs.bib", tmp_path / "b" / "refs.bib"]
    )
    report = sync(conn, [reader])
    sources = [r[0] for r in conn.execute("SELECT source FROM reference_sources ORDER BY source")]
    assert sources == ["bibtex:refs.bib", "bibtex:refs.bib#2"] and report.conflicts == 1


def test_crossref_abstract_drops_citation_numbers_and_signed_blurbs():
    msg = {
        "abstract": "<jats:p>Through an enormous experimental effort<jats:sup>1–4</jats:sup>,"
        " the structures\n  of around 100,000 proteins</jats:p>"
    }
    assert (
        library_readers._crossref_abstract(msg)
        == "Through an enormous experimental effort, the structures of around 100,000 proteins"
    )
    blurb = {"abstract": "<jats:p>Deep learning takes on protein folding. In 1972 ... —VV</jats:p>"}
    assert library_readers._crossref_abstract(blurb) is None


# ---------------------------------------------------------------------------
# search at scale; the graph keeps a reader's keywords
# ---------------------------------------------------------------------------


def test_filtered_semantic_search_survives_more_than_4096_vectors(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    for i in range(4097):
        _store_with(conn, bib_key=f"x{i}", title=f"Filler paper {i}", year=2020 + (i % 2))
    library.embed_missing(conn, fake_embedder, None)
    res = library.search(conn, "filler paper", embedder=fake_embedder, year=2021, limit=3)
    assert res.semantic is True and all(h.year == 2021 for h in res.hits)


def test_the_emitted_similarity_is_the_number_the_order_was_made_from(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="a", title="Diffusion models for docking")
    _store_with(conn, bib_key="b", title="Ion transport in solids")
    library.embed_missing(conn, fake_embedder, None)
    res = library.search(conn, "ion channels", embedder=fake_embedder, limit=2)
    sims = [h.similarity for h in res.hits]
    assert sims == sorted(sims, reverse=True)


def test_substring_escapes_like_wildcards(tmp_path):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="x", title="axb paper")
    _store_with(conn, bib_key="y", title="a_b paper")
    assert [h.bib_key for h in library.search(conn, "a_b").hits] == ["y"]
    assert (
        library.search(conn, "schnet").caveat is None
        or "direct lookup" not in library.search(conn, "nothing here").caveat
    )


def test_a_query_that_is_a_key_says_it_was_a_direct_lookup(tmp_path):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="gap", title="Gaussian approximation potentials")
    res = library.search(conn, "gap")
    assert res.semantic is False and "direct lookup" in res.caveat


def test_a_feed_papers_bib_keywords_join_its_item_node(tmp_path):
    """Round 1 dropped every feed-derived reference's tags; a reader's own
    keywords are the one human tag signal and must reach the graph -- once
    per (paper, tag), under the item's id."""
    from attestation import kg
    from attestation.features import tag_vocabulary

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO items(id, title, content_hash) VALUES (1, 'i', 'h')")
    conn.execute("INSERT INTO items(id, title, content_hash) VALUES (2, 'j', 'h2')")
    conn.executemany("INSERT INTO item_tags VALUES (1, ?)", [("gnn",)])
    r1 = _store_with(conn, source="feed", source_key="1", title="i", arxiv_id="2106.02347")
    upsert(
        conn,
        ReferenceRecord(
            source="bibtex:a.bib",
            source_key="i",
            bib_key="i",
            title="i",
            arxiv_id="2106.02347",
            tags=["gnn", "force-fields"],
        ),
    )
    _store_with(conn, source="feed", source_key="2", title="j", arxiv_id="2106.99999")
    upsert(
        conn,
        ReferenceRecord(
            source="bibtex:a.bib",
            source_key="j",
            bib_key="j",
            title="j",
            arxiv_id="2106.99999",
            tags=["force-fields", "equivariance"],
        ),
    )
    other = _store_with(conn, bib_key="o", title="o", tags=["gnn"])
    pairs = sorted(kg.tag_assignments(conn))
    assert pairs == [
        (-other, "gnn"),
        (1, "force-fields"),
        (1, "gnn"),
        (2, "equivariance"),
        (2, "force-fields"),
    ]
    assert r1 not in {abs(p) for p, _ in pairs if p < 0}
    assert tag_vocabulary(conn) == ["force-fields", "gnn", "equivariance"]


def test_migration_008_backfills_title_key(tmp_path):
    from attestation import db as dbmod

    path = tmp_path / "v7.db"
    conn = dbmod.get_db(path)
    _store_with(conn, bib_key="a", title='Schr{\\"o}dinger Equation', year=2000)
    conn.execute('UPDATE "references" SET title_key = NULL')
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()
    conn = dbmod.get_db(path)
    assert (
        conn.execute('SELECT title_key FROM "references"').fetchone()[0] == "schrodinger equation"
    )
