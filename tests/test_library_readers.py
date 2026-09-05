"""library_readers: the offline readers and, behind flags, the enrichers."""

from pathlib import Path

import httpx
from test_citations import _add_item, _zotero_db

from attestation import library_readers
from attestation.db import get_db

FIX = Path(__file__).parent / "fixtures" / "library"


def test_bibtex_records_carry_abstract_venue_and_arxiv_from_eprint():
    recs = {r.bib_key: r for r in library_readers.BibtexRecords([FIX / "sample.bib"]).records()}
    s = recs["schutt2017schnet"]
    assert s.arxiv_id == "1706.08566" and s.year == 2017
    assert s.venue == "Advances in Neural Information Processing Systems"
    assert s.abstract.startswith("Deep learning")
    assert s.source == "bibtex:sample.bib" and s.fetched_at is None
    # "and others" is a truncation marker, not an author. LaTeX escapes are
    # kept verbatim: de-escaping is a rendering concern the citations spec
    # keeps at the presentation edge.
    assert s.authors == ['Sch{\\"u}tt, Kristof T.', "Kindermans, Pieter-Jan", "Sauceda, Huziel E."]
    n = recs["batzner2022nequip"]
    assert n.doi == "10.1038/s41467-022-29939-5" and n.venue == "Nature Communications"


def test_bib_keywords_and_cites_fields_are_read():
    from attestation.library_readers import _bib_cites, _bib_tags

    assert _bib_tags("Force Fields, equivariant GNN; Molecular Dynamics") == [
        "force-fields",
        "equivariant-gnn",
        "molecular-dynamics",
    ]
    assert _bib_tags("!!!, ok, ok") == ["ok"]
    assert _bib_tags("") == []
    assert _bib_cites("doi:10.5555/schnet|SchNet: a CNN; arxiv:2101.03164") == [
        ("doi:10.5555/schnet", "SchNet: a CNN"),
        ("arxiv:2101.03164", None),
    ]
    assert _bib_cites("") == []


def test_bibtex_records_carry_keywords_and_cites(tmp_path):
    bib = tmp_path / "k.bib"
    bib.write_text(
        "@article{k,\n  title = {K},\n  keywords = {force-fields, gnn},\n"
        "  cites = {doi:10.5555/schnet|SchNet},\n}\n"
    )
    (rec,) = library_readers.BibtexRecords([bib]).records()
    assert rec.tags == ["force-fields", "gnn"]
    assert rec.cites == [("doi:10.5555/schnet", "SchNet")]


def test_a_missing_bib_is_an_absent_source(tmp_path):
    assert list(library_readers.BibtexRecords([tmp_path / "nope.bib"]).records()) == []


def test_feed_records_only_for_items_with_an_id(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://a', 'arXiv')")
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash, arxiv_id, published)"
        " VALUES (1, 'g1', 'NequIP', 'https://arxiv.org/abs/2106.02347', 'abs', 'h1',"
        " '2106.02347', '2021-06-04')"
    )
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'g2', 'HN post', 'http://x', 's', 'h2')"
    )
    recs = list(library_readers.FeedRecords(conn).records())
    assert [(r.source, r.source_key, r.arxiv_id, r.abstract, r.year) for r in recs] == [
        ("feed", "1", "2106.02347", "abs", 2021)
    ]


def test_zotero_records_carry_the_key_as_bib_key(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = _zotero_db(db)
    _add_item(conn, 1, "ABCD1234", title="Stored", year=2019, doi="10.1/z", authors=[("A", "B")])
    _add_item(conn, 2, "GONE0000", title="Trashed", deleted=True)
    conn.close()
    recs = list(library_readers.ZoteroRecords(db).records())
    assert [(r.source, r.source_key, r.bib_key, r.doi, r.year, r.authors) for r in recs] == [
        ("zotero", "ABCD1234", "ABCD1234", "10.1/z", 2019, ["B, A"])
    ]


def test_an_absent_zotero_yields_nothing(tmp_path):
    assert list(library_readers.ZoteroRecords(tmp_path / "none.sqlite").records()) == []


# ---------------------------------------------------------------------------
# enrichers: fill-only, cached, armed by flags read at construction
# ---------------------------------------------------------------------------


def _fake_transport(responses: dict):
    """URL-prefix -> (status, body, headers). Records every request URL."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        for prefix, (status, body, headers) in responses.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, content=body, headers=headers, request=request)
        return httpx.Response(404, request=request)

    return httpx.MockTransport(handler), calls


def _store_with(conn, **kw):
    from attestation.library import ReferenceRecord, upsert

    kw.setdefault("source", "bibtex:/a.bib")
    kw.setdefault("source_key", "k")
    return upsert(conn, ReferenceRecord(**kw))[0]


def test_arxiv_enricher_fills_abstract_authors_and_doi(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="SchNet", arxiv_id="1706.08566")
    transport, calls = _fake_transport(
        {"https://export.arxiv.org/api/query": (200, (FIX / "arxiv_query.xml").read_bytes(), {})}
    )
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    cache = tmp_path / "cache"
    recs = list(library_readers.ArxivEnricher(cache_dir=cache).records(conn, None))
    assert len(recs) == 1
    assert recs[0].doi == "10.5555/schnet" and recs[0].abstract.startswith("Deep learning")
    assert recs[0].authors == ["Kristof T. Schütt", "Pieter-Jan Kindermans"]
    assert recs[0].year == 2017 and recs[0].source == "arxiv"
    assert recs[0].source_key == "arxiv:1706.08566"  # the row's identity, so upsert finds it
    assert recs[0].fetched_at is not None
    # Cached: a second pass makes no request and keeps the ORIGINAL fetched_at.
    first = recs[0].fetched_at
    recs2 = list(library_readers.ArxivEnricher(cache_dir=cache).records(conn, None))
    assert len(calls) == 1 and recs2[0].fetched_at == first


def test_crossref_enricher_fills_venue_and_strips_jats(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/s41467-022-29939-5")
    transport, _calls = _fake_transport(
        {"https://api.crossref.org/works/": (200, (FIX / "crossref_work.json").read_bytes(), {})}
    )
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    recs = list(library_readers.CrossrefEnricher(cache_dir=tmp_path / "c").records(conn, None))
    assert recs[0].venue == "Nature Communications" and recs[0].year == 2022
    assert (
        recs[0].abstract == "This work presents Neural Equivariant Interatomic Potentials (NequIP)."
    )
    assert recs[0].authors == ["Batzner, Simon", "Musaelian, Albert"]


def test_s2_enricher_yields_cites_and_backs_off_on_429(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/s41467-022-29939-5")
    body = (FIX / "s2_paper.json").read_bytes()
    seq = iter([(429, b"", {"Retry-After": "0"}), (200, body, {})])
    slept: list[float] = []

    def handler(request):
        status, content, headers = next(seq)
        return httpx.Response(status, content=content, headers=headers, request=request)

    monkeypatch.setattr(
        library_readers, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    monkeypatch.setattr(library_readers, "_sleep", slept.append)
    recs = list(library_readers.S2Enricher(cache_dir=tmp_path / "c").records(conn, None))
    assert recs[0].arxiv_id == "2101.03164" and recs[0].source == "s2"
    # Two traceable references; no ids and no title, or a journal-abbreviation title, is dropped.
    assert recs[0].cites == [
        ("doi:10.5555/schnet", "SchNet ..."),
        (
            "title:an untraceable reference with a long title:-",
            "An untraceable reference with a long title",
        ),
    ]
    assert slept == [10.0, 3.0]  # Retry-After floored at 10 s, then the per-request pace


def test_a_dead_network_leaves_the_row_for_the_next_sync(tmp_path, monkeypatch):
    """A transport failure is not a miss: nothing is written, the row is
    retried next time, and the sync report counts it as failed. The first
    generation of examples/molecular-ai marked 24 rate-limited papers as tried
    and could then never repair them."""
    from attestation.library import sync

    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/x")

    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    monkeypatch.setattr(
        library_readers, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    enricher = library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")
    assert list(enricher.records(conn, None)) == []
    assert enricher.errors == ["doi:10.1038/x: ConnectError: no route"]
    report = sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
    assert report.sources["crossref"]["failed"] == 1
    assert (
        conn.execute("SELECT count(*) FROM reference_sources WHERE source='crossref'").fetchone()[0]
        == 0
    )


def test_a_404_is_a_definite_miss_and_marks_the_row_tried(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/x")
    transport, _calls = _fake_transport({})  # everything 404s
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    recs = list(library_readers.CrossrefEnricher(cache_dir=tmp_path / "c").records(conn, None))
    assert len(recs) == 1 and recs[0].title is None  # a miss: the row is marked tried


def test_an_exhausted_429_is_transient_not_a_miss(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/x")
    slept: list[float] = []
    transport, calls = _fake_transport(
        {"https://api.semanticscholar.org/": (429, b"", {"Retry-After": "1"})}
    )
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(library_readers, "_sleep", slept.append)
    enricher = library_readers.S2Enricher(cache_dir=tmp_path / "c")
    assert list(enricher.records(conn, None)) == []
    # One back-off (Retry-After floored at 10 s), one more try, then the row
    # is given up for this pass; the per-request pace still applies after.
    assert len(calls) == 2 and enricher.errors[0].endswith("HTTP 429")
    assert slept == [10.0, 3.0]


def test_no_request_is_made_with_the_flags_unset(tmp_path, monkeypatch):
    from attestation.library import sync

    conn = get_db(tmp_path / "t.db")

    def explode(*a, **k):
        raise AssertionError("network touched")

    monkeypatch.setattr(httpx, "Client", explode)
    monkeypatch.setattr(httpx, "get", explode)
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    monkeypatch.delenv("ATTEST_CITATION_SCHOLAR", raising=False)
    readers = library_readers.readers_from_env(
        conn, bib_paths=[FIX / "sample.bib"], zotero_path=tmp_path / "none.sqlite"
    )
    assert [r.name for r in readers] == ["bibtex", "feed"]
    sync(conn, readers)


def test_flags_arm_the_enrichers_at_construction(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")
    monkeypatch.setenv("ATTEST_CITATION_SCHOLAR", "1")
    readers = library_readers.readers_from_env(conn, bib_paths=[], zotero_path=tmp_path / "n")
    assert [r.name for r in readers] == ["feed", "arxiv", "crossref", "s2"]
    monkeypatch.delenv("ATTEST_CITATION_SCHOLAR")
    assert [r.name for r in readers] == ["feed", "arxiv", "crossref", "s2"]  # already built
    assert [
        r.name
        for r in library_readers.readers_from_env(
            conn, bib_paths=[], zotero_path=tmp_path / "n", sources=["feed"]
        )
    ] == ["feed"]
