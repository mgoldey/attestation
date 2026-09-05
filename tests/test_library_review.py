"""Regression tests from review round 1 (2026-09-05): six lenses over the
library store, each finding reproduced here as the failing test the lens
named. Grouped by the invariant each protects rather than by module.
"""

import math

import httpx
import numpy as np
import pytest

from attestation import library, library_readers
from attestation.db import embed_dims, get_db
from attestation.library import ReferenceRecord, sync, upsert


def _store_with(conn, **kw):
    kw.setdefault("source", "bibtex:/a.bib")
    kw.setdefault("source_key", kw.get("bib_key", "k"))
    return upsert(conn, ReferenceRecord(**kw))[0]


def _counting_transport(status: int, body: bytes = b"", headers: dict | None = None):
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(status, content=body, headers=headers or {}, request=request)

    return httpx.MockTransport(handler), calls


# ---------------------------------------------------------------------------
# an enricher's answer attaches to the row it was fetched FOR
# ---------------------------------------------------------------------------


def test_a_404_marks_the_row_tried_through_sync(tmp_path, monkeypatch):
    """Two syncs over a 404 make ONE wire request: the miss writes the source
    row that `_todo` excludes. It was counted as failed and re-fetched forever
    (three lenses named this; the old test stopped at the reader)."""
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/x")
    conn.commit()
    transport, calls = _counting_transport(404)
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    for _ in range(2):
        report = sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
    assert len(calls) == 1
    assert report.sources["crossref"] == {
        "seen": 0,
        "added": 0,
        "merged": 0,
        "unchanged": 0,
        "enriched": 0,
        "failed": 0,
    }
    n = conn.execute("SELECT count(*) FROM reference_sources WHERE source = 'crossref'").fetchone()
    assert n[0] == 1


def test_an_enricher_never_introduces_a_row(tmp_path):
    """A Semantic Scholar answer carrying ids the store has never seen still
    lands on the row it was fetched for (spec §3.1: enrichers never introduce
    a reference)."""
    conn = get_db(tmp_path / "t.db")
    rid = _store_with(
        conn, source="zotero", source_key="Z", title="P", doi="10.48550/arxiv.2101.03164"
    )
    conn.commit()

    class S2:
        name, network = "s2", True
        errors: list[str] = []

        def records(self, conn, limit):
            yield ReferenceRecord(
                source="s2",
                source_key="arxiv:2101.03164",
                title="P",
                doi="10.1038/s41467-022-29939-5",
                arxiv_id="2101.03164",
                fetched_at="2026-09-05",
            )

    report = sync(conn, [S2()])
    assert conn.execute('SELECT count(*) FROM "references"').fetchone()[0] == 1
    assert report.sources["s2"]["enriched"] == 1
    src = conn.execute("SELECT reference_id FROM reference_sources WHERE source='s2'").fetchone()
    assert src[0] == rid
    row = conn.execute('SELECT identity, doi FROM "references"').fetchone()
    assert (row["identity"], row["doi"]) == (
        "doi:10.1038/s41467-022-29939-5",
        "10.1038/s41467-022-29939-5",
    )


def test_an_id_already_held_by_another_row_is_refused_and_recorded(tmp_path):
    """Feed row by arXiv id, .bib row by DOI, and the arXiv API says they are
    one paper: the ids are refused (folding two rows is not done here), the
    collision is recorded, and the arXiv row is marked tried -- not fetched
    again on every sync, and never a UNIQUE violation."""
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, source="feed", source_key="1", title="P", arxiv_id="2106.02347")
    b = _store_with(conn, title="P (journal)", doi="10.1/p")
    conn.commit()

    class Arxiv:
        name, network = "arxiv", True
        errors: list[str] = []

        def records(self, conn, limit):
            yield ReferenceRecord(
                source="arxiv",
                source_key="arxiv:2106.02347",
                title="P",
                arxiv_id="2106.02347",
                doi="10.1/p",
                fetched_at="2026-09-05",
            )

    sync(conn, [Arxiv()])
    sync(conn, [Arxiv()])
    rows = {r["id"]: r for r in conn.execute('SELECT * FROM "references"')}
    assert set(rows) == {a, b} and rows[a]["doi"] is None
    srcs = conn.execute(
        "SELECT reference_id, count(*) FROM reference_sources WHERE source='arxiv'"
    ).fetchall()
    assert [tuple(s) for s in srcs] == [(a, 1)]
    out = library.lookup_row(conn, "arxiv:2106.02347")
    assert out["id"] == a


# ---------------------------------------------------------------------------
# identity: DataCite arXiv DOIs, Zotero's category suffix, LaTeX, title forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["2106.02347 [cs.LG]", "arXiv: 2106.02347v2 [cs.LG]", "2106.02347", " 2106.02347v1 "]
)
def test_normalise_arxiv_drops_zoteros_category_suffix(raw):
    assert library.normalise_arxiv(raw) == "2106.02347"


def test_a_datacite_arxiv_doi_is_the_arxiv_id_it_names():
    assert library.identity("10.48550/arxiv.2302.14231", None, "CHGNet", 2023) == "arxiv:2302.14231"
    assert library.identity("10.48550/ARXIV.2302.14231v2", None, "x", 2023) == "arxiv:2302.14231"
    assert library.normalise_doi("10.48550/arxiv.2302.14231") is None
    assert library.arxiv_from_doi("https://doi.org/10.48550/arXiv.2302.14231") == "2302.14231"
    assert library.arxiv_from_doi("10.1038/x") is None
    fields = ReferenceRecord(source="s", source_key="k", doi="10.48550/arxiv.2302.14231").fields()
    assert fields == {"arxiv_id": "2302.14231"}


def test_a_datacite_doi_row_merges_with_the_journal_doi_row(tmp_path):
    """The three-way split the molecular-AI generation showed: arXiv seed, S2's
    DataCite DOI, Zotero's journal DOI -- one row, identity upgraded to the
    publisher's DOI."""
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, source="feed", source_key="1", title="CHGNet", arxiv_id="2302.14231")
    b = _store_with(
        conn,
        source="zotero",
        source_key="Z",
        title="CHGNet",
        doi="10.48550/arxiv.2302.14231",
        year=2023,
    )
    c = _store_with(
        conn,
        source="zotero",
        source_key="Z2",
        title="CHGNet",
        doi="10.1038/s42256-023-00716-3",
        arxiv_id="2302.14231 [cond-mat.mtrl-sci]",
    )
    assert a == b == c
    row = conn.execute('SELECT identity, doi, arxiv_id FROM "references"').fetchone()
    assert tuple(row) == (
        "doi:10.1038/s42256-023-00716-3",
        "10.1038/s42256-023-00716-3",
        "2302.14231",
    )


def test_normalise_title_folds_latex_accents_and_commands():
    assert library.normalise_title('Schr{\\"o}dinger equation') == "schrodinger equation"
    assert library.normalise_title("Schrödinger equation") == "schrodinger equation"
    assert library.normalise_title("\\emph{Ab initio} {M}ethods") == "ab initio methods"
    assert library._same("authors", ['Sch{\\"u}tt, K.'], ["Schütt, K."])


def test_a_title_only_bib_row_meets_the_doi_record_for_the_same_paper(tmp_path):
    """A hand-typed .bib entry with no DOI and the feed's DOI-bearing item are
    one row, and the row is upgraded to the DOI identity."""
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, bib_key="schnet", title="SchNet: A continuous-filter CNN", year=2017)
    b = _store_with(
        conn,
        source="feed",
        source_key="9",
        title="SchNet: A continuous-filter CNN",
        year=2017,
        doi="10.5555/x",
    )
    assert a == b
    row = conn.execute('SELECT identity, bib_key FROM "references"').fetchone()
    assert tuple(row) == ("doi:10.5555/x", "schnet")


def test_a_venue_brings_its_year(tmp_path):
    """`Nature Communications 2021` for a paper published in 2022 is a wrong
    citation: the source that names the journal names the year, recorded as
    a conflict like any other first-wins exception."""
    merged, conflicts = library.merge(
        {"title": "NequIP", "year": 2021, "venue": None},
        {"title": "NequIP", "year": 2022, "venue": "Nature Communications"},
    )
    assert merged["year"] == 2022 and merged["venue"] == "Nature Communications"
    assert conflicts["year"] == {"kept": 2022, "offered": 2021, "note": "the venue's year wins"}
    # A second venue does NOT move the year again: first venue wins, as ever.
    merged2, conflicts2 = library.merge(merged, {"year": 2023, "venue": "Some Reprint"})
    assert merged2["year"] == 2022 and conflicts2["venue"]["offered"] == "Some Reprint"


# ---------------------------------------------------------------------------
# an armed sync finishes: headers, bodies and ids off the wire never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("Fri, 31 Dec 1999 23:59:59 GMT", 10.0),  # an HTTP-date in the past: the floor
        ("1e12", 60.0),
        ("nan", 10.0),
        ("inf", 10.0),
        ("junk", 10.0),
        (None, 10.0),
        ("30", 30.0),
        ("2", 10.0),
    ],
)
def test_retry_after_is_parsed_defensively_and_capped(value, want):
    got = library_readers._retry_after(value)
    assert math.isfinite(got) and got == want


def test_a_malformed_200_body_is_not_cached_and_not_a_crash(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", arxiv_id="1706.08566")
    conn.commit()
    transport, calls = _counting_transport(200, b"<feed><entry>not closed")
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    for _ in range(2):
        report = sync(conn, [library_readers.ArxivEnricher(cache_dir=tmp_path / "c")])
        assert report.sources["arxiv"]["failed"] == 1
    assert len(calls) == 2  # not cached: the next sync asks again
    assert not list((tmp_path / "c").glob("*.json"))


def test_a_bad_stored_id_is_an_absent_source_not_an_error(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="X", doi="10.1/a\nb")
    conn.commit()
    transport, calls = _counting_transport(200, b"{}")
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    report = sync(conn, [library_readers.CrossrefEnricher(cache_dir=tmp_path / "c")])
    assert report.sources["crossref"]["failed"] == 1


def test_crossref_shapes_off_the_wire_are_tolerated():
    assert library_readers._crossref_year({"issued": {"date-parts": [[]]}}) is None
    assert library_readers._crossref_year({"issued": {"date-parts": [["2019", 3]]}}) == 2019
    assert library_readers._crossref_authors({"author": ["x", {"family": "Jumper"}]}) == ["Jumper"]
    rec = library_readers._crossref_record(
        {
            "title": ["A &amp; B"],
            "container-title": ["Multiscale Modeling &amp; Simulation"],
            "abstract": "<jats:p>Abstract Deep learning takes on folding.</jats:p>",
            "DOI": ["not-a-string"],
        },
        {"identity": "doi:x", "doi": "x", "arxiv_id": None},
        None,
    )
    assert rec.title == "A & B" and rec.venue == "Multiscale Modeling & Simulation"
    assert rec.abstract == "Deep learning takes on folding." and rec.doi == "x"


def test_s2_shapes_off_the_wire_are_tolerated():
    row = {"identity": "doi:x", "title": "kept"}
    rec = library_readers._s2_record(
        {"paperId": "p", "externalIds": ["oops"], "references": "nope", "title": None}, row, None
    )
    assert rec.title == "kept" and rec.doi is None and rec.cites == []


def test_parse_arxiv_keeps_old_style_ids_whole():
    body = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        b"<id>http://arxiv.org/abs/cond-mat/0301234v1</id><title>T</title></entry></feed>"
    )
    assert list(library_readers._parse_arxiv(body, None)) == ["cond-mat/0301234"]


# ---------------------------------------------------------------------------
# citation edges: junk stubs out, identities normalised, stubs ordered
# ---------------------------------------------------------------------------


def test_s2_reference_list_junk_is_dropped_but_id_bearing_stubs_stay():
    refs = [
        {"title": "Phys. Rev. B", "externalIds": {}},
        {"title": "J. Chem. Phys", "externalIds": {}},
        {"title": "and as an in", "externalIds": {}},
        {"title": "AUTHOR CONTRIBUTIONS", "externalIds": {}},
        {"title": "Learn", "externalIds": {"DOI": "10.1/learn"}},  # an id: traceable
        {
            "title": "Tensor field networks: rotation- and translation-equivariant",
            "externalIds": {},
        },
    ]
    assert library_readers._s2_cites(refs) == [
        ("doi:10.1/learn", "Learn"),
        (
            "title:tensor field networks rotation and translation equivariant:-",
            "Tensor field networks: rotation- and translation-equivariant",
        ),
    ]


def test_bib_cites_identities_are_normalised():
    assert library_readers._bib_cites("DOI:10.1/X|T; arxiv:2101.03164v2; Title:Foo Bar:-") == [
        ("doi:10.1/x", "T"),
        ("arxiv:2101.03164", None),
        ("title:foo bar:-", None),
    ]
    assert library_readers._bib_cites("doi:10.48550/arxiv.2302.14231") == [
        ("arxiv:2302.14231", None)
    ]


def test_related_orders_id_stubs_before_title_only_stubs(tmp_path):
    conn = get_db(tmp_path / "t.db")
    _store_with(
        conn,
        bib_key="a",
        title="A",
        cites=[
            ("title:zzz a long enough title here:-", "Zzz a long enough title here"),
            ("doi:10.1/b", "B paper"),
            ("arxiv:1706.08566", "SchNet"),
        ],
    )
    _store_with(conn, bib_key="schnet", title="SchNet", arxiv_id="1706.08566")
    rel = library.related(conn, "a")
    assert [n.identity for n in rel.cites] == [
        "arxiv:1706.08566",
        "doi:10.1/b",
        "title:zzz a long enough title here:-",
    ]


# ---------------------------------------------------------------------------
# search: filters before the floor, whole-token boost, word-AND fallback
# ---------------------------------------------------------------------------


def test_fielded_filters_restrict_candidates_before_the_floor(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, bib_key="a", title="Equivariant potentials", authors=["Batzner, Simon"])
    c = _store_with(conn, bib_key="c", title="Sourdough", authors=["Musaelian, Albert"])
    for i in range(30):
        _store_with(conn, bib_key=f"x{i}", title=f"Filler paper {i}", authors=["Nobody, N."])
    library.embed_missing(conn, fake_embedder, None)
    q = fake_embedder.embed_query("equivariant force fields")
    conn.execute("DELETE FROM reference_vectors WHERE rowid = ?", (a,))
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)", (a, q.tobytes()))
    # Both papers carry the query vector; only the filter separates them, and
    # the floor must be computed over the filtered set (one candidate).
    conn.execute("DELETE FROM reference_vectors WHERE rowid = ?", (c,))
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)", (c, q.tobytes()))
    res = library.search(
        conn, "equivariant force fields", embedder=fake_embedder, author="musaelian", limit=1
    )
    assert res.semantic is True and [h.id for h in res.hits] == [c]
    none = library.search(conn, "equivariant force fields", embedder=fake_embedder, year=1999)
    assert none.semantic is False and none.hits == [] and "no semantic hit" in none.caveat


def test_literal_boost_counts_whole_tokens_only(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, bib_key="a", title="Diffusion models for docking")
    b = _store_with(conn, bib_key="b", title="Ion transport in solids")
    library.embed_missing(conn, fake_embedder, None)
    q = fake_embedder.embed_query("ion channels")
    for rid in (a, b):
        conn.execute("DELETE FROM reference_vectors WHERE rowid = ?", (rid,))
        conn.execute(
            "INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)", (rid, q.tobytes())
        )
    res = library.search(conn, "ion channels", embedder=fake_embedder, limit=2)
    assert [h.id for h in res.hits] == [b, a]  # "ion" inside "diffusion" earns nothing


def test_substring_fallback_is_word_and_not_phrase(tmp_path):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="mace", title="MACE: higher order equivariant message passing")
    _store_with(conn, bib_key="bread", title="Sourdough", abstract="equivariant nothing")
    hits = library.search(conn, "equivariant message passing").hits
    assert [h.bib_key for h in hits] == ["mace"]
    assert [h.bib_key for h in library.search(conn, "passing equivariant").hits] == ["mace"]


def test_embed_missing_does_not_hold_a_transaction_across_embed_calls(tmp_path):
    conn = get_db(tmp_path / "t.db")
    for k in "abc":
        _store_with(conn, bib_key=k, title=k)
    conn.commit()
    seen = []

    class E:
        def embed_document(self, t, x):
            seen.append(conn.in_transaction)
            return np.zeros(embed_dims(), dtype=np.float32)

    library.embed_missing(conn, E(), None)
    assert seen == [False, False, False]


# ---------------------------------------------------------------------------
# the feed's own papers count once; a typo in --sources raises
# ---------------------------------------------------------------------------


def test_a_feed_derived_reference_is_not_counted_twice(tmp_path):
    from attestation import kg
    from attestation.features import tag_vocabulary

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO items(id, title, content_hash) VALUES (1, 'i', 'h')")
    conn.executemany("INSERT INTO item_tags VALUES (1, ?)", [("a",), ("b",)])
    rid = _store_with(conn, source="feed", source_key="1", title="i", arxiv_id="2106.02347")
    conn.executemany("INSERT INTO reference_tags VALUES (?, ?)", [(rid, "a"), (rid, "b")])
    other = _store_with(conn, bib_key="o", title="o", tags=["c"])
    assert sorted(kg.tag_assignments(conn)) == [(-other, "c"), (1, "a"), (1, "b")]
    adjacency, _ = kg.build_graph(kg.tag_assignments(conn))
    assert adjacency == {}  # one paper cannot make an edge on its own
    assert tag_vocabulary(conn) == ["a", "b", "c"]


def test_unknown_source_names_raise(tmp_path):
    conn = get_db(tmp_path / "t.db")
    with pytest.raises(ValueError, match="unknown source\\(s\\) bib"):
        library_readers.readers_from_env(
            conn, bib_paths=[], zotero_path=tmp_path / "n", sources=["bib"]
        )
    assert library_readers.unarmed(["s2", "feed"]) == ["s2"]
