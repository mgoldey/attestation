"""library.py: identity, merge, upsert, sync, search."""

from pathlib import Path

import httpx
import pytest

from attestation import library

FIX = Path(__file__).parent / "fixtures" / "library"


@pytest.mark.parametrize(
    ("doi", "arxiv", "title", "year", "want"),
    [
        ("10.1038/S41467-026-74391-4", None, "x", 2026, "doi:10.1038/s41467-026-74391-4"),
        ("https://doi.org/10.1000/ABC", None, "x", 2020, "doi:10.1000/abc"),
        ("doi:10.1000/abc", None, "x", 2020, "doi:10.1000/abc"),
        # DOI beats arXiv: a published preprint keeps its arXiv id but gains a DOI.
        ("10.1000/abc", "2106.02347v3", "x", 2021, "doi:10.1000/abc"),
        (None, "2106.02347v3", "x", 2021, "arxiv:2106.02347"),
        (None, "arXiv:2106.02347", "x", 2021, "arxiv:2106.02347"),
        (None, "cond-mat/0301234", "x", 2003, "arxiv:cond-mat/0301234"),
        (
            None,
            None,
            "SchNet: A continuous-filter CNN",
            2017,
            "title:schnet a continuous filter cnn:2017",
        ),
        (None, None, "  Équivariant  Force-Fields! ", None, "title:equivariant force fields:-"),
    ],
)
def test_identity_prefers_doi_then_arxiv_then_title(doi, arxiv, title, year, want):
    assert library.identity(doi, arxiv, title, year) == want


def test_identity_needs_something():
    with pytest.raises(ValueError):
        library.identity(None, None, "", None)


def test_merge_fills_empty_keeps_first_and_records_conflicts():
    existing = {"title": "SchNet", "abstract": None, "year": 2017, "authors": ["Schütt, K."]}
    incoming = {
        "title": "SchNet: a CNN",
        "abstract": "We present...",
        "year": 2018,
        "authors": ["Schütt, K.", "Kindermans, P."],
    }
    merged, conflicts = library.merge(existing, incoming)
    assert merged["abstract"] == "We present..."  # filled
    assert merged["title"] == "SchNet"  # kept
    assert merged["year"] == 2017  # kept
    assert conflicts["title"] == {"kept": "SchNet", "offered": "SchNet: a CNN"}
    assert conflicts["year"] == {"kept": 2017, "offered": 2018}
    # A longer author list EXTENDS rather than conflicts (a .bib truncated with "and others").
    assert merged["authors"] == ["Schütt, K.", "Kindermans, P."]
    assert "authors" not in conflicts


def test_merge_author_disagreement_is_a_conflict():
    merged, conflicts = library.merge({"authors": ["A, B"]}, {"authors": ["C, D"]})
    assert merged["authors"] == ["A, B"]
    assert conflicts["authors"] == {"kept": ["A, B"], "offered": ["C, D"]}


# ---------------------------------------------------------------------------
# upsert and sync
# ---------------------------------------------------------------------------


def _rec(**kw):
    kw.setdefault("source", "bibtex:/a.bib")
    kw.setdefault("source_key", kw.get("bib_key", "k"))
    return library.ReferenceRecord(**kw)


def test_upsert_merges_zotero_and_bib_under_one_row(tmp_path):
    import json

    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid1, how1 = library.upsert(
        conn,
        _rec(
            bib_key="schnet",
            title="SchNet",
            year=2017,
            doi="10.5555/schnet",
            authors=["Schütt, K."],
        ),
    )
    rid2, how2 = library.upsert(
        conn,
        _rec(
            source="zotero",
            source_key="ABCD1234",
            bib_key="ABCD1234",
            title="SchNet: a CNN",
            doi="10.5555/SCHNET",
            abstract="We present",
            authors=["Schütt, K.", "Kindermans, P."],
        ),
    )
    assert rid1 == rid2 and (how1, how2) == ("added", "merged")
    row = conn.execute('SELECT * FROM "references"').fetchone()
    assert row["identity"] == "doi:10.5555/schnet"
    assert row["title"] == "SchNet" and row["abstract"] == "We present"
    assert row["bib_key"] == "schnet"  # first key seen
    assert json.loads(row["authors"]) == [
        "Schütt, K.",
        "Kindermans, P.",
    ]  # extended, not conflicted
    sources = conn.execute(
        "SELECT source, source_key, raw FROM reference_sources ORDER BY source"
    ).fetchall()
    assert [(s["source"], s["source_key"]) for s in sources] == [
        ("bibtex:/a.bib", "schnet"),
        ("zotero", "ABCD1234"),
    ]
    assert json.loads(sources[1]["raw"])["conflicts"]["title"]["offered"] == "SchNet: a CNN"


def test_upsert_finds_a_row_by_arxiv_id_and_upgrades_its_identity(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(
        conn, _rec(source="feed", source_key="9", title="NequIP", arxiv_id="2106.02347v2")
    )
    assert conn.execute('SELECT identity FROM "references"').fetchone()[0] == "arxiv:2106.02347"
    rid2, how = library.upsert(
        conn,
        _rec(
            bib_key="nequip",
            title="NequIP",
            arxiv_id="2106.02347",
            doi="10.1038/s41467-022-29939-5",
        ),
    )
    assert rid2 == rid and how == "merged"
    ident = conn.execute('SELECT identity FROM "references"').fetchone()[0]
    assert ident == "doi:10.1038/s41467-022-29939-5"


def test_upsert_records_citation_edges(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(
        conn,
        _rec(
            source="s2",
            source_key="doi:x",
            title="Citing",
            doi="10.1/x",
            fetched_at="2026-09-05",
            cites=[("doi:10.5555/schnet", "SchNet"), ("title:untraceable:-", "Untraceable")],
        ),
    )
    rows = conn.execute(
        "SELECT cited_identity, cited_title, source FROM reference_cites WHERE citing_id = ?"
        " ORDER BY cited_identity",
        (rid,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("doi:10.5555/schnet", "SchNet", "s2"),
        ("title:untraceable:-", "Untraceable", "s2"),
    ]


def test_upsert_writes_tags_from_the_record_without_deleting_others(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(conn, _rec(bib_key="k", title="K", tags=["force-fields"]))
    conn.execute("INSERT INTO reference_tags VALUES (?, 'from-the-tagger')", (rid,))
    library.upsert(
        conn, _rec(source="zotero", source_key="Z", bib_key="Z", title="K", tags=["gnn"])
    )
    tags = {
        r["tag"]
        for r in conn.execute("SELECT tag FROM reference_tags WHERE reference_id = ?", (rid,))
    }
    assert tags == {"force-fields", "from-the-tagger", "gnn"}


def test_sync_is_idempotent(tmp_path):
    from attestation.db import get_db
    from attestation.library_readers import BibtexRecords

    conn = get_db(tmp_path / "t.db")
    readers = [BibtexRecords([FIX / "sample.bib"])]
    first = library.sync(conn, readers).to_dict()
    second = library.sync(conn, readers).to_dict()
    assert first["sources"]["bibtex"] == {
        "seen": 2,
        "added": 2,
        "merged": 0,
        "unchanged": 0,
        "enriched": 0,
        "failed": 0,
    }
    assert second["sources"]["bibtex"]["unchanged"] == 2
    assert second["sources"]["bibtex"]["added"] == 0
    assert first["unembedded"] == 2  # no embedder given: reported, not an error
    assert library.status(conn)["references"] == 2
    assert library.status(conn)["sources"] == {f"bibtex:{FIX / 'sample.bib'}": 2}


# ---------------------------------------------------------------------------
# embeddings and search
# ---------------------------------------------------------------------------


def _store_with(conn, **kw):
    kw.setdefault("source", "bibtex:/a.bib")
    kw.setdefault("source_key", kw.get("bib_key", "k"))
    return library.upsert(conn, library.ReferenceRecord(**kw))[0]


def test_search_is_semantic_with_an_embedder_and_says_so(tmp_path, fake_embedder):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    a = _store_with(
        conn,
        bib_key="a",
        title="E(3)-equivariant interatomic potentials",
        abstract="force fields",
        year=2022,
    )
    _store_with(
        conn, bib_key="b", title="Sourdough starter maintenance", abstract="bread", year=2019
    )
    embedded, missing, err = library.embed_missing(conn, fake_embedder, None)
    assert (embedded, missing, err) == (2, 0, None)
    # FakeEmbedder is hash-based, so plant the query vector on `a` to make the
    # ranking deterministic: what is tested is the plumbing (KNN + floor +
    # envelope), not embeddinggemma.
    q = fake_embedder.embed_query("equivariant force fields")
    conn.execute("DELETE FROM reference_vectors WHERE rowid = ?", (a,))
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)", (a, q.tobytes()))
    res = library.search(conn, "equivariant force fields", embedder=fake_embedder, limit=5)
    assert res.semantic is True and res.caveat is None
    assert res.hits[0].id == a and res.hits[0].similarity > 0.99
    assert res.hits[0].to_row()["sources"] == ["bibtex:/a.bib"]
    assert res.hits[0].to_row()["key"] == "a"


def test_search_without_an_embedder_is_fielded_and_says_so(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="a", title="SchNet", year=2017, authors=["Schütt, Kristof"])
    _store_with(conn, bib_key="b", title="NequIP", year=2022, authors=["Batzner, Simon"])
    res = library.search(conn, "schn")
    assert res.semantic is False and "no embedder" in res.caveat
    assert [h.bib_key for h in res.hits] == ["a"]
    assert [h.bib_key for h in library.search(conn, "", author="batzner").hits] == ["b"]
    assert [h.bib_key for h in library.search(conn, "", year_from=2020).hits] == ["b"]
    assert [h.bib_key for h in library.search(conn, "", year=2017).hits] == ["a"]


def test_an_identifier_query_is_a_direct_lookup(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    _store_with(
        conn,
        bib_key="n",
        title="NequIP",
        doi="10.1038/s41467-022-29939-5",
        arxiv_id="2101.03164",
    )
    for q in (
        "10.1038/S41467-022-29939-5",
        "arXiv:2101.03164v1",
        "N",
        "doi:10.1038/s41467-022-29939-5",
    ):
        res = library.search(conn, q)
        assert [h.bib_key for h in res.hits] == ["n"], q
        assert res.n_matches == 1


def test_an_unreachable_embedder_degrades_to_substring(tmp_path):
    from attestation.db import get_db

    class Dead:
        def embed_query(self, text):
            raise httpx.ConnectError("refused")

        def embed_document(self, title, text):
            raise httpx.ConnectError("refused")

    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="a", title="SchNet", year=2017)
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (1, ?)", (b"\x00" * 1024,))
    done, missing, err = library.embed_missing(conn, Dead(), None)
    assert (done, missing) == (0, 0) and err is None  # nothing to embed
    res = library.search(conn, "schnet", embedder=Dead())
    assert res.semantic is False and [h.bib_key for h in res.hits] == ["a"]


def test_to_reference_keeps_the_provenance_pair(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    rid = _store_with(conn, bib_key="a", title="SchNet", doi="10.5555/schnet")
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="arxiv",
            source_key="doi:10.5555/schnet",
            doi="10.5555/schnet",
            abstract="filled",
            fetched_at="2026-09-05",
        ),
    )
    row = conn.execute('SELECT * FROM "references" WHERE id = ?', (rid,)).fetchone()
    ref = library.to_reference(conn, row)
    assert ref.key == "a" and ref.source == "library:bibtex:/a.bib" and ref.fetched_at is None
    assert ref.to_row()["doi"] == "10.5555/schnet"


# ---------------------------------------------------------------------------
# citation neighbourhood
# ---------------------------------------------------------------------------


def test_related_resolves_edges_both_ways(tmp_path):
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    nequip = _store_with(
        conn,
        bib_key="nequip",
        title="NequIP",
        doi="10.1038/s41467-022-29939-5",
        arxiv_id="2101.03164",
    )
    schnet = _store_with(conn, bib_key="schnet", title="SchNet", arxiv_id="1706.08566")
    # An edge recorded by arXiv id must still find SchNet after it gains a DOI.
    library.upsert(
        conn,
        _rec(
            source="s2",
            source_key="x",
            doi="10.1038/s41467-022-29939-5",
            fetched_at="2026-09-05",
            cites=[("arxiv:1706.08566", "SchNet"), ("title:elsewhere:-", "Elsewhere")],
        ),
    )
    library.upsert(
        conn, _rec(bib_key="schnet2", title="SchNet", arxiv_id="1706.08566", doi="10.5555/schnet")
    )
    rel = library.related(conn, "nequip")
    assert rel is not None and rel.reference.id == nequip
    assert [(n.identity, n.in_library, n.key) for n in rel.cites] == [
        ("arxiv:1706.08566", True, "schnet"),
        ("title:elsewhere:-", False, None),
    ]
    assert rel.n_cites == 2 and rel.cited_by == [] and rel.n_cited_by == 0
    back = library.related(conn, "schnet")
    assert back is not None and back.reference.id == schnet
    assert [n.key for n in back.cited_by] == ["nequip"]
    assert library.related(conn, "nope") is None
    row = rel.to_row()
    assert row["reference"]["key"] == "nequip" and row["cites"][1]["title"] == "Elsewhere"
