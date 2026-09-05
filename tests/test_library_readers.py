"""library_readers: the offline readers and, behind flags, the enrichers."""

from pathlib import Path

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
    assert s.source == f"bibtex:{FIX / 'sample.bib'}" and s.fetched_at is None
    # "and others" is a truncation marker, not an author. LaTeX escapes are
    # kept verbatim: de-escaping is a rendering concern the citations spec
    # keeps at the presentation edge.
    assert s.authors == ['Sch{\\"u}tt, Kristof T.', "Kindermans, Pieter-Jan", "Sauceda, Huziel E."]
    n = recs["batzner2022nequip"]
    assert n.doi == "10.1038/s41467-022-29939-5" and n.venue == "Nature Communications"


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
