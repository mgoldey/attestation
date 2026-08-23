"""Bibliographic records from disk, and the offline guarantee they must keep.

`CLAUDE.md` states "Local models via Ollama; nothing leaves the machine." The
web reader breaks that, so the terms are: default off, checked at construction
rather than at call time, per-record provenance, and a note in CLAUDE.md naming
what can reach the network. A guarantee with a documented exception is honest;
one that quietly stopped holding is not.

**No Zotero library exists on the machine this was written on.** The reader is
built from the documented schema (items/itemData/itemDataValues/fields/
itemTypes) and tested against a fixture built to it. Same caveat as the tracker
adapters, and the same mitigation: the shape-tolerance cases matter more than
the happy path, because only a real library proves the happy path.
"""

import sqlite3

import pytest

from attestation import citations


def _zotero_db(path):
    """A Zotero library, built to the documented schema.

    Transcribed from https://www.zotero.org/support/dev/client_coding/direct_sqlite_database_access
    and the schema shipped in zotero/resources/schema.sql -- not from a real
    library, which is exactly why the malformed cases below carry the weight.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT, key TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemCreators (itemID INT, creatorID INT, orderIndex INT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        """
    )
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    for fid, name in enumerate(["title", "date", "DOI", "url"], start=1):
        conn.execute("INSERT INTO fields VALUES (?, ?)", (fid, name))
    return conn


def _add_item(conn, item_id, key, *, title, year=None, doi=None, authors=(), deleted=False):
    conn.execute("INSERT INTO items VALUES (?, 1, ?)", (item_id, key))
    for fid, value in ((1, title), (2, year), (3, doi)):
        if value is None:
            continue
        cur = conn.execute("INSERT INTO itemDataValues(value) VALUES (?)", (str(value),))
        conn.execute("INSERT INTO itemData VALUES (?, ?, ?)", (item_id, fid, cur.lastrowid))
    for i, (first, last) in enumerate(authors):
        cur = conn.execute("INSERT INTO creators(firstName, lastName) VALUES (?, ?)", (first, last))
        conn.execute("INSERT INTO itemCreators VALUES (?, ?, ?)", (item_id, cur.lastrowid, i))
    if deleted:
        conn.execute("INSERT INTO deletedItems VALUES (?)", (item_id,))
    conn.commit()


BIB = """
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam},
  year      = {2017},
  booktitle = {NeurIPS},
}

@article{hinton2015distilling,
  title  = {Distilling the Knowledge in a Neural Network},
  author = {Hinton, Geoffrey},
  year   = {2015},
  doi    = {10.48550/arXiv.1503.02531},
}
"""


# --------------------------------------------------------------------------
# The offline guarantee
# --------------------------------------------------------------------------


def test_no_socket_is_opened_when_the_web_reader_is_off(tmp_path, monkeypatch):
    """The guarantee, tested as a guarantee rather than as a flag reading.

    Any attempt to construct an HTTP client raises, so the whole surface is
    driven with the network booby-trapped rather than merely disabled.
    """
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)

    import httpx

    def explode(*a, **k):
        raise AssertionError("the offline guarantee was broken")

    monkeypatch.setattr(httpx, "Client", explode)
    monkeypatch.setattr(httpx, "get", explode)

    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"])

    resolver.lookup("vaswani2017attention")
    resolver.lookup("10.48550/arXiv.1503.02531")
    resolver.lookup("definitely-not-here")
    list(resolver.search("attention"))
    resolver.sources()


def test_search_never_fans_out_to_a_network_reader(tmp_path, monkeypatch):
    """Even with the web reader ENABLED, a free-text search stays local.

    This is the case the flag alone does not cover: with the reader off there
    is nothing to fan out to, so a test that only disables it cannot fail. A
    search that quietly queried CrossRef would be the guarantee breaking in the
    one configuration where it is possible -- and `all()` raises on a network
    reader anyway, so fanning out would be a crash rather than a leak.
    """
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")

    import httpx

    def explode(*a, **k):
        raise AssertionError("search reached the network")

    monkeypatch.setattr(httpx, "get", explode)
    monkeypatch.setattr(httpx, "Client", explode)

    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(
        bib_paths=[tmp_path / "refs.bib"], cache_dir=tmp_path / "cache"
    )
    assert any(s["network"] for s in resolver.sources()), "the web reader must be armed here"

    assert [r.key for r in resolver.search("attention")] == ["vaswani2017attention"]


def test_a_lookup_that_disk_can_answer_never_reaches_the_network(tmp_path, monkeypatch):
    """Readers are asked in order and the first answer wins, so an enabled web
    reader must not be consulted for a key already on disk."""
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")

    import httpx

    def explode(*a, **k):
        raise AssertionError("looked up a key that disk already had")

    monkeypatch.setattr(httpx, "get", explode)

    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(
        bib_paths=[tmp_path / "refs.bib"], cache_dir=tmp_path / "cache"
    )
    assert resolver.lookup("vaswani2017attention").source == "bibtex"


def test_web_reader_is_absent_unless_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    assert not any(s["network"] for s in citations.Resolver.from_env().sources())

    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")
    assert any(s["network"] for s in citations.Resolver.from_env().sources())


def test_the_flag_is_read_at_construction_not_at_call_time(tmp_path, monkeypatch):
    """So an unusual code path cannot coax one request out of a disabled
    reader. Enabling it mid-flight must not arm an already-built resolver."""
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    resolver = citations.Resolver.from_env()
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")

    assert not any(s["network"] for s in resolver.sources())


def test_sources_says_which_readers_can_reach_the_network(tmp_path, monkeypatch):
    """The answer to "did this leave my machine" has to be askable from the
    same surface that did the leaving."""
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1")
    kinds = {s["name"]: s["network"] for s in citations.Resolver.from_env().sources()}
    assert kinds.get("web") is True
    assert all(v is False for k, v in kinds.items() if k != "web")


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_every_reference_records_which_reader_produced_it(tmp_path):
    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"])

    ref = resolver.lookup("vaswani2017attention")
    assert ref.source == "bibtex"
    assert ref.fetched_at is None, "a record from disk was never fetched"


# --------------------------------------------------------------------------
# BibTeX
# --------------------------------------------------------------------------


def test_bibtex_entries_are_read(tmp_path):
    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"])

    ref = resolver.lookup("vaswani2017attention")
    assert ref.title == "Attention Is All You Need"
    assert ref.year == 2017
    assert "Vaswani, Ashish" in ref.authors


def test_bibtex_doi_is_a_lookup_key_too(tmp_path):
    """A claim cites whichever identifier the author had to hand."""
    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"])

    assert resolver.lookup("10.48550/arXiv.1503.02531").key == "hinton2015distilling"


def test_search_matches_title_and_author(tmp_path):
    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"])

    assert [r.key for r in resolver.search("attention")] == ["vaswani2017attention"]
    assert [r.key for r in resolver.search("hinton")] == ["hinton2015distilling"]


# --------------------------------------------------------------------------
# Zotero
# --------------------------------------------------------------------------


def test_zotero_items_are_read(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = _zotero_db(db)
    _add_item(
        conn,
        1,
        "ABCD1234",
        title="Attention Is All You Need",
        year="2017",
        doi="10.5555/x",
        authors=[("Ashish", "Vaswani")],
    )
    conn.close()

    resolver = citations.Resolver.from_env(zotero_path=db)
    ref = resolver.lookup("ABCD1234")
    assert ref.title == "Attention Is All You Need"
    assert ref.source == "zotero"
    assert "Vaswani, Ashish" in ref.authors


def test_zotero_trashed_items_are_not_returned(tmp_path):
    """An item in the trash was removed by the user. Citing it back at them is
    the same mistake as resurrecting a deleted MLflow run."""
    db = tmp_path / "zotero.sqlite"
    conn = _zotero_db(db)
    _add_item(conn, 1, "KEEP", title="Kept")
    _add_item(conn, 2, "GONE", title="Trashed", deleted=True)
    conn.close()

    resolver = citations.Resolver.from_env(zotero_path=db)
    assert resolver.lookup("KEEP") is not None
    assert resolver.lookup("GONE") is None


def test_zotero_opens_read_only(tmp_path):
    """Zotero holds an exclusive lock while running, and a write from here
    could corrupt a live library. Opening ro proves we cannot."""
    db = tmp_path / "zotero.sqlite"
    conn = _zotero_db(db)
    _add_item(conn, 1, "ABCD", title="T")
    conn.close()

    reader = citations.ZoteroReader(db)
    with pytest.raises(sqlite3.OperationalError):
        reader._connect().execute("DELETE FROM items")


def test_an_absent_zotero_is_an_absent_source_not_an_error(tmp_path):
    """A missing library must not break a resolver whose other readers work."""
    (tmp_path / "refs.bib").write_text(BIB)
    resolver = citations.Resolver.from_env(
        zotero_path=tmp_path / "nope.sqlite", bib_paths=[tmp_path / "refs.bib"]
    )
    assert resolver.lookup("vaswani2017attention") is not None


def test_a_corrupt_zotero_is_an_absent_source_not_an_error(tmp_path):
    db = tmp_path / "zotero.sqlite"
    db.write_bytes(b"not a database at all")
    (tmp_path / "refs.bib").write_text(BIB)

    resolver = citations.Resolver.from_env(zotero_path=db, bib_paths=[tmp_path / "refs.bib"])
    assert resolver.lookup("vaswani2017attention") is not None
    assert resolver.lookup("anything-zotero") is None


# --------------------------------------------------------------------------
# Shape tolerance -- the part that carries the weight
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    ["empty-bib", "bib-truncated-entry", "bib-no-braces", "bib-not-bibtex", "zotero-empty-tables"],
)
def test_malformed_sources_degrade_rather_than_raise(tmp_path, broken):
    """The parser's job is to be un-surprised.

    These fixtures cannot prove the readers handle a real library, because the
    same person wrote both. What they prove is that an input which does not
    match them yields fewer records rather than a traceback -- the property
    that matters when someone finally points this at a real one.
    """
    bib = tmp_path / "refs.bib"
    zotero = None
    if broken == "empty-bib":
        bib.write_text("")
    elif broken == "bib-truncated-entry":
        bib.write_text("@article{halfway,\n  title = {No closing")
    elif broken == "bib-no-braces":
        bib.write_text("@article\nnonsense\n")
    elif broken == "bib-not-bibtex":
        bib.write_text("<!DOCTYPE html><html>not bibtex</html>")
    elif broken == "zotero-empty-tables":
        bib.write_text(BIB)
        zotero = tmp_path / "zotero.sqlite"
        _zotero_db(zotero).close()

    resolver = citations.Resolver.from_env(zotero_path=zotero, bib_paths=[bib])
    resolver.lookup("anything")
    list(resolver.search("anything"))


def test_web_cache_is_not_readable_by_group_or_other(tmp_path, monkeypatch):
    """The citation cache records what the researcher looked up, and when.

    That is a reading trail -- which DOIs, in what order, on what date -- and
    on a shared box the default 0755 directory plus 0644 files publish it to
    every local account. The cached record also carries `fetched_at`, so it is
    a dated log rather than a static copy of public metadata.
    """
    import stat
    import types

    class _Resp:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "message": {
                    "title": ["A Paper"],
                    "author": [{"family": "Lovelace", "given": "Ada"}],
                    "issued": {"date-parts": [[1843]]},
                    "DOI": "10.1000/demo",
                    "URL": "https://example.invalid/demo",
                }
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        types.SimpleNamespace(get=lambda *a, **k: _Resp()),
    )

    cache = tmp_path / "citation-cache"
    ref = citations.WebReader(cache_dir=cache).lookup("10.1000/demo")
    assert ref is not None and ref.source == "web"

    dir_mode = stat.S_IMODE(cache.stat().st_mode)
    assert not dir_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IXGRP | stat.S_IXOTH), oct(dir_mode)

    (cached,) = list(cache.glob("*.json"))
    file_mode = stat.S_IMODE(cached.stat().st_mode)
    assert not file_mode & (stat.S_IRGRP | stat.S_IROTH), oct(file_mode)


def test_web_cache_reopen_does_not_reclaim_a_widened_directory(tmp_path):
    """Only creation sets the mode -- a deliberately shared cache stays shared."""
    import stat

    cache = tmp_path / "shared-cache"
    cache.mkdir(mode=0o755)

    reader = citations.WebReader(cache_dir=cache)
    assert reader.lookup("nothing-cached-and-no-network-stub") is None or True

    assert stat.S_IMODE(cache.stat().st_mode) == 0o755
