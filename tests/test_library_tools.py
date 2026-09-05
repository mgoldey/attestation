"""The cite.* tools over the library store: envelopes, fallbacks, payload size."""

import json

from test_response_size import HARD_RESPONSE_CEILING

from attestation.db import get_db
from attestation.library import ReferenceRecord, upsert
from attestation.mcp import citation


def _db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    return db


def test_cite_lookup_shows_every_source_and_the_conflicts(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    conn = get_db(db)
    upsert(
        conn,
        ReferenceRecord(
            source="bibtex:/a.bib",
            source_key="schnet",
            bib_key="schnet",
            title="SchNet",
            doi="10.5555/schnet",
            year=2017,
        ),
    )
    upsert(
        conn,
        ReferenceRecord(
            source="zotero",
            source_key="Z1",
            bib_key="Z1",
            title="SchNet: a CNN",
            doi="10.5555/schnet",
            year=2018,
        ),
    )
    conn.commit()
    conn.close()
    out = citation._lookup("10.5555/schnet")
    assert out["ok"] is True
    assert (
        out["reference"]["key"] == "schnet"
        and out["reference"]["source"] == "library:bibtex:/a.bib"
    )
    assert [s["source"] for s in out["sources"]] == ["bibtex:/a.bib", "zotero"]
    assert out["conflicts"]["zotero"]["year"] == {"kept": 2017, "offered": 2018}


def test_cite_lookup_falls_back_to_the_disk_readers(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    (tmp_path / "refs.bib").write_text(
        "@article{disk,\n  title = {On disk},\n  year = {2020},\n}\n"
    )
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "refs.bib"))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "none.sqlite"))
    out = citation._lookup("disk")
    assert out["ok"] is True and out["reference"]["source"] == "bibtex" and out["sources"] == []
    missing = citation._lookup("nope")
    assert missing["ok"] is False and "library store: 0 references" in missing["message"]


def test_cite_search_on_an_empty_store_says_so(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "absent.bib"))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "none.sqlite"))
    out = citation._search("anything", 5)
    assert out["ok"] is True and out["semantic"] is False
    assert "library store is empty" in out["caveat"]


def test_cite_search_is_bounded_at_the_cap(tmp_path, monkeypatch):
    """The worst realistic row, 30 times, at the cap: still under the ceiling
    a 2B model can render (test_response_size's measured 7000 characters)."""
    db = _db(tmp_path, monkeypatch)
    conn = get_db(db)
    long_title = "Retraction Note: " + "Photocatalytic Degradation of Organic Pollutants " * 4
    for i in range(30):
        rid = upsert(
            conn,
            ReferenceRecord(
                source="bibtex:/a.bib",
                source_key=f"k{i}",
                bib_key=f"k{i}",
                title=f"{long_title} {i}",
                year=2020,
                authors=[f"Author{j}, Name" for j in range(12)],
                abstract="pollutants " * 200,
                doi=f"10.1/{i}",
                venue="A Journal With A Very Long Name Indeed",
            ),
        )[0]
        conn.executemany(
            "INSERT INTO reference_tags VALUES (?, ?)", [(rid, f"tag-{j}") for j in range(6)]
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(citation, "_embedder", lambda: None)
    out = citation._search("pollutants", 13)
    assert out["ok"] is True and len(out["references"]) == 13 and out["n_matches"] == 30
    assert len(json.dumps(out)) <= HARD_RESPONSE_CEILING, len(json.dumps(out))
    row = out["references"][0]
    assert len(row["authors"]) == 6 and row["n_authors"] == 12
    assert len(row["tags"]) == 3 and row["n_tags"] == 6


def test_cite_sources_reports_the_store_and_the_s2_flag(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "absent.bib"))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "none.sqlite"))
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    monkeypatch.delenv("ATTEST_CITATION_SCHOLAR", raising=False)
    out = citation._sources()
    assert out["offline"] is True and out["store"]["references"] == 0
    monkeypatch.setenv("ATTEST_CITATION_SCHOLAR", "1")
    armed = citation._sources()
    assert armed["offline"] is False and {"name": "s2", "network": True} in armed["sources"]


def test_cite_sync_reads_a_bib_and_reports_structure(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@article{schnet,\n  title = {SchNet},\n  author = {Schütt, K.},\n  year = {2017},\n"
        "  doi = {10.5555/schnet},\n}\n"
    )
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(bib))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "none.sqlite"))
    monkeypatch.setattr(citation, "_embedder", lambda: None)
    out = citation._sync(["bibtex"], None)
    assert out["ok"] is True
    assert out["sources"]["bibtex"]["added"] == 1 and out["unembedded"] == 1
    assert out["message"].startswith("bibtex: +1 added")
    again = citation._sync(["bibtex"], None)
    assert again["sources"]["bibtex"]["unchanged"] == 1


def test_cite_check_resolves_through_the_store(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    conn = get_db(db)
    upsert(conn, ReferenceRecord(source="zotero", source_key="Z1", bib_key="Z1", title="Stored"))
    conn.commit()
    conn.close()
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "absent.bib"))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "none.sqlite"))
    draft = tmp_path / "draft.md"
    draft.write_text(
        "Stored says so. <!-- claim: proj/run metric=wer value=1 cite=Z1 -->\n"
        "Nobody says this. <!-- claim: proj/run metric=wer value=2 cite=ghost -->\n"
    )
    out = citation._check(str(draft))
    assert out["ok"] is True
    assert [u["key"] for u in out["uncited"]] == ["ghost"]
