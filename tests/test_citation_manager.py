"""The citation manager: BibTeX round trip, CSL formatting, edits, merges,
imports, cited_by, the dashboard fields, repeated_keys and cite.format.

No model and no network: citeproc-py renders from the bundled CSL styles and
its own bundled locale, which `test_formatting_needs_no_network` proves by
making every socket connection fail.
"""

import json
import socket

import pytest

from attestation import (
    bibtex_io,
    citation_manager,
    cite_format,
    dashboard,
    latex,
    library,
    library_readers,
)
from attestation.db import get_db

BIB = r"""@string{nc = "Nature Communications"}
@article{batzner2022,
  author = {Batzner, Simon and Sch{\"u}tt, Kristof},
  title = {{E(3)}-equivariant graph neural networks},
  journal = nc,
  year = 2022,
  volume = {13},
  number = {1},
  pages = {2453--2460},
  doi = {10.1038/s41467-022-29939-5},
  publisher = {Springer Nature},
  custom-field = {kept verbatim},
  file = {/home/someone/papers/batzner.pdf}
}
@inproceedings{vaswani2017,
  title = "Attention is all you need",
  author = {Vaswani, Ashish and Shazeer, Noam},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  pages = {5998--6008}
}
@book{bishop2006,
  title = {Pattern Recognition and Machine Learning},
  author = {Bishop, Christopher M.},
  publisher = {Springer},
  address = {New York},
  year = {2006}
}
@misc{doe2021preprint,
  title = {A preprint about diffusion},
  author = {Doe, Jane},
  year = {2021},
  eprint = {2101.03164},
  archivePrefix = {arXiv}
}
@misc{openai2023,
  title = {Some report},
  author = {{OpenAI}},
  year = {2023},
  url = {https://example.org/report}
}
"""


@pytest.fixture
def lib(tmp_path):
    """A store synced from BIB, as `attest library sync` would."""
    bib = tmp_path / "uploads" / "refs.bib"
    bib.parent.mkdir()
    bib.write_text(BIB)
    conn = get_db(tmp_path / "t.db")
    library.sync(conn, [library_readers.BibtexRecords([bib])])
    yield conn, bib
    conn.close()


def _row(conn, key):
    return citation_manager.require(conn, key)


# --------------------------------------------------------------------------- BibTeX


def test_the_parser_reads_what_the_regex_reader_missed():
    parsed = bibtex_io.parse(BIB + "@misc{broken, title = {never closed\n")
    by_key = {e.key: e for e in parsed.entries}
    assert by_key["batzner2022"].fields["journal"] == "Nature Communications"  # @string
    assert by_key["batzner2022"].fields["year"] == "2022"  # unbraced
    assert by_key["batzner2022"].entry_type == "article"
    assert by_key["vaswani2017"].fields["title"] == "Attention is all you need"  # quoted
    assert parsed.failed and parsed.failed[0].startswith("line ")


def test_a_key_defined_twice_keeps_the_first_and_is_reported():
    parsed = bibtex_io.parse("@misc{a, title={First}}\n@misc{a, title={Second}}\n")
    assert [e.fields["title"] for e in parsed.entries] == ["First"]
    assert parsed.duplicate_keys == ["a"]


def test_an_imported_entry_round_trips_with_the_fields_it_carried(lib):
    conn, _ = lib
    row = _row(conn, "batzner2022")
    text = library.bibtex(row)
    back = bibtex_io.parse(text).entries[0]
    assert back.key == "batzner2022" and back.entry_type == "article"
    original = bibtex_io.parse(BIB).entries[0].fields
    for name in ("author", "title", "journal", "year", "volume", "number", "pages", "doi"):
        assert back.fields[name] == original[name], name
    assert back.fields["publisher"] == "Springer Nature"
    assert back.fields["custom-field"] == "kept verbatim"
    assert "file" not in back.fields, "a machine path must not travel into exports"
    assert library.bibtex(row) == text  # deterministic


def test_entry_types_and_venue_fields_survive(lib):
    conn, _ = lib
    vaswani = bibtex_io.parse(library.bibtex(_row(conn, "vaswani2017"))).entries[0]
    assert vaswani.entry_type == "inproceedings"
    assert vaswani.fields["booktitle"].startswith("Advances in Neural")
    preprint = bibtex_io.parse(library.bibtex(_row(conn, "doe2021preprint"))).entries[0]
    assert preprint.entry_type == "misc" and preprint.fields["eprint"] == "2101.03164"


def test_plain_text_is_escaped_once_and_latex_is_left_alone(tmp_path):
    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(
        conn,
        library.ReferenceRecord(
            source="feed", source_key="1", title="Proteins & 50% of structure", year=2023
        ),
    )
    out = library.bibtex(conn.execute('SELECT * FROM "references" WHERE id=?', (rid,)).fetchone())
    assert r"Proteins \& 50\% of structure" in out
    assert library._tex(r"Sch{\"u}tt \& Co") == r"Sch{\"u}tt \& Co"


# --------------------------------------------------------------------------- CSL


def test_csl_json_for_each_entry_kind(lib):
    conn, _ = lib
    article = cite_format.to_csl(_row(conn, "batzner2022"))
    assert article["type"] == "article-journal"
    assert article["author"][1] == {"family": "Schütt", "given": "Kristof"}
    assert article["title"] == "E(3)-equivariant graph neural networks"
    assert article["container-title"] == "Nature Communications"
    assert article["page"] == "2453-2460" and article["issue"] == "1"
    assert article["issued"] == {"date-parts": [[2022]]} and article["DOI"].startswith("10.1038")
    assert cite_format.to_csl(_row(conn, "vaswani2017"))["type"] == "paper-conference"
    book = cite_format.to_csl(_row(conn, "bishop2006"))
    assert book["type"] == "book" and book["publisher"] == "Springer"
    assert book["publisher-place"] == "New York"
    pre = cite_format.to_csl(_row(conn, "doe2021preprint"))
    assert pre["type"] == "article" and pre["publisher"] == "arXiv"
    assert pre["number"] == "2101.03164" and pre["URL"] == "https://arxiv.org/abs/2101.03164"
    misc = cite_format.to_csl(_row(conn, "openai2023"))
    assert misc["type"] == "document" and misc["author"] == [{"literal": "OpenAI"}]


def test_every_bundled_style_renders_without_a_number_label(lib):
    conn, _ = lib
    row = _row(conn, "batzner2022")
    for sid in cite_format.STYLES:
        text = cite_format.format_row(row, sid)
        assert "Batzner" in text and "2022" in text, (sid, text)
        assert not text.startswith(("[", "1.")), (sid, text)
        assert text == cite_format.format_row(row, sid)  # deterministic
    assert cite_format.format_row(row, "apa").startswith("Batzner, S., & Schütt, K. (2022).")


def test_formatting_needs_no_network(lib, monkeypatch):
    conn, _ = lib

    def refuse(*args, **kwargs):
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    cite_format._style.cache_clear()
    assert cite_format.format_row(_row(conn, "bishop2006"), "vancouver").startswith("Bishop CM.")


def test_style_names_are_validated_and_the_default_is_configurable(monkeypatch):
    assert cite_format.style_id("APA") == "apa" and cite_format.style_id("nlm") == "vancouver"
    with pytest.raises(ValueError, match="unknown citation style"):
        cite_format.style_id("harvard")
    assert cite_format.default_style() == "apa"
    monkeypatch.setenv("ATTEST_CITATION_STYLE", "ieee")
    assert cite_format.default_style() == "ieee"
    monkeypatch.setenv("ATTEST_CITATION_STYLE", "typo")
    with pytest.raises(ValueError):
        cite_format.default_style()
    assert [s["id"] for s in cite_format.styles()] == list(cite_format.STYLES)


def test_every_bundled_style_file_ships_with_its_license():
    files = {p.name for p in cite_format.STYLE_DIR.glob("*.csl")}
    assert files == {f for f, _ in cite_format.STYLES.values()}
    notice = (cite_format.STYLE_DIR / "LICENSE.md").read_text()
    assert "CC BY-SA 3.0" in notice or "Attribution-ShareAlike 3.0" in notice
    for name in files:
        assert name in notice
        assert (
            "creativecommons.org/licenses/by-sa/3.0" in (cite_format.STYLE_DIR / name).read_text()
        )


def test_the_cache_serves_current_renderings_and_misses_changed_rows(lib):
    conn, _ = lib
    rows = [_row(conn, "batzner2022"), _row(conn, "bishop2006")]
    assert cite_format.cached(conn, rows, "apa") == {}
    stored = cite_format.formatted(conn, rows, "apa", store=True)
    assert cite_format.cached(conn, rows, "apa") == stored
    citation_manager.edit(conn, "bishop2006", {"title": "Pattern Recognition"})
    fresh = [_row(conn, "batzner2022"), _row(conn, "bishop2006")]
    assert set(cite_format.cached(conn, fresh, "apa")) == {fresh[0]["id"]}


# --------------------------------------------------------------------------- edits


def test_an_edit_reaches_the_owner_bib_and_never_the_upload(lib, tmp_path, monkeypatch):
    conn, upload = lib
    out_dir = tmp_path / "generated"
    monkeypatch.setenv("ATTEST_BIB_OUT", str(out_dir))
    before = upload.read_bytes()
    out = citation_manager.edit(
        conn,
        "vaswani2017",
        {"pages": "5998--6010", "publisher": "Curran Associates"},
        note="the transformer paper",
        add_tags=["Attention"],
    )
    assert out["changed"] == ["pages", "publisher"] and out["owner"] == "owner"
    assert out["cite_key"] == "vaswani2017" and out["file"] == "owner.bib"
    owner_bib = (out_dir / "owner.bib").read_text()
    assert "pages = {5998--6010}" in owner_bib and "publisher = {Curran Associates}" in owner_bib
    assert upload.read_bytes() == before, "an uploaded .bib is never written"
    rid = _row(conn, "vaswani2017")["id"]
    assert citation_manager.note_of(conn, rid) == "the transformer paper"
    assert "attention" in citation_manager.tags_of(conn, rid)
    logged = {r["field"] for r in conn.execute("SELECT field FROM reference_edits")}
    assert {"pages", "publisher", "note", "tag"} <= logged


def test_edits_refuse_stable_names_blank_titles_and_taken_ids(lib):
    conn, _ = lib
    with pytest.raises(ValueError, match="cannot be edited"):
        citation_manager.edit(conn, "bishop2006", {"key": "other"})
    with pytest.raises(ValueError, match="keep a title"):
        citation_manager.edit(conn, "bishop2006", {"title": ""})
    with pytest.raises(ValueError, match="merge"):
        citation_manager.edit(conn, "bishop2006", {"doi": "10.1038/s41467-022-29939-5"})
    with pytest.raises(LookupError):
        citation_manager.edit(conn, "no-such-key", {"title": "x"})
    assert _row(conn, "bishop2006")["doi"] is None, "a refused edit changes nothing"


def test_an_edited_doi_moves_the_identity(lib):
    conn, _ = lib
    citation_manager.edit(conn, "bishop2006", {"doi": "https://doi.org/10.1007/978-0-387"})
    row = _row(conn, "10.1007/978-0-387")
    assert row["bib_key"] == "bishop2006" and row["identity"] == "doi:10.1007/978-0-387"


# --------------------------------------------------------------------------- merge


def _split_pair(conn):
    """Two rows that are one paper, kept apart by a refused id -- the store's
    own evidence of a duplicate (a .bib entry with the arXiv id, a Zotero
    item with the DOI and the same arXiv id)."""
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="bibtex:refs.bib",
            source_key="nequip",
            bib_key="nequip",
            title="NequIP preprint",
            year=2021,
            arxiv_id="2101.03164",
        ),
    )
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="zotero",
            source_key="Z9",
            bib_key="Z9",
            title="E(3)-equivariant networks for potentials",
            year=2022,
            doi="10.5555/nequip",
        ),
    )
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="crossref",
            source_key="10.5555/nequip",
            title="E(3)-equivariant networks for potentials",
            doi="10.5555/nequip",
            arxiv_id="2101.03164",
        ),
    )
    conn.commit()


def test_duplicates_come_from_the_stores_own_evidence(tmp_path):
    conn = get_db(tmp_path / "t.db")
    _split_pair(conn)
    pairs = citation_manager.duplicates(conn)
    assert len(pairs) == 1
    assert pairs[0]["keep_key"] == "Z9" and pairs[0]["drop_key"] == "nequip"
    assert pairs[0]["why"] == "arxiv"


def test_a_merge_keeps_one_row_and_answers_to_the_old_names(tmp_path, monkeypatch):
    out_dir = tmp_path / "generated"
    monkeypatch.setenv("ATTEST_BIB_OUT", str(out_dir))
    conn = get_db(tmp_path / "t.db")
    _split_pair(conn)
    saved = citation_manager.save_for_owner(conn, _row(conn, "nequip")["id"])
    assert saved["cite_key"] == "nequip"
    out = citation_manager.merge(conn, "Z9", "nequip")
    assert out["kept"] == "Z9" and "key:nequip" in out["aliases"]
    assert conn.execute('SELECT count(*) FROM "references"').fetchone()[0] == 1
    kept = _row(conn, "nequip")  # the old key resolves through the alias
    assert kept["bib_key"] == "Z9" and kept["arxiv_id"] == "2101.03164"
    assert library.lookup_row(conn, "arxiv:2101.03164")["id"] == kept["id"]
    sources = {r["source"] for r in conn.execute("SELECT source FROM reference_sources")}
    assert {"bibtex:refs.bib", "zotero", "crossref"} <= sources
    # The owner's entry moved with its key, so a \cite{nequip} still builds.
    assert "@misc{nequip," in (out_dir / "owner.bib").read_text()
    assert citation_manager.duplicates(conn) == []


def test_a_resync_after_a_merge_lands_on_the_kept_row(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@misc{nequip, title={NequIP preprint}, year={2021}, eprint={2101.03164},"
        " archivePrefix={arXiv}}\n"
    )
    before = bib.read_bytes()
    conn = get_db(tmp_path / "t.db")
    reader = library_readers.BibtexRecords([bib])
    library.sync(conn, [reader])
    library.upsert(
        conn,
        library.ReferenceRecord(
            source="zotero", source_key="Z9", bib_key="Z9", title="Published", doi="10.5555/n"
        ),
    )
    conn.commit()
    citation_manager.merge(conn, "Z9", "nequip")
    report = library.sync(conn, [library_readers.BibtexRecords([bib])])
    assert report.sources["bibtex"]["added"] == 0
    assert conn.execute('SELECT count(*) FROM "references"').fetchone()[0] == 1
    assert bib.read_bytes() == before


def test_merging_a_row_into_itself_is_refused(lib):
    conn, _ = lib
    with pytest.raises(ValueError, match="already one reference"):
        citation_manager.merge(conn, "bishop2006", "bishop2006")


# --------------------------------------------------------------------------- import


def test_import_reports_counts_clashes_and_unreadable_entries(lib, tmp_path):
    conn, _ = lib
    other = tmp_path / "other.bib"
    other.write_text(
        "@article{bishop2006, title={A different paper entirely}, year={1999}}\n"
        "@article{fresh, title={Fresh}, year={2020}}\n"
        "@article{fresh, title={Fresh again}, year={2020}}\n"
        "@misc{notitle, year={2020}}\n"
        "@misc{broken, title = {never closed\n"
    )
    before = other.read_bytes()
    out = citation_manager.import_bib(conn, other)
    assert out["file"] == "other.bib" and out["entries"] == 3 and out["added"] == 2
    assert out["duplicate_keys"] == ["fresh"] and out["untitled"] == ["notitle"]
    assert [c["key"] for c in out["key_clashes"]] == ["bishop2006"]
    assert out["failed"] and other.read_bytes() == before


def test_import_refuses_what_it_cannot_read(lib, tmp_path):
    conn, _ = lib
    with pytest.raises(ValueError, match="no such file"):
        citation_manager.import_bib(conn, tmp_path / "absent.bib")
    junk = tmp_path / "junk.bib"
    junk.write_text("@article{x, title = {open\n")
    with pytest.raises(ValueError, match="no entry could be read"):
        citation_manager.import_bib(conn, junk)
    with pytest.raises(ValueError, match="no Zotero database"):
        citation_manager.import_zotero(conn, tmp_path / "zotero.sqlite")


# --------------------------------------------------------------------------- cited_by


def test_cited_by_scans_tex_and_markdown_drafts(lib, tmp_path):
    conn, _ = lib
    ws = tmp_path / "ws"
    (ws / "paper").mkdir(parents=True)
    (ws / ".hidden").mkdir()
    (ws / "paper" / "main.tex").write_text(
        "Intro.\n% \\cite{bishop2006} is a comment\n"
        "As shown \\citep[p.~3]{batzner2022,Vaswani2017}.\n"
    )
    (ws / "notes.md").write_text(
        "Text.\n<!-- claim: proj/run metric=acc value=0.9 cite=bishop2006 -->\n"
    )
    (ws / ".hidden" / "x.tex").write_text("\\cite{bishop2006}\n")
    found = citation_manager.cited_by(conn, ws)
    by_key = {_row(conn, k)["id"]: k for k in ("batzner2022", "vaswani2017", "bishop2006")}
    named = {by_key[rid]: places for rid, places in found.items()}
    assert named["batzner2022"] == [{"path": "paper/main.tex", "line": 3}]
    assert named["vaswani2017"] == [{"path": "paper/main.tex", "line": 3}]  # case folded
    assert named["bishop2006"] == [{"path": "notes.md", "line": 2}]


# --------------------------------------------------------------------------- dashboard


def test_the_dashboard_carries_the_citation_fields(lib, tmp_path):
    conn, _ = lib
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "draft.tex").write_text("\\cite{bishop2006}\n")
    assert dashboard.warm(conn, "apa") == 5
    doc = dashboard.build(conn, style="apa", workspace=ws)
    assert doc["default_style"] == "apa"
    assert doc["citation_styles"][0] == {"id": "apa", "name": "APA 7th edition"}
    rows = {r["key"]: r for r in doc["references"]}
    bishop = rows["bishop2006"]
    assert bishop["formatted"]["apa"].startswith("Bishop, C. M. (2006).")
    assert bishop["bibtex"].startswith("@book{bishop2006,")
    assert bishop["cited_by"] == [{"path": "draft.tex", "line": 1}]
    assert bishop["abstract"] is None and rows["batzner2022"]["cited_by"] == []
    assert doc["references_trimmed"] == 0
    json.dumps(doc)


def test_without_warm_the_build_formats_nothing_and_writes_nothing(lib):
    conn, _ = lib
    doc = dashboard.build(conn)
    assert all(r["formatted"] == {} for r in doc["references"])
    assert conn.execute("SELECT count(*) FROM reference_formatted").fetchone()[0] == 0


def test_fit_sheds_abstracts_from_the_oldest_rows_before_dropping_any():
    refs = [
        {
            "key": f"k{i}",
            "title": "t" * 200,
            "abstract": "a" * 300,
            "formatted": {"apa": "f" * 200},
            "bibtex": "b" * 400,
        }
        for i in range(20)
    ]
    doc = {"references": refs, "references_truncated": False, "references_limit": 20}
    full = len(dashboard.render(doc).encode())
    fitted = dashboard.fit(doc, max_bytes=full - 1000)
    assert len(fitted["references"]) == 20 and fitted["references_truncated"] is False
    shed = [r["abstract"] is None for r in fitted["references"]]
    assert shed[-1] and not shed[0], "the oldest (last) lose theirs first"
    assert fitted["references_trimmed"] == sum(shed)
    assert len(dashboard.render(fitted).encode()) <= full - 1000
    budget = len(dashboard.render({**doc, "references": refs[:3]}).encode())
    tight = dashboard.fit(doc, max_bytes=budget)
    assert tight["references_truncated"] is True and 3 <= len(tight["references"]) < 20
    assert tight["references_limit"] == len(tight["references"])
    assert [r["key"] for r in tight["references"]] == [
        f"k{i}" for i in range(len(tight["references"]))
    ]
    assert all(r["title"] for r in tight["references"])
    assert len(dashboard.render(tight).encode()) <= budget


def test_an_oversized_bibtex_entry_is_withheld_not_cut():
    assert dashboard.bibtex_block("@misc{k,\n  title = {T}\n}\n") == "@misc{k,\n  title = {T}\n}\n"
    assert dashboard.bibtex_block("x" * 5000) is None


# --------------------------------------------------------------------------- latex


def test_repeated_keys_names_keys_two_loaded_files_define(tmp_path):
    (tmp_path / "a.bib").write_text("@misc{shared, title={A}}\n@misc{only_a, title={A2}}\n")
    (tmp_path / "b.bib").write_text("@misc{shared, title={B}}\n")
    (tmp_path / "c.bib").write_text("@misc{only_a, title={C}}\n")  # not loaded
    paper = tmp_path / "p.tex"
    paper.write_text("\\documentclass{article}\n\\cite{shared}\n\\bibliography{a,b}\n")
    out = latex.check_paper(paper)
    assert out["repeated_keys"] == {"shared": ["a.bib", "b.bib"]}
    merged = latex.check_citations(tmp_path, extra_bibs=[])
    assert merged["repeated_keys"] == [
        {"key": "shared", "files": ["a.bib", "b.bib"], "paper": str(paper.resolve())}
    ]
    paper.write_text("\\documentclass{article}\n\\cite{shared}\n")
    assert latex.check_paper(paper)["repeated_keys"] == {}, "no named .bib: BibTeX loads none"


# --------------------------------------------------------------------------- MCP


def _mcp_db(lib, monkeypatch):
    conn, _ = lib
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ATTEST_DB", path)


def test_cite_format_returns_text_and_bibtex(lib, monkeypatch):
    from attestation.mcp import citation_format

    _mcp_db(lib, monkeypatch)
    out = citation_format._format(["bishop2006", "10.1038/s41467-022-29939-5", "nope"], "ieee")
    assert out["ok"] is True and out["style"] == "ieee" and out["missing"] == ["nope"]
    first = out["references"][0]
    assert first["cite_key"] == "bishop2006" and first["text"].startswith("C. M. Bishop,")
    assert first["bibtex"].startswith("@book{bishop2006,")
    assert "not in the library: nope" in out["message"]
    refused = citation_format._format(["nope"])
    assert refused["ok"] is False and refused["references"] == []


def test_cite_format_stays_inside_the_response_ceiling(tmp_path, monkeypatch):
    from test_response_size import HARD_RESPONSE_CEILING, emitted

    from attestation.mcp import citation_format

    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    conn = get_db(db)
    keys = []
    for i in range(citation_format.MAX_KEYS):
        library.upsert(
            conn,
            library.ReferenceRecord(
                source="bibtex:x.bib",
                source_key=f"k{i}",
                bib_key=f"k{i}",
                title="W" * 300,
                authors=[f"Author{j}, Given" for j in range(40)],
                year=2020,
                venue="V" * 200,
                doi=f"10.5555/{i}",
                url="https://example.org/" + "u" * 200,
                volume="12",
                pages="1--2",
                extra={"note": "n" * 300, "publisher": "P" * 100},
            ),
        )
        keys.append(f"k{i}")
    conn.commit()
    conn.close()
    out = citation_format._format(keys)
    assert out["ok"] and len(out["references"]) == citation_format.MAX_KEYS
    assert emitted(out) <= HARD_RESPONSE_CEILING, emitted(out)


def test_cite_search_rows_carry_the_arxiv_id(lib):
    conn, _ = lib
    hit = library.search(conn, "doe2021preprint").hits[0]
    assert hit.to_row()["arxiv"] == "2101.03164"


# --------------------------------------------------------------------------- store


def test_a_version_13_database_gains_the_citation_manager_schema(tmp_path):
    path = tmp_path / "old.db"
    raw = get_db(path)  # the vec0 extension loaded: the delete trigger names it
    for table in ("reference_aliases", "reference_notes", "reference_edits", "reference_formatted"):
        raw.execute(f"DROP TABLE {table}")
    for col in ("entry_type", "volume", "pages", "extra"):
        raw.execute(f'ALTER TABLE "references" DROP COLUMN {col}')
    raw.execute("PRAGMA user_version = 13")
    raw.commit()
    raw.close()
    conn = get_db(path)
    cols = {r["name"] for r in conn.execute('PRAGMA table_info("references")')}
    assert {"entry_type", "volume", "pages", "extra"} <= cols
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"reference_aliases", "reference_notes", "reference_edits", "reference_formatted"} <= (
        tables
    )
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 14


# --------------------------------------------------------------------------- CLI


def test_the_bib_commands_end_to_end(lib, tmp_path, monkeypatch, capsys):
    from attestation.cli import main

    conn, _ = lib
    db = conn.execute("PRAGMA database_list").fetchone()["file"]
    conn.close()
    monkeypatch.setenv("ATTEST_BIB_OUT", str(tmp_path / "generated"))
    assert main(["bib", "--db", db, "format", "bishop2006", "--style", "nature"]) == 0
    assert capsys.readouterr().out.startswith("Bishop, C. M. Pattern Recognition")
    assert main(["bib", "--db", db, "format", "nope"]) == 1
    assert main(["bib", "--db", db, "show", "vaswani2017"]) == 0
    assert "@inproceedings{vaswani2017," in capsys.readouterr().out
    assert main(["bib", "--db", db, "edit", "vaswani2017", "--set", "volume=30"]) == 0
    assert "volume" in capsys.readouterr().out
    assert main(["bib", "--db", db, "edit", "vaswani2017", "--set", "volume"]) == 1
    assert main(["bib", "--db", db, "merge"]) == 0
    assert "0 duplicate pair(s)" in capsys.readouterr().out
    target = tmp_path / "export.bib"
    assert main(["bib", "--db", db, "export", "--out", str(target)]) == 0
    assert len(bibtex_io.parse(target.read_text()).entries) == 5
    assert main(["bib", "--db", db, "export", "--out", str(target)]) == 1, "no silent overwrite"
    assert main(["bib", "--db", db, "import", str(tmp_path / "nope.bib")]) == 1
