"""feed.research: envelope, storing to the library, offline honesty, payload size."""

import json

from test_response_size import HARD_RESPONSE_CEILING

from attestation import research
from attestation.db import get_db
from attestation.mcp import research as tool_mod


def _db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    get_db(db).close()
    return db


def _paper(i: int):
    return research.Paper(
        client="arxiv",
        external_id=f"2101.{i:05d}",
        title=f"Paper {i} " + "x" * 200,
        abstract="A" * 500,
        authors=tuple(f"Author{j}, Given" for j in range(12)),
        published="2021-01-08",
        arxiv_id=f"2101.{i:05d}",
        url=f"https://arxiv.org/abs/2101.{i:05d}",
        venue="V" * 80,
    )


class Fixed:
    name, offline = "arxiv", False

    def __init__(self, papers):
        self.papers = papers

    def search(self, query, *, journal=None, since=None, limit=50):
        return self.papers[:limit]


def _clients(papers):
    return {
        "arxiv": Fixed(papers),
        "pubmed": research.NullClient("pubmed"),
        "crossref": research.NullClient("crossref"),
    }


def test_research_stores_to_the_library_not_the_feed(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = tool_mod._research("equivariant", sources="arxiv")
    assert out["ok"] and out["n_found"] == 1 and out["stored"] == 1 and out["offline"] is False
    assert out["papers"][0]["stored"] is True and out["papers"][0]["arxiv_id"] == "2101.00001"
    conn = get_db(db)
    assert conn.execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    again = tool_mod._research("equivariant", sources="arxiv")
    assert again["stored"] == 0  # unchanged on the second call


def test_research_preview_does_not_store(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = tool_mod._research("x", sources="arxiv", store=False)
    assert out["stored"] == 0 and out["papers"][0]["stored"] is False
    assert get_db(db).execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 0


def test_research_offline_says_so(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", research.null_clients())
    out = tool_mod._research("x")
    assert out["ok"] and out["offline"] is True and out["papers"] == []
    assert "ATTEST_RESEARCH_WEB" in out["message"]


def test_research_refuses_unknown_source_and_journal_on_arxiv(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([]))
    out = tool_mod._research("x", sources="scholar")
    assert out["ok"] is False and "arxiv, pubmed, crossref" in out["message"]
    out = tool_mod._research("x", sources="arxiv", journal="Nature")
    assert out["ok"] is False and "categories" in out["message"]


def test_research_payload_fits_at_the_cap(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(i) for i in range(20)]))
    out = tool_mod._research("x", sources="arxiv", limit=16, store=False)
    assert len(out["papers"]) == tool_mod.MAX_RESEARCH_LIMIT
    assert len(json.dumps(out, indent=2)) < HARD_RESPONSE_CEILING


def test_feed_ask_executes_research_and_track(tmp_path, monkeypatch):
    from conftest import seeded_db

    from attestation.mcp import ask

    db = _db(tmp_path, monkeypatch)
    seeded_db(db).close()
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = ask._feed_ask("researcher", "search arxiv for equivariant force fields")
    assert out["ok"] and out["tool_used"] == "feed.research" and "1 paper" in out["answer"]
    out = ask._feed_ask("researcher", "track equivariant interatomic potentials for me")
    assert out["ok"] and out["tool_used"] == "feed.source_add"
    row = (
        get_db(db).execute("SELECT url, added_by FROM feeds WHERE url LIKE 'research:%'").fetchone()
    )
    assert row["url"] == "research:arxiv,pubmed?q=equivariant+interatomic+potentials"
    assert row["added_by"] is not None
