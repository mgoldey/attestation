"""research.py: topic URLs, payload parsing, conversions, the flag. No network."""

import time
from datetime import date
from pathlib import Path

import pytest

from attestation import research
from attestation.db import get_db
from attestation.library import ReferenceRecord, upsert

FIX = Path(__file__).parent / "fixtures" / "research"


def _pdf_bytes(text: str = "Hello full text") -> bytes:
    """The committed one-page PDF fixture; `text` must be its literal contents.

    pypdf's own writer API is fragile to poke at directly for a one-off test
    PDF (see tests/fixtures/research/make_one_page_pdf.py's docstring), so the
    fixture is generated once and read here instead of built per-call.
    """
    assert text == "Hello full text", "the committed fixture only carries this text"
    return (FIX / "one_page.pdf").read_bytes()


def test_parse_topic_two_forms():
    t = research.parse_topic("research:arxiv,pubmed?q=equivariant+force+fields")
    assert t.clients == ("arxiv", "pubmed") and t.query == "equivariant force fields"
    assert t.journal is None
    t2 = research.parse_topic("research:crossref?q=protein+language+models&journal=Nature+Methods")
    assert t2.clients == ("crossref",) and t2.journal == "Nature Methods"
    assert t2.url == "research:crossref?q=protein+language+models&journal=Nature+Methods"
    assert t2.title == "crossref: protein language models in Nature Methods"


@pytest.mark.parametrize(
    "url, needle",
    [
        ("research:scholar?q=x", "unknown research client"),
        ("research:arxiv?q=", "needs q="),
        ("research:arxiv?q=x&journal=Nature", "categories, not journals"),
        ("https://example.com/rss", "not a research: URL"),
    ],
)
def test_parse_topic_refuses(url, needle):
    with pytest.raises(research.TopicError, match=needle):
        research.parse_topic(url)


def test_topic_url_round_trips():
    url = research.topic_url(("arxiv",), "graph neural networks", None)
    assert research.parse_topic(url).query == "graph neural networks"
    assert research.is_topic_url(url) and not research.is_topic_url("http://x")


def _paper(**kw):
    base = dict(
        client="arxiv",
        external_id="2106.02347",
        title="E(3)-equivariant graph neural networks",
        abstract="We propose NequIP.",
        authors=("Batzner, Simon", "Musaelian, Albert"),
        published="2021-06-04",
        arxiv_id="2106.02347",
        url="https://arxiv.org/abs/2106.02347",
    )
    base.update(kw)
    return research.Paper(**base)


def test_as_entries_is_feedparser_shaped():
    [e] = research.as_entries([_paper()])
    assert e["id"] == "arxiv:2106.02347"
    assert e["title"].startswith("E(3)") and e["summary"] == "We propose NequIP."
    assert e["link"] == "https://arxiv.org/abs/2106.02347"
    assert e["arxiv_id"] == "2106.02347" and e["doi"] is None
    assert time.strftime("%Y-%m-%d", e["published_parsed"]) == "2021-06-04"


def test_as_records_carries_what_the_payload_had():
    [r] = research.as_records([_paper(doi="10.1/x", pmcid="PMC1", venue=None)], "2026-09-10")
    assert r.source == "research:arxiv" and r.source_key == "2106.02347"
    assert r.authors == ["Batzner, Simon", "Musaelian, Albert"] and r.year == 2021
    assert r.doi == "10.1/x" and r.pmcid == "PMC1" and r.fetched_at == "2026-09-10"


def test_store_papers_upserts_once_and_is_idempotent(tmp_path):
    """The shared home for `attest research` and `feed.research`: two distinct
    hits store as two new rows the first time, and re-storing the same two
    changes nothing the second time (an unchanged row counts 0)."""
    conn = get_db(tmp_path / "t.db")
    papers = [
        _paper(external_id="2106.02347", arxiv_id="2106.02347", doi=None),
        _paper(external_id="2106.99999", arxiv_id="2106.99999", doi=None, title="A second paper"),
    ]

    assert research.store_papers(conn, papers) == 2
    assert research.store_papers(conn, papers) == 0

    n = conn.execute("SELECT COUNT(*) FROM reference_sources").fetchone()[0]
    assert n == len(papers)


def test_flag_default_on_and_off(monkeypatch):
    monkeypatch.delenv("ATTEST_RESEARCH_WEB", raising=False)
    assert research.research_enabled()
    for off in ("0", "false"):
        monkeypatch.setenv("ATTEST_RESEARCH_WEB", off)
        assert not research.research_enabled()
        clients = research.clients_from_env()
        assert set(clients) == set(research.CLIENT_NAMES)
        assert all(c.offline for c in clients.values())
        assert clients["arxiv"].search("x") == []
    monkeypatch.setenv("ATTEST_RESEARCH_WEB", "1")
    assert not any(c.offline for c in research.clients_from_env().values())


def _fetcher(routes: dict[str, str]):
    """fetch(url) -> bytes from the first route whose key is a substring of the url."""
    asked: list[str] = []

    def fetch(url: str) -> bytes:
        asked.append(url)
        for needle, name in routes.items():
            if needle in url:
                return (FIX / name).read_bytes()
        raise AssertionError(f"unexpected url {url}")

    return fetch, asked


def test_parse_arxiv_search():
    papers = research.parse_arxiv((FIX / "arxiv_search.xml").read_bytes())
    assert [p.arxiv_id for p in papers] == ["2101.03164", "2206.07697"]
    p = papers[0]
    assert p.client == "arxiv" and p.external_id == "2101.03164"
    assert p.title.startswith("E(3)-Equivariant Graph Neural Networks for Data-Efficient")
    assert "  " not in p.title
    assert p.authors == ("Batzner, Simon", "Musaelian, Albert")
    assert p.published == "2021-01-08" and p.doi == "10.1038/s41467-022-29939-5"
    assert p.url == "https://arxiv.org/abs/2101.03164"
    assert p.pdf_url == "https://arxiv.org/pdf/2101.03164" and p.venue is None
    assert papers[1].doi is None


def test_parse_pubmed():
    assert research.parse_pubmed_ids((FIX / "pubmed_esearch.xml").read_bytes()) == [
        "36702928",
        "34961817",
    ]
    papers = research.parse_pubmed((FIX / "pubmed_efetch.xml").read_bytes())
    p = papers[0]
    assert p.client == "pubmed" and p.external_id == "36702928"
    assert p.abstract == "Language models learn. They predict structure."
    assert p.authors == ("Lin, Zeming", "Rives, Alexander") and p.venue == "Nature methods"
    assert p.published == "2023-01-27" and p.doi == "10.1038/s41592-022-01760-3"
    assert p.pmcid == "PMC9912345" and p.url == "https://pubmed.ncbi.nlm.nih.gov/36702928/"
    assert papers[1].authors == ("The Consortium",) and papers[1].published == "2021-12-01"


def test_parse_crossref_skips_titleless():
    [p] = research.parse_crossref((FIX / "crossref_works.json").read_bytes())
    assert p.client == "crossref" and p.external_id == p.doi == "10.1038/s41592-022-01760-3"
    assert p.venue == "Nature Methods" and p.authors == ("Lin, Zeming", "Rives, Alexander")
    assert p.published == "2023-01-27" and p.abstract == "Language models learn."
    assert p.url == "https://doi.org/10.1038/s41592-022-01760-3"


def test_arxiv_search_builds_the_query_and_paces(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(research, "_sleep", slept.append)
    fetch, asked = _fetcher({"export.arxiv.org": "arxiv_search.xml"})
    client = research.ArxivSearch(fetch=fetch)
    papers = client.search("equivariant force fields", since=date(2021, 1, 1), limit=20)
    assert len(papers) == 2
    assert "search_query=all%3Aequivariant+AND+all%3Aforce+AND+all%3Afields" in asked[0]
    assert "submittedDate%3A%5B202101010000+TO" in asked[0] and "max_results=20" in asked[0]
    client.search("again")
    assert slept and slept[0] == pytest.approx(3.0, abs=0.5)  # 3 s between requests, as arXiv asks


def test_pubmed_search_two_calls_journal_and_key(monkeypatch):
    monkeypatch.setattr(research, "_sleep", lambda s: None)
    fetch, asked = _fetcher({"esearch": "pubmed_esearch.xml", "efetch": "pubmed_efetch.xml"})
    client = research.PubmedSearch(fetch=fetch, api_key="K")
    papers = client.search(
        "protein language models", journal="Nature Methods", since=date(2022, 1, 1)
    )
    assert len(papers) == 2 and len(asked) == 2
    assert "%22Nature+Methods%22%5BJournal%5D" in asked[0] and "mindate=2022%2F01%2F01" in asked[0]
    assert "api_key=K" in asked[0] and "id=36702928%2C34961817" in asked[1]


def test_pubmed_search_with_no_ids_makes_one_call():
    fetch, asked = _fetcher({"esearch": "pubmed_esearch_empty.xml"})
    assert research.PubmedSearch(fetch=fetch).search("nothing") == [] and len(asked) == 1


def test_crossref_search_filters():
    fetch, asked = _fetcher(
        {
            "crossref.org/journals": "crossref_journals_empty.json",
            "crossref.org/works": "crossref_works.json",
        }
    )
    papers = research.CrossrefSearch(fetch=fetch).search(
        "x", journal="Nature Methods", since=date(2023, 1, 1)
    )
    assert len(papers) == 1
    # unresolved journal: the name boost plus the date filter, venue checked on the payload
    assert "query.container-title=Nature+Methods" in asked[1]
    assert "filter=from-pub-date%3A2023-01-01" in asked[1]


def test_fetch_topic_collects_across_clients_and_records_failures():
    class Boom:
        name, offline = "pubmed", False

        def search(self, *a, **k):
            raise research.httpx.ConnectError("down")

    fetch, _ = _fetcher({"export.arxiv.org": "arxiv_search.xml"})
    clients = {"arxiv": research.ArxivSearch(fetch=fetch), "pubmed": Boom()}
    out = research.fetch_topic(research.Topic(("arxiv", "pubmed"), "x"), clients)
    assert len(out.papers) == 2 and len(out.entries) == 2 and not out.offline
    assert out.errors == ["pubmed: ConnectError: down"]
    null = research.fetch_topic(research.Topic(("arxiv",), "x"), research.null_clients())
    assert null.offline and null.papers == []


def test_pdf_text_extracts_and_tolerates_garbage():
    """pypdf reads the fixture's text; garbage bytes return None, never raise."""
    assert "Hello full text" in (research.pdf_text(_pdf_bytes("Hello full text")) or "")
    assert research.pdf_text(b"not a pdf") is None


def test_parse_pmc_joins_body_paragraphs_and_none_without_body():
    """PMC XML's <body> paragraphs and section titles join into one text; no body -> None."""
    xml = (
        b"<pmc-articleset><article><front><article-meta/></front>"
        b"<body><sec><title>Intro</title><p>First para.</p>"
        b"<p>Second <italic>para</italic>.</p></sec></body>"
        b"</article></pmc-articleset>"
    )
    assert research.parse_pmc(xml) == "Intro\nFirst para.\nSecond para."
    assert (
        research.parse_pmc(b"<pmc-articleset><article><front/></article></pmc-articleset>") is None
    )


def test_fetch_fulltext_caps_marks_none_and_retries_only_transient(tmp_path):
    """A capped pass: PMC/arXiv bodies fetched, an unreadable PDF marked 'none', a transient
    transport failure left unrecorded so the next run retries it, and the flag off is a no-op."""
    conn = get_db(tmp_path / "t.db")
    ids = []
    for i in range(3):
        rid, _ = upsert(
            conn,
            ReferenceRecord(
                source="research:arxiv",
                source_key=f"2101.0000{i}",
                title=f"P{i}",
                arxiv_id=f"2101.0000{i}",
            ),
        )
        ids.append(rid)
    pm, _ = upsert(
        conn, ReferenceRecord(source="research:pubmed", source_key="9", title="PM", pmcid="PMC9")
    )
    none_id, _ = upsert(conn, ReferenceRecord(source="zotero", source_key="Z", title="no ids"))
    conn.commit()
    calls = []

    def fetch(url):
        calls.append(url)
        if "2101.00002" in url:
            raise research.httpx.ConnectError("down")
        if "2101.00001" in url:
            return b"garbage"
        if "pmc" in url:
            return (
                b"<pmc-articleset><article><body><p>PMC body.</p></body></article></pmc-articleset>"
            )
        return _pdf_bytes("Hello full text")

    out = research.fetch_fulltext(conn, limit=2, fetch=fetch)
    assert out == {"fetched": 1, "none": 0, "failed": 1} or out == {
        "fetched": 1,
        "none": 1,
        "failed": 0,
    }
    out2 = research.fetch_fulltext(conn, limit=10, fetch=fetch)
    rows = {r["reference_id"]: r for r in conn.execute("SELECT * FROM reference_fulltext")}
    assert none_id not in rows  # nothing to fetch, never selected
    assert rows[pm]["source"] == "pmc-xml" and rows[pm]["text"] == "PMC body."
    assert (
        rows[ids[1]]["source"] == "none" and rows[ids[1]]["text"] is None
    )  # unreadable pdf: tried
    assert ids[2] not in rows  # transient failure: no row, retried next run
    assert any("2101.00002" in u for u in calls[-3:])
    assert out2["failed"] == 1
    # the flag off does nothing
    assert research.fetch_fulltext(
        conn, limit=10, fetch=fetch, clients=research.null_clients()
    ) == {
        "fetched": 0,
        "none": 0,
        "failed": 0,
    }


def test_fetch_fulltext_marks_malformed_pmc_body_none_not_failed(tmp_path):
    """A garbled PMC response (HTML error page, truncated XML) is an unreadable body, not a
    transport failure: it writes source='none' and is never retried, same as an unreadable PDF."""
    conn = get_db(tmp_path / "t.db")
    pm, _ = upsert(
        conn, ReferenceRecord(source="research:pubmed", source_key="9", title="PM", pmcid="PMC9")
    )
    conn.commit()

    def fetch(url):
        return b"<html>not xml"

    out = research.fetch_fulltext(conn, limit=10, fetch=fetch)
    assert out == {"fetched": 0, "none": 1, "failed": 0}
    row = conn.execute(
        "SELECT source, text FROM reference_fulltext WHERE reference_id = ?", (pm,)
    ).fetchone()
    assert row["source"] == "none" and row["text"] is None


def test_crossref_journal_resolves_to_an_issn_filter():
    """MEASURED 2026-09-11: `query.container-title` only RANKS by venue -- a search
    scoped to the Journal of Chemical Physics returned Chemical Engineering
    Science 12 times out of 12 under date order and 0 of 12 of its own papers
    under relevance order, while CrossRef's server-side `container-title`
    filter is exact-match (0 hits without the leading "The"). `/journals`
    resolves the name to an ISSN and `filter=issn:` is exact."""
    fetch, asked = _fetcher(
        {
            "crossref.org/journals": "crossref_journals.json",
            "crossref.org/works": "crossref_works_mixed.json",
        }
    )
    client = research.CrossrefSearch(fetch=fetch)

    papers = client.search(
        "force fields", journal="Journal of Chemical Physics", limit=2, since=date(2025, 1, 1)
    )
    assert "journals?query=Journal+of+Chemical+Physics" in asked[0]
    assert "filter=issn%3A0021-9606%2Cfrom-pub-date%3A2025-01-01" in asked[1]
    assert "rows=2" in asked[1] and "container-title" not in asked[1]
    assert len(papers) == 2, "the ISSN filter is exact: the payload is trusted as-is"

    client.search("other", journal="the journal of chemical physics", limit=1)
    assert len(asked) == 3 and "works" in asked[2], "resolution cached per name, spelling aside"


def test_crossref_unresolved_journal_falls_back_to_venue_match():
    """An abbreviation `/journals` does not know ("J Chem Phys" -> no items): boost
    by name, over-fetch, keep items whose long or short container title is the
    same name after normalisation -- equality, so "Nature" never claims Nature
    Methods -- then cut to `limit`."""
    fetch, asked = _fetcher(
        {
            "crossref.org/journals": "crossref_journals_empty.json",
            "crossref.org/works": "crossref_works_mixed.json",
        }
    )
    client = research.CrossrefSearch(fetch=fetch)

    papers = client.search("force fields", journal="J Chem Phys", limit=2)
    assert [p.doi for p in papers] == ["10.1063/5.0000001", "10.1063/5.0000002"]
    assert "rows=8" in asked[1] and "query.container-title=J+Chem+Phys" in asked[1]

    assert client.search("force fields", journal="Nature", limit=5) == []
    assert len(client.search("force fields", limit=5)) == 4, "no journal: nothing dropped"
    assert "rows=5" in asked[-1] and "journals" not in asked[-1]
