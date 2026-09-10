"""research.py: topic URLs, payload parsing, conversions, the flag. No network."""

import time

import pytest

from attestation import research


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
