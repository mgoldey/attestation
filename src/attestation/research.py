"""Paper research: search clients over arXiv, PubMed and CrossRef, topics as feeds.

Spec: docs/superpowers/specs/2026-09-10-paper-research-design.md. A standing
topic is a `feeds` row whose URL is `research:<clients>?q=<query>[&journal=]`;
`run_ingest` dispatches on that scheme and the clients here return entries
shaped like feedparser's, so the three ingest passes run unchanged. Parsing
is pure over committed fixture payloads; the HTTP call is one injected
`fetch(url) -> bytes`. Nothing here ranks or summarises: the caller is the
model.

`ATTEST_RESEARCH_WEB` (unset or 1 = on) is read once, in `clients_from_env`,
never per call -- the offline guarantee's construction-time rule.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, quote_plus, urlencode

from attestation.library import ReferenceRecord

CLIENT_NAMES = ("arxiv", "pubmed", "crossref")
SCHEME = "research:"
DEFAULT_SINCE_DAYS = 365


class TopicError(ValueError):
    """A research URL the caller can fix: unknown client, no query, journal on arXiv."""


@dataclass(frozen=True)
class Topic:
    """One standing query: which clients, what to ask, optionally which journal."""

    clients: tuple[str, ...]
    query: str
    journal: str | None = None

    @property
    def url(self) -> str:
        """The canonical `research:` URL, so two spellings of one topic are one row."""
        return topic_url(self.clients, self.query, self.journal)

    @property
    def title(self) -> str:
        """The default feed title: `<clients>: <query>[ in <journal>]`."""
        tail = f" in {self.journal}" if self.journal else ""
        return f"{','.join(self.clients)}: {self.query}{tail}"


def is_topic_url(url: str) -> bool:
    """True for the `research:` scheme this module owns."""
    return url.startswith(SCHEME)


def topic_url(clients, query: str, journal: str | None = None) -> str:
    """Render a topic URL; `parse_topic` reads it back."""
    params = {"q": " ".join(query.split())}
    if journal:
        params["journal"] = " ".join(journal.split())
    return f"{SCHEME}{','.join(clients)}?{urlencode(params, quote_via=quote_plus)}"


def parse_topic(url: str) -> Topic:
    """A `research:` URL as a Topic, or TopicError naming what to fix."""
    if not is_topic_url(url):
        raise TopicError(f"{url!r} is not a research: URL (research:<clients>?q=<query>)")
    path, _, qs = url[len(SCHEME) :].partition("?")
    clients = tuple(c.strip() for c in path.split(",") if c.strip())
    unknown = [c for c in clients if c not in CLIENT_NAMES]
    if not clients or unknown:
        raise TopicError(
            f"unknown research client(s) {', '.join(unknown) or '(none)'};"
            f" the clients are {', '.join(CLIENT_NAMES)}"
        )
    params = parse_qs(qs, keep_blank_values=True)
    query = " ".join(params.get("q", [""])[0].split())
    if not query:
        raise TopicError(
            "a research topic needs q=<query>, e.g. research:arxiv?q=graph+neural+networks"
        )
    journal = " ".join(params.get("journal", [""])[0].split()) or None
    if journal and "arxiv" in clients:
        raise TopicError("arXiv has categories, not journals: drop journal= or drop arxiv")
    return Topic(clients, query, journal)


@dataclass(frozen=True)
class Paper:
    """One search hit, whichever client found it. `external_id` is the client's own handle."""

    client: str
    external_id: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    published: str | None = None  # ISO date, the paper's own
    doi: str | None = None
    arxiv_id: str | None = None
    pmcid: str | None = None
    url: str | None = None
    venue: str | None = None
    pdf_url: str | None = None


def as_entries(papers) -> list[dict]:
    """Papers as the entry dicts `ingest._new_entries` and the insert read.

    `id` is `<client>:<external id>` so UNIQUE(feed_id, guid) holds across the
    clients one topic names; `published_parsed` is the paper's date, never the
    fetch time, so a 2019 hit does not surface as new.
    """
    out = []
    for p in papers:
        entry = {
            "title": p.title,
            "summary": p.abstract,
            "id": f"{p.client}:{p.external_id}",
            "link": p.url,
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
        }
        if p.published:
            try:
                entry["published_parsed"] = time.strptime(p.published[:10], "%Y-%m-%d")
            except ValueError:
                pass
        out.append(entry)
    return out


def as_records(papers, fetched_at: str) -> list[ReferenceRecord]:
    """Papers as library records under source `research:<client>`, with the
    authors, venue and abstract the payload carried (the stated exception to
    the enricher rule -- the reader asked for these by query)."""
    return [
        ReferenceRecord(
            source=f"research:{p.client}",
            source_key=p.external_id,
            title=p.title,
            authors=list(p.authors),
            year=int(p.published[:4]) if p.published and p.published[:4].isdigit() else None,
            doi=p.doi,
            arxiv_id=p.arxiv_id,
            pmcid=p.pmcid,
            venue=p.venue,
            abstract=p.abstract or None,
            url=p.url,
            fetched_at=fetched_at,
        )
        for p in papers
    ]


def research_enabled() -> bool:
    """ATTEST_RESEARCH_WEB: on unless set to 0/false. Read by `clients_from_env` only."""
    return os.environ.get("ATTEST_RESEARCH_WEB", "1").strip().lower() not in ("0", "false")


class NullClient:
    """What every client is when the flag is off: answers nothing, says it is offline."""

    offline = True

    def __init__(self, name: str):
        self.name = name

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """No results; the caller reports `offline` rather than 'no papers'."""
        return []


def null_clients() -> dict:
    """One NullClient per name -- `attest ingest --no-research` and the flag off."""
    return {name: NullClient(name) for name in CLIENT_NAMES}


def clients_from_env(*, fetch=None) -> dict:
    """The three clients, or NullClients when ATTEST_RESEARCH_WEB is off.

    The flag is read HERE, once, when the ingest run or the MCP server builds
    its clients -- a disabled client cannot be coaxed into a request later.
    """
    if not research_enabled():
        return null_clients()
    return {
        "arxiv": ArxivSearch(fetch=fetch),
        "pubmed": PubmedSearch(fetch=fetch, api_key=os.environ.get("NCBI_API_KEY")),
        "crossref": CrossrefSearch(fetch=fetch),
    }


class ArxivSearch(NullClient):  # replaced in Task 3
    """Replaced in Task 3."""

    offline = False

    def __init__(self, fetch=None):
        super().__init__("arxiv")


class PubmedSearch(NullClient):  # replaced in Task 3
    """Replaced in Task 3."""

    offline = False

    def __init__(self, fetch=None, api_key=None):
        super().__init__("pubmed")


class CrossrefSearch(NullClient):  # replaced in Task 3
    """Replaced in Task 3."""

    offline = False

    def __init__(self, fetch=None):
        super().__init__("crossref")
