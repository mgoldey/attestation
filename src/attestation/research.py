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

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, quote_plus, urlencode

import httpx
from defusedxml import ElementTree as SafeET

from attestation.library import ReferenceRecord, normalise_arxiv
from attestation.library_readers import _client, _crossref_abstract, _crossref_authors

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


# ---------------------------------------------------------------------------
# the clients: one injected fetch, pure parsers, paced
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
FETCH_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError, SafeET.ParseError)


def _sleep(seconds: float) -> None:
    """time.sleep, broken out so tests can monkeypatch pacing without waiting."""
    time.sleep(seconds)


def _default_fetch(url: str) -> bytes:
    """GET a URL through the library's client (redirects followed); raises httpx.HTTPError."""
    with _client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _collapse(text: str | None) -> str:
    """Collapse internal whitespace/newlines to single spaces; None -> ''."""
    return " ".join((text or "").split())


class _Client:
    """Shared: the injected fetch, the request log, and pacing between requests."""

    name = ""
    offline = False
    pace_seconds = 0.0

    def __init__(self, fetch=None):
        """Wrap `fetch` (default: a real HTTP GET) and start with an empty request log."""
        self.fetch = fetch or _default_fetch
        self.requests: list[str] = []
        self._last: float | None = None

    def _get(self, url: str) -> bytes:
        """Fetch `url`, pacing since the previous request and logging the URL."""
        if self._last is not None and self.pace_seconds:
            wait = self.pace_seconds - (time.monotonic() - self._last)
            if wait > 0:
                _sleep(wait)
        self.requests.append(url)
        self._last = time.monotonic()
        return self.fetch(url)


def _family_first(name: str) -> str:
    """'Simon Batzner' -> 'Batzner, Simon'; a single token or 'Family, Given' stays."""
    name = _collapse(name)
    if "," in name or " " not in name:
        return name
    given, _, family = name.rpartition(" ")
    return f"{family}, {given}"


def parse_arxiv(body: bytes) -> list[Paper]:
    """arXiv Atom search results as Papers; versionless ids; venue stays None."""
    out = []
    for entry in SafeET.fromstring(body).iter(f"{_ATOM}entry"):
        aid = normalise_arxiv((entry.findtext(f"{_ATOM}id") or "").partition("/abs/")[2])
        if not aid:
            continue
        doi_el = entry.find(f"{_ARXIV_NS}doi")
        published = entry.findtext(f"{_ATOM}published") or ""
        out.append(
            Paper(
                client="arxiv",
                external_id=aid,
                title=_collapse(entry.findtext(f"{_ATOM}title")),
                abstract=_collapse(entry.findtext(f"{_ATOM}summary")),
                authors=tuple(
                    _family_first(a.findtext(f"{_ATOM}name") or "")
                    for a in entry.iter(f"{_ATOM}author")
                ),
                published=published[:10] or None,
                doi=doi_el.text.strip() if doi_el is not None and doi_el.text else None,
                arxiv_id=aid,
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=f"https://arxiv.org/pdf/{aid}",
            )
        )
    return out


class ArxivSearch(_Client):
    """The arXiv export API; 3 s between requests, as arXiv asks."""

    name = "arxiv"
    pace_seconds = 3.0

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers matching every word of `query`, newest first, submitted since `since`."""
        terms = " AND ".join(f"all:{w}" for w in query.split())
        if since:
            terms += f" AND submittedDate:[{since:%Y%m%d}0000 TO 209912312359]"
        params = {
            "search_query": terms,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": limit,
        }
        url = "https://export.arxiv.org/api/query?" + urlencode(params, quote_via=quote_plus)
        return parse_arxiv(self._get(url))


def parse_pubmed_ids(body: bytes) -> list[str]:
    """PMIDs from an esearch response."""
    return [e.text for e in SafeET.fromstring(body).iter("Id") if e.text]


def _pubmed_date(article) -> str | None:
    """The article's pubmed-history date, falling back to the journal issue's PubDate."""
    node = article.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']")
    if node is None:
        node = article.find(".//Article/Journal/JournalIssue/PubDate")
    if node is None:
        return None
    year = node.findtext("Year")
    if not year or not year.isdigit():
        return None
    month = (node.findtext("Month") or "1").strip().lower()
    m = int(month) if month.isdigit() else _MONTHS.get(month[:3], 1)
    day = node.findtext("Day") or "1"
    return f"{int(year):04d}-{m:02d}-{int(day):02d}"


def _pubmed_authors(article) -> tuple[str, ...]:
    """'Family, Given' per Author, or the CollectiveName when there is no personal name."""
    out = []
    for a in article.iter("Author"):
        family = a.findtext("LastName")
        given = a.findtext("ForeName")
        group = a.findtext("CollectiveName")
        if family:
            out.append(f"{family}, {given}" if given else family)
        elif group:
            out.append(group)
    return tuple(out)


def _pubmed_paper(article, pmid: str, title: str) -> Paper:
    """One PubmedArticle, already known to have a PMID and a title, as a Paper."""
    ids = {a.get("IdType"): (a.text or "").strip() for a in article.iter("ArticleId")}
    return Paper(
        client="pubmed",
        external_id=pmid,
        title=title.rstrip("."),
        abstract=" ".join(_collapse(t.text) for t in article.iter("AbstractText") if t.text),
        authors=_pubmed_authors(article),
        published=_pubmed_date(article),
        doi=ids.get("doi") or None,
        pmcid=ids.get("pmc") or None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        venue=_collapse(article.findtext(".//Journal/Title")) or None,
    )


def parse_pubmed(body: bytes) -> list[Paper]:
    """efetch (rettype=abstract, retmode=xml) records as Papers."""
    out = []
    for article in SafeET.fromstring(body).iter("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = _collapse(article.findtext(".//ArticleTitle"))
        if pmid and title:
            out.append(_pubmed_paper(article, pmid, title))
    return out


class PubmedSearch(_Client):
    """NCBI E-utilities: esearch for ids, efetch for records. 3 req/s, 10 with a key."""

    name = "pubmed"
    _BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, fetch=None, api_key: str | None = None):
        """Wrap `fetch`; an `api_key` raises the NCBI rate limit and paces faster."""
        super().__init__(fetch)
        self.api_key = api_key
        self.pace_seconds = 0.1 if api_key else 0.34

    def _params(self, **params) -> str:
        """Encode query params, adding `api_key` when one was configured."""
        if self.api_key:
            params["api_key"] = self.api_key
        return urlencode(params, quote_via=quote_plus)

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers for `query` (AND a `[Journal]` term), entered since `since`."""
        term = f"({query})" if journal else query
        if journal:
            term += f' AND "{journal}"[Journal]'
        params = {"db": "pubmed", "term": term, "retmax": limit, "sort": "date"}
        if since:
            params.update({"datetype": "edat", "mindate": f"{since:%Y/%m/%d}", "maxdate": "3000"})
        ids = parse_pubmed_ids(self._get(self._BASE + "esearch.fcgi?" + self._params(**params)))
        if not ids:
            return []
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        return parse_pubmed(self._get(self._BASE + "efetch.fcgi?" + self._params(**fetch_params)))


def _crossref_date(item: dict) -> str | None:
    """`issued.date-parts` as an ISO date; missing month/day default to the 1st."""
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    if not parts or not parts[0]:
        return None
    y, m, d = (list(parts) + [1, 1])[:3]
    return f"{int(y):04d}-{int(m or 1):02d}-{int(d or 1):02d}"


def _crossref_paper(item: dict, doi: str, title: str) -> Paper:
    """One `/works` item, already known to have a DOI and a title, as a Paper."""
    venue = _collapse((item.get("container-title") or [""])[0]) or None
    return Paper(
        client="crossref",
        external_id=doi,
        title=title,
        abstract=_crossref_abstract(item) or "",
        authors=tuple(_crossref_authors(item)),
        published=_crossref_date(item),
        doi=doi,
        url=item.get("URL") or f"https://doi.org/{doi}",
        venue=venue,
    )


def parse_crossref(body: bytes) -> list[Paper]:
    """`/works` items as Papers; an item with no title cannot name a paper and is skipped."""
    items = ((json.loads(body) or {}).get("message") or {}).get("items") or []
    out = []
    for item in items:
        doi = (item.get("DOI") or "").lower()
        title = _collapse((item.get("title") or [""])[0])
        if doi and title:
            out.append(_crossref_paper(item, doi, title))
    return out


class CrossrefSearch(_Client):
    """CrossRef `/works`: the only way to search a journal on neither arXiv nor PubMed."""

    name = "crossref"

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers matching `query`, optionally within `journal`, published since `since`."""
        params: dict = {"query": query, "rows": limit, "sort": "published", "order": "desc"}
        if journal:
            params["query.container-title"] = journal
        if since:
            params["filter"] = f"from-pub-date:{since.isoformat()}"
        url = "https://api.crossref.org/works?" + urlencode(params, quote_via=quote_plus)
        return parse_crossref(self._get(url))


@dataclass
class Fetched:
    """What one topic run produced: entries for ingest, papers for the library, and why not."""

    entries: list[dict]
    papers: list[Paper]
    errors: list[str]
    offline: bool


def fetch_topic(
    topic: Topic, clients: dict, *, since: date | None = None, limit: int = 50
) -> Fetched:
    """Run every client a topic names.

    A client's failure is one error string, never an exception: the other
    clients and the other feeds still run.
    """
    papers: list[Paper] = []
    errors: list[str] = []
    for name in topic.clients:
        client = clients[name]
        try:
            papers.extend(
                client.search(topic.query, journal=topic.journal, since=since, limit=limit)
            )
        except FETCH_ERRORS as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    offline = all(clients[n].offline for n in topic.clients)
    return Fetched(as_entries(papers), papers, errors, offline)
