"""Sources of ReferenceRecords: three from disk and the feed, three enrichers
behind flags.

Only the offline readers can INTRODUCE a reference. The network readers
(arXiv, CrossRef, Semantic Scholar) fill fields and citation edges on rows
that already exist, which is what keeps the library the reader's and not the
web's, and keeps `cite.sources`' `offline` answer honest (spec §3.1).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
from defusedxml import ElementTree as SafeET

from attestation import citations
from attestation.library import ReferenceRecord, identity

DEFAULT_CACHE = Path.home() / ".hermes" / "citation-cache"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_JATS = re.compile(r"<[^>]+>")
_ARXIV_BATCH = 50
_S2_FIELDS = "title,externalIds,references.title,references.externalIds"


def _bib_authors(value: str) -> list[str]:
    """`A and B and others` -> ["A", "B"]; the `others` marker is not a name."""
    return [a.strip() for a in value.split(" and ") if a.strip() and a.strip() != "others"]


class BibtexRecords:
    """Every entry of every `.bib` file given, through citations' one parser."""

    name = "bibtex"
    network = False

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def records(self) -> Iterator[ReferenceRecord]:
        """One record per entry with a title; a missing file is an absent source."""
        for path in self.paths:
            if not path.is_file():
                continue
            for key, f in citations._parse_bib_entries(path.read_text(errors="replace")):
                is_arxiv = f.get("archiveprefix", "").lower() == "arxiv"
                yield ReferenceRecord(
                    source=f"bibtex:{path}",
                    source_key=key,
                    title=f["title"],
                    authors=_bib_authors(f.get("author", "")),
                    year=citations._year(f.get("year") or f.get("date")),
                    doi=f.get("doi"),
                    arxiv_id=(f.get("eprint") if is_arxiv else None) or f.get("arxivid"),
                    venue=f.get("journal") or f.get("booktitle"),
                    abstract=f.get("abstract"),
                    url=f.get("url"),
                    bib_key=key,
                )


class ZoteroRecords:
    """A local Zotero library, through `ZoteroReader.raw_items` (read-only, tolerant)."""

    name = "zotero"
    network = False

    def __init__(self, path: Path | None = None):
        self.reader = citations.ZoteroReader(path)

    def records(self) -> Iterator[ReferenceRecord]:
        """One record per titled, undeleted Zotero item; nothing on any sqlite error."""
        for key, data, authors in self.reader.raw_items():
            arxiv = None
            for line in (data.get("extra") or "").splitlines():
                if line.lower().startswith("arxiv:"):
                    arxiv = line.split(":", 1)[1].strip()
            yield ReferenceRecord(
                source="zotero",
                source_key=key,
                title=data["title"],
                authors=list(authors),
                year=citations._year(data.get("date")),
                doi=data.get("DOI"),
                arxiv_id=arxiv,
                venue=data.get("publicationTitle"),
                abstract=data.get("abstractNote"),
                url=data.get("url"),
                bib_key=key,
            )


class FeedRecords:
    """Feed items that carry a DOI or an arXiv id -- what the reader read."""

    name = "feed"
    network = False

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def records(self) -> Iterator[ReferenceRecord]:
        """One record per item with a DOI or arXiv id; the summary is its abstract."""
        rows = self.conn.execute(
            "SELECT id, title, url, summary, published, doi, arxiv_id FROM items"
            " WHERE doi IS NOT NULL OR arxiv_id IS NOT NULL ORDER BY id"
        ).fetchall()
        for r in rows:
            yield ReferenceRecord(
                source="feed",
                source_key=str(r["id"]),
                title=r["title"],
                year=citations._year(r["published"]),
                doi=r["doi"],
                arxiv_id=r["arxiv_id"],
                abstract=r["summary"] or None,
                url=r["url"],
            )


# ---------------------------------------------------------------------------
# the network: enrichers, cached content-addressed, armed only by a flag
# ---------------------------------------------------------------------------


def _client() -> httpx.Client:
    """One place to build the HTTP client, so tests can swap the transport."""
    return httpx.Client(timeout=15.0, headers={"User-Agent": "attestation/library"})


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"


def _cached_get(cache_dir: Path, url: str) -> tuple[bytes | None, str | None, str | None]:
    """(body, fetched_at, error). A cache hit keeps the ORIGINAL fetched_at.

    Three attempts on 429/5xx, sleeping Retry-After or 2 s. Every other
    failure is returned as `error`, never raised: a dead network is an absent
    source, and the sync must finish. The cache is the citations spec's:
    content-addressed, never expiring, 0700 dir / 0600 file, and it must not
    launder a wire record into one that looks local.
    """
    path = _cache_path(cache_dir, url)
    if path.is_file():
        rec = json.loads(path.read_text())
        return rec["body"].encode(), rec["fetched_at"], None
    error = None
    for _attempt in range(3):
        try:
            with _client() as client:
                resp = client.get(url)
        except httpx.HTTPError as exc:
            return None, None, f"{type(exc).__name__}: {exc}"
        if resp.status_code == 429 or resp.status_code >= 500:
            error = f"HTTP {resp.status_code}"
            _sleep(float(resp.headers.get("Retry-After") or 2))
            continue
        if resp.status_code != 200:
            return None, None, f"HTTP {resp.status_code}"
        fetched = datetime.now(UTC).date().isoformat()
        existed = cache_dir.is_dir()
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not existed:
            citations._chmod(cache_dir, 0o700)
        path.write_text(json.dumps({"url": url, "fetched_at": fetched, "body": resp.text}))
        citations._chmod(path, 0o600)
        return resp.content, fetched, None
    return None, None, error


class _Enricher:
    """Base: selects the rows it has not yet touched, oldest `updated` first."""

    network = True
    name = ""
    where = "1=1"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE

    def _todo(self, conn: sqlite3.Connection, limit: int | None):
        sql = (
            'SELECT * FROM "references" r WHERE '
            + self.where
            + " AND NOT EXISTS (SELECT 1 FROM reference_sources s"
            " WHERE s.reference_id = r.id AND s.source = ?) ORDER BY r.updated, r.id"
        )
        params: tuple = (self.name,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        return conn.execute(sql, params).fetchall()

    def _miss(self, row, fetched: str | None) -> ReferenceRecord:
        """A record for a row the wire had nothing for: marks the row as tried."""
        return ReferenceRecord(
            source=self.name,
            source_key=row["identity"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            fetched_at=fetched,
        )


class ArxivEnricher(_Enricher):
    """Abstract, authors, title and DOI from the arXiv API, 50 ids per request."""

    name = "arxiv"
    where = "r.arxiv_id IS NOT NULL"

    def records(self, conn: sqlite3.Connection, limit: int | None) -> Iterator[ReferenceRecord]:
        """A record per untouched row with an arXiv id; a miss marks the row tried."""
        rows = self._todo(conn, limit)
        for i in range(0, len(rows), _ARXIV_BATCH):
            batch = rows[i : i + _ARXIV_BATCH]
            ids = ",".join(r["arxiv_id"] for r in batch)
            url = f"http://export.arxiv.org/api/query?id_list={ids}&max_results={len(batch)}"
            body, fetched, _error = _cached_get(self.cache_dir, url)
            found = _parse_arxiv(body, fetched) if body else {}
            for r in batch:
                rec = found.get(r["arxiv_id"])
                if rec is None:
                    yield self._miss(r, fetched)
                else:
                    rec.source_key = r["identity"]
                    yield rec


def _parse_arxiv(body: bytes, fetched: str | None) -> dict[str, ReferenceRecord]:
    """arXiv Atom entries keyed by versionless id.

    defusedxml rather than the stdlib parser: the body came off the wire, and
    an entity-expansion payload would otherwise be parsed with no limit.
    """
    out: dict[str, ReferenceRecord] = {}
    for entry in SafeET.fromstring(body).iter(f"{_ATOM}entry"):
        aid = re.sub(r"v\d+$", "", (entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1])
        if aid:
            out[aid] = _arxiv_record(entry, aid, fetched)
    return out


def _text(entry, tag: str) -> str | None:
    """An Atom element's text, whitespace-collapsed, None when absent or empty."""
    return " ".join((entry.findtext(f"{_ATOM}{tag}") or "").split()) or None


def _arxiv_record(entry, aid: str, fetched: str | None) -> ReferenceRecord:
    doi_el = entry.find(f"{_ARXIV_NS}doi")
    published = entry.findtext(f"{_ATOM}published") or ""
    return ReferenceRecord(
        source="arxiv",
        source_key=aid,
        title=_text(entry, "title"),
        authors=[a.findtext(f"{_ATOM}name") or "" for a in entry.iter(f"{_ATOM}author")],
        year=int(published[:4]) if published[:4].isdigit() else None,
        doi=doi_el.text if doi_el is not None else None,
        arxiv_id=aid,
        abstract=_text(entry, "summary"),
        url=f"https://arxiv.org/abs/{aid}",
        fetched_at=fetched,
    )


class CrossrefEnricher(_Enricher):
    """Venue, authors, title, year and abstract from CrossRef, one request per DOI."""

    name = "crossref"
    where = "r.doi IS NOT NULL"

    def records(self, conn: sqlite3.Connection, limit: int | None) -> Iterator[ReferenceRecord]:
        """A record per untouched row with a DOI, one request each, cached."""
        for r in self._todo(conn, limit):
            url = f"https://api.crossref.org/works/{r['doi']}"
            body, fetched, _error = _cached_get(self.cache_dir, url)
            msg = _json_message(body)
            if msg is None:
                yield self._miss(r, fetched)
            else:
                yield _crossref_record(msg, r["identity"], fetched)


def _joined(msg: dict, key: str) -> str | None:
    """CrossRef lists its title and container-title; one string or None."""
    return " ".join(msg.get(key) or []) or None


def _crossref_authors(msg: dict) -> list[str]:
    names = []
    for a in msg.get("author", []):
        parts = [p for p in (a.get("family"), a.get("given")) if p]
        names.append(", ".join(parts))
    return names


def _crossref_year(msg: dict) -> int | None:
    issued = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return int(issued) if issued else None


def _crossref_record(msg: dict, source_key: str, fetched: str | None) -> ReferenceRecord:
    """A CrossRef `message` as a record; JATS markup stripped from the abstract."""
    return ReferenceRecord(
        source="crossref",
        source_key=source_key,
        title=_joined(msg, "title"),
        authors=_crossref_authors(msg),
        year=_crossref_year(msg),
        doi=msg.get("DOI"),
        venue=_joined(msg, "container-title"),
        abstract=_JATS.sub("", msg.get("abstract") or "").strip() or None,
        url=msg.get("URL"),
        fetched_at=fetched,
    )


def _json_message(body: bytes | None) -> dict | None:
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    msg = data.get("message") if isinstance(data, dict) else None
    return msg if isinstance(msg, dict) else None


class S2Enricher(_Enricher):
    """Reference lists from Semantic Scholar: the only source of citation edges.

    One request per second when the wire is actually touched (cache hits are
    free), honouring Retry-After through `_cached_get`. Each reference with a
    DOI, an arXiv id, or at least a title becomes a `reference_cites` row; one
    with none of those is untraceable and dropped.
    """

    name = "s2"
    where = "(r.doi IS NOT NULL OR r.arxiv_id IS NOT NULL)"

    def records(self, conn: sqlite3.Connection, limit: int | None) -> Iterator[ReferenceRecord]:
        """A record with `cites` per untouched row that has a DOI or arXiv id."""
        for r in self._todo(conn, limit):
            paper = f"DOI:{r['doi']}" if r["doi"] else f"arXiv:{r['arxiv_id']}"
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper}?fields={_S2_FIELDS}"
            on_wire = not _cache_path(self.cache_dir, url).is_file()
            body, fetched, _error = _cached_get(self.cache_dir, url)
            if on_wire:
                _sleep(1.0)
            data = _s2_paper(body)
            if data is None:
                yield self._miss(r, fetched)
                continue
            ext = data.get("externalIds") or {}
            yield ReferenceRecord(
                source=self.name,
                source_key=r["identity"],
                title=data.get("title") or r["title"],
                doi=ext.get("DOI"),
                arxiv_id=ext.get("ArXiv"),
                fetched_at=fetched,
                cites=_s2_cites(data.get("references") or []),
            )


def _s2_paper(body: bytes | None) -> dict | None:
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data if isinstance(data, dict) and "paperId" in data else None


def _s2_cites(references: list) -> list[tuple[str, str | None]]:
    out = []
    for ref in references:
        ext = ref.get("externalIds") or {}
        title = ref.get("title")
        try:
            ident = identity(ext.get("DOI"), ext.get("ArXiv"), title, None)
        except ValueError:
            continue  # no id and no title: untraceable
        out.append((ident, title))
    return out


def readers_from_env(
    conn: sqlite3.Connection,
    *,
    bib_paths=None,
    zotero_path=None,
    cache_dir: Path | None = None,
    sources=None,
) -> list:
    """The reader list, in sync order; the two network flags are read HERE.

    Offline readers first (bibtex, zotero, feed), then the enrichers that
    `ATTEST_CITATION_WEB` and `ATTEST_CITATION_SCHOLAR` arm. `sources` filters by
    reader name so a caller can run one at a time.
    """
    readers = _offline_readers(conn, bib_paths, zotero_path) + _enrichers(cache_dir)
    if sources:
        wanted = set(sources)
        readers = [r for r in readers if r.name in wanted]
    return readers


def _offline_readers(conn, bib_paths, zotero_path) -> list:
    zp = Path(zotero_path) if zotero_path else citations.zotero_path_from_env()
    paths = (
        [Path(p) for p in bib_paths] if bib_paths is not None else citations.bib_paths_from_env()
    )
    readers: list = []
    if paths:
        readers.append(BibtexRecords(paths))
    if zp.is_file():
        readers.append(ZoteroRecords(zp))
    readers.append(FeedRecords(conn))
    return readers


def _enrichers(cache_dir: Path | None) -> list:
    """The network readers the two flags arm -- read here, at construction."""
    readers: list = []
    if citations.web_enabled():
        readers += [ArxivEnricher(cache_dir), CrossrefEnricher(cache_dir)]
    if citations.s2_enabled():
        readers.append(S2Enricher(cache_dir))
    return readers
