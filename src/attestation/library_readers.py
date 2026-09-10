"""Sources of ReferenceRecords: three from disk and the feed, three enrichers
behind flags.

Only the offline readers can INTRODUCE a reference. The network readers
(arXiv, CrossRef, Semantic Scholar) fill fields and citation edges on rows
that already exist, which is what keeps the library the reader's and not the
web's, and keeps `cite.sources`' `offline` answer honest (spec §3.1).
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import httpx
from defusedxml import DefusedXmlException
from defusedxml import ElementTree as SafeET

from attestation import citations
from attestation.library import (
    ReferenceRecord,
    arxiv_from_doi,
    identity,
    normalise_arxiv,
    normalise_doi,
    normalise_title,
)

DEFAULT_CACHE = Path.home() / ".hermes" / "citation-cache"
# Every reader name `readers_from_env(sources=...)` accepts; a typo RAISES,
# the ATTEST_TOOLS rule, rather than silently syncing nothing.
SOURCE_NAMES = ("bibtex", "zotero", "feed", "arxiv", "crossref", "s2")
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_JATS = re.compile(r"<[^>]+>")
_ARXIV_BATCH = 50
_S2_FIELDS = "title,externalIds,references.title,references.externalIds"


def _bib_authors(value: str) -> list[str]:
    """`A and B and others` -> ["A", "B"]; the `others` marker is not a name."""
    return [a.strip() for a in value.split(" and ") if a.strip() and a.strip() != "others"]


_TAG_SPLIT = re.compile(r"[,;]")


def _bib_tags(value: str) -> list[str]:
    """A `keywords` field as tags, folded the way ItemTags folds; unusable ones dropped.

    BibTeX's standard `keywords` field is how a Zotero or JabRef export, or
    the generated molecular-AI example, carries tags into the graph with no
    model call. Same normalisation as the tagger's output so the two agree.
    """
    from attestation.features import TAG_PATTERN

    out: list[str] = []
    for raw in _TAG_SPLIT.split(value or ""):
        tag = raw.strip().lower().replace(" ", "-")
        if tag and re.match(TAG_PATTERN, tag) and tag not in out:
            out.append(tag)
    return out


def _bib_cites(value: str) -> list[tuple[str, str | None]]:
    """A `cites` field (`identity|title; identity`) as (identity, title) pairs.

    Not a standard BibTeX field: it is what `examples/molecular-ai/generate.py`
    writes so that citation edges fetched from Semantic Scholar survive as a
    committed file and load offline. Identities are re-normalised through the
    same rules as `library.identity`, so a hand-typed `DOI:10.1/X` or
    `arxiv:2101.03164v2` still lands on the row the store holds.
    """
    out: list[tuple[str, str | None]] = []
    for raw in (value or "").split(";"):
        entry = raw.strip()
        if not entry:
            continue
        ident, _, title = entry.partition("|")
        out.append((_normalise_identity(ident.strip()), title.strip() or None))
    return out


def _normalise_identity(ident: str) -> str:
    """`kind:value` with the value put through the kind's normaliser."""
    kind, sep, value = ident.partition(":")
    if not sep:
        return ident
    kind = kind.lower()
    if kind == "doi":
        norm = normalise_doi(value)
        return f"doi:{norm}" if norm else _arxiv_form(arxiv_from_doi(value), ident)
    if kind == "arxiv":
        return _arxiv_form(normalise_arxiv(value), ident)
    if kind == "title":
        key, _, year = value.rpartition(":")
        return f"title:{normalise_title(key)}:{year if year.isdigit() else '-'}"
    return ident


def _arxiv_form(arxiv_id: str | None, fallback: str) -> str:
    return f"arxiv:{arxiv_id}" if arxiv_id else fallback


class BibtexRecords:
    """Every entry of every `.bib` file given, through citations' one parser."""

    name = "bibtex"
    network = False

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        # The source string is the file NAME (a path would put a machine
        # string into every tool reply); two files sharing a name are told
        # apart with a suffix so neither's provenance is lost.
        seen: dict[str, int] = {}
        self.labels: dict[Path, str] = {}
        for p in self.paths:
            n = seen.get(p.name, 0) + 1
            seen[p.name] = n
            self.labels[p] = f"bibtex:{p.name}" if n == 1 else f"bibtex:{p.name}#{n}"

    def records(self) -> Iterator[ReferenceRecord]:
        """One record per entry with a title; a missing file is an absent source."""
        for path in self.paths:
            if not path.is_file():
                continue
            for key, f in citations._parse_bib_entries(path.read_text(errors="replace")):
                is_arxiv = f.get("archiveprefix", "").lower() == "arxiv"
                yield ReferenceRecord(
                    source=self.labels[path],
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
                    tags=_bib_tags(f.get("keywords", "")),
                    cites=_bib_cites(f.get("cites", "")),
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


_S2_PACE_SECONDS = 3.0  # unauthenticated S2 rate-limited 1 rps flat (429 x3) on 2026-09-05
_RETRY_AFTER_DEFAULT = 10.0
_RETRY_AFTER_MAX = 60.0  # one header must not park attest-mcp for an hour
_FETCH_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError)  # a bad stored id is an InvalidURL


def _retry_after(value: str | None) -> float:
    """Seconds to wait from a Retry-After header: an integer, an HTTP-date, or
    junk (`nan`, `1e12`, a typo) -- never a crash, never more than a minute."""
    if not value:
        return _RETRY_AFTER_DEFAULT
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            return _RETRY_AFTER_DEFAULT
    if not math.isfinite(seconds):
        return _RETRY_AFTER_DEFAULT
    return min(max(seconds, _RETRY_AFTER_DEFAULT), _RETRY_AFTER_MAX)


def _client() -> httpx.Client:
    """One place to build the HTTP client, so tests can swap the transport.

    Redirects are followed: export.arxiv.org answers plain http with a 301 to
    https, which read as a miss for every arXiv seed on the first generation
    of examples/molecular-ai.
    """
    return httpx.Client(
        timeout=15.0, headers={"User-Agent": "attestation/library"}, follow_redirects=True
    )


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
    resp, error = _fetch_with_backoff(url)
    if resp is None:
        return None, None, error
    fetched = datetime.now(UTC).date().isoformat()
    _write_cache(cache_dir, path, url, fetched, resp.text)
    return resp.content, fetched, None


def _fetch_with_backoff(url: str) -> tuple[httpx.Response | None, str | None]:
    """A 200 response, or (None, why). One back-off for a 429, then the row
    is given up for this pass: a rate limit that outlasts one Retry-After is
    the pool being exhausted, and hammering it delays every row behind it. A
    5xx gets a second retry, since those are usually momentary."""
    error = None
    for attempt in range(3):
        try:
            with _client() as client:
                resp = client.get(url)
        except _FETCH_ERRORS as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if resp.status_code == 200:
            return resp, None
        error = f"HTTP {resp.status_code}"
        if resp.status_code != 429 and resp.status_code < 500:
            return None, error
        if resp.status_code == 429 and attempt >= 1:
            return None, error
        if attempt < 2:  # no sleep after the last attempt: nothing follows it
            _sleep(_retry_after(resp.headers.get("Retry-After")))
    return None, error


# What an identifier must look like before it is put on a URL. A stored id
# that fails this is one row marked tried-with-error, never a batch poisoned
# (one `#frag` in an arXiv id truncated the id_list and marked 49 other rows
# as answered; review round 2).
_ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})$")
_DOI = re.compile(r"^10\.\d+/[^\s#?]+$")  # separators are the concern, not the prefix width


def _well_formed(row) -> str | None:
    """Why this row's identifiers cannot go on a URL, or None when they can."""
    if row["arxiv_id"] and not _ARXIV_ID.match(row["arxiv_id"]):
        return f"malformed arXiv id {row['arxiv_id']!r}"
    if row["doi"] and (not _DOI.match(row["doi"]) or "/../" in f"/{row['doi']}/"):
        return f"malformed DOI {row['doi']!r}"
    return None


def _forget(cache_dir: Path, url: str) -> None:
    """Drop a cached body that did not parse, so the next sync asks the wire
    again instead of raising from the same file forever."""
    path = _cache_path(cache_dir, url)
    if path.is_file():
        path.unlink()


def _write_cache(cache_dir: Path, path: Path, url: str, fetched: str, body: str) -> None:
    """0700 dir / 0600 file, applied at creation only (see citations._write_cached)."""
    existed = cache_dir.is_dir()
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not existed:
        citations._chmod(cache_dir, 0o700)
    path.write_text(json.dumps({"url": url, "fetched_at": fetched, "body": body}))
    citations._chmod(path, 0o600)


class _Enricher:
    """Base: selects the rows it has not yet touched, oldest `updated` first.

    A row is "touched" once this source has a `reference_sources` row for it.
    A definite answer (a record, or a 404 / empty payload) writes one, so the
    row is not asked again; a transport failure or an exhausted 429 writes
    NOTHING, so the row is retried on the next sync -- the first generation of
    examples/molecular-ai marked 24 papers as tried after S2 rate-limited them,
    which a re-run could then never repair. Those failures are counted in
    `errors` and reported as `failed`.
    """

    network = True
    name = ""
    where = "1=1"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self.errors: list[str] = []

    def _transient(self, row, error: str | None) -> bool:
        """Record and skip a failure that a later sync should retry."""
        if error is None or error.startswith("HTTP 404"):
            return False
        self.errors.append(f"{row['identity']}: {error}")
        return True

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
        rows = []
        for r in self._todo(conn, limit):
            if _well_formed(r):
                yield self._miss(r, None)  # marked tried: an id that cannot go on a URL never will
            else:
                rows.append(r)
        for i in range(0, len(rows), _ARXIV_BATCH):
            batch = rows[i : i + _ARXIV_BATCH]
            found, fetched, error = self._fetch_batch(batch)
            for r in batch:
                if not found and self._transient(r, error):
                    continue
                rec = found.get(r["arxiv_id"])
                if rec is None:
                    yield self._miss(r, fetched)
                else:
                    rec.source_key = r["identity"]
                    yield rec

    def _fetch_batch(self, batch) -> tuple[dict[str, ReferenceRecord], str | None, str | None]:
        """One id_list request: (records by id, fetched_at, error)."""
        ids = quote(",".join(r["arxiv_id"] for r in batch), safe=",/")
        url = f"https://export.arxiv.org/api/query?id_list={ids}&max_results={len(batch)}"
        body, fetched, error = _cached_get(self.cache_dir, url)
        found: dict[str, ReferenceRecord] = {}
        if body is not None:
            try:
                found = _parse_arxiv(body, fetched)
            except (SafeET.ParseError, DefusedXmlException) as exc:
                # A 200 that is not the feed it claims: not a miss (the
                # row is not tried), not a crash (the sync finishes).
                _forget(self.cache_dir, url)
                error = f"unparseable body: {exc}"
        return found, fetched, error


def _parse_arxiv(body: bytes, fetched: str | None) -> dict[str, ReferenceRecord]:
    """arXiv Atom entries keyed by versionless id.

    defusedxml rather than the stdlib parser: the body came off the wire, and
    an entity-expansion payload would otherwise be parsed with no limit.
    """
    out: dict[str, ReferenceRecord] = {}
    for entry in SafeET.fromstring(body).iter(f"{_ATOM}entry"):
        # `http://arxiv.org/abs/cond-mat/0301234v1`: everything after /abs/,
        # so an old-style id keeps its archive prefix, as `identity` does.
        aid = normalise_arxiv((entry.findtext(f"{_ATOM}id") or "").partition("/abs/")[2])
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
            if _well_formed(r):
                yield self._miss(r, None)  # marked tried: an id that cannot go on a URL never will
                continue
            url = f"https://api.crossref.org/works/{quote(r['doi'], safe='/')}"
            body, fetched, error = _cached_get(self.cache_dir, url)
            msg = None
            if body is not None:  # an empty 200 is unparseable, not an answer
                try:
                    msg = _json_message(body)
                except ValueError as exc:
                    _forget(self.cache_dir, url)
                    error = f"unparseable body: {exc}"
            if msg is None and self._transient(r, error):
                continue
            if msg is None:
                yield self._miss(r, fetched)
            else:
                yield _crossref_record(msg, r, fetched)


def _joined(msg: dict, key: str) -> str | None:
    """CrossRef lists its title and container-title; one string or None.

    HTML entities are decoded: CrossRef sent `Multiscale Modeling &amp;amp;
    Simulation` for SIAM's journal in the first molecular-AI generation.
    """
    value = msg.get(key)
    parts = (
        [html.unescape(p) for p in value if isinstance(p, str)] if isinstance(value, list) else []
    )
    return " ".join(parts) or None


def _crossref_authors(msg: dict) -> list[str]:
    names = []
    for a in msg.get("author") or []:
        if not isinstance(a, dict):
            continue
        parts = [p for p in (a.get("family"), a.get("given")) if isinstance(p, str) and p]
        if parts:
            names.append(", ".join(parts))
    return names


def _crossref_year(msg: dict) -> int | None:
    issued = msg.get("issued")
    parts = issued.get("date-parts") if isinstance(issued, dict) else None
    first = parts[0] if isinstance(parts, list) and parts else None
    year = first[0] if isinstance(first, list) and first else None
    return int(year) if isinstance(year, int | str) and str(year).isdigit() else None


_ABSTRACT_HEADING = re.compile(r"^\s*abstract\s*[:.]?\s*", re.IGNORECASE)


_JATS_SUP = re.compile(r"<jats:sup>.*?</jats:sup>", re.DOTALL)
# Science's "abstract" for some papers is the editor's summary, signed with
# initials ("...Anfinsen won a Nobel prize... —VV"); that is not the paper's
# abstract and the embedding should not see it.
_SIGNED_BLURB = re.compile(r"[—–-]\s*[A-Z]{2,4}\s*$")


def _crossref_abstract(msg: dict) -> str | None:
    """JATS markup stripped (citation superscripts with their numbers),
    entities decoded, a leading `Abstract` heading (Nature's and Science's
    habit) dropped, a signed editor's summary dropped entirely."""
    raw = msg.get("abstract")
    if not isinstance(raw, str):
        return None
    text = html.unescape(_JATS.sub("", _JATS_SUP.sub("", raw)))
    text = " ".join(text.split())
    text = _ABSTRACT_HEADING.sub("", text, count=1).strip()
    if not text or _SIGNED_BLURB.search(text):
        return None
    return text


def _crossref_record(msg: dict, row, fetched: str | None) -> ReferenceRecord:
    """A CrossRef `message` as a record for `row`; JATS markup stripped from
    the abstract. The row's own ids ride along, so a shapeless message (no
    DOI, no title) is a titleless miss on that row rather than a record with
    nothing to identify it."""
    doi = msg.get("DOI") if isinstance(msg.get("DOI"), str) else None
    return ReferenceRecord(
        source="crossref",
        source_key=row["identity"],
        title=_joined(msg, "title"),
        authors=_crossref_authors(msg),
        year=_crossref_year(msg),
        doi=doi or row["doi"],
        arxiv_id=row["arxiv_id"],
        venue=_joined(msg, "container-title"),
        abstract=_crossref_abstract(msg),
        url=msg.get("URL") if isinstance(msg.get("URL"), str) else None,
        fetched_at=fetched,
    )


def _json_object(body: bytes | None) -> dict:
    """The JSON object a 200 body must be; anything else raises ValueError so
    the caller forgets the cached body and leaves the row for the next sync
    (an interstitial page cached as an answer marked the row tried forever,
    review round 2)."""
    if not body:
        raise ValueError("empty body")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _json_message(body: bytes | None) -> dict | None:
    """CrossRef's `message` object; None when the reply parsed but holds none.
    Raises ValueError when the body is not JSON at all."""
    msg = _json_object(body).get("message")
    return msg if isinstance(msg, dict) else None


class S2Enricher(_Enricher):
    """Reference lists from Semantic Scholar: the only source of citation edges.

    One request every `_S2_PACE_SECONDS` when the wire is actually touched
    (cache hits are free), honouring Retry-After through `_cached_get`. Each
    reference with a DOI, an arXiv id, or a title of at least four words
    becomes a `reference_cites` row; one with none of those is untraceable
    and dropped -- see `_s2_cites` for what the short-title rule removes.
    """

    name = "s2"
    where = "(r.doi IS NOT NULL OR r.arxiv_id IS NOT NULL)"

    def records(self, conn: sqlite3.Connection, limit: int | None) -> Iterator[ReferenceRecord]:
        """A record with `cites` per untouched row that has a DOI or arXiv id."""
        for r in self._todo(conn, limit):
            if _well_formed(r):
                yield self._miss(r, None)  # marked tried: an id that cannot go on a URL never will
                continue
            data, fetched, error = self._fetch_paper(r)
            if data is None and self._transient(r, error):
                continue
            if data is None:
                yield self._miss(r, fetched)
                continue
            yield _s2_record(data, r, fetched)

    def _fetch_paper(self, r) -> tuple[dict | None, str | None, str | None]:
        """One paper request, paced when it touches the wire: (paper, fetched_at, error)."""
        paper = f"DOI:{r['doi']}" if r["doi"] else f"arXiv:{r['arxiv_id']}"
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{quote(paper, safe=':/')}?fields={_S2_FIELDS}"
        )
        on_wire = not _cache_path(self.cache_dir, url).is_file()
        body, fetched, error = _cached_get(self.cache_dir, url)
        if on_wire:
            _sleep(_S2_PACE_SECONDS)
        data = None
        if body is not None:
            try:
                data = _s2_paper(body)
            except ValueError as exc:
                _forget(self.cache_dir, url)
                error = f"unparseable body: {exc}"
        return data, fetched, error


def _s2_record(data: dict, row, fetched: str | None) -> ReferenceRecord:
    """A Semantic Scholar paper as a record: its ids and its reference list."""
    ext = _ids(data)
    title = data.get("title")
    refs = data.get("references")
    return ReferenceRecord(
        source="s2",
        source_key=row["identity"],
        title=title if isinstance(title, str) and title else row["title"],
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        fetched_at=fetched,
        cites=_s2_cites(refs if isinstance(refs, list) else []),
    )


def _ids(paper: dict) -> dict:
    """A paper's `externalIds` as a dict of strings, whatever shape the wire sent."""
    ext = paper.get("externalIds")
    if not isinstance(ext, dict):
        return {}
    return {k: v for k, v in ext.items() if isinstance(v, str) and v}


_MIN_TITLE_WORDS = 4
_WORD = re.compile(r"[a-zA-Z]{3,}")


def _traceable_title(title) -> bool:
    """A reference-list entry with no id is kept only when its title could
    name a paper: at least four words of three or more letters. Semantic Scholar's parsed
    reference lists carry `Phys. Rev. B`, `Learn`, `AND T`, `AUTHOR
    CONTRIBUTIONS` and NeurIPS checklist questions as titled references;
    a third of GAP's 19 edges were such stubs in the first generation."""
    if not isinstance(title, str) or "{" in title or "}" in title:
        return False  # a brace is an equation fragment ("Eτ} and Υ and Γ are MLPs"), never a title
    return len(_WORD.findall(title)) >= _MIN_TITLE_WORDS


def _s2_paper(body: bytes | None) -> dict | None:
    """A Semantic Scholar paper object; None when the reply parsed but is not
    one. Raises ValueError when the body is not JSON at all."""
    data = _json_object(body)
    return data if "paperId" in data else None


def _s2_cites(references: list) -> list[tuple[str, str | None]]:
    out = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        ext = _ids(ref)
        title = ref.get("title") if isinstance(ref.get("title"), str) else None
        if not ext.get("DOI") and not ext.get("ArXiv") and not _traceable_title(title):
            continue  # no id and no title that could name a paper: untraceable
        try:
            ident = identity(ext.get("DOI"), ext.get("ArXiv"), title, None)
        except ValueError:
            continue
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
        if unknown := sorted(wanted - set(SOURCE_NAMES)):
            raise ValueError(
                f"unknown source(s) {', '.join(unknown)}; the readers are {', '.join(SOURCE_NAMES)}"
            )
        readers = [r for r in readers if r.name in wanted]
    return readers


def unarmed(sources) -> list[str]:
    """The requested enrichers whose flag is unset -- so `cite.sync(sources=["s2"])`
    with no flag says "not armed" instead of the "no sources configured" that
    reads as a sync that happened."""
    wanted = set(sources or ())
    out = []
    if not citations.web_enabled():
        out += sorted(wanted & {"arxiv", "crossref"})
    if not citations.s2_enabled():
        out += sorted(wanted & {"s2"})
    return out


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
