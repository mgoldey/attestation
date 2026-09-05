"""The reference library: one row per paper, fed by many sources.

Identity is a pure function (DOI, else versionless arXiv id, else normalised
title and year) so that Zotero, three `.bib` files and the feed can all fill
ONE row. Merge never overwrites: an empty field takes a value, a differing
value is recorded as a conflict on the contributing source row. See
docs/superpowers/specs/2026-09-05-library-store-design.md.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import numpy as np

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^(?:arxiv:)\s*", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_doi(doi: str | None) -> str | None:
    """Lowercase, scheme and doi.org prefix stripped; None for empty."""
    if not doi:
        return None
    out = _DOI_PREFIX.sub("", doi.strip()).lower()
    return out or None


def normalise_arxiv(arxiv_id: str | None) -> str | None:
    """`arXiv:2106.02347v3` -> `2106.02347`; old-style ids kept whole."""
    if not arxiv_id:
        return None
    out = _ARXIV_PREFIX.sub("", arxiv_id.strip())
    out = _ARXIV_VERSION.sub("", out)
    return out or None


def normalise_title(title: str) -> str:
    """NFKD, combining marks dropped, lowercase, non-alphanumerics collapsed.

    Leading articles are kept on purpose: dropping them merges "A survey"
    with "Survey", which are different papers more often than not.
    """
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", ascii_only.lower()).strip()


def identity(doi: str | None, arxiv_id: str | None, title: str | None, year: int | None) -> str:
    """The one string two records must share to be the same paper.

    DOI beats arXiv because a preprint that is later published gains a DOI
    while keeping its arXiv id; a row carries both columns so a record that
    knows only the arXiv id still finds it (see `upsert`).
    """
    if d := normalise_doi(doi):
        return f"doi:{d}"
    if a := normalise_arxiv(arxiv_id):
        return f"arxiv:{a}"
    if title and (t := normalise_title(title)):
        return f"title:{t}:{year if year is not None else '-'}"
    raise ValueError("a reference needs a DOI, an arXiv id, or a title")


def _authors_extend(existing: list, incoming: list) -> bool:
    """True when `incoming` is `existing` plus more names (a truncated list filled in)."""
    if len(incoming) <= len(existing):
        return False
    norm = [normalise_title(a) for a in existing]
    return [normalise_title(a) for a in incoming[: len(existing)]] == norm


def _same(name: str, kept, offered) -> bool:
    """Equal after the normalisation the field `name` deserves."""
    if name == "authors":
        return [normalise_title(a) for a in kept] == [normalise_title(a) for a in offered]
    if name == "title":
        return normalise_title(kept) == normalise_title(offered)
    return kept == offered


def merge(existing: dict, incoming: dict) -> tuple[dict, dict]:
    """Fill empty fields from `incoming`; keep non-empty ones; record conflicts.

    Deliberately dumb -- longer abstracts do not win, first does -- so that
    nothing is ever overwritten silently. `cite.lookup` shows every source
    row, so a reader can see a disagreement rather than lose it.
    """
    merged = dict(existing)
    conflicts: dict = {}
    for name, offered in incoming.items():
        if offered in (None, "", [], {}):
            continue
        kept = merged.get(name)
        if kept in (None, "", [], {}):
            merged[name] = offered
            continue
        if name == "authors" and _authors_extend(kept, offered):
            merged[name] = list(offered)
            continue
        if _same(name, kept, offered):
            continue
        conflicts[name] = {"kept": kept, "offered": offered}
    return merged, conflicts


@dataclass
class ReferenceRecord:
    """What one source says about one paper. The incoming shape for `upsert`.

    `source` names the reader (`bibtex:<path>`, `zotero`, `feed`, `arxiv`,
    `crossref`, `s2`); `source_key` is that reader's own handle on the record
    (a .bib key, a Zotero key, an items.id, an external id). `fetched_at` is
    None for anything read from disk -- the provenance pair the citations spec
    put on every record, kept here per contribution.
    """

    source: str
    source_key: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    bib_key: str | None = None
    fetched_at: str | None = None
    tags: list[str] = field(default_factory=list)
    cites: list[tuple[str, str | None]] = field(default_factory=list)

    def fields(self) -> dict:
        """The `references` columns this record can fill (empty ones omitted)."""
        out = {
            "doi": normalise_doi(self.doi),
            "arxiv_id": normalise_arxiv(self.arxiv_id),
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "url": self.url,
            "bib_key": self.bib_key,
        }
        return {k: v for k, v in out.items() if v not in (None, "", [])}


# ---------------------------------------------------------------------------
# upsert and sync
# ---------------------------------------------------------------------------

_COLUMNS = ("doi", "arxiv_id", "title", "authors", "year", "venue", "abstract", "url", "bib_key")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _find(conn: sqlite3.Connection, fields: dict, ident: str):
    """The row this record belongs to: by identity, then DOI, then arXiv id."""
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', ident),
        ('SELECT * FROM "references" WHERE doi = ?', fields.get("doi")),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', fields.get("arxiv_id")),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


def _insert(conn: sqlite3.Connection, ident: str, fields: dict, now: str) -> int:
    cur = conn.execute(
        'INSERT INTO "references"(identity, doi, arxiv_id, title, authors, year, venue,'
        " abstract, url, bib_key, first_seen, updated)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ident,
            fields.get("doi"),
            fields.get("arxiv_id"),
            fields["title"],
            json.dumps(fields.get("authors", [])),
            fields.get("year"),
            fields.get("venue"),
            fields.get("abstract"),
            fields.get("url"),
            fields.get("bib_key"),
            now,
            now,
        ),
    )
    if cur.lastrowid is None:  # pragma: no cover - sqlite always sets it after INSERT
        raise RuntimeError("INSERT returned no rowid")
    return cur.lastrowid


def _update(conn: sqlite3.Connection, row, fields: dict, now: str) -> tuple[str, dict]:
    """Merge `fields` into `row`; returns (merged|unchanged, conflicts)."""
    existing = {c: row[c] for c in _COLUMNS}
    existing["authors"] = json.loads(row["authors"])
    merged, conflicts = merge(existing, fields)
    changed = {c: merged[c] for c in _COLUMNS if merged.get(c) != existing.get(c)}
    new_ident = identity(
        merged.get("doi"), merged.get("arxiv_id"), merged.get("title"), merged.get("year")
    )
    if new_ident != row["identity"]:
        changed["identity"] = new_ident
    if not changed:
        return "unchanged", conflicts
    sets = ", ".join(f"{c} = ?" for c in changed) + ", updated = ?"
    vals = [json.dumps(v) if c == "authors" else v for c, v in changed.items()] + [now]
    conn.execute(f'UPDATE "references" SET {sets} WHERE id = ?', (*vals, row["id"]))
    return "merged", conflicts


def upsert(conn: sqlite3.Connection, rec: ReferenceRecord) -> tuple[int, str]:
    """Merge one record into the store. Returns (id, added|merged|unchanged).

    Lookup order: identity, DOI, arXiv id -- so a record that knows only the
    arXiv id still finds the row a DOI-bearing record created, and a row
    created from an arXiv id is upgraded to the DOI identity when one arrives.
    The source row is written once per (source, key) and carries the fields
    that source offered plus any conflict merge() refused.
    """
    fields = rec.fields()
    ident = identity(
        fields.get("doi"), fields.get("arxiv_id"), fields.get("title"), fields.get("year")
    )
    now = _now()
    row = _find(conn, fields, ident)
    if row is None:
        rid, how, conflicts = _insert(conn, ident, fields, now), "added", {}
    else:
        rid = row["id"]
        how, conflicts = _update(conn, row, fields, now)
    seen = conn.execute(
        "SELECT 1 FROM reference_sources WHERE reference_id = ? AND source = ? AND source_key = ?",
        (rid, rec.source, rec.source_key),
    ).fetchone()
    if seen is None:
        raw = {
            "fields": {k: v for k, v in fields.items() if k != "abstract"},
            "conflicts": conflicts,
        }
        conn.execute(
            "INSERT INTO reference_sources(reference_id, source, source_key, fetched_at, raw)"
            " VALUES (?, ?, ?, ?, ?)",
            (rid, rec.source, rec.source_key, rec.fetched_at, json.dumps(raw)),
        )
    if rec.tags:
        # Tags from a file (a .bib `keywords` field) only ever ADD: they never
        # delete what the tagger wrote, and the tagger never deletes them.
        conn.executemany(
            "INSERT OR IGNORE INTO reference_tags(reference_id, tag) VALUES (?, ?)",
            [(rid, t) for t in dict.fromkeys(rec.tags)],
        )
    for cited_identity, cited_title in rec.cites:
        conn.execute(
            "INSERT OR IGNORE INTO reference_cites"
            "(citing_id, cited_identity, cited_title, source, fetched_at) VALUES (?, ?, ?, ?, ?)",
            (rid, cited_identity, cited_title, rec.source, rec.fetched_at or now),
        )
    return rid, how


@dataclass
class SyncReport:
    """Structure, not prose: the caller is a model or a CLI printer."""

    sources: dict = field(default_factory=dict)
    embedded: int = 0
    unembedded: int = 0
    embed_error: str | None = None
    conflicts: int = 0
    conflict_samples: list = field(default_factory=list)

    def bucket(self, name: str) -> dict:
        """The per-source counters for `name`, created on first use."""
        return self.sources.setdefault(
            name, {"seen": 0, "added": 0, "merged": 0, "unchanged": 0, "enriched": 0, "failed": 0}
        )

    def to_dict(self) -> dict:
        """The wire shape of the report; conflict samples capped at five."""
        return {
            "sources": self.sources,
            "embedded": self.embedded,
            "unembedded": self.unembedded,
            "embed_error": self.embed_error,
            "conflicts": self.conflicts,
            "conflict_samples": self.conflict_samples[:5],
        }


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


_UNEMBEDDED = (
    'SELECT count(*) FROM "references" r WHERE NOT EXISTS'
    " (SELECT 1 FROM reference_vectors v WHERE v.rowid = r.id)"
)


def _conflicts_of(conn: sqlite3.Connection, rec: ReferenceRecord) -> dict:
    raw = conn.execute(
        "SELECT raw FROM reference_sources WHERE source = ? AND source_key = ?",
        (rec.source, rec.source_key),
    ).fetchone()
    return json.loads(raw["raw"]).get("conflicts", {}) if raw else {}


def sync(conn: sqlite3.Connection, readers, *, embedder=None, limit: int | None = None):
    """Run every reader in order, then embed rows without a vector.

    Offline readers introduce rows; enrichers (`network=True`) only fill rows
    that exist -- their `records(conn, limit)` selects the rows themselves.
    Each reader is its own short transaction, the ingest discipline, so
    `attest serve` keeps working alongside.
    """
    report = SyncReport()
    for reader in readers:
        bucket = report.bucket(reader.name)
        records = reader.records(conn, limit) if reader.network else reader.records()
        for rec in records:
            bucket["seen"] += 1
            if rec.title is None:
                # An offline record with no title cannot be a row; an enricher
                # that found nothing still marks the row as tried.
                bucket["failed"] += 1
                continue
            _, how = upsert(conn, rec)
            bucket["enriched" if reader.network else how] += 1
            if how == "merged" or reader.network:
                conflicts = _conflicts_of(conn, rec)
                report.conflicts += len(conflicts)
                report.conflict_samples.extend((rec.source_key, f) for f in conflicts)
        conn.commit()
    if embedder is not None:
        report.embedded, report.unembedded, report.embed_error = embed_missing(
            conn, embedder, limit
        )
    else:
        report.unembedded = _count(conn, _UNEMBEDDED)
    return report


def embed_missing(conn, embedder, limit: int | None) -> tuple[int, int, str | None]:
    """Embed rows with no vector: (embedded, still_missing, error).

    One embed call per row outside any transaction, then one short write --
    the ingest discipline. An embedder that cannot be reached stops the pass
    and is reported once; the rows stay unembedded and `search` degrades to
    fielded, which is rank.py's policy (serve what you have, never 500).
    """
    sql = (
        'SELECT r.id, r.title, r.abstract FROM "references" r WHERE NOT EXISTS'
        " (SELECT 1 FROM reference_vectors v WHERE v.rowid = r.id) ORDER BY r.id"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    done, error = 0, None
    for row in rows:
        try:
            vec = embedder.embed_document(row["title"], row["abstract"] or "")
        except (httpx.HTTPError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        conn.execute(
            "INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)",
            (row["id"], np.asarray(vec, dtype=np.float32).tobytes()),
        )
        done += 1
    conn.commit()
    return done, _count(conn, _UNEMBEDDED), error


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

_ID_QUERY = re.compile(
    r"^(10\.\d{4,9}/\S+|(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?|(?:arxiv:)?[a-z\-]+/\d{7})$",
    re.IGNORECASE,
)
# A literal match moves a hit up; it never excludes one. Measured on the feed:
# flooring on a literal made all 711 "llm" matches tie.
_LITERAL_BOOST = 0.02


@dataclass
class SearchHit:
    """One library row as a search result, with its sources and tags beside it."""

    id: int
    identity: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    url: str | None
    bib_key: str | None
    venue: str | None
    sources: list[str]
    tags: list[str]
    n_tags: int
    similarity: float | None = None

    def to_row(self) -> dict:
        """The wire projection: the same budgets as `Reference.to_row` and
        `RankedItem.to_row` (authors 6, tags 3, the true counts beside them,
        title and venue clipped with the cut made visible). Measured against
        test_library_tools' worst-case row: 13 rows at 223-char titles came to
        7,736 characters, over the 7,000 a 2B model can render; at 90 they fit.
        """
        from attestation.rank import MAX_SOURCE_CHARS, MAX_URL_CHARS, _clip_field, _clip_title

        return {
            "id": self.id,
            "key": self.bib_key or self.identity,
            "title": _clip_title(self.title),
            "authors": self.authors[:6],
            "n_authors": len(self.authors),
            "year": self.year,
            "doi": self.doi,
            "url": _clip_field(self.url, MAX_URL_CHARS) if self.url else None,
            "venue": _clip_field(self.venue, MAX_SOURCE_CHARS) if self.venue else None,
            "sources": self.sources,
            "tags": self.tags[:3],
            "n_tags": self.n_tags,
            "similarity": self.similarity,
        }


@dataclass
class SearchResult:
    """Hits plus the two facts a caller must relay: was it semantic, and why not."""

    hits: list[SearchHit]
    semantic: bool
    caveat: str | None
    n_matches: int


def _hit(conn: sqlite3.Connection, row, similarity: float | None = None) -> SearchHit:
    sources = [
        r["source"]
        for r in conn.execute(
            "SELECT source FROM reference_sources WHERE reference_id = ? ORDER BY source",
            (row["id"],),
        )
    ]
    tags = [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM reference_tags WHERE reference_id = ? ORDER BY tag", (row["id"],)
        )
    ]
    return SearchHit(
        id=row["id"],
        identity=row["identity"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        year=row["year"],
        doi=row["doi"],
        url=row["url"],
        bib_key=row["bib_key"],
        venue=row["venue"],
        sources=sources,
        tags=tags,
        n_tags=len(tags),
        similarity=similarity,
    )


def _fielded_where(author, year, year_from, year_to, tag, source) -> tuple[str, list]:
    where: list[str] = []
    params: list = []
    if author:
        where.append("EXISTS (SELECT 1 FROM json_each(r.authors) a WHERE lower(a.value) LIKE ?)")
        params.append(f"%{author.lower()}%")
    if year is not None:
        where.append("r.year = ?")
        params.append(year)
    if year_from is not None:
        where.append("r.year >= ?")
        params.append(year_from)
    if year_to is not None:
        where.append("r.year <= ?")
        params.append(year_to)
    if tag:
        where.append(
            "EXISTS (SELECT 1 FROM reference_tags t WHERE t.reference_id = r.id AND t.tag = ?)"
        )
        params.append(tag)
    if source:
        where.append(
            "EXISTS (SELECT 1 FROM reference_sources s"
            " WHERE s.reference_id = r.id AND s.source LIKE ?)"
        )
        params.append(f"{source}%")
    return (" AND ".join(where) or "1=1"), params


def lookup_row(conn: sqlite3.Connection, key: str):
    """A row by identity, DOI, arXiv id, or bib key -- the direct forms."""
    k = key.strip()
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', k.lower()),
        ('SELECT * FROM "references" WHERE doi = ?', normalise_doi(k)),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', normalise_arxiv(k)),
        ('SELECT * FROM "references" WHERE bib_key = ? COLLATE NOCASE', k),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


def _semantic(conn, embedder, q: str, where: str, params: list, limit: int) -> SearchResult | None:
    """KNN over reference_vectors, the relative floor, the filters, the boost;
    None when the wire could not be reached or nothing cleared the floor."""
    from attestation.rank import apply_relevance_floor, vector_search

    try:
        raw = vector_search(conn, embedder, q, k=4 * limit, table="reference_vectors")
    except (httpx.HTTPError, OSError):
        return None
    sims = apply_relevance_floor(raw)
    if not sims:
        return None
    marks = ",".join("?" * len(sims))
    rows = conn.execute(
        f'SELECT * FROM "references" r WHERE r.id IN ({marks}) AND {where}',
        (*sims.keys(), *params),
    ).fetchall()
    words = [w for w in normalise_title(q).split() if len(w) > 2]

    def _score(r) -> float:
        text = normalise_title(f"{r['title']} {r['abstract'] or ''}")
        return sims[r["id"]] + _LITERAL_BOOST * sum(w in text for w in words)

    rows.sort(key=_score, reverse=True)
    hits = [_hit(conn, r, round(sims[r["id"]], 4)) for r in rows[:limit]]
    n_vectors = _count(conn, "SELECT count(*) FROM reference_vectors")
    n_refs = _count(conn, 'SELECT count(*) FROM "references"')
    caveat = None
    if n_vectors < n_refs:
        caveat = f"{n_vectors} of {n_refs} references are embedded; run `attest library embed`"
    return SearchResult(hits, semantic=True, caveat=caveat, n_matches=len(rows))


def _substring(conn, q: str, where: str, params: list, limit: int, reason: str) -> SearchResult:
    like = f"%{q.lower()}%" if q else "%"
    rows = conn.execute(
        f'SELECT * FROM "references" r WHERE {where} AND ('
        " lower(r.title) LIKE ? OR lower(coalesce(r.abstract, '')) LIKE ?"
        " OR lower(r.authors) LIKE ? OR lower(coalesce(r.bib_key, '')) LIKE ?)"
        " ORDER BY r.year DESC, r.title",
        (*params, like, like, like, like),
    ).fetchall()
    return SearchResult(
        [_hit(conn, r) for r in rows[:limit]], semantic=False, caveat=reason, n_matches=len(rows)
    )


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    embedder=None,
    author: str | None = None,
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    tag: str | None = None,
    source: str | None = None,
    limit: int = 10,
) -> SearchResult:
    """Semantic when it can be, fielded when it must be, and it says which.

    A query that is an identifier or a bib key is a direct lookup. Otherwise
    the feed's search shape: KNN over reference_vectors for 4x limit
    candidates, the relative relevance floor, fielded filters, a literal
    boost that never excludes. Falls back to substring over title, abstract,
    authors and key when there is no embedder, no vectors, or no semantic
    hit -- with a caveat a caller cannot mistake for a semantic answer.
    """
    q = (query or "").strip()
    where, params = _fielded_where(author, year, year_from, year_to, tag, source)
    if q and (row := lookup_row(conn, q)):
        return SearchResult([_hit(conn, row)], semantic=False, caveat=None, n_matches=1)
    n_vectors = _count(conn, "SELECT count(*) FROM reference_vectors")
    if q and embedder is not None and n_vectors:
        result = _semantic(conn, embedder, q, where, params, limit)
        if result is not None:
            return result
        reason = "no semantic hit cleared the relevance floor; substring results"
    elif embedder is None:
        reason = "substring search only (no embedder); run `attest library embed` for semantic"
    elif not n_vectors:
        reason = "substring search only (no vectors yet); run `attest library embed`"
    else:
        reason = "substring search only (empty query)"
    return _substring(conn, q, where, params, limit, reason)


def to_reference(conn: sqlite3.Connection, row):
    """A store row as the `Reference` the cite.* tools already emit.

    `source` is `library:<first contributing source>` and `fetched_at` that
    source's, so the provenance pair survives the projection.
    """
    from attestation.citations import Reference

    first = conn.execute(
        "SELECT source, fetched_at FROM reference_sources WHERE reference_id = ?"
        " ORDER BY fetched_at IS NOT NULL, source LIMIT 1",
        (row["id"],),
    ).fetchone()
    return Reference(
        key=row["bib_key"] or row["identity"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        year=row["year"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        url=row["url"],
        source="library:" + (first["source"] if first else "?"),
        fetched_at=first["fetched_at"] if first else None,
    )


# ---------------------------------------------------------------------------
# citation neighbourhood
# ---------------------------------------------------------------------------

MAX_NEIGHBOURS = 20


@dataclass
class Neighbour:
    """One end of a citation edge: a library row when it resolves, else a stub."""

    identity: str
    title: str | None
    in_library: bool
    key: str | None
    id: int | None

    def to_row(self) -> dict:
        """The wire shape; a stub's title is whatever the reference list carried."""
        return {
            "identity": self.identity,
            "title": (self.title or "")[:90] or None,
            "in_library": self.in_library,
            "key": self.key,
            "id": self.id,
        }


@dataclass
class Related:
    """What a paper cites and what in the library cites it, capped, with true counts."""

    reference: SearchHit
    cites: list[Neighbour]
    cited_by: list[Neighbour]
    n_cites: int
    n_cited_by: int

    def to_row(self) -> dict:
        """The wire shape `cite.related` returns."""
        return {
            "reference": self.reference.to_row(),
            "cites": [n.to_row() for n in self.cites],
            "cited_by": [n.to_row() for n in self.cited_by],
            "n_cites": self.n_cites,
            "n_cited_by": self.n_cited_by,
        }


def _identity_forms(row) -> list[str]:
    """Every identity string that could name this row: its own, its DOI's, its arXiv id's."""
    forms = [row["identity"]]
    if row["doi"]:
        forms.append(f"doi:{row['doi']}")
    if row["arxiv_id"]:
        forms.append(f"arxiv:{row['arxiv_id']}")
    return list(dict.fromkeys(forms))


def _row_for_identity(conn: sqlite3.Connection, ident: str):
    """The row an identity names, by identity, DOI or arXiv column."""
    kind, _, value = ident.partition(":")
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', ident),
        ('SELECT * FROM "references" WHERE doi = ?', value if kind == "doi" else None),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', value if kind == "arxiv" else None),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


def _neighbour(conn: sqlite3.Connection, ident: str, title: str | None) -> Neighbour:
    row = _row_for_identity(conn, ident)
    if row is None:
        return Neighbour(ident, title, False, None, None)
    return Neighbour(ident, row["title"], True, row["bib_key"] or row["identity"], row["id"])


def _neighbour_order(n: Neighbour) -> tuple[bool, str]:
    return (not n.in_library, (n.title or "").lower())


def related(conn: sqlite3.Connection, key: str) -> Related | None:
    """What a paper cites and what in the library cites it, deterministic.

    Edges come from reference_cites (Semantic Scholar via sync, or a .bib
    `cites` field). A cited identity resolves to a library row by identity,
    DOI or arXiv form, so an edge recorded before a paper gained its DOI still
    lands. In-library first, then by title; capped with the true counts
    beside the lists. Never fetches: a paper not in the library is a stub.
    """
    row = lookup_row(conn, key)
    if row is None:
        return None
    cites = [
        _neighbour(conn, r["cited_identity"], r["cited_title"])
        for r in conn.execute(
            "SELECT cited_identity, cited_title FROM reference_cites WHERE citing_id = ?",
            (row["id"],),
        )
    ]
    forms = _identity_forms(row)
    marks = ",".join("?" * len(forms))
    citing = conn.execute(
        f'SELECT DISTINCT r.* FROM "references" r JOIN reference_cites c ON c.citing_id = r.id'
        f" WHERE c.cited_identity IN ({marks})",
        forms,
    ).fetchall()
    cited_by = [
        Neighbour(r["identity"], r["title"], True, r["bib_key"] or r["identity"], r["id"])
        for r in citing
    ]
    cites.sort(key=_neighbour_order)
    cited_by.sort(key=_neighbour_order)
    return Related(
        _hit(conn, row),
        cites[:MAX_NEIGHBOURS],
        cited_by[:MAX_NEIGHBOURS],
        len(cites),
        len(cited_by),
    )


def status(conn: sqlite3.Connection) -> dict:
    """Counts a caller can act on: what is stored, embedded, tagged, linked."""
    return {
        "references": _count(conn, 'SELECT count(*) FROM "references"'),
        "with_vectors": _count(conn, "SELECT count(*) FROM reference_vectors"),
        "with_tags": _count(conn, "SELECT count(DISTINCT reference_id) FROM reference_tags"),
        "with_cites": _count(conn, "SELECT count(DISTINCT citing_id) FROM reference_cites"),
        "sources": {
            r["source"]: r["n"]
            for r in conn.execute(
                "SELECT source, count(*) n FROM reference_sources GROUP BY source ORDER BY source"
            )
        },
    }
