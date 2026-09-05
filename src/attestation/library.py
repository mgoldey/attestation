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
        return self.sources.setdefault(
            name, {"seen": 0, "added": 0, "merged": 0, "unchanged": 0, "enriched": 0, "failed": 0}
        )

    def to_dict(self) -> dict:
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
    """Embed rows with no vector: (embedded, still_missing, error). Filled in Task 7."""
    return 0, _count(conn, _UNEMBEDDED), None


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
