"""Bibliographic records, read from disk by default.

`claims.py` can verify that a number in prose matches a run in the ledger. It
could not verify that a *citation* in prose points at a real paper, and had no
way to express "supported by someone else's published result" as distinct from
"supported by my run". A reference had no representation here at all.

**The offline guarantee and its exception.** `CLAUDE.md` states "Local models
via Ollama; nothing leaves the machine." The `web` reader breaks that, so:

  1. It is absent unless `ATTEST_CITATION_WEB` is set, checked when the
     resolver is BUILT rather than when it is called -- a disabled reader
     cannot be coaxed into one request by an unusual code path.
  2. Every record carries `source` and `fetched_at`, so any answer can be
     asked where it came from.
  3. `cite.sources` reports which readers can reach the network, from the same
     surface that would have done the reaching.

A guarantee with a documented exception is honest. One that quietly stopped
holding is not.

**No Zotero library existed on the machine where this was written.** The reader
is built from Zotero's documented schema and tested against a fixture built to
the same document, so it is plausible rather than verified. If you have a real
library, point this at it -- the shape-tolerance tests say what should happen
when the layout differs, but only a real one proves what does.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ZOTERO = Path.home() / "Zotero" / "zotero.sqlite"


@dataclass(frozen=True)
class Reference:
    """One bibliographic record, and where it came from.

    `source` and `fetched_at` are the provenance pair. A record from disk has
    `fetched_at=None`; one from the network carries a date. A cache keeps the
    original date rather than refreshing it -- the cache must not launder a
    network record into something that looks local.
    """

    key: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    source: str = ""
    fetched_at: str | None = None

    def matches(self, needle: str) -> bool:
        """Whether a free-text query plausibly names this record."""
        hay = " ".join([self.key, self.title, *self.authors]).lower()
        return needle.lower() in hay

    def to_row(self) -> dict:
        """This reference's `cite.*` wire projection.

        Truncates `authors` to 6 and adds `n_authors` (the true count) so a
        long author list does not blow out a response -- the same 3-of-n
        budget pattern `RankedItem.to_row` applies to tags.

        `arxiv_id` is deliberately OMITTED: it is redundant with `doi`/`url`
        for most records, and dropping it here is a stated decision rather
        than a silent gap a future editor might "fix" back in inconsistently.
        """
        return {
            "key": self.key,
            "title": self.title,
            "authors": self.authors[:6],
            "n_authors": len(self.authors),
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            # The provenance pair, on every record. This is what makes the
            # offline guarantee's exception inspectable rather than merely
            # documented.
            "source": self.source,
            "fetched_at": self.fetched_at,
        }


def _year(value: str | None) -> int | None:
    """A four-digit year out of whatever a date field holds.

    Zotero stores '2017-06-12', '2017', and '2017-06-12 2017' depending on how
    the item was imported.
    """
    if not value:
        return None
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", str(value))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.DOTALL)
_FIELD = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", re.MULTILINE | re.DOTALL)


def _parse_bib_entries(text: str) -> Iterator[tuple[str, dict]]:
    """(key, {lowercased field: single-spaced value}) for each entry with a title.

    The ONE `.bib` parser: `BibtexReader` (the cite.* lookup path) and
    `library_readers.BibtexRecords` (the library sync) both read through it,
    so a grammar fix lands in both.
    """
    for _kind, key, body in _ENTRY.findall(text):
        fields = {k.lower(): " ".join(v.split()) for k, v in _FIELD.findall(body)}
        if fields.get("title"):
            yield key, fields


class BibtexReader:
    """`.bib` files on disk.

    A hand-rolled reader rather than a dependency: the grammar used here is
    `@type{key, field = {value},}`, which is what every tool emits, and this
    codebase has a standing preference against pulling a parser in for a
    machine-generated format (see ledger_adapters/generic._config_shape).

    A truncated entry yields nothing rather than raising: a `.bib` killed
    mid-write is the commonest way one ends, and losing the whole file for one
    bad entry is silent data loss.
    """

    name = "bibtex"
    network = False

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def all(self) -> Iterator[Reference]:
        """Every entry each `.bib` file yields, skipping ones with no title.

        A missing file is silently skipped rather than an error, matching
        `ZoteroReader`: a `.bib` named in config but not (yet) present is an
        absent source, not a broken one.
        """
        for path in self.paths:
            if not path.is_file():
                continue
            for key, fields in _parse_bib_entries(path.read_text(errors="replace")):
                authors = [a.strip() for a in fields.get("author", "").split(" and ") if a.strip()]
                yield Reference(
                    key=key,
                    title=fields["title"],
                    authors=authors,
                    year=_year(fields.get("year") or fields.get("date")),
                    doi=fields.get("doi"),
                    url=fields.get("url"),
                    source=self.name,
                )

    def lookup(self, key: str) -> Reference | None:
        """The entry whose citation key or DOI matches `key`, case-insensitive.

        Scans `all()` rather than an index: `.bib` files are small and this
        reader has no persistent state to keep one in sync with a file that
        may have changed on disk between calls.
        """
        needle = key.lower()
        for ref in self.all():
            if ref.key.lower() == needle or (ref.doi or "").lower() == needle:
                return ref
        return None


# Zotero's schema is normalised three deep: items -> itemData -> itemDataValues,
# with `fields` naming the column. Documented at
# https://www.zotero.org/support/dev/client_coding/direct_sqlite_database_access
_ZOTERO_SQL = """
SELECT i.key AS key,
       f.fieldName AS field,
       v.value AS value
  FROM items i
  JOIN itemData d ON d.itemID = i.itemID
  JOIN fields f ON f.fieldID = d.fieldID
  JOIN itemDataValues v ON v.valueID = d.valueID
 WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
"""

_ZOTERO_CREATORS = """
SELECT i.key AS key, c.lastName AS last, c.firstName AS first
  FROM items i
  JOIN itemCreators ic ON ic.itemID = i.itemID
  JOIN creators c ON c.creatorID = ic.creatorID
 WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
 ORDER BY ic.orderIndex
"""


class ZoteroReader:
    """A local Zotero library, opened read-only.

    Zotero holds an exclusive lock while running, so this opens with
    `mode=ro&immutable=1`: it reads a live library without being able to write
    to it. If that fails -- no library, a corrupt file, a schema this does not
    recognise -- the reader returns no records rather than raising. A missing
    Zotero is an absent source, not an error, and a resolver whose other
    readers work must keep working.
    """

    name = "zotero"
    network = False

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DEFAULT_ZOTERO

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def all(self) -> Iterator[Reference]:
        """Every item in the library with a title, read-only and tolerant.

        No library, a corrupt file, or a schema this does not recognise all
        yield nothing rather than raising -- see the class docstring. This
        reader ships tested only against a fixture built to Zotero's
        documented schema; there was no real library on the machine where it
        was written, so the fixture is plausible, not verified. If you have a
        real one, point this at it.
        """
        for key, data, authors in self.raw_items():
            yield Reference(
                key=key,
                title=data["title"],
                authors=authors,
                year=_year(data.get("date")),
                doi=data.get("DOI"),
                url=data.get("url"),
                source=self.name,
            )

    def raw_items(self) -> Iterator[tuple[str, dict, list[str]]]:
        """(zotero key, {field: value}, [authors]) for every titled, undeleted item.

        The library sync reads this rather than `all()` because it wants
        fields `Reference` does not carry (abstractNote, publicationTitle,
        extra). Same tolerance as `all()`: nothing on any sqlite error.
        """
        if not self.path.is_file():
            return
        try:
            conn = self._connect()
            rows = conn.execute(_ZOTERO_SQL).fetchall()
            creators = conn.execute(_ZOTERO_CREATORS).fetchall()
            conn.close()
        except sqlite3.Error:
            # Narrow on purpose: a locked, corrupt or unrecognised library is
            # an absent source. Anything else here is a real bug and should
            # surface rather than be swallowed as "no citations".
            return

        by_key: dict[str, dict] = {}
        for row in rows:
            by_key.setdefault(row["key"], {})[row["field"]] = row["value"]
        authors: dict[str, list[str]] = {}
        for row in creators:
            name = ", ".join(p for p in (row["last"], row["first"]) if p)
            authors.setdefault(row["key"], []).append(name)

        for key, data in by_key.items():
            if data.get("title"):
                yield key, data, authors.get(key, [])

    def lookup(self, key: str) -> Reference | None:
        """The item whose Zotero key or DOI matches `key`, case-insensitive.

        Scans `all()` -- the library is opened fresh each call (see
        `_connect`), so there is no cached index to keep in sync with a
        library that changed since the last lookup.
        """
        needle = key.lower()
        for ref in self.all():
            if ref.key.lower() == needle or (ref.doi or "").lower() == needle:
                return ref
        return None


def _chmod(path: Path, mode: int) -> None:
    """Best-effort chmod. A filesystem that cannot express POSIX modes
    (Windows, most network mounts) loses the hardening, not the data."""
    try:
        path.chmod(mode)
    except OSError:
        pass


def _write_cached(cache_dir: Path, cached: Path, ref) -> None:
    """Write one cached record, 0700 dir / 0600 file.

    What is cached here is not the paper -- it is the record that this machine
    looked this key up, and the date it did. A directory listing of it is a
    reading trail, so the default 0755/0644 publishes to every local account
    something the researcher never chose to share.

    Applied at CREATION only: `mode=` on mkdir is ignored for a directory that
    already exists, so a cache someone deliberately opened up stays open.
    """
    import json

    existed = cache_dir.is_dir()
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not existed:
        _chmod(cache_dir, 0o700)
    cached.write_text(json.dumps(ref.__dict__))
    _chmod(cached, 0o600)


class WebReader:
    """Metadata by DOI or arXiv id, from CrossRef and arXiv.

    **This is the only thing in the project that leaves the machine**, and it
    exists only when `ATTEST_CITATION_WEB` was set at construction. Records are
    cached content-addressed and never expire: a published paper's metadata
    does not change, and an expiring cache turns one network call into a
    recurring one, which is the opposite of the guarantee.
    """

    name = "web"
    network = True

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (Path.home() / ".hermes" / "citation-cache")

    def all(self) -> Iterator[Reference]:
        """Not supported: a network source has no fixed set to enumerate.

        `Resolver.search` relies on this raising -- it skips every reader
        with `network=True` before calling `all()`, so a search that fanned
        out to CrossRef here would be the offline guarantee failing quietly.
        This exists so that bypassing that guard is loud rather than a silent
        empty result.
        """
        raise NotImplementedError("a network source cannot be enumerated")

    def lookup(self, key: str) -> Reference | None:
        """DOI or arXiv id metadata from CrossRef, cached content-addressed.

        A cache hit returns the ORIGINAL `fetched_at`, never today's date --
        the cache must not launder a network record into one that looks
        local. Any failure (unreachable network, a 404, a changed payload
        shape) returns `None` rather than raising: an unreachable network is
        an absent source, exactly like a missing Zotero library, and should
        not break a lookup whose other readers can still answer.
        """
        import json
        from datetime import UTC, datetime

        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
        cached = self.cache_dir / f"{safe}.json"
        if cached.is_file():
            # Keep the ORIGINAL fetched_at: the cache must not launder a
            # network record into one that looks local.
            return Reference(**json.loads(cached.read_text()))

        import httpx

        try:
            resp = httpx.get(
                f"https://api.crossref.org/works/{key}",
                timeout=10.0,
                headers={"User-Agent": "attestation/citations"},
            )
            resp.raise_for_status()
            msg = resp.json()["message"]
        except Exception:  # noqa: BLE001 -- an unreachable network is an absent
            # source, exactly like an absent Zotero. Every failure mode here
            # (DNS, timeout, 404, a changed payload shape) means the same thing
            # to the caller, and none of them should break a lookup whose other
            # readers can answer.
            return None

        authors = [
            ", ".join(p for p in (a.get("family"), a.get("given")) if p)
            for a in msg.get("author", [])
        ]
        ref = Reference(
            key=key,
            title=" ".join(msg.get("title") or ["(untitled)"]),
            authors=authors,
            year=_year(str((msg.get("issued", {}).get("date-parts") or [[None]])[0][0])),
            doi=msg.get("DOI"),
            url=msg.get("URL"),
            source=self.name,
            fetched_at=datetime.now(UTC).date().isoformat(),
        )
        _write_cached(self.cache_dir, cached, ref)
        return ref


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------


class Resolver:
    """The configured readers, asked in order, recording which one answered."""

    def __init__(self, readers):
        self.readers = list(readers)

    @classmethod
    def from_env(cls, *, zotero_path=None, bib_paths=None, cache_dir=None) -> Resolver:
        """Build from the environment.

        `ATTEST_CITATION_WEB` is read HERE and nowhere else. Reading it at call
        time would mean a resolver built while disabled could still make a
        request if the variable changed under it.
        """
        readers: list = (
            [ZoteroReader(zotero_path)] if zotero_path or DEFAULT_ZOTERO.is_file() else []
        )
        paths = list(bib_paths) if bib_paths else sorted(Path.cwd().glob("*.bib"))
        if paths:
            readers.append(BibtexReader(paths))
        if os.environ.get("ATTEST_CITATION_WEB", "").strip() not in ("", "0", "false"):
            readers.append(WebReader(cache_dir))
        return cls(readers)

    def lookup(self, key: str) -> Reference | None:
        """The first configured reader's answer for `key`, tried in order.

        Order is the constructor's reader list, which `from_env` fixes as
        zotero, then bibtex, then web -- so a network lookup is only ever
        tried after every local, offline reader has already said no.
        """
        for reader in self.readers:
            found = reader.lookup(key)
            if found is not None:
                return found
        return None

    def search(self, needle: str) -> list[Reference]:
        """Free-text search across the readers that can be enumerated.

        Network readers are skipped rather than queried: `all()` raises on
        them, and a search that silently fanned out to CrossRef would be the
        offline guarantee failing quietly.
        """
        out: list[Reference] = []
        seen: set[str] = set()
        for reader in self.readers:
            if reader.network:
                continue
            for ref in reader.all():
                if ref.matches(needle) and ref.key not in seen:
                    seen.add(ref.key)
                    out.append(ref)
        return out

    def sources(self) -> list[dict]:
        """Which readers are configured, and which of them can reach the
        network -- so `cite.sources` reports the offline exception from the
        same surface that would have done the reaching, rather than a
        separate claim about it."""
        return [{"name": r.name, "network": r.network} for r in self.readers]
