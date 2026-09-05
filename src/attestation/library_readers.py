"""Sources of ReferenceRecords: three from disk and the feed, three enrichers
behind flags.

Only the offline readers can INTRODUCE a reference. The network readers
(arXiv, CrossRef, Semantic Scholar) fill fields and citation edges on rows
that already exist, which is what keeps the library the reader's and not the
web's, and keeps `cite.sources`' `offline` answer honest (spec §3.1).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from attestation import citations
from attestation.library import ReferenceRecord


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
