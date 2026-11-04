"""Papers a reader likes, and a first search of their field: the cold start.

A new reader's ranking is the embedding of one sentence ("I work on X") until
they rate things, and the click classifier stays silent until both labels
appear. Feedback from a research customer (2026-10-29): the first paper was
not their field, and nothing said how to steer -- keywords? titles? So two
things here make the first list theirs:

- `like`: a paper the reader names (DOI, arXiv id or title; or every paper in
  a .bib or a PDF they point at) is resolved, stored as an item and in the
  library, recorded as a useful click from the reader (`source='agent'`), and
  logged in `engagement` with kind `seed`. `profile_text` folds those titles
  into the text the profile vector embeds, so a liked paper moves the ranking
  at once -- the classifier alone would wait for a "not useful" too.
- `first_search`: "I work on X" searches arXiv, PubMed and OpenAlex for X
  over the list's own 14-day window, so the very next "what should I read?"
  holds papers from that field even when no subscribed feed covers it.

Network use is the research clients' (`ATTEST_RESEARCH_WEB`, on unless 0),
not citations.WebReader: that one is off by default and carries no abstract,
and an abstract is what makes a liked paper embed like the reader's field.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx

from attestation import paths
from attestation.library import (
    _row_by_ids,
    arxiv_from_doi,
    normalise_arxiv,
    normalise_doi,
    normalise_title,
)
from attestation.research import FETCH_ERRORS, Paper

SEED_KIND = "seed"
# A .bib can hold hundreds of entries and each one may be a network lookup;
# a chat turn has seconds. The rest are named, not dropped silently.
MAX_FILE_SEEDS = 10
# Titles folded into the profile text. A long tail of seeds would drown the
# reader's own sentence, which is still the best single statement of intent.
MAX_PROFILE_SEEDS = 12
# The first search matches the window feed.list ranks over (rank_items'
# since_days default), so what it finds is what the next list can show.
FIRST_SEARCH_DAYS = 14
FIRST_SEARCH_LIMIT = 8
FIRST_SEARCH_CLIENTS = ("arxiv", "pubmed", "openalex")
# arXiv ANDs every word, so a long sentence finds nothing; the first words
# of the reader's own description carry the subject.
FIRST_SEARCH_WORDS = 8
# A title lookup is accepted only this close to what the reader typed:
# OpenAlex returns its best guess for anything, and a wrong paper seeded as
# liked is worse than none.
TITLE_MATCH = 0.9

_ARXIV_NEW = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ARXIV_OLD = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b")
_ARXIV_MARK = re.compile(r"arxiv[:\s]|arxiv\.org/", re.IGNORECASE)
_DOI_IN = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


@dataclass
class Liked:
    """What one `like` call did: papers seeded, and what could not be."""

    seeded: list[dict] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    skipped: int = 0


def parse_ref(text: str) -> tuple[str, str]:
    """('doi'|'arxiv'|'title', value) for what the reader typed."""
    text = text.strip().strip("\"'“”").strip()
    if m := _DOI_IN.search(text):
        doi = m.group(1).rstrip(".,;)")
        if aid := arxiv_from_doi(doi):
            return "arxiv", aid
        return "doi", normalise_doi(doi) or doi.lower()
    if m := _ARXIV_NEW.fullmatch(text) or (_ARXIV_MARK.search(text) and _ARXIV_NEW.search(text)):
        return "arxiv", normalise_arxiv(m.group(1)) or m.group(1)
    if _ARXIV_MARK.search(text) and (m := _ARXIV_OLD.search(text)):
        return "arxiv", m.group(1)
    return "title", text.rstrip(".")


def _from_library(conn: sqlite3.Connection, kind: str, value: str) -> Paper | None:
    """The paper as the library already holds it -- an uploaded .bib, a research hit."""
    row = _row_by_ids(
        conn,
        doi=value if kind == "doi" else None,
        arxiv_id=value if kind == "arxiv" else None,
        title_key=normalise_title(value) if kind == "title" else None,
    )
    if row is None or not row["title"]:
        return None
    return Paper(
        client="library",
        external_id=str(row["id"]),
        title=row["title"],
        abstract=row["abstract"] or "",
        published=f"{row['year']}-01-01" if row["year"] else None,
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        url=row["url"],
        venue=row["venue"],
    )


def _close_title(wanted: str, papers: list[Paper]) -> Paper | None:
    """The hit whose title is the one the reader typed, or None."""
    key = normalise_title(wanted)
    best, score = None, 0.0
    for p in papers:
        s = difflib.SequenceMatcher(None, key, normalise_title(p.title)).ratio()
        if s > score:
            best, score = p, s
    return best if score >= TITLE_MATCH else None


def _online(clients: dict, name: str):
    """The named client, or None when it is absent or the flag turned it off."""
    client = clients.get(name)
    return None if client is None or client.offline else client


def _first(papers: list[Paper]) -> Paper | None:
    return papers[0] if papers else None


def _arxiv_paper(clients: dict, arxiv_id: str) -> Paper | None:
    """arXiv's own record for an id, else OpenAlex's under arXiv's DataCite DOI."""
    if (ax := _online(clients, "arxiv")) is not None:
        if found := [p for p in ax.by_id(arxiv_id) if p.arxiv_id == arxiv_id]:
            return found[0]
    if (oa := _online(clients, "openalex")) is None:
        return None
    return _first(oa.works(f"doi:https://doi.org/10.48550/arxiv.{arxiv_id}", limit=1))


def _from_network(clients: dict, kind: str, value: str) -> Paper | None:
    """One lookup: OpenAlex for a DOI or a title, arXiv (then OpenAlex) for an id."""
    if kind == "arxiv":
        return _arxiv_paper(clients, value)
    if (oa := _online(clients, "openalex")) is None:
        return None
    if kind == "doi":
        return _first(oa.works(f"doi:https://doi.org/{value}", limit=1))
    return _close_title(value, oa.works(f"title.search:{value}", limit=5))


def resolve(conn: sqlite3.Connection, clients: dict, ref: str) -> Paper | None:
    """The paper `ref` names: the library first (no network), then the clients.

    None when nothing confidently matches or the network failed -- the caller
    reports it as not found rather than seeding a guess.
    """
    kind, value = parse_ref(ref)
    if not value:
        return None
    if (paper := _from_library(conn, kind, value)) is not None:
        return paper
    try:
        return _from_network(clients, kind, value)
    except FETCH_ERRORS:
        return None


def _resolve_path(text: str) -> Path | None:
    """A file the reader named: as given, else inside the agent's workspace."""
    raw = Path(text.strip().strip("\"'`")).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, paths.workspace_dir() / raw]
    if not raw.is_absolute():
        candidates += list(paths.workspace_dir().glob(f"*/{raw}"))
    return next((p for p in candidates if p.is_file()), None)


def _refs_in_bib(path: Path) -> list[str]:
    from attestation import bibtex_io

    refs = []
    for entry in bibtex_io.parse(path.read_text(errors="replace")).entries:
        f = entry.fields
        eprint = f.get("eprint") if "arxiv" in (f.get("archiveprefix") or "arxiv").lower() else None
        if ref := f.get("doi") or (f"arXiv:{eprint}" if eprint else None) or f.get("title"):
            refs.append(ref.strip("{}"))
    return refs


def _refs_in_pdf(path: Path) -> list[str]:
    """The DOI or arXiv id printed on a PDF's first page, if it has one."""
    from attestation.research import pdf_text

    text = (pdf_text(path.read_bytes()) or "")[:4000]
    if m := _DOI_IN.search(text):
        return [m.group(1).rstrip(".,;)")]
    if m := re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", text):
        return [f"arXiv:{m.group(1)}"]
    return []


def refs_in_file(text: str) -> tuple[list[str], str | None]:
    """(refs, problem) for a .bib or PDF the reader pointed at."""
    path = _resolve_path(text)
    if path is None:
        return [], f"no file at {text.strip()}"
    suffix = path.suffix.lower()
    if suffix == ".bib":
        return _refs_in_bib(path), None
    if suffix == ".pdf":
        refs = _refs_in_pdf(path)
        return refs, None if refs else f"{path.name} shows no DOI or arXiv id on its first page"
    return [], f"{path.name} is not a .bib or a PDF"


def record_seed(conn: sqlite3.Connection, user_id: int, item_id: int) -> None:
    """A useful click from the reader plus the `seed` engagement row. One commit."""
    # 'agent', as a verdict given in chat: the reader said it, the agent typed it.
    conn.execute(
        "INSERT OR REPLACE INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, 1, 'agent')",
        (user_id, item_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO engagement(user_id, item_id, kind) VALUES (?, ?, ?)",
        (user_id, item_id, SEED_KIND),
    )
    conn.commit()


def like(conn, embedder, clients: dict, user_id: int, refs: list[str]) -> Liked:
    """Seed every paper `refs` names (at most MAX_FILE_SEEDS) for this reader."""
    from attestation import ingest, research
    from attestation.rank import forget_profile_vector

    out = Liked(skipped=max(0, len(refs) - MAX_FILE_SEEDS))
    papers = []
    for ref in refs[:MAX_FILE_SEEDS]:
        if (paper := resolve(conn, clients, ref)) is None:
            out.unresolved.append(ref)
        else:
            papers.append(paper)
    papers = research.dedup_papers(papers)
    if not papers:
        return out
    research.store_papers(conn, [p for p in papers if p.client != "library"])
    for paper, item_id in zip(papers, ingest.add_papers(conn, embedder, None, papers), strict=True):
        record_seed(conn, user_id, item_id)
        out.seeded.append({"item_id": item_id, "title": paper.title, "url": paper.url})
    # The profile text changed, so its vector and every cached explanation
    # (keyed by user and item, not by profile) are stale -- as _write_interests.
    conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_id,))
    conn.commit()
    forget_profile_vector(conn, user_id)
    return out


def seed_titles(conn: sqlite3.Connection, user_id: int) -> list[str]:
    """Titles of the papers this reader seeded and still marks useful, newest first."""
    return [
        r["title"]
        for r in conn.execute(
            "SELECT i.title FROM engagement e JOIN items i ON i.id = e.item_id"
            " JOIN clicks c ON c.user_id = e.user_id AND c.item_id = e.item_id AND c.useful = 1"
            " WHERE e.user_id = ? AND e.kind = ? ORDER BY e.created_at DESC, e.item_id DESC"
            " LIMIT ?",
            (user_id, SEED_KIND, MAX_PROFILE_SEEDS),
        )
    ]


def profile_text(conn: sqlite3.Connection, user) -> str:
    """What the profile vector embeds: the reader's words, then papers they like.

    `_profile_vector` caches by this text's hash, so seeding a paper (or a
    reset dropping the clicks) re-embeds on its own.
    """
    base = user["interests"] or user["name"]
    titles = seed_titles(conn, user["id"])
    return f"{base}\nPapers I like: " + "; ".join(titles) if titles else base


def first_search(conn, embedder, clients: dict, interests: str) -> dict:
    """Recent papers on the reader's field, stored as items so the next list has them.

    Bounded: three clients, FIRST_SEARCH_LIMIT each, the list's own window.
    Never raises -- a failure is a note, and the interests are already set.
    """
    from attestation import ingest, research

    names = tuple(n for n in FIRST_SEARCH_CLIENTS if not getattr(clients.get(n), "offline", True))
    query = " ".join(interests.split()[:FIRST_SEARCH_WORDS])
    if not names or len(query) < 2:
        return {"found": 0, "note": None}
    topic = research.Topic(clients=names, query=query)
    since = date.today() - timedelta(days=FIRST_SEARCH_DAYS)
    fetched = research.fetch_topic(topic, clients, since=since, limit=FIRST_SEARCH_LIMIT)
    if not fetched.papers:
        why = "the search failed" if fetched.errors else "nothing new in the last two weeks"
        return {"found": 0, "note": f"No recent papers found on {query} ({why})."}
    try:
        ids = ingest.add_papers(conn, embedder, _topic_feed(conn, query), fetched.papers)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        # The embedder being down must not undo "I work on X": the interests
        # are set, and the hourly ingest searches this topic again.
        return {"found": 0, "note": f"Found papers on {query} but could not store them: {exc}"}
    research.store_papers(conn, fetched.papers)
    return {"found": len(ids), "note": None}


def _topic_feed(conn: sqlite3.Connection, query: str) -> int:
    """The `research:` feed the first search's items belong to, so they carry a source."""
    from attestation import research

    url = research.topic_url(FIRST_SEARCH_CLIENTS, query)
    conn.execute(
        "INSERT OR IGNORE INTO feeds(url, title) VALUES (?, ?)", (url, f"Your field: {query}")
    )
    conn.commit()
    return conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()["id"]
