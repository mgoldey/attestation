"""Deterministic feed ingest: fetch -> dedup -> clean -> embed -> store. No LLM."""

import hashlib
import logging
import os
import re
import sqlite3
import time
import tomllib
from pathlib import Path

import feedparser

from attestation.ports import backend_unreachable

log = logging.getLogger(__name__)

ARXIV_RE = re.compile(r"arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


_ARXIV_ID = re.compile(
    r"(?:oai:arXiv\.org:|arxiv\.org/(?:abs|pdf)/)([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_DOI = re.compile(r"(?:doi\.org/|doi:)?(10\.\d{4,9}/[^\s#?]+)", re.IGNORECASE)
_NATURE = re.compile(r"nature\.com/articles/([a-z0-9\-]+)", re.IGNORECASE)


def extract_ids(guid: str | None, url: str | None) -> tuple[str | None, str | None]:
    """(doi, arxiv_id) from an entry's guid and url, or Nones.

    Pure and network-free; used by migration 007's backfill and by every new
    item. Formats are the ones measured on the live database 2026-09-05:
    arXiv guids `oai:arXiv.org:1003.0563v2`, Nature URLs carrying the DOI
    suffix under the 10.1038 prefix. Anything else stays NULL.
    """
    doi = arxiv = None
    for text in (guid or "", url or ""):
        if arxiv is None and (m := _ARXIV_ID.search(text)):
            arxiv = m.group(1)
        if doi is None and (m := _DOI.search(text)):
            doi = m.group(1).lower().rstrip(".")
        if doi is None and (m := _NATURE.search(text)):
            doi = f"10.1038/{m.group(1).lower()}"
    return doi, arxiv


def strip_boilerplate(text: str) -> str:
    """Drop HTML tags and arXiv's own "Announce Type / Abstract:" preamble,
    collapsing whitespace -- so a stored summary is the actual abstract, not
    the feed entry's markup and boilerplate around it."""
    text = TAG_RE.sub(" ", text or "")
    text = ARXIV_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(title: str, summary: str) -> str:
    """A dedup key for one item: title+summary, SHA-256. Two feeds syndicating
    the same paper hash identically without any cross-feed ID to rely on."""
    return hashlib.sha256(f"{title}\n{summary}".encode()).hexdigest()


def sync_feeds(conn: sqlite3.Connection, feeds_path: str | Path) -> None:
    """Seed the feeds table from feeds.toml.

    INSERT OR IGNORE, so this is a no-op for feeds already present: the
    database -- not the TOML file -- is the source of truth once seeded.
    Use attestation.feeds.add_source / remove_source to change the feed set.
    """
    path = Path(feeds_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no feeds file at {path} -- pass --feeds to point at one, or run from a checkout"
        )
    cfg = tomllib.loads(path.read_text())
    for feed in cfg.get("feeds", []):
        conn.execute(
            "INSERT OR IGNORE INTO feeds(url, title) VALUES (?, ?)",
            (feed["url"], feed.get("title")),
        )
    conn.commit()


def _published_iso(entry) -> str | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return time.strftime("%Y-%m-%dT%H:%M:%S", parsed) if parsed else None


def _entry_ids(entry) -> tuple[str | None, str | None]:
    """(doi, arxiv_id): what the entry states (a research hit carries both keys),
    else what its guid/url imply."""
    doi, arxiv_id = extract_ids(entry.get("id"), entry.get("link"))
    return entry.get("doi") or doi, entry.get("arxiv_id") or arxiv_id


def _exists(conn, feed_id: int, guid: str | None, chash: str, doi=None, arxiv_id=None) -> bool:
    """True if this item is already stored: same feed+guid, same content hash,
    or -- across ANY feed -- the same DOI/arXiv id (a cross-list or a topic hit
    the RSS already carried; see the comment on the identifier loop below)."""
    if guid is not None:
        row = conn.execute(
            "SELECT 1 FROM items WHERE feed_id = ? AND guid = ?", (feed_id, guid)
        ).fetchone()
        if row:
            return True
    if conn.execute("SELECT 1 FROM items WHERE content_hash = ?", (chash,)).fetchone():
        return True
    # Same paper under ANY feed: a cs.LG/chem-ph cross-list (207 of 7,787
    # identified items, measured 2026-09-05) or a topic hit the RSS already
    # carried with a slightly different abstract. The library merges these
    # into one reference; this stops feed.list showing one paper twice.
    for column, value in (("doi", doi), ("arxiv_id", arxiv_id)):
        if value and conn.execute(f"SELECT 1 FROM items WHERE {column} = ?", (value,)).fetchone():
            return True
    return False


def _duplicate_in_batch(seen_guids, seen_hashes, seen_ids, guid, chash, doi, arxiv_id) -> bool:
    """True if this entry repeats one already accepted earlier in the same
    batch -- by guid, content hash, or DOI/arXiv id -- split out of
    `_new_entries` to keep that function's own branching down."""
    return (
        chash in seen_hashes
        or (guid is not None and guid in seen_guids)
        or any(v in seen_ids for v in (doi, arxiv_id) if v)
    )


def _new_entries(conn, feed_id: int, entries) -> tuple[list, int]:
    """Entries worth storing, plus how many were skipped as duplicates.

    Dedup checks are read-only (SELECT), so this never opens a write
    transaction -- other connections can still write while feeds are being
    fetched and parsed.

    Dedup runs within the batch as well as against the database. `_exists()`
    only sees committed rows, so two entries sharing a GUID both passed, the
    second hit the UNIQUE constraint during the write, and the rollback
    discarded every good item alongside it -- a whole feed lost to one
    republished post.
    """
    new_entries: list = []
    seen_guids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    skipped = 0
    for entry in entries:
        title = (entry.get("title") or "").strip()
        summary = strip_boilerplate(entry.get("summary", ""))
        guid = entry.get("id")
        doi, arxiv_id = _entry_ids(entry)
        try:
            chash = content_hash(title, summary)
        except UnicodeEncodeError:
            # A lone surrogate -- reachable from a bare `&#xD800;` character
            # reference in ordinary feed XML -- used to raise past every good
            # entry to the per-feed handler, which rolled the whole batch back.
            # That is the blast radius this function's docstring says was
            # already fixed for duplicate GUIDs; the guard just never covered
            # encoding. One bad post costs itself, not the feed.
            log.warning("skipping an entry with unencodable text: %s", (title or guid)[:60])
            skipped += 1
            continue
        duplicate_in_batch = _duplicate_in_batch(
            seen_guids, seen_hashes, seen_ids, guid, chash, doi, arxiv_id
        )
        if duplicate_in_batch or _exists(conn, feed_id, guid, chash, doi, arxiv_id):
            skipped += 1
            continue
        if guid is not None:
            seen_guids.add(guid)
        seen_hashes.add(chash)
        seen_ids.update(v for v in (doi, arxiv_id) if v)
        new_entries.append((entry, title, summary, guid, chash))
    return new_entries, skipped


def _fetch_feed(feed, parse, clients):
    """One feed's parsed payload: feedparser for an RSS URL, the research
    clients for a `research:` one (entries feedparser-shaped, plus `.papers`
    for the library). Returns None when research is disabled for this run."""
    from datetime import date

    from attestation import research

    if not research.is_topic_url(feed["url"]):
        return parse(feed["url"])
    topic = research.parse_topic(feed["url"])
    if all(clients[n].offline for n in topic.clients):
        return None
    since = date.fromisoformat(feed["last_fetched"][:10]) if feed["last_fetched"] else None
    fetched = research.fetch_topic(topic, clients, since=since)
    if fetched.errors and not fetched.papers:
        raise RuntimeError("; ".join(fetched.errors))
    for err in fetched.errors:
        log.warning("research client failed: %s -- %s", feed["url"], err)
    return fetched


def _store_references(conn, parsed) -> None:
    """After a topic's items are committed, the same hits enter the library
    under `research:<client>` with the authors and venue the payload had --
    the spec's stated exception to 'the wire never introduces a reference'."""
    from datetime import UTC, datetime

    from attestation import library, research

    papers = getattr(parsed, "papers", None)
    if not papers:
        return
    today = datetime.now(UTC).date().isoformat()
    for rec in research.as_records(papers, today):
        library.upsert(conn, rec)
    conn.commit()


def _disabled_outcome(url: str) -> dict:
    """The per-feed outcome for a research topic skipped this run because
    ATTEST_RESEARCH_WEB (or --no-research) is off -- counted, never a failure."""
    return {
        "feed": url,
        "new": 0,
        "skipped": 0,
        "error": None,
        "embedder_down": False,
        "research_disabled": True,
    }


def _ingest_outcome(outcomes: list[dict]) -> dict:
    """The decision over one run's per-feed results, held apart from the I/O
    that produced them.

    Each outcome is `{"feed": str, "new": int, "skipped": int,
    "error": str | None, "embedder_down": bool}` -- one entry per feed
    `run_ingest` actually attempted (a feed skipped outright, e.g. because the
    embedder was already known down, contributes no entry). `added` and
    `skipped` sum across feeds; `failed_feeds` counts entries with a non-None
    `error`; `embedder_down` LATCHES true if any outcome reports it, since one
    dead backend outlasts whichever feed first discovered it -- and, matching
    the shape `run_ingest` returned before this split, the key is present only
    when true, so a normal run's dict is exactly `{added, skipped,
    failed_feeds}`. `research_disabled` counts topics skipped because
    `ATTEST_RESEARCH_WEB` is off (or `--no-research`); present only when
    nonzero, and never a failure.

    Pure: no I/O, no logging -- just the list and this dict. Logging which
    feed failed and why stays in `run_ingest`, next to the exception it is
    describing.
    """
    stats = {
        "added": sum(o["new"] for o in outcomes),
        "skipped": sum(o["skipped"] for o in outcomes),
        "failed_feeds": sum(1 for o in outcomes if o["error"] is not None),
    }
    if any(o["embedder_down"] for o in outcomes):
        stats["embedder_down"] = True
    disabled = sum(1 for o in outcomes if o.get("research_disabled"))
    if disabled:
        stats["research_disabled"] = disabled
    return stats


def run_ingest(
    conn, embedder, feeds_path: str | Path, parse=feedparser.parse, *, clients=None
) -> dict:
    """Fetch every registered feed, dedup, embed, and store -- deterministic
    throughout, per the module docstring; no LLM runs here.

    Embedding happens in a pass separate from the dedup/store transaction
    (see the comment below `_new_entries`): the embed call is the slow HTTP
    round trip to the model server, and holding a DB lock across it would
    block every other reader and writer for that long. One feed's failure is
    counted and does not stop the others -- `_ingest_outcome` makes that call
    over the outcomes this loop collects.

    A `research:` feed is searched through `clients` (built once here from the
    flag) instead of parsed, and its hits also enter the reference library.
    """
    sync_feeds(conn, feeds_path)
    if clients is None:
        from attestation import research

        clients = research.clients_from_env()  # the flag is read here, once per run
    outcomes: list[dict] = []
    already_down = False
    for feed in conn.execute("SELECT * FROM feeds").fetchall():
        try:
            parsed = _fetch_feed(feed, parse, clients)
            if parsed is None:
                outcomes.append(_disabled_outcome(feed["url"]))
                continue

            new_entries, skipped = _new_entries(conn, feed["id"], parsed.entries)

            # Pass 2: embed everything outside of any transaction. These are
            # the slow HTTP calls to Ollama -- no db lock is held while they run.
            embedded = [
                (entry, title, summary, guid, chash, embedder.embed_document(title, summary))
                for entry, title, summary, guid, chash in new_entries
            ]

            # Pass 3: short write transaction -- just the inserts + last_fetched
            # update. `added_here` is counted locally and folded into the
            # outcome only after the commit: the rollback below undoes the
            # rows, so counting as we go reported items that no longer exist.
            added_here = 0
            for entry, title, summary, guid, chash, vec in embedded:
                doi, arxiv_id = _entry_ids(entry)
                cur = conn.execute(
                    "INSERT INTO items(feed_id, guid, title, url, summary, published,"
                    " content_hash, doi, arxiv_id)"
                    " VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?, ?, ?)",
                    (
                        feed["id"],
                        guid,
                        title,
                        entry.get("link"),
                        summary,
                        _published_iso(entry),
                        chash,
                        doi,
                        arxiv_id,
                    ),
                )
                conn.execute(
                    "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, vec.tobytes()),
                )
                added_here += 1
            conn.execute(
                "UPDATE feeds SET last_fetched = datetime('now') WHERE id = ?", (feed["id"],)
            )
            conn.commit()
            _store_references(conn, parsed)
            outcomes.append(
                {
                    "feed": feed["url"],
                    "new": added_here,
                    "skipped": skipped,
                    "error": None,
                    "embedder_down": False,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- one bad feed must not end
            # the run, and the handler below sorts the two cases that matter:
            # an unreachable embedding backend (fatal for every feed, so stop
            # and say so once) versus this feed being broken (report and go on).
            conn.rollback()
            # An unreachable embedder is not a broken feed, and reporting it as
            # one sends a new user to debug their network or feeds.toml while
            # the actual cause is that Ollama is not running. Measured: with the
            # backend down this printed one full httpx traceback PER FEED --
            # 22 of them, ~880 lines -- every one headed "feed failed: <url>".
            down = backend_unreachable(exc)
            outcomes.append(
                {
                    "feed": feed["url"],
                    "new": 0,
                    "skipped": 0,
                    "error": str(exc),
                    "embedder_down": down,
                }
            )
            if down:
                if not already_down:
                    # No `attestation.llm` import here -- domain modules may not
                    # name the concrete client (test_domain_reaches_models_only_
                    # through_ports). So this cannot resolve the URL the way
                    # cli.py's sibling message does via base_url(); it names the
                    # env var honestly instead of guessing at a default.
                    configured = os.environ.get("LLM_BASE_URL")
                    where = f"LLM_BASE_URL={configured}" if configured else "LLM_BASE_URL is unset"
                    log.warning(
                        "embedding model unreachable (%s) -- is ollama"
                        " running? (`attest install --check` diagnoses this). Skipping"
                        " the remaining feeds; nothing can be embedded until it is back.",
                        where,
                    )
                already_down = True
                break
            # A genuine per-feed failure still names the feed. No stack trace:
            # the exception text is the diagnosis, and a traceback for an
            # expected condition trains people to ignore the output.
            log.warning("feed failed: %s -- %s: %s", feed["url"], type(exc).__name__, exc)
    return _ingest_outcome(outcomes)
