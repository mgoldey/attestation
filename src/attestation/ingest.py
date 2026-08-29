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
    Use attestation.feeds.add_feed / remove_feed to change the feed set.
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


def _exists(conn, feed_id: int, guid: str | None, chash: str) -> bool:
    if guid is not None:
        row = conn.execute(
            "SELECT 1 FROM items WHERE feed_id = ? AND guid = ?", (feed_id, guid)
        ).fetchone()
        if row:
            return True
    row = conn.execute("SELECT 1 FROM items WHERE content_hash = ?", (chash,)).fetchone()
    return row is not None


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
    skipped = 0
    for entry in entries:
        title = (entry.get("title") or "").strip()
        summary = strip_boilerplate(entry.get("summary", ""))
        guid = entry.get("id")
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
        duplicate_in_batch = chash in seen_hashes or (guid is not None and guid in seen_guids)
        if duplicate_in_batch or _exists(conn, feed_id, guid, chash):
            skipped += 1
            continue
        if guid is not None:
            seen_guids.add(guid)
        seen_hashes.add(chash)
        new_entries.append((entry, title, summary, guid, chash))
    return new_entries, skipped


def run_ingest(conn, embedder, feeds_path: str | Path, parse=feedparser.parse) -> dict:
    """Fetch every registered feed, dedup, embed, and store -- deterministic
    throughout, per the module docstring; no LLM runs here.

    Embedding happens in a pass separate from the dedup/store transaction
    (see the comment below `_new_entries`): the embed call is the slow HTTP
    round trip to the model server, and holding a DB lock across it would
    block every other reader and writer for that long. One feed's failure is
    counted and does not stop the others.
    """
    sync_feeds(conn, feeds_path)
    stats = {"added": 0, "skipped": 0, "failed_feeds": 0}
    for feed in conn.execute("SELECT * FROM feeds").fetchall():
        try:
            parsed = parse(feed["url"])

            new_entries, skipped = _new_entries(conn, feed["id"], parsed.entries)
            stats["skipped"] += skipped

            # Pass 2: embed everything outside of any transaction. These are
            # the slow HTTP calls to Ollama -- no db lock is held while they run.
            embedded = [
                (entry, title, summary, guid, chash, embedder.embed_document(title, summary))
                for entry, title, summary, guid, chash in new_entries
            ]

            # Pass 3: short write transaction -- just the inserts + last_fetched
            # update. `added` is counted locally and folded into stats only
            # after the commit: the rollback below undoes the rows, so counting
            # as we go reported items that no longer exist.
            added_here = 0
            for entry, title, summary, guid, chash, vec in embedded:
                cur = conn.execute(
                    "INSERT INTO items(feed_id, guid, title, url, summary, published, content_hash)"
                    " VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?)",
                    (
                        feed["id"],
                        guid,
                        title,
                        entry.get("link"),
                        summary,
                        _published_iso(entry),
                        chash,
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
            stats["added"] += added_here
        except Exception as exc:  # noqa: BLE001 -- one bad feed must not end
            # the run, and the handler below sorts the two cases that matter:
            # an unreachable embedding backend (fatal for every feed, so stop
            # and say so once) versus this feed being broken (report and go on).
            conn.rollback()
            stats["failed_feeds"] += 1
            # An unreachable embedder is not a broken feed, and reporting it as
            # one sends a new user to debug their network or feeds.toml while
            # the actual cause is that Ollama is not running. Measured: with the
            # backend down this printed one full httpx traceback PER FEED --
            # 22 of them, ~880 lines -- every one headed "feed failed: <url>".
            if backend_unreachable(exc):
                if not stats.get("embedder_down"):
                    stats["embedder_down"] = True
                    log.warning(
                        "embedding model unreachable (LLM_BASE_URL=%s) -- is ollama"
                        " running? (`attest install --check` diagnoses this). Skipping"
                        " the remaining feeds; nothing can be embedded until it is back.",
                        os.environ.get("LLM_BASE_URL", "default"),
                    )
                break
            # A genuine per-feed failure still names the feed. No stack trace:
            # the exception text is the diagnosis, and a traceback for an
            # expected condition trains people to ignore the output.
            log.warning("feed failed: %s -- %s: %s", feed["url"], type(exc).__name__, exc)
    return stats
