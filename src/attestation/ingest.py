"""Deterministic feed ingest: fetch -> dedup -> clean -> embed -> store. No LLM."""

import hashlib
import logging
import re
import sqlite3
import time
import tomllib
from pathlib import Path

import feedparser

log = logging.getLogger(__name__)

ARXIV_RE = re.compile(r"arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def strip_boilerplate(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = ARXIV_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(title: str, summary: str) -> str:
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
        chash = content_hash(title, summary)
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
        except Exception:
            log.exception("feed failed: %s", feed["url"])
            conn.rollback()
            stats["failed_feeds"] += 1
    return stats
