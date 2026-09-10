"""Source curation: register, list, remove, preview, and suggest feeds.

The database is the source of truth for the feed set. `feeds.toml` seeds a
fresh database on first ingest (ingest.sync_feeds uses INSERT OR IGNORE, so
it is a no-op afterwards); these functions are the supported way to change
which feeds are tracked, and they work with no checkout present.

Named after the `feed.source_*` tool vocabulary these functions back
one-for-one (`add_source` <- `feed.source_add`, and so on) rather than after
"feed" -- "feed" is ambiguous in this codebase between an RSS subscription
(this module) and the personalized ranked-item product (`mcp/feed.py`,
`feed.list`, `feed.digest`); "source" names only the former.

add_source is register-only by design: it validates the URL parses and
inserts the row, leaving the fetch to the next `attest ingest`. Ingesting
inline would mean network I/O plus one embedding per item -- minutes for a
busy feed, inside a tool call an agent may time out on.
"""

import sqlite3
import tomllib
from pathlib import Path

import feedparser

CANDIDATES_PATH = Path(__file__).resolve().parent / "feed_candidates.toml"


class FeedError(ValueError):
    """A caller-fixable refusal.

    Raised for a URL that does not parse, an unknown feed id, and similar --
    instead of returned as `ok: False`, so the MCP layer maps it to ToolError
    the way it does every other refusal."""


def _looks_like_feed(parsed) -> bool:
    """A usable feed has entries, or at least a title we can show."""
    if getattr(parsed, "entries", None):
        return True
    feed_meta = getattr(parsed, "feed", None) or {}
    return bool(feed_meta.get("title")) and not getattr(parsed, "bozo", 0)


def add_source(
    conn: sqlite3.Connection,
    url: str,
    title: str | None = None,
    parse=feedparser.parse,
    *,
    added_by: int | None = None,
) -> tuple[int, str]:
    """Register a feed after checking it parses. Does NOT ingest its items.

    A `research:` URL (a standing topic, see research.parse_topic) is
    validated by parsing the URL itself -- no network at registration -- and
    stored in its canonical spelling so two spellings are one row. `added_by`
    is the registering persona's id: provenance, not scoping.

    Returns (feed_id, message). Raises FeedError if the URL does not parse
    as a feed -- a caller-fixable refusal, not a bug.
    """
    from attestation import research

    if research.is_topic_url(url):
        return _register_topic(conn, url, title, added_by)

    parsed = parse(url)
    if not _looks_like_feed(parsed):
        raise FeedError(f"{url} did not parse as an RSS/Atom feed; nothing was added")
    resolved_title = title or (getattr(parsed, "feed", None) or {}).get("title") or url
    feed_id, existed = _register(conn, url, resolved_title, added_by)
    if existed:
        return feed_id, f"already subscribed to {url}"
    return feed_id, (
        f"subscribed to {resolved_title!r}. Items appear after the next ingest "
        "(run `attest ingest`, or wait for the hourly refresh)."
    )


def _register_topic(
    conn: sqlite3.Connection, url: str, title: str | None, added_by: int | None
) -> tuple[int, str]:
    """Validate and register a `research:` URL. Raises FeedError for a bad topic."""
    from attestation import research

    try:
        topic = research.parse_topic(url)
    except research.TopicError as exc:
        raise FeedError(str(exc)) from exc
    feed_id, existed = _register(conn, topic.url, title or topic.title, added_by)
    if existed:
        return feed_id, f"already tracking {topic.url}"
    return feed_id, (
        f"tracking {title or topic.title!r}. Papers appear after the next ingest "
        "(run `attest ingest`, or wait for the hourly refresh)."
    )


def _register(
    conn: sqlite3.Connection, url: str, title: str, added_by: int | None
) -> tuple[int, bool]:
    """Insert the row; returns (feed_id, already_existed).

    Owns the race the old add_source handled inline: someone subscribing
    between the caller's check and this write -- and the window is a whole
    network round trip, since parse() sits inside it for an RSS URL -- means
    the subscription the caller asked for already exists, which is the
    outcome the serial path already calls success, not an error.
    """
    existing = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()
    if existing is not None:
        return existing["id"], True
    try:
        cur = conn.execute(
            "INSERT INTO feeds(url, title, added_by) VALUES (?, ?, ?)", (url, title, added_by)
        )
        conn.commit()
        feed_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.rollback()
        raced = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()
        if raced is None:
            raise
        return raced["id"], True
    if feed_id is None:
        # cur.lastrowid is None only when the statement was not an INSERT, or
        # the table has no rowid -- neither is possible here, so this is a
        # real failure worth surfacing rather than asserting past.
        raise FeedError(f"insert for {url} did not return a row id")
    return feed_id, False


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    """Every registered feed with its item count -- the DB is the source of
    truth (see the module docstring): feeds.toml only seeds the first ingest."""
    rows = conn.execute(
        "SELECT f.id, f.title, f.url, f.last_fetched, u.name AS added_by,"
        " COUNT(i.id) AS item_count"
        " FROM feeds f LEFT JOIN items i ON i.feed_id = f.id"
        " LEFT JOIN users u ON u.id = f.added_by"
        " GROUP BY f.id ORDER BY f.title"
    ).fetchall()
    return [
        {
            "feed_id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "kind": "research" if r["url"].startswith("research:") else "rss",
            "added_by": r["added_by"],
            "item_count": r["item_count"],
            "last_fetched": r["last_fetched"],
        }
        for r in rows
    ]


def remove_source(conn: sqlite3.Connection, feed_id: int) -> tuple[int, str]:
    """Unsubscribe. Items are ORPHANED, never deleted -- their clicks trained
    the ranker, and cascading would destroy that feedback.

    Returns (orphaned_items, message). Raises FeedError for an unknown feed_id.
    """
    row = conn.execute("SELECT title FROM feeds WHERE id = ?", (feed_id,)).fetchone()
    if row is None:
        raise FeedError(f"unknown feed_id: {feed_id}")

    orphaned = conn.execute(
        "SELECT COUNT(*) AS n FROM items WHERE feed_id = ?", (feed_id,)
    ).fetchone()["n"]
    # items.feed_id REFERENCES feeds(id) and get_db turns PRAGMA foreign_keys
    # ON, so the row must be detached before the feed can be deleted -- a
    # straight DELETE FROM feeds raises IntegrityError with an item still
    # pointing at it.
    conn.execute("UPDATE items SET feed_id = NULL WHERE feed_id = ?", (feed_id,))
    conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
    conn.commit()
    return orphaned, (
        f"unsubscribed from {row['title']!r}; {orphaned} existing item(s) and all "
        "feedback on them were kept"
    )


def preview_source(url: str, limit: int = 5, parse=feedparser.parse) -> dict:
    """Fetch and show recent entries WITHOUT subscribing.

    Returns {title, entries, message} -- no `ok` key. Raises FeedError if the
    URL does not parse as a feed.
    """
    parsed = parse(url)
    if not _looks_like_feed(parsed):
        raise FeedError(f"{url} did not parse as an RSS/Atom feed")
    feed_meta = getattr(parsed, "feed", None) or {}
    entries = [
        {"title": e.get("title"), "url": e.get("link")} for e in list(parsed.entries)[:limit]
    ]
    return {
        "message": f"{len(entries)} recent entrie(s); not subscribed",
        "title": feed_meta.get("title") or url,
        "entries": entries,
    }


def _load_candidates() -> list[dict]:
    return tomllib.loads(CANDIDATES_PATH.read_text()).get("candidates", [])


def _score_candidates(
    liked: set[str], subscribed: set[str], candidates: list[dict], limit: int
) -> list[dict]:
    """Rank candidates by tag overlap with `liked`, dropping `subscribed` URLs.

    Pure: no database, no file I/O -- just the sets and the candidate list.
    Ties break by title so the order is deterministic. Highest score first.
    """
    scored = []
    for cand in candidates:
        if cand["url"] in subscribed:
            continue
        overlap = liked & set(cand.get("tags", []))
        score = len(overlap)
        scored.append(
            (
                score,
                {
                    "url": cand["url"],
                    "title": cand["title"],
                    "score": score,
                    "matched_tags": sorted(overlap),
                },
            )
        )
    scored.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
    return [entry for _, entry in scored[:limit]]


def suggest_sources(conn: sqlite3.Connection, user_id: int, limit: int = 5) -> list[dict]:
    """Score the curated candidate list against tags this user marked useful."""
    liked = {
        r["tag"]
        for r in conn.execute(
            "SELECT DISTINCT t.tag FROM clicks c JOIN item_tags t ON t.item_id = c.item_id"
            " WHERE c.user_id = ? AND c.useful = 1",
            (user_id,),
        )
    }
    subscribed = {r["url"] for r in conn.execute("SELECT url FROM feeds")}
    return _score_candidates(liked, subscribed, _load_candidates(), limit)
