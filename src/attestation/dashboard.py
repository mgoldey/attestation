"""The library dashboard: ONE JSON file a host page reads to show what was ingested.

AgentMarkit's Files tab draws "a dashboard of references ingested" and "a
searchable reference library" from this file and nothing else -- no database
access, no tool call -- so the file is a contract (`schema: 1`): fields may be
ADDED, never renamed, removed or retyped. See docs/guides/library-dashboard.md
for the field-by-field meaning; the semantics that are not obvious are decided
here and stated where they are decided.

Pure over a connection (`build`), one atomic write (`write`), no network and no
model. Every string that came from a feed, a .bib or a web enricher is
untrusted text, so every string field is stripped of control characters and
length-capped, and a `url` is http(s) or null -- the consumer renders text, but
a field meant to become an href must not be able to carry `javascript:`.

Deterministic: the same database gives the same bytes except `generated_at`
(sorted keys, stable orderings with an id tie-break, tags and sources sorted).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from attestation import paths

SCHEMA = 1
WEEKS = 26
DEFAULT_LIMIT = 5000
MAX_LIMIT = 5000  # --limit is clamped to this
MAX_BYTES = 6 * 1024 * 1024  # the consumer refuses a file over 8 MB; leave headroom
PERSONA_CAP = 40
MAX_SOURCES = 20
MAX_AUTHORS = 8
MAX_TAGS = 12
MAX_REF_SOURCES = 5
PATH_ENV = "ATTEST_LIBRARY_DASHBOARD"
DEFAULT_PERSONA = "owner"

CAP = {
    "title": 300,
    "venue": 200,
    "author": 120,
    "tag": 60,
    "url": 500,
    "source": 120,
    "key": 200,
    "id": 200,
    "arxiv": 40,
}
_CHUNK = 500  # below SQLITE_LIMIT_VARIABLE_NUMBER, like features._SQL_VAR_CHUNK


def default_path() -> Path:
    """`$ATTEST_LIBRARY_DASHBOARD`, else the Research Desk's library folder.

    The folder is `<parent of HERMES_HOME>/.hermes/workspace/research-desk/library`
    -- the same convention the host's installer uses for the desk page and the
    bibliography folder -- so on the usual layout (`HERMES_HOME=<dir>/.hermes`)
    it is `$HERMES_HOME/workspace/research-desk/library/library.json`.
    """
    configured = (os.environ.get(PATH_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    raw = (os.environ.get("HERMES_HOME") or "").strip()
    home = Path(raw).expanduser() if raw else Path.home() / paths.DEFAULT_HOME_DIRNAME
    if home.parent.name == "profiles":
        # A Hermes profile (<root>/profiles/<name>): the consumer reads the ROOT's
        # workspace, not one beside the profiles directory.
        root = home.parent.parent
    elif raw:
        root = home.parent / paths.DEFAULT_HOME_DIRNAME
    else:
        root = home
    return root / "workspace/research-desk/library/library.json"


# ---------------------------------------------------------------------------
# untrusted text
# ---------------------------------------------------------------------------


def clean(value, limit: int) -> str:
    """`value` as text with every control/format character replaced by a space,
    whitespace collapsed, and at most `limit` characters AND `2 * limit` UTF-8 bytes
    (cut text ends in an ellipsis). A character cap alone is no bound: a 4-byte
    emoji costs four of the consumer's bytes for one of ours.

    HTML is NOT escaped: a title `<script>` stays the literal text it is, and
    the consumer renders text only. Escaping here would show `&lt;` to every
    reader and still protect nothing, because the one field that can become a
    link (`url`) is validated separately.
    """
    if value is None:
        return ""
    text = str(value)
    text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in text)
    text = " ".join(text.split())
    if len(text) <= limit and len(text.encode()) <= 2 * limit:
        return text
    text = text[: limit - 1]
    while len(text.encode()) > 2 * limit - 3:  # room for the 3-byte ellipsis
        text = text[: max(len(text) * 9 // 10, 0)]
    return text.rstrip() + "\u2026"


def clean_or_none(value, limit: int) -> str | None:
    """`clean(value, limit)`, or None when nothing is left of it."""
    return clean(value, limit) or None


def safe_url(value) -> str | None:
    """`value` when it is an http(s) URL with a host, no credentials and no
    whitespace, at most 500 characters; else None. Never truncated: a cut URL
    points somewhere else."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or len(url) > CAP["url"] or any(unicodedata.category(c)[0] in "CZ" for c in url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host or parts.username is not None:
        return None
    return url


def _iso(value) -> str | None:
    """A stored timestamp as `YYYY-MM-DDTHH:MM:SSZ`, or None when unparseable.

    The database holds three shapes: the library's `...+00:00`, SQLite's
    `datetime('now')` (`YYYY-MM-DD HH:MM:SS`, UTC) and a feed's own
    `YYYY-MM-DDTHH:MM:SS` (also UTC: ingest formats `published_parsed`).
    A naive value is read as UTC. A bare date is midnight.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(value) -> date | None:
    iso = _iso(value)
    return date.fromisoformat(iso[:10]) if iso else None


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return conn.execute(sql, params).fetchone()[0] or 0


def _vector_count(conn: sqlite3.Connection, table: str) -> int:
    """Rows in a vec0 table, 0 when there is none: a Python without the
    sqlite-vec extension never creates them (db._ensure_vec_tables)."""
    try:
        return _scalar(conn, f"SELECT count(*) FROM {table}")
    except sqlite3.OperationalError:
        return 0


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _weekly(days: dict[date, int], today: date) -> list[tuple[date, int]]:
    """The last WEEKS ISO weeks ending with today's, oldest first, zero-filled."""
    this = _week_start(today)
    buckets = {this - timedelta(weeks=i): 0 for i in range(WEEKS)}
    for day, n in days.items():
        start = _week_start(day)
        if start in buckets:
            buckets[start] += n
    return sorted(buckets.items())


def _days(conn: sqlite3.Connection, sql: str) -> dict[date, int]:
    out: dict[date, int] = {}
    for row in conn.execute(sql):
        day = _day(row[0])
        if day is not None:
            out[day] = out.get(day, 0) + row[1]
    return out


def _ingested(conn: sqlite3.Connection, today: date) -> list[dict]:
    """Weekly counts. Items use `items.published` and references `first_seen`.

    `items` has no ingest-time column. `published` is the feed's own date when
    the entry carried one and the ingest moment (`datetime('now')`) when it did
    not, so it is the nearest thing there is: right for an hourly refresh (a new
    item is published within hours of being ingested), and wrong only for a
    backfill, where a first ingest of an old feed lands items in the weeks they
    were published rather than the week they arrived. `references.first_seen`
    IS an ingest time -- the moment the library first stored the row.
    """
    items = _weekly(
        _days(conn, "SELECT substr(published, 1, 10), count(*) FROM items GROUP BY 1"), today
    )
    refs = dict(
        _weekly(
            _days(conn, 'SELECT substr(first_seen, 1, 10), count(*) FROM "references" GROUP BY 1'),
            today,
        )
    )
    out = []
    for start, n in items:
        iso = start.isocalendar()
        out.append(
            {
                "week": f"{iso.year}-W{iso.week:02d}",
                "start": start.isoformat(),
                "items": n,
                "references": refs.get(start, 0),
            }
        )
    return out


def _kind(source: str) -> str:
    if source.startswith("bibtex"):
        return "bib"
    if source == "zotero":
        return "zotero"
    return "library"


def _by_source(conn: sqlite3.Connection) -> list[dict]:
    """Items per feed (kind `feed`) and references per contributing source
    (`bib` for a .bib file, `zotero`, else `library`: arXiv/CrossRef/S2
    enrichers, research topics, a feed item promoted to a reference)."""
    rows: list[dict] = []
    for r in conn.execute(
        "SELECT coalesce(nullif(f.title, ''), f.url) AS name, count(*) AS n FROM items i"
        " JOIN feeds f ON f.id = i.feed_id GROUP BY f.id"
    ):
        rows.append({"source": clean(r["name"], CAP["source"]), "kind": "feed", "items": r["n"]})
    for r in conn.execute(
        "SELECT source, count(DISTINCT reference_id) AS n FROM reference_sources GROUP BY source"
    ):
        rows.append(
            {
                "source": clean(r["source"], CAP["source"]),
                "kind": _kind(r["source"]),
                "items": r["n"],
            }
        )
    rows = [r for r in rows if r["source"]]
    rows.sort(key=lambda r: (-r["items"], r["kind"], r["source"]))
    return rows[:MAX_SOURCES]


def _persona_id(conn: sqlite3.Connection, persona: str) -> int | None:
    from attestation.rank import get_user

    row = get_user(conn, persona)
    return row["id"] if row is not None else None


def _in(conn: sqlite3.Connection, sql: str, ids: list[int]) -> list[sqlite3.Row]:
    """`sql` with one `{}` placeholder list, over `ids` in SQLite-safe chunks."""
    out: list[sqlite3.Row] = []
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i : i + _CHUNK]
        out.extend(conn.execute(sql.format(",".join("?" * len(chunk))), chunk).fetchall())
    return out


def _authors(raw) -> list[str]:
    try:
        names = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        return []
    out = (
        [clean(n, CAP["author"]) for n in names if isinstance(n, str)]
        if isinstance(names, list)
        else []
    )
    return [n for n in out if n][:MAX_AUTHORS]


def _grouped(conn: sqlite3.Connection, sql: str, ids: list[int], cap: int) -> dict[int, list[str]]:
    """{reference id: sorted, cleaned, de-duplicated values} for a two-column query
    `(reference_id, value) ... WHERE reference_id IN ({})`."""
    found: dict[int, set[str]] = {}
    for r in _in(conn, sql, ids):
        value = clean(r[1], cap)
        if value:
            found.setdefault(r[0], set()).add(value)
    return {k: sorted(v) for k, v in found.items()}


def _saved_ids(conn: sqlite3.Connection, persona_id: int | None) -> set[int]:
    if persona_id is None:
        return set()
    return {
        r["reference_id"]
        for r in conn.execute(
            "SELECT reference_id FROM bibliography WHERE user_id = ? AND removed_at IS NULL",
            (persona_id,),
        )
    }


def _row(r: sqlite3.Row, tags: list[str], sources: list[str], saved: bool, fulltext: bool) -> dict:
    year = r["year"] if isinstance(r["year"], int) and not isinstance(r["year"], bool) else None
    return {
        "key": clean(r["bib_key"] or r["identity"], CAP["key"]),
        "title": clean(r["title"], CAP["title"]),
        "authors": _authors(r["authors"]),
        "year": year,
        "venue": clean_or_none(r["venue"], CAP["venue"]),
        "doi": clean_or_none(r["doi"], CAP["id"]),
        "arxiv": clean_or_none(r["arxiv_id"], CAP["arxiv"]),
        "url": safe_url(r["url"]),
        "tags": tags[:MAX_TAGS],
        "sources": sources[:MAX_REF_SOURCES],
        "added": _iso(r["first_seen"]),
        "saved": saved,
        "fulltext": fulltext,
    }


def _references(conn: sqlite3.Connection, persona_id: int | None, limit: int) -> tuple[list, bool]:
    total = _scalar(conn, 'SELECT count(*) FROM "references"')
    rows = conn.execute(
        "SELECT id, identity, bib_key, title, authors, year, venue, doi, arxiv_id, url, first_seen"
        ' FROM "references" ORDER BY first_seen DESC, id DESC LIMIT ?',
        (limit,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    tags = _grouped(
        conn,
        "SELECT reference_id, tag FROM reference_tags WHERE reference_id IN ({})",
        ids,
        CAP["tag"],
    )
    sources = _grouped(
        conn,
        "SELECT reference_id, source FROM reference_sources WHERE reference_id IN ({})",
        ids,
        CAP["source"],
    )
    fulltext = {
        r["reference_id"]
        for r in _in(
            conn,
            "SELECT reference_id FROM reference_fulltext WHERE reference_id IN ({})"
            " AND source != 'none' AND coalesce(text, '') != ''",
            ids,
        )
    }
    saved = _saved_ids(conn, persona_id)
    out = [
        _row(
            r,
            tags.get(r["id"], []),
            sources.get(r["id"], []),
            r["id"] in saved,
            r["id"] in fulltext,
        )
        for r in rows
    ]
    return out, total > len(rows)


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def build(
    conn: sqlite3.Connection,
    *,
    persona: str = DEFAULT_PERSONA,
    limit: int = DEFAULT_LIMIT,
    now: datetime | None = None,
) -> dict:
    """The v1 document for `persona` -- see docs/guides/library-dashboard.md.

    Read-only: it never folds a bibliography, creates a persona or writes a
    row. An unknown persona (a fresh machine before the first conversation)
    has `saved` 0 and no saved reference, not an error.

    `failures` is `null` throughout because Attestation records no failure
    counts: ingest and tagging print theirs per run and keep nothing. The
    honest substitute is `backlog` (an ADDED field): items and references that
    are still untagged or unembedded right now, which is a real count of work
    outstanding but is NOT a failure count -- a fresh sync has a backlog and
    no failures.
    """
    limit = min(max(int(limit), 0), MAX_LIMIT)
    now = (now or datetime.now(UTC)).astimezone(UTC)
    # One snapshot: counts and rows from a single read transaction (a refresh may
    # land mid-export), with undecodable bytes replaced rather than raised -- one
    # bad byte in one row must not stop the export every hour.
    began = not conn.in_transaction
    if began:
        conn.execute("BEGIN")
    previous, conn.text_factory = conn.text_factory, _decode
    try:
        return fit(_build(conn, persona, limit, now))
    finally:
        conn.text_factory = previous
        if began:
            conn.rollback()


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def _build(conn: sqlite3.Connection, persona: str, limit: int, now: datetime) -> dict:
    persona_id = _persona_id(conn, persona)
    n_items = _scalar(conn, "SELECT count(*) FROM items")
    n_refs = _scalar(conn, 'SELECT count(*) FROM "references"')
    items_tagged = _scalar(conn, "SELECT count(DISTINCT item_id) FROM item_tags")
    refs_tagged = _scalar(conn, "SELECT count(DISTINCT reference_id) FROM reference_tags")
    items_embedded = _vector_count(conn, "item_vectors")
    refs_embedded = _vector_count(conn, "reference_vectors")
    references, truncated = _references(conn, persona_id, limit)
    saved = (
        _scalar(
            conn,
            "SELECT count(*) FROM bibliography WHERE user_id = ? AND removed_at IS NULL",
            (persona_id,),
        )
        if persona_id is not None
        else 0
    )
    return {
        "schema": SCHEMA,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "persona": clean(persona, PERSONA_CAP),
        "totals": {
            "items": n_items,
            "references": n_refs,
            # Records, not items: tagged/embedded count BOTH stores (a feed item
            # and a library reference are different rows even when they are
            # the same paper). The split is in items_*/references_* below.
            "tagged": items_tagged + refs_tagged,
            "embedded": items_embedded + refs_embedded,
            "with_fulltext": _scalar(
                conn,
                "SELECT count(*) FROM reference_fulltext"
                " WHERE source != 'none' AND coalesce(text, '') != ''",
            ),
            "saved": saved,
            "items_tagged": items_tagged,
            "items_embedded": items_embedded,
            "references_tagged": refs_tagged,
            "references_embedded": refs_embedded,
        },
        # The last successful feed fetch -- feeds.last_fetched is stamped when a
        # feed's items are committed, so it is the only recorded ingest time.
        "last_refresh": _iso(conn.execute("SELECT max(last_fetched) FROM feeds").fetchone()[0]),
        "ingested": _ingested(conn, now.date()),
        "by_source": _by_source(conn),
        "failures": {"fetch": None, "tag": None, "embed": None, "since": None},
        "backlog": {
            "items_untagged": max(n_items - items_tagged, 0),
            "references_untagged": max(n_refs - refs_tagged, 0),
            "references_unembedded": max(n_refs - refs_embedded, 0),
        },
        "references": references,
        "references_truncated": truncated,
        "references_limit": limit,
    }


def fit(doc: dict, max_bytes: int = MAX_BYTES) -> dict:
    """`doc` guaranteed to render in at most `max_bytes`, by dropping the OLDEST
    references (the list is newest first) and saying so: `references_truncated`
    true, `references_limit` the number kept. Totals are not touched -- they stay
    the database's. Character and byte caps per field make the common case fit;
    this is the guarantee for the worst case."""
    refs = doc["references"]
    if len(render(doc).encode()) <= max_bytes:
        return doc
    base = len(render({**doc, "references": []}).encode())
    sizes = [
        len(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()) + 1
        for r in refs
    ]
    keep, used = 0, base
    for size in sizes:
        if used + size > max_bytes:
            break
        used += size
        keep += 1
    while keep and len(render(_cut(doc, keep)).encode()) > max_bytes:
        keep -= 1
    return _cut(doc, keep)


def _cut(doc: dict, keep: int) -> dict:
    return {
        **doc,
        "references": doc["references"][:keep],
        "references_truncated": True,
        "references_limit": keep,
    }


def render(doc: dict) -> str:
    """The file's text: sorted keys, compact separators, UTF-8, trailing newline.
    Compact, because a researcher with 5,000 references is ~2 MB and the host
    page downloads it whole."""
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def write(path: str | Path, doc: dict) -> Path:
    """Write `doc` to `path` atomically: a temp file in the same directory,
    fsynced, then renamed over the target, so a reader sees the old file or the
    new one and never half of either. Mode 0600 (the file is a view of a
    private library). On any failure the temp file is removed and the existing
    target is untouched."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render(doc)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target
