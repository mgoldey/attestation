"""The Reading desk: a private page of today's ranked papers whose verdicts
come back as clicks.

The page keeps its verdicts in one JSON blob that the hosting platform stores
in a small SQLite file. This module reads that file READ-ONLY and imports the
verdicts through `rank.record_click` before any ranking a person sees -- a
verdict changes nothing until a ranking is computed, so importing then loses
nothing. See docs/superpowers/specs/2026-09-29-reading-desk-design.md.

Configured from the environment (the checkout `.env`, loaded at the entry
points): ATTEST_DESK_STATE (the state file), ATTEST_DESK_USER (the persona the
verdicts belong to), ATTEST_DESK_PUBLISH (a command run after a build).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from attestation.rank import get_user, record_click

log = logging.getLogger(__name__)

STATE_VERSION = 1
# Seconds a read waits on a lock the page host holds mid-save. The host's own
# write is a single-row replace, so anything longer means something is wrong
# and ranking should go ahead on the clicks it already has.
READ_TIMEOUT_S = 5


@dataclass(frozen=True)
class Imported:
    """What one import did. `unchanged` is a verdict already recorded the same
    way; `skipped` is an entry naming no known item or carrying no bool."""

    recorded: int = 0
    unchanged: int = 0
    skipped: int = 0


def desk_config() -> tuple[Path | None, str | None]:
    """(state file, persona name) from the environment; each None when blank."""
    raw_state = (os.environ.get("ATTEST_DESK_STATE") or "").strip()
    raw_user = (os.environ.get("ATTEST_DESK_USER") or "").strip()
    return (Path(raw_state).expanduser() if raw_state else None, raw_user or None)


def read_state(path: str | Path) -> dict:
    """The page's state blob, or {} when it is missing, unreadable, or not v1.

    Opened with `mode=ro` so this can never write, create, or hold a write
    lock on a file another program owns.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(
            path.absolute().as_uri() + "?mode=ro", uri=True, timeout=READ_TIMEOUT_S
        )
        try:
            row = conn.execute("SELECT value FROM state WHERE id = 1").fetchone()
        finally:
            conn.close()
        value = json.loads(row[0]) if row else {}
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: cannot read page state %s: %s", path, exc)
        return {}
    if not isinstance(value, dict) or value.get("v") != STATE_VERSION:
        if value:
            log.warning("desk: page state %s is not version %d; ignored", path, STATE_VERSION)
        return {}
    return value


def import_verdicts(conn, user_id: int, state: dict) -> Imported:
    """Record each `verdicts[<item_id>] = {"useful": bool, ...}` as a `ui` click.

    Idempotent: a verdict already recorded the same way is left alone, so the
    hourly import and every ranking call can run it without rewriting rows.
    """
    verdicts = state.get("verdicts")
    if not isinstance(verdicts, dict):
        return Imported()
    recorded = unchanged = skipped = 0
    for key, entry in verdicts.items():
        useful = entry.get("useful") if isinstance(entry, dict) else None
        if not (isinstance(key, str) and key.isdecimal()) or not isinstance(useful, bool):
            skipped += 1
            continue
        item_id = int(key)
        if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
            skipped += 1
            continue
        row = conn.execute(
            "SELECT useful FROM clicks WHERE user_id = ? AND item_id = ?", (user_id, item_id)
        ).fetchone()
        if row is not None and bool(row[0]) == useful:
            unchanged += 1
            continue
        record_click(conn, user_id, item_id, useful, source="ui")
        recorded += 1
    return Imported(recorded, unchanged, skipped)


def import_pending(conn) -> Imported | None:
    """Import the configured page's verdicts; None when unconfigured or failed.

    Never raises: this runs in front of every ranking, and a broken state file
    must degrade to ranking on the clicks already recorded, not to an error.
    Never creates a persona either -- verdicts for a name that does not exist
    yet wait in the state file until it does.
    """
    state_path, name = desk_config()
    if state_path is None or name is None:
        return None
    user = get_user(conn, name)
    if user is None:
        log.warning("desk: ATTEST_DESK_USER=%r is not a persona yet; nothing imported", name)
        return None
    try:
        return import_verdicts(conn, user["id"], read_state(state_path))
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: import failed, ranking on existing clicks: %s", exc)
        return None
