"""The citation manager: the owner's edits, notes, tags, merges and imports.

Everything here acts on the library STORE. An uploaded `.bib` is never
written: an edit changes the store row, is logged in `reference_edits`, and
reaches the owner's generated `.bib` (`<ATTEST_BIB_OUT>/<owner>.bib`) through
`bibliography.save` and `bibliography.write` -- the existing writer. A merge
folds one row into another and records the folded row's names as aliases, so a
`\\cite` key or a re-sync that still uses them lands on the kept row; the files
those names came from are untouched, and the next sync of them lands on the
kept row too, because their source rows moved with it.

The owner is `ATTEST_DESK_USER`, else `owner` -- the persona the dashboard
marks saved papers for. See docs/guides/citation-manager.md.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from attestation import bibtex_io, library

DEFAULT_OWNER = "owner"
MAX_IMPORT_BYTES = 50 * 1024 * 1024
MAX_SCAN_FILES = 2000
MAX_SCAN_BYTES = 2 * 1024 * 1024
MAX_CITED_BY = 10

# Editable fields: the column a name writes, or None for `authors`/`arxiv`,
# which need parsing or normalising. Any other BibTeX field name goes to `extra`.
_COLUMN_FOR = {
    "title": "title",
    "authors": "authors",
    "author": "authors",
    "year": "year",
    "venue": "venue",
    "journal": "venue",
    "booktitle": "venue",
    "doi": "doi",
    "arxiv": "arxiv_id",
    "arxiv_id": "arxiv_id",
    "url": "url",
    "abstract": "abstract",
    "type": "entry_type",
    "entry_type": "entry_type",
    "volume": "volume",
    "pages": "pages",
}
# Never editable: identity and bookkeeping are derived, and a key is stable
# once given (bibliography.py) -- a renamed key would break every \cite of it.
_REFUSED = {"key", "bib_key", "identity", "id", "title_key", "first_seen", "updated", "file"}
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def owner_name() -> str:
    """`ATTEST_DESK_USER`, else `owner` -- the dashboard's persona convention."""
    return (os.environ.get("ATTEST_DESK_USER") or "").strip() or DEFAULT_OWNER


def find(conn: sqlite3.Connection, key: str):
    """The row a key, DOI, arXiv id, identity or merged-away alias names --
    or a key only a persona's bibliography or a second .bib file uses (a row
    keeps the FIRST .bib key it saw as `bib_key`). None when nothing does."""
    row = library.lookup_row(conn, key)
    if row is not None:
        return row
    hit = conn.execute(
        "SELECT reference_id FROM bibliography WHERE cite_key = ? COLLATE NOCASE"
        " UNION ALL SELECT reference_id FROM reference_sources WHERE source_key = ?"
        " AND (source LIKE 'bibtex:%' OR source = 'zotero') LIMIT 1",
        (key.strip(), key.strip()),
    ).fetchone()
    return _row(conn, hit[0]) if hit else None


def require(conn: sqlite3.Connection, key: str):
    """`find`, or LookupError with a message the caller can act on."""
    row = find(conn, key)
    if row is None:
        raise LookupError(
            f"no library reference matches {key!r}: pass a citation key, DOI or"
            " arXiv id (`attest bib search` finds one)"
        )
    return row


def _row(conn: sqlite3.Connection, rid: int):
    return conn.execute('SELECT * FROM "references" WHERE id = ?', (rid,)).fetchone()


def _log(conn, rid: int, name: str, old, new) -> None:
    conn.execute(
        "INSERT INTO reference_edits(reference_id, field, old, new, at) VALUES (?, ?, ?, ?, ?)",
        (rid, name, None if old is None else str(old), None if new is None else str(new), _now()),
    )


# ---------------------------------------------------------------------------
# edits
# ---------------------------------------------------------------------------


def _authors(value: str) -> list[str]:
    sep = ";" if ";" in value else " and "
    return [a.strip() for a in value.split(sep) if a.strip()]


def _year(value: str) -> int | None:
    if value and not value.isdigit():
        raise ValueError(f"year must be a number, got {value!r}")
    return int(value) if value else None


# How an edit's text becomes what a column stores; anything else is the text
# itself, or NULL when empty.
_PARSERS = {
    "authors": _authors,
    "year": _year,
    "doi": library.normalise_doi,
    "arxiv_id": library.normalise_arxiv,
    "entry_type": lambda v: v.lower() or None,
}


def _parse_value(column: str, value: str):
    """An edit's text as the value its column stores."""
    value = value.strip()
    parse = _PARSERS.get(column)
    return parse(value) if parse else (value or None)


def _split_changes(changes: dict[str, str]) -> tuple[dict, dict]:
    """(column changes, extra-field changes), refusing what may not be edited."""
    columns: dict = {}
    extra: dict = {}
    for raw_name, value in changes.items():
        name = raw_name.strip().lower()
        if name in _REFUSED or not _FIELD_NAME.match(name):
            raise ValueError(f"{raw_name!r} cannot be edited (keys and ids are stable)")
        column = _COLUMN_FOR.get(name)
        if column is None:
            extra[name] = value.strip()
        else:
            columns[column] = _parse_value(column, value)
    if "title" in columns and not columns["title"]:
        raise ValueError("a reference must keep a title")
    return columns, extra


def _check_ids(conn, row, columns: dict) -> str:
    """The identity the edited row will have; refuses an id another row holds."""
    merged = {**library.row_fields(row), **columns}
    for col in ("doi", "arxiv_id"):
        if columns.get(col):
            holder = library.lookup_row(conn, columns[col])
            if holder is not None and holder["id"] != row["id"]:
                raise ValueError(
                    f"{columns[col]} already names {holder['bib_key'] or holder['identity']}:"
                    f" merge the two instead (attest bib merge)"
                )
    return library.identity(merged["doi"], merged["arxiv_id"], merged["title"], merged["year"])


def _column_sets(conn, row, old: dict, columns: dict) -> tuple[dict, list[str]]:
    sets: dict = {}
    names: list[str] = []
    for col, value in columns.items():
        if value != old[col]:
            sets[col] = json.dumps(value) if col == "authors" else value
            names.append(col)
            _log(conn, row["id"], col, old[col], value)
    return sets, names


def _extra_sets(conn, row, old: dict, extra: dict) -> tuple[dict, list[str]]:
    names = []
    for k, v in extra.items():
        if old["extra"].get(k) != (v or None):
            names.append(k)
            _log(conn, row["id"], k, old["extra"].get(k), v or None)
    new_extra = {k: v for k, v in {**old["extra"], **extra}.items() if v}
    if new_extra == old["extra"]:
        return {}, names
    return {"extra": json.dumps(new_extra, sort_keys=True)}, names


def _apply_columns(conn, row, columns: dict, extra: dict) -> list[str]:
    """Write the changes; returns the names that actually changed."""
    old = library.row_fields(row)
    sets, names = _column_sets(conn, row, old, columns)
    extra_sets, extra_names = _extra_sets(conn, row, old, extra)
    sets.update(extra_sets)
    if not sets:
        return []
    sets["identity"] = _check_ids(conn, row, columns)
    if "title" in sets:
        sets["title_key"] = library.normalise_title(sets["title"]) or None
    sets["updated"] = _now()
    assign = ", ".join(f"{c} = ?" for c in sets)
    conn.execute(f'UPDATE "references" SET {assign} WHERE id = ?', (*sets.values(), row["id"]))
    return names + extra_names


def set_note(conn: sqlite3.Connection, rid: int, note: str) -> None:
    """The owner's note on a reference; an empty note deletes it."""
    old = note_of(conn, rid)
    if note.strip():
        conn.execute(
            "INSERT OR REPLACE INTO reference_notes(reference_id, note, updated) VALUES (?, ?, ?)",
            (rid, note.strip(), _now()),
        )
    else:
        conn.execute("DELETE FROM reference_notes WHERE reference_id = ?", (rid,))
    if old != (note.strip() or None):
        _log(conn, rid, "note", old, note.strip() or None)


def note_of(conn: sqlite3.Connection, rid: int) -> str | None:
    """The owner's note on a reference, or None."""
    r = conn.execute("SELECT note FROM reference_notes WHERE reference_id = ?", (rid,)).fetchone()
    return r["note"] if r else None


def tags_of(conn: sqlite3.Connection, rid: int) -> list[str]:
    """A reference's tags, sorted."""
    return [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM reference_tags WHERE reference_id = ? ORDER BY tag", (rid,)
        )
    ]


def _fold_tag(tag: str) -> str:
    from attestation.features import TAG_PATTERN

    folded = tag.strip().lower().replace(" ", "-")
    if not re.match(TAG_PATTERN, folded):
        raise ValueError(f"tag {tag!r} is not lowercase letters, digits and hyphens (max 32)")
    return folded


def change_tags(conn, rid: int, add: Iterable[str] = (), remove: Iterable[str] = ()) -> None:
    """Add and remove the owner's tags, folded the way the tagger folds them."""
    for tag in [_fold_tag(t) for t in add]:
        if conn.execute(
            "INSERT OR IGNORE INTO reference_tags(reference_id, tag) VALUES (?, ?)", (rid, tag)
        ).rowcount:
            _log(conn, rid, "tag", None, tag)
    for tag in [_fold_tag(t) for t in remove]:
        if conn.execute(
            "DELETE FROM reference_tags WHERE reference_id = ? AND tag = ?", (rid, tag)
        ).rowcount:
            _log(conn, rid, "tag", tag, None)


def edit(
    conn: sqlite3.Connection,
    key: str,
    changes: dict[str, str] | None = None,
    *,
    note: str | None = None,
    add_tags: Iterable[str] = (),
    remove_tags: Iterable[str] = (),
    user: str | None = None,
) -> dict:
    """The owner's edit: fields, note, tags -- then the owner's .bib follows.

    One transaction: a refused field (an id another row holds, a blank title)
    changes nothing. The edited reference is saved to the owner's
    bibliography, so `<ATTEST_BIB_OUT>/<owner>.bib` carries the edited entry
    under its stable key; uploaded `.bib` files are never written.
    """
    row = require(conn, key)
    columns, extra = _split_changes(changes or {})
    try:
        changed = _apply_columns(conn, row, columns, extra)
        if note is not None:
            set_note(conn, row["id"], note)
        change_tags(conn, row["id"], add_tags, remove_tags)
    except ValueError:
        conn.rollback()
        raise
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError(
            "another reference already has that identity: merge the two instead (attest bib merge)"
        ) from exc
    conn.commit()
    return {"key": key, "changed": changed, **save_for_owner(conn, row["id"], user)}


def save_for_owner(conn: sqlite3.Connection, rid: int, user: str | None = None) -> dict:
    """Put the reference in the owner's bibliography and re-render the file.

    The owner persona is created when absent (a fresh machine edits before
    its first conversation); `file` is None when ATTEST_BIB_OUT is unset --
    the entry is still recorded and written once it is set.
    """
    from attestation import bibliography
    from attestation.rank import autocreate_user

    owner, _ = autocreate_user(conn, user or owner_name())
    kept = bibliography.keep(conn, owner["id"], rid)
    return {"owner": owner["name"], "cite_key": kept["entry"]["key"], "file": kept["file"]}


# ---------------------------------------------------------------------------
# duplicates and merging
# ---------------------------------------------------------------------------


def _rank(row) -> tuple:
    """Which of two duplicates is kept: a DOI, then an arXiv id, then the older row."""
    return (row["doi"] is None, row["arxiv_id"] is None, row["id"])


def _pair(conn, a_id: int, b_id: int, why: str) -> dict | None:
    if a_id == b_id:
        return None
    a, b = _row(conn, a_id), _row(conn, b_id)
    if a is None or b is None:
        return None
    keep, drop = sorted((a, b), key=_rank)
    return {"keep": keep["id"], "drop": drop["id"], "why": why}


def _column_groups(conn) -> list[tuple[list[int], str]]:
    groups = []
    for sql, why in (
        ('SELECT group_concat(id) ids FROM "references" WHERE doi IS NOT NULL GROUP BY doi', "doi"),
        (
            'SELECT group_concat(id) ids FROM "references" WHERE arxiv_id IS NOT NULL'
            " GROUP BY arxiv_id",
            "arxiv",
        ),
        (
            'SELECT group_concat(id) ids FROM "references" WHERE title_key IS NOT NULL'
            " AND year IS NOT NULL GROUP BY title_key, year",
            "title+year",
        ),
    ):
        for r in conn.execute(sql + " HAVING count(*) > 1"):
            groups.append((sorted(int(i) for i in r["ids"].split(",")), why))
    return groups


_ID_CONFLICTS = {"identity": None, "doi": "doi", "arxiv_id": "arxiv"}


def _refused_ids(conn) -> Iterable[tuple[int, int, str]]:
    """(row, other row, why) for every id a sync REFUSED because another row
    held it -- the store's own record of two rows that are one paper."""
    for r in conn.execute(
        "SELECT reference_id, raw FROM reference_sources WHERE raw NOT LIKE '%\"conflicts\": {}%'"
    ):
        conflicts = json.loads(r["raw"]).get("conflicts", {})
        for name, why in _ID_CONFLICTS.items():
            offered = conflicts.get(name, {}).get("offered")
            if not isinstance(offered, str):
                continue
            other = library.lookup_row(conn, offered)
            if other is not None:
                yield r["reference_id"], other["id"], why or offered.partition(":")[0]


def duplicates(conn: sqlite3.Connection) -> list[dict]:
    """Pairs of rows that are one paper: the same DOI, arXiv id, or normalised
    title and year, or an id a sync refused because another row held it.
    Each pair names the row a merge keeps (a DOI first, then an arXiv id, then
    the older row). Deterministic; changes nothing."""
    seen: dict[tuple[int, int], dict] = {}
    for ids, why in _column_groups(conn):
        for other in ids[1:]:
            if p := _pair(conn, ids[0], other, why):
                seen.setdefault((p["keep"], p["drop"]), p)
    for a, b, why in _refused_ids(conn):
        if p := _pair(conn, a, b, why):
            seen.setdefault((p["keep"], p["drop"]), p)
    out = []
    for p in sorted(seen.values(), key=lambda p: (p["keep"], p["drop"])):
        keep, drop = _row(conn, p["keep"]), _row(conn, p["drop"])
        out.append({**p, "keep_key": _key(keep), "drop_key": _key(drop), "title": keep["title"]})
    return out


def _key(row) -> str:
    return row["bib_key"] or row["identity"]


def _alias_names(conn, drop) -> list[str]:
    """Every name the dropped row answered to: its identity, id forms, keys."""
    names = [drop["identity"]]
    if drop["doi"]:
        names.append(f"doi:{drop['doi']}")
    if drop["arxiv_id"]:
        names.append(f"arxiv:{drop['arxiv_id']}")
    keys = {drop["bib_key"]} if drop["bib_key"] else set()
    keys |= {
        r["source_key"]
        for r in conn.execute(
            "SELECT source_key FROM reference_sources WHERE reference_id = ?"
            " AND (source LIKE 'bibtex:%' OR source = 'zotero')",
            (drop["id"],),
        )
    }
    keys |= {
        r["cite_key"]
        for r in conn.execute(
            "SELECT cite_key FROM bibliography WHERE reference_id = ?", (drop["id"],)
        )
    }
    names += [f"key:{k.lower()}" for k in sorted(keys)]
    return list(dict.fromkeys(names))


def _move_bibliography(conn, keep_id: int, drop_id: int) -> list[int]:
    """Each persona's entry moves to the kept row unless that row has a live
    one; the dropped entry's cite key becomes an alias. Returns the users."""
    users = []
    entries = conn.execute(
        "SELECT * FROM bibliography WHERE reference_id = ?", (drop_id,)
    ).fetchall()
    for e in entries:
        users.append(e["user_id"])
        have = conn.execute(
            "SELECT removed_at FROM bibliography WHERE user_id = ? AND reference_id = ?",
            (e["user_id"], keep_id),
        ).fetchone()
        if have is not None and (have["removed_at"] is None or e["removed_at"] is not None):
            continue  # the kept row's entry stands
        conn.execute(
            "DELETE FROM bibliography WHERE user_id = ? AND reference_id = ?",
            (e["user_id"], keep_id),
        )
        conn.execute(
            "UPDATE bibliography SET reference_id = ? WHERE user_id = ? AND reference_id = ?",
            (keep_id, e["user_id"], drop_id),
        )
    return users


def _move_children(conn, keep_id: int, drop_id: int) -> None:
    """Sources, tags, citation edges, full text, notes and history follow the
    kept row; where the kept row already has one, it wins."""
    for sql in (
        "UPDATE OR IGNORE reference_sources SET reference_id = ? WHERE reference_id = ?",
        "UPDATE OR IGNORE reference_tags SET reference_id = ? WHERE reference_id = ?",
        "UPDATE OR IGNORE reference_cites SET citing_id = ? WHERE citing_id = ?",
        "UPDATE OR IGNORE reference_fulltext SET reference_id = ? WHERE reference_id = ?",
        "UPDATE reference_edits SET reference_id = ? WHERE reference_id = ?",
        "UPDATE reference_aliases SET reference_id = ? WHERE reference_id = ?",
    ):
        conn.execute(sql, (keep_id, drop_id))
    dropped_note, kept_note = note_of(conn, drop_id), note_of(conn, keep_id)
    if dropped_note and dropped_note != kept_note:
        joined = f"{kept_note}\n\n{dropped_note}" if kept_note else dropped_note
        set_note(conn, keep_id, joined)


def merge(conn: sqlite3.Connection, keep_key: str, drop_key: str) -> dict:
    """Fold `drop_key`'s row into `keep_key`'s, in one transaction.

    The kept row fills its empty fields from the dropped one (never
    overwrites), takes its sources, tags, edges, full text, note and
    bibliography entries, and answers to every name the dropped row did
    (`reference_aliases`). The dropped row is deleted from the STORE only;
    no .bib file is touched, and a re-sync of the file it came from lands on
    the kept row. The owner's .bib is re-rendered for every persona whose
    entry moved.
    """
    keep, drop = require(conn, keep_key), require(conn, drop_key)
    if keep["id"] == drop["id"]:
        raise ValueError(f"{keep_key!r} and {drop_key!r} are already one reference")
    now = _now()
    try:
        fields = library.row_fields(drop)
        names = _alias_names(conn, drop)
        users = _move_bibliography(conn, keep["id"], drop["id"])
        _move_children(conn, keep["id"], drop["id"])
        conn.execute('DELETE FROM "references" WHERE id = ?', (drop["id"],))
        conn.executemany(
            "INSERT OR REPLACE INTO reference_aliases(alias, reference_id, merged_from, merged_at)"
            " VALUES (?, ?, ?, ?)",
            [(n, keep["id"], drop["identity"], now) for n in names],
        )
        refused = library.fill(conn, _row(conn, keep["id"]), fields)
        _log(conn, keep["id"], "merged", drop["identity"], keep["identity"])
    except (ValueError, sqlite3.IntegrityError):
        conn.rollback()
        raise
    conn.commit()
    _rewrite(conn, users)
    return {
        "kept": _key(_row(conn, keep["id"])),
        "dropped": drop["identity"],
        "aliases": names,
        "refused": refused,
    }


def _rewrite(conn, users: Iterable[int]) -> None:
    from attestation import bibliography

    for uid in sorted(set(users)):
        bibliography.write(conn, uid)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def _key_holder(conn, key: str):
    return conn.execute(
        'SELECT id, identity FROM "references" WHERE bib_key = ? COLLATE NOCASE', (key,)
    ).fetchone()


def import_bib(conn: sqlite3.Connection, path: Path) -> dict:
    """Read one `.bib` into the store; the file itself is never written.

    Validates first (a file, readable, at most 50 MB, at least one entry with
    a title) and reports what it could not read. `key_clashes` names each key
    that already belonged to a DIFFERENT paper in the store -- both rows keep
    it, so the owner should rename one in its own file -- and `duplicate_keys`
    each key defined twice in this file (BibTeX keeps the first; so does this).
    """
    from attestation.library_readers import bib_record

    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"no such file: {path}")
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError(f"{path.name} is over {MAX_IMPORT_BYTES // (1024 * 1024)} MB")
    parsed = bibtex_io.parse(path.read_text(errors="replace"))
    source = f"bibtex:{path.name}"
    counts = {"added": 0, "merged": 0, "unchanged": 0}
    clashes, untitled = [], []
    for entry in parsed.entries:
        rec = bib_record(entry, source)
        if rec is None:
            untitled.append(entry.key)
            continue
        holder = _key_holder(conn, entry.key)
        rid, how = library.upsert(conn, rec)
        counts[how] += 1
        if holder is not None and holder["id"] != rid:
            clashes.append({"key": entry.key, "store": holder["identity"], "file": rec.title})
    conn.commit()
    if not parsed.entries and parsed.failed:
        raise ValueError(f"{path.name}: no entry could be read ({parsed.failed[0]})")
    return {
        "file": path.name,
        "entries": len(parsed.entries),
        **counts,
        "untitled": untitled,
        "failed": parsed.failed,
        "duplicate_keys": parsed.duplicate_keys,
        "key_clashes": clashes,
    }


def import_zotero(conn: sqlite3.Connection, path: Path | None = None) -> dict:
    """Read a local zotero.sqlite (read-only, the existing reader) into the store."""
    from attestation.library_readers import ZoteroRecords

    reader = ZoteroRecords(path)
    if not reader.reader.path.is_file():
        raise ValueError(f"no Zotero database at {reader.reader.path}")
    report = library.sync(conn, [reader])
    b = report.sources.get("zotero", {})
    return {"file": reader.reader.path.name, "entries": b.get("seen", 0), **b}


# ---------------------------------------------------------------------------
# show, format
# ---------------------------------------------------------------------------


def show(conn: sqlite3.Connection, key: str, style: str | None = None) -> dict:
    """One reference in full, for `attest bib show`: fields, sources, tags,
    note, aliases, its BibTeX and its formatted citation."""
    from attestation import cite_format

    row = require(conn, key)
    sid = cite_format.style_id(style) if style else cite_format.default_style()
    fields = library.row_fields(row)
    return {
        "key": library.bibtex_key(row),
        "identity": row["identity"],
        **{k: v for k, v in fields.items() if v not in (None, "", [], {})},
        "sources": [
            r["source"]
            for r in conn.execute(
                "SELECT DISTINCT source FROM reference_sources WHERE reference_id = ?"
                " ORDER BY source",
                (row["id"],),
            )
        ],
        "tags": tags_of(conn, row["id"]),
        "note": note_of(conn, row["id"]),
        "aliases": [
            r["alias"]
            for r in conn.execute(
                "SELECT alias FROM reference_aliases WHERE reference_id = ? ORDER BY alias",
                (row["id"],),
            )
        ],
        "bibtex": library.bibtex(row),
        "formatted": {sid: cite_format.formatted(conn, [row], sid, store=True)[row["id"]]},
    }


def format_keys(conn: sqlite3.Connection, keys: list[str], style: str | None = None) -> dict:
    """`cite.format`'s whole act: each key's formatted citation and BibTeX.

    Keys are resolved like cite.lookup (a key, DOI, arXiv id or a merged-away
    alias); `missing` lists the ones nothing resolved, in the order given.
    """
    from attestation import cite_format

    sid = cite_format.style_id(style) if style else cite_format.default_style()
    found, missing = [], []
    for key in dict.fromkeys(k.strip() for k in keys if k.strip()):
        row = find(conn, key)
        (found if row is not None else missing).append((key, row))
    texts = cite_format.formatted(conn, [r for _, r in found], sid, store=True)
    return {
        "style": sid,
        "styles": [s["id"] for s in cite_format.styles()],
        "references": [
            {
                "key": key,
                "cite_key": library.bibtex_key(row),
                "text": texts[row["id"]],
                "bibtex": library.bibtex(row),
            }
            for key, row in found
        ],
        "missing": [k for k, _ in missing],
    }


# ---------------------------------------------------------------------------
# cited_by: where the workspace's drafts cite each reference
# ---------------------------------------------------------------------------


def _drafts(root: Path) -> list[Path]:
    """`*.tex` and `*.md` under `root`, hidden and vendored folders pruned,
    sorted, at most MAX_SCAN_FILES, each at most MAX_SCAN_BYTES."""
    found: list[Path] = []
    for here, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "node_modules")
        for name in sorted(files):
            path = Path(here) / name
            if name.endswith((".tex", ".md")) and _small(path):
                found.append(path)
        if len(found) >= MAX_SCAN_FILES:
            break
    return found[:MAX_SCAN_FILES]


def _small(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size <= MAX_SCAN_BYTES
    except OSError:
        return False


def _keys_in(path: Path) -> Iterable[tuple[str, int]]:
    """(key, line): every `\\cite`-family key in a .tex, and every claim's
    `cite=` key in either kind of file."""
    from attestation import claims, latex

    if latex.is_tex(path):
        yield from latex.cited_keys(path.read_text(errors="replace"))
    found, _ = claims.parse_file(path)
    for claim in found:
        if claim.cite:
            yield claim.cite, claim.line


def scan_citations(root: Path) -> dict[str, list[tuple[str, int]]]:
    """{key: [(path relative to root, line)]} for every key the drafts cite."""
    out: dict[str, list[tuple[str, int]]] = {}
    if not root.is_dir():
        return out
    for path in _drafts(root):
        rel = path.relative_to(root).as_posix()
        try:
            for key, line in _keys_in(path):
                out.setdefault(key, []).append((rel, line))
        except OSError:
            continue  # a file that vanished or cannot be read is not cited from
    return out


def _key_index(conn) -> dict[str, set[int]]:
    """Every key a draft may use for a row, lowercased: its .bib/Zotero keys,
    each persona's cite key, merged-away keys, its identity."""
    index: dict[str, set[int]] = {}
    for sql in (
        'SELECT id, bib_key FROM "references" WHERE bib_key IS NOT NULL',
        'SELECT id, identity FROM "references"',
        "SELECT reference_id, source_key FROM reference_sources"
        " WHERE source LIKE 'bibtex:%' OR source = 'zotero'",
        "SELECT reference_id, cite_key FROM bibliography",
        "SELECT reference_id, substr(alias, 5) FROM reference_aliases WHERE alias LIKE 'key:%'",
    ):
        for r in conn.execute(sql):
            index.setdefault(str(r[1]).lower(), set()).add(r[0])
    return index


def cited_by(conn: sqlite3.Connection, root: Path | None = None) -> dict[int, list[dict]]:
    """{reference id: [{path, line}]}, at most MAX_CITED_BY each, sorted.

    `root` defaults to `paths.workspace_dir()`. Paths are relative to it, so
    no machine path reaches the dashboard. A key no row answers to is skipped.
    """
    from attestation import paths

    root = Path(root) if root is not None else paths.workspace_dir()
    index = _key_index(conn)
    hits: dict[int, set[tuple[str, int]]] = {}
    for key, places in scan_citations(root).items():
        for rid in index.get(key.lower(), ()):
            hits.setdefault(rid, set()).update(places)
    return {
        rid: [{"path": p, "line": n} for p, n in sorted(places)[:MAX_CITED_BY]]
        for rid, places in hits.items()
    }
