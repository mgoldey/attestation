"""Formatted citations: a library row as CSL-JSON, rendered by citeproc-py.

Five styles ship in `csl_styles/` (CC BY-SA 3.0, see its LICENSE.md): APA 7th,
IEEE, Chicago author-date, Nature and Vancouver. citeproc-py (BSD) renders
them with the `en-US` locale bundled in its own package, so formatting reads
no network and no file outside the installed packages.

Each reference is rendered ALONE, never as part of a whole-library
bibliography. Two reasons, both measured on 5,000 references: a joint
bibliography disambiguates across entries (APA's 2022a/2022b), so one
reference's text would change when an unrelated paper was added; and that
disambiguation is quadratic -- 81 s for 5,000 APA entries against a linear
~20 ms each. A numeric style's citation number (`[1]`, `1.`) is stripped for
the same reason: alone, every reference is number one.

Rendering is deterministic (same row, same style, same text), and cached in
`reference_formatted` keyed by a digest of the CSL-JSON and the style file,
so the hourly dashboard renders only rows that changed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sqlite3
import warnings
from pathlib import Path

from attestation import library

STYLE_DIR = Path(__file__).resolve().parent / "csl_styles"
STYLE_ENV = "ATTEST_CITATION_STYLE"
DEFAULT_STYLE = "apa"

# id -> (file in csl_styles/, display name). The order is the order a style
# switcher lists them in.
STYLES: dict[str, tuple[str, str]] = {
    "apa": ("apa.csl", "APA 7th edition"),
    "ieee": ("ieee.csl", "IEEE"),
    "chicago-author-date": ("chicago-author-date.csl", "Chicago (author-date, 18th edition)"),
    "nature": ("nature.csl", "Nature"),
    "vancouver": ("nlm-citation-sequence.csl", "Vancouver (NLM)"),
}
# What people type for a style, folded onto its id.
_ALIASES = {
    "apa7": "apa",
    "chicago": "chicago-author-date",
    "nlm": "vancouver",
    "nlm-citation-sequence": "vancouver",
}

# BibTeX entry type -> CSL type.
_CSL_TYPE = {
    "article": "article-journal",
    "inproceedings": "paper-conference",
    "conference": "paper-conference",
    "proceedings": "book",
    "book": "book",
    "incollection": "chapter",
    "inbook": "chapter",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "report",
    "manual": "report",
    "unpublished": "manuscript",
    "online": "webpage",
    "misc": "document",
}
# `extra` (BibTeX) field -> CSL variable, for the plain-text ones.
_CSL_EXTRA = {
    "publisher": "publisher",
    "school": "publisher",
    "institution": "publisher",
    "organization": "publisher",
    "address": "publisher-place",
    "location": "publisher-place",
    "edition": "edition",
    "series": "collection-title",
    "isbn": "ISBN",
    "issn": "ISSN",
}
_MONTHS: dict[str, int] = {
    m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)
}
_NUMBER_LABEL = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s*")
_PREPRINT_VENUE = re.compile(r"^\s*(arxiv|corr|biorxiv|chemrxiv|medrxiv|ssrn|preprint)", re.I)


def style_id(name: str | None) -> str:
    """A style name as one of `STYLES`' ids. Raises ValueError naming the
    valid ids for anything else -- a typo must not silently pick a style."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in STYLES:
        raise ValueError(f"unknown citation style {name!r}: expected one of {', '.join(STYLES)}")
    return key


def default_style() -> str:
    """`ATTEST_CITATION_STYLE`, else APA. Read at call time; validated."""
    configured = (os.environ.get("ATTEST_CITATION_STYLE") or "").strip()
    return style_id(configured) if configured else DEFAULT_STYLE


def styles() -> list[dict]:
    """Every bundled style as `{id, name}`, in switcher order."""
    return [{"id": sid, "name": name} for sid, (_, name) in STYLES.items()]


# ---------------------------------------------------------------------------
# CSL-JSON
# ---------------------------------------------------------------------------


def plain(text: str | None) -> str:
    """LaTeX folded to plain text for display (`Sch{\\"u}tt` -> Schütt,
    `{E(3)}` -> E(3), `$\\alpha$` -> α). Text with no backslash, brace or `$`
    is returned unchanged: a feed title is already plain, and LaTeX would read
    its `%` as a comment."""
    if not text:
        return ""
    if not any(c in text for c in "\\{$"):
        return text
    from pylatexenc.latex2text import LatexNodes2Text

    return " ".join(LatexNodes2Text(math_mode="text").latex_to_text(text).split())


def _name(raw: str) -> dict:
    """One author string as a CSL name: `Family, Given`, `Given Family`, or a
    literal for a braced or one-word name (`{OpenAI Team}`, `OpenAI`)."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return {"literal": plain(raw[1:-1])}
    if "," in raw:
        family, _, given = raw.partition(",")
        return {"family": plain(family.strip()), "given": plain(given.strip())}
    parts = raw.split()
    if len(parts) < 2:
        return {"literal": plain(raw)}
    return {"family": plain(parts[-1]), "given": plain(" ".join(parts[:-1]))}


def _get(row, name: str):
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def _issued(year, month: str | None) -> dict | None:
    if not year:
        return None
    parts = [int(year)]
    m = (month or "").strip().lower()[:3]
    if m.isdigit() and 1 <= int(m) <= 12:
        parts.append(int(m))
    elif m in _MONTHS:
        parts.append(_MONTHS[m])
    return {"date-parts": [parts]}


def _csl_type(row, preprint: bool) -> str:
    stored = (_get(row, "entry_type") or "").lower()
    if preprint and stored in ("", "misc", "article", "unpublished", "online"):
        return "article"  # CSL's type for a preprint
    if stored in _CSL_TYPE:
        return _CSL_TYPE[stored]
    return "article-journal" if row["venue"] else "document"


def _is_preprint(row) -> bool:
    venue = row["venue"] or ""
    return bool(row["arxiv_id"]) and (not venue or bool(_PREPRINT_VENUE.match(venue)))


def _people(raw) -> list[dict]:
    return [_name(a) for a in raw if a and a.strip()]


def _csl_people(item: dict, row, extra: dict) -> None:
    """author, editor and issued."""
    if authors := _people(library.authors_of(row)):
        item["author"] = authors
    if editors := _people((extra.get("editor") or "").split(" and ")):
        item["editor"] = editors
    if issued := _issued(row["year"], extra.get("month")):
        item["issued"] = issued


def _csl_source(item: dict, row, extra: dict, preprint: bool) -> None:
    """container-title, volume, page, issue/number and the extra fields."""
    if row["venue"] and not preprint:
        item["container-title"] = plain(row["venue"])
    for col, var in (("volume", "volume"), ("pages", "page")):
        if value := _get(row, col):
            item[var] = value.replace("--", "-")
    if number := extra.get("number"):
        item["issue" if item["type"] == "article-journal" else "number"] = number
    for bib, var in _CSL_EXTRA.items():
        if extra.get(bib):
            item.setdefault(var, plain(extra[bib]))


def _csl_links(item: dict, row, preprint: bool) -> None:
    """The preprint's arXiv fields, DOI and URL (the abs page when no URL)."""
    arxiv = row["arxiv_id"]
    if preprint:
        item.setdefault("publisher", "arXiv")
        item["number"] = arxiv  # bare: styles add their own "arXiv" label
    if row["doi"]:
        item["DOI"] = row["doi"]
    if url := row["url"] or (f"https://arxiv.org/abs/{arxiv}" if arxiv else None):
        item["URL"] = url


def to_csl(row) -> dict:
    """A library row as one CSL-JSON item (`id` "ref"). Pure and deterministic.

    article / inproceedings / book / preprint / misc are handled explicitly;
    an arXiv paper with no journal is a preprint (`article`, publisher arXiv,
    number the arXiv id, the abs page as URL when the row has none). LaTeX in
    titles, names and venues is folded to plain text.
    """
    extra = library.extra_of(row)
    preprint = _is_preprint(row)
    item: dict = {"id": "ref", "type": _csl_type(row, preprint), "title": plain(row["title"])}
    _csl_people(item, row, extra)
    _csl_source(item, row, extra, preprint)
    _csl_links(item, row, preprint)
    return item


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


@functools.cache
def _style(sid: str):
    from citeproc import CitationStylesStyle

    path = STYLE_DIR / STYLES[sid][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CitationStylesStyle(str(path), validate=False, locale="en-US")


@functools.cache
def _style_hash(sid: str) -> str:
    return hashlib.sha256((STYLE_DIR / STYLES[sid][0]).read_bytes()).hexdigest()[:16]


def render(item: dict, style: str) -> str:
    """One CSL-JSON item as plain text in `style`, number label stripped."""
    from citeproc import Citation, CitationItem, CitationStylesBibliography, formatter
    from citeproc.source.json import CiteProcJSON

    sid = style_id(style)
    with warnings.catch_warnings():
        # citeproc-py warns about CSL variables it does not use; the item is
        # ours and complete, so those warnings carry nothing actionable.
        warnings.simplefilter("ignore")
        bib = CitationStylesBibliography(_style(sid), CiteProcJSON([item]), formatter.plain)
        bib.register(Citation([CitationItem(item["id"])]))
        text = str(bib.bibliography()[0])
    return " ".join(_NUMBER_LABEL.sub("", text, count=1).split())


def format_row(row, style: str) -> str:
    """A library row rendered in `style` (no cache)."""
    return render(to_csl(row), style)


def digest(item: dict, style: str) -> str:
    """What a cached rendering is valid for: the item and the style file."""
    sid = style_id(style)
    payload = json.dumps(item, sort_keys=True, ensure_ascii=False) + _style_hash(sid)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


_CHUNK = 500
_COMMIT_EVERY = 50  # ~1 s of APA rendering per write transaction, at most


def _cached(conn: sqlite3.Connection, ids: list[int], sid: str) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i : i + _CHUNK]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT reference_id, digest, text FROM reference_formatted"
            f" WHERE style = ? AND reference_id IN ({marks})",
            (sid, *chunk),
        ):
            out[r["reference_id"]] = (r["digest"], r["text"])
    return out


def cached(conn: sqlite3.Connection, rows, style: str) -> dict[int, str]:
    """{row id: text} for the rows whose cached rendering is CURRENT (its
    digest matches the row and the style today). Renders nothing and writes
    nothing: the dashboard's read-only build uses this after `warm`."""
    sid = style_id(style)
    rows = list(rows)
    hits = _cached(conn, [r["id"] for r in rows], sid)
    return {
        r["id"]: hits[r["id"]][1]
        for r in rows
        if r["id"] in hits and hits[r["id"]][0] == digest(to_csl(r), sid)
    }


def formatted(conn: sqlite3.Connection, rows, style: str, *, store: bool = False) -> dict[int, str]:
    """{row id: text} for `rows` in `style`, from the cache where it is current.

    `store=False` renders misses in memory only; `store=True` (the
    dashboard's `warm`, `attest bib format`, `cite.format`) writes them,
    committing every 50 rows so a first 5,000-row APA pass (~100 s) never
    holds the write lock for long.
    """
    sid = style_id(style)
    rows = list(rows)
    cached = _cached(conn, [r["id"] for r in rows], sid)
    out: dict[int, str] = {}
    pending = 0
    for r in rows:
        item = to_csl(r)
        want = digest(item, sid)
        hit = cached.get(r["id"])
        if hit and hit[0] == want:
            out[r["id"]] = hit[1]
            continue
        out[r["id"]] = render(item, sid)
        if store:
            conn.execute(
                "INSERT OR REPLACE INTO reference_formatted(reference_id, style, digest, text)"
                " VALUES (?, ?, ?, ?)",
                (r["id"], sid, want, out[r["id"]]),
            )
            pending += 1
            if pending % _COMMIT_EVERY == 0:
                conn.commit()
    if store:
        conn.commit()
    return out
