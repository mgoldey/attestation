"""The reference library: one row per paper, fed by many sources.

Identity is a pure function (DOI, else versionless arXiv id, else normalised
title and year) so that Zotero, three `.bib` files and the feed can all fill
ONE row. Merge never overwrites: an empty field takes a value, a differing
value is recorded as a conflict on the contributing source row. See
docs/superpowers/specs/2026-09-05-library-store-design.md.
"""

from __future__ import annotations

import re
import unicodedata

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^(?:arxiv:)\s*", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_doi(doi: str | None) -> str | None:
    """Lowercase, scheme and doi.org prefix stripped; None for empty."""
    if not doi:
        return None
    out = _DOI_PREFIX.sub("", doi.strip()).lower()
    return out or None


def normalise_arxiv(arxiv_id: str | None) -> str | None:
    """`arXiv:2106.02347v3` -> `2106.02347`; old-style ids kept whole."""
    if not arxiv_id:
        return None
    out = _ARXIV_PREFIX.sub("", arxiv_id.strip())
    out = _ARXIV_VERSION.sub("", out)
    return out or None


def normalise_title(title: str) -> str:
    """NFKD, combining marks dropped, lowercase, non-alphanumerics collapsed.

    Leading articles are kept on purpose: dropping them merges "A survey"
    with "Survey", which are different papers more often than not.
    """
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", ascii_only.lower()).strip()


def identity(doi: str | None, arxiv_id: str | None, title: str | None, year: int | None) -> str:
    """The one string two records must share to be the same paper.

    DOI beats arXiv because a preprint that is later published gains a DOI
    while keeping its arXiv id; a row carries both columns so a record that
    knows only the arXiv id still finds it (see `upsert`).
    """
    if d := normalise_doi(doi):
        return f"doi:{d}"
    if a := normalise_arxiv(arxiv_id):
        return f"arxiv:{a}"
    if title and (t := normalise_title(title)):
        return f"title:{t}:{year if year is not None else '-'}"
    raise ValueError("a reference needs a DOI, an arXiv id, or a title")


def _authors_extend(existing: list, incoming: list) -> bool:
    """True when `incoming` is `existing` plus more names (a truncated list filled in)."""
    if len(incoming) <= len(existing):
        return False
    norm = [normalise_title(a) for a in existing]
    return [normalise_title(a) for a in incoming[: len(existing)]] == norm


def _same(field: str, kept, offered) -> bool:
    """Equal after the normalisation that field deserves."""
    if field == "authors":
        return [normalise_title(a) for a in kept] == [normalise_title(a) for a in offered]
    if field == "title":
        return normalise_title(kept) == normalise_title(offered)
    return kept == offered


def merge(existing: dict, incoming: dict) -> tuple[dict, dict]:
    """Fill empty fields from `incoming`; keep non-empty ones; record conflicts.

    Deliberately dumb -- longer abstracts do not win, first does -- so that
    nothing is ever overwritten silently. `cite.lookup` shows every source
    row, so a reader can see a disagreement rather than lose it.
    """
    merged = dict(existing)
    conflicts: dict = {}
    for field, offered in incoming.items():
        if offered in (None, "", [], {}):
            continue
        kept = merged.get(field)
        if kept in (None, "", [], {}):
            merged[field] = offered
            continue
        if field == "authors" and _authors_extend(kept, offered):
            merged[field] = list(offered)
            continue
        if _same(field, kept, offered):
            continue
        conflicts[field] = {"kept": kept, "offered": offered}
    return merged, conflicts
