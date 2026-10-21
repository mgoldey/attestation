"""BibTeX in and out through bibtexparser v2 (MIT): the ONE `.bib` parser.

`citations._parse_bib_entries` (the cite.* disk reader and the claim checker),
`library_readers.BibtexRecords` (the library sync) and `attest bib import` all
read through `parse`; `library.bibtex` writes through `write_entry`. A grammar
fix therefore lands everywhere at once, which is why the hand-rolled regex
reader this replaced was also kept to one function.

What the library adds over that regex reader, each found in a real `.bib`:
unbraced values (`year = 2022`), `@string` macros, quoted values containing
braces, and the entry type -- which formatting needs (an `@inproceedings`
cites its `booktitle`, an `@article` its `journal`).

Tolerance is unchanged: a truncated or malformed entry is one failed block,
reported, and the entries around it still parse. A key defined twice in one
file keeps its FIRST entry, as BibTeX does, and is reported.

Read-only and pure: nothing here touches a file. Callers read the text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

# bibtexparser logs every failed block at WARNING through its own logger,
# which reached stderr as a multi-line message in the middle of CLI output.
# `parse` reports failures itself, as data, so the library's copy is noise.
logging.getLogger("bibtexparser").setLevel(logging.ERROR)


@dataclass
class BibEntry:
    """One entry: its key, its lowercase type, and its fields.

    Field names are lowercased; values have their enclosing braces or quotes
    removed (inner braces, which protect case, are kept) and whitespace runs
    collapsed to one space -- the shape the regex reader returned.
    """

    key: str
    entry_type: str
    fields: dict[str, str]


@dataclass
class ParsedBib:
    """Everything one `.bib` text holds, plus what could not be read."""

    entries: list[BibEntry] = field(default_factory=list)
    # "line N: <why>" for each block that did not parse (1-based lines).
    failed: list[str] = field(default_factory=list)
    # Keys defined more than once in this text; the first definition stands.
    duplicate_keys: list[str] = field(default_factory=list)


def _collapse(value) -> str:
    return " ".join(str(value).split())


def parse(text: str) -> ParsedBib:
    """Every entry in `text`, in file order, with failures as data."""
    import bibtexparser
    from bibtexparser.model import DuplicateBlockKeyBlock

    library = bibtexparser.parse_string(text)
    out = ParsedBib()
    for entry in library.entries:
        out.entries.append(
            BibEntry(
                key=entry.key,
                entry_type=entry.entry_type.lower(),
                fields={f.key.lower(): _collapse(f.value) for f in entry.fields},
            )
        )
    for block in library.failed_blocks:
        if isinstance(block, DuplicateBlockKeyBlock):
            out.duplicate_keys.append(block.key)
            continue
        line = (block.start_line or 0) + 1
        why = _collapse(getattr(block, "error", "") or "") or "entry not closed or not BibTeX"
        out.failed.append(f"line {line}: {why[:160]}")
    return out


def write_entry(key: str, entry_type: str, fields: list[tuple[str, str | None]]) -> str:
    """One entry as BibTeX text: `@type{key,` then `  name = {value}` lines.

    Values are written as given -- escaping is the caller's decision, because
    only the caller knows whether a value is already LaTeX (from a `.bib`) or
    plain text (from a feed). Empty values are dropped. Deterministic: the
    same arguments always give the same text.
    """
    import bibtexparser
    from bibtexparser.model import Entry, Field

    entry = Entry(entry_type, key, [Field(name, value) for name, value in fields if value])
    fmt = bibtexparser.BibtexFormat()
    fmt.indent = "  "
    fmt.block_separator = "\n"
    return bibtexparser.write_string(bibtexparser.Library([entry]), bibtex_format=fmt)
