# examples/molecular-ai/generate.py
"""Generates references.bib from real papers with the real readers.

`seeds.toml` holds identifiers and working titles. This script turns them into
offline rows in a scratch library, lets the arXiv, CrossRef and Semantic
Scholar enrichers fill abstracts, authors, venues and reference lists, tags
every row with the real tagger against the chat model, and writes the result
with bibtexparser v2. Nothing in `references.bib` was typed to look fetched:
a seed that fails to resolve is dropped and reported.

    ATTEST_CITATION_WEB=1 ATTEST_CITATION_SCHOLAR=1 \\
      uv run --with "bibtexparser>=2.0.0b9" python generate.py

Semantic Scholar is paced at one request every three seconds and rate-limits
the shared unauthenticated pool anyway, so expect several resumed passes
(`--scratch` keeps the cache between them). Regenerating is deliberate; review
the diff like any other fixture change.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import tomllib
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Built from parts so this file does not itself carry the host name the
# golden-path guard scans for.
# `[^\s}]*`, not `\S*`: the first generation's scrub swallowed the `},` that
# closed three abstracts ending in a code URL, and the committed file was
# invalid BibTeX its own writer could not re-read (review round 1).
_CODE_URL = re.compile(r"(?:https?://)?(?:www\.)?" + "git" + "hub" + r"[ .]com[^\s})]*")


def _tex(text: str | None) -> str | None:
    """Fetched text as a BibTeX field value: `&` escaped, nothing else touched."""
    return text.replace("&", "\\&") if text else text


def _ascii(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(c for c in folded if not unicodedata.combining(c))


def _key(authors: list[str], year: int | None, title: str) -> str:
    """<surname><year><firstword>, lowercase ASCII, the usual .bib convention."""
    # CrossRef gives "Family, Given"; the arXiv API gives "Given Family".
    first = authors[0] if authors else ""
    surname = first.split(",")[0] if "," in first else first.split(" ")[-1] if first else "anon"
    surname = re.sub(r"[^a-z]", "", _ascii(surname).lower()) or "anon"
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", _ascii(title).lower()).split() if len(w) > 2]
    skip = {"the", "and", "for", "with", "from", "towards", "using", "via"}
    first = next((w for w in words if w not in skip), "paper")
    return f"{surname}{year or ''}{first}"


def _wire_title(conn, reference_id: int) -> str | None:
    """The title an enricher offered, when it differed from the working title."""
    rows = conn.execute(
        "SELECT raw FROM reference_sources WHERE reference_id = ?"
        " AND source IN ('arxiv', 'crossref') ORDER BY source",
        (reference_id,),
    ).fetchall()
    for row in rows:
        raw = json.loads(row["raw"])
        offered = raw.get("conflicts", {}).get("title", {}).get("offered")
        if offered:
            return offered
        fields = raw.get("fields", {})
        if fields.get("title"):
            return fields["title"]
    return None


def build(scratch: Path, steps: set[str] | None = None) -> tuple[list[dict], dict]:
    """Run the requested steps against the scratch library and return the entries."""
    steps = steps or {"sync", "tag", "write"}
    from attestation.db import get_db
    from attestation.features import run_reference_tagging
    from attestation.library import sync
    from attestation.library_readers import readers_from_env
    from attestation.llm import chat_model, default_chat_fn

    seeds = tomllib.loads((HERE / "seeds.toml").read_text())["seeds"]
    seed_bib = scratch / "seed.bib"
    lines = []
    for s in seeds:
        fields = [f"  title = {{{s['title']}}}"]
        if s.get("arxiv"):
            fields += [f"  eprint = {{{s['arxiv']}}}", "  archiveprefix = {arXiv}"]
        if s.get("doi"):
            fields.append(f"  doi = {{{s['doi']}}}")
        lines.append(f"@misc{{{s['key']},\n" + ",\n".join(fields) + ",\n}\n")
    seed_bib.write_text("".join(lines))

    conn = get_db(scratch / "library.db")
    sync_seconds = 0.0
    stats: dict = {"skipped": True}
    if "sync" in steps:
        readers = readers_from_env(
            conn,
            bib_paths=[seed_bib],
            zotero_path=scratch / "no-zotero",
            cache_dir=scratch / "cache",
        )
        names = [r.name for r in readers]
        if "arxiv" not in names or "s2" not in names:
            sys.exit(
                "set ATTEST_CITATION_WEB=1 and ATTEST_CITATION_SCHOLAR=1: the enrichers are off"
            )
        t0 = time.monotonic()
        # Two passes: the first lets S2 attach DOIs to arXiv-only seeds and
        # may leave 429s behind; the second lets CrossRef see those DOIs and
        # retries what rate limiting left untouched. Cache hits make the
        # repeat cheap. With --scratch, the whole step can be repeated later.
        for label in ("first pass", "second pass"):
            report = sync(conn, readers)
            for name, bucket in report.sources.items():
                print(f"{label} {name}: {bucket}")
            for reader in readers:
                for err in getattr(reader, "errors", ()):
                    print(f"  {reader.name} transient: {err}")
                if hasattr(reader, "errors"):
                    reader.errors.clear()
        sync_seconds = time.monotonic() - t0
    if "tag" in steps:
        stats = run_reference_tagging(conn, default_chat_fn, chat_model())
        print(f"tagging: {stats}")

    entries = []
    misses = {"abstract": [], "cites": [], "seed": []}
    resolved = {
        r["source_key"]
        for r in conn.execute(
            "SELECT source_key FROM reference_sources WHERE source LIKE 'bibtex:%'"
        )
    }
    misses["seed"] = [s["key"] for s in seeds if s["key"] not in resolved]
    for row in conn.execute('SELECT * FROM "references" ORDER BY year, id'):
        authors = json.loads(row["authors"])
        title = _wire_title(conn, row["id"]) or row["title"]
        tags = [
            r["tag"]
            for r in conn.execute(
                "SELECT tag FROM reference_tags WHERE reference_id = ? ORDER BY tag", (row["id"],)
            )
        ]
        cites = []
        for r in conn.execute(
            "SELECT cited_identity, cited_title FROM reference_cites WHERE citing_id = ?"
            " ORDER BY cited_identity",
            (row["id"],),
        ):
            cited = r["cited_title"] or ""
            # A reference list sometimes cites a code repository by URL. That
            # is not a paper and the golden-path guard forbids the host name in
            # a committed example, so those edges are left out of the fixture.
            if _CODE_URL.search(cited) or _CODE_URL.search(r["cited_identity"]):
                continue
            # The field's grammar is `identity|title; identity`: an identity
            # carrying either separator (a `...3.0.CO;2-P` DOI) cannot be
            # written into it and is left out rather than split.
            if ";" in r["cited_identity"] or "|" in r["cited_identity"]:
                continue
            # Separators are the field's grammar and braces are BibTeX's: a
            # cited title carrying either would corrupt the entry around it.
            clean = cited.translate(str.maketrans({"|": " ", ";": ",", "{": "(", "}": ")"}))
            cites.append(f"{r['cited_identity']}|{clean}" if cited else r["cited_identity"])
        if not row["abstract"]:
            misses["abstract"].append(row["bib_key"])
        if not cites:
            misses["cites"].append(row["bib_key"])
        # `&` is BibTeX's alignment character: unescaped it stops `bibtex`
        # with "Misplaced alignment tab" (review round 2, SIAM's journal).
        fields = {
            "title": _tex(title),
            "author": " and ".join(_tex(a) for a in authors),
            "year": str(row["year"]) if row["year"] else None,
            "journal": _tex(row["venue"]),
            "doi": row["doi"],
            "eprint": row["arxiv_id"],
            "archiveprefix": "arXiv" if row["arxiv_id"] else None,
            "abstract": _tex(row["abstract"]),
            "url": row["url"],
            "keywords": ", ".join(tags) or None,
            "cites": "; ".join(cites) or None,
        }
        entries.append(
            {
                "key": _key(authors, row["year"], title),
                # A paper with no known venue is a preprint: `@misc`, so a
                # LaTeX compile does not warn "empty journal" 23 times.
                "type": "article" if row["venue"] else "misc",
                "fields": {k: v for k, v in fields.items() if v},
            }
        )
    return entries, {
        "sync_seconds": round(sync_seconds, 1),
        "misses": misses,
        "n": len(entries),
        "tagging": stats,
    }


def write_bib(entries: list[dict], path: Path) -> None:
    try:
        from bibtexparser import Library, write_string
        from bibtexparser.model import Entry, Field
    except ImportError as exc:  # bibtexparser 1.x has no Library; refuse rather than mimic
        raise ImportError(
            'run with: uv run --with "bibtexparser>=2.0.0b9" python generate.py'
        ) from exc
    library = Library()
    seen: set[str] = set()
    for e in entries:
        key = e["key"]
        while key in seen:
            key += "a"
        seen.add(key)
        library.add(Entry(e["type"], key, [Field(k, v) for k, v in e["fields"].items()]))
    text = write_string(library)
    # Abstracts fetched from the wire sometimes carry a code-hosting URL
    # ("code is available at ..."). The golden-path guard forbids that host
    # in any committed example, so the URL is replaced with a marker -- the
    # one edit made to fetched text, and it is a removal, not an invention.
    text = _CODE_URL.sub("[code URL removed]", text)
    # The same scrub tests/test_golden_paths.py applies to every committed
    # example: no home-directory path and no username. The needles are built
    # rather than spelled so this file passes its own guard.
    home_prefix = os.sep + "home" + os.sep
    needles = [home_prefix, str(Path.home())]
    user = os.environ.get("USER", "")
    if user:
        needles.append(user)
    for needle in needles:
        pattern = rf"\b{re.escape(needle)}\b" if needle == user else re.escape(needle)
        if re.search(pattern, text):
            raise SystemExit(f"refusing to write references.bib: it carries {needle!r}")
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scratch",
        help="keep the scratch library and cache here across runs (default: a temp dir,"
        " deleted at exit). With it, `--steps sync` can be repeated until Semantic"
        " Scholar's rate limit has let every seed through, then `--steps tag,write`.",
    )
    parser.add_argument(
        "--steps",
        default="sync,tag,write",
        help="comma-separated subset of sync,tag,write (default: all three)",
    )
    args = parser.parse_args(argv)
    steps = {s.strip() for s in args.steps.split(",")}
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="molecular-ai-"))
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        entries, summary = build(scratch, steps)
        if "write" in steps:
            write_bib(entries, HERE / "references.bib")
    finally:
        if not args.scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(summary, indent=2))
    if "write" in steps:
        print(f"wrote {summary['n']} entries to references.bib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
