"""`attest bib`: the citation manager from the shell.

Its own module so cli.py does not grow by seven commands; `cli.build_parser`
calls `register` and imports this lazily, so `attest --help` pays nothing
for it. Every command works on the library store (`attest library sync` fills
it); edits go to the owner's generated .bib, never to an uploaded one. See
docs/guides/citation-manager.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attestation.cli import HELP, _documented, _embedder_or_none, fail, open_db


def register(sub, add_db) -> None:
    """Attach `attest bib <command>` to the top-level subparsers."""
    sp = sub.add_parser("bib", help=HELP["bib"])
    add_db(sp)
    bib = sp.add_subparsers(dest="bib_command", required=True)

    p = bib.add_parser("search", help=HELP["bib.search"])
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--author", help="surname filter")
    p.add_argument("--year", type=int)
    p.add_argument("--tag")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_bib_search)

    p = bib.add_parser("show", help=HELP["bib.show"])
    p.add_argument("key", help="citation key, DOI, arXiv id, or library identity")
    p.add_argument("--style", help="citation style (default: $ATTEST_CITATION_STYLE, else apa)")
    p.add_argument("--json", action="store_true", help="print the record as JSON")
    p.set_defaults(func=cmd_bib_show)

    p = bib.add_parser("format", help=HELP["bib.format"])
    p.add_argument("keys", nargs="+", help="citation keys, DOIs or arXiv ids")
    p.add_argument("--style", help="apa, ieee, chicago-author-date, nature or vancouver")
    p.add_argument("--bibtex", action="store_true", help="print BibTeX instead")
    p.set_defaults(func=cmd_bib_format)

    p = bib.add_parser("import", help=HELP["bib.import"])
    p.add_argument("path", nargs="?", help="a .bib file (or zotero.sqlite with --zotero)")
    p.add_argument(
        "--zotero", action="store_true", help="read a local zotero.sqlite (default: ~/Zotero)"
    )
    p.set_defaults(func=cmd_bib_import)

    p = bib.add_parser("edit", help=HELP["bib.edit"])
    p.add_argument("key", help="citation key, DOI or arXiv id")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="title, authors (A and B), year, venue, doi, arxiv, url, abstract, type,"
        " volume, pages, or any other BibTeX field; an empty value clears it",
    )
    p.add_argument("--note", help="the owner's note (empty clears it)")
    p.add_argument("--tag", action="append", default=[], help="add a tag")
    p.add_argument("--untag", action="append", default=[], help="remove a tag")
    p.add_argument("--user", help="whose .bib it goes to (default: $ATTEST_DESK_USER, else owner)")
    p.set_defaults(func=cmd_bib_edit)

    p = bib.add_parser("merge", help=HELP["bib.merge"])
    p.add_argument("keep", nargs="?", help="the reference to keep")
    p.add_argument("drop", nargs="?", help="the duplicate folded into it")
    p.add_argument("--all", action="store_true", help="merge every listed duplicate pair")
    p.set_defaults(func=cmd_bib_merge)

    p = bib.add_parser("export", help=HELP["bib.export"])
    p.add_argument("keys", nargs="*", help="references to export (default: the whole library)")
    p.add_argument("--out", help="write to this file (refuses to overwrite without --force)")
    p.add_argument("--style", help="formatted text in this style instead of BibTeX")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_bib_export)


def _style(arg: str | None) -> str:
    """The style id for a --style flag (or the configured default). Raises
    ValueError naming the valid ids."""
    from attestation import cite_format

    return cite_format.style_id(arg) if arg else cite_format.default_style()


@_documented("bib.search")
def cmd_bib_search(args: argparse.Namespace) -> int:
    """The same search as `attest library search` and cite.search: semantic
    when the library is embedded and the model server answers, substring
    otherwise -- and the last line says which."""
    from attestation import cite_format, library

    with open_db(args.db, need_vectors=True) as conn:
        res = library.search(
            conn,
            args.query,
            embedder=_embedder_or_none(),
            author=args.author,
            year=args.year,
            tag=args.tag,
            limit=args.limit,
        )
    for h in res.hits:
        who = h.authors[0].split(",")[0] if h.authors else "?"
        title = cite_format.plain(h.title)[:70]
        print(f"{h.bib_key or h.identity:32s} {h.year or '----'}  {who:16.16s} {title}")
    print(f"{res.n_matches} match(es); " + ("semantic" if res.semantic else res.caveat or ""))
    return 0


@_documented("bib.show")
def cmd_bib_show(args: argparse.Namespace) -> int:
    from attestation import citation_manager

    with open_db(args.db) as conn:
        try:
            rec = citation_manager.show(conn, args.key, _style(args.style))
        except (LookupError, ValueError) as exc:
            return fail(str(exc))
    if args.json:
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0
    _print_record(rec)
    return 0


def _shown(value) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value)


def _print_record(rec: dict) -> None:
    for name in ("key", "identity", "title", "authors", "year", "venue", "entry_type"):
        if name in rec:
            value = " and ".join(rec[name]) if name == "authors" else rec[name]
            print(f"{name:11s} {value}")
    for name in ("volume", "pages", "doi", "arxiv_id", "url", "tags", "sources", "aliases", "note"):
        if rec.get(name):
            print(f"{name:11s} {_shown(rec[name])}")
    for sid, text in rec["formatted"].items():
        print(f"\n{sid}: {text}")
    print("\n" + rec["bibtex"].rstrip())


@_documented("bib.format")
def cmd_bib_format(args: argparse.Namespace) -> int:
    """Exit 1 when a key resolves to nothing; the ones that do still print."""
    from attestation import citation_manager

    with open_db(args.db) as conn:
        try:
            out = citation_manager.format_keys(conn, args.keys, _style(args.style))
        except ValueError as exc:
            return fail(str(exc))
    for ref in out["references"]:
        print(ref["bibtex"].rstrip() if args.bibtex else ref["text"])
        if args.bibtex:
            print()
    for key in out["missing"]:
        print(f"no library reference matches {key!r}", file=sys.stderr)
    return 1 if out["missing"] else 0


def _print_import(out: dict) -> None:
    print(
        f"{out['file']}: {out['entries']} entr{'y' if out['entries'] == 1 else 'ies'} --"
        f" +{out['added']} added, {out['merged']} merged, {out['unchanged']} unchanged"
    )
    for line in out.get("failed", []):
        print(f"  unreadable  {line}")
    for key in out.get("untitled", []):
        print(f"  untitled    {key}  (no title; not imported)")
    for key in out.get("duplicate_keys", []):
        print(f"  repeated    {key}  (defined twice in the file; the first was kept)")
    for c in out.get("key_clashes", []):
        print(f"  key clash   {c['key']}  (already names {c['store']} in the library)")


@_documented("bib.import")
def cmd_bib_import(args: argparse.Namespace) -> int:
    """Reads the file; never writes it. Exit 1 when nothing could be read."""
    from attestation import citation_manager

    if not args.zotero and not args.path:
        return fail("name a .bib file to import, or pass --zotero")
    with open_db(args.db) as conn:
        try:
            if args.zotero:
                path = Path(args.path).expanduser() if args.path else None
                out = citation_manager.import_zotero(conn, path)
            else:
                out = citation_manager.import_bib(conn, Path(args.path))
        except ValueError as exc:
            return fail(str(exc))
    _print_import({"added": 0, "merged": 0, "unchanged": 0, **out})
    return 0


def _parse_sets(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"--set takes FIELD=VALUE, got {pair!r}")
        out[name.strip()] = value
    return out


@_documented("bib.edit")
def cmd_bib_edit(args: argparse.Namespace) -> int:
    """Changes the store row, logs the edit, and saves the reference to the
    owner's bibliography so `<ATTEST_BIB_OUT>/<owner>.bib` carries it.
    Uploaded .bib files are never written."""
    from attestation import citation_manager

    with open_db(args.db) as conn:
        try:
            out = citation_manager.edit(
                conn,
                args.key,
                _parse_sets(args.set),
                note=args.note,
                add_tags=args.tag,
                remove_tags=args.untag,
                user=args.user,
            )
        except (LookupError, ValueError) as exc:
            return fail(str(exc))
    changed = ", ".join(out["changed"]) or "no field changed"
    where = out["file"] or "not written: set ATTEST_BIB_OUT"
    print(f"{out['cite_key']}: {changed}; in {out['owner']}'s bibliography ({where})")
    return 0


@_documented("bib.merge")
def cmd_bib_merge(args: argparse.Namespace) -> int:
    """With no keys, lists the duplicate pairs (the store's own evidence: the
    same DOI, arXiv id or title and year) and changes nothing."""
    from attestation import citation_manager

    with open_db(args.db) as conn:
        if args.keep and args.drop:
            pairs = [{"keep_key": args.keep, "drop_key": args.drop}]
        elif args.keep:
            return fail("name both references: attest bib merge KEEP DROP")
        else:
            pairs = citation_manager.duplicates(conn)
            if not args.all:
                for p in pairs:
                    print(
                        f"  {p['keep_key']}  <-  {p['drop_key']}  ({p['why']})  {p['title'][:60]}"
                    )
                print(f"{len(pairs)} duplicate pair(s); merge one with: attest bib merge KEEP DROP")
                return 0
        for p in pairs:
            try:
                out = citation_manager.merge(conn, p["keep_key"], p["drop_key"])
            except (LookupError, ValueError) as exc:
                print(f"  skipped {p['drop_key']}: {exc}", file=sys.stderr)
                continue
            print(f"merged {out['dropped']} into {out['kept']} ({len(out['aliases'])} alias(es))")
    return 0


@_documented("bib.export")
def cmd_bib_export(args: argparse.Namespace) -> int:
    """BibTeX through the same writer as the owner's .bib (fields the source
    carried are kept); `--style` writes formatted text, one per line."""
    target = Path(args.out).expanduser() if args.out else None
    if target is not None and target.exists() and not args.force:
        return fail(f"{target} exists; pass --force to overwrite")
    with open_db(args.db) as conn:
        try:
            texts, missing, n = _export_texts(conn, args.keys, args.style)
        except ValueError as exc:
            return fail(str(exc))
    if not texts:
        return fail("no references match; nothing written")
    _emit(("\n".join(texts) + "\n") if args.style else "\n".join(texts), target, n)
    for key in missing:
        print(f"no library reference matches {key!r}", file=sys.stderr)
    return 1 if missing else 0


def _emit(text: str, target: Path | None, n: int) -> None:
    """To stdout, or to `target` with a count on stderr (stdout stays the data)."""
    if target is None:
        sys.stdout.write(text)
        return
    target.write_text(text)
    print(f"wrote {n} reference(s) to {target}", file=sys.stderr)


def _export_texts(conn, keys: list[str], style: str | None) -> tuple[list[str], list[str], int]:
    """(texts, keys nothing matched, references exported) for `bib export`."""
    from attestation import citation_manager, library

    if not keys:
        rows = library.select_rows(conn)
        return _whole_library(conn, rows, style), [], len(rows)
    out = citation_manager.format_keys(conn, keys, _style(style))
    texts = [r["text"] if style else r["bibtex"] for r in out["references"]]
    return texts, out["missing"], len(out["references"])


def _whole_library(conn, rows, style: str | None) -> list[str]:
    """Every row, as formatted text in `style`, else as BibTeX entries whose
    repeated keys take a letter suffix (library.export_bib's rule)."""
    from attestation import cite_format, library

    if style:
        texts = cite_format.formatted(conn, rows, _style(style), store=True)
        return [texts[r["id"]] for r in rows]
    return [library.export_bib(rows)] if rows else []
