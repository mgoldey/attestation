"""Link the figures a draft includes to the files that produced them.

The claim checker ties a number in prose to a run. A figure has no such tie:
`loss.png` keeps being included after the results behind it moved. This module
closes that with an annotation beside the figure, in the claim comment's own
style so it renders as nothing:

    ![training loss](figs/loss.png)
    <!-- figure: figs/loss.png from=results/a.json,results/b.json sha256=<hex> -->

    \\includegraphics{figs/loss}
    % figure: figs/loss.png from=results/a.json,results/b.json

Four verdicts: `fresh`, `stale` (a `from=` file is newer than the figure, or
`sha256=` no longer matches its bytes), `orphan` (included, not annotated) and
`missing` (the figure or a `from=` file is absent). `fresh` means only that no
declared source is newer -- never that the figure was drawn from them, and an
image is never opened. Deterministic, read-only, no network and no model.
Design: docs/superpowers/specs/2026-11-19-figure-provenance-design.md.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote

from attestation import latex

# Extensions LaTeX tries for an extensionless \includegraphics path, in the
# order a pdflatex build prefers them.
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

# The MCP block's caps. Measured against the 7000-char response ceiling with a
# 50-figure draft of long paths in tests/test_response_size.py.
MAX_FIGURE_FINDINGS = 8
MAX_FIGURE_MALFORMED = 4
MAX_MESSAGE_CHARS = 120
MAX_PATH_CHARS = 80

NOTE = "fresh means no declared source is newer than the figure, not that it was drawn from them"

_ALLOWED_FIELDS = frozenset({"from", "sha256"})
_FIELD_TOKEN = re.compile(r"(\w+)=(\S+)")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]+:|//)")

_FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_NOTE = re.compile(r"<!--\s*figure:\s*(?P<body>.*?)\s*-->", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?:<(?P<a>[^>]*)>|(?P<b>[^)\s]+))")
_HTML_IMG = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?:\"(?P<d>[^\"]*)\"|'(?P<s>[^']*)')", re.DOTALL | re.IGNORECASE
)
_TEX_NOTE = re.compile(r"%*\s*figure:\s*(?P<body>.*?)\s*$")
_TEX_GRAPHIC = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\]\s*)*\{(?P<p>[^{}]*)\}")
_TEX_GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\}\s*)*)\}")
_TEX_GROUP = re.compile(r"\{([^{}]*)\}")


class FigureVerdict(StrEnum):
    """The closed set of figure outcomes (a str, like `claims.VerdictKind`)."""

    FRESH = "fresh"
    STALE = "stale"
    ORPHAN = "orphan"
    MISSING = "missing"


@dataclass
class Figure:
    """One distinct figure file a draft includes, and what was found.

    `path` is the reference as written in the document, `resolved` the file it
    names on disk (or would, when `missing`), `sources` the `from=` files.
    """

    path: str
    file: str
    line: int
    resolved: str
    verdict: FigureVerdict
    message: str
    sources: list[str] = field(default_factory=list)


@dataclass
class _Note:
    """One parsed `figure:` annotation."""

    doc: Path
    line: int
    path: str
    sources: list[str]
    sha256: str | None


@dataclass
class _Place:
    """One figure reference in one document, with the files it could mean."""

    doc: Path
    line: int
    ref: str
    candidates: list[Path]

    @property
    def resolved(self) -> Path:
        """The first candidate that exists, else the first candidate."""
        return next((c for c in self.candidates if c.is_file()), self.candidates[0])


# --------------------------------------------------------------------------- reading


def _blank(match: re.Match) -> str:
    """Same-shape whitespace: every newline kept, so line numbers stay true."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _mask_code(text: str) -> str:
    """Markdown with fenced and inline code blanked: code shows text, not a figure."""
    return _INLINE_CODE.sub(_blank, _FENCE.sub(_blank, text))


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _markdown_doc(text: str):
    """(references, annotations, graphicspath) of a Markdown document.

    References come from text with comments AND code masked; annotations from
    text with only code masked, since an annotation IS a comment.
    """
    coded = _mask_code(text)
    visible = _HTML_COMMENT.sub(_blank, coded)
    refs = []
    for m in _MD_IMAGE.finditer(visible):
        refs.append((_line_of(visible, m.start()), unquote(m.group("a") or m.group("b"))))
    for m in _HTML_IMG.finditer(visible):
        refs.append((_line_of(visible, m.start()), m.group("d") or m.group("s") or ""))
    notes = [(_line_of(coded, m.start()), m.group("body")) for m in _MD_NOTE.finditer(coded)]
    return sorted(refs), notes, []


def _tex_notes(text: str):
    """(line, body) for every `% figure:` comment in `text`."""
    for number, line in enumerate(text.split("\n"), 1):
        k = latex.comment_start(line)
        match = _TEX_NOTE.match(line, k + 1) if k >= 0 else None
        if match:
            yield number, match.group("body")


def _tex_doc(text: str):
    """(references, annotations, graphicspath) of a LaTeX source, comments ignored."""
    visible = latex.strip_comments(text)
    refs = [
        (_line_of(visible, m.start()), m.group("p").strip()) for m in _TEX_GRAPHIC.finditer(visible)
    ]
    dirs = [g for m in _TEX_GRAPHICSPATH.finditer(visible) for g in _TEX_GROUP.findall(m.group(1))]
    return refs, list(_tex_notes(text)), dirs


def _read_doc(doc: Path):
    text = doc.read_text(errors="replace")
    return _tex_doc(text) if latex.is_tex(doc) else _markdown_doc(text)


def _is_literal(ref: str) -> bool:
    """A local path we can resolve: not empty, not a URL, not built from macros."""
    if not ref or _SCHEME.match(ref):
        return False
    return not any(ch in ref for ch in "\\#$")


# --------------------------------------------------------------------------- resolving


def _extensions(ref: str, doc: Path) -> tuple[str, ...]:
    """The suffixes to try: LaTeX extends an extensionless name, Markdown never."""
    if not latex.is_tex(doc) or Path(ref).suffix.lower() in GRAPHIC_EXTENSIONS:
        return ("",)
    return GRAPHIC_EXTENSIONS


def _bases(doc: Path, root: Path, graphicspath: list[str]) -> list[Path]:
    """Folders a reference may be relative to, nearest first, each once."""
    found = [doc.parent, root.parent]
    found += [root.parent / g for g in graphicspath]
    return list(dict.fromkeys(found))


def _candidates(ref: str, doc: Path, root: Path, graphicspath: list[str]) -> list[Path]:
    out = []
    for base in _bases(doc, root, graphicspath):
        for ext in _extensions(ref, doc):
            out.append(Path(os.path.normpath(base / (ref + ext))))
    return list(dict.fromkeys(out))


def _source_file(name: str, note: _Note, root: Path) -> Path | None:
    """A `from=` path as an existing file: beside the annotation, then the root's folder."""
    for base in (note.doc.parent, root.parent):
        candidate = Path(os.path.normpath(base / name))
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- annotations


def _fields(tokens: list[str]) -> dict[str, str] | str:
    """The `name=value` tokens after the path, or the first one that is not allowed."""
    fields: dict[str, str] = {}
    for token in tokens:
        match = _FIELD_TOKEN.fullmatch(token)
        if match is None or match.group(1) not in _ALLOWED_FIELDS:
            return f"unexpected {token!r} (fields: from=a,b sha256=<hex>)"
        fields.setdefault(match.group(1), match.group(2))
    return fields


def _parse_note(doc: Path, line: int, body: str) -> _Note | str:
    """One annotation body as a `_Note`, or the complaint about it."""
    where = f"{doc}:{line}"
    tokens = body.split()
    if not tokens or "=" in tokens[0]:
        return f"{where}: expected a figure path first, got {body[:60]!r}"
    fields = _fields(tokens[1:])
    if isinstance(fields, str):
        return f"{where}: {fields}"
    problem = _field_problem(fields)
    if problem:
        return f"{where}: {problem}"
    sha = fields.get("sha256")
    sources = [s for s in fields.get("from", "").split(",") if s]
    return _Note(doc, line, tokens[0], sources, sha.lower() if sha else None)


def _field_problem(fields: dict[str, str]) -> str | None:
    """What is wrong with the parsed fields, or None."""
    sha = fields.get("sha256")
    if sha is not None and not _SHA256.fullmatch(sha):
        return f"sha256= must be 64 hex characters, got {sha[:20]!r}"
    if sha is None and not [s for s in fields.get("from", "").split(",") if s]:
        return "annotation links nothing; add from=<files> or sha256=<hex>"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- judging


def _judge(place: _Place, note: _Note | None, root: Path) -> tuple[FigureVerdict, str, list[str]]:
    """The verdict, message and `from=` files for one figure."""
    figure = place.resolved
    if not figure.is_file():
        return FigureVerdict.MISSING, f"figure file not found: {place.ref}", []
    if note is None:
        return (
            FigureVerdict.ORPHAN,
            "no figure annotation links this to the files that produced it",
            [],
        )
    found = {s: _source_file(s, note, root) for s in note.sources}
    gone = [s for s, p in found.items() if p is None]
    names = list(found)
    if gone:
        return FigureVerdict.MISSING, f"source not found: {', '.join(gone)}", names
    reasons = _stale_reasons(figure, note, [p for p in found.values() if p])
    if reasons:
        return FigureVerdict.STALE, "; ".join(reasons), names
    return FigureVerdict.FRESH, "no declared source is newer than the figure", names


def _stale_reasons(figure: Path, note: _Note, sources: list[Path]) -> list[str]:
    """Why the figure no longer matches what it was linked to; empty when it does."""
    reasons = []
    made = figure.stat().st_mtime_ns
    newer = [p.name for p in sources if p.stat().st_mtime_ns > made]
    if newer:
        reasons.append(f"{', '.join(newer)} changed after the figure was written")
    if note.sha256 is not None and _sha256(figure) != note.sha256:
        reasons.append("figure bytes no longer match sha256=")
    return reasons


# --------------------------------------------------------------------------- scanning


def _scan_unit(root: Path, docs: list[Path]) -> tuple[list[Figure], list[str], int]:
    """(figures, malformed, skipped) for one root document and its `\\input` tree."""
    places: list[_Place] = []
    parsed: list[_Note] = []
    problems: list[str] = []
    skipped = 0
    graphicspath: list[str] = []
    raw_refs = []
    for doc in docs:
        refs, notes, dirs = _read_doc(doc)
        graphicspath += dirs
        raw_refs += [(doc, line, ref) for line, ref in refs]
        for line, body in notes:
            note = _parse_note(doc, line, body)
            if isinstance(note, _Note):
                parsed.append(note)
            else:
                problems.append(note)
    for doc, line, ref in raw_refs:
        if _is_literal(ref):
            places.append(_Place(doc, line, ref, _candidates(ref, doc, root, graphicspath)))
        else:
            skipped += 1
    attached = _attach(places, parsed, root, graphicspath, problems)
    figures = [_figure(p, attached.get(id(p)), root) for p in places]
    return figures, problems, skipped


def _attach(places, notes, root, graphicspath, problems) -> dict[int, _Note]:
    """Each note onto the figure(s) whose candidate files it names; a second
    note for one figure is reported, never silently preferred."""
    attached: dict[int, _Note] = {}
    for note in notes:
        wanted = set(_candidates(note.path, note.doc, root, graphicspath))
        for place in places:
            if not wanted.intersection(place.candidates):
                continue
            if id(place) in attached:
                problems.append(
                    f"{note.doc}:{note.line}: duplicate annotation for {place.ref}"
                    f" (first at line {attached[id(place)].line})"
                )
            else:
                attached[id(place)] = note
    return attached


def _figure(place: _Place, note: _Note | None, root: Path) -> Figure:
    verdict, message, sources = _judge(place, note, root)
    return Figure(
        path=place.ref,
        file=str(place.doc),
        line=place.line,
        resolved=str(place.resolved),
        verdict=verdict,
        message=message,
        sources=sources,
    )


def _markdown_files(root: Path) -> list[Path]:
    """Every `*.md` below a folder, skipping hidden and vendored folders BELOW it."""
    out = []
    for doc in sorted(root.rglob("*.md")):
        parts = doc.relative_to(root).parts
        if not any(p.startswith(".") or p == "node_modules" for p in parts):
            out.append(doc.resolve())
    return out


def _units(root: Path) -> list[tuple[Path, list[Path]]]:
    """(root document, documents it reads) for a file or folder.

    A `.tex` paper is a tree read from its root, so a chapter's figure path
    resolves from the paper's folder; files no paper reaches stand alone.
    """
    if root.is_file():
        return [(root.resolve(), latex.tree(root) if latex.is_tex(root) else [root.resolve()])]
    units = [(p.resolve(), latex.tree(p)) for p in latex.paper_roots(root)]
    covered = {d for _, docs in units for d in docs}
    loose = [*latex.tex_files(root), *_markdown_files(root)]
    units += [(d, [d]) for d in sorted({p.resolve() for p in loose} - covered)]
    return units


# When sightings of one file disagree, the most severe wins: a stale or
# missing annotation in one document must not be hidden by a fresh one in
# another (it was, and `attest claims --figures` exited 0). An annotated
# sighting still beats an orphan.
_MERGE_RANK = {
    FigureVerdict.ORPHAN: 0,
    FigureVerdict.FRESH: 1,
    FigureVerdict.STALE: 2,
    FigureVerdict.MISSING: 3,
}


def _merge(found: list[Figure]) -> list[Figure]:
    """One finding per distinct figure file: the most severe sighting of it."""
    best: dict[str, Figure] = {}
    for fig in found:
        seen = best.get(fig.resolved)
        if seen is None or _MERGE_RANK[fig.verdict] > _MERGE_RANK[seen.verdict]:
            best[fig.resolved] = fig
    return list(best.values())


def check(root: Path) -> dict:
    """Every figure under `root` (a Markdown or `.tex` file, or a folder), judged.

    Read-only. Returns `figures` (the findings, one per distinct file), `total`,
    `counts` by verdict, `malformed` annotations, `skipped` (URLs and macro-built
    paths, never fetched) and `base`, the folder findings are shown relative to.
    """
    root = Path(root).expanduser()
    found: list[Figure] = []
    malformed: list[str] = []
    skipped = 0
    for doc_root, docs in _units(root):
        figs, bad, n = _scan_unit(doc_root, docs)
        found += figs
        malformed += bad
        skipped += n
    figures = _merge(found)
    return {
        "total": len(figures),
        "counts": Counter(f.verdict for f in figures),
        "figures": figures,
        "malformed": malformed,
        "skipped": skipped,
        "base": root.resolve() if root.is_dir() else root.resolve().parent,
    }


# --------------------------------------------------------------------------- reporting

_SEVERITY = {FigureVerdict.MISSING: 0, FigureVerdict.STALE: 1, FigureVerdict.ORPHAN: 2}


def failing(report: dict) -> int:
    """How many findings should fail a gate: stale, missing, malformed.

    Orphans do not -- like `claims.coverage`, an unannotated figure is a lint.
    """
    counts = report["counts"]
    bad = counts.get(FigureVerdict.STALE, 0) + counts.get(FigureVerdict.MISSING, 0)
    return bad + len(report["malformed"])


def _show(path: str, base: Path) -> str:
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return path


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def empty_block() -> dict:
    """The `figures` block of a draft with no figures: same keys, nothing in them."""
    return {
        "total": 0,
        "counts": {},
        "findings": [],
        "malformed": [],
        "truncated": 0,
        "skipped": 0,
        "note": "",
    }


def block(report: dict) -> dict:
    """The bounded `figures` block of `runs.claims_check`.

    `total` and `counts` are always complete; `findings` lists only non-fresh
    figures, worst first, capped at MAX_FIGURE_FINDINGS with `truncated` saying
    how many were cut. A cap is a response-size decision, not a verdict.
    """
    base = report["base"]
    bad = [f for f in report["figures"] if f.verdict != FigureVerdict.FRESH]
    bad.sort(key=lambda f: _SEVERITY[f.verdict])
    shown = bad[:MAX_FIGURE_FINDINGS]
    return {
        "total": report["total"],
        "counts": dict(report["counts"]),
        "findings": [
            {
                "verdict": f.verdict,
                "file": _clip(_show(f.file, base), MAX_PATH_CHARS),
                "line": f.line,
                "figure": _clip(f.path, MAX_PATH_CHARS),
                "message": _clip(f.message, MAX_MESSAGE_CHARS),
            }
            for f in shown
        ],
        "malformed": [
            _clip(_show(m, base), MAX_PATH_CHARS + MAX_MESSAGE_CHARS)
            for m in report["malformed"][:MAX_FIGURE_MALFORMED]
        ],
        "truncated": len(bad)
        - len(shown)
        + max(0, len(report["malformed"]) - MAX_FIGURE_MALFORMED),
        "skipped": report["skipped"],
        "note": NOTE if report["total"] else "",
    }


def message_suffix(report: dict) -> str:
    """`; figures: 1 stale, 1 missing` when a figure needs attention, else empty.

    Appended to `runs.claims_check`'s message so a caller that relays only the
    message cannot miss it; a draft with no failing figure keeps its message
    byte for byte.
    """
    counts = report["counts"]
    parts = [
        f"{counts[v]} {v}" for v in (FigureVerdict.STALE, FigureVerdict.MISSING) if counts.get(v)
    ]
    return f"; figures: {', '.join(parts)}" if parts else ""
