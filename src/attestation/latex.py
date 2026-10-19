"""LaTeX sources as first-class documents for the claim checker and `cite.check`.

`claims.py` reads Markdown, where a claim is an HTML comment. A paper is
usually LaTeX, so this module gives `.tex` the same three things:

- **Claims** are LaTeX comments, in the same grammar as the Markdown form:

      WER fell to \\num{0.0642}. % claim: speech/kdsweep_t4 metric=wer value=0.0642

  A comment, not a `\\claim{}` macro, for the reason the Markdown form is an
  HTML comment: it compiles to nothing with no package to install, `grep` and
  `git diff` read it, and it sits on the line (or the table row) it describes.
  A macro would need a `.sty` in every paper's tree and would break the build
  of anyone who copied the paragraph without it. The design spec
  (2026-08-30-latex-and-reviewer-audits-design.md, Option A) chose this form.

- **Coverage** reads numbers out of prose, table cells, `\\num{}` and
  `\\SI{}{}`, and a number typeset alone in inline math (`$0.053$`), and
  ignores what is structural: comments, equations, `\\label`/`\\ref`/`\\cite`
  arguments, lengths (`0.5\\textwidth`, `2pt`) and package options.
  `1{,}234.5` is read as 1234.5.

- **Citations**: every `*cite*` command's keys (`\\cite`, `\\citep[see][p.~3]{a,b}`,
  `\\parencite`, `\\textcite`, `\\autocite`, `\\footcite`, `\\cites{a}{b}` ...) are
  checked against the paper's `.bib` files. Once the paper names its files
  (`\\bibliography{}` / `\\addbibresource{}`), a key counts only if one of
  those files defines it, as BibTeX would read it; a key found only in some
  other file is reported with `defined_in_unloaded`. The rules match
  AgentMarkit's `latex-paper` skill, so the two never disagree on one paper.

A paper is a tree: `\\input{}`, `\\include{}` and `\\subfile{}` are followed
from the root file with a cycle guard, and every finding names its file.
Read-only throughout; nothing here edits a `.tex` or a `.bib`.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

# `% claim: ...` -- the comment text after the first unescaped `%`. Extra `%`s
# are allowed so `%% claim:` (a common comment style) is read the same way.
_CLAIM_MARK = re.compile(r"%*\s*claim:\s*(?P<body>.*?)\s*$")

_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]*)\}")
_DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b")

# Any macro with "cite" in its name; optional arguments come first, keys last.
_CITE_RE = re.compile(r"\\([A-Za-z]*[Cc]ite[A-Za-z]*)\*?")
# Macros with "cite" in the name whose argument is not a key: natbib's
# \setcitestyle{numbers} and \citestyle{plain}, biblatex's \DeclareCiteCommand.
NOT_CITATIONS = frozenset({"setcitestyle", "citestyle", "DeclareCiteCommand"})
_BIB_RESOURCE_RE = re.compile(r"\\(?:addbibresource|bibliography)\s*(?:\[[^\]]*\])?\{([^}]*)\}")
_BIB_ENTRY_RE = re.compile(r"^\s*@\s*([A-Za-z]+)\s*[{(]\s*([^,\s{}()]+)\s*,", re.MULTILINE)
_NON_ENTRIES = frozenset({"comment", "string", "preamble"})


def is_tex(path: Path) -> bool:
    """Whether `path` is a LaTeX source, by suffix."""
    return path.suffix.lower() == ".tex"


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


# --------------------------------------------------------------------------- comments


def comment_start(line: str) -> int:
    """Index of the `%` that opens this line's comment, or -1.

    `\\%` is a literal percent sign, not a comment; `\\\\%` is a line break
    followed by one. Stepping over every backslash pair gets both right.
    """
    i = 0
    while i < len(line):
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == "%":
            return i
        i += 1
    return -1


def strip_comments(text: str) -> str:
    """`text` with every comment removed and every newline kept."""
    out = []
    for line in text.split("\n"):
        k = comment_start(line)
        out.append(line if k < 0 else line[:k])
    return "\n".join(out)


def claim_annotations(text: str) -> Iterator[tuple[int, str]]:
    """(line number, claim body) for every `% claim:` comment in `text`."""
    for number, line in enumerate(text.split("\n"), 1):
        k = comment_start(line)
        if k < 0:
            continue
        match = _CLAIM_MARK.match(line, k + 1)
        if match:
            yield number, match.group("body")


# --------------------------------------------------------------------------- the tree


def _resolve_include(name: str, dirs: Iterable[Path]) -> Path | None:
    for base in dirs:
        candidate = base / name
        for path in (candidate, candidate.with_name(candidate.name + ".tex")):
            if path.is_file():
                return path
    return None


def tree(root: Path) -> list[Path]:
    """`root` and every file it reaches by `\\input`/`\\include`/`\\subfile`.

    Depth-first in document order, each file once (a cycle, or a table
    included twice, is read once). Names resolve against the root's folder,
    where LaTeX runs, then against the including file's own folder. A name
    that resolves to nothing is skipped: the build will say so, and this
    checker has nothing to read there.
    """
    root = Path(root).resolve()
    seen: dict[Path, None] = {}
    stack = [root]
    while stack:
        path = stack.pop()
        key = path.resolve()
        if key in seen:
            continue
        seen[key] = None
        children = []
        for match in _INCLUDE_RE.finditer(strip_comments(_read(path))):
            child = _resolve_include(match.group(1).strip(), (root.parent, path.parent))
            if child is not None:
                children.append(child)
        stack.extend(reversed(children))
    return list(seen)


def tex_files(root: Path) -> Iterator[Path]:
    """Every `*.tex` under `root`, skipping hidden and vendored folders below it."""
    for path in sorted(root.rglob("*.tex")):
        parts = path.relative_to(root).parts
        if any(p.startswith(".") or p == "node_modules" for p in parts):
            continue
        yield path


def paper_roots(root: Path) -> list[Path]:
    """The papers at `root`: the file itself, or each `\\documentclass` file below it."""
    root = Path(root).expanduser()
    if root.is_file():
        return [root] if is_tex(root) else []
    if not root.is_dir():
        return []
    return [p for p in tex_files(root) if _DOCUMENTCLASS_RE.search(strip_comments(_read(p)))]


# --------------------------------------------------------------------------- numbers

_DISPLAY_ENVS = r"(?:equation|align|alignat|gather|multline|flalign|eqnarray|displaymath|math)\*?"
_DISPLAY_RE = re.compile(r"\\begin\{(" + _DISPLAY_ENVS + r")\}.*?\\end\{\1\}", re.DOTALL)
_DISPLAY_DELIMS = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\\)\$(?P<d>[^$]+?)(?<!\\)\$|\\\((?P<p>.*?)\\\)", re.DOTALL)
# What may sit beside a number in math and still leave "just a number":
# signs, \pm, \%, thin spaces, braces and the {,} thousands separator.
_MATH_NOISE = re.compile(r"\\(?:pm|mp|%|,|;|!)|[{}\s]")
_PURE_NUMBER = re.compile(r"[-+−\d.,]*\d[-+−\d.,]*")

# siunitx: the first argument is the number (\SI and \qty also take a unit).
_SIUNITX = re.compile(r"\\(?:num|SI|qty)\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}(?:\s*\{[^{}]*\})?")
_UNIT_ONLY = re.compile(r"\\(?:si|unit)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}")

# Macros whose arguments are never a measurement. Their whole call is blanked.
# `(?![A-Za-z])` ends the name: without it `include` matched the start of
# `\includegraphics` and left its options unmasked.
_STRUCTURAL = re.compile(
    r"\\(?:label|ref|eqref|[cC]ref|autoref|pageref|nameref|url|href|input|include|subfile"
    r"|includegraphics|usepackage|RequirePackage|documentclass|bibliography|addbibresource"
    r"|bibliographystyle|[A-Za-z]*[Cc]ite[A-Za-z]*|vspace|hspace|setlength|addtolength"
    r"|setcounter|setstretch|linespread|renewcommand|rule|resizebox|scalebox)(?![A-Za-z])\*?"
    r"(?:\s*\[[^\]]*\])*(?:\s*\{[^{}]*\})*"
)
# Lengths: a unit glued to the number, or a number before a \...width macro.
_LENGTH = re.compile(
    r"[-+]?\d*\.?\d+(?:pt|em|ex|cm|mm|in|bp|pc|sp|mu)\b"
    r"|[-+]?\d*\.?\d+\s*\\(?:text|line|column|paper)(?:width|height)"
    r"|[-+]?\d*\.?\d+\s*\\(?:baselineskip|hsize|vsize)"
)


def _blank(match: re.Match) -> str:
    """Same-shape whitespace: every newline kept, so line numbers stay true."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _inline_math(match: re.Match) -> str:
    """A number typeset alone in math (`$0.053$`, `$12.2\\%$`, `$0.91 \\pm 0.02$`)
    is a result and is kept; math with a symbol in it (`$\\alpha = 0.5$`) is a
    parameter or an equation and is blanked."""
    body = match.group("d") if match.group("d") is not None else match.group("p")
    bare = _MATH_NOISE.sub("", body.replace("{,}", ",").replace("\\pm", " "))
    if bare and _PURE_NUMBER.fullmatch(bare):
        return " " + body + " "
    return _blank(match)


def masked_prose(text: str) -> str:
    """The text a reader sees as numbers, with structural ones blanked.

    Newlines are kept, so a match's line number in the result is its line in
    the source; offsets within a line are not (siunitx calls are unwrapped
    to their number, `{,}` becomes `,`).
    """
    text = strip_comments(text)
    text = _DISPLAY_RE.sub(_blank, text)
    text = _DISPLAY_DELIMS.sub(_blank, text)
    text = _INLINE_MATH.sub(_inline_math, text)
    text = _SIUNITX.sub(lambda m: " " + m.group(1) + " ", text)
    text = _UNIT_ONLY.sub(_blank, text)
    text = _STRUCTURAL.sub(_blank, text)
    text = _LENGTH.sub(_blank, text)
    return text.replace("{,}", ",").replace("{.}", ".")


# --------------------------------------------------------------------------- citations


def _braced(text: str, pos: int) -> tuple[str, int] | None:
    """The balanced `{...}` group starting at `pos`, and the index after it."""
    if pos >= len(text) or text[pos] != "{":
        return None
    depth = 0
    for i in range(pos, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[pos + 1 : i], i + 1
    return None


def _skip_optional(text: str, pos: int) -> int:
    """Past spaces and `[...]` groups, so `\\citep[see][p.~3]{key}` lands on `{key}`."""
    while True:
        while pos < len(text) and text[pos] in " \t\n":
            pos += 1
        if pos < len(text) and text[pos] == "[":
            close = text.find("]", pos)
            if close < 0:
                return pos
            pos = close + 1
            continue
        return pos


def _keys_after(text: str, name: str, pos: int) -> Iterator[str]:
    """The keys of one cite command whose name ends at `pos`."""
    # \cites{a}{b}[..]{c} takes several groups; every other cite command one.
    groups = 20 if name.lower().endswith("cites") else 1
    for _ in range(groups):
        got = _braced(text, _skip_optional(text, pos))
        if got is None:
            return
        body, pos = got
        for key in body.split(","):
            key = key.strip()
            if key and key != "*":
                yield key


def cited_keys(text: str) -> Iterator[tuple[str, int]]:
    """(key, line) for every citation key in one file, comments ignored."""
    text = strip_comments(text)
    for match in _CITE_RE.finditer(text):
        name = match.group(1)
        if name in NOT_CITATIONS:
            continue
        line = text.count("\n", 0, match.start()) + 1
        for key in _keys_after(text, name, match.end()):
            yield key, line


def bib_keys(path: Path) -> list[str]:
    """The entry keys one `.bib` defines (title or not: BibTeX reads them all)."""
    try:
        text = _read(path)
    except OSError:
        return []
    return [
        m.group(2) for m in _BIB_ENTRY_RE.finditer(text) if m.group(1).lower() not in _NON_ENTRIES
    ]


def _resolve_resource(name: str, base: Path, candidates: list[Path]) -> Path | None:
    """A `\\bibliography`/`\\addbibresource` name as a file: beside the paper
    first, then any configured `.bib` with that name or stem."""
    for path in (base / name, base / f"{name}.bib"):
        if path.is_file():
            return path
    short = Path(name).name
    return next((c for c in candidates if short in (c.name, c.stem)), None)


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: dict[Path, Path] = {}
    for p in paths:
        if p.is_file():
            seen.setdefault(p.resolve(), p)
    return list(seen.values())


def _named_resources(files: list[Path]) -> list[str]:
    names: list[str] = []
    for path in files:
        for group in _BIB_RESOURCE_RE.findall(strip_comments(_read(path))):
            names.extend(n.strip() for n in group.split(",") if n.strip())
    return names


def _defined(paths: list[Path]) -> dict[str, str]:
    """key -> name of the first file that defines it."""
    out: dict[str, str] = {}
    for path in paths:
        for key in bib_keys(path):
            out.setdefault(key, path.name)
    return out


def _uses(paper: Path, files: list[Path]) -> dict[str, str]:
    """key -> where it is first cited, as `file:line` relative to the paper's folder."""
    out: dict[str, str] = {}
    for path in files:
        try:
            shown = path.relative_to(paper.parent)
        except ValueError:
            shown = path
        for key, line in cited_keys(_read(path)):
            out.setdefault(key, f"{shown}:{line}")
    return out


Lookup = Callable[[str], object | None]


def _elsewhere(key: str, defined: dict[str, str], lookup: Lookup | None) -> str | None:
    """Where an unusable key IS defined: another .bib file, or another source."""
    if key in defined:
        return defined[key]
    found = lookup(key) if lookup is not None else None
    if found is None:
        return None
    # A store or Zotero record, not a file: name the source it came from.
    return getattr(found, "source", "") or "library"


def _bibliography(paper: Path, files: list[Path], extra_bibs: Iterable[Path]):
    """(candidates, named, loaded, unresolved) for one paper's bibliography."""
    candidates = _dedupe([*sorted(paper.parent.glob("*.bib")), *extra_bibs])
    named = _named_resources(files)
    found = {n: _resolve_resource(n, paper.parent, candidates) for n in named}
    loaded = _dedupe(p for p in found.values() if p is not None)
    unresolved = [n for n, p in found.items() if p is None]
    return candidates, bool(named), loaded, unresolved


def _missing(key: str, where: str, usable: set[str], defined: dict[str, str], lookup, named):
    """The finding for one cited key, or None when it resolves."""
    if key in usable:
        return None
    if not named and lookup is not None and lookup(key) is not None:
        return None
    entry: dict = {
        "key": key,
        "where": where,
        "closest": difflib.get_close_matches(key, sorted(usable), n=3, cutoff=0.6),
    }
    other = _elsewhere(key, defined, lookup) if named else None
    if other:
        entry["defined_in_unloaded"] = other
    return entry


def check_paper(paper: Path, *, extra_bibs: Iterable[Path] = (), lookup: Lookup | None = None):
    """One paper's citation keys against its bibliography.

    Candidate `.bib` files are those beside the root file plus `extra_bibs`
    (the configured ones: ATTEST_BIB_PATHS and the generated owner .bib).
    With no `\\bibliography`/`\\addbibresource`, any candidate or `lookup`
    (the library store, Zotero) resolves a key. With one, only the named
    files do, and a key found elsewhere is reported `defined_in_unloaded`.
    """
    paper = Path(paper).resolve()
    files = tree(paper)
    uses = _uses(paper, files)
    candidates, named, loaded, unresolved = _bibliography(paper, files, extra_bibs)
    defined = _defined(candidates + loaded)
    usable = set(_defined(loaded)) if named else set(defined)
    found = (_missing(k, w, usable, defined, lookup, named) for k, w in uses.items())
    return {
        "paper": str(paper),
        "files": [str(f) for f in files],
        "citation_keys_used": len(uses),
        "bib_files": [p.name for p in (loaded if named else candidates)],
        "missing_keys": [m for m in found if m is not None],
        "unresolved_bib_resources": unresolved,
    }


def _merge(reports: list[dict]) -> dict:
    missing = [{**m, "paper": r["paper"]} for r in reports for m in r["missing_keys"]]
    return {
        "papers": [r["paper"] for r in reports],
        "citation_keys_used": sum(r["citation_keys_used"] for r in reports),
        "bib_files": sorted({b for r in reports for b in r["bib_files"]}),
        "missing_keys": missing,
        "unresolved_bib_resources": [n for r in reports for n in r["unresolved_bib_resources"]],
    }


def check_citations(
    root: Path, *, extra_bibs: Iterable[Path] | None = None, lookup: Lookup | None = None
) -> dict:
    """`check_paper` for every paper at `root`, merged; empty for a Markdown target.

    `extra_bibs` defaults to `configured_bibs()`, read only when there is a
    paper to check.
    """
    papers = paper_roots(root)
    if extra_bibs is None:
        extra_bibs = configured_bibs() if papers else []
    extra = list(extra_bibs)
    return _merge([check_paper(p, extra_bibs=extra, lookup=lookup) for p in papers])


def configured_bibs() -> list[Path]:
    """The configured `.bib` files: ATTEST_BIB_PATHS (else the cwd's), plus
    the owner bibliography Attestation writes into ATTEST_BIB_OUT.

    Imported lazily: the claim checker imports this module, and must not
    depend on the bibliographic readers at import time.
    """
    from attestation import bibliography, citations

    out = list(citations.bib_paths_from_env())
    owner = bibliography.output_dir()
    if owner is not None and owner.is_dir():
        out.extend(sorted(owner.glob("*.bib")))
    return out


def resolver_bib_paths(target: Path) -> list[Path] | None:
    """`.bib` files for a `Resolver` checking a LaTeX target's `cite=` claims:
    the configured ones plus every `.bib` beside a paper. None for a target
    with no `.tex` paper, so a Markdown check resolves exactly as before."""
    papers = paper_roots(target)
    if not papers:
        return None
    beside = [b for p in papers for b in sorted(p.parent.glob("*.bib"))]
    return _dedupe([*configured_bibs(), *beside])
