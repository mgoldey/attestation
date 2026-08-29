"""Docstring ratchet: every public def under src/attestation gets a docstring.

Mirrors `scripts/check_complexity.py`'s shape: a baseline pinned here, only
ever lowered, never raised without a reason in the commit message. Measured
2026-08-29 against `main` at e5e511b: 190 of 292 public defs had a docstring
(103 missing, concentrated in `cli.py`'s `cmd_*` handlers -- whose one-line
purpose already existed as argparse `help=`, see `test_cli.py`'s
`test_every_cmd_docstring_is_its_helps_first_line` -- and in `citations.py`'s
reader `all`/`lookup` methods). This task brought the count to 0; BASELINE
stays 0 so a new undocumented public def fails the suite immediately, naming
its `file:line` rather than waiting for someone to notice coverage drifted.

"Public" here means a module, class, or function whose name does not start
with `_`, at ANY nesting depth -- a decorator's inner `wrapper`, or a closure
like `cli.py`'s `add_db`, is still something a reader can end up looking at
(via `__doc__`, a traceback, or just reading the file) and still counts.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "attestation"

BASELINE = 0


def _undocumented(path: Path) -> list[str]:
    """`file:line kind name` for every undocumented public def in `path`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(SRC.parent.parent)
    out: list[str] = []

    if ast.get_docstring(tree) is None:
        out.append(f"{rel}:1 module {path.stem}")

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if not child.name.startswith("_"):
                    if ast.get_docstring(child) is None:
                        kind = "class" if isinstance(child, ast.ClassDef) else "function"
                        out.append(f"{rel}:{child.lineno} {kind} {prefix}{child.name}")
                is_class = isinstance(child, ast.ClassDef)
                visit(child, f"{prefix}{child.name}." if is_class else prefix)
            else:
                visit(child, prefix)

    visit(tree)
    return out


def test_every_public_def_has_a_docstring():
    missing: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        missing.extend(_undocumented(path))

    assert len(missing) <= BASELINE, (
        f"{len(missing)} undocumented public def(s), baseline is {BASELINE}:\n" + "\n".join(missing)
    )
