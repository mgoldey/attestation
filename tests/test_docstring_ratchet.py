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

A def is documented if the AST shows a literal docstring, OR -- for a
MODULE-LEVEL def only -- if it carries a decorator and its runtime
`__doc__` is non-empty. The second arm exists for exactly one case:
`cli.py`'s `cmd_*` handlers get `__doc__` from `@_documented(name)`, which
reads the argparse `help=` text out of `HELP` at import time rather than
repeating it as a second literal string (see cli.py's own docstring on
`HELP` and `_documented` for why). It is deliberately NOT extended to a
nested function/closure (`add_db`, `wrapper`, ...): those cannot carry a
decorator in the first place, so a missing literal docstring there is
never explained by this mechanism and must stay a real finding.
"""

import ast
import importlib
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "attestation"

BASELINE = 0


def _module_name(path: Path) -> str:
    """`src/attestation/mcp/_tool.py` -> `attestation.mcp._tool`;
    `.../__init__.py` -> the package name itself, dropping the filename."""
    rel = path.relative_to(SRC.parent.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.parts[-1] == "__init__" else rel.parts
    return ".".join(parts)


def _undocumented(path: Path) -> list[str]:
    """`file:line kind name` for every undocumented public def in `path`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(SRC.parent.parent)
    out: list[str] = []

    if ast.get_docstring(tree) is None:
        out.append(f"{rel}:1 module {path.stem}")
        return out  # a module that fails to state itself is reported once,
        # not once per def inside it.

    module = None  # imported lazily, only if a decorated reachable def needs it

    def visit(node: ast.AST, prefix: str, attr_path: list[str] | None) -> None:
        """`attr_path` is the module-attribute chain reaching `node`'s own
        children (e.g. [] at module scope, ["Resolver"] inside a class), or
        None once we are inside a plain function -- a closure local to it
        (like cli.py's `add_db`) is never reachable as an attribute, so no
        decorator could set its `__doc__` from outside, and a missing
        literal docstring there is always a real finding."""
        nonlocal module
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if not child.name.startswith("_"):
                    documented = ast.get_docstring(child) is not None
                    if not documented and attr_path is not None and child.decorator_list:
                        # A decorator can set __doc__ at import time (see
                        # cli.py's `_documented`) without a literal string
                        # in the body -- checkable only for a def reachable
                        # as a module/class attribute.
                        if module is None:
                            module = importlib.import_module(_module_name(path))
                        obj = module
                        for part in [*attr_path, child.name]:
                            obj = getattr(obj, part, None)
                            if obj is None:
                                break
                        documented = bool(obj is not None and getattr(obj, "__doc__", None))
                    if not documented:
                        kind = "class" if isinstance(child, ast.ClassDef) else "function"
                        out.append(f"{rel}:{child.lineno} {kind} {prefix}{child.name}")
                if isinstance(child, ast.ClassDef):
                    next_path = [*attr_path, child.name] if attr_path is not None else None
                    visit(child, f"{prefix}{child.name}.", next_path)
                else:
                    # Descending into a function body: nothing inside it is
                    # attribute-reachable from here on.
                    visit(child, prefix, None)
            else:
                visit(child, prefix, attr_path)

    visit(tree, "", [])
    return out


def test_every_public_def_has_a_docstring():
    missing: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        missing.extend(_undocumented(path))

    assert len(missing) <= BASELINE, (
        f"{len(missing)} undocumented public def(s), baseline is {BASELINE}:\n" + "\n".join(missing)
    )
