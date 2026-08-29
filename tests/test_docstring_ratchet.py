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
MODULE-LEVEL def only -- if its decorator list names `_documented`
specifically and its runtime `__doc__` is non-empty. That arm exists for
exactly one case: `cli.py`'s `cmd_*` handlers get `__doc__` from
`@_documented(name)`, which reads the argparse `help=` text out of `HELP`
at import time rather than repeating it as a second literal string (see
cli.py's own docstring on `HELP` and `_documented` for why). It is
deliberately NOT extended to a nested function/closure (`add_db`,
`wrapper`, ...): those cannot carry a decorator in the first place, so a
missing literal docstring there is always a real finding.

Narrowed to the named decorator (not "any decorator") 2026-08-29: `@dataclass`
also sets a runtime `__doc__` (it synthesizes `Foo(a: int)` as the class's
docstring), so the original "any decorator + non-empty `__doc__`" fallback
let an undocumented public `@dataclass` pass silently. Only `@_documented`
is a real docstring-setting mechanism this repo has written; nothing else
gets the benefit of the doubt.
"""

import ast
import importlib
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "attestation"

BASELINE = 0

# The only decorator this ratchet trusts to set __doc__ at import time
# without a literal docstring in the body. See cli.py's HELP/_documented.
DOC_SETTING_DECORATOR = "_documented"


def _module_name(path: Path) -> str:
    """`src/attestation/mcp/_tool.py` -> `attestation.mcp._tool`;
    `.../__init__.py` -> the package name itself, dropping the filename.

    Relative to `SRC.parent` (the `src/` directory), not `SRC.parent.parent`
    (the repo root) -- the latter produced `src.attestation.cli`, an import
    path nothing installs under, which resolved by accident under one
    pytest invocation (rootdir on sys.path put a `src` namespace package
    there too) and raised ModuleNotFoundError under another (the package
    installed normally, with no bare `src` on sys.path at all).
    """
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.parts[-1] == "__init__" else rel.parts
    return ".".join(parts)


def _decorator_names(child: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The base name of each decorator on `child` -- `@foo`, `@foo(...)`,
    `@mod.foo`, and `@mod.foo(...)` all resolve to `"foo"`. Only the base
    name is ever compared against `DOC_SETTING_DECORATOR`, so an aliased or
    namespaced spelling of an unrelated decorator can't collide with it."""
    names: set[str] = set()
    for dec in child.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _load_module_for_import_check(path: Path):
    """The module object `_undocumented` consults for a `@_documented`-style
    runtime `__doc__`. Under `src/attestation`, this is the installed
    package module (so it reflects what a real import sees); for an
    arbitrary path outside the package (a temp module in a regression test),
    it is loaded directly from the file, since `_module_name` only resolves
    paths relative to `SRC.parent`."""
    try:
        path.relative_to(SRC.parent)
    except ValueError:
        spec = importlib.util.spec_from_file_location(f"_ratchet_check_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(_module_name(path))


def _undocumented(path: Path) -> list[str]:
    """`file:line kind name` for every undocumented public def in `path`.

    Reads and parses `path` itself, so it is callable on any `.py` file --
    not only one under `SRC` -- which is what lets a regression test point
    it at a temp module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    try:
        rel = path.relative_to(SRC.parent.parent)
    except ValueError:
        rel = path
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
                    if (
                        not documented
                        and attr_path is not None
                        and DOC_SETTING_DECORATOR in _decorator_names(child)
                    ):
                        # @_documented can set __doc__ at import time (see
                        # cli.py's HELP/_documented) without a literal string
                        # in the body -- checkable only for a def reachable
                        # as a module/class attribute. No other decorator
                        # gets this benefit of the doubt (see module
                        # docstring: @dataclass also sets a runtime __doc__).
                        if module is None:
                            module = _load_module_for_import_check(path)
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


def test_undocumented_dataclass_is_reported_despite_synthesized_doc(tmp_path):
    """`@dataclass` sets a runtime `__doc__` (`Foo(a: int)`) with no literal
    docstring anywhere -- the exact shape the pre-2026-08-29 fallback let
    through, since it trusted "any decorator + non-empty `__doc__`" rather
    than checking which decorator. This must still be a finding."""
    module = tmp_path / "undocumented_dataclass_module.py"
    module.write_text(
        '"""A module that states itself but not its dataclass."""\n'
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Foo:\n"
        "    a: int\n"
    )

    missing = _undocumented(module)

    assert any("class Foo" in line for line in missing), missing


def test_documented_decorator_without_literal_docstring_is_accepted(tmp_path):
    """The narrowed fallback still accepts the one real case it exists for:
    a module-level def whose decorator list names `_documented` and whose
    runtime `__doc__` is non-empty, even with no literal docstring in the
    body."""
    module = tmp_path / "documented_via_decorator_module.py"
    module.write_text(
        '"""A module that states itself and documents a def via decorator."""\n'
        "\n"
        "\n"
        "def _documented(func):\n"
        '    """Set on the module for the test to find, not a real decorator\n'
        '    factory -- just enough to exercise the ratchet\'s fallback."""\n'
        '    func.__doc__ = "set at import time, not as a literal string"\n'
        "    return func\n"
        "\n"
        "\n"
        "@_documented\n"
        "def documented_thing():\n"
        "    return 1\n"
    )

    missing = _undocumented(module)

    assert not any("documented_thing" in line for line in missing), missing
