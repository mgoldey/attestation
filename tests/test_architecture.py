"""The onion's rules, enforced mechanically.

A layering doc nobody re-reads cannot stop the next contributor -- human or
agent -- from reaching through a boundary. These tests can. Each one fails
loudly the first time a rule is broken, and names the rule in its message.

The rules here are deliberately narrow. An earlier draft of the design spec
proposed banning deferred imports inside function bodies, on the theory that
they hide import cycles. Measurement refuted it: 29 of the 30 are lazy loads
that keep `attest --help` at 0.22s, because importing sklearn alone costs
929ms. Banning them would have enforced a one-second regression on every CLI
invocation. `test_import_graph_is_acyclic` asserts the property that draft was
reaching for, and `test_cli_help_stays_fast` protects the optimization it would
have destroyed.
"""

import ast
import pathlib
import subprocess
import sys
import time
from collections import defaultdict

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "attestation"

# Layers that may import sqlite3. Until stage 2 lands, db.py is the only
# infrastructure that exists; the tuple grows as modules move, and shrinks to
# just `infrastructure/` when stage 3 finishes.
SQLITE_ALLOWED = {
    "db.py",
    # Stage 3 migrates these off sqlite3 one at a time. Delete each entry in
    # the PR that migrates it -- this list is the stage-3 burndown, and an
    # empty set is the stage's definition of done.
    "claims.py",
    "explain.py",
    "features.py",
    "feeds.py",
    "ingest.py",
    "kg.py",
    "ledger.py",
    "rank.py",
    "server.py",
}


def _modules():
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(SRC))


def _imports_sqlite3(path: pathlib.Path) -> int | None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == "sqlite3" for a in node.names):
            return node.lineno
        if isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
            return node.lineno
    return None


def test_sqlite3_confined_to_allowed_modules():
    """The onion's one mechanical rule: only infrastructure talks to the store.

    Domain logic names a protocol, never a sqlite3.Connection. When this fails
    for a module not on the burndown list, the fix is to route through that
    module's repository -- not to add it to SQLITE_ALLOWED.
    """
    offenders = [
        f"{_rel(p)}:{ln}" for p in _modules() if (ln := _imports_sqlite3(p)) and _rel(p) not in SQLITE_ALLOWED
    ]
    assert not offenders, (
        "these modules import sqlite3 but are not infrastructure: "
        + ", ".join(offenders)
        + " -- route the queries through a repository instead"
    )


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    return ".".join(rel.parts)


def _top_level_deps(path: pathlib.Path) -> set[str]:
    """Intra-package imports at module scope. Function-body imports excluded:
    a lazy import cannot cause an import-time cycle, which is the whole reason
    the one real cycle in this tree is survivable today."""
    tree = ast.parse(path.read_text())
    in_function = {
        id(n)
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for n in ast.walk(fn)
    }
    deps: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in in_function:
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("attestation"):
            tail = node.module[len("attestation") :].lstrip(".")
            deps |= {tail.split(".")[0]} if tail else {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            deps |= {
                a.name[len("attestation.") :].split(".")[0]
                for a in node.names
                if a.name.startswith("attestation.")
            }
    return deps


def test_import_graph_is_acyclic():
    """No import cycles at module scope.

    This is what the deferred-import ban was actually reaching for. It catches
    a real cycle without touching the deliberate lazy loads, because a cycle
    among top-level imports is the thing that breaks at import time.

    symbolic/symbolic_ops is NOT flagged, correctly: only symbolic_ops imports
    symbolic at module scope, while symbolic defers its half inside _worker().
    That is a lazy cycle, which never breaks at import time. Verified 2026-08-21
    by injecting embed -> features -> embed, which this test caught by path.
    """
    graph = defaultdict(set)
    for path in _modules():
        me = _module_name(path)
        graph[me] = {d for d in _top_level_deps(path) if d != me}

    cycles: list[str] = []
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = defaultdict(int)

    def visit(node: str, stack: list[str]) -> None:
        colour[node] = GREY
        for dep in sorted(graph.get(node, ())):
            if colour[dep] == GREY:
                cycles.append(" -> ".join(stack[stack.index(dep) :] + [dep]))
            elif colour[dep] == WHITE:
                visit(dep, stack + [dep])
        colour[node] = BLACK

    for node in sorted(graph):
        if colour[node] == WHITE:
            visit(node, [node])

    assert not cycles, "import cycles at module scope: " + "; ".join(cycles)


def test_cli_help_stays_fast():
    """`attest --help` must not regress past 0.5s (measured 0.22s, 2026-08-21).

    This test exists to make a non-obvious optimization visible. The lazy
    imports in cli.py look like untidiness and invite cleanup; sklearn alone
    costs 929ms to import, so promoting one to module scope silently puts a
    second onto every invocation. If this fails, someone promoted an import --
    move it back into the function body rather than raising the budget.
    """
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", "import attestation.cli; attestation.cli.main()", "--help"],
        capture_output=True,
        timeout=30,
    )
    elapsed = time.perf_counter() - start
    assert proc.returncode == 0, proc.stderr.decode()[:2000]
    assert elapsed < 0.5, (
        f"attest --help took {elapsed:.2f}s (budget 0.5s). A lazy import in "
        "cli.py was probably promoted to module scope; move it back."
    )
