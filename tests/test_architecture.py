"""The onion's rules, enforced mechanically.

A layering doc nobody re-reads cannot stop the next contributor -- human or
agent -- from reaching through a boundary. These tests can. Each one fails
loudly the first time a rule is broken, and names the rule in its message.

One rule that used to be here is gone: sqlite3 confined to an infrastructure
package. That guarded a repository layer the onion spec proposed and two
reviews then talked us out of -- see `2026-08-21-onion-refactor-design.md`,
superseded. A test enforcing a boundary that does not exist is worse than no
test: it passes forever and reads like coverage.

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

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "attestation"


def _modules():
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(SRC))


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


def test_mcp_domain_modules_stay_small():
    """The split's actual goal: no module big enough to lose an agent in.

    mcp_server.py was 1454 lines holding 34 tools written twice each. Line
    count is a weak proxy for quality in general -- ledger.py is 637 lines of
    one coherent argument and should stay that way -- but for the tool surface
    specifically it tracks the thing that went wrong, which was ritual repeated
    per tool rather than depth.
    """
    # knowledge.py went 150 -> 175 on 2026-08-21 for kg_concepts, a fifth tool.
    # The cap is meant to catch ritual repeated per tool, not to price a module
    # out of gaining a genuinely new one -- so a real tool buys headroom, while
    # the limit still binds against boilerplate creeping back in.
    limits = {"feed.py": 720, "provenance.py": 300, "knowledge.py": 175, "symbolic.py": 150}
    mcp_dir = SRC / "mcp"
    oversized = [
        f"{p.name}={len(p.read_text().splitlines())} (max {limits[p.name]})"
        for p in mcp_dir.glob("*.py")
        if p.name in limits and len(p.read_text().splitlines()) > limits[p.name]
    ]
    assert not oversized, "mcp domain modules grew: " + ", ".join(oversized)


def test_every_tool_body_is_reachable_without_fastmcp():
    """Tools must be callable directly, or they can only be tested through a server.

    symbolic.py briefly defined all seven of its tools as closures inside
    register(), which made them unreachable -- the seven `_sym_*_impl` tests
    could not import anything to call. Each domain keeps its implementations at
    module level and registers thin wrappers over them.
    """
    from attestation.mcp import feed, knowledge, provenance, symbolic

    for mod, names in (
        (feed, ["_list_feed", "_digest_body", "_search_feed"]),
        (knowledge, ["_neighbors", "_path", "_central", "_communities"]),
        (provenance, ["_scan", "_list", "_compare", "_detail"]),
        (symbolic, ["_sym_simplify", "_sym_solve", "_sym_verify"]),
    ):
        for name in names:
            assert callable(getattr(mod, name, None)), f"{mod.__name__}.{name} not reachable"
