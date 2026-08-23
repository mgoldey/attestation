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
    # Best of three. A single timing under a loaded machine is a coin flip --
    # this test failed during a parallel review run while `attest --help`
    # measured 0.15-0.27s on an idle box. Taking the minimum measures the
    # import cost, which is what the budget is about, rather than whatever
    # else the CPU was doing.
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", "import attestation.cli; attestation.cli.main()", "--help"],
            capture_output=True,
            timeout=30,
        )
        timings.append(time.perf_counter() - start)
        assert proc.returncode == 0, proc.stderr.decode()[:2000]

    elapsed = min(timings)
    assert elapsed < 0.5, (
        f"attest --help took {elapsed:.2f}s at best of {len(timings)} (budget 0.5s). "
        "A lazy import in cli.py was probably promoted to module scope; move it back."
    )


def test_mcp_domain_modules_stay_small():
    """The split's actual goal: no module big enough to lose an agent in.

    Counts CODE lines, not total lines. 28% of feed.py is docstrings, and in
    an MCP module the docstrings ARE the product -- they are what a calling
    agent reads to choose a tool, and two reviews said the surface needed more
    of that guidance, not less. A raw line cap taxes the fix.

    Line count is a weak proxy generally: ledger.py is 637 lines of one
    coherent argument and should stay that way. For the tool surface it tracks
    the thing that went wrong, which was ritual repeated per tool.
    """
    # Measured plus ~40 lines of headroom, so a real new tool fits and a slow
    # accretion of ritual does not.
    #
    # feed.py carries 19 of the 37 tools -- ranking, search, personas, feeds,
    # explanations and feedback -- because the split was by domain and "feed"
    # is one domain holding six concerns. It has now hit this cap three times
    # in a day. The next tool that lands here should come with a split
    # (personas and subscriptions are the obvious seams) rather than another
    # raised number; the cap exists to force that conversation, not to be
    # edited past.
    # Raised once for schema constraints -- Annotated[...] on every bounded
    # argument is annotation, not behaviour, and it buys a client-side reject
    # instead of a failed call. The seam note below still stands for the next
    # genuine tool.
    limits = {
        # feed.py shed the five subscription tools to subscriptions.py -- the
        # seam the code drew itself, since every one of them imported
        # attestation.feeds and nothing else in the domain did.
        "feed.py": 710,
        "subscriptions.py": 150,
        "knowledge.py": 140,
        "symbolic.py": 130,
        # ask.py was 540 code lines and entirely unguarded until the default
        # below existed. Split at the seam it already had: routing.py holds the
        # rule tables (pure functions over a string, no model, no database --
        # the regression guard for the measured 13/15), ask.py holds the four
        # tools that call them and touch the rest of the system.
        "ask.py": 315,
        "routing.py": 265,
        "provenance.py": 250,
    }
    # Anything not named above still gets a cap. `if name not in limits:
    # continue` meant a module was exempt until someone remembered to enrol it
    # -- so a NEW domain module started life unguarded, and ask.py reached 662
    # code lines (the second-largest in mcp/, serving the four routers CLAUDE.md
    # calls the primary entry points) without this test ever looking at it.
    # Verified: appending ~1200 junk lines to ask.py left this test green.
    #
    # Same asymmetry as the .env.sample allowlist: enrolling a file that later
    # shrinks is harmless, failing to enrol one that grows is the costly
    # direction, and only that one was unguarded.
    default_limit = 250

    oversized = []
    for path in (SRC / "mcp").glob("*.py"):
        limit = limits.get(path.name, default_limit)
        tree = ast.parse(path.read_text())
        doc_lines = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                doc = ast.get_docstring(node)
                if doc:
                    doc_lines += len(doc.splitlines()) + 2
        code = len(path.read_text().splitlines()) - doc_lines
        if code > limit:
            oversized.append(f"{path.name}={code} code lines (max {limit})")
    assert not oversized, "mcp domain modules grew: " + ", ".join(oversized)


# Modules whose tools predate mcp/_tool.py's @tool decorator, so their bodies
# carry no __wrapped__ marker and must be counted plainly. Named, not detected:
# an empty set here is how a new module silently gained the exemption.
PRE_DECORATOR_MODULES = {"attestation.mcp.symbolic", "attestation.mcp.ask"}

# Helpers inside those modules, so their bodies can be counted exactly rather
# than with slack. symbolic.py had 8 underscore callables for 7 tool bodies --
# `_call` being the eighth -- which gave one free slot: a tool could become a
# closure and the count still matched. Naming the helpers is a short, stable
# list; guessing which names are bodies is not, since bodies are called
# `_list_feed` for feed.list and `_digest_body` for feed.digest.
PRE_DECORATOR_HELPERS = {
    "attestation.mcp.symbolic": {"_call"},
    # ask.py is exempted from the exact count entirely rather than listed:
    # its four `.tools` tools all delegate to ONE body (_tools_listing), so it
    # legitimately serves 8 tools from 5 bodies and any count-based rule is
    # wrong for it. The four `.ask` bodies are reachable and asserted by name
    # below instead.
}

# Modules where tools legitimately share a body, so counting cannot apply.
SHARED_BODY_MODULES = {"attestation.mcp.ask"}


def test_every_tool_body_is_reachable_without_fastmcp():
    """Tools must be callable directly, or they can only be tested through a server.

    symbolic.py briefly defined all seven of its tools as closures inside
    register(), which made them unreachable. An earlier version of this test
    hardcoded 14 names and so covered 14 of 37 impls -- renaming _sym_integrate
    to _HIDDEN_sym_integrate left it green, which is the regression it names.

    The list now comes from the registry, so a tool cannot be added without
    being covered.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server
    from attestation.mcp import DOMAINS

    served = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert served, "no tools registered"

    # Every tool's body must be a module-level callable in the module that
    # serves it. Counted PER MODULE, not in total: the global count carries 16
    # slack (66 module-level underscore callables against 50 tools, the surplus
    # being helpers like _has/_label/_item_row), so up to 16 real tool bodies
    # could become closures before the total even dipped. Demonstrated: making
    # all five kg.* bodies unreachable left the total check green.
    #
    # Names are not matched to tools -- that would re-encode the naming
    # convention rather than check anything. Per-module counting needs no
    # convention and localises the failure to the module that broke.
    served_by_module: dict[str, list[str]] = {}
    impls_by_module = {
        module.__name__: {
            attr
            for attr in dir(module)
            if attr.startswith("_")
            and not attr.startswith("__")
            and callable(getattr(module, attr))
        }
        for module in DOMAINS
    }
    # Attribute each served tool to the module whose register() created it, by
    # re-registering against a recording stub. That is the only mapping that
    # does not assume a name shape.
    for module in DOMAINS:
        recorded: list[str] = []

        class _Recorder:
            @staticmethod
            def tool(*a, name=None, **kw):
                def deco(fn):
                    recorded.append(name or fn.__name__)
                    return fn

                return deco

        try:
            module.register(_Recorder())
        except Exception:  # noqa: BLE001 -- a module that cannot register
            # against a stub is a separate failure, caught by the tool-listing
            # tests; here it simply contributes no expectations.
            continue
        served_by_module[module.__name__] = recorded

    # Count only DECORATED bodies. Plain counting -- even per module -- lets a
    # helper mask a tool body one for one: feed.py has 8 helpers against 8
    # slack, so eight of its bodies could become closures undetected.
    #
    # Name-matching was tried and rejected: bodies are named `_list_feed` for
    # `feed.list`, `_digest_body` for `feed.digest`, and so on, so matching by
    # name would encode a convention the code does not actually follow.
    # @tool sets __wrapped__, which identifies a body exactly and assumes
    # nothing about what it is called. symbolic.py predates the decorator and
    # is counted the old way.
    unreachable = []
    for module in DOMAINS:
        names = served_by_module.get(module.__name__, [])
        if not names:
            continue
        attrs = [getattr(module, a) for a in impls_by_module.get(module.__name__, set())]
        bodies = [a for a in attrs if hasattr(a, "__wrapped__")]
        # No fallback for "this module has no decorated bodies" -- that was
        # tried and it IS the hole: a probe module with two closure-bodied
        # tools and two helpers passed, because zero decorated bodies looked
        # like a pre-decorator module. Pre-decorator modules are named
        # explicitly instead, so a NEW module that forgets @tool fails rather
        # than inheriting the exemption.
        if module.__name__ in SHARED_BODY_MODULES:
            # Named check instead of a count: these four must stay reachable.
            missing = [
                n
                for n in ("_feed_ask", "_runs_ask", "_kg_ask", "_sym_ask", "_tools_listing")
                if not callable(getattr(module, n, None))
            ]
            if missing:
                unreachable.append(f"{module.__name__}: {missing} not reachable")
            continue
        if module.__name__ in PRE_DECORATOR_MODULES:
            helpers = PRE_DECORATOR_HELPERS.get(module.__name__, set())
            found = len(
                [a for a in impls_by_module.get(module.__name__, set()) if a not in helpers]
            )
        else:
            found = len(bodies)
        if found < len(names):
            unreachable.append(
                f"{module.__name__}: {found} reachable bodies for {len(names)} tools"
            )

    assert not unreachable, (
        "these tools have no module-level implementation to call directly: "
        + ", ".join(unreachable)
    )


def test_every_tool_is_namespaced():
    """The change the tool-surface spec named as its primary goal.

    36 tools in one flat namespace presented `runs_compare`, `kg_path`,
    `sym_integrate` and `digest` to a calling agent as peers with equal claim
    on any question. Splitting the file did nothing about that -- a calling
    agent cannot see which file a tool lives in. The prefix is the part it can
    see: a 37-way choice becomes a 4-way choice and then a smaller one.

    A tool with no namespace is almost always a new one that skipped
    `@mcp.tool(name=...)`, which is exactly when this should fail.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server

    names = sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools()))
    assert names, "no tools registered"

    flat = [n for n in names if "." not in n]
    assert not flat, f"tools with no namespace: {flat}"

    namespaces = {n.split(".", 1)[0] for n in names}
    assert namespaces == {"cite", "feed", "kg", "runs", "sym"}, (
        f"unexpected namespaces: {namespaces}"
    )


def test_no_tool_repeats_its_own_namespace():
    """`kg.kg_path` and `feed.list_feed` carry the domain twice.

    The namespace already says it; repeating it is the noise the rename was
    meant to remove, and it creeps back one tool at a time.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server

    redundant = []
    for name in sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools())):
        namespace, _, leaf = name.partition(".")
        singular = namespace.rstrip("s")
        if leaf.startswith(f"{namespace}_") or leaf.endswith(f"_{singular}"):
            redundant.append(name)
    assert not redundant, f"these repeat their namespace: {redundant}"


def test_no_message_or_docstring_names_a_tool_that_does_not_exist():
    """A recovery message is only useful if the tool it names is callable.

    After namespacing, two error messages still said "call runs_scan(confirm=
    true)" and "call kg_concepts()" -- an agent following either would call a
    name the server no longer serves, turning a helpful message into a dead
    end. Docstrings had the same drift.
    """
    import asyncio
    import os
    import re
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server

    served = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    leaves = {n.split(".", 1)[1] for n in served}
    # The pre-namespacing spellings: a leaf name prefixed by its own domain
    # with an underscore, e.g. runs_scan for runs.scan.
    stale = {f"{n.split('.')[0]}_{n.split('.', 1)[1]}" for n in served} | {
        "list_feed",
        "list_feeds",
        "record_feedback",
        "explain_item",
        "create_persona",
        "propose_interests",
        "profile_status",
        "search_feed",
    }
    stale -= leaves  # a leaf that is genuinely its own name is not stale

    # Only STRINGS: docstrings an agent reads and messages it is handed. The
    # Python identifiers (`def _sym_solve`, `_sym_solve(...)`) are internal and
    # deliberately keep the flat spelling.
    offenders = []
    for path in (SRC / "mcp").glob("*.py"):
        tree = ast.parse(path.read_text())
        strings = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        for text in strings:
            for name in sorted(stale):
                # Backticked and bare mentions count too. The guard matched
                # only `name(` and "call name", so seven live descriptions told
                # an agent to use `kg_path`, `search_feed` and `list_feed` --
                # none of which the server serves since namespacing -- and one
                # description mixed both spellings in a single sentence.
                # Backticked mentions count too. The guard matched only
                # `name(` and "call name", so seven live descriptions told an
                # agent to use `kg_path`, `search_feed` and `list_feed` -- none
                # of which the server serves since namespacing.
                #
                # NOT a bare-word match: that also hits Python identifiers like
                # the `_list_feed` impl and `_cite_lookup`, which legitimately
                # exist. A backtick means "this is a thing you can call", which
                # is exactly the claim being checked.
                mentioned = (
                    re.search(rf"(?<![_.\w]){re.escape(name)}\s*\(", text)
                    or f"call {name}" in text
                    or f"`{name}`" in text
                )
                if mentioned:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, "these name tools that no longer exist: " + ", ".join(offenders)


# Numeric arguments that are deliberately unbounded, with the reason. Anything
# not listed here must declare a minimum -- the default is guarded, so a new
# argument is safe until someone argues otherwise.
UNBOUNDED_BY_DESIGN: set[str] = set()


def test_tool_schemas_constrain_their_arguments():
    """A weak schema is how a small model sends garbage.

    `content_type` accepted any string when only six values exist, and `limit`
    had no bounds -- so limit=0 and since_days=-30 reached the tool and had to
    be refused at runtime with a message the model then had to read and act
    on. A constraint in the schema is enforced by the CLIENT before the call
    is made, which is a round trip and a failed tool call cheaper.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server

    tools = {t.name: t.inputSchema for t in asyncio.run(mcp_server.mcp.list_tools())}

    problems = []
    for name, schema in tools.items():
        for arg, spec in (schema.get("properties") or {}).items():
            flat = str(spec)
            # Derived, not enumerated. The old version checked four hardcoded
            # names and so missed `since_days` -- the argument this docstring
            # cites as its own motivating example -- plus every `timeout` and
            # `sym.differentiate.order`, which feeds a CAS subprocess where an
            # unbounded order is a resource question, not just a bad input.
            types = [spec.get("type"), *(o.get("type") for o in spec.get("anyOf", []))]
            numeric = "integer" in types or "number" in types
            bounded = "minimum" in spec or any("minimum" in o for o in spec.get("anyOf", []))
            if numeric and not bounded and arg not in UNBOUNDED_BY_DESIGN:
                problems.append(f"{name}.{arg}: no minimum")
            if arg == "content_type" and "enum" not in flat:
                problems.append(f"{name}.{arg}: not an enum")
            if arg == "metric" and name == "kg.central" and "enum" not in flat:
                problems.append(f"{name}.{arg}: not an enum")
    assert not problems, "unconstrained tool arguments: " + ", ".join(problems)


def test_claude_md_tool_counts_match_the_live_surface():
    """A count in the always-loaded doc must not drift from the code.

    CLAUDE.md is read into context every session, so a stale number there is
    repeated with confidence for as long as it survives. It said "37 tools
    NAMESPACED as feed.*(19) kg.*(5) runs.*(6) sym.*(7)" and "serves all 41"
    on 2026-08-22, when the live surface was 46 as 22/9/8/7 -- both written
    when they were true, neither noticed going wrong, and both were quoted
    into a design spec before being measured.

    This asserts the per-namespace counts rather than only the total: a total
    can stay right while two namespaces drift in opposite directions.
    """
    import asyncio
    import re
    from collections import Counter

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("count-check")
    register_all(server)
    names = [t.name for t in asyncio.run(server.list_tools())]
    live = Counter(n.split(".", 1)[0] for n in names)

    text = (SRC.parent.parent / "CLAUDE.md").read_text()
    line = next(ln for ln in text.splitlines() if "MCP surface:" in ln)

    # EVERY "<n> MCP tools" / "<n> tools" claim in the file, not just the one
    # on the surface line. Line 5's opening summary said 37 while line 40 said
    # 50 -- the guard matched only the line it was written against, so the
    # stale number sat in the always-loaded doc's first paragraph.
    for stale in re.finditer(r"(\d+)\s+MCP tools", text):
        assert int(stale.group(1)) == len(names), (
            f"CLAUDE.md claims {stale.group(1)} MCP tools; live surface has {len(names)}"
        )

    total = int(re.search(r"MCP surface: (\d+) tools", line).group(1))
    assert total == len(names), (
        f"CLAUDE.md says {total} tools, live surface has {len(names)}. Update the line."
    )

    claimed = {m.group(1): int(m.group(2)) for m in re.finditer(r"(\w+)\.\*\((\d+)\)", line)}
    assert claimed == dict(live), f"CLAUDE.md claims {claimed}, live surface is {dict(live)}"


def test_readme_tool_count_and_table_match_the_live_surface():
    """The same drift as CLAUDE.md, in the doc a new reader actually opens.

    CLAUDE.md learned this lesson on 2026-08-22 and got a test; README did not,
    and went on saying "exposes 37 tools" while the live surface reached 50.
    Its table was worse than the number: it silently omitted every `cite.*`
    tool, all four `.ask` routers, all four `.tools` listings, and
    `feed.read`/`feed.harvest_engagement`/`feed.simulate_ratings` -- so a
    reader counting the rows got a third answer again.

    A table is a claim about completeness. This asserts the count AND that
    every live tool has a row, because a correct total above an incomplete
    table is the drift that is hardest to notice.
    """
    import asyncio
    import re

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("readme-check")
    register_all(server)
    names = {t.name for t in asyncio.run(server.list_tools())}

    text = (SRC.parent.parent / "README.md").read_text()

    claimed = int(re.search(r"exposes (\d+) tools", text).group(1))
    assert claimed == len(names), (
        f"README says {claimed} tools, live surface has {len(names)}. Update the line."
    )

    documented = set(re.findall(r"`([a-z_]+\.[a-z_]+)\(", text))
    missing = sorted(names - documented)
    assert not missing, f"live tools with no README row: {missing}"


def test_a_schema_bound_is_never_looser_than_the_code_enforces():
    """A schema that advertises what the code will not honour is worse than
    no bound, because a client validates against it and is told yes.

    `sym.*` declared `timeout` up to 120 while symbolic.py clamps to
    MAX_TIMEOUT=30, so a caller asking for 120 got 30 and was never told.
    Stricter-than-enforced is fine (cite.search caps at 25 under a global 50);
    looser is the bug.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server
    from attestation.mcp._shared import MAX_LIST_LIMIT
    from attestation.symbolic import MAX_TIMEOUT

    ceilings = {"timeout": MAX_TIMEOUT, "limit": MAX_LIST_LIMIT}

    looser = []
    for tool in asyncio.run(mcp_server.mcp.list_tools()):
        for arg, spec in (tool.inputSchema.get("properties") or {}).items():
            ceiling = ceilings.get(arg)
            declared = spec.get("maximum")
            if ceiling and declared and declared > ceiling:
                looser.append(f"{tool.name}.{arg}: schema says {declared}, code enforces {ceiling}")

    assert not looser, "schema bounds looser than the code: " + ", ".join(looser)
