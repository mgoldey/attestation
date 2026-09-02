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
import re
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
        "feed.py": 775,
        "subscriptions.py": 150,
        "knowledge.py": 165,
        "symbolic.py": 130,
        # ask.py was 540 code lines and entirely unguarded until the default
        # below existed. Split at the seam it already had: routing.py holds the
        # rule tables (pure functions over a string, no model, no database --
        # the regression guard for the measured 13/15), ask.py holds the four
        # tools that call them and touch the rest of the system.
        "ask.py": 325,
        # Raised for runs.record (2026-09-01): a new tool plus its Arm
        # pydantic model in provenance.py, one new routing rule (with its
        # own ordering comment) in routing.py.
        "routing.py": 285,
        "provenance.py": 350,
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
        # Namespaced spelling of the tool O1 folded into feed.persona_status:
        # not derivable from the served set the way the flat spellings above
        # are, since it was removed outright rather than renamed leaf-only.
        "feed.personas",
    }
    stale -= leaves  # a leaf that is genuinely its own name is not stale

    # Only STRINGS: docstrings an agent reads and messages it is handed. The
    # Python identifiers (`def _sym_solve`, `_sym_solve(...)`) are internal and
    # deliberately keep the flat spelling.
    #
    # Not just mcp/: cli.py's `attest eval` failure message told a reader "the
    # `feed.personas` MCP tool lists them" after that tool was folded into
    # `feed.persona_status` -- a user-facing message naming a retired tool,
    # caught nowhere because the walk stopped at mcp/. server.py is walked too
    # since it is the other place a caller-visible message could name a tool.
    scanned_paths = [*(SRC / "mcp").glob("*.py"), SRC / "cli.py", SRC / "server.py"]
    offenders = []
    for path in scanned_paths:
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

    The tool table itself moved from README to `docs/guides/agents.md` on
    2026-08-29 (README became a front door under 200 lines) -- this test
    moved with it, since the fact it guards lives there now, not in README.
    """
    import asyncio
    import re

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("readme-check")
    register_all(server)
    names = {t.name for t in asyncio.run(server.list_tools())}

    text = (SRC.parent.parent / "docs" / "guides" / "agents.md").read_text()

    claimed = int(re.search(r"exposes (\d+) tools", text).group(1))
    assert claimed == len(names), (
        f"docs/guides/agents.md says {claimed} tools, live surface has {len(names)}."
        " Update the line."
    )

    documented = set(re.findall(r"`([a-z_]+\.[a-z_]+)\(", text))
    missing = sorted(names - documented)
    assert not missing, f"live tools with no docs/guides/agents.md row: {missing}"


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

    # feed.digest's `limit` counts items CONSIDERED before grouping, not rows
    # returned, so it is not bounded by MAX_LIST_LIMIT -- MAX_DIGEST_ITEMS
    # bounds what it renders. Same name, different quantity.
    ceilings = {"timeout": MAX_TIMEOUT, "limit": MAX_LIST_LIMIT}
    exempt = {("feed.digest", "limit")}

    looser = []
    for tool in asyncio.run(mcp_server.mcp.list_tools()):
        for arg, spec in (tool.inputSchema.get("properties") or {}).items():
            ceiling = ceilings.get(arg)
            declared = spec.get("maximum")
            if (tool.name, arg) in exempt:
                continue
            if ceiling and declared and declared > ceiling:
                looser.append(f"{tool.name}.{arg}: schema says {declared}, code enforces {ceiling}")

    assert not looser, "schema bounds looser than the code: " + ", ".join(looser)


def test_nothing_holds_a_sqlite_connection_across_requests():
    """A connection shared between threads is a 500 generator.

    The web UI held one "for the whole app" while FastAPI ran sync routes in a
    threadpool, so concurrent requests interleaved cursors: get_user returned
    None for a user that exists, and that None reached autocreate_user. The MCP
    surface never had it -- `@tool` opens a connection per call -- so the rule
    is one holder's mistake, not a design.

    Enforced structurally: module-level or closure-scoped `get_db(...)` results
    that outlive a request are what to look for.
    """
    import re

    server = (SRC / "server.py").read_text()

    # get_db may be called, but its result must not be bound once and reused --
    # it goes through a per-thread accessor.
    direct = re.findall(r"^\s+(\w+)\s*=\s*get_db\(", server, flags=re.MULTILINE)
    allowed = {"existing"}  # the per-thread cache inside connection()
    leaked = [name for name in direct if name not in allowed]
    assert not leaked, (
        f"server.py binds a shared connection: {leaked}."
        " Use the per-thread accessor; a shared connection interleaves cursors."
    )


def test_documented_response_limits_match_the_constants():
    """README and CLAUDE.md both stated feed.list's limits, both wrong, in
    opposite directions: README "capped at 50" against a real cap of 13, and
    CLAUDE.md "defaults to limit=5" against a real default of 4.

    The cap matters most. It is enforced as Field(le=MAX_SEARCH_LIMIT), so an
    agent author budgeting from README's 50 plans a payload the schema rejects
    -- and 13 was derived by measurement against a 7000-char ceiling, so it is
    exactly the kind of number that moves.

    The tool table (and this "capped at" line) moved from README to
    `docs/guides/agents.md` on 2026-08-29 -- this test moved with it.
    """
    import re
    from pathlib import Path

    from attestation.mcp.feed import DEFAULT_LIST_LIMIT, MAX_SEARCH_LIMIT

    root = Path(__file__).resolve().parents[1]
    agents_guide = (root / "docs" / "guides" / "agents.md").read_text()
    claude = (root / "CLAUDE.md").read_text()

    capped = re.search(r"capped at (\d+)", agents_guide)
    assert capped, "docs/guides/agents.md no longer states feed.list's cap"
    assert int(capped.group(1)) == MAX_SEARCH_LIMIT, (
        f"docs/guides/agents.md says capped at {capped.group(1)},"
        f" MAX_SEARCH_LIMIT is {MAX_SEARCH_LIMIT}"
    )

    default = re.search(r"feed\.list defaults to limit=(\d+)", claude)
    assert default, "CLAUDE.md no longer states feed.list's default"
    assert int(default.group(1)) == DEFAULT_LIST_LIMIT, (
        f"CLAUDE.md says default {default.group(1)}, DEFAULT_LIST_LIMIT is {DEFAULT_LIST_LIMIT}"
    )


def test_claude_md_noqa_inventory_matches_the_tree():
    """CLAUDE.md states the BLE001 count as a policy ("4 inline sites, each
    carrying its reason") and then lists them with line numbers. Live count was
    7 and three of the four line numbers had rotted.

    The count is the part that matters -- it is a claim a reviewer would check
    by counting, and it is how the repo asserts there is no blanket suppression.
    Line numbers are deliberately NOT asserted here: they rot on every edit
    above them, and citing them was the mistake.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "attestation"
    # Real suppressions only: a noqa suppresses nothing unless it trails code,
    # and _tool.py's module docstring DISCUSSES these sites (carrying its own
    # copy of the stale count, which is how this rots in two places at once).
    live = sum(
        1
        for path in root.rglob("*.py")
        for line in path.read_text().splitlines()
        if "# noqa: BLE001" in line and not line.lstrip().startswith(("#", "`", "*"))
    )
    claude = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text()
    stated = re.search(r"(\d+) inline `# noqa: BLE001` sites", claude)
    assert stated, "CLAUDE.md no longer states the BLE001 site count"
    assert int(stated.group(1)) == live, (
        f"CLAUDE.md says {stated.group(1)} BLE001 sites; the tree has {live}."
        " Each must carry its reason -- add it there and update the count."
    )


def test_the_docs_index_lists_every_source_and_test_file():
    """CLAUDE.md's index is a contributor's map, and the file says to read the
    listed docs before writing code. It had drifted 11 files behind -- including
    BOTH personas.py files, right after the persona split, so a contributor
    told to read first was pointed at feed.py where those tools no longer live.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    index = (root / "CLAUDE.md").read_text()
    index_block = index[
        index.index("[Project Docs Index]") : index.index(
            "```", index.index("[Project Docs Index]")
        )
    ]
    listed = set(re.findall(r"[\w.-]+\.(?:py|md|toml|yml|yaml|sample)", index_block))

    actual: set[str] = set()
    for pattern in ("src/attestation/**/*.py", "tests/*.py"):
        for path in root.glob(pattern):
            if "__pycache__" not in str(path):
                actual.add(path.name)

    missing = sorted(actual - listed)
    assert not missing, (
        f"CLAUDE.md's docs index is behind the tree by {len(missing)} file(s): {missing}"
    )


def test_no_tool_declares_a_default_its_own_schema_rejects():
    """`kg.neighbors` declared `limit: Limit = 20` and `kg.concepts` `= 50`,
    while Limit is `Field(le=16)`. An agent reading the schema sees
    `default=50, maximum=16` -- a contradiction in the one artifact it has to
    reason from. Checks every numeric bound on every tool, not just limit.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    mcp = FastMCP("audit")
    register_all(mcp)
    bad: list[str] = []
    for tool in asyncio.run(mcp.list_tools()):
        for name, spec in (tool.inputSchema.get("properties") or {}).items():
            default = spec.get("default")
            if not isinstance(default, int | float) or isinstance(default, bool):
                continue
            low, high = spec.get("minimum"), spec.get("maximum")
            if high is not None and default > high:
                bad.append(f"{tool.name}.{name} default={default} > maximum={high}")
            if low is not None and default < low:
                bad.append(f"{tool.name}.{name} default={default} < minimum={low}")
    assert not bad, "tools declare defaults their own schema rejects: " + "; ".join(bad)


def test_truncation_messages_do_not_name_a_limit_the_schema_rejects():
    """`kg.neighbors` said "raise limit to see more" and `runs.list` said
    "raise limit (max 25)", while the schema caps limit at 16. A caller obeying
    either got a pydantic validation dump -- measured on gemma4:e2b, the model
    read the message, sent limit=20, and abandoned the task.

    Scans the source for a message naming a numeric ceiling above the schema's.
    """
    import re
    from pathlib import Path

    from attestation.mcp._shared import MAX_LIST_LIMIT

    root = Path(__file__).resolve().parents[1] / "src" / "attestation" / "mcp"
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue  # a comment explaining the bug is not the bug
            for match in re.finditer(r"raise limit \(max (\d+)\)", line):
                if int(match.group(1)) > MAX_LIST_LIMIT:
                    offenders.append(f"{path.name}:{lineno} advertises max {match.group(1)}")
    assert not offenders, f"messages name a limit above the schema's {MAX_LIST_LIMIT}: {offenders}"


def test_the_feed_entry_points_say_what_the_corpus_holds():
    """An agent with these tools available answered a request for arXiv papers
    on KV-cache optimization with "I do not have a tool that can execute live
    searches on external academic repositories" -- and told the user to go
    search arxiv.org by hand.

    The tools were registered and working: the same query through feed.search
    returns four directly on-topic KV-cache papers, and the corpus holds 3,106
    arXiv items. The failure was the DESCRIPTION. "Search items by keyword"
    does not tell a caller what the items are, so a request phrased as
    "find me papers" never matched a tool that only advertised "items".

    A tool's description is its API. These two are the entry points a model
    reaches for first, so they must name the thing they contain.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    mcp = FastMCP("audit")
    register_all(mcp)
    described = {t.name: (t.description or "") for t in asyncio.run(mcp.list_tools())}

    # The FIRST line specifically. Many tools mention papers somewhere in a
    # long docstring; a model choosing among 50 tools reads the summary line,
    # and the original failure was a first line that said only "Search items by
    # keyword". Checking the whole description passes even when that line is
    # restored verbatim, which makes the guard useless -- verified by mutation.
    for name in ("feed.ask", "feed.search"):
        first_line = described[name].splitlines()[0].lower()
        assert any(word in first_line for word in ("paper", "arxiv", "research", "article")), (
            f"{name}'s summary line never says its corpus holds papers, so an"
            f" agent asked for papers cannot tell this tool applies:"
            f" {described[name].splitlines()[0]!r}"
        )


def test_every_tool_summary_line_is_a_complete_sentence():
    """A router scans the FIRST line to choose among tools, and six tools had
    a first line that stopped mid-clause -- "Topic clusters in the reading
    graph, each labelled by its most". The rest of the docstring is never
    weighed at selection time, so a truncated summary is the whole pitch.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    mcp = FastMCP("audit")
    register_all(mcp)
    truncated = [
        f"{t.name}: {t.description.splitlines()[0][-40:]!r}"
        for t in asyncio.run(mcp.list_tools())
        if t.description and not t.description.splitlines()[0].rstrip().endswith((".", "?", "!"))
    ]
    assert not truncated, "summary lines break mid-sentence: " + "; ".join(truncated)


def test_no_description_names_a_tool_that_does_not_exist():
    """Namespacing renamed every tool, and two descriptions still pointed at
    the old flat names: `feed.source_add` said "Use preview_feed first" and
    `runs.claims_coverage` said "the inverse of `claims_check`". A model that
    follows a cross-reference to a name that is not registered wastes a turn on
    a hard error.
    """
    import asyncio
    import re

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    mcp = FastMCP("audit")
    register_all(mcp)
    tools = asyncio.run(mcp.list_tools())
    registered = {t.name for t in tools}

    # A bare snake_case name in backticks is a tool reference when the same
    # WORDS exist as a registered tool, in any order: `preview_feed` against
    # `feed.source_preview`, `claims_check` against `runs.claims_check`.
    # Matching on the local half alone missed the first of those, since the
    # rename reordered the words -- verified by mutation.
    def words(name: str) -> frozenset[str]:
        return frozenset(name.replace(".", "_").split("_"))

    registered_words = {words(n) for n in registered}
    phantom: list[str] = []
    for tool in tools:
        for ref in re.findall(r"`([a-z][a-z0-9_]*_[a-z0-9_]+)`", tool.description or ""):
            if ref not in registered and words(ref) <= set().union(*registered_words):
                if any(words(ref) <= rw for rw in registered_words):
                    phantom.append(f"{tool.name} -> `{ref}`")
    assert not phantom, "descriptions name pre-namespacing tools: " + "; ".join(phantom)


def test_every_tool_that_returns_a_url_tells_the_agent_to_show_it():
    """A link the agent receives and does not render is a link the reader lacks.

    A watched Slack session asked "what should I read first today?", got five
    ranked papers, and rendered title/source/tags/item_id for each -- dropping
    every url. The reader replied "you didn't give links". The urls were all
    present in the payload (verified against the live DB: all five items had
    real arxiv.org/abs addresses, and 5499 of 5499 items carried one), so this
    was never a data or ranking bug. SKILL.md has said "present each item as
    one line: a linked title" from the start, but that is 700 lines of body
    loaded on invoke, whereas the tool description is in the prompt on EVERY
    turn -- and it listed `url` as a returned field without ever saying to
    render it. `item_id` looks like an identifier and is not an address, so it
    is what a small model reaches for.

    Keyed off the tools that actually EMIT a url, listed here explicitly. A
    first version keyed off `outputSchema`, which only the four `ask` routers
    declare -- so it inspected four tools, silently skipped `feed.list` and
    `feed.search`, and passed while both of the tools from the transcript were
    unguarded. Deleting either instruction wholesale did not fail it. The
    adjacency requirement matters too: every docstring already says "Returns
    ... url" in passing, so a bare `"url" in text and "show" in text` check
    passed with the whole instruction gone.
    """
    import asyncio
    import os
    import tempfile

    os.environ.setdefault("RSS_DB", tempfile.mkdtemp() + "/t.db")
    from attestation import mcp_server

    # Tools whose rows carry a url a human is meant to open. Verified against
    # a live payload rather than assumed: feed.list and feed.search both
    # return `url` on every row.
    EMIT_URLS = {"feed.list", "feed.search", "feed.ask", "runs.ask", "kg.ask", "sym.ask"}

    served = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    missing = EMIT_URLS - set(served)
    assert not missing, f"guard names tools that no longer exist: {missing}"

    silent = []
    for name in sorted(EMIT_URLS):
        tool = served[name]

        # Prose only. Scanning the serialized schema matched a field literally
        # NAMED url ("title": "Url") sitting in the same unsplittable JSON blob
        # as some unrelated verb, so deleting every real instruction still
        # passed. Collect the human-written `description` strings instead.
        def _descriptions(node) -> list[str]:
            found = []
            if isinstance(node, dict):
                if isinstance(node.get("description"), str):
                    found.append(node["description"])
                for value in node.values():
                    found.extend(_descriptions(value))
            elif isinstance(node, list):
                for value in node:
                    found.extend(_descriptions(value))
            return found

        prose = [tool.description or "", *_descriptions(tool.outputSchema or {})]
        sentences = [part for chunk in prose for part in re.split(r"(?<=[.!?])\s+|\n", chunk)]
        instructs = any(
            "url" in part.lower()
            and re.search(r"\b(show|render|display|linked|link)\b", part.lower())
            for part in sentences
        )
        if not instructs:
            silent.append(name)
    assert not silent, f"these tools return a url but never tell the agent to show it: {silent}"


def test_no_doc_quotes_a_stale_attestation_tool_count():
    """Two docs had guards; the drift moved to the docs that did not.

    CLAUDE.md and README each got a test after each was caught claiming 37
    tools against a live 46. The lesson did not generalise: on 2026-08-24
    `docs/architecture/research-profile.md` still said "attestation MCP | one
    combined server, 41 tools", and `docs/measurement-lessons.md` -- a file
    written that same day ABOUT numbers going stale -- quoted 46 and 22 in
    prose with nothing pinning either.

    So this guard is keyed off the docs tree rather than a list of filenames:
    a new doc is covered the day it is written, which is the failure mode the
    per-file tests kept having.

    Only attestation's own counts are asserted. `research-profile.md` also
    cites 67/27/40 for Hermes' whole tool budget and `filament`'s share of it
    -- real numbers about another system, which a naive "every integer near
    the word tools" check would report as failures forever.

    The file set and phrase set both grew after the round-two final review
    found three sites this guard's original phrasing missed even though its
    docs tree already covered two of them: CLAUDE.md's "unset serves all 46"
    and "feed 22 …" (a per-surface count, not the total, so the total-only
    `OURS` pattern below cannot see it even now that its wording matches),
    SKILL.md's "Unset serves all 46 tools", and agents.md's "does not need
    all 46" -- three different phrasings of the same total, none containing
    the literal words the original `OURS` alternatives required ("MCP
    surface", "live surface", "the full", or a bare "N tools" near
    "attestation"/"attest-mcp"). SKILL.md lives under `src/attestation/
    skills/`, outside the `docs/` tree the file list walked, so it is added
    by path alongside the pre-existing README.md/CLAUDE.md additions.

    Deliberately phrase-narrow, not a blanket digit-plus-"tools" scan. A blanket
    match false-positives on legitimate numbers that share these files with
    the real count: CLAUDE.md's "67 tool schemas" is Hermes' whole prompt
    budget, not attestation's surface, and "flat-37" names a historically
    measured routing score, not a live tool count -- both would read as
    stale claims about attestation under a pattern that does not check what
    the number is ABOUT. Each phrase added here is added because a real doc
    used it and the guard missed it, not preemptively -- narrower coverage
    that catches real drift beats broad coverage that also flags text this
    guard was never meant to police.
    """
    import asyncio
    import re

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("doc-count-check")
    register_all(server)
    live_total = len(asyncio.run(server.list_tools()))

    root = SRC.parent.parent
    docs = [p for p in (root / "docs").rglob("*.md") if "superpowers" not in p.parts]
    docs += [
        root / "README.md",
        root / "CLAUDE.md",
        root / "docs/guides/agents.md",
        *sorted((root / "src/attestation/skills").glob("*/SKILL.md")),
    ]
    docs = list(dict.fromkeys(docs))  # de-dupe: agents.md is already under docs/

    # A count is attestation's only when the sentence says so. The nearby
    # words are what disambiguate it from Hermes' 67 or filament's 40.
    # The word "tools" is always required: dropping it to catch a bare "the
    # full 46" matched "all 711 matches" and "claims 127" instead. A count
    # that omits the noun is unguardable by pattern, so the docs say it.
    OURS = re.compile(
        r"(?:attestation|attest-mcp|MCP surface|live surface|the full)"
        r"\D{0,40}?(\d+)\s+tools"
        r"|(\d+)\s+tools?\b(?=\D{0,40}?(?:attestation|attest-mcp))"
        # "of the N tools": still carries the word "tools" like the two
        # alternatives above, but "of the" alone is not a project-specific
        # anchor -- demonstrated false positive: "a handful of the 40 tools
        # in its provider ecosystem" has nothing to do with attestation.
        # Anchored the SAME way the first alternative above is: an anchor
        # word (attestation/attest-mcp/MCP surface/live surface/the full)
        # leading the match within the same 40-char window, not "already
        # scoped by surrounding prose" the way SERVES_ALL's phrasings are
        # (those are unambiguous possessives, "serves all N"/"need all N",
        # on their own -- "of the N tools" is not).
        # The gap uses `.` here, not the `\D` the first alternative uses --
        # a real anchored sentence can carry its OWN digits before "of the"
        # (e.g. "attestation serves 46 of the 46 tools"), and `\D` cannot
        # skip over that "46" to reach the anchor's target count. `.` is
        # still bounded to 40 chars and still requires the anchor word, so
        # it does not reopen the unanchored gap the review flagged.
        r"|(?:attestation|attest-mcp|MCP surface|live surface|the full)"
        r".{0,40}?of\s+the\s+(\d+)\s+tools\b",
        re.I,
    )
    # "serves all N" / "does not need all N": both name the unscoped total
    # without the word "tools" close enough for OURS to match, and without
    # any of "attestation"/"attest-mcp"/"MCP surface"/"live surface" nearby
    # either -- the two phrasings that survived the round-two review's
    # I2 finding (CLAUDE.md's "unset serves all 46", agents.md's "does not
    # need all 46"). Both sentences are already scoped to this project by
    # the surrounding prose (ATTEST_TOOLS, the tool surface), so no nearby-
    # word disambiguation is needed the way OURS needs one for a bare count.
    #
    # "N-tool" (e.g. "46-tool surface") was considered and dropped: a plain
    # `(\d+)-tool\b` also matches the tail of a dated filename like
    # `2026-08-21-tool-surface-design.md` (its "21-tool" reads exactly like
    # a live-count claim to the regex) and collides with a legitimate
    # historical count this repo deliberately keeps around --
    # agents.md's "a flat 37-tool list" names the swarm-refutation
    # measurement's baseline, not attestation's live total, and CLAUDE.md
    # says as much. Exactly the false-positive risk this test's docstring
    # warns a blanket pattern invites; "N-tool" stays out.
    SERVES_ALL = re.compile(r"(?:serves|need)\s+all\s+(\d+)\b", re.I)
    stale = []
    for path in docs:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            for match in SERVES_ALL.finditer(line):
                claimed = int(match.group(1))
                if claimed != live_total:
                    stale.append(
                        f"{path.relative_to(root)}: claims 'all {claimed}', live is {live_total}"
                    )
            for match in OURS.finditer(line):
                claimed = int(next(g for g in match.groups() if g))
                if claimed != live_total:
                    stale.append(
                        f"{path.relative_to(root)}: claims {claimed}, live is {live_total}"
                    )
    assert not stale, "stale attestation tool counts:\n  " + "\n  ".join(stale)

    # Per-surface counts drift independently of the total, and CLAUDE.md warns
    # these "MOVE". A doc citing ATTEST_TOOLS=<surface> with a number is making
    # a checkable claim; the total-guard above cannot see it.
    import os

    for path in docs:
        if not path.exists():
            continue
        for match in re.finditer(r"ATTEST_TOOLS=(\w+)\D{0,60}?\((\d+)\s+tools\)", path.read_text()):
            surface, claimed = match.group(1), int(match.group(2))
            previous = os.environ.get("ATTEST_TOOLS"), os.environ.get("ATTEST_EXPAND")
            os.environ["ATTEST_TOOLS"], os.environ["ATTEST_EXPAND"] = surface, "1"
            try:
                scoped = FastMCP(f"surface-{surface}")
                register_all(scoped)
                actual = len(asyncio.run(scoped.list_tools()))
            finally:
                for key, value in zip(("ATTEST_TOOLS", "ATTEST_EXPAND"), previous):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            assert claimed == actual, (
                f"{path.relative_to(root)}: ATTEST_TOOLS={surface} claims {claimed} "
                f"tools, live surface has {actual}"
            )

    def _expanded_surface_count(surface: str) -> int:
        previous = os.environ.get("ATTEST_TOOLS"), os.environ.get("ATTEST_EXPAND")
        os.environ["ATTEST_TOOLS"], os.environ["ATTEST_EXPAND"] = surface, "1"
        try:
            scoped = FastMCP(f"surface-{surface}")
            register_all(scoped)
            return len(asyncio.run(scoped.list_tools()))
        finally:
            for key, value in zip(("ATTEST_TOOLS", "ATTEST_EXPAND"), previous):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # CLAUDE.md's "Agent surfaces" line states the per-surface counts as
    # bare "<surface> <n>" pairs rather than the `ATTEST_TOOLS=<surface>
    # (<n> tools)` shape the loop above matches -- "feed 22 / provenance 9 /
    # knowledge 12 / symbolic 9 with ATTEST_EXPAND=1" is the exact sentence
    # I2 found stale (feed claimed 22, live is 21). Scoped to the four real
    # surface names so this cannot fire on an unrelated "feed 22" elsewhere.
    SURFACE_NAMES = {"feed", "provenance", "knowledge", "symbolic"}
    for path in docs:
        if not path.exists():
            continue
        text = path.read_text()
        for match in re.finditer(r"\b(feed|provenance|knowledge|symbolic)\s+(\d+)\b", text):
            surface, claimed = match.group(1), int(match.group(2))
            if surface not in SURFACE_NAMES:
                continue
            # Only within a sentence that is actually about the
            # ATTEST_TOOLS/ATTEST_EXPAND surface split -- otherwise "feed 22"
            # could be any unrelated mention of the word "feed" near a number.
            window = text[max(0, match.start() - 200) : match.end() + 60]
            if "ATTEST_TOOLS" not in window and "ATTEST_EXPAND" not in window:
                continue
            actual = _expanded_surface_count(surface)
            if claimed != actual:
                stale.append(
                    f"{path.relative_to(root)}: claims {surface} {claimed}, live is {actual}"
                )
    assert not stale, "stale attestation tool counts:\n  " + "\n  ".join(stale)


def test_no_doc_quotes_a_stale_attestation_skill_count():
    """The tool-count guard above has a skill-count sibling now.

    WS-T1 fixed two live docs (README.md, docs/guides/agents.md) still
    saying "five skills" after attestation-record and attestation-annotate
    brought the bundled total to seven. Same failure shape as the tool
    count: a number that is true the day it's written and never re-checked.

    Anchored the same way OURS/SERVES_ALL are above -- narrow phrasing that
    matched a REAL sentence, not a blanket "number near the word skill(s)"
    scan. A blanket scan false-positives constantly in this repo's own
    prose: CLAUDE.md alone has "68 skills cost ~7 KB" (Hermes' whole index,
    not attestation's bundle) and "6/6 -> 3/6" (a routing score) sitting a
    few words from "skill". Two phrasings are anchored because two real
    sentences use them: "N under `src/attestation/skills`" (agents.md,
    CLAUDE.md) and "(N skills:" (README.md).

    Deliberately excluded: agents.md's "split into five `attestation-*`
    skills on 2026-08-30" is retrospective, not a live-total claim -- it
    names the count AT THE TIME OF THAT SPLIT and says so with the date
    immediately after, the same reason the tool-count guard above excludes
    "N-tool" (a dated split, like a dated filename, is history this repo
    deliberately keeps rather than a live count to police). Requiring the
    match to skip a trailing date within a few words would work but is
    unnecessary complexity for one sentence that already says which day it
    is about; simpler to leave "split into" out of the anchor set the way
    the tool-count guard left "N-tool" out of its own.

    The skill counts here are spelled out in words ("seven", "five"), not
    digits, unlike the tool counts -- so the anchors match a small word list
    rather than a bare digit pattern.
    """
    import attestation.install as install

    root = SRC.parent.parent
    docs = [p for p in (root / "docs").rglob("*.md") if "superpowers" not in p.parts]
    docs += [
        root / "README.md",
        root / "CLAUDE.md",
        root / "docs/guides/agents.md",
        *sorted((root / "src/attestation/skills").glob("*/SKILL.md")),
    ]
    docs = list(dict.fromkeys(docs))

    live_total = len(install.SKILL_NAMES)
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    words_pattern = "|".join(number_words)
    SKILL_COUNT = re.compile(
        rf"(?:({words_pattern})\s+under\s+`?src/attestation/skills"
        rf"|\(({words_pattern})\s+skills?:)",
        re.I,
    )

    stale = []
    for path in docs:
        if not path.exists():
            continue
        text = path.read_text()
        for match in SKILL_COUNT.finditer(text):
            word = (match.group(1) or match.group(2)).lower()
            claimed = number_words[word]
            if claimed != live_total:
                stale.append(
                    f"{path.relative_to(root)}: claims {word} ({claimed}) skills,"
                    f" install.SKILL_NAMES has {live_total}"
                )
    assert not stale, "stale attestation skill counts:\n  " + "\n  ".join(stale)


DOMAIN = {
    "explain",
    "features",
    "ingest",
    "simulate",
    "rank",
    "kg",
    "claims",
    "ledger",
    "corpus",
    "citations",
    "implicit",
    "personas",
    "feeds",
}


def _imports_of(path: pathlib.Path, module: str) -> list[str]:
    """Names imported from `module` anywhere in the file, function bodies included:
    a lazy import of the concrete client is still the concrete client.

    `module` is a dotted path such as "attestation.llm". Three import forms
    all count: `from attestation.llm import x`, `import attestation.llm`, and
    `from attestation import llm [as x]` -- the last is this codebase's own
    dominant deferred-import idiom (`cli.py` uses it eight times,
    `mcp/subscriptions.py` and `mcp/feed.py` once each), so a guard that only
    caught the first two forms had a hole through the likeliest evasion.
    """
    package, _, leaf = module.rpartition(".")
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.extend(a.name for a in node.names)
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names if a.name == module)
        if isinstance(node, ast.ImportFrom) and node.module == package:
            names.extend(f"{module} as {a.asname}" for a in node.names if a.name == leaf)
    return names


def test_domain_reaches_models_only_through_ports():
    """ports.py is load-bearing only if the domain uses it. At 787823d explain,
    features and ingest imported the concrete client; a second provider would
    have meant editing domain code."""
    offenders = {}
    for path in _modules():
        name = _module_name(path)
        if name in DOMAIN:
            found = _imports_of(path, "attestation.llm")
            if found:
                offenders[_rel(path)] = found
    assert not offenders, f"domain modules importing the concrete client: {offenders}"


_SQL = re.compile(r"""["'](SELECT|INSERT|UPDATE|DELETE|WITH) """)
MCP_SQL_BASELINE = 21  # measured after the Wave-1 seams: feed 15, personas 5, _tool 1


def test_mcp_layer_sql_only_ratchets_down():
    """The presentation layer writing its own queries is the braid the onion
    was for. Pinned, not banned: it falls as seams move queries into domain
    readers, and a new query up here needs a reason in a spec."""
    counts = {
        _rel(p): len(_SQL.findall(p.read_text())) for p in _modules() if p.parent.name == "mcp"
    }
    total = sum(counts.values())
    assert total <= MCP_SQL_BASELINE, f"SQL in mcp/ rose to {total}: {counts}"


def test_no_mcp_module_imports_a_private_domain_name():
    """A private name crossing a module boundary is a missing public function."""
    offenders = {}
    for path in _modules():
        if path.parent.name != "mcp":
            continue
        tree = ast.parse(path.read_text())
        private = [
            f"{n.module}.{a.name}"
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("attestation.")
            and not (n.module or "").startswith("attestation.mcp")
            for a in n.names
            if a.name.startswith("_")
        ]
        if private:
            offenders[_rel(path)] = private
    assert not offenders, f"mcp modules importing private domain names: {offenders}"
