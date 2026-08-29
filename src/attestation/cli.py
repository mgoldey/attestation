"""attest CLI: ingest, tag, serve, runs, claims, browse, kg-report, install."""

import argparse
import contextlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

import httpx


def _default_feeds_path() -> str:
    """Prefer a cwd-relative feeds.toml (dev checkout), else the packaged copy.

    Mirrors feeds.py::CANDIDATES_PATH and kg.py::_ALIAS_PATH: a
    Path(__file__)-relative fallback so `attest ingest` works from a wheel
    install with no checkout present, not just from the repo root.
    """
    local = Path("feeds.toml")
    if local.exists():
        return str(local)
    return str(Path(__file__).resolve().parent / "feeds.toml")


@contextlib.contextmanager
def open_db(db_arg: str | None):
    """Open a connection for a CLI arg's DB path and guarantee it closes.

    `sqlite3.Connection.__enter__` manages transactions, not handle
    lifetime, so a plain `with get_db(...)` would leave the handle open --
    this wraps it in `contextlib.closing` around the resolved path.
    """
    import sqlite3

    from attestation.db import get_db, resolve_db_path

    path = resolve_db_path(db_arg)
    try:
        # A path whose directory does not exist yet is an ordinary first run,
        # not an error -- but a path under a directory that cannot be created
        # is a typo, and it should read as one.
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(path)
    except (OSError, sqlite3.Error) as exc:
        # A typo in ATTEST_DB, or a cwd the user cannot write, printed 18 lines of
        # traceback ending in "unable to open database file" -- which does not
        # say WHICH file, and so does not say which typo. Every other failure
        # in this CLI is a sentence; this one was the exception because it fired
        # before any command body ran, outside every command's own error
        # handling. Both arms are needed: mkdir raises OSError, sqlite3 raises
        # OperationalError, and OperationalError is not an OSError.
        raise SystemExit(fail(f"cannot open database at {path}: {exc}")) from exc
    with contextlib.closing(conn):
        yield conn


def fail(message: str) -> int:
    """Report a failure on stderr and return the exit code.

    Every failure path printed to stdout, so `attest runs list > runs.txt`
    wrote "no runs recorded" into the data file and `| jq` fed the complaint to
    jq. Exit codes are for programs; streams are how a program separates its
    answer from its excuse.
    """
    print(message, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Assemble every `attest` subcommand, each `help=` reused verbatim as its
    handler's one-line docstring (see the `test_help_matches_docstring`
    family in test_cli.py) so `--help` and `cmd_<name>.__doc__` cannot drift
    the way `eval --help` once did (see `test_a_subcommand_help_describes_
    what_it_prints`)."""
    p = argparse.ArgumentParser(prog="attest")
    p.add_argument("--version", action="version", version=f"attest {version('attestation')}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_db(sp):
        """Add the --db flag every subcommand but warmup/emit/install shares."""
        sp.add_argument(
            "--db",
            default=None,
            help="DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env var > "
            "~/.hermes/skills/science-recommendations/data/hermes.db (if it exists) > ./hermes.db",
        )

    sp = sub.add_parser("ingest", help="fetch feeds, embed, store")
    add_db(sp)
    sp.add_argument("--feeds", default=_default_feeds_path())
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("tag", help="LLM-tag untagged items (topic tags + content type)")
    add_db(sp)
    sp.add_argument("--limit", type=int, default=None, help="max items to tag this run")
    sp.set_defaults(func=cmd_tag)

    sp = sub.add_parser("serve", help="run the web UI")
    add_db(sp)
    sp.add_argument("--port", type=int, default=8899)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("eval", help="cross-validated AUC for a persona's click classifier")
    add_db(sp)
    sp.add_argument("--user", required=True)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("warmup", help="pin chat + embedding models in VRAM")
    sp.set_defaults(func=cmd_warmup)

    sp = sub.add_parser("reload", help="restart running MCP servers so code edits take effect")
    sp.set_defaults(func=cmd_reload)

    sp = sub.add_parser("backup", help="write a consistent copy of the database")
    add_db(sp)
    sp.add_argument("dest", help="path to write; must not already exist")
    sp.set_defaults(func=cmd_backup)

    sp = sub.add_parser("emit", help="agent configs generated from the tool surfaces")
    sp.add_argument(
        "--write", action="store_true", help="write the Claude agent files (default: report only)"
    )
    sp.set_defaults(func=cmd_emit)

    sp = sub.add_parser("kg-report", help="knowledge-graph health + topic clusters")
    add_db(sp)
    sp.add_argument("--min-size", type=int, default=3, help="smallest cluster to list")
    sp.set_defaults(func=cmd_kg_report)

    sp = sub.add_parser("claims", help="verify claims written in Markdown against runs")
    add_db(sp)
    sp.add_argument("path", nargs="?", help="file or directory (default: $RESEARCH_ROOT)")
    sp.add_argument("--verdict", help="show only this verdict")
    sp.add_argument(
        "--coverage",
        action="store_true",
        help="instead: list numbers in prose that no claim covers",
    )
    sp.set_defaults(func=cmd_claims)

    sp = sub.add_parser("browse", help="open the ledger in Datasette (read-only)")
    add_db(sp)
    sp.add_argument("--port", type=int, default=8898)
    sp.add_argument("--open", action="store_true", help="open a browser window")
    sp.set_defaults(func=cmd_browse)

    sp = sub.add_parser("runs", help="experiment ledger read from artifacts on disk")
    add_db(sp)
    runs_sub = sp.add_subparsers(dest="runs_command", required=True)

    rp = runs_sub.add_parser("scan", help="re-read runs from a workspace directory")
    rp.add_argument("--root", help="workspace dir (default: $RESEARCH_ROOT)")
    rp.add_argument("--project", help="scan only this project")
    rp.set_defaults(func=cmd_runs_scan)

    rp = runs_sub.add_parser("list", help="runs in the ledger")
    rp.add_argument("--project")
    rp.add_argument("--family")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_runs_list)

    rp = runs_sub.add_parser("compare", help="rank the arms of an experiment family")
    rp.add_argument("family")
    rp.add_argument("--metric", help="default: the metric most arms share")
    rp.add_argument("--project", help="required when the family exists in more than one project")
    rp.set_defaults(func=cmd_runs_compare)

    rp = runs_sub.add_parser("show", help="one run in full")
    rp.add_argument("project")
    rp.add_argument("name")
    rp.set_defaults(func=cmd_runs_show)

    sp = sub.add_parser("bootstrap-persona", help="write pseudo-clicks for a persona")
    add_db(sp)
    sp.add_argument("name")
    sp.add_argument("-k", type=int, default=30)
    sp.set_defaults(func=cmd_bootstrap_persona)

    sp = sub.add_parser("install", help="idempotent setup + --check doctor mode")
    sp.add_argument("--check", action="store_true", help="detect only, change nothing")
    sp.add_argument("--yes", action="store_true", help="non-interactive consent")
    sp.add_argument("--now", action="store_true", help="also run the tag backfill inline")
    sp.set_defaults(func=cmd_install)
    return p


def warmup() -> None:
    """Pin chat + embed models in VRAM. Ollama-specific by design: derives the
    native /api base from the /v1 base URL; other backends don't need pinning."""
    from attestation.llm import base_url, chat_model, embed_model

    # How long Ollama holds the models in RAM after warmup.
    #
    # This was `-1` -- forever, until Ollama restarts. On a 23 GB machine that
    # pinned 5.4 GB across two llama-server processes and the kernel started
    # OOM-killing whatever else was running: a browser, a quantum-chemistry
    # job, the terminal. The pin exists so a demo does not stall on a cold
    # model load, and a demo is minutes long, not permanent.
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
    native = base_url().rstrip("/").removesuffix("/v1")
    try:
        httpx.post(
            f"{native}/api/chat",
            json={
                "model": chat_model(),
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_ctx": 8192},
            },
            timeout=300,
        ).raise_for_status()
        httpx.post(
            f"{native}/api/embed",
            json={"model": embed_model(), "input": "warmup", "keep_alive": keep_alive},
            timeout=300,
        ).raise_for_status()
    except httpx.HTTPError:
        print("warmup is Ollama-only; skipping for this backend")
        return
    print(f"models loaded (chat={chat_model()}, keep_alive={keep_alive})")


def _running_mcp_pids() -> list[int]:
    """PIDs of live `attest-mcp` processes, excluding the `uv run` wrappers.

    Matching the console script rather than the command line: a wrapper shares
    the same argv, and signalling it kills the shim while the watchdog
    respawns the pair, which looks like it worked and does not reload.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-f", "bin/attest-mcp$"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.split() if line.isdigit() and int(line) != os.getpid()]


def _emit_agent_files(root, write: bool) -> int:
    """Report -- or with `write`, create -- the .claude/agents files.

    Split from `cmd_emit` for the complexity ratchet, and because "is the MCP
    config right" and "are the agent files right" are two questions with two
    answers.
    """
    from attestation import emit

    agents_dir = root / ".claude" / "agents"
    generated = emit.claude_agents(root)

    if not write:
        stale = [
            name
            for name, body in generated.items()
            if not (agents_dir / f"attestation-{name}.md").is_file()
            or (agents_dir / f"attestation-{name}.md").read_text() != body
        ]
        if stale:
            print(f"{len(stale)} agent file(s) missing or differing: {', '.join(sorted(stale))}")
            print("  run `attest emit --write` to regenerate (an existing file that differs")
            print("  is reported, never overwritten -- a deliberate edit survives)")
        else:
            print(f"claude agents: all {len(generated)} present and current")
        return 0

    agents_dir.mkdir(parents=True, exist_ok=True)
    wrote, refused = [], []
    for name, body in generated.items():
        path = agents_dir / f"attestation-{name}.md"
        # Write only what is missing or already matches. An existing file whose
        # contents differ is someone's edit, and losing it is not this
        # command's call: the realistic path is not running --write on a file
        # you just changed, it is adding a fifth surface and silently losing an
        # unrelated edit as a side effect.
        if path.is_file() and path.read_text() != body:
            refused.append(name)
            continue
        path.write_text(body)
        wrote.append(name)

    if wrote:
        print(f"wrote {len(wrote)} agent file(s) to {agents_dir}")
    if refused:
        print(f"{len(refused)} file(s) differ from what would be generated and were LEFT")
        for name in sorted(refused):
            print(f"  {agents_dir / f'attestation-{name}.md'}")
        print("  delete one to accept the generated version, or keep your edit")
        return 1
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """write a consistent copy of the database

    A single-file copy, exact and restorable -- exists because `cp hermes.db
    backup.db` is what an operator types and it silently drops the WAL -- the
    copy opens, looks intact, and is missing the newest writes. Five such
    copies were found beside the live database.
    """
    from attestation.db import backup_db, get_db, resolve_db_path

    src = resolve_db_path(args.db)
    if not src.exists():
        return fail(f"no database at {src}")
    try:
        dest = backup_db(get_db(src), args.dest)
    except FileExistsError as exc:
        print(str(exc))
        return 1
    size = dest.stat().st_size / 1e6
    print(f"wrote {dest} ({size:.1f} MB) — restore by copying it back over {src}")
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    """agent configs generated from the tool surfaces

    Reports by default -- or with --write, produces -- the per-surface agent
    configs. Reporting is the default because nothing here overwrites: a
    difference between generated and on-disk is a fact the user acts on, not
    one this command resolves for them. See emit.py's module docstring.
    """
    import subprocess

    from attestation import emit
    from attestation.install import _checkout_root, _find_agent_binary

    root = _checkout_root()
    if root is None:
        print("not running from a checkout; nothing to point a config at")
        return 1

    agent = _find_agent_binary()
    if agent:
        proc = subprocess.run(
            [agent, "config", "get", "mcp_servers"], capture_output=True, text=True, timeout=60
        )
        findings = emit.check_hermes(emit.parse_config_dump(proc.stdout), root)
        if findings:
            print(f"{len(findings)} config problem(s):")
            for f in findings:
                print(f"  [{f.kind}] {f.detail}")
        else:
            print(f"mcp surfaces: all {len(emit.hermes_servers(root))} present and current")
    else:
        print("no agent binary found; skipping the MCP config check")

    return _emit_agent_files(root, args.write)


def cmd_reload(args: argparse.Namespace) -> int:
    """restart running MCP servers so code edits take effect

    An MCP server is spawned once per session and never reloads. Both live
    servers here were once found running code five commits stale -- the Hermes
    gateway and a Claude Code session, against commits made hours later -- so
    every fix landed in between was invisible to the agent using them.

    `hermes mcp test` does not catch that: it spawns a FRESH process, so it
    reports the code on disk rather than what a session is serving.

    Signalling is enough, but the respawn is LAZY: measured against a live
    Hermes gateway, nothing restarted for at least ten seconds after the kill
    and the new process appeared only when a tool was next called. So a reload
    leaves the server down rather than instantly fresh, which is fine for a
    chat session and worth saying rather than implying otherwise.
    """
    import signal

    pids = _running_mcp_pids()
    if not pids:
        print("no running attest-mcp servers; nothing to reload")
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            # A watchdog may reap or restart one between listing and
            # signalling. That is a race, not a failure, and must not stop
            # the remaining servers from being reloaded.
            print(f"  {pid}: already gone ({type(exc).__name__})")
            continue
        print(f"  {pid}: signalled")
    print(
        f"stopped {len(pids)} server(s). Each respawns on its client's next tool"
        " call, so the first call after this is slower and picks up your edits."
    )
    return 0


def cmd_warmup(args: argparse.Namespace) -> int:
    """pin chat + embedding models in VRAM"""
    import attestation.cli  # self-import so tests can monkeypatch attestation.cli.warmup

    attestation.cli.warmup()
    return 0


def _citation_resolver():
    """The configured bibliographic sources.

    Imported here rather than at module scope, matching
    `attestation.mcp.claims_tools._citation_resolver`: `claims.check` takes
    the resolver as an injected argument precisely so the ledger does not
    depend on the citation readers at import time, and this wiring should not
    undo that. `Resolver.from_env()` only stores paths and reads
    `ATTEST_CITATION_WEB` -- it does not touch a `.bib` file or Zotero's
    sqlite database until a lookup happens, so there is nothing here to guard
    against with a broad except.
    """
    from attestation import citations

    return citations.Resolver.from_env()


def cmd_claims(args: argparse.Namespace) -> int:
    """verify claims written in Markdown against runs"""
    from attestation import claims as claims_mod
    from attestation import ledger

    target = Path(args.path).expanduser() if args.path else ledger.workspace_root()
    if target is None:
        print("pass a path, or set RESEARCH_ROOT")
        return 1
    if not target.exists():
        return fail(f"no such path: {target}")

    if args.coverage:
        cov = claims_mod.coverage(target)
        for u in cov["uncovered"]:
            where = f"{Path(u['file']).name}:{u['line']}"
            print(f"  uncovered  {where:24s} {u['value']:<12g} {u['context'][:60]}")
        print(
            f"\n{cov['covered']}/{cov['numbers']} number(s) covered by a claim"
            f" across {cov['files']} file(s)"
        )
        return 0

    with open_db(args.db) as conn:
        out = claims_mod.check(conn, target, resolver=_citation_resolver())

    for problem in out["malformed"]:
        print(f"  malformed  {problem}")
    for v in out["verdicts"]:
        if args.verdict and v.verdict != args.verdict:
            continue
        where = f"{Path(v.claim.path).name}:{v.claim.line}"
        print(f"  {v.verdict:13s} {where:28s} {v.claim.metric}={v.claim.value:g}  {v.message}")

    if not out["claims"]:
        print(f"no claims found under {target}")
        print("annotate one beside the prose it describes:")
        print("  <!-- claim: project/run metric=wer value=0.053 tol=0.001 -->")
        return 0

    summary = ", ".join(f"{n} {k}" for k, n in sorted(out["counts"].items()))
    print(f"\n{out['claims']} claim(s): {summary}")
    if out["malformed"]:
        print(f"{len(out['malformed'])} malformed")
    # a contradicted claim is a document asserting something false; that is
    # worth a non-zero exit so it can gate a commit if anyone wants it to
    bad = out["counts"].get("contradicted", 0) + len(out["malformed"])
    return 1 if bad else 0


def cmd_browse(args: argparse.Namespace) -> int:
    """open the ledger in Datasette (read-only)"""
    import shutil
    import subprocess

    from attestation.db import resolve_db_path

    datasette = shutil.which("datasette") or str(Path(sys.executable).parent / "datasette")
    if not Path(datasette).exists():
        print("datasette is not installed -- `uv sync --group dev`")
        return 1
    db_path = resolve_db_path(args.db)
    if not Path(db_path).exists():
        return fail(f"no database at {db_path}")
    config = Path(__file__).resolve().parents[2] / "datasette.yml"
    cmd = [datasette, "--immutable", str(db_path), "--port", str(args.port)]
    # The database holds a vec0 virtual table for embeddings. Datasette's
    # own SQLite has no such module and refuses to open the file at all
    # ("no such module: vec0") unless the extension is loaded the same way
    # db.get_db does it.
    try:
        import sqlite_vec

        cmd += ["--load-extension", str(sqlite_vec.loadable_path())]
    except Exception:  # noqa: BLE001 - sqlite-vec is optional here; any
        # import or path failure just means vec0 tables are unavailable in
        # the viewer, which is a warning rather than a reason to not launch.
        print("warning: sqlite-vec not loadable; tables using vec0 will be unavailable")
    if config.exists():
        # canned queries are the point: named SQL with shareable URLs
        cmd += ["--metadata", str(config)]
    if args.open:
        cmd.append("-o")
    # --immutable, always: a viewer must never be able to write. It also
    # lets Datasette cache counts, which matters at 62k metric rows.
    print(f"browsing {db_path} read-only at http://localhost:{args.port}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def cmd_runs_scan(args: argparse.Namespace) -> int:
    """re-read runs from a workspace directory"""
    from attestation import ledger

    with open_db(args.db) as conn:
        root = ledger.workspace_root(args.root)
        if root is None:
            print("set RESEARCH_ROOT (or pass --root) to the directory holding your projects")
            return 1
        out = ledger.scan(conn, root, project=args.project)
        if out.get("message"):
            print(out["message"])
            return 1
        for name, count in sorted(out["scanned"].items()):
            print(f"  {name:28s} {count} run(s)")
        print(f"{sum(out['scanned'].values())} run(s) across {len(out['scanned'])} project(s)")
        if out["empty"]:
            # reported with a reason, not hidden: "found nothing" is not
            # "nothing there", and a bare count leaves the user no next step.
            print()
            for name in out["empty"]:
                print(f"  no runs in {name}: {out['diagnostics'].get(name, 'unrecognised layout')}")
        return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    """runs in the ledger"""
    from attestation import ledger

    with open_db(args.db) as conn:
        found = ledger.list_runs(conn, args.project, args.family, args.limit)
        if not found:
            return fail("no runs recorded -- run `attest runs scan` first")
        for r in found:
            fam = f"[{r['family']}]" if r["family"] else ""
            print(f"  {r['project']:20s} {r['name']:38s} {r['status']:10s} {fam}")
        print()
        for f in ledger.families(conn, args.project):
            print(f"  family {f['family']:32s} {f['n']} run(s)  ({f['project']})")
        return 0


def cmd_runs_compare(args: argparse.Namespace) -> int:
    """rank the arms of an experiment family"""
    from attestation import ledger

    with open_db(args.db) as conn:
        try:
            result = ledger.compare(conn, args.family, metric=args.metric, project=args.project)
        except ValueError as exc:
            return fail(str(exc))
        if not result["arms"]:
            # say which families exist rather than dead-ending: `compare
            # <project>` is the intuitive first guess and is not a family
            return fail(result.get("message") or f"no runs in family {args.family!r}")
        header = f"{result['family']} — ranked by {result['metric']} ({result['direction']})"
        # Naming the shared corpus says the comparison was *checked*, not
        # assumed -- the reader cannot tell those apart otherwise.
        if result.get("corpus"):
            header += f", all arms on {result['corpus']}"
        print(header + "\n")
        print(f"  {'arm':44s} {result['metric']:>10s} {'n':>6s}  {'step':>8s}  source")
        print(f"  {'-' * 44} {'-' * 10} {'-' * 6}  {'-' * 8}  {'-' * 6}")
        for arm in result["arms"]:
            # every row carries where the number came from: an auditor's
            # first question is "from which file?"
            src = arm.get("source_path") or ""
            if arm["value"] is None:
                print(f"  {arm['name']:44s} {'(none)':>10s} {'':>6s}  {'':>8s}  {src}")
                continue
            n = str(arm["n"]) if arm.get("n") is not None else "?"
            step = str(arm["step"]) if arm["step"] is not None else ""
            print(f"  {arm['name']:44s} {arm['value']:>10.4f} {n:>6s}  {step:>8s}  {src}")

        print(f"\nwinner: {result['winner']}")
        for caveat in result.get("caveats", []):
            print(f"  caveat: {caveat}")
        if result["without_metric"]:
            print(
                f"  {len(result['without_metric'])} arm(s) have no {result['metric']}:"
                f" {', '.join(result['without_metric'])}"
            )
        return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    """one run in full"""
    from attestation import ledger

    with open_db(args.db) as conn:
        found = ledger.detail(conn, args.project, args.name)
        if found is None:
            print(f"no run {args.name!r} in project {args.project!r}")
            return 1
        print(f"{found['project']}/{found['name']}  [{found['status']}]")
        print(f"source: {found['source_path']}")
        if found.get("notes"):
            print(f"\n{found['notes']}\n")
        for m in found["metrics"]:
            step = f" step={m['step']}" if m["step"] is not None else ""
            split = f" split={m['split']}" if m["split"] else ""
            print(f"  {m['metric']:24s} {m['value']:>14.6f}{step}{split}")
        return 0


def cmd_kg_report(args: argparse.Namespace) -> int:
    """knowledge-graph health + topic clusters"""
    from attestation import kg

    with open_db(args.db) as conn:
        health = kg.health(conn)
        # Ten zeros do not answer "what does my reading graph look like", and
        # they hide the actual fix: the graph is DERIVED from the tagging pass,
        # so an empty one means nothing has been tagged yet. `attest runs list`
        # already sets the house pattern for this.
        if not health.get("nodes"):
            tagged = conn.execute("SELECT COUNT(*) FROM item_tags").fetchone()[0]
            items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            # fail(), like every other empty/error state. This branch predates
            # that convention and kept printing guidance to stdout with exit 0,
            # so a pipeline could not tell "no graph yet" from "here is your
            # graph" -- while `runs compare` and `eval` both exit 1 to stderr
            # for the same shape of nothing-to-report.
            if not items:
                return fail("no items yet -- run `attest ingest` to fetch some, then `attest tag`")
            if not tagged:
                return fail(f"{items} item(s), none tagged -- run `attest tag` to build the graph")
            return fail(
                f"{tagged} tag assignment(s) but no concepts: a tag must be used at least"
                f" {kg.MIN_TAG_USES} times to become one. Tag more items."
            )

        for key, value in health.items():
            print(f"{key:24s} {value}")
        print()
        for i, c in enumerate(kg.communities(conn, min_size=args.min_size), 1):
            print(f"{i}. [{c['label']}] — {len(c['members'])} concepts")
            print(f"   {', '.join(c['members'])}\n")
        return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """fetch feeds, embed, store"""
    from attestation.embed import Embedder
    from attestation.ingest import run_ingest

    with open_db(args.db) as conn:
        stats = run_ingest(conn, Embedder(), args.feeds)
    print(stats)
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    """LLM-tag untagged items (topic tags + content type)"""
    import attestation.features
    from attestation.llm import base_url

    with open_db(args.db) as conn:
        stats = attestation.features.run_tagging(conn, limit=args.limit)
    print(stats)
    if stats.get("chat_down"):
        # Same words ingest uses for the same condition. The stats dict alone
        # said `failed: 2` and left the cause -- Ollama is not running -- to be
        # guessed.
        print(
            f"chat model unreachable at {base_url()} -- is ollama running?"
            " (`attest install --check` diagnoses this)",
            file=sys.stderr,
        )
        return 1
    return 1 if (stats["tagged"] == 0 and stats["failed"] > 0) else 0


def cmd_serve(args: argparse.Namespace) -> int:
    """run the web UI"""
    import uvicorn

    from attestation.db import resolve_db_path
    from attestation.server import create_app

    uvicorn.run(create_app(resolve_db_path(args.db)), host="127.0.0.1", port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """cross-validated AUC for a persona's click classifier"""
    from attestation.rank import evaluate_user, get_user
    from attestation.simulate import source_skew_caveat

    with open_db(args.db) as conn:
        user = get_user(conn, args.user)
        result = evaluate_user(conn, user["id"]) if user else None
        skew = source_skew_caveat(conn, user["id"]) if user else None
    if result is None:
        # Nonzero, like every other failure path here. "I could not measure
        # this" is not the answer the command exists to produce, and a script
        # gating on `attest eval` read the old exit 0 as a pass. Both causes
        # land here -- an unknown persona and a real one with too few mixed
        # clicks -- and both are told what would fix them.
        message = "insufficient click data for a meaningful holdout (need 10+ mixed clicks)"
        if user is None:
            # There is no `attest personas`: an earlier version of this line
            # advised one, and it exits 2 with "invalid choice". Personas are
            # listed by the `feed.personas` MCP tool and by the web UI, so
            # name something that exists rather than something that reads well.
            message += (
                f"\n  (no persona named {args.user!r} -- the `feed.personas`"
                " MCP tool lists them, as does `attest serve`)"
            )
        return fail(message)

    # "leave-last-5-out" named an approach evaluate_user's own docstring says
    # it abandoned -- it is a shuffled StratifiedKFold, and for a 22-click
    # persona that is 2 folds, neither "last" nor "5".
    #
    # And the caveat warned about the wrong thing. Sample size is real, but the
    # larger limit is WHAT is measured: replacing a persona's interests with
    # unrelated text changed its top five to 1-of-5 overlap and left this
    # number bit-identical, because it scores the click classifier and not the
    # two terms that moved.
    print(
        f"click-classifier AUC: {result['auc']:.3f}"
        f"  ({result['n_splits']}-fold over {result['n_clicks']} clicks)"
    )
    print(f"  measures {result['measures']}")
    # The repo's own skew warning, routed to the number it warns about. It says
    # verbatim "rate some items from that feed as not-useful before trusting an
    # evaluation score", computes correctly, and fires at 94% on this database
    # -- and its only caller was feed.simulate_feedback, a one-shot write tool.
    # The right warning existed and never reached the score.
    if skew:
        print(f"  WARNING: {skew}")

    provenance = result.get("provenance_auc")
    if provenance is not None and provenance >= result["auc"]:
        # The number above is not measuring what it appears to. Printed
        # whenever provenance separates at least as well as usefulness, because
        # at that point the classifier cannot be shown to have learned anything
        # about this reader that it did not learn about where the labels came
        # from. Measured on the live database: 1.000 against 0.964.
        print(
            f"  WARNING: predicting where each label CAME FROM scores"
            f" {provenance:.3f} on the same data -- at or above the score above."
            " Harvested positives are items the ranker surfaced; generated"
            " negatives are sampled to be rejected. The classifier may be"
            " separating those two populations rather than useful from not."
        )
    return 0


def cmd_bootstrap_persona(args: argparse.Namespace) -> int:
    """write pseudo-clicks for a persona"""
    from attestation.db import SEED_USERS
    from attestation.embed import Embedder
    from attestation.rank import bootstrap_persona, create_user, get_user

    with open_db(args.db) as conn:
        # A new database has no personas. The demo ones exist only when asked
        # for by name, and this command is how the README asks.
        if args.name in SEED_USERS and get_user(conn, args.name) is None:
            create_user(conn, args.name, SEED_USERS[args.name])
        try:
            n = bootstrap_persona(conn, Embedder(), args.name, k=args.k)
        except ValueError as exc:
            # bootstrap_persona raises on an unknown name, and this was the one
            # user-triggerable traceback in the CLI -- on a command the README
            # recommends verbatim. Print the reason and the fix, like
            # cmd_runs_compare does with its own ValueError.
            print(exc)
            names = [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]
            print(
                f"  known personas: {', '.join(names)}"
                if names
                else "  no personas exist yet -- name a demo persona to create it:"
                f" {', '.join(SEED_USERS)}"
            )
            return 1
    print(f"wrote {n} pseudo-clicks for {args.name}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """idempotent setup + --check doctor mode"""
    import attestation.install

    return attestation.install.run_install(check=args.check, yes=args.yes, now=args.now)


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to the matching `cmd_*`, and normalise its exit code.

    A console-script entry point returning anything other than an int, or
    raising instead of returning, is invisible until something inspects the
    process exit status -- `open_db` reports a bad DB path this way (see its
    docstring), so a SystemExit it raises is converted back to a plain int
    here, which is the one place that keeps main()'s contract true for the
    tests and for anything embedding the CLI directly, not just the console
    script.
    """
    from attestation.llm import load_env

    load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as exc:
        # open_db reports an unusable database path and exits. It is a context
        # manager wrapping ten command bodies, so it cannot return an exit code
        # the way a command body does -- it raises one. Converting it back here
        # keeps main()'s contract ("returns an int") true for callers that are
        # not the console script: the tests, and anything embedding the CLI.
        return exc.code if isinstance(exc.code, int) else 1
