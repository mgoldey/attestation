"""attest CLI: ingest, tag, serve, runs, claims, browse, kg-report, install."""

import argparse
import contextlib
import inspect
import json
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


# Every subcommand's one-line purpose, typed ONCE. build_parser reads from
# this for `add_parser(..., help=HELP[name])`; `_documented` (below) reads
# the same entry to set each `cmd_*` handler's docstring first line, so
# `attest <cmd> --help` and `cmd_<name>.__doc__` share one string rather
# than two that can drift apart -- the failure `test_a_subcommand_help_
# describes_what_it_prints` exists to catch. `runs`'s sub-subcommands use a
# dotted key ("runs.scan") since they are not top-level attest subcommands.
HELP: dict[str, str] = {
    "ingest": "fetch feeds, embed, store",
    "tag": "LLM-tag untagged items (topic tags + content type)",
    "serve": "run the web UI",
    "eval": "cross-validated AUC for a persona's click classifier",
    "warmup": "pin chat + embedding models in VRAM",
    "reload": "restart running MCP servers so code edits take effect",
    "backup": "write a consistent copy of the database",
    "emit": "agent configs generated from the tool surfaces",
    "kg-report": "knowledge-graph health + topic clusters",
    "claims": "verify claims written in Markdown against runs",
    "browse": "open the ledger in Datasette (read-only)",
    "runs": "experiment ledger read from artifacts on disk",
    "runs.scan": "re-read runs from a workspace directory",
    "runs.list": "runs in the ledger",
    "runs.compare": "rank the arms of an experiment family",
    "runs.show": "one run in full",
    "runs.record": "write per-arm result/config files the ledger can scan",
    "bootstrap-persona": "write pseudo-clicks for a persona",
    "install": "idempotent setup + --check doctor mode",
    "library": "the deduplicated reference library (BibTeX, Zotero, feed, opt-in web)",
    "library.sync": "read every configured source into the store",
    "library.search": "search the library (semantic when embedded)",
    "library.tag": "LLM-tag untagged references",
    "library.embed": "embed references that have no vector",
    "library.status": "counts per source, vectors, tags, citation edges",
}


def _documented(name: str):
    """Decorator: set a `cmd_*` handler's docstring from `HELP[name]`.

    Applied at the def site, so the docstring exists the moment the module
    is imported -- unlike setting `__doc__` inside `build_parser`, which
    would leave it `None` until that function is first called. `HELP[name]`
    becomes the docstring's first line; any rationale already written below
    it (the function's own literal docstring, holding only the rationale
    body with no summary line -- see the functions below) is kept as the
    paragraph(s) that follow, never retyped here.
    """

    def decorate(func):
        """Apply to one `cmd_*` function: derive its `__doc__` and return it
        unchanged otherwise -- this is a docstring rewrite, not a wrapper."""
        body = inspect.cleandoc(func.__doc__ or "")
        func.__doc__ = HELP[name] + (f"\n\n{body}" if body else "")
        return func

    return decorate


def build_parser() -> argparse.ArgumentParser:
    """Assemble every `attest` subcommand, each `help=` read from `HELP`
    rather than retyped (see `HELP` and `_documented` above), so `--help`
    and `cmd_<name>.__doc__` share one string and cannot drift the way
    `eval --help` once did (see `test_a_subcommand_help_describes_
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

    sp = sub.add_parser("ingest", help=HELP["ingest"])
    add_db(sp)
    sp.add_argument("--feeds", default=_default_feeds_path())
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("tag", help=HELP["tag"])
    add_db(sp)
    sp.add_argument("--limit", type=int, default=None, help="max items to tag this run")
    sp.set_defaults(func=cmd_tag)

    sp = sub.add_parser("serve", help=HELP["serve"])
    add_db(sp)
    sp.add_argument("--port", type=int, default=8899)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("eval", help=HELP["eval"])
    add_db(sp)
    sp.add_argument("--user", required=True)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("warmup", help=HELP["warmup"])
    sp.set_defaults(func=cmd_warmup)

    sp = sub.add_parser("reload", help=HELP["reload"])
    sp.set_defaults(func=cmd_reload)

    sp = sub.add_parser("backup", help=HELP["backup"])
    add_db(sp)
    sp.add_argument("dest", help="path to write; must not already exist")
    sp.set_defaults(func=cmd_backup)

    sp = sub.add_parser("emit", help=HELP["emit"])
    sp.add_argument(
        "--write", action="store_true", help="write the Claude agent files (default: report only)"
    )
    sp.set_defaults(func=cmd_emit)

    sp = sub.add_parser("kg-report", help=HELP["kg-report"])
    add_db(sp)
    sp.add_argument("--min-size", type=int, default=3, help="smallest cluster to list")
    sp.set_defaults(func=cmd_kg_report)

    sp = sub.add_parser("claims", help=HELP["claims"])
    add_db(sp)
    sp.add_argument("path", nargs="?", help="file or directory (default: $RESEARCH_ROOT)")
    sp.add_argument("--verdict", help="show only this verdict")
    sp.add_argument(
        "--coverage",
        action="store_true",
        help="instead: list numbers in prose that no claim covers",
    )
    sp.set_defaults(func=cmd_claims)

    sp = sub.add_parser("browse", help=HELP["browse"])
    add_db(sp)
    sp.add_argument("--port", type=int, default=8898)
    sp.add_argument("--open", action="store_true", help="open a browser window")
    sp.set_defaults(func=cmd_browse)

    sp = sub.add_parser("runs", help=HELP["runs"])
    add_db(sp)
    runs_sub = sp.add_subparsers(dest="runs_command", required=True)

    rp = runs_sub.add_parser("scan", help=HELP["runs.scan"])
    rp.add_argument("--root", help="workspace dir (default: $RESEARCH_ROOT)")
    rp.add_argument("--project", help="scan only this project")
    rp.set_defaults(func=cmd_runs_scan)

    rp = runs_sub.add_parser("list", help=HELP["runs.list"])
    rp.add_argument("--project")
    rp.add_argument("--family")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_runs_list)

    rp = runs_sub.add_parser("compare", help=HELP["runs.compare"])
    rp.add_argument("family")
    rp.add_argument("--metric", help="default: the metric most arms share")
    rp.add_argument("--project", help="required when the family exists in more than one project")
    rp.set_defaults(func=cmd_runs_compare)

    rp = runs_sub.add_parser("show", help=HELP["runs.show"])
    rp.add_argument("project")
    rp.add_argument("name")
    rp.set_defaults(func=cmd_runs_show)

    rp = runs_sub.add_parser("record", help=HELP["runs.record"])
    rp.add_argument("family")
    rp.add_argument(
        "--arm",
        dest="arms",
        action="append",
        nargs="+",
        metavar=("NAME", "METRIC=VALUE"),
        required=True,
        help="one arm: its name, then one or more METRIC=VALUE pairs",
    )
    rp.add_argument("--corpus", help="corpus name; declares it in corpora.toml")
    rp.add_argument(
        "--direction",
        dest="directions",
        action="append",
        default=[],
        metavar="METRIC=lower_is_better|higher_is_better",
        help="declare a metric not already in ledger.METRIC_DIRECTION",
    )
    rp.add_argument(
        "--config",
        dest="config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra provenance pair written into each arm's config file",
    )
    rp.add_argument("--root", default=".", help="where to write files (default: cwd)")
    rp.add_argument("--dry-run", action="store_true", help="print the manifest, write nothing")
    rp.add_argument("--force", action="store_true", help="overwrite existing files/entries")
    rp.add_argument(
        "--scan", action="store_true", help="also run `runs scan` and print `runs compare`"
    )
    rp.set_defaults(func=cmd_runs_record)

    sp = sub.add_parser("library", help=HELP["library"])
    add_db(sp)
    lib_sub = sp.add_subparsers(dest="library_command", required=True)

    lp = lib_sub.add_parser("sync", help=HELP["library.sync"])
    lp.add_argument(
        "--sources", help="comma-separated subset: bibtex,zotero,feed,arxiv,crossref,s2"
    )
    lp.add_argument("--limit", type=int, help="max rows per enricher and per embed pass")
    lp.set_defaults(func=cmd_library_sync)

    lp = lib_sub.add_parser("search", help=HELP["library.search"])
    lp.add_argument("query", nargs="?", default="")
    lp.add_argument("--author", help="surname filter")
    lp.add_argument("--year", type=int)
    lp.add_argument("--tag")
    lp.add_argument("--limit", type=int, default=10)
    lp.set_defaults(func=cmd_library_search)

    lp = lib_sub.add_parser("tag", help=HELP["library.tag"])
    lp.add_argument("--limit", type=int)
    lp.set_defaults(func=cmd_library_tag)

    lp = lib_sub.add_parser("embed", help=HELP["library.embed"])
    lp.add_argument("--limit", type=int)
    lp.set_defaults(func=cmd_library_embed)

    lp = lib_sub.add_parser("status", help=HELP["library.status"])
    lp.set_defaults(func=cmd_library_status)

    sp = sub.add_parser("bootstrap-persona", help=HELP["bootstrap-persona"])
    add_db(sp)
    sp.add_argument("name")
    sp.add_argument("-k", type=int, default=30)
    sp.set_defaults(func=cmd_bootstrap_persona)

    sp = sub.add_parser("install", help=HELP["install"])
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


@_documented("backup")
def cmd_backup(args: argparse.Namespace) -> int:
    """
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


@_documented("emit")
def cmd_emit(args: argparse.Namespace) -> int:
    """
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


@_documented("reload")
def cmd_reload(args: argparse.Namespace) -> int:
    """
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


@_documented("warmup")
def cmd_warmup(args: argparse.Namespace) -> int:
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


@_documented("claims")
def cmd_claims(args: argparse.Namespace) -> int:
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


@_documented("browse")
def cmd_browse(args: argparse.Namespace) -> int:
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


@_documented("runs.scan")
def cmd_runs_scan(args: argparse.Namespace) -> int:
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


@_documented("runs.list")
def cmd_runs_list(args: argparse.Namespace) -> int:
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


def _print_compare(result: dict) -> None:
    """Render one `ledger.compare()` result the way `runs compare` always
    has -- factored out so `runs record --scan` can print the identical
    table for the family it just wrote, rather than a second rendering that
    could drift from this one."""
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


@_documented("runs.compare")
def cmd_runs_compare(args: argparse.Namespace) -> int:
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
        _print_compare(result)
        return 0


@_documented("runs.show")
def cmd_runs_show(args: argparse.Namespace) -> int:
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


def _parse_kv_pairs(pairs: list[str], *, label: str) -> dict[str, str]:
    """`["k=v", ...]` to `{k: v}`, raising `ValueError` naming `label` on a
    pair missing `=` -- shared by `--direction` and `--config`, both of
    which take the same `KEY=VALUE` shape on the command line."""
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"{label} {pair!r} must be KEY=VALUE")
        out[key] = value
    return out


def _parse_arms(raw_arms: list[list[str]]) -> dict[str, dict[str, float]]:
    """argparse's `--arm` groups (`[[name, "m1=v1", "m2=v2"], ...]`) to
    `{name: {metric: value}}`, validating each metric name and parsing each
    value to `float` -- the one place `runs record` turns argv into the
    plain data `record.plan`/`record.undeclared` take. Raises `ValueError`
    (metric name, value, or duplicate arm) rather than returning a partial
    result, so the caller can refuse before writing anything.
    """
    from attestation import record

    arms: dict[str, dict[str, float]] = {}
    for group in raw_arms:
        if not group:
            raise ValueError("--arm needs a name and at least one METRIC=VALUE")
        name, *pairs = group
        record.validate_name(name, label="--arm name")
        if name in arms:
            raise ValueError(f"--arm {name!r} given more than once")
        if not pairs:
            raise ValueError(f"--arm {name!r} has no METRIC=VALUE pairs")
        metrics: dict[str, float] = {}
        for pair in pairs:
            metric, sep, value = pair.partition("=")
            if not sep:
                raise ValueError(f"--arm {name!r}: {pair!r} must be METRIC=VALUE")
            record.validate_metric_name(metric)
            metrics[metric] = record.parse_metric_value(value)
        arms[name] = metrics
    return arms


def _parse_record_args(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """`(arms, declared_directions, config)` from `args`, or raise
    `ValueError` naming the first bad `--arm`/`--direction`/`--config`.
    Split out of `cmd_runs_record` so that function's own branching stays
    about ORCHESTRATION (parse, refuse, plan, write, scan) rather than
    argv shape."""
    from attestation import record

    record.validate_name(args.family, label="family")
    if args.corpus is not None:
        record.validate_name(args.corpus, label="--corpus")
    arms = _parse_arms(args.arms)
    declared = _parse_kv_pairs(args.directions, label="--direction")
    for metric, direction in declared.items():
        record.validate_metric_name(metric)
        record.validate_direction(metric, direction)
    config = _parse_kv_pairs(args.config, label="--config")
    return arms, declared, config


def _direction_conflict_detail(overridden: dict[str, tuple[str, str]]) -> str:
    """`[metric_direction] entr{y,ies}: m: 'old' -> 'new', ...` -- the one
    phrase both the refusal message and the --force override message build
    from, so the two read as the same finding stated two ways rather than
    two hand-written phrasings that could drift."""
    detail = ", ".join(f"{m}: {old!r} -> {new!r}" for m, (old, new) in overridden.items())
    plural = "y" if len(overridden) == 1 else "ies"
    return f"[metric_direction] entr{plural}: {detail}"


def _direction_conflict_message(overridden: dict[str, tuple[str, str]]) -> str:
    """The refusal `cmd_runs_record` prints when `--force` is not given."""
    detail = _direction_conflict_detail(overridden)
    return f"refusing to overwrite an existing {detail} without --force"


def _apply_direction_override(manifest: dict[str, str], declared: dict[str, str]) -> dict[str, str]:
    """`manifest` with its `metric_direction.toml` entry replaced by one
    covering ALL of `declared` (redundant, non-conflicting declarations
    included) -- `plan()`'s own redundancy elision only sees "already
    known", not "already known and a --force override was requested", so
    `plan()` alone would leave `--force` with nothing new to write."""
    return {**manifest, "metric_direction.toml": _metric_direction_toml_fragment(declared)}


def _metric_direction_toml_fragment(overridden: dict[str, str]) -> str:
    """`[metric_direction]\nkey = "value"\n...` for exactly the metrics
    `--force` is overriding -- the same fresh-file shape `plan()` builds,
    parseable by `record.toml_tables()` and mergeable by `record.
    merge_toml_table(force=True)`, which is what actually overwrites the
    file's differing value; this only supplies the entry `plan()` itself
    omits (its own redundancy elision only sees "already known", not
    "already known and a --force override was requested"), so `--force`
    has something to write at all.
    """
    lines = ["[metric_direction]"]
    for metric, direction in overridden.items():
        lines.append(f'{metric} = "{direction}"')
    return "\n".join(lines) + "\n"


def _toml_target(relpath: str, root: Path):
    """Where one manifest TOML entry actually gets written.
    `metric_direction.toml` is NOT root-relative: the ledger always reads it
    from `ledger._metric_direction_path()` (the LEDGER_METRIC_DIRECTION_FILE
    env var, else ~/.hermes/), same as `runs.compare`'s own refusal names --
    writing it under `root` instead would declare a direction `compare`
    never looks at. `corpora.toml` is the one genuinely root-relative merge
    target the spec names ("corpora.toml at the root")."""
    from attestation import ledger

    return ledger._metric_direction_path() if relpath == "metric_direction.toml" else root / relpath


def _merge_toml_files(toml_files: dict[str, str], root: Path, *, force: bool) -> dict[Path, str]:
    """Every manifest TOML entry, merged (never yet written) into whatever
    text already exists at its real target (see `_toml_target`) --
    computed ENTIRELY IN MEMORY so a conflict (`record.merge_toml_table`'s
    `ValueError` without `--force`) is raised before any file this call
    touches is written, matching `record.write`'s own "refuse before
    writing anything" guarantee for the per-arm results/configs files.
    Split out of the old `_write_toml_files` for exactly that reordering --
    `cmd_runs_record` used to call `record.write` (which already had this
    property) and then this function (which did not: it merged AND wrote
    in the same loop, so a second TOML table's conflict crashed after the
    first table's merge, and after the per-arm files, had already reached
    disk).

    Each entry holds exactly one declared table's worth of new entries
    (parsed back out of the fresh-file content `plan()` built):
    `metric_direction.toml` has one flat `[metric_direction]` table;
    `corpora.toml` has two, `[corpus.<name>]` and `[assign.family]`, merged
    in turn so neither clobbers the other.
    """
    from attestation import record

    merged: dict[Path, str] = {}
    for relpath, content in toml_files.items():
        path = _toml_target(relpath, root)
        existing_text = path.read_text() if path.exists() else ""
        for table, entries in record.toml_tables(content):
            existing_text = record.merge_toml_table(existing_text, table, entries, force=force)
        merged[path] = existing_text
    return merged


def _write_merged_toml(merged: dict[Path, str]) -> list[Path]:
    """Write every already-merged TOML text (see `_merge_toml_files`) and
    return the paths written -- the only I/O half of the old
    `_write_toml_files`."""
    written = []
    for path, text in merged.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written


def _run_record_scan(args: argparse.Namespace, root: Path) -> int:
    """The `--scan` half of `runs record`: fold `runs scan` + `runs compare`
    for the just-written family into the same invocation."""
    from attestation import ledger

    with open_db(args.db) as conn:
        scan_out = ledger.scan(conn, root, project=None)
        if scan_out.get("message"):
            print(scan_out["message"])
            return 1
        for name, count in sorted(scan_out["scanned"].items()):
            print(f"  {name:28s} {count} run(s)")
        try:
            result = ledger.compare(conn, args.family)
        except ValueError as exc:
            return fail(str(exc))
        if result["arms"]:
            _print_compare(result)
    return 0


def _plan_with_direction_check(
    family: str,
    arms: dict[str, dict[str, float]],
    corpus: str | None,
    declared: dict[str, str],
    config: dict[str, str],
    known: dict[str, str],
    *,
    force: bool,
) -> tuple[dict[str, str], dict[str, tuple[str, str]], str | None]:
    """`(manifest, overridden, error)` -- `cmd_runs_record`'s own direction-
    conflict branch, split out to keep that function's complexity down to
    ORCHESTRATION (parse, refuse, build, write, scan). `error` is set
    (manifest empty) exactly when a differing declared direction was found
    and `force` is false; otherwise `manifest` is `record.plan`'s result,
    with `_apply_direction_override` layered on top when `force` did
    override something.
    """
    from attestation import record

    overridden = record.differing_directions(declared, known)
    if overridden and not force:
        return {}, overridden, _direction_conflict_message(overridden)

    manifest = record.plan(
        family,
        arms,
        corpus=corpus,
        directions=declared,
        config=config,
        known_directions=known,
    )
    if overridden:
        manifest = _apply_direction_override(manifest, declared)
    return manifest, overridden, None


def _write_record_manifest(
    manifest: dict[str, str], root: Path, *, force: bool
) -> tuple[list[Path], str | None]:
    """`(written, error)` -- writes `manifest` under `root` and returns the
    paths written, or an empty list and an error message on refusal.

    TOML merges are computed BEFORE any write -- `record.write`'s own
    per-arm collision check already refuses before writing anything; a
    differing TOML entry (no `--force`) must refuse just as cleanly, not
    crash with a raw traceback after the per-arm files already landed on
    disk (found in review: assigning family `asr` to a second corpus
    without `--force` wrote the second call's results/config files, THEN
    raised an uncaught `ValueError`).
    """
    from attestation import record

    per_arm = {k: v for k, v in manifest.items() if not k.endswith(".toml")}
    toml_files = {k: v for k, v in manifest.items() if k.endswith(".toml")}

    try:
        merged_toml = _merge_toml_files(toml_files, root, force=force)
    except ValueError as exc:
        return [], str(exc)

    try:
        written = record.write(root, per_arm, force=force)
    except FileExistsError as exc:
        return [], str(exc)
    written += _write_merged_toml(merged_toml)
    return written, None


@_documented("runs.record")
def cmd_runs_record(args: argparse.Namespace) -> int:
    """
    New files only (results/configs refuse on an existing target unless
    `--force`); the direction and corpus files always merge, keeping every
    foreign entry, refusing to clobber a differing value without `--force`.
    `--dry-run` prints the manifest -- `{"files": {relpath: content}}` -- and
    writes nothing; `evals/run_record_eval.py --command` drives this same
    path to score the planner deterministically, no model involved.
    """
    from attestation import ledger, record

    # Resolved to an absolute path: `--scan` derives the project name from
    # `root.name`, and an unresolved "." has none -- every recorded run
    # would land under project "" instead of the directory's real name.
    root = Path(args.root).expanduser().resolve()

    try:
        arms, declared, config = _parse_record_args(args)
    except ValueError as exc:
        return fail(str(exc))

    known = ledger.metric_directions()
    missing = record.undeclared(arms, {**known, **declared})
    if missing:
        # Same sentence `runs.compare` prints for one named metric, so an
        # agent following this refusal and one following `compare`'s learn
        # the identical remedy rather than two phrasings of the same rule.
        return fail("\n".join(ledger.unknown_direction_message(m) for m in missing))

    manifest, overridden, error = _plan_with_direction_check(
        args.family, arms, args.corpus, declared, config, known, force=args.force
    )
    if error:
        return fail(error)

    if args.dry_run:
        print(json.dumps({"files": manifest}, indent=2, sort_keys=True))
        return 0

    written, error = _write_record_manifest(manifest, root, force=args.force)
    if error:
        return fail(error)

    for path in written:
        print(f"wrote {path}")
    if overridden:
        # --force was used to change an already-declared direction: say so
        # explicitly, not just "wrote metric_direction.toml" -- a silent
        # override is the exact failure this whole check exists to prevent.
        print(f"overrode existing {_direction_conflict_detail(overridden)} (--force)")

    return _run_record_scan(args, root) if args.scan else 0


def _embedder_or_none():
    """An Embedder when the model server answers, else None (fielded search).

    One cheap probe rather than letting every row's embed call fail: the
    library's search and sync both degrade cleanly without an embedder, and a
    dead server should cost one round trip, not one per reference.
    """
    from attestation.embed import Embedder
    from attestation.llm import base_url

    try:
        httpx.get(f"{base_url().rstrip('/')}/models", timeout=2.0)
    except httpx.HTTPError:
        return None
    return Embedder()


@_documented("library.sync")
def cmd_library_sync(args: argparse.Namespace) -> int:
    from attestation import library, library_readers

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    with open_db(args.db) as conn:
        readers = library_readers.readers_from_env(conn, sources=sources)
        report = library.sync(conn, readers, embedder=_embedder_or_none(), limit=args.limit)
    for name, b in report.sources.items():
        line = f"{name}: +{b['added']} added, {b['merged']} merged, {b['unchanged']} unchanged"
        if b["enriched"]:
            line += f", {b['enriched']} enriched"
        if b["failed"]:
            line += f", {b['failed']} failed"
        print(line)
    tail = f" ({report.embed_error})" if report.embed_error else ""
    print(f"embedded {report.embedded}, {report.unembedded} without a vector{tail}")
    if report.conflicts:
        print(
            f"{report.conflicts} field conflict(s) recorded; first: {report.conflict_samples[:5]}"
        )
    return 0


@_documented("library.search")
def cmd_library_search(args: argparse.Namespace) -> int:
    from attestation import library

    with open_db(args.db) as conn:
        res = library.search(
            conn,
            args.query,
            embedder=_embedder_or_none(),
            author=args.author,
            year=args.year,
            tag=args.tag,
            limit=args.limit,
        )
    for h in res.hits:
        sim = f" {h.similarity:.3f}" if h.similarity is not None else ""
        print(f"{h.year or '----'}{sim}  {h.title[:90]}  [{h.bib_key or h.identity}]")
    print(
        f"{res.n_matches} match(es); " + ("semantic" if res.semantic else res.caveat or "fielded")
    )
    return 0


@_documented("library.tag")
def cmd_library_tag(args: argparse.Namespace) -> int:
    from attestation.features import run_reference_tagging
    from attestation.llm import base_url, chat_model, default_chat_fn

    with open_db(args.db) as conn:
        stats = run_reference_tagging(conn, default_chat_fn, chat_model(), limit=args.limit)
    print(stats)
    if stats.get("chat_down"):
        print(f"chat model unreachable at {base_url()} -- is ollama running?", file=sys.stderr)
        return 1
    return 1 if (stats["tagged"] == 0 and stats["failed"] > 0) else 0


@_documented("library.embed")
def cmd_library_embed(args: argparse.Namespace) -> int:
    from attestation import library
    from attestation.embed import Embedder

    with open_db(args.db) as conn:
        done, missing, error = library.embed_missing(conn, Embedder(), args.limit)
    print(f"embedded {done}, {missing} still without a vector" + (f" ({error})" if error else ""))
    return 1 if error and done == 0 else 0


@_documented("library.status")
def cmd_library_status(args: argparse.Namespace) -> int:
    from attestation import library

    with open_db(args.db) as conn:
        print(json.dumps(library.status(conn), indent=2))
    return 0


@_documented("kg-report")
def cmd_kg_report(args: argparse.Namespace) -> int:
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


@_documented("ingest")
def cmd_ingest(args: argparse.Namespace) -> int:
    from attestation.embed import Embedder
    from attestation.ingest import run_ingest

    with open_db(args.db) as conn:
        stats = run_ingest(conn, Embedder(), args.feeds)
    print(stats)
    return 0


@_documented("tag")
def cmd_tag(args: argparse.Namespace) -> int:
    import attestation.features
    from attestation.llm import base_url, chat_model, default_chat_fn

    with open_db(args.db) as conn:
        stats = attestation.features.run_tagging(
            conn, default_chat_fn, chat_model(), limit=args.limit
        )
    print(stats)
    if stats.get("chat_down"):
        # Same diagnosis ingest gives for the same condition (is ollama
        # running? run --check): this composition root may import llm and
        # print the resolved URL; ingest.py is a domain module and may not,
        # so its sibling message names LLM_BASE_URL instead. The stats dict
        # alone said `failed: 2` and left the cause -- Ollama is not running
        # -- to be guessed.
        print(
            f"chat model unreachable at {base_url()} -- is ollama running?"
            " (`attest install --check` diagnoses this)",
            file=sys.stderr,
        )
        return 1
    return 1 if (stats["tagged"] == 0 and stats["failed"] > 0) else 0


@_documented("serve")
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from attestation.db import resolve_db_path
    from attestation.server import create_app

    uvicorn.run(create_app(resolve_db_path(args.db)), host="127.0.0.1", port=args.port)
    return 0


@_documented("eval")
def cmd_eval(args: argparse.Namespace) -> int:
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
            # listed by calling `feed.persona_status` with no `user` argument
            # (folded from the retired feed.personas tool, which this message
            # used to name) and by the web UI, so name something that exists
            # rather than something that reads well.
            message += (
                f"\n  (no persona named {args.user!r} -- call"
                " `feed.persona_status` with no user to list them, as does"
                " `attest serve`)"
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


@_documented("bootstrap-persona")
def cmd_bootstrap_persona(args: argparse.Namespace) -> int:
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


@_documented("install")
def cmd_install(args: argparse.Namespace) -> int:
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
