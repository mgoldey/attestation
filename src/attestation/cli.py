"""attest CLI: ingest, tag, serve, runs, claims, browse, kg-report, install."""

import argparse
import contextlib
import sys
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
    from attestation.db import get_db, resolve_db_path

    with contextlib.closing(get_db(resolve_db_path(db_arg))) as conn:
        yield conn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attest")
    sub = p.add_subparsers(dest="command", required=True)

    def add_db(sp):
        sp.add_argument(
            "--db",
            default=None,
            help="DB path. Resolution order if omitted: RSS_DB env var > "
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

    sp = sub.add_parser("eval", help="leave-last-N-out AUC for a user")
    add_db(sp)
    sp.add_argument("--user", required=True)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("warmup", help="pin chat + embedding models in VRAM")
    sp.set_defaults(func=cmd_warmup)

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

    native = base_url().rstrip("/").removesuffix("/v1")
    try:
        httpx.post(
            f"{native}/api/chat",
            json={
                "model": chat_model(),
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_ctx": 8192},
            },
            timeout=300,
        ).raise_for_status()
        httpx.post(
            f"{native}/api/embed",
            json={"model": embed_model(), "input": "warmup", "keep_alive": -1},
            timeout=300,
        ).raise_for_status()
    except httpx.HTTPError:
        print("warmup is Ollama-only; skipping for this backend")
        return
    print(f"models loaded and pinned (chat={chat_model()}, keep_alive=-1)")


def cmd_warmup(args: argparse.Namespace) -> int:
    import attestation.cli  # self-import so tests can monkeypatch attestation.cli.warmup

    attestation.cli.warmup()
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from attestation import claims as claims_mod
    from attestation import ledger

    target = Path(args.path).expanduser() if args.path else ledger.workspace_root()
    if target is None:
        print("pass a path, or set RESEARCH_ROOT")
        return 1
    if not target.exists():
        print(f"no such path: {target}")
        return 1

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
        out = claims_mod.check(conn, target)

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
    import shutil
    import subprocess

    from attestation.db import resolve_db_path

    datasette = shutil.which("datasette") or str(Path(sys.executable).parent / "datasette")
    if not Path(datasette).exists():
        print("datasette is not installed -- `uv sync --group dev`")
        return 1
    db_path = resolve_db_path(args.db)
    if not Path(db_path).exists():
        print(f"no database at {db_path}")
        return 1
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
    from attestation import ledger

    with open_db(args.db) as conn:
        found = ledger.list_runs(conn, args.project, args.family, args.limit)
        if not found:
            print("no runs recorded -- run `attest runs scan` first")
            return 1
        for r in found:
            fam = f"[{r['family']}]" if r["family"] else ""
            print(f"  {r['project']:20s} {r['name']:38s} {r['status']:10s} {fam}")
        print()
        for f in ledger.families(conn, args.project):
            print(f"  family {f['family']:32s} {f['n']} run(s)  ({f['project']})")
        return 0


def cmd_runs_compare(args: argparse.Namespace) -> int:
    from attestation import ledger

    with open_db(args.db) as conn:
        try:
            result = ledger.compare(conn, args.family, metric=args.metric)
        except ValueError as exc:
            print(exc)
            return 1
        if not result["arms"]:
            # say which families exist rather than dead-ending: `compare
            # <project>` is the intuitive first guess and is not a family
            print(result.get("message") or f"no runs in family {args.family!r}")
            return 1
        print(f"{result['family']} — ranked by {result['metric']} ({result['direction']})\n")
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
    from attestation import kg

    with open_db(args.db) as conn:
        for key, value in kg.health(conn).items():
            print(f"{key:24s} {value}")
        if kg.is_stale(conn):
            print("\nNOTE: stored graph is stale; `attest tag` rebuilds it")
        print()
        for i, c in enumerate(kg.communities(conn, min_size=args.min_size), 1):
            print(f"{i}. [{c['label']}] — {len(c['members'])} concepts")
            print(f"   {', '.join(c['members'])}\n")
        return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from attestation.embed import Embedder
    from attestation.ingest import run_ingest

    with open_db(args.db) as conn:
        stats = run_ingest(conn, Embedder(), args.feeds)
    print(stats)
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    import attestation.features

    with open_db(args.db) as conn:
        stats = attestation.features.run_tagging(conn, limit=args.limit)
    print(stats)
    return 1 if (stats["tagged"] == 0 and stats["failed"] > 0) else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from attestation.db import resolve_db_path
    from attestation.server import create_app

    uvicorn.run(create_app(resolve_db_path(args.db)), host="127.0.0.1", port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from attestation.rank import evaluate_user, get_user

    with open_db(args.db) as conn:
        user = get_user(conn, args.user)
        auc = evaluate_user(conn, user["id"]) if user else None
    if auc is None:
        print("insufficient click data for a meaningful holdout (need 10+ mixed clicks)")
    else:
        print(f"leave-last-5-out AUC: {auc:.3f}  (noise at small n -- not evidence)")
    return 0


def cmd_bootstrap_persona(args: argparse.Namespace) -> int:
    from attestation.embed import Embedder
    from attestation.rank import bootstrap_persona

    with open_db(args.db) as conn:
        n = bootstrap_persona(conn, Embedder(), args.name, k=args.k)
    print(f"wrote {n} pseudo-clicks for {args.name}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    import attestation.install

    return attestation.install.run_install(check=args.check, yes=args.yes, now=args.now)


def main(argv: list[str] | None = None) -> int:
    from attestation.llm import load_env

    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)
