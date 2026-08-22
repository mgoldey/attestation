"""Experiment ledger and claim checking: the `runs.*` namespace.

The ledger reads artifacts already on disk -- it is deliberately not an
experiment tracker and asks for no `log_metric()` calls. See `ledger.py`'s
module docstring for why adoption cost is the design constraint.

Two rules keep it honest and both live in the domain, not here: record what is
unambiguous and refuse to guess the rest, and never rank a metric whose
direction is undeclared. An undeclared direction reaches the caller as a
ToolError with the reason spelled out, because it is caller-fixable.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from attestation import claims, ledger
from attestation.mcp._shared import MAX_LIST_LIMIT
from attestation.mcp._tool import ToolError, open_db, tool

# Argument constraints live in the SCHEMA, not just in runtime checks, so an
# MCP client rejects a bad call before it is made. limit=0 and since_days=-30
# both reached the tools and had to be refused with a message the model then
# had to read and act on -- a round trip and a failed call more expensive than
# a bound the client already knows about.
Limit = Annotated[int, Field(ge=1, le=50)]


NO_ROOT = (
    "no workspace configured -- set RESEARCH_ROOT to the directory holding your"
    " projects, or pass root explicitly"
)


def register(mcp) -> None:
    """Attach every runs.* tool to the server."""

    @mcp.tool(name="runs.scan")
    def runs_scan(
        root: str | None = None, project: str | None = None, confirm: bool = False
    ) -> dict:
        """Read experiment runs from artifacts already on disk into the ledger.

        Walks a workspace directory, treating each subdirectory as a project, and
        reads the conventions research repos already use -- `results/`, `logs/`,
        `configs/`, `outputs/`, `benchmarks/` holding JSON, JSONL, YAML or TOML.
        Nothing needs to be instrumented and no project needs to be registered.

        `root` defaults to the RESEARCH_ROOT environment variable. Requires
        `confirm=true` since it replaces each scanned project's rows. Directories
        with nothing recognisable are listed in `empty` rather than omitted, so
        "found nothing" is never mistaken for "nothing was there".

        """
        return _scan(root, project, confirm)

    @mcp.tool(name="runs.list")
    def runs_list(
        project: str | None = None,
        family: str | None = None,
        limit: Limit = 20,
    ) -> dict:
        """Experiment runs in the ledger, with the families they group into.

        A `family` is a set of sibling runs -- the arms of a sweep, or one run's
        checkpoints over training. Use it with `runs_compare` to answer which arm
        won. Also returns the family list, so you can see what is comparable.

        """
        return _list(project, family, limit)

    @mcp.tool(name="runs.compare")
    def runs_compare(family: str, metric: str | None = None) -> dict:
        """Rank the arms of an experiment family by a metric.

        The question a sweep exists to answer and that usually lives only in
        filenames: which variant won, on what metric, by how much. Omit `metric` to
        use the one most arms share.

        Refuses to rank a metric whose direction is undeclared rather than guessing
        -- ranking WER as if higher were better would name the worst arm the
        winner. Arms with no value for the metric are listed in `without_metric`
        rather than dropped: an arm that was never evaluated is a finding.

        """
        return _compare(family, metric)

    @mcp.tool(name="runs.detail")
    def runs_detail(project: str, name: str) -> dict:
        """One run in full: config shape, every metric, source path, and the
        header comment from its config if it had one.

        That header is often where the hypothesis and the single changed variable
        are written down. It is stored verbatim and never interpreted.

        """
        return _detail(project, name)

    @mcp.tool(name="runs.claims_coverage")
    def claims_coverage(path: str | None = None) -> dict:
        """Numbers asserted in Markdown that no claim annotation covers.

        The inverse of `claims_check`: that verifies the claims that exist, this
        finds assertions nobody made checkable. A document with zero contradicted
        claims looks healthy while asserting a dozen unverifiable numbers, and
        nothing else surfaces the difference.

        Only decimals count as measurements -- integers in prose are overwhelmingly
        versions, counts and dates. Versions, ISO dates, URLs, package pins and
        anything inside an HTML comment are excluded, since a comment renders as
        nothing and asserts nothing to a reader.

        """
        return _coverage(path)

    @mcp.tool(name="runs.claims_check")
    def claims_check(path: str | None = None, verdict: str | None = None) -> dict:
        """Verify numeric claims written in Markdown against runs in the ledger.

        A claim is an HTML comment beside the prose it describes, so it renders as
        nothing and the document reads exactly as before:

        <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->

        Five verdicts, and the differences matter. `supported`: a run agrees.
        `contradicted`: a run disagrees — the document or the run is wrong.
        `unsupported`: no run matches, so the claim may still be true but nothing
        backs it. `ambiguous`: a wildcard matched several runs, so which is meant is
        undecidable. `stale`: the value matches but the artifact changed after
        `as_of`, so it is worth re-verifying.

        Filter with `verdict` to answer "what in my documentation is unsupported".
        Read-only: it reports, it never edits a document.

        """
        return _check(path, verdict)


@tool(empty={"scanned": {}, "empty": [], "diagnostics": {}}, needs_db=False, label="runs_scan")
def _scan(root: str | None = None, project: str | None = None, confirm: bool = False) -> dict:
    if not confirm:
        raise ToolError(
            "refusing to scan without confirm=true. This replaces the ledger's"
            " rows for each project scanned (they are re-read from disk, so"
            " nothing unrecoverable is lost)."
        )
    target = ledger.workspace_root(root)
    if target is None:
        raise ToolError(NO_ROOT)
    with open_db() as conn:
        out = ledger.scan(conn, target, project=project)
    total = sum(out["scanned"].values())
    return {
        "message": f"{total} run(s) across {len(out['scanned'])} project(s)",
        "scanned": out["scanned"],
        "empty": out.get("empty", []),
        # why each empty project was empty: the caller is a model, and a bare
        # "0 run(s)" gives it nothing to tell the user or act on
        "diagnostics": out.get("diagnostics", {}),
    }


@tool(empty={"runs": [], "families": [], "n_families": 0}, label="runs_list")
def _list(conn, project: str | None = None, family: str | None = None, limit: int = 20) -> dict:
    found = ledger.list_runs(conn, project=project, family=family, limit=min(limit, MAX_LIST_LIMIT))
    if not found:
        raise ToolError("no runs recorded -- call runs.scan(confirm=true) first")
    families = ledger.families(conn, project=project)
    # A sample, not a listing. Fifty of 403 was 73% of this response and an
    # agent could not act on any of it -- a caller asking "which arm won?"
    # got a wall of names ahead of the runs it asked for. A handful shows the
    # SHAPE of what is comparable; the count and the narrowing hint do the
    # rest, and runs.compare's own error names the alternatives when a guess
    # misses.
    FAMILY_SAMPLE = 8
    # Cap the family list. `limit` bounds `runs` only, so a workspace with
    # hundreds of families returned all of them alongside a handful of runs --
    # the field advertised as the bridge to runs_compare was the one that blew
    # the caller's context. Truncation is reported, never silent.
    shown = families[:FAMILY_SAMPLE]
    message = f"{len(found)} run(s)"
    if len(families) > len(shown):
        message += (
            f"; showing {len(shown)} of {len(families)} families -- pass project= to narrow them"
        )
    return {
        "message": message,
        "runs": found,
        "families": shown,
        "n_families": len(families),
    }


@tool(
    empty={
        "family": None,
        "metric": None,
        "arms": [],
        "winner": None,
        # These four are as load-bearing as the ranking. SKILL.md tells an
        # agent that a comparison with no caveats has earned that silence, so
        # `caveats` missing on failure would be indistinguishable from a clean
        # comparison. Declared here so failure and success have one shape.
        "caveats": [],
        "direction": None,
        "corpus": None,
        "without_metric": [],
    },
    label="runs_compare",
)
def _compare(conn, family: str, metric: str | None = None) -> dict:
    try:
        result = ledger.compare(conn, family, metric=metric)
    except ValueError as exc:
        # an undeclared metric direction is a caller-fixable problem, so the
        # reason is surfaced rather than flattened to "internal error"
        raise ToolError(str(exc)) from exc
    if not result["arms"]:
        # ledger.compare already built the recovery message for this case --
        # it distinguishes "no such family" from "you named a project, not a
        # family" and lists what IS comparable. An earlier version of this
        # wrapper replaced it with a bare "no runs in family 'x'", throwing
        # away the one thing that tells the caller what to do next.
        raise ToolError(result.get("message") or f"no runs in family {family!r}")
    return {"message": f"{len(result['arms'])} arm(s)", **result}


@tool(empty={"run": None}, label="runs_detail")
def _detail(conn, project: str, name: str) -> dict:
    found = ledger.detail(conn, project, name)
    if found is None:
        raise ToolError(f"no run {name!r} in project {project!r}")
    return {"message": f"{len(found['metrics'])} metric row(s)", "run": found}


def _target(path: str | None) -> Path:
    """Resolve an explicit path, else the configured workspace root.

    Both failure modes are caller-fixable and get a ToolError rather than a
    logged traceback: no root configured at all, or a path that is not there.
    """
    target = Path(path).expanduser() if path else ledger.workspace_root()
    if target is None:
        raise ToolError(NO_ROOT)
    if not target.exists():
        raise ToolError(f"no such path: {target}")
    return target


@tool(empty={"claims": [], "counts": {}, "malformed": []}, label="claims_check")
def _check(conn, path: str | None = None, verdict: str | None = None) -> dict:
    target = _target(path)
    out = claims.check(conn, target)
    rows = [
        {
            "verdict": v.verdict,
            "file": v.claim.path,
            "line": v.claim.line,
            "run": f"{v.claim.project}/{v.claim.run}",
            "metric": v.claim.metric,
            "claimed": v.claim.value,
            "actual": v.actual,
            "message": v.message,
            "source_path": v.source_path,
        }
        for v in out["verdicts"]
        if verdict is None or v.verdict == verdict
    ]
    summary = ", ".join(f"{n} {k}" for k, n in sorted(out["counts"].items()))
    return {
        "message": f"{out['claims']} claim(s): {summary}" if out["claims"] else "no claims found",
        "claims": rows,
        "counts": out["counts"],
        "malformed": out["malformed"],
    }


@tool(empty={"uncovered": [], "numbers": 0}, needs_db=False, label="claims_coverage")
def _coverage(path: str | None = None) -> dict:
    target = _target(path)
    out = claims.coverage(target)
    return {
        "message": f"{out['covered']}/{out['numbers']} number(s) covered by a claim",
        "uncovered": out["uncovered"],
        "numbers": out["numbers"],
        "covered": out["covered"],
    }
