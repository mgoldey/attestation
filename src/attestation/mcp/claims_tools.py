"""The claim checker behind `runs.claims_check` and `runs.claims_coverage`.

Split out of provenance.py, which hit its size cap. It is a real seam rather
than a size dodge: the ledger reads experiment artifacts, while this reads
Markdown prose and checks its numbers against what the ledger found. They
share a workspace root and nothing else.

The tools keep the `runs.*` namespace -- a claim is verified against runs, and
a reader looking for `claims.*` would be looking for a surface that does not
exist. This is a source split, not a namespace change.
"""

from pathlib import Path

from attestation import claims, ledger
from attestation.mcp._tool import ToolError, tool
from attestation.mcp.provenance import NO_ROOT


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


def _citation_resolver():
    """The configured bibliographic sources, or None if they cannot be built.

    Imported here rather than at module scope: `claims.check` takes the
    resolver as an argument precisely so the ledger does not depend on the
    citation readers, and this wiring should not undo that at import time.

    A resolver that cannot be constructed means the lint does not run --
    the numeric verdicts, which are what this tool is chiefly for, still do.
    """
    try:
        from attestation import citations

        return citations.Resolver.from_env()
    except Exception:  # noqa: BLE001 -- an unbuildable resolver is an absent
        # lint, exactly as an absent Zotero is an absent source. Anything from
        # a missing optional dependency to an unreadable library means the same
        # thing here, and none of them should fail a claim check.
        return None


@tool(empty={"claims": [], "counts": {}, "malformed": [], "checked": []}, label="claims_check")
def _check(conn, path: str | None = None, verdict: str | None = None) -> dict:
    target = _target(path)
    # The citation lint runs alongside the numeric check: a claim whose number
    # agrees but whose `cite=` key no source has is not `supported`, and this
    # is the tool whose name says it checks claims. Claims with no `cite=` are
    # untouched -- see claims.check_citations.
    resolver = _citation_resolver()
    out = claims.check(conn, target, resolver=resolver)
    # States the pairing with cite.check as data, not only in prose: a caller
    # comparing the two tools' payloads sees the scope difference directly.
    # "citation" only appears when a resolver could be built -- an unbuildable
    # resolver means the lint did not run, so claiming it was checked would be
    # the same false-clean-bill-of-health failure cite.check's own comment
    # warns about.
    checked = ["numeric", "citation"] if resolver is not None else ["numeric"]
    rows = [
        {
            "verdict": v.verdict,
            "file": v.claim.path,
            "line": v.claim.line,
            "run": f"{v.claim.project}/{v.claim.run}",
            "metric": v.claim.metric,
            "claimed": v.claim.value,
            "actual": v.actual,
            # Present on every row so the shape does not change between
            # verdicts; it is the whole subject of an `uncited` one.
            "cite": v.claim.cite,
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
        "checked": checked,
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
