"""Turn the existing checkers' answers into run records.

Duck-typed over `claims.Verdict`/`claims.Claim` rather than importing
`claims`, so recording a check adds no dependency to the checker and this
package stays free of `sqlite3` (the checker already ran; this only writes
down what it said).

`attest claims` runs the number check AND the citation lint, so its record
carries two check entries, `claims` and `cite.check`, kept apart for the same
reason `claims.check` keeps the verdicts apart: a claim can be contradicted
on its number and uncited on its key.
"""

from __future__ import annotations

from pathlib import Path

from attestation.run_record.build import CheckResult, Finding, RunRecorder

UNCITED = "uncited"
RESOLVED = "resolved"


def _root_of(target: Path) -> Path:
    target = Path(target).expanduser().resolve()
    return target if target.is_dir() else target.parent


def _where(rec: RunRecorder, path: str, line: int) -> str:
    try:
        rel = Path(path).expanduser().resolve().relative_to(rec.root).as_posix()
    except ValueError:
        rel = Path(path).name
    return f"{rel}:{line}"


def _claim_data(verdict) -> dict:
    claim = verdict.claim
    data = {"ref": f"{claim.project}/{claim.run}", "metric": claim.metric, "value": claim.value}
    if verdict.actual is not None:
        data["actual"] = verdict.actual
    if claim.cite:
        data["cite"] = claim.cite
    if verdict.source_path:
        data["source"] = Path(verdict.source_path).name
    return data


def _add_documents(rec: RunRecorder, target: Path, claims_) -> None:
    target = Path(target).expanduser()
    if target.is_file():
        rec.add_input(target.resolve(), "draft")
    for path in sorted({c.path for c in claims_}):
        rec.add_input(Path(path).expanduser().resolve(), "draft")


def _add_bibs(rec: RunRecorder, bib_paths) -> None:
    for bib in bib_paths or ():
        if Path(bib).is_file():
            rec.add_input(Path(bib).resolve(), "bib")


def citation_check(rec: RunRecorder, claims_, uncited) -> CheckResult:
    """Every claim with a `cite=` key: `uncited` if the lint flagged it,
    else `resolved`. A record of only the failures would hide how many
    keys were checked."""
    flagged = {id(v.claim): v for v in uncited}
    results = []
    for claim in claims_:
        if not claim.cite:
            continue
        v = flagged.get(id(claim))
        results.append(
            Finding(
                _where(rec, claim.path, claim.line),
                UNCITED if v else RESOLVED,
                v.message if v else "",
                {"cite": claim.cite},
            )
        )
    return CheckResult("cite.check", results)


def record_claims_check(
    target: Path,
    verdicts,
    malformed,
    *,
    bib_paths=(),
    started_at: str | None = None,
    tool: str = "attest claims",
) -> RunRecorder:
    """A recorder for one `claims.check` run over `target`. Every claim gets
    exactly one number verdict, so the claims are read back off those."""
    numeric = [v for v in verdicts if v.verdict != UNCITED]
    claims_ = [v.claim for v in numeric]
    rec = RunRecorder(tool, root=_root_of(target), started_at=started_at)
    _add_documents(rec, target, claims_)
    _add_results(rec, numeric)
    _add_bibs(rec, bib_paths)
    rec.add_check(claims_check(rec, numeric, malformed))
    rec.add_check(citation_check(rec, claims_, [v for v in verdicts if v.verdict == UNCITED]))
    return rec


def _add_results(rec: RunRecorder, verdicts) -> None:
    """The run artifacts the ledger read for each verdict, when still on disk."""
    for v in verdicts:
        if v.source_path and Path(v.source_path).is_file():
            rec.add_input(Path(v.source_path).resolve(), "results")


def claims_check(rec: RunRecorder, numeric, malformed) -> CheckResult:
    """The number verdicts, plus each malformed annotation as `malformed`."""
    findings = [
        Finding(_where(rec, v.claim.path, v.claim.line), str(v.verdict), v.message, _claim_data(v))
        for v in numeric
    ]
    findings += [Finding(problem, "malformed") for problem in malformed]
    return CheckResult("claims", findings)


def record_citation_check(
    target: Path,
    claims_,
    uncited,
    *,
    bib_paths=(),
    started_at: str | None = None,
) -> RunRecorder:
    """A recorder for one `cite.check` run over `target`."""
    rec = RunRecorder("cite.check", root=_root_of(target), started_at=started_at)
    _add_documents(rec, target, claims_)
    _add_bibs(rec, bib_paths)
    rec.add_check(citation_check(rec, claims_, uncited))
    return rec
