"""Check assertions written in Markdown against runs in the ledger.

A README says "MAE 0.353 eV vs experiment @ aDZ". That number was transcribed
by hand from an artifact, and nothing checks it: re-run the benchmark, and the
document keeps asserting 0.353 forever. This closes that loop.

The format is an HTML comment beside the prose it describes:

    The cut leaves WER essentially unchanged (**0.053 vs. 0.043** baseline).
    <!-- claim: ablation/whisper-small-ablated metric=wer value=0.053 tol=0.001 -->

An HTML comment because it is the only annotation that is invisible in every
Markdown renderer, plain text so grep and git diff work on it, and adjacent to
the claim rather than in a separate file that drifts. The prose is never
touched; a claim is added *beside* an assertion that already exists, so
annotating is incremental -- one claim is useful on its own.

Five verdicts, and the distinctions between them are the point:

    supported     a run matches, within tolerance
    contradicted  a run matches the reference but DISAGREES on the value
    unsupported   no run matches -- the claim may be true, nothing backs it
    ambiguous     the reference matched several runs; which is meant is unknown
    stale         the artifact changed after the claim was last verified

`unsupported` and `contradicted` must never collapse together: one means "I
never recorded this", the other means "this is wrong", and the response differs.
`ambiguous` exists because silently taking the first of several matches is how
a checker reports a confident wrong answer.
"""

import datetime as _dt
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->
CLAIM_RE = re.compile(r"<!--\s*claim:\s*(?P<body>.*?)\s*-->", re.DOTALL)
_REF_RE = re.compile(r"^(?P<project>[^/\s]+)/(?P<run>\S+)")
_FIELD_RE = re.compile(r"(\w+)=([^\s]+)")

# Effectively exact. A claim whose number was rounded for the prose must say so
# with tol=, which keeps that judgement with the person who knows how the
# number was produced.
DEFAULT_TOL = 1e-9


@dataclass
class Claim:
    path: str
    line: int
    project: str
    run: str
    metric: str
    value: float
    tol: float = DEFAULT_TOL
    as_of: str | None = None
    split: str | None = None
    step: int | None = None
    raw: str = ""


class VerdictKind(StrEnum):
    """The closed set of claim-check outcomes.

    A StrEnum member IS a str -- every existing `verdict.verdict == "supported"`
    comparison, `Counter` tally, and `dict(...) == {"supported": 1}` assertion
    keeps working untouched -- but a misspelled verdict at a construction site
    is now a `ty` error instead of a silently-tallied new category in `counts`.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"


@dataclass
class Verdict:
    claim: Claim
    verdict: VerdictKind
    message: str
    actual: float | None = None
    matched: list[str] | None = None
    source_path: str | None = None
    split: str | None = None
    step: int | None = None


def parse_file(path: Path) -> tuple[list[Claim], list[str]]:
    """Claims in one Markdown file, plus complaints about malformed ones.

    A malformed annotation is reported, never skipped: silent skipping is how a
    claim disappears from review without anyone deciding to remove it.
    """
    claims: list[Claim] = []
    problems: list[str] = []
    text = path.read_text(errors="replace")

    for match in CLAIM_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        body = match.group("body").strip()
        where = f"{path}:{line_no}"

        ref = _REF_RE.match(body)
        if not ref:
            problems.append(f"{where}: expected 'project/run', got {body[:60]!r}")
            continue

        fields = dict(_FIELD_RE.findall(body[ref.end() :]))
        missing = [k for k in ("metric", "value") if k not in fields]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue

        try:
            value = float(fields["value"])
            tol = float(fields.get("tol", DEFAULT_TOL))
            step = int(fields["step"]) if "step" in fields else None
        except ValueError:
            problems.append(f"{where}: value/tol/step must be numbers, got {body[:60]!r}")
            continue

        claims.append(
            Claim(
                path=str(path),
                line=line_no,
                project=ref.group("project"),
                run=ref.group("run"),
                metric=fields["metric"].lower(),
                value=value,
                tol=tol,
                as_of=fields.get("as_of"),
                split=fields.get("split"),
                step=step,
                raw=body,
            )
        )
    return claims, problems


def find_claims(root: Path) -> tuple[list[Claim], list[str]]:
    """Every claim in every Markdown file under `root`."""
    root = Path(root).expanduser()
    if root.is_file():
        return parse_file(root)

    claims: list[Claim] = []
    problems: list[str] = []
    for md in sorted(root.rglob("*.md")):
        if any(part.startswith(".") or part == "node_modules" for part in md.parts):
            continue
        found, bad = parse_file(md)
        claims.extend(found)
        problems.extend(bad)
    return claims, problems


def _matching_runs(conn: sqlite3.Connection, claim: Claim) -> list[sqlite3.Row]:
    if claim.run.endswith("*"):
        pattern = claim.run[:-1] + "%"
        sql = (
            "SELECT id, name, source_path FROM runs WHERE project = ? AND name LIKE ? ORDER BY name"
        )
        params = (claim.project, pattern)
    else:
        sql = "SELECT id, name, source_path FROM runs WHERE project = ? AND name = ?"
        params = (claim.project, claim.run)
    return conn.execute(sql, params).fetchall()


def _is_stale(claim: Claim, source_path: str | None) -> bool:
    """Has the evidence moved since someone last looked?

    mtime rather than a content hash: hashing means storing a hash per claim,
    which is state that must itself be kept fresh. mtime answers the only
    question actually asked, and a claim with no as_of is simply never stale.
    """
    if not claim.as_of or not source_path:
        return False
    artifact = Path(source_path)
    if not artifact.exists():
        return False
    try:
        as_of = _dt.date.fromisoformat(claim.as_of)
    except ValueError:
        return False
    changed = _dt.date.fromtimestamp(artifact.stat().st_mtime)
    return changed > as_of


def check_claim(conn: sqlite3.Connection, claim: Claim) -> Verdict:
    runs = _matching_runs(conn, claim)
    if not runs:
        return Verdict(
            claim,
            VerdictKind.UNSUPPORTED,
            f"no run {claim.project}/{claim.run} in the ledger"
            " -- the claim may be true, but nothing here backs it",
        )
    if len(runs) > 1:
        names = [r["name"] for r in runs]
        return Verdict(
            claim,
            VerdictKind.AMBIGUOUS,
            f"{len(runs)} runs match {claim.run!r}; which one is meant is undecidable",
            matched=names,
        )

    run = runs[0]
    sql = "SELECT value, step, split FROM run_metrics WHERE run_id = ? AND metric = ?"
    params: list = [run["id"], claim.metric]
    if claim.split is not None:
        sql += " AND split = ?"
        params.append(claim.split)
    if claim.step is not None:
        sql += " AND step = ?"
        params.append(claim.step)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return Verdict(
            claim,
            VerdictKind.UNSUPPORTED,
            f"run {run['name']} records no metric {claim.metric!r}",
            source_path=run["source_path"],
        )

    # A run may hold the same metric at several steps or splits -- a train
    # split, a baseline arm, a mid-training checkpoint. Picking the row
    # closest to the claimed value would make the check self-fulfilling: the
    # claim would select its own evidence. Without a disambiguator, more than
    # one row is undecidable -- exactly what `ambiguous` means -- not a cue to
    # guess. Distance-to-claim is legitimate only once exactly one row remains.
    if len(rows) > 1:
        available = sorted(
            {f"split={r['split']}" if r["split"] else f"step={r['step']}" for r in rows}
        )
        return Verdict(
            claim,
            VerdictKind.AMBIGUOUS,
            f"{len(rows)} rows for {claim.metric!r} in {run['name']}; "
            f"add split=/step= to disambiguate ({', '.join(available)})",
            matched=available,
            source_path=run["source_path"],
        )

    row = rows[0]
    actual = row["value"]
    where = f" [split={row['split']}]" if row["split"] else ""
    where += f" [step={row['step']}]" if row["step"] is not None else ""

    if abs(actual - claim.value) > claim.tol:
        return Verdict(
            claim,
            VerdictKind.CONTRADICTED,
            f"document says {claim.value:g}, run records {actual:g}{where} "
            f"(tolerance {claim.tol:g})",
            actual=actual,
            source_path=run["source_path"],
            split=row["split"],
            step=row["step"],
        )

    if not Path(run["source_path"]).exists():
        return Verdict(
            claim,
            VerdictKind.STALE,
            f"value matches, but evidence file {run['source_path']} no longer exists -- re-verify",
            actual=actual,
            source_path=run["source_path"],
            split=row["split"],
            step=row["step"],
        )

    if _is_stale(claim, run["source_path"]):
        return Verdict(
            claim,
            VerdictKind.STALE,
            f"value still matches, but {Path(run['source_path']).name} changed"
            f" after as_of={claim.as_of} -- re-verify",
            actual=actual,
            source_path=run["source_path"],
            split=row["split"],
            step=row["step"],
        )

    return Verdict(
        claim,
        VerdictKind.SUPPORTED,
        f"{claim.metric}={actual:g} in {run['name']}{where}",
        actual=actual,
        source_path=run["source_path"],
        split=row["split"],
        step=row["step"],
    )


# A measurement is written as a decimal. Integers in prose are overwhelmingly
# versions, counts, dates and identifiers -- on a real index, 212 numbers reduce
# to 30 decimals, and the decimals are the results. Requiring a decimal point is
# a crude filter that happens to match how people write measured quantities.
# The trailing guard rejects only a further DIGIT. Earlier versions also
# rejected a following period, which silently dropped every number that ended a
# sentence -- in prose, most of them. Version strings are excluded by
# _NOT_A_MEASUREMENT instead, where the intent is explicit.
#
# Two branches, and the ORDER is load-bearing: grouped-thousands must be tried
# first, or "1,234.5" matches the second branch as bare "234.5" -- a corrupted
# value, not a missed one. The second branch also accepts a scientific
# exponent, since the claim parser reads `value=3.2e-4` with bare float() and
# a prose scan that reads the same number as 3.2 can never agree with it.
_NUMBER_RE = re.compile(
    r"(?<![\w.])("
    r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|[-−]?\d+\.\d+(?:[eE][+-]?\d+)?"
    r")(?!\d)"
)

# Contexts where a decimal is structural rather than a result.
_NOT_A_MEASUREMENT = (
    re.compile(r"\bv?\d+\.\d+\.\d+"),  # semver / dotted versions
    re.compile(r"https?://\S*\d+\.\d+"),  # URLs
    re.compile(r"\d{4}-\d{2}-\d{2}"),  # ISO dates
    # Package pins and dotted paths only -- NOT all code spans. An earlier
    # version masked any backticked run containing a decimal, which on a
    # table-heavy document matched whole rows and hid every real number.
    re.compile(r"`[^`\s]*[a-zA-Z_][^`\s]*\d+\.\d+[^`\s]*`"),
)


def _masked_prose(text: str) -> str:
    """Blank out claim annotations and structural numbers, keeping offsets.

    Replacing with spaces rather than deleting keeps every surviving match's
    index valid, so reported line numbers stay true to the file.
    """

    def blank(m):
        return " " * len(m.group(0))

    # ALL HTML comments, not just claim annotations: a comment renders as
    # nothing, so nothing in one is an assertion the reader ever sees. Masking
    # only `<!-- claim: -->` reported a note explaining why a number was left
    # unannotated as itself an unannotated number.
    text = re.sub(r"<!--.*?-->", blank, text, flags=re.DOTALL)
    for pattern in _NOT_A_MEASUREMENT:
        text = pattern.sub(blank, text)
    return text


def coverage(root: Path) -> dict:
    """Numbers in prose that no claim annotation covers.

    The inverse of `check`: that verifies claims that exist, this finds
    assertions that were never made checkable. A document with zero
    contradicted claims looks healthy while asserting a dozen unverifiable
    numbers, and nothing surfaces the difference.

    Reports a decimal as uncovered when no claim in the same file states that
    value (within its own tolerance). Deliberately line-agnostic: a claim
    annotation usually sits a line or two from the prose it describes, and
    requiring adjacency would produce false alarms for correctly-annotated
    documents.
    """
    root = Path(root).expanduser()
    files = [root] if root.is_file() else sorted(root.rglob("*.md"))
    out: list[dict] = []
    total_numbers = 0

    for md in files:
        if any(p.startswith(".") or p == "node_modules" for p in md.parts):
            continue
        text = md.read_text(errors="replace")
        claims, _ = parse_file(md)
        covered = [(c.value, c.tol) for c in claims]
        prose = _masked_prose(text)

        for match in _NUMBER_RE.finditer(prose):
            value = float(match.group(1).replace("−", "-").replace(",", ""))
            total_numbers += 1
            if any(abs(value - cv) <= max(ctol, 1e-9) for cv, ctol in covered):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            line = text.splitlines()[line_no - 1] if line_no <= text.count("\n") + 1 else ""
            out.append(
                {
                    "file": str(md),
                    "line": line_no,
                    "value": value,
                    "context": line.strip()[:120],
                }
            )

    return {
        "numbers": total_numbers,
        "uncovered": out,
        "covered": total_numbers - len(out),
        "files": len(files),
    }


def check(conn: sqlite3.Connection, root: Path) -> dict:
    """Verify every claim under `root`. Read-only; never edits a document."""
    claims, problems = find_claims(root)
    verdicts = [check_claim(conn, c) for c in claims]
    counts = Counter(v.verdict for v in verdicts)
    return {
        "claims": len(claims),
        "counts": counts,
        "verdicts": verdicts,
        "malformed": problems,
    }
