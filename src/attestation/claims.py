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
from dataclasses import dataclass
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
    raw: str = ""


@dataclass
class Verdict:
    claim: Claim
    verdict: str
    message: str
    actual: float | None = None
    matched: list[str] | None = None
    source_path: str | None = None


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
        except ValueError:
            problems.append(f"{where}: value/tol must be numbers, got {body[:60]!r}")
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
            "unsupported",
            f"no run {claim.project}/{claim.run} in the ledger"
            " -- the claim may be true, but nothing here backs it",
        )
    if len(runs) > 1:
        names = [r["name"] for r in runs]
        return Verdict(
            claim,
            "ambiguous",
            f"{len(runs)} runs match {claim.run!r}; which one is meant is undecidable",
            matched=names,
        )

    run = runs[0]
    rows = conn.execute(
        "SELECT value FROM run_metrics WHERE run_id = ? AND metric = ?",
        (run["id"], claim.metric),
    ).fetchall()
    if not rows:
        return Verdict(
            claim,
            "unsupported",
            f"run {run['name']} records no metric {claim.metric!r}",
            source_path=run["source_path"],
        )

    # closest recorded value: a run may hold the metric at several steps, and
    # the claim is about the number the author saw, not a particular checkpoint
    actual = min((r["value"] for r in rows), key=lambda v: abs(v - claim.value))
    if abs(actual - claim.value) > claim.tol:
        return Verdict(
            claim,
            "contradicted",
            f"document says {claim.value:g}, run records {actual:g} (tolerance {claim.tol:g})",
            actual=actual,
            source_path=run["source_path"],
        )

    if _is_stale(claim, run["source_path"]):
        return Verdict(
            claim,
            "stale",
            f"value still matches, but {Path(run['source_path']).name} changed"
            f" after as_of={claim.as_of} -- re-verify",
            actual=actual,
            source_path=run["source_path"],
        )

    return Verdict(
        claim,
        "supported",
        f"{claim.metric}={actual:g} in {run['name']}",
        actual=actual,
        source_path=run["source_path"],
    )


# A measurement is written as a decimal. Integers in prose are overwhelmingly
# versions, counts, dates and identifiers -- on a real index, 212 numbers reduce
# to 30 decimals, and the decimals are the results. Requiring a decimal point is
# a crude filter that happens to match how people write measured quantities.
# The trailing guard rejects only a further DIGIT. Earlier versions also
# rejected a following period, which silently dropped every number that ended a
# sentence -- in prose, most of them. Version strings are excluded by
# _NOT_A_MEASUREMENT instead, where the intent is explicit.
_NUMBER_RE = re.compile(r"(?<![\w.])([-−]?\d+\.\d+)(?!\d)")

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
            value = float(match.group(1).replace("−", "-"))
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
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return {
        "claims": len(claims),
        "counts": counts,
        "verdicts": verdicts,
        "malformed": problems,
    }
