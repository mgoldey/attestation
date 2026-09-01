"""Scoring for the attestation-annotate skill's output, model-free.

A trial is a `runs.detail`-shaped payload (project, run name, metrics with
value/step/split) plus a topic sentence; the model answers with a Markdown
paragraph stating a result and annotating it per the skill's contract: a
`<!-- claim: ... -->` beside each decimal, `cite=<key>` only when the
scenario supplied a bibliography entry for it (these scenarios supply none,
so ANY `cite=` is an invented key and a fail).

`score_one` writes the paragraph to a tmp `.md`, builds a tmp ledger holding
exactly the payload's run (via the real `ledger.scan` over a `results/` JSON
file -- the same shape `tests/test_claims.py`'s `ledgered` fixture uses, not
a hand-inserted row, so the scorer exercises the same adapter production
code does), then runs the REAL `claims.coverage` + `claims.check`:

    (a) every decimal in the paragraph is covered by a claim
        (`coverage(...)["uncovered"] == []`)
    (b) every claim comes back SUPPORTED (none unsupported/contradicted/
        ambiguous/stale)
    (c) no claim carries a `cite=` key at all -- the scenario's fixture
        ledger has no bibliography, so an invented `cite=` is checked by
        `claims.check_citations` against an empty `citations.Resolver([])`
        and any key present resolves to UNCITED
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from attestation import citations, claims, db, ledger

CASES_PATH = pathlib.Path(__file__).parent / "annotate_cases.json"


def load_cases(path: pathlib.Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _write_fixture_ledger(workspace: pathlib.Path, payload: dict) -> None:
    """Materialise `payload` (a `runs.detail`-shaped dict: project, run,
    metrics=[{metric,value,step,split}, ...]) as one result JSON the real
    generic adapter recognises, then scan it -- so the ledger this scorer
    checks against was produced by the same reader `attest runs scan` uses.
    """
    project = payload["project"]
    run = payload["run"]
    metrics = payload["metrics"]
    splits = {m.get("split") for m in metrics}
    if splits == {None}:
        # The generic adapter's simplest recognised shape: a flat
        # {metric: value} object, split=None on every row.
        body: dict = {m["metric"]: m["value"] for m in metrics}
    else:
        # A metric name recurs under different splits (e.g. bleu on both
        # test and train) -- nest by split so each row stays distinct;
        # `metrics_from_payload`'s `_descend` reads the nesting key as
        # `split`. A metric with split=None alongside split-nested ones
        # sits at the top level, same as the flat case.
        body = {}
        for m in metrics:
            split = m.get("split")
            if split is None:
                body[m["metric"]] = m["value"]
            else:
                body.setdefault(split, {})[m["metric"]] = m["value"]
    results_dir = workspace / project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run}.json").write_text(json.dumps(body))


def score_one(payload: dict, paragraph: str, *, workspace: pathlib.Path | None = None) -> dict:
    """Score one trial. Returns per-check booleans plus an overall `pass`."""
    result: dict = {
        "id": payload.get("id", f"{payload['project']}/{payload['run']}"),
        "errors": [],
        "checks": {
            "coverage_complete": False,
            "claims_supported": False,
            "no_invented_cite": False,
        },
    }

    def _run(ws: pathlib.Path) -> None:
        _write_fixture_ledger(ws, payload)
        doc = ws / "draft.md"
        doc.write_text(paragraph)

        cov = claims.coverage(doc)
        if cov["uncovered"]:
            result["errors"].append(
                f"{len(cov['uncovered'])} uncovered decimal(s): "
                + ", ".join(str(u["value"]) for u in cov["uncovered"])
            )
        else:
            result["checks"]["coverage_complete"] = True

        conn = db.get_db(ws / "eval.db")
        try:
            ledger.scan(conn, ws)
            resolver = citations.Resolver([])  # no bibliography: any cite= is invented
            out = claims.check(conn, doc, resolver=resolver)
            bad = [
                v
                for v in out["verdicts"]
                if v.verdict != claims.VerdictKind.SUPPORTED
                and v.verdict != claims.VerdictKind.UNCITED
            ]
            if bad:
                result["errors"].append(
                    "not all claims supported: "
                    + ", ".join(f"{v.claim.metric}={v.claim.value}->{v.verdict}" for v in bad)
                )
            elif not out["verdicts"]:
                result["errors"].append("no claims found in the paragraph")
            else:
                result["checks"]["claims_supported"] = True

            uncited = [v for v in out["verdicts"] if v.verdict == claims.VerdictKind.UNCITED]
            if uncited:
                result["errors"].append(
                    "invented cite= key(s): " + ", ".join(v.claim.cite or "?" for v in uncited)
                )
            else:
                result["checks"]["no_invented_cite"] = True
        finally:
            conn.close()

    if workspace is not None:
        _run(workspace)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            _run(pathlib.Path(tmp))

    result["pass"] = all(result["checks"].values())
    return result
