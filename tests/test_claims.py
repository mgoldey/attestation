"""Claim checker tests."""

import json
import os
import time
from pathlib import Path

import pytest

from attestation import claims, ledger
from attestation.db import get_db


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def ledgered(tmp_path):
    """A workspace with two runs recorded, ready for claims to point at."""
    ws = tmp_path / "ws"
    write(ws / "proj" / "results" / "eval_a.json", json.dumps({"wer": 0.053, "cer": 0.02}))
    write(ws / "proj" / "results" / "eval_b.json", json.dumps({"wer": 0.900}))
    conn = get_db(tmp_path / "c.db")
    ledger.scan(conn, ws)
    yield conn, ws
    conn.close()


def test_supported_when_the_run_agrees(ledgered):
    conn, ws = ledgered
    doc = write(
        ws / "README.md", "WER is low.\n<!-- claim: proj/eval_a metric=wer value=0.053 -->\n"
    )

    out = claims.check(conn, doc)

    assert out["counts"] == {"supported": 1}
    assert "eval_a" in out["verdicts"][0].message


def test_contradicted_when_the_run_disagrees(ledgered):
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.010 -->\n")

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "contradicted"
    assert verdict.actual == pytest.approx(0.053)
    assert "0.053" in verdict.message


def test_unsupported_is_not_contradicted(ledgered):
    """The distinction is the point: "I never recorded this" must not read as
    "this is false". The fix for each is different."""
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/nonexistent metric=wer value=0.5 -->\n")

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "unsupported"
    assert "may be true" in verdict.message


def test_a_run_without_that_metric_is_unsupported(ledgered):
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/eval_b metric=bleu value=0.5 -->\n")

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "unsupported"
    assert "no metric" in verdict.message


def test_wildcard_matching_several_runs_is_ambiguous(ledgered):
    """Silently taking the first match is how a checker reports a confident
    wrong answer."""
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/eval_* metric=wer value=0.053 -->\n")

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "ambiguous"
    assert sorted(verdict.matched) == ["eval_a", "eval_b"]


def test_tolerance_admits_a_rounded_transcription(ledgered):
    """A number in prose is rounded. Exact-match would flag every claim
    contradicted and get the checker switched off within a day."""
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.05 tol=0.01 -->\n")

    assert claims.check(conn, doc)["verdicts"][0].verdict == "supported"


def test_tolerance_does_not_admit_a_real_difference(ledgered):
    conn, ws = ledgered
    doc = write(ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.05 tol=0.0001 -->\n")

    assert claims.check(conn, doc)["verdicts"][0].verdict == "contradicted"


def test_stale_when_the_artifact_changed_after_as_of(ledgered):
    """The value still matches, but the evidence moved -- worth re-verifying,
    and distinct from being wrong."""
    conn, ws = ledgered
    artifact = ws / "proj" / "results" / "eval_a.json"
    future = time.time() + 86400 * 2
    os.utime(artifact, (future, future))
    doc = write(
        ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.053 as_of=2020-01-01 -->\n"
    )

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "stale"
    assert verdict.actual == pytest.approx(0.053)


def test_a_claim_without_as_of_is_never_stale(ledgered):
    conn, ws = ledgered
    artifact = ws / "proj" / "results" / "eval_a.json"
    future = time.time() + 86400 * 2
    os.utime(artifact, (future, future))
    doc = write(ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.053 -->\n")

    assert claims.check(conn, doc)["verdicts"][0].verdict == "supported"


@pytest.mark.parametrize(
    "body",
    [
        "claim: no-slash metric=wer value=1",
        "claim: proj/run value=1",
        "claim: proj/run metric=wer",
        "claim: proj/run metric=wer value=not-a-number",
    ],
)
def test_malformed_claims_are_reported_not_skipped(ledgered, body):
    """A silently skipped claim disappears from review without anyone deciding
    to remove it."""
    conn, ws = ledgered
    doc = write(ws / "R.md", f"<!-- {body} -->\n")

    out = claims.check(conn, doc)

    assert out["claims"] == 0
    assert len(out["malformed"]) == 1
    assert str(doc) in out["malformed"][0]


def test_claims_are_found_across_a_directory_tree(ledgered):
    conn, ws = ledgered
    write(ws / "README.md", "<!-- claim: proj/eval_a metric=wer value=0.053 -->\n")
    write(ws / "docs" / "deep" / "notes.md", "<!-- claim: proj/eval_b metric=wer value=0.9 -->\n")

    out = claims.check(conn, ws)

    assert out["claims"] == 2
    assert out["counts"]["supported"] == 2


def test_the_checker_never_edits_a_document(ledgered):
    """Read-only: it reports, it does not correct."""
    conn, ws = ledgered
    text = "WER is low.\n<!-- claim: proj/eval_a metric=wer value=0.010 -->\n"
    doc = write(ws / "R.md", text)

    claims.check(conn, doc)

    assert doc.read_text() == text


def test_prose_is_untouched_by_annotation(ledgered):
    """The comment renders as nothing, so the document reads exactly as before
    -- which is why this format can be adopted one claim at a time."""
    conn, ws = ledgered
    doc = write(
        ws / "R.md",
        "The cut leaves WER unchanged (**0.053 vs 0.043**).\n"
        "<!-- claim: proj/eval_a metric=wer value=0.053 -->\n",
    )

    out = claims.check(conn, doc)

    assert out["counts"] == {"supported": 1}
    assert "**0.053 vs 0.043**" in doc.read_text()


def test_coverage_finds_numbers_no_claim_covers(tmp_path):
    """The inverse of check(): a document with zero contradicted claims looks
    healthy while asserting unverifiable numbers, and nothing else shows it."""
    doc = write(
        tmp_path / "R.md",
        "Baseline WER is 0.043 and the cut gives 0.053.\n"
        "<!-- claim: p/r metric=wer value=0.043 -->\n",
    )

    out = claims.coverage(doc)

    assert out["numbers"] == 2
    assert [u["value"] for u in out["uncovered"]] == [0.053]


def test_coverage_ignores_integers(tmp_path):
    """Integers in prose are versions, counts and dates: on a real index, 212
    numbers reduce to 30 decimals and the decimals are the results."""
    doc = write(tmp_path / "R.md", "Ran 1709 tests across 8 projects in 2026.\n")

    assert claims.coverage(doc)["numbers"] == 0


@pytest.mark.parametrize(
    "line",
    [
        "Tested against hermes-agent v0.20.0 here.",
        "Released on 2026-05-28 after review.",
        "See https://example.com/v1.2/docs for details.",
        "Pin it with `numpy==1.26.4` exactly.",
    ],
)
def test_coverage_excludes_structural_numbers(tmp_path, line):
    """Versions, ISO dates, URLs and package pins are structure, not results."""
    doc = write(tmp_path / "R.md", line + "\n")

    assert claims.coverage(doc)["uncovered"] == []


def test_coverage_ignores_html_comments(tmp_path):
    """A comment renders as nothing, so nothing in one is asserted to a reader.
    Masking only claim annotations reported a note explaining why a number was
    left unannotated as itself an unannotated number."""
    doc = write(
        tmp_path / "R.md",
        "Nothing asserted here.\n<!-- an aside mentioning 0.353 and 0.414 -->\n",
    )

    assert claims.coverage(doc)["uncovered"] == []


def test_coverage_respects_claim_tolerance(tmp_path):
    """A claim with a loose tolerance covers the rounded number in the prose;
    a tight one does not, which is how a precise annotation can leave the
    number a reader actually sees unverified."""
    loose = write(
        tmp_path / "loose.md",
        "Correlation was 0.41.\n<!-- claim: p/r metric=rho value=0.4063 tol=0.01 -->\n",
    )
    tight = write(
        tmp_path / "tight.md",
        "Correlation was 0.41.\n<!-- claim: p/r metric=rho value=0.4063 tol=0.0005 -->\n",
    )

    assert claims.coverage(loose)["uncovered"] == []
    assert [u["value"] for u in claims.coverage(tight)["uncovered"]] == [0.41]


def test_coverage_handles_grouped_thousands(tmp_path):
    """1,234.5 must read as 1234.5, not truncate to 234.5 at the comma."""
    doc = write(tmp_path / "R.md", "Dataset totals 1,234.5 GB.\n")

    out = claims.coverage(doc)

    assert [u["value"] for u in out["uncovered"]] == [1234.5]


def test_coverage_handles_scientific_notation(tmp_path):
    """3.2e-4 must read as 0.00032, not truncate to 3.2 at the exponent."""
    doc = write(tmp_path / "R.md", "Learning rate was 3.2e-4 and 1.5e3 warmup steps.\n")

    out = claims.coverage(doc)

    values = sorted(u["value"] for u in out["uncovered"])
    assert values == pytest.approx([0.00032, 1500.0])


def test_coverage_reports_true_line_numbers(tmp_path):
    """Masking blanks with spaces rather than deleting, so offsets survive."""
    doc = write(tmp_path / "R.md", "\n".join(["intro", "<!-- x -->", "value is 0.99"]) + "\n")

    assert claims.coverage(doc)["uncovered"][0]["line"] == 3


@pytest.fixture
def multi_split(tmp_path):
    """One artifact, one run, three splits of the same metric -- the shape
    that let a claim select its own evidence before the fix."""
    ws = tmp_path / "ws"
    write(
        ws / "proj" / "results" / "bench.json",
        json.dumps(
            {
                "summary": {
                    "variants": {
                        "raw": {"mae": 0.353},
                        "corrected": {"mae": 0.212},
                        "baseline": {"mae": 0.900},
                    }
                }
            }
        ),
    )
    conn = get_db(tmp_path / "c.db")
    ledger.scan(conn, ws)
    yield conn, ws
    conn.close()


def test_multiple_splits_without_disambiguator_is_ambiguous(multi_split):
    """A claim must not be able to select its own evidence: two different
    claimed values against the same run cannot both come back 'supported'."""
    conn, ws = multi_split
    doc_raw = write(ws / "raw.md", "<!-- claim: proj/bench metric=mae value=0.353 -->\n")
    doc_baseline = write(ws / "baseline.md", "<!-- claim: proj/bench metric=mae value=0.900 -->\n")

    v_raw = claims.check(conn, doc_raw)["verdicts"][0]
    v_baseline = claims.check(conn, doc_baseline)["verdicts"][0]

    assert v_raw.verdict == "ambiguous"
    assert v_baseline.verdict == "ambiguous"
    assert not (v_raw.verdict == "supported" and v_baseline.verdict == "supported")


def test_split_disambiguator_resolves_the_row(multi_split):
    conn, ws = multi_split
    doc = write(
        ws / "R.md",
        "<!-- claim: proj/bench metric=mae value=0.353 split=summary.variants.raw -->\n",
    )

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "supported"
    assert verdict.split == "summary.variants.raw"


def test_split_disambiguator_can_contradict(multi_split):
    """The baseline split really is 0.900 -- claiming it is 0.353 must not
    quietly verify against a different split."""
    conn, ws = multi_split
    doc = write(
        ws / "R.md",
        "<!-- claim: proj/bench metric=mae value=0.353 split=summary.variants.baseline -->\n",
    )

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict == "contradicted"
    assert verdict.actual == pytest.approx(0.900)


def test_missing_evidence_file_is_not_supported(ledgered):
    """Deleting the evidence must not leave the claim reading as verified."""
    conn, ws = ledgered
    artifact = ws / "proj" / "results" / "eval_a.json"
    doc = write(ws / "R.md", "<!-- claim: proj/eval_a metric=wer value=0.053 -->\n")
    artifact.unlink()

    verdict = claims.check(conn, doc)["verdicts"][0]

    assert verdict.verdict != "supported"
    assert verdict.verdict == "stale"


@pytest.mark.parametrize("minus", ["-", "−"])
def test_coverage_captures_negative_numbers(tmp_path, minus):
    """A negative result is written with a sign. Capturing only the digits made
    a correctly-annotated -0.0988 read as an uncovered +0.10 -- a sign error
    that inverts the finding. Unicode minus counts: real prose uses it."""
    doc = write(
        tmp_path / "R.md",
        f"One convention gives +0.60, the other {minus}0.10.\n"
        "<!-- claim: p/r metric=rho value=-0.0988 tol=0.005 -->\n"
        "<!-- claim: p/r metric=rho value=0.5997 tol=0.005 -->\n",
    )

    out = claims.coverage(doc)

    assert out["uncovered"] == [], f"both signed values should be covered, got {out['uncovered']}"
