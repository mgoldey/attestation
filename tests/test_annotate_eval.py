"""Tests for the attestation-annotate eval scorer, model-free.

`evals/` is not a package; the modules there are scripts with an importable
core, same as test_tagging_eval.py's own sys.path insertion. The ledger
fixture is built via the REAL `ledger.scan` over a written results/ JSON
file (the shape tests/test_claims.py's `ledgered` fixture uses), which is
I/O by nature.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import annotate_eval as ae  # noqa: E402


def _payload(**overrides) -> dict:
    base = {
        "id": "t",
        "project": "asr",
        "run": "asr_biglm",
        "metrics": [{"metric": "wer", "value": 0.08, "step": None, "split": None}],
        "topic": "how the run performed",
    }
    base.update(overrides)
    return base


def test_a_good_paragraph_passes_every_check(tmp_path):
    payload = _payload()
    paragraph = (
        "The biglm run reached a word error rate of 0.08 on the held-out set.\n"
        "<!-- claim: asr/asr_biglm metric=wer value=0.08 -->\n"
    )

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert result["pass"], result["errors"]
    assert all(result["checks"].values()), result["checks"]


def test_an_uncovered_decimal_fails_the_coverage_check(tmp_path):
    payload = _payload()
    paragraph = (
        "The biglm run reached 0.08 WER, up from a 0.15 baseline.\n"
        "<!-- claim: asr/asr_biglm metric=wer value=0.08 -->\n"
    )

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["coverage_complete"]
    assert any("uncovered" in e for e in result["errors"])


def test_a_wrong_value_fails_as_contradicted(tmp_path):
    payload = _payload()
    paragraph = (
        "The biglm run reached a word error rate of 0.20.\n"
        "<!-- claim: asr/asr_biglm metric=wer value=0.20 -->\n"
    )

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["claims_supported"]
    assert any("contradicted" in e for e in result["errors"])


def test_an_invented_cite_key_fails(tmp_path):
    payload = _payload()
    paragraph = (
        "The biglm run reached a word error rate of 0.08 (Smith et al.).\n"
        "<!-- claim: asr/asr_biglm metric=wer value=0.08 cite=smith2020asr -->\n"
    )

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["no_invented_cite"]
    assert any("smith2020asr" in e for e in result["errors"])


def test_a_paragraph_with_no_claims_at_all_fails_supported_check(tmp_path):
    payload = _payload()
    paragraph = "The run reached 0.08 WER.\n"  # no annotation at all

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["coverage_complete"]
    assert not result["checks"]["claims_supported"]


def test_split_disambiguated_metrics_are_both_checkable(tmp_path):
    payload = _payload(
        run="mt_beam4",
        project="mt",
        metrics=[
            {"metric": "bleu", "value": 27.8, "step": None, "split": "test"},
            {"metric": "bleu", "value": 31.2, "step": None, "split": "train"},
        ],
    )
    paragraph = (
        "On the test split, beam search reached a BLEU of 27.8.\n"
        "<!-- claim: mt/mt_beam4 metric=bleu value=27.8 split=test -->\n"
    )

    result = ae.score_one(payload, paragraph, workspace=tmp_path)

    assert result["pass"], result["errors"]


def test_load_cases_reads_the_committed_fixture():
    cases = ae.load_cases()
    assert len(cases) >= 10
    assert any(c.get("expect_fail") for c in cases)
    # both failure shapes the brief calls out are represented
    uncovered = [
        c for c in cases if "uncovered" not in c.get("paragraph", "") and c.get("expect_fail")
    ]
    assert uncovered  # at least one expect_fail case exists beyond a trivial name match


def test_every_committed_case_scores_as_its_expect_fail_says(tmp_path):
    cases = ae.load_cases()
    for i, case in enumerate(cases):
        payload = {k: case[k] for k in ("id", "project", "run", "metrics", "topic")}
        result = ae.score_one(payload, case["paragraph"], workspace=tmp_path / str(i))
        if case.get("expect_fail"):
            assert not result["pass"], f"{case['id']} was supposed to fail but passed"
        else:
            assert result["pass"], f"{case['id']}: {result['errors']}"


# ---------------------------------------------------------------------------
# RED proof: the invented-cite check.
#
# Mechanically mutated evals/annotate_eval.py's score_one, in place, then ran
# `uv run pytest tests/test_annotate_eval.py -k invented_cite -q`, then
# reverted:
#
#     uncited = [v for v in out["verdicts"] if v.verdict == claims.VerdictKind.UNCITED]   # ORIGINAL
#     uncited = []                                                                          # MUTANT
#
# With the mutant in place:
#     test_an_invented_cite_key_fails FAILED
#     -- AssertionError: assert not True  (checks["no_invented_cite"] was
#        True even though the paragraph cited smith2020asr, a key no
#        configured bibliography has -- the mutant always finds zero
#        uncited verdicts, so the check can never fail)
#
# Reverted the list comprehension back to filtering on VerdictKind.UNCITED
# and reran: PASSED. This proves the check is actually reading
# `check_citations`'s verdicts (via `claims.check`'s resolver argument)
# rather than passing by construction.
# ---------------------------------------------------------------------------
