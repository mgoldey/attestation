"""The shipped example workspace must keep working.

Example data rots quietly: an adapter convention changes, the demo silently
produces different output, and nobody notices until someone runs it in front of
an audience. These tests are cheap and they pin the behaviour the README
promises, including the three deliberately-wrong claims.
"""

import pathlib

import pytest

from attestation import claims, ledger
from attestation.db import get_db

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "workspace"
FINDINGS = WORKSPACE / "speech-distill" / "FINDINGS.md"


@pytest.fixture
def scanned(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, WORKSPACE)
    yield conn
    conn.close()


def test_the_sweep_groups_into_one_family(scanned):
    """Family comes from the filename stem. If the convention changes, the
    example stops demonstrating a sweep and nobody would notice."""
    runs = ledger.list_runs(scanned, family="kdsweep", limit=50)
    assert {r["name"] for r in runs} == {
        "kdsweep_baseline",
        "kdsweep_t2",
        "kdsweep_t4",
        "kdsweep_t4b",
    }


def test_comparing_the_sweep_warns_about_seeds_and_closeness(scanned):
    out = ledger.compare(scanned, "kdsweep", metric="wer")
    assert out["winner"] == "kdsweep_t4"
    caveats = " ".join(out.get("caveats", []))
    assert "seed replication" in caveats, "single-run arms must be caveated"
    assert "too close to call" in caveats, "a 2.6% gap must be caveated"


def test_an_undeclared_metric_direction_is_refused_not_guessed(scanned):
    """The ledger's second rule: WER 0.043 -> 0.053 is a regression and
    accuracy 0.90 -> 0.94 is an improvement, so a metric with no known
    direction gets no ranking at all."""
    with pytest.raises(ValueError, match="known direction"):
        ledger.compare(scanned, "rank-method")


def test_a_config_without_results_records_no_metrics(scanned):
    detail = ledger.detail(scanned, "retrieval-ablation", "planned_colbert")
    assert detail is not None
    assert detail["metrics"] == [], "a spec must not be given an invented number"


def test_the_paper_produces_every_verdict(scanned):
    """Five supported, one contradicted, one unsupported, one malformed --
    the point of the example is that it exercises all of them."""
    out = claims.check(scanned, FINDINGS)
    assert out["counts"] == {"supported": 5, "contradicted": 1, "unsupported": 1}
    assert len(out["malformed"]) == 1
    assert "missing metric" in out["malformed"][0]


def test_the_contradicted_claim_names_both_numbers(scanned):
    """A verdict that says only "wrong" makes the reader go digging."""
    out = claims.check(scanned, FINDINGS)
    bad = [v for v in out["verdicts"] if v.verdict == "contradicted"]
    assert len(bad) == 1
    assert "0.0701" in bad[0].message and "0.0688" in bad[0].message


def test_coverage_finds_prose_numbers_no_claim_backs(scanned):
    out = claims.coverage(FINDINGS)
    assert out["numbers"] > out["covered"]
    assert out["uncovered"], "the example deliberately leaves numbers uncovered"
