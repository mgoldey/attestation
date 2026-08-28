"""The shipped example workspace must keep working.

Example data rots quietly: an adapter convention changes, the demo silently
produces different output, and nobody notices until someone runs it in front of
an audience. These tests are cheap and they pin the behaviour the README
promises, including the three deliberately-wrong claims.
"""

import importlib.util
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
    direction gets no ranking at all.

    This used `rank-method` as its example until ndcg was added to
    METRIC_DIRECTION, at which point the family ranked and the test failed --
    correctly. The RULE is what matters, not which family happens to violate
    it, so the example is now a metric that is genuinely ambiguous rather than
    merely absent: a "rate" can be a hit rate or an error rate, and no table
    should ever declare a direction for it.
    """
    with pytest.raises(ValueError, match="known direction"):
        ledger.compare(scanned, "rank-method", metric="n_records")


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


def test_the_suite_does_not_read_the_developers_own_metric_directions(tmp_path, monkeypatch):
    """metric_directions() overlays ~/.hermes/metric_direction.toml, which is
    the extension point ledger.py's own "unknown direction" error tells users
    to create. A contributor who followed that advice then watched an unrelated
    test fail: this file asserts ndcg_at_10 has NO declared direction, true only
    of a machine where nobody had declared one. CI passed because its runners
    have no ~/.hermes. conftest's _hermetic_env now repoints the ladder."""
    import os

    from attestation.ledger import METRIC_DIRECTION_PATH_ENV, _metric_direction_path

    configured = os.environ.get(METRIC_DIRECTION_PATH_ENV)
    assert configured, "the hermetic fixture must pin the metric-direction ladder"
    assert not _metric_direction_path().exists(), (
        "tests must run against an absent override file, not the developer's own"
    )
    assert str(pathlib.Path.home()) not in configured, (
        f"the ladder still resolves inside $HOME: {configured}"
    )


def test_the_readme_quickstart_runs_without_a_model_server(tmp_path, monkeypatch):
    """README's "Try it in 60 seconds" promises the ledger and claim checker
    work with no LLM backend reachable. That promise is the whole point of the
    block -- it moves first value from ~40 minutes (install Ollama, pull 7GB,
    ingest, tag) to under a second -- so it must not rot into a lie.

    Runs the documented sequence against a deliberately dead backend.
    """
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RSS_DB", str(tmp_path / "quickstart.db"))
    root = pathlib.Path(__file__).resolve().parents[1]

    conn = get_db(tmp_path / "quickstart.db")
    scanned = ledger.scan(conn, root / "examples" / "workspace")
    assert scanned, "the documented `runs scan --root examples/workspace` found nothing"

    ranked = ledger.compare(conn, "kdsweep")
    assert ranked["winner"] == "kdsweep_t4", (
        "README prints kdsweep_t4 as the winner of the documented command"
    )
    caveats = " ".join(ranked.get("caveats", []))
    assert "seed replication" in caveats, (
        "README quotes the seed-replication caveat as the reason this output"
        " earns its keep; without it the block oversells"
    )

    out = claims.check(conn, FINDINGS)
    assert out["counts"].get("contradicted") == 1, (
        f"README promises a contradicted verdict from this exact file: {out['counts']}"
    )


TRAINING = pathlib.Path(__file__).resolve().parent.parent / "examples" / "flows" / "training"


def _train_mlflow():
    spec = importlib.util.spec_from_file_location("train_mlflow", TRAINING / "train_mlflow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _c_sweep_result(C, auc, run_id="deadbeef01234567"):
    return {
        "C": C,
        "run_id": run_id,
        "accuracy": auc,
        "precision": auc,
        "recall": auc,
        "auc": auc,
    }


def test_the_committed_mlflow_directory_is_read_as_four_arms_of_one_family(tmp_path):
    """The tracker reader was written against documented layouts and said so
    in capitals. examples/flows/training/mlruns is the output of a real
    mlflow-skinny run (train_mlflow.py, 2026-08-28), committed so this test
    needs no mlflow: this is the first real directory it has read."""
    conn = get_db(tmp_path / "t.db")
    out = ledger.scan(conn, TRAINING.parent, project="training")
    assert out["scanned"] == {"training": 4}, out
    rows = conn.execute(
        "SELECT name, family, adapter FROM runs WHERE project = 'training' ORDER BY name"
    ).fetchall()
    assert {r["family"] for r in rows} == {"c_sweep"}
    assert {r["adapter"] for r in rows} == {"mlflow"}
    metrics = conn.execute(
        "SELECT DISTINCT metric FROM run_metrics rm JOIN runs r ON r.id = rm.run_id"
        " WHERE r.project = 'training'"
    ).fetchall()
    assert {m["metric"] for m in metrics} >= {
        "accuracy",
        "precision",
        "recall",
        "auc",
        "train_loss",
    }
    steps = conn.execute(
        "SELECT step FROM run_metrics rm JOIN runs r ON r.id = rm.run_id"
        " WHERE r.project = 'training' AND rm.metric = 'train_loss'"
    ).fetchall()
    assert all(s["step"] == 9 for s in steps), "final value of a ten-step curve, step recorded"


def test_the_mlflow_family_compares_and_its_findings_carry_one_contradiction(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, TRAINING.parent, project="training")
    result = ledger.compare(conn, "c_sweep", metric="auc", project="training")
    assert result["winner"], result
    parsed, errors = claims.parse_file(TRAINING / "FINDINGS.md")
    assert not errors
    verdicts = [claims.check_claim(conn, c) for c in parsed]
    kinds = [v.verdict for v in verdicts]
    assert kinds.count(claims.VerdictKind.CONTRADICTED) == 1, kinds
    assert kinds.count(claims.VerdictKind.SUPPORTED) == 4, kinds


def test_the_committed_mlruns_carries_no_personal_or_machine_attribution():
    """train_mlflow.py's scrub() strips what mlflow writes by default -- the
    reader (_mlflow_runs) never touches any of it. This is the guard that
    keeps a future regeneration honest: forget to call scrub() and this
    fails before the fixture is committed with a real name and a real
    machine path in it, which is exactly the class of content this repo has
    previously rewritten history to remove."""
    forbidden = ("/home/", "matt", "github.com", "mlflow.user")
    offenders = []
    for path in TRAINING.joinpath("mlruns").rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        hits = [needle for needle in forbidden if needle in text or needle in path.name]
        if hits:
            offenders.append((str(path.relative_to(TRAINING)), hits))
    assert not offenders, offenders


def test_write_findings_refuses_to_write_a_demo_that_would_not_contradict(tmp_path):
    """A regeneration under a different sklearn or seed can shift the four
    real AUCs close enough to the deliberately-wrong claim (best - 0.05) that
    it would land inside claim tolerance and be SUPPORTED, not CONTRADICTED --
    silently breaking the one thing FINDINGS.md exists to demonstrate. This
    must fail loudly before anything is written, not commit a broken demo."""
    tm = _train_mlflow()
    results = [
        _c_sweep_result(0.01, 0.90, "aaaaaaaa1111"),
        _c_sweep_result(0.1, 0.92, "bbbbbbbb2222"),
        _c_sweep_result(1.0, 0.95, "cccccccc3333"),
        # best - 0.05 == 0.90 exactly: the "deliberately wrong" claim would
        # match this arm's own real value.
        _c_sweep_result(10.0, 0.95, "dddddddd4444"),
    ]
    with pytest.raises(ValueError, match="CONTRADICTED"):
        tm.write_findings(tmp_path / "FINDINGS.md", results, project="training")
    assert not (tmp_path / "FINDINGS.md").exists(), "must not write a broken demo"


def test_write_findings_writes_five_claims_one_under_the_wrong_heading(tmp_path):
    tm = _train_mlflow()
    results = [
        _c_sweep_result(0.01, 0.90, "aaaaaaaa1111"),
        _c_sweep_result(0.1, 0.93, "bbbbbbbb2222"),
        _c_sweep_result(1.0, 0.97, "cccccccc3333"),
        _c_sweep_result(10.0, 0.99, "dddddddd4444"),
    ]
    out = tmp_path / "FINDINGS.md"
    tm.write_findings(out, results, project="training")
    lines = out.read_text().splitlines()
    parsed, errors = claims.parse_file(out)
    assert not errors
    assert len(parsed) == 5
    heading_line = next(
        i for i, line in enumerate(lines, start=1) if line.startswith("### Deliberately wrong")
    )
    wrong_claims = [c for c in parsed if c.line > heading_line]
    assert len(wrong_claims) == 1
