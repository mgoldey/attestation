"""Tests for the generic ledger adapter's provenance handling.

Split out from test_ledger.py (which owns the discovery/scan/compare
integration tests) because these exercise adapter internals directly:
per-sample spread on aggregated metrics, and seed retention across the JSON
and CSV result shapes.
"""

import json
from pathlib import Path

import pytest

from attestation import ledger
from attestation.db import get_db
from attestation.ledger_adapters import generic


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def conn(tmp_path):
    c = get_db(tmp_path / "l.db")
    yield c
    c.close()


def test_bimodal_and_tight_distributions_are_distinguishable_by_std():
    """Both reduce to the same mean (~0.337); only the std tells them apart."""
    bimodal = generic.metrics_from_payload(
        [{"wer": 0.01}, {"wer": 0.01}, {"wer": 0.99}], step=None, split=None
    )
    tight = generic.metrics_from_payload(
        [{"wer": 0.34}, {"wer": 0.34}, {"wer": 0.33}], step=None, split=None
    )

    bimodal_by_name = {m.metric: m.value for m in bimodal}
    tight_by_name = {m.metric: m.value for m in tight}

    assert bimodal_by_name["wer"] == pytest.approx(0.3367, abs=1e-4)
    assert tight_by_name["wer"] == pytest.approx(0.3367, abs=1e-4)

    assert "wer_std" in bimodal_by_name
    assert "wer_std" in tight_by_name
    assert bimodal_by_name["wer_std"] > tight_by_name["wer_std"]
    assert bimodal_by_name["wer_std"] == pytest.approx(0.4619, abs=1e-4)
    assert tight_by_name["wer_std"] == pytest.approx(0.0047, abs=1e-4)


def test_single_sample_std_is_zero_not_an_error():
    out = generic.metrics_from_payload([{"wer": 0.5}], step=None, split=None)
    by_name = {m.metric: m.value for m in out}

    assert by_name["wer_std"] == 0.0
    assert by_name["n_records"] == 1.0


def test_n_records_is_not_duplicated_as_a_redundant_count_metric():
    """n_records already carries the count; a `<metric>_n` would be redundant."""
    out = generic.metrics_from_payload([{"wer": 0.5}, {"wer": 0.6}], step=None, split=None)
    names = {m.metric for m in out}

    assert names == {"wer", "wer_std", "n_records"}


def test_seed_in_a_json_payload_is_retained_as_provenance():
    out = generic.metrics_from_payload({"seed": 1337, "wer": 0.053}, step=None, split=None)
    names = {m.metric for m in out}

    # seed must not leak into run_metrics -- it is not a rankable quantity
    assert "seed" not in names
    assert names == {"wer"}


def test_seed_survives_discovery_via_config_not_metrics(conn, tmp_path):
    write(
        tmp_path / "proj" / "results" / "eval_run.json",
        json.dumps({"seed": 1337, "wer": 0.053}),
    )

    ledger.scan(conn, tmp_path)
    run = ledger.detail(conn, "proj", "eval_run")

    assert run["config"] == {"seed": "1337"}
    metric_names = {m["metric"] for m in run["metrics"]}
    assert "seed" not in metric_names
    assert metric_names == {"wer"}


def test_csv_with_a_numeric_seed_column_retains_it_in_config(conn, tmp_path):
    write(
        tmp_path / "proj" / "results" / "sweep.csv",
        "config_name,seed,wer\narm_a,1337,0.05\narm_b,42,0.06\n",
    )

    ledger.scan(conn, tmp_path)
    run_a = ledger.detail(conn, "proj", "sweep/arm_a")

    assert run_a["config"]["seed"] == 1337.0
    metric_names = {m["metric"] for m in run_a["metrics"]}
    assert "seed" not in metric_names, "seed is provenance, not a rankable metric"
    assert "wer" in metric_names


def test_runs_compare_never_auto_selects_a_std_metric(conn, tmp_path):
    """METRIC_DIRECTION is the only source runs_compare auto-picks from; a new
    `wer_std` metric must not become eligible just by existing."""
    write(
        tmp_path / "proj" / "results" / "eval_a.json",
        json.dumps([{"wer": 0.01}, {"wer": 0.01}, {"wer": 0.99}]),
    )
    write(
        tmp_path / "proj" / "results" / "eval_b.json",
        json.dumps([{"wer": 0.34}, {"wer": 0.34}, {"wer": 0.33}]),
    )

    ledger.scan(conn, tmp_path)
    result = ledger.compare(conn, family="eval")

    assert result["metric"] == "wer"
    assert "wer_std" not in ledger.METRIC_DIRECTION
