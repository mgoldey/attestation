"""W&B and MLflow local artifact directories, read through `generic`.

**Both readers have now been run against a real directory, not just the
transcribed fixtures below.** The MLflow reader was run against a real
directory on 2026-08-28 (examples/flows/training/mlruns, written by
mlflow-skinny 3.x via train_mlflow.py) and read four runs with final values
and steps; run_name did land in meta.yaml as documented, so no fallback to
tags/mlflow.runName was needed. tests/test_examples.py pins that committed
directory directly.

The W&B reader was run against a real directory the same day
(examples/wandb/wandb, written by wandb 0.17.6 via generate.py), and
`test_the_reader_scans_the_real_committed_wandb_fixture` below pins it
directly. The run directory is named `offline-run-<timestamp>-<id>`, not
`run-<timestamp>-<id>` as the fixtures below (still transcribed from
https://docs.wandb.ai/guides/track/save-restore, unchanged) assume -- but
`_wandb_runs` never filtered on that prefix, so both names already worked
and no reader code changed. The real surprise was upstream of naming:
offline W&B does not write wandb-summary.json or config.yaml to files/ at
all until the run is synced to a server. examples/wandb/generate.py's
docstring has the full finding, upstream references, and the local decode
step (via `wandb.sdk.internal.datastore`, a maintainer-endorsed workaround
for reading an offline `.wandb` log without syncing) that makes the
committed fixture real data rather than a fabricated stand-in.

The fixtures below stay synthetic on purpose: they are what the
shape-tolerance tests at the bottom exercise, targeted at specific edge
cases (a crashed run, a malformed file) a training run would not
reliably reproduce on demand. `CLAUDE.md` names this repo's recurring
failure mode as "tests that pass against the bug they were written to
catch"; the mitigation is that the shape-tolerance tests are no longer the
only line of defense -- the real fixture is now load-bearing too.
"""

import json
import math
from pathlib import Path

import pytest

from attestation.ledger_adapters import generic

WANDB_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "wandb"


def _wandb_run(root, run_id, summary, config=None, metadata=None):
    d = root / "wandb" / run_id / "files"
    d.mkdir(parents=True)
    (d / "wandb-summary.json").write_text(json.dumps(summary))
    if config is not None:
        lines = ["wandb_version: 1", ""]
        for k, v in config.items():
            lines += [f"{k}:", "  desc: null", f"  value: {v}"]
        lines += ["_wandb:", "  desc: null", "  value:", "    cli_version: 0.16.0"]
        (d / "config.yaml").write_text("\n".join(lines) + "\n")
    if metadata is not None:
        (d / "wandb-metadata.json").write_text(json.dumps(metadata))
    return d


def _mlflow_run(
    root,
    exp,
    run_id,
    *,
    name,
    metrics,
    params=None,
    status="FINISHED",
    lifecycle="active",
    start_ms=1755000000000,
):
    d = root / "mlruns" / exp / run_id
    (d / "metrics").mkdir(parents=True)
    (d / "params").mkdir(parents=True)
    (d / "meta.yaml").write_text(
        f"artifact_uri: file://{d}/artifacts\n"
        f"end_time: {start_ms + 60000}\n"
        f"experiment_id: '{exp}'\n"
        f"lifecycle_stage: {lifecycle}\n"
        f"run_id: {run_id}\n"
        f"run_name: {name}\n"
        f"start_time: {start_ms}\n"
        f"status: {status}\n"
    )
    for metric, lines in metrics.items():
        (d / "metrics" / metric).write_text("".join(f"{t} {v} {s}\n" for t, v, s in lines))
    for k, v in (params or {}).items():
        (d / "params" / k).write_text(str(v))
    return d


# --------------------------------------------------------------------------
# The gap this exists to close
# --------------------------------------------------------------------------


def test_a_project_with_only_tracker_dirs_is_not_invisible(tmp_path):
    """The whole point. `discover` walks RESULT_DIRS and CONFIG_DIRS at the
    project root; `wandb` and `mlruns` are in neither, so before this change a
    project whose entire experimental record lived in one of them scanned to
    zero runs and gave no indication why."""
    proj = tmp_path / "myproj"
    _wandb_run(proj, "run-20260814_101133-a1b2c3d4", {"wer": 0.12, "loss": 2.1})
    _mlflow_run(proj, "0", "abc123", name="baseline", metrics={"wer": [(0, 0.31, 5)]})

    runs = generic.discover(proj)
    assert len(runs) == 2, [r.name for r in runs]


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------


def test_wandb_summary_becomes_metrics(tmp_path):
    proj = tmp_path / "p"
    _wandb_run(proj, "run-20260814_101133-a1b2c3d4", {"wer": 0.0688, "val_loss": 2.29})

    (run,) = generic.discover(proj)
    assert {m.metric: m.value for m in run.metrics} == {"wer": 0.0688, "val_loss": 2.29}


def test_wandb_run_is_named_for_its_program_not_its_hash(tmp_path):
    """`run-20260814_101133-a1b2c3d4` names a timestamp and a hash. A ledger
    listing forty of those is unreadable, so the program name from
    wandb-metadata.json leads and the short id disambiguates."""
    proj = tmp_path / "p"
    _wandb_run(
        proj,
        "run-20260814_101133-a1b2c3d4",
        {"wer": 0.1},
        metadata={"program": "train_asr.py", "startedAt": "2026-08-14T10:11:33Z"},
    )

    (run,) = generic.discover(proj)
    assert "train_asr" in run.name
    assert "a1b2c3d4" in run.name, "the id must survive: two runs of one program collide"


def test_wandb_config_is_unwrapped_and_internals_dropped(tmp_path):
    """W&B wraps every config entry as {desc, value} and injects `_wandb`.
    Storing the wrapper would make every config value a dict."""
    proj = tmp_path / "p"
    _wandb_run(proj, "run-1-abc", {"wer": 0.1}, config={"lr": 0.0003, "epochs": 40})

    (run,) = generic.discover(proj)
    assert run.config["lr"] == 0.0003
    assert run.config["epochs"] == 40
    assert not any(k.startswith("_") for k in run.config), run.config


def test_wandb_start_time_comes_from_metadata(tmp_path):
    """`started` is otherwise unknowable for these runs -- no other artifact
    this adapter reads carries one."""
    proj = tmp_path / "p"
    _wandb_run(
        proj,
        "run-1-abc",
        {"wer": 0.1},
        metadata={"program": "t.py", "startedAt": "2026-08-14T10:11:33Z"},
    )

    (run,) = generic.discover(proj)
    assert run.started == "2026-08-14T10:11:33Z"


def test_wandb_run_without_summary_yields_nothing(tmp_path):
    """A crashed run leaves the directory and no summary. No metrics, no run --
    the same rule the rest of the adapter follows."""
    proj = tmp_path / "p"
    (proj / "wandb" / "run-1-abc" / "files").mkdir(parents=True)

    assert generic.discover(proj) == []


def test_the_reader_scans_the_real_committed_wandb_fixture():
    """examples/wandb/wandb is real: written by wandb 0.17.6 (generate.py,
    2026-08-28), not transcribed from documentation like the fixtures above.

    Its run directories are named `offline-run-<timestamp>-<id>`, which the
    reader's docstring once claimed it did not look for -- it does, because
    `_wandb_runs` never filtered on the directory name at all. This is the
    test that would have failed had that been false.
    """
    runs = generic.discover(WANDB_EXAMPLE)
    assert len(runs) == 4, [r.name for r in runs]
    assert all(r.adapter == "wandb" for r in runs)
    assert all(r.name.startswith("generate/") for r in runs), [r.name for r in runs]
    by_lr = {r.config["lr"]: r for r in runs}
    assert set(by_lr) == {0.001, 0.01, 0.1, 1.0}
    for run in runs:
        metrics = {m.metric for m in run.metrics}
        assert {"train_loss", "accuracy", "auc"} <= metrics, metrics
        assert run.started and run.started.startswith("2026-"), run.started
        assert not any(k.startswith("_") for k in run.config), run.config


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------


def test_mlflow_reads_the_final_value_of_each_metric(tmp_path):
    """The decision recorded in the spec: last line, not the whole curve.
    Recording history would make MLflow runs structurally unlike every other
    run in the ledger and flood run_metrics -- 200 epochs becomes 200 rows."""
    proj = tmp_path / "p"
    _mlflow_run(
        proj,
        "0",
        "abc",
        name="run-a",
        metrics={"wer": [(1000, 0.51, 0), (2000, 0.33, 1), (3000, 0.29, 2)]},
    )

    (run,) = generic.discover(proj)
    (metric,) = run.metrics
    assert metric.value == 0.29
    assert metric.step == 2, "the step of the value actually recorded"


def test_mlflow_uses_its_run_name_not_the_uuid(tmp_path):
    proj = tmp_path / "p"
    _mlflow_run(proj, "0", "9f8e7d6c5b4a", name="dense-baseline", metrics={"wer": [(0, 0.2, 0)]})

    (run,) = generic.discover(proj)
    assert "dense-baseline" in run.name


def test_mlflow_deleted_runs_are_skipped(tmp_path):
    """lifecycle_stage: deleted means the user deleted it in the UI.
    Resurrecting it is worse than missing it."""
    proj = tmp_path / "p"
    _mlflow_run(proj, "0", "keep", name="kept", metrics={"wer": [(0, 0.2, 0)]})
    _mlflow_run(
        proj, "0", "gone", name="trashed", metrics={"wer": [(0, 0.1, 0)]}, lifecycle="deleted"
    )

    names = [r.name for r in generic.discover(proj)]
    assert any("kept" in n for n in names)
    assert not any("trashed" in n for n in names), names


def test_mlflow_params_become_config(tmp_path):
    proj = tmp_path / "p"
    _mlflow_run(
        proj,
        "0",
        "abc",
        name="r",
        metrics={"wer": [(0, 0.2, 0)]},
        params={"lr": "0.0003", "model": "conformer"},
    )

    (run,) = generic.discover(proj)
    assert run.config["lr"] == "0.0003"
    assert run.config["model"] == "conformer"


def test_mlflow_run_with_no_metrics_yields_nothing(tmp_path):
    proj = tmp_path / "p"
    _mlflow_run(proj, "0", "abc", name="r", metrics={})

    assert generic.discover(proj) == []


# --------------------------------------------------------------------------
# Coexistence and shape tolerance
# --------------------------------------------------------------------------


def test_tracker_dirs_coexist_with_a_results_tree(tmp_path):
    """A project may keep both; neither reader may hide the other."""
    proj = tmp_path / "p"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "baseline.json").write_text(json.dumps({"wer": 0.4}))
    _wandb_run(proj, "run-1-abc", {"wer": 0.12})

    runs = generic.discover(proj)
    assert len(runs) == 2
    assert len({r.name for r in runs}) == 2


def test_a_tracker_run_colliding_with_a_results_run_is_recorded_once(tmp_path):
    """The readers must share `discover`'s `seen` set, not carry their own.

    UNIQUE(project, name) on `runs` means a duplicate name is a scan-time
    error rather than a duplicate row, so appending blindly would break the
    scan instead of merely inflating it. Names collide in practice: a driver
    that writes results/train_asr.json and also logs to W&B from train_asr.py
    produces the same stem from both.
    """
    proj = tmp_path / "p"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "train_asr").mkdir()
    (proj / "results" / "train_asr" / "a1b2c3d4.json").write_text(json.dumps({"wer": 0.4}))
    _wandb_run(
        proj,
        "run-20260814_101133-a1b2c3d4",
        {"wer": 0.12},
        metadata={"program": "train_asr.py"},
    )

    runs = generic.discover(proj)
    names = [r.name for r in runs]
    assert names.count("train_asr/a1b2c3d4") == 1, names


def test_no_metric_direction_is_inferred_from_tracker_metadata(tmp_path):
    """ledger.py line 21: "Never rank a metric whose direction is undeclared."
    W&B records a `goal` on some metrics. Reading it would put a second source
    of truth beside METRIC_DIRECTION, which is how an ablation gets ranked
    backwards. RunRecord has nowhere to put it, and that is deliberate."""
    proj = tmp_path / "p"
    d = _wandb_run(proj, "run-1-abc", {"custom_score": 0.9})
    (d / "wandb-summary.json").write_text(
        json.dumps({"custom_score": 0.9, "_wandb": {"goal": "maximize"}})
    )

    (run,) = generic.discover(proj)
    assert [m.metric for m in run.metrics] == ["custom_score"]
    assert "maximize" not in json.dumps(run.config or {})


@pytest.mark.parametrize(
    "broken",
    [
        "empty-summary",
        "summary-not-json",
        "no-metadata",
        "meta-yaml-missing",
        "metric-file-empty",
        "metric-file-garbage",
    ],
)
def test_malformed_tracker_dirs_degrade_rather_than_raise(tmp_path, broken):
    """The parser's job is to be un-surprised.

    These fixtures were written by the same person as the parser, so they
    cannot prove it handles a real directory. What they CAN prove is that a
    directory not matching them produces fewer runs, never a traceback -- which
    is the property that matters when someone finally points it at a real one.
    """
    proj = tmp_path / "p"
    if broken == "empty-summary":
        _wandb_run(proj, "run-1-abc", {})
    elif broken == "summary-not-json":
        d = _wandb_run(proj, "run-1-abc", {"wer": 0.1})
        (d / "wandb-summary.json").write_text("<!DOCTYPE html>not json")
    elif broken == "no-metadata":
        _wandb_run(proj, "run-1-abc", {"wer": 0.1})
    elif broken == "meta-yaml-missing":
        _mlflow_run(proj, "0", "abc", name="r", metrics={"wer": [(0, 0.2, 0)]})
        (proj / "mlruns" / "0" / "abc" / "meta.yaml").unlink()
    elif broken == "metric-file-empty":
        _mlflow_run(proj, "0", "abc", name="r", metrics={"wer": []})
    elif broken == "metric-file-garbage":
        _mlflow_run(proj, "0", "abc", name="r", metrics={"wer": [(0, 0.2, 0)]})
        (proj / "mlruns" / "0" / "abc" / "metrics" / "wer").write_text("not a metric line\n")

    generic.discover(proj)  # must not raise


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "Infinity", "NaN"])
def test_a_non_finite_mlflow_metric_is_not_recorded(tmp_path, bad):
    """NaN and inf must not become metrics, and MLflow can produce both.

    `_numeric_items` filters them and its docstring says why: NaN compares
    false to everything, so it loses every ranking it appears in and is
    reported as a legitimate last place, and `statistics.pstdev` raises on one
    -- "one such file took down every project in the workspace".

    That docstring also says it is "the one place every numeric metric passes
    through". The MLflow reader made that untrue: it builds Metric directly
    from float(), which accepts every spelling above.
    """
    proj = tmp_path / "p"
    d = _mlflow_run(proj, "0", "abc", name="diverged", metrics={"loss": [(0, 0.5, 0)]})
    (d / "metrics" / "loss").write_text(f"1000 0.5 0\n2000 {bad} 1\n")

    runs = generic.discover(proj)
    values = [m.value for r in runs for m in r.metrics]
    assert all(math.isfinite(v) for v in values), values


def test_a_finite_value_before_a_non_finite_one_is_still_lost(tmp_path):
    """Skipping the bad final line and reporting the last GOOD one would be
    worse: it would report a diverged run's mid-training loss as its result.
    The run diverged; "not measured" is the honest answer."""
    proj = tmp_path / "p"
    d = _mlflow_run(proj, "0", "abc", name="diverged", metrics={"loss": [(0, 0.5, 0)]})
    (d / "metrics" / "loss").write_text("1000 0.5 0\n2000 nan 1\n")

    assert generic.discover(proj) == []


def test_a_non_finite_wandb_config_value_stays_a_string(tmp_path):
    """Same root cause, lower stakes: config is not ranked, but silently
    turning the string "nan" into a float NaN is still wrong."""
    proj = tmp_path / "p"
    _wandb_run(proj, "run-1-abc", {"wer": 0.1}, config={"threshold": "nan"})

    (run,) = generic.discover(proj)
    assert run.config["threshold"] == "nan"


# --------------------------------------------------------------------------
# Which reader produced a run, and the caveat that comes with it
# --------------------------------------------------------------------------


def test_each_reader_labels_the_runs_it_produced(tmp_path):
    """A wandb-derived run must be distinguishable from a hand-written one.

    The caveats on these two readers are documented in this module's source
    and reached no user: they have never been run against a real directory,
    and they record final metric values rather than curves. A run carrying no
    trace of which reader produced it cannot surface either.
    """
    proj = tmp_path / "proj"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "eval.json").write_text(json.dumps({"wer": 0.3}))
    _wandb_run(proj, "run-20260814_101133-a1b2c3d4", {"wer": 0.12})
    _mlflow_run(proj, "0", "b" * 32, name="mlflow-arm", metrics={"wer": [(1, 0.2, 0)]})

    by_name = {r.name: r.adapter for r in generic.discover(proj)}

    assert by_name["eval"] == "generic"
    assert by_name["run-20260814_101133-a1b2c3d4"] == "wandb"
    assert by_name["mlflow-arm/" + "b" * 8] == "mlflow"


def test_the_adapter_label_survives_a_scan(tmp_path, monkeypatch):
    """The label is only useful if it reaches the ledger."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "eval.json").write_text(json.dumps({"wer": 0.3}))
    _wandb_run(proj, "run-1-abc", {"wer": 0.12})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)

    rows = {r["name"]: r["adapter"] for r in conn.execute("SELECT name, adapter FROM runs")}
    assert rows == {"eval": "generic", "run-1-abc": "wandb"}
    conn.close()


def test_detail_attaches_the_tracker_caveat(tmp_path):
    """`runs.detail` on a tracker-derived run says what the reader cannot do.

    Two things a reader would otherwise assume wrongly: the value is the final
    logged one rather than the curve or the best step, and neither reader has
    been exercised against a real directory.
    """
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    proj.mkdir(parents=True)
    _wandb_run(proj, "run-1-abc", {"wer": 0.12})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    found = ledger.detail(conn, "proj", "run-1-abc")

    assert found["adapter"] == "wandb"
    caveats = " ".join(found["caveats"])
    assert "wandb" in caveats
    assert "final" in caveats.lower()
    conn.close()


def test_detail_of_an_ordinary_run_carries_no_tracker_caveat(tmp_path):
    """A caveat on every run is boilerplate, which is what trains a reader to
    skip them. The generic reader has no such limitation to report."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    (ws / "proj" / "results").mkdir(parents=True)
    (ws / "proj" / "results" / "eval.json").write_text(json.dumps({"wer": 0.3}))

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    found = ledger.detail(conn, "proj", "eval")

    assert found["adapter"] == "generic"
    assert found["caveats"] == []
    conn.close()


def test_compare_warns_when_an_arm_came_from_a_tracker(tmp_path):
    """Ranking a tracker-derived final value against a hand-recorded best
    value compares two different things, and the ranking should say so."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    proj.mkdir(parents=True)
    _wandb_run(proj, "sweep-1-aaa", {"wer": 0.12}, metadata={"program": "sweep.py"})
    _wandb_run(proj, "sweep-2-bbb", {"wer": 0.30}, metadata={"program": "sweep.py"})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    out = ledger.compare(conn, "sweep", metric="wer")

    assert out["winner"] == "sweep/aaa"
    joined = " ".join(out["caveats"])
    assert "wandb" in joined, out["caveats"]
    assert "final" in joined.lower()
    conn.close()


def test_wandb_bookkeeping_keys_are_not_metrics(tmp_path):
    """W&B writes its own `_step`/`_runtime`/`_timestamp` into the summary.

    `_wandb_config` already drops underscore-prefixed keys; the metric path did
    not, so a wall-clock timestamp landed in run_metrics and showed up in
    runs.detail beside the real numbers. compare() refuses to rank them
    (undeclared direction), so this was noise rather than a wrong verdict --
    but a ledger that lists `_timestamp: 170000001.0` as a measurement invites
    exactly the misreading the direction rule exists to prevent.
    """
    proj = tmp_path / "p"
    _wandb_run(
        proj,
        "run-1-abc",
        {"accuracy": 0.71, "_step": 1001, "_runtime": 451.2, "_timestamp": 170000001.0},
    )

    (run,) = generic.discover(proj)
    assert [m.metric for m in run.metrics] == ["accuracy"], [m.metric for m in run.metrics]


def test_a_config_does_not_shadow_a_result_of_the_same_name(tmp_path):
    """A spec claimed a run's name and the measured result was dropped --
    naming the wrong winner.

    `discover` walks CONFIG_DIRS before RESULT_DIRS sharing one `seen` set of
    bare stems, so `configs/asr_biglm.yaml` claimed `asr_biglm` and
    `results/asr_biglm.json` was skipped. compare then reported the real winner
    (WER 0.05) in `without_metric` -- "an arm that was never evaluated", the
    phrase the docstring uses to justify surfacing it as a finding -- and named
    the loser (0.09) the winner.

    A config is a SPEC and a result is a MEASUREMENT. When both exist for one
    name the measurement is the run; the spec is what it was going to be.
    """
    proj = tmp_path / "asr"
    (proj / "configs").mkdir(parents=True)
    (proj / "results").mkdir()
    (proj / "configs" / "asr_biglm.yaml").write_text("lr: 0.001\n")
    (proj / "results" / "asr_biglm.json").write_text(json.dumps({"tag": "asr_biglm", "wer": 0.05}))

    runs = {r.name: r for r in generic.discover(proj)}
    assert "asr_biglm" in runs
    assert runs["asr_biglm"].status == "recorded", (
        f"the config shadowed the result: status={runs['asr_biglm'].status}, "
        f"source={runs['asr_biglm'].source_path}"
    )
    assert runs["asr_biglm"].metrics, "the measured value was dropped"


def test_two_result_directories_do_not_collide_on_a_bare_stem(tmp_path):
    """`results/baseline.json` and `eval/baseline.json` recorded one run.

    Both are RESULT_DIRS and the `seen` set holds bare stems, so the second was
    silently skipped with no diagnostic -- in a sweep where final numbers moved
    to eval/, that discards the real scores and ranks against stale ones, with
    no caveat because the tool cannot see what it dropped.
    """
    proj = tmp_path / "asr"
    (proj / "results").mkdir(parents=True)
    (proj / "eval").mkdir()
    (proj / "results" / "baseline.json").write_text(json.dumps({"tag": "baseline", "wer": 0.30}))
    (proj / "eval" / "baseline.json").write_text(json.dumps({"tag": "baseline", "wer": 0.05}))

    runs = generic.discover(proj)
    assert len(runs) == 2, f"two files became {len(runs)} run(s): {[r.name for r in runs]}"
    assert len({r.name for r in runs}) == 2, "the two runs must be distinguishable by name"
