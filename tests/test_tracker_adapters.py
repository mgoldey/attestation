"""W&B, MLflow, Sacred, DVC and Hydra local artifact directories, read
through `generic`.

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

Sacred's reader was added 2026-08-28, verified the same way against a real
directory (examples/sacred/sacred_runs, sacred 0.8.7 via generate.py) from
the start rather than retrofitted -- `test_the_reader_scans_the_real_committed_sacred_fixture`
below pins it. DVC's reader followed the same day, against a real `dvc repro`
(examples/dvc/, dvc 3.x via generate.sh); see
`test_the_reader_scans_the_real_committed_dvc_fixture`. Hydra's reader
followed the same day, against a real `--multirun` sweep (examples/hydra/,
hydra-core 1.3.5 via generate.sh); see
`test_the_reader_scans_the_real_committed_hydra_fixture`.

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
SACRED_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sacred"
DVC_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dvc"
HYDRA_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hydra"


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


def _sacred_run(
    root,
    n,
    *,
    experiment_name="lr_sweep",
    status="COMPLETED",
    config=None,
    metrics=None,
    result=None,
    write_metrics=True,
    write_config=True,
):
    d = root / "sacred_runs" / str(n)
    d.mkdir(parents=True)
    run_json = {
        "experiment": {"name": experiment_name},
        "status": status,
        "start_time": "2026-08-28T10:00:00.000000",
        "stop_time": "2026-08-28T10:00:10.000000",
    }
    if result is not None:
        run_json["result"] = result
    (d / "run.json").write_text(json.dumps(run_json))
    if write_config:
        (d / "config.json").write_text(json.dumps(config if config is not None else {}))
    if write_metrics and metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics))
    return d


def _dvc_project(
    root,
    *,
    stage="train",
    foreach_param="lr",
    items=("0.01", "0.1", "1", "10"),
    metrics=None,
    write_lock=True,
    write_params=True,
):
    """A `dvc.yaml` `foreach` stage over `params.yaml`'s `lr` list, with the
    metric files and `dvc.lock` a real `dvc repro` produces.

    `metrics` maps each item (a string, exactly as it appears in `items` and
    in the generated filenames) to its metrics dict; a missing key means that
    metric file is not written, which is how the "listed metric file missing"
    tolerance case is built.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "dvc.yaml").write_text(
        "stages:\n"
        f"  {stage}:\n"
        f"    foreach: ${{{foreach_param}}}\n"
        "    do:\n"
        f"      cmd: python train.py ${{item}}\n"
        "      params:\n"
        f"        - {foreach_param}\n"
        "      metrics:\n"
        "        - metrics/${item}.json\n"
    )
    if write_params:
        (root / "params.yaml").write_text(f"{foreach_param}: [{', '.join(items)}]\n")
    if metrics:
        (root / "metrics").mkdir(exist_ok=True)
        for item, payload in metrics.items():
            (root / "metrics" / f"{item}.json").write_text(json.dumps(payload))
    if write_lock:
        lock_lines = ["schema: '2.0'", "stages:"]
        for item in items:
            lock_lines.append(f"  {stage}@{item}:")
            lock_lines.append(f"    cmd: python train.py {item}")
            lock_lines.append("    params:")
            lock_lines.append("      params.yaml:")
            lock_lines.append(f"        {foreach_param}:")
            for i in items:
                lock_lines.append(f"        - {i}")
            lock_lines.append("    outs:")
            lock_lines.append(f"    - path: metrics/{item}.json")
            lock_lines.append("      hash: md5")
            lock_lines.append("      md5: deadbeef")
            lock_lines.append("      size: 100")
        (root / "dvc.lock").write_text("\n".join(lock_lines) + "\n")
    return root


def _hydra_arm(
    root,
    n,
    *,
    sweep_date="2026-01-01",
    sweep_time="10-00-00",
    job_name="train",
    config=None,
    metrics=None,
    write_hydra_yaml=True,
    write_config_yaml=True,
    metrics_filename="metrics.json",
):
    """One arm of a Hydra `--multirun` sweep: `multirun/<date>/<time>/<n>/`
    with `.hydra/config.yaml`, `.hydra/hydra.yaml` (naming `job_name` at
    `hydra.job.name`, and the sweep dir at `hydra.sweep.dir`), and a metrics
    file -- the shape a real sweep produces (examples/hydra/generate.sh),
    trimmed to just the keys `_hydra_runs` reads.
    """
    d = root / "multirun" / sweep_date / sweep_time / str(n) / ".hydra"
    d.mkdir(parents=True)
    if write_config_yaml:
        lines = [f"{k}: {v}" for k, v in (config or {}).items()]
        (d / "config.yaml").write_text("\n".join(lines) + ("\n" if lines else ""))
    if write_hydra_yaml:
        (d / "hydra.yaml").write_text(
            "hydra:\n"
            "  sweep:\n"
            f"    dir: multirun/{sweep_date}/{sweep_time}\n"
            "  overrides:\n"
            "    task:\n"
            f"    - lr={config.get('lr') if config else ''}\n"
            "  job:\n"
            f"    name: {job_name}\n"
        )
    if metrics is not None:
        (d.parent / metrics_filename).write_text(json.dumps(metrics))
    return d.parent


# --------------------------------------------------------------------------
# The gap this exists to close
# --------------------------------------------------------------------------


def test_a_project_with_only_tracker_dirs_is_not_invisible(tmp_path):
    """The whole point. `discover` walks RESULT_DIRS and CONFIG_DIRS at the
    project root; `wandb`, `mlruns`, `sacred_runs` and `multirun` are in none
    of them (and dvc.yaml/dvc.lock are not results at all), so before this
    change a project whose entire experimental record lived in one of them
    scanned to zero runs and gave no indication why."""
    proj = tmp_path / "myproj"
    _wandb_run(proj, "run-20260814_101133-a1b2c3d4", {"wer": 0.12, "loss": 2.1})
    _mlflow_run(proj, "0", "abc123", name="baseline", metrics={"wer": [(0, 0.31, 5)]})
    _sacred_run(proj, 1, config={"lr": 0.01}, metrics={"auc": {"steps": [0], "values": [0.9]}})
    _dvc_project(proj / "dvcproj", metrics={"0.01": {"auc": 0.9}})
    _hydra_arm(proj / "hydraproj", 0, config={"lr": 0.01}, metrics={"auc": 0.9})

    runs = generic.discover(proj)
    assert len(runs) == 3, [r.name for r in runs]
    dvc_runs = generic.discover(proj / "dvcproj")
    assert len(dvc_runs) == 1, [r.name for r in dvc_runs]
    hydra_runs = generic.discover(proj / "hydraproj")
    assert len(hydra_runs) == 1, [r.name for r in hydra_runs]


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
# Sacred
# --------------------------------------------------------------------------


def test_sacred_metrics_become_final_values_not_curves(tmp_path):
    """metrics.json holds every logged point; the decision this adapter
    inherits from MLflow's is the last value and its last step, not the
    curve -- a 10-step train_loss log becomes one Metric, not ten rows."""
    proj = tmp_path / "p"
    _sacred_run(
        proj,
        1,
        config={"lr": 0.01},
        metrics={
            "train_loss": {"steps": [0, 1, 2], "values": [0.9, 0.5, 0.3]},
            "auc": {"steps": [0], "values": [0.97]},
        },
    )

    (run,) = generic.discover(proj)
    by_name = {m.metric: m for m in run.metrics}
    assert by_name["train_loss"].value == 0.3
    assert by_name["train_loss"].step == 2
    assert by_name["auc"].value == 0.97


def test_sacred_run_is_named_experiment_slash_number(tmp_path):
    proj = tmp_path / "p"
    _sacred_run(
        proj,
        3,
        experiment_name="lr_sweep",
        config={},
        metrics={"auc": {"steps": [0], "values": [0.9]}},
    )

    (run,) = generic.discover(proj)
    assert run.name == "lr_sweep/3"
    assert run.family == "lr_sweep"


def test_sacred_config_json_becomes_config(tmp_path):
    proj = tmp_path / "p"
    _sacred_run(
        proj, 1, config={"lr": 0.01, "seed": 7}, metrics={"auc": {"steps": [0], "values": [0.9]}}
    )

    (run,) = generic.discover(proj)
    assert run.config == {"lr": 0.01, "seed": 7}


def test_sacred_numeric_result_becomes_a_metric(tmp_path):
    """run.json's own `result` field -- the value `@ex.main` returned -- is
    Sacred's own headline number and is not duplicated in metrics.json."""
    proj = tmp_path / "p"
    _sacred_run(
        proj,
        1,
        config={"lr": 0.01},
        metrics={"train_loss": {"steps": [0], "values": [0.5]}},
        result=0.987,
    )

    (run,) = generic.discover(proj)
    by_name = {m.metric: m for m in run.metrics}
    assert by_name["result"].value == 0.987


def test_sacred_non_numeric_result_is_not_recorded_as_a_metric(tmp_path):
    """Sacred's `result` can be any JSON-serialisable object -- a dict, a
    list, a string. Only a number is a metric; anything else is silently
    not one, the same refusal `_numeric_items` makes everywhere else."""
    proj = tmp_path / "p"
    d = _sacred_run(proj, 1, config={}, metrics={"auc": {"steps": [0], "values": [0.9]}})
    run_json = json.loads((d / "run.json").read_text())
    run_json["result"] = {"not": "a number"}
    (d / "run.json").write_text(json.dumps(run_json))

    (run,) = generic.discover(proj)
    assert "result" not in {m.metric for m in run.metrics}


def test_sacred_only_completed_runs_are_recorded(tmp_path):
    """A run.json with status FAILED or INTERRUPTED is still on disk after a
    crash. Recording it as a result the same way as a completed run would
    misreport a crash as a measurement."""
    proj = tmp_path / "p"
    _sacred_run(
        proj, 1, status="COMPLETED", config={}, metrics={"auc": {"steps": [0], "values": [0.9]}}
    )
    _sacred_run(
        proj, 2, status="FAILED", config={}, metrics={"auc": {"steps": [0], "values": [0.1]}}
    )

    runs = generic.discover(proj)
    assert len(runs) == 1, [r.name for r in runs]
    assert runs[0].name.endswith("/1")


def test_sacred_missing_metrics_json_is_skipped_like_mlflow(tmp_path):
    """A run with config but no metrics.json (or an empty one) is a spec with
    no measurement attached -- the same rule the MLflow reader follows for a
    run with no metric files."""
    proj = tmp_path / "p"
    _sacred_run(proj, 1, config={"lr": 0.01}, metrics=None, write_metrics=False)

    assert generic.discover(proj) == []


def test_sacred_missing_config_json_means_no_config(tmp_path):
    proj = tmp_path / "p"
    _sacred_run(
        proj,
        1,
        metrics={"auc": {"steps": [0], "values": [0.9]}},
        write_config=False,
    )

    (run,) = generic.discover(proj)
    assert run.config is None


def test_sacred_unknown_keys_in_run_json_are_ignored(tmp_path):
    """A future Sacred version, or a custom observer, adding fields to
    run.json must not raise -- the shape-tolerance rule every convention
    here follows."""
    proj = tmp_path / "p"
    d = _sacred_run(proj, 1, config={}, metrics={"auc": {"steps": [0], "values": [0.9]}})
    run_json = json.loads((d / "run.json").read_text())
    run_json["some_future_field"] = {"nested": ["stuff"]}
    (d / "run.json").write_text(json.dumps(run_json))

    (run,) = generic.discover(proj)
    assert run.name == "lr_sweep/1"


def test_the_reader_scans_the_real_committed_sacred_fixture():
    """examples/sacred/sacred_runs is real: written by sacred 0.8.7
    (generate.py, 2026-08-28), not transcribed from documentation."""
    runs = generic.discover(SACRED_EXAMPLE)
    assert len(runs) == 4, [r.name for r in runs]
    assert all(r.adapter == "sacred" for r in runs)
    assert all(r.name.startswith("lr_sweep/") for r in runs), [r.name for r in runs]
    assert all(r.family == "lr_sweep" for r in runs)
    by_lr = {r.config["lr"]: r for r in runs}
    assert set(by_lr) == {0.001, 0.01, 0.1, 1.0}
    for run in runs:
        metrics = {m.metric for m in run.metrics}
        assert {"train_loss", "auc", "result"} <= metrics, metrics


def test_detail_attaches_the_sacred_caveat(tmp_path):
    """Same rule as wandb/mlflow: a value read from metrics.json's series is
    the last one, not the curve or the best step, and runs.detail must say
    so -- ledger.ADAPTER_CAVEATS needs a "sacred" entry for this reader."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    proj.mkdir(parents=True)
    _sacred_run(proj, 1, config={"lr": 0.01}, metrics={"auc": {"steps": [0], "values": [0.9]}})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    found = ledger.detail(conn, "proj", "lr_sweep/1")

    assert found["adapter"] == "sacred"
    caveats = " ".join(found["caveats"])
    assert "sacred" in caveats
    assert "final" in caveats.lower() or "last" in caveats.lower()
    conn.close()


# --------------------------------------------------------------------------
# DVC
# --------------------------------------------------------------------------


def test_dvc_foreach_stage_expands_to_one_run_per_item(tmp_path):
    """`dvc.yaml`'s `foreach: ${lr}` over `params.yaml`'s `lr` list names one
    stage instance per item -- `train@0.01`, `train@0.1`, ... -- the way a
    real `dvc repro` does, read without ever running dvc itself."""
    proj = tmp_path / "p"
    _dvc_project(
        proj,
        items=("0.01", "0.1"),
        metrics={"0.01": {"auc": 0.9}, "0.1": {"auc": 0.95}},
    )

    runs = generic.discover(proj)
    assert {r.name for r in runs} == {"train@0.01", "train@0.1"}, [r.name for r in runs]


def test_dvc_family_is_the_stage_name_before_the_at(tmp_path):
    proj = tmp_path / "p"
    _dvc_project(proj, items=("0.01", "0.1"), metrics={"0.01": {"auc": 0.9}, "0.1": {"auc": 0.95}})

    runs = generic.discover(proj)
    assert all(r.family == "train" for r in runs), [r.family for r in runs]


def test_dvc_metrics_file_becomes_metrics(tmp_path):
    proj = tmp_path / "p"
    _dvc_project(proj, items=("0.1",), metrics={"0.1": {"auc": 0.95, "accuracy": 0.9}})

    (run,) = generic.discover(proj)
    assert {m.metric: m.value for m in run.metrics} == {"auc": 0.95, "accuracy": 0.9}


def test_dvc_config_comes_from_params_yaml_and_the_lock(tmp_path):
    """`params.yaml` names which keys the stage reads (`lr`); `dvc.lock`
    records the value actually used for each stage instance. The recorded
    value for `train@0.1` is `0.1`, not the whole `lr` list `params.yaml`
    keys -- reading only the list would put every arm's config in every
    other arm's row."""
    proj = tmp_path / "p"
    _dvc_project(
        proj,
        items=("0.01", "0.1"),
        metrics={"0.01": {"auc": 0.9}, "0.1": {"auc": 0.95}},
    )

    runs = {r.name: r for r in generic.discover(proj)}
    assert runs["train@0.1"].config.get("lr") == "0.1"
    assert runs["train@0.01"].config.get("lr") == "0.01"


def test_dvc_missing_lock_still_reads_existing_metric_files(tmp_path):
    """Tolerance: `dvc.lock` is written only after `dvc repro` runs. A stage
    declared in `dvc.yaml` whose metric file already exists on disk must
    still be read -- the lock adds config, it is not required for a run to
    be recorded."""
    proj = tmp_path / "p"
    _dvc_project(
        proj,
        items=("0.01", "0.1"),
        metrics={"0.01": {"auc": 0.9}, "0.1": {"auc": 0.95}},
        write_lock=False,
    )

    runs = generic.discover(proj)
    assert {r.name for r in runs} == {"train@0.01", "train@0.1"}, [r.name for r in runs]


def test_dvc_missing_metric_file_yields_fewer_metrics_not_fewer_runs(tmp_path):
    """Tolerance: `dvc.yaml` lists four arms; only two have run so far
    (`dvc repro` is incremental). The two with no metric file yet must not
    appear as empty or broken runs -- they are simply not there yet, the
    same rule the MLflow and Sacred readers apply to a run with nothing
    measured."""
    proj = tmp_path / "p"
    _dvc_project(
        proj,
        items=("0.01", "0.1", "1", "10"),
        metrics={"0.01": {"auc": 0.9}, "0.1": {"auc": 0.95}},
    )

    runs = generic.discover(proj)
    assert {r.name for r in runs} == {"train@0.01", "train@0.1"}, [r.name for r in runs]


def test_dvc_missing_params_yaml_means_no_config(tmp_path):
    proj = tmp_path / "p"
    _dvc_project(proj, items=("0.1",), metrics={"0.1": {"auc": 0.95}}, write_params=False)

    (run,) = generic.discover(proj)
    assert not run.config


def test_dvc_stage_declaring_no_metrics_is_not_a_run(tmp_path):
    """A `dvc.yaml` stage with no `metrics:` key is a data or preprocessing
    stage, not a measured result -- the same refusal the generic reader
    makes for a config file with no result attached."""
    proj = tmp_path / "p"
    proj.mkdir(parents=True)
    (proj / "dvc.yaml").write_text(
        "stages:\n  prepare:\n    cmd: python prepare.py\n    outs:\n      - data/clean.csv\n"
    )

    assert generic.discover(proj) == []


def test_dvc_non_foreach_stage_declaring_metrics_is_one_run(tmp_path):
    """Not every DVC stage is a `foreach` sweep -- a single named stage that
    declares `metrics:` directly is one run named for the stage itself."""
    proj = tmp_path / "p"
    proj.mkdir(parents=True)
    (proj / "dvc.yaml").write_text(
        "stages:\n"
        "  evaluate:\n"
        "    cmd: python evaluate.py\n"
        "    metrics:\n"
        "      - metrics/eval.json\n"
    )
    (proj / "metrics").mkdir()
    (proj / "metrics" / "eval.json").write_text(json.dumps({"auc": 0.93}))

    (run,) = generic.discover(proj)
    assert run.name == "evaluate"
    assert run.family is None


def test_dvc_metrics_dir_does_not_double_count_an_unrelated_file(tmp_path):
    """`metrics/` is one of the generic reader's own RESULT_DIRS -- unlike
    wandb/, mlruns/ or sacred_runs/, which live in a directory name of
    their own. A project with a real DVC sweep AND an unrelated hand-
    written `metrics/baseline.json` (not declared by any dvc.yaml stage)
    must read the DVC files exactly once, as `dvc`, and the unrelated file
    exactly once, as `generic` -- no double-count from the ordinary
    metrics/ walk re-reading a DVC output, and no over-exclusion of a file
    DVC never claimed."""
    proj = tmp_path / "p"
    _dvc_project(proj, items=("0.1",), metrics={"0.1": {"auc": 0.95}})
    (proj / "metrics" / "baseline.json").write_text(json.dumps({"wer": 0.4}))

    runs = generic.discover(proj)
    by_name = {r.name: r for r in runs}
    assert set(by_name) == {"train@0.1", "baseline"}, [r.name for r in runs]
    assert by_name["train@0.1"].adapter == "dvc"
    assert by_name["train@0.1"].family == "train"
    assert by_name["baseline"].adapter == "generic"
    assert by_name["baseline"].family != "train"


def test_dvc_metrics_list_item_with_a_trailing_comment_is_still_read(tmp_path):
    """A hand-edited dvc.yaml can carry a trailing `# comment` on a
    `metrics:` list item -- `dvc repro` itself never writes one, but a
    person editing the file by hand routinely does. Before the comment was
    stripped, the metric path was read as
    "metrics/${item}.json  # note" -- a path that never exists on disk --
    so the stage produced zero metrics, the DVC run vanished, and
    metrics/0.1.json reappeared as a family-less generic run: a wrong run,
    not merely a missing one. The fix must produce the real DVC run, not
    just avoid a crash.
    """
    proj = tmp_path / "p"
    proj.mkdir(parents=True)
    (proj / "dvc.yaml").write_text(
        "stages:\n"
        "  train:\n"
        "    foreach: ${lr}\n"
        "    do:\n"
        "      cmd: python train.py ${item}\n"
        "      params:\n"
        "        - lr\n"
        "      metrics:\n"
        "        - metrics/${item}.json  # note\n"
    )
    (proj / "params.yaml").write_text("lr: [0.1]\n")
    (proj / "metrics").mkdir()
    (proj / "metrics" / "0.1.json").write_text(json.dumps({"auc": 0.95}))

    (run,) = generic.discover(proj)
    assert run.name == "train@0.1"
    assert run.adapter == "dvc"
    assert run.family == "train"


def test_dvc_flow_style_metrics_list_is_still_read(tmp_path):
    """A hand-edited dvc.yaml can declare `metrics: [a, b]` inline instead
    of DVC's own block style (`metrics:` then `- a` on the next line) --
    both are legal YAML, and `dvc repro` reads either. Before the flow
    style was parsed, `_list_after` only ever looked for block-style items
    following the key, so `metrics:` with an inline `[...]` value read as
    an empty list -- the stage was excluded as having no metrics entirely,
    the DVC run vanished, and metrics/0.1.json reappeared as a family-less
    generic run.
    """
    proj = tmp_path / "p"
    proj.mkdir(parents=True)
    (proj / "dvc.yaml").write_text(
        "stages:\n"
        "  train:\n"
        "    foreach: ${lr}\n"
        "    do:\n"
        "      cmd: python train.py ${item}\n"
        "      params: [lr]\n"
        "      metrics: [metrics/${item}.json]\n"
    )
    (proj / "params.yaml").write_text("lr: [0.1]\n")
    (proj / "metrics").mkdir()
    (proj / "metrics" / "0.1.json").write_text(json.dumps({"auc": 0.95}))

    (run,) = generic.discover(proj)
    assert run.name == "train@0.1"
    assert run.adapter == "dvc"
    assert run.family == "train"
    assert run.config.get("lr") == "0.1"


@pytest.mark.parametrize(
    "quoted",
    [
        "lr: '1 # not a comment'",
        'note: "see docs # section 3"',
    ],
    ids=["single-quoted", "double-quoted"],
)
def test_a_hash_inside_a_quoted_value_is_not_a_comment(quoted):
    """`_strip_inline_comment` first stripped a trailing `# ...` with a
    plain whitespace-then-hash regex, which does not know about quoting --
    `lr: '1 # not a comment'` truncated to `lr: '1`, and `note: "see docs #
    section 3"` truncated to `note: "see docs`, both silently. DVC's own
    writer never quotes a scalar (this module's long-standing assumption),
    but a hand-edited dvc.yaml or params.yaml can, so the value must
    survive whichever quote style wraps it."""
    assert generic._strip_inline_comment(quoted) == quoted


@pytest.mark.parametrize(
    ("quoted", "expected"),
    [
        ("lr: '1 # not a comment'  # real comment", "lr: '1 # not a comment'"),
        ('note: "see docs # section 3"  # real comment', 'note: "see docs # section 3"'),
    ],
    ids=["single-quoted", "double-quoted"],
)
def test_a_real_trailing_comment_after_a_quoted_value_is_still_stripped(quoted, expected):
    """The quote-aware fix must not overcorrect into never stripping a
    comment at all -- a genuine `# comment` after a closed quote is still a
    comment and must still go, whichever quote style precedes it."""
    assert generic._strip_inline_comment(quoted) == expected


def test_a_quoted_hash_in_params_yaml_survives_through_the_dvc_reader(tmp_path):
    """End-to-end version of the two tests above, through the actual
    params.yaml parsing path (`_dvc_params_yaml`, built on
    `_indented_lines`/`_strip_inline_comment`) rather than the helper in
    isolation -- proves the fix reaches the code path the bug was found in,
    not just the unit that was patched."""
    proj = tmp_path / "p"
    _dvc_project(proj, items=("0.1",), metrics={"0.1": {"auc": 0.95}})
    params_path = proj / "params.yaml"
    params_path.write_text(params_path.read_text() + 'note: "see docs # section 3"\n')

    parsed = generic._dvc_params_yaml(params_path)
    assert parsed["note"] == '"see docs # section 3"'

    (run,) = generic.discover(proj)
    assert run.name == "train@0.1"
    assert run.adapter == "dvc"


def test_the_reader_scans_the_real_committed_dvc_fixture():
    """examples/dvc/ is real: written by `dvc repro` (dvc 3.x, generate.sh,
    2026-08-28), not transcribed from documentation."""
    runs = generic.discover(DVC_EXAMPLE)
    assert len(runs) == 4, [r.name for r in runs]
    assert all(r.adapter == "dvc" for r in runs)
    assert all(r.name.startswith("train@") for r in runs), [r.name for r in runs]
    assert all(r.family == "train" for r in runs)
    by_lr = {r.config["lr"]: r for r in runs}
    assert set(by_lr) == {"0.01", "0.1", "1", "10"}
    for run in runs:
        metrics = {m.metric for m in run.metrics}
        assert {"accuracy", "precision", "recall", "auc"} <= metrics, metrics


def test_detail_attaches_the_dvc_caveat(tmp_path):
    """Same rule as wandb/mlflow/sacred: `ledger.ADAPTER_CAVEATS` needs a
    "dvc" entry, or `runs.compare` ranks DVC arms with no caveat while every
    other tracker carries one."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    _dvc_project(proj, items=("0.1",), metrics={"0.1": {"auc": 0.95}})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    found = ledger.detail(conn, "proj", "train@0.1")

    assert found["adapter"] == "dvc"
    caveats = " ".join(found["caveats"])
    assert "dvc" in caveats
    conn.close()


# --------------------------------------------------------------------------
# Hydra
# --------------------------------------------------------------------------


def test_hydra_multirun_sweep_becomes_one_run_per_arm(tmp_path):
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, config={"lr": 0.01}, metrics={"auc": 0.9})
    _hydra_arm(proj, 1, config={"lr": 0.1}, metrics={"auc": 0.95})

    runs = generic.discover(proj)
    assert {r.name for r in runs} == {"train/0", "train/1"}, [r.name for r in runs]


def test_hydra_family_is_job_name_from_hydra_yaml(tmp_path):
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, job_name="sweep", config={"lr": 0.01}, metrics={"auc": 0.9})

    (run,) = generic.discover(proj)
    assert run.family == "sweep"
    assert run.name == "sweep/0"


def test_hydra_metrics_json_becomes_metrics(tmp_path):
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, config={"lr": 0.1}, metrics={"auc": 0.95, "accuracy": 0.9})

    (run,) = generic.discover(proj)
    assert {m.metric: m.value for m in run.metrics} == {"auc": 0.95, "accuracy": 0.9}


def test_hydra_config_comes_from_hydra_config_yaml(tmp_path):
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, config={"lr": 0.1}, metrics={"auc": 0.95})

    (run,) = generic.discover(proj)
    assert run.config == {"lr": 0.1}


def test_hydra_missing_hydra_yaml_falls_back_to_the_sweep_directory_as_family(tmp_path):
    """Tolerance: an older Hydra version, or a directory edited by hand,
    might not write `.hydra/hydra.yaml`. `_hydra_runs` must still record the
    arm -- named after the sweep directory it sits under, since there is no
    `hydra.job.name` to read."""
    proj = tmp_path / "p"
    _hydra_arm(
        proj,
        0,
        sweep_date="2026-01-01",
        sweep_time="10-00-00",
        config={"lr": 0.1},
        metrics={"auc": 0.95},
        write_hydra_yaml=False,
    )

    (run,) = generic.discover(proj)
    assert run.family == "10-00-00"
    assert run.name == "10-00-00/0"


def test_hydra_arm_with_no_metrics_file_is_skipped(tmp_path):
    """A spec with no measurement attached -- same refusal as MLflow/Sacred/
    DVC: `.hydra/config.yaml` alone, with no metrics.json or any other
    JSON/CSV file in the arm directory, is not a run."""
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, config={"lr": 0.1}, metrics=None)

    assert generic.discover(proj) == []


def test_hydra_two_sweeps_of_the_same_job_qualify_the_second_with_its_time_dir(tmp_path):
    """Two sweeps of `train` both produce an arm `0` -- `train/0` would
    collide. The second sweep's arm 0 is re-qualified with its own time
    directory rather than silently dropped, the same `seen`-based dedup
    every other tracker reader here participates in."""
    proj = tmp_path / "p"
    _hydra_arm(proj, 0, sweep_date="2026-01-01", sweep_time="10-00-00", metrics={"auc": 0.9})
    _hydra_arm(proj, 0, sweep_date="2026-01-01", sweep_time="11-00-00", metrics={"auc": 0.95})

    runs = generic.discover(proj)
    assert {r.name for r in runs} == {"train/0", "train/11-00-00/0"}, [r.name for r in runs]


def test_hydra_metrics_come_from_a_csv_too(tmp_path):
    """The brief's own tolerance: metrics come from any JSON/CSV file in the
    arm directory, not only `metrics.json` -- reusing the same
    `_csv_rows`/`metrics_from_payload` path the ordinary `results/` scan
    uses, not a Hydra-specific format. A single-row CSV goes through the
    same "independent samples" aggregation as any other CSV `generic` reads
    (mean plus `_std` plus `n_records`, `_std` a guaranteed 0.0 for one row)
    -- this is `metrics_from_payload`'s own existing behaviour, not
    something `_hydra_runs` special-cases."""
    proj = tmp_path / "p"
    arm_dir = _hydra_arm(proj, 0, config={"lr": 0.1}, metrics=None)
    (arm_dir / "metrics.csv").write_text("auc,accuracy\n0.95,0.9\n")

    (run,) = generic.discover(proj)
    values = {m.metric: m.value for m in run.metrics}
    assert values["auc"] == 0.95
    assert values["accuracy"] == 0.9
    assert values["n_records"] == 1.0


def test_the_reader_scans_the_real_committed_hydra_fixture():
    """examples/hydra/multirun is real: written by hydra-core 1.3.5
    (generate.sh, 2026-08-28), not transcribed from documentation. One extra
    `generic`-adapter run is expected alongside the four Hydra arms:
    examples/hydra/conf/ is itself one of `CONFIG_DIRS`, so `conf/
    config.yaml` -- Hydra's own input, not a Hydra output -- is read as an
    ordinary config spec named `config` by the ordinary CONFIG_DIRS scan,
    the same honest "a spec with no result attached" every config file
    gets; see the README's own note on this."""
    runs = generic.discover(HYDRA_EXAMPLE)
    hydra_runs = [r for r in runs if r.adapter == "hydra"]
    assert len(hydra_runs) == 4, [r.name for r in runs]
    assert all(r.name.startswith("train/") for r in hydra_runs), [r.name for r in hydra_runs]
    assert all(r.family == "train" for r in hydra_runs)
    by_lr = {r.config["lr"]: r for r in hydra_runs}
    assert set(by_lr) == {0.01, 0.1, 1, 10}
    for run in hydra_runs:
        metrics = {m.metric for m in run.metrics}
        assert {"accuracy", "precision", "recall", "auc"} <= metrics, metrics


def test_detail_attaches_the_hydra_caveat(tmp_path):
    """Same rule as wandb/mlflow/sacred/dvc: `ledger.ADAPTER_CAVEATS` needs a
    "hydra" entry, or `runs.compare` ranks Hydra arms with no caveat while
    every other tracker carries one."""
    from attestation import ledger
    from attestation.db import get_db

    ws = tmp_path / "ws"
    proj = ws / "proj"
    _hydra_arm(proj, 0, config={"lr": 0.1}, metrics={"auc": 0.95})

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    found = ledger.detail(conn, "proj", "train/0")

    assert found["adapter"] == "hydra"
    caveats = " ".join(found["caveats"])
    assert "hydra" in caveats
    conn.close()


def test_yaml_path_scalar_reads_a_nested_key(tmp_path):
    """`_yaml_path_scalar`, the small generalisation of `_yaml_scalars` this
    reader needed for `hydra.job.name` (three levels deep, unlike the flat
    top-level keys `meta.yaml`/`config.yaml` use)."""
    lines = generic._indented_lines("hydra:\n  job:\n    name: train\n")
    assert generic._yaml_path_scalar(lines, ("hydra", "job", "name")) == "train"
    assert generic._yaml_path_scalar(lines, ("hydra", "job", "missing")) is None
    assert generic._yaml_path_scalar(lines, ("nope",)) is None


def test_yaml_path_list_reads_hydras_same_indent_block_list(tmp_path):
    """Hydra's own dumper writes a block list's `- item` lines at the SAME
    indent as the key introducing them (`task:` and `- lr=0.01` both at
    indent 4), not one level deeper as DVC's writer does for `metrics:` --
    found only by running a real sweep, not by assuming DVC's own style
    generalised."""
    text = "hydra:\n  overrides:\n    task:\n    - lr=0.01\n    - seed=3\n  job:\n    name: train\n"
    lines = generic._indented_lines(text)
    assert generic._yaml_path_list(lines, ("hydra", "overrides", "task")) == ["lr=0.01", "seed=3"]
    assert generic._yaml_path_list(lines, ("hydra", "overrides", "missing")) == []


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


def test_a_sacred_run_colliding_with_a_results_run_is_recorded_once(tmp_path):
    """Same rule, for Sacred's own naming: `lr_sweep/1` collides with a
    results/ file that happens to share that stem."""
    proj = tmp_path / "p"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "lr_sweep").mkdir()
    (proj / "results" / "lr_sweep" / "1.json").write_text(json.dumps({"wer": 0.4}))
    _sacred_run(proj, 1, config={}, metrics={"auc": {"steps": [0], "values": [0.9]}})

    runs = generic.discover(proj)
    names = [r.name for r in runs]
    assert names.count("lr_sweep/1") == 1, names


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
        "hydra-yaml-garbage",
        "hydra-metrics-not-json",
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
    elif broken == "hydra-yaml-garbage":
        arm_dir = _hydra_arm(proj, 0, config={"lr": 0.1}, metrics={"auc": 0.9})
        (arm_dir / ".hydra" / "hydra.yaml").write_text("not: [valid, yaml: at all")
    elif broken == "hydra-metrics-not-json":
        arm_dir = _hydra_arm(proj, 0, config={"lr": 0.1}, metrics=None)
        (arm_dir / "metrics.json").write_text("<!DOCTYPE html>not json")

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
