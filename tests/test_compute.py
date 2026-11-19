"""Compute tracking: wall time per arm and the not-compute-matched caveat.

See docs/superpowers/specs/2026-11-19-compute-tracking-design.md. WALL TIME IS
NOT COMPUTE: every assertion here is about seconds on a clock, never a cost.
Each guard was mutation-checked (measurement-lessons section 4): removing the
`compute.caveats` call from `ledger._all_caveats` fails the ratio tests; flipping
`>=` to `>` in `_ratio_caveat` fails the exactly-2.0x test; summing only the
named runs in `compute.attach` fails the seven-run test; making the "not
recorded" caveat unconditional fails the no-timing tests.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from attestation import compute, ledger
from attestation.db import get_db
from attestation.ledger_adapters import generic
from attestation.run_record import build, keys, verify

ROOT = Path(__file__).resolve().parent.parent


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run_arm(name, value, wall=None, **extra):
    """A per-run arm dict shaped like ledger._arms_for_run's."""
    return {
        "name": name,
        "status": "recorded",
        "value": value,
        "step": None,
        "split": None,
        "source_path": f"/r/{name}.json",
        "n": 100,
        **extra,
    }


def timed(name, value, wall, **extra):
    """A ranked arm that already carries its wall time."""
    return run_arm(name, value, **extra) | {"wall_time_s": wall}


def scan_dir(tmp_path, files: dict[str, dict]):
    """Write `results/<name>.json` files and scan them into a fresh ledger."""
    for name, payload in files.items():
        write(tmp_path / "proj" / "results" / f"{name}.json", json.dumps(payload))
    conn = get_db(tmp_path / "db.sqlite")
    ledger.scan(conn, tmp_path)
    return conn


# --- parsers -------------------------------------------------------------------


def test_iso_duration_is_stop_minus_start():
    a, b = "2026-08-28T10:00:00.000000", "2026-08-28T10:00:10.500000"
    assert compute.iso_duration(a, b) == 10.5
    assert compute.iso_duration(b, a) is None, "a stop before the start is not a duration"
    assert compute.iso_duration(a, None) is None
    assert compute.iso_duration("yesterday", b) is None
    assert compute.iso_duration(a, "2026-08-28T10:00:10+00:00") is None, "naive vs aware"


def test_epoch_ms_duration_refuses_a_run_that_has_not_ended():
    assert compute.epoch_ms_duration(1000, 61000) == 60.0
    assert compute.epoch_ms_duration("1000", "3500") == 2.5, "YAML scalars arrive as strings"
    assert compute.epoch_ms_duration(1000, 0) is None
    assert compute.epoch_ms_duration(1000, None) is None
    assert compute.epoch_ms_duration(5000, 1000) is None


def test_seconds_rejects_what_is_not_a_duration():
    for bad in (True, None, "fast", -1, float("nan"), float("inf")):
        assert compute.seconds(bad) is None, bad
    assert compute.seconds("12") == 12.0
    assert compute.seconds(0) == 0.0


def test_run_wall_prefers_wall_time_and_takes_the_largest_row():
    rows = [
        {"metric": "runtime_s", "value": 9.0},
        {"metric": "wall_time_s", "value": 3.0},
        {"metric": "wall_time_s", "value": 7.0},
        {"metric": "accuracy", "value": 0.9},
    ]
    assert compute.run_wall(rows) == 7.0
    assert compute.run_wall(rows[:1]) == 9.0, "runtime_s is an accepted synonym"
    assert compute.run_wall(rows[3:]) is None


# --- tracker adapters ------------------------------------------------------------


def _sacred(root, n, start, stop, result=0.5):
    d = root / "sacred_runs" / str(n)
    d.mkdir(parents=True)
    run = {
        "experiment": {"name": "lr_sweep"},
        "status": "COMPLETED",
        "start_time": start,
        "stop_time": stop,
        "result": result,
    }
    (d / "run.json").write_text(json.dumps(run))
    return d


def _wall_of(record):
    return [m.value for m in record.metrics if m.metric == compute.WALL_METRIC]


def test_sacred_wall_time_is_stop_minus_start(tmp_path):
    proj = tmp_path / "p"
    _sacred(proj, 1, "2026-08-28T10:00:00.000000", "2026-08-28T10:02:30.250000")
    (run,) = generic.discover(proj)
    assert _wall_of(run) == [150.25]


def test_sacred_without_a_stop_time_records_no_wall_time(tmp_path):
    proj = tmp_path / "p"
    _sacred(proj, 1, "2026-08-28T10:00:00.000000", None)
    (run,) = generic.discover(proj)
    assert _wall_of(run) == []


def _mlflow(root, run_id, start, end, value=0.5):
    d = root / "mlruns" / "0" / run_id
    (d / "metrics").mkdir(parents=True)
    (d / "metrics" / "accuracy").write_text(f"1000 {value} 0\n")
    end_line = f"end_time: {end}\n" if end is not None else ""
    (d / "meta.yaml").write_text(
        f"{end_line}lifecycle_stage: active\nrun_id: {run_id}\nrun_name: r\n"
        f"start_time: {start}\nstatus: FINISHED\n"
    )


def test_mlflow_wall_time_is_end_minus_start_in_ms(tmp_path):
    proj = tmp_path / "p"
    _mlflow(proj, "abc", 1_755_000_000_000, 1_755_000_090_000)
    (run,) = generic.discover(proj)
    assert _wall_of(run) == [90.0]


def test_mlflow_run_still_going_has_no_wall_time(tmp_path):
    proj = tmp_path / "p"
    _mlflow(proj, "abc", 1_755_000_000_000, None)
    (run,) = generic.discover(proj)
    assert _wall_of(run) == []


def test_a_tracker_metric_already_named_wall_time_is_not_shadowed(tmp_path):
    proj = tmp_path / "p"
    _mlflow(proj, "abc", 1_755_000_000_000, 1_755_000_090_000)
    d = proj / "mlruns" / "0" / "abc" / "metrics"
    (d / "wall_time_s").write_text("1000 12.5 3\n")
    (run,) = generic.discover(proj)
    assert _wall_of(run) == [12.5]


def test_wandb_runtime_is_read_before_the_underscore_filter(tmp_path):
    proj = tmp_path / "p"
    files = proj / "wandb" / "offline-run-20260828_000000-abcd1234" / "files"
    write(files / "wandb-summary.json", json.dumps({"loss": 0.1, "_runtime": 42.5, "_step": 3}))
    (run,) = generic.discover(proj)
    assert _wall_of(run) == [42.5]
    assert {m.metric for m in run.metrics} == {"loss", compute.WALL_METRIC}


def test_hydra_and_dvc_record_no_wall_time():
    for example in ("hydra", "dvc"):
        for run in generic.discover(ROOT / "examples" / example):
            assert _wall_of(run) == [], (example, run.name)


def test_an_explicit_key_in_a_plain_result_is_the_wall_time(tmp_path):
    proj = tmp_path / "p"
    write(proj / "results" / "a_seed1.json", json.dumps({"accuracy": 0.9, "wall_time_s": 33.0}))
    (run,) = generic.discover(proj)
    assert _wall_of(run) == [33.0]


# --- per arm -----------------------------------------------------------------------


def test_compare_reports_each_runs_wall_time_from_the_ledger(tmp_path):
    conn = scan_dir(
        tmp_path,
        {
            "m_fast": {"accuracy": 0.90, "wall_time_s": 10.0},
            "m_slow": {"accuracy": 0.80, "wall_time_s": 12.0},
        },
    )
    out = ledger.compare(conn, "m", metric="accuracy")
    assert {a["name"]: a["wall_time_s"] for a in out["arms"]} == {"m_fast": 10.0, "m_slow": 12.0}


def test_a_pooled_arm_sums_wall_time_over_all_its_runs_not_the_five_named(tmp_path):
    """The mutation to catch: summing `arm["runs"]` (capped at 5) gives 50."""
    files = {}
    for i in range(1, 8):
        files[f"fam_a_seed{i}"] = {"seed": i, "accuracy": 0.90 + i / 1000, "wall_time_s": 10.0 * i}
        files[f"fam_b_seed{i}"] = {"seed": i, "accuracy": 0.80 + i / 1000, "wall_time_s": 10.0}
    out = ledger.compare(scan_dir(tmp_path, files), "fam", metric="accuracy")
    a, b = (
        {x["name"]: x for x in out["arms"]}["fam_a"],
        {x["name"]: x for x in out["arms"]}["fam_b"],
    )
    assert a["n_runs"] == 7 and len(a["runs"]) == 5
    assert a["wall_time_s"] == 280.0, "10+20+...+70, all seven runs"
    assert a["wall_time_mean_s"] == 40.0
    assert b["wall_time_s"] == 70.0


def test_one_untimed_run_withholds_the_pooled_total(tmp_path):
    """A sum over part of the runs would read as the arm's whole cost."""
    files = {
        "fam_a_seed1": {"seed": 1, "accuracy": 0.91, "wall_time_s": 10.0},
        "fam_a_seed2": {"seed": 2, "accuracy": 0.92},
        "fam_a_seed3": {"seed": 3, "accuracy": 0.93, "wall_time_s": 10.0},
        "fam_b_seed1": {"seed": 1, "accuracy": 0.81, "wall_time_s": 10.0},
        "fam_b_seed2": {"seed": 2, "accuracy": 0.82, "wall_time_s": 10.0},
        "fam_b_seed3": {"seed": 3, "accuracy": 0.83, "wall_time_s": 10.0},
    }
    out = ledger.compare(scan_dir(tmp_path, files), "fam", metric="accuracy")
    by = {x["name"]: x for x in out["arms"]}
    assert "wall_time_s" not in by["fam_a"] and "wall_time_mean_s" not in by["fam_a"]
    assert by["fam_b"]["wall_time_s"] == 30.0
    assert any("compute not recorded for fam_a" in c for c in out["caveats"])


def test_a_nested_arm_gets_no_wall_time_from_its_file(tmp_path):
    payload = {
        "wall_time_s": 20.0,
        "arms": {"A": {"accuracy": 0.9, "n_records": 50}, "B": {"accuracy": 0.8, "n_records": 50}},
    }
    out = ledger.compare(scan_dir(tmp_path, {"nest_a": payload}), "nest", metric="accuracy")
    assert all("wall_time_s" not in a for a in out["arms"]), out["arms"]


def test_no_timing_means_no_new_key_and_no_new_caveat(tmp_path):
    files = {f"m_{k}": {"accuracy": v} for k, v in (("a", 0.9), ("b", 0.8))}
    out = ledger.compare(scan_dir(tmp_path, files), "m", metric="accuracy")
    assert all(not k.startswith("wall_time") for a in out["arms"] for k in a)
    assert not any("wall time" in c or "compute" in c for c in out["caveats"])


# --- the caveat ----------------------------------------------------------------------


def test_three_x_in_wall_time_warns_and_says_it_is_not_compute():
    out = compute.caveats([timed("a", 0.9, 300.0), timed("b", 0.8, 100.0)])
    (text,) = out
    assert "a 300s" in text and "b 100s" in text and "3.0x" in text
    assert "wall time is not compute" in text.lower()
    assert "compute-matched" in text


def test_one_point_two_x_does_not_warn():
    assert compute.caveats([timed("a", 0.9, 120.0), timed("b", 0.8, 100.0)]) == []


def test_exactly_the_threshold_warns():
    out = compute.caveats([timed("a", 0.9, 200.0), timed("b", 0.8, 100.0)])
    assert len(out) == 1 and "2.0x" in out[0]


def test_just_under_the_threshold_does_not_warn():
    assert compute.caveats([timed("a", 0.9, 199.0), timed("b", 0.8, 100.0)]) == []


def test_the_faster_arm_may_be_the_winner():
    """Direction of the mismatch does not matter: a cheap winner is as unmatched."""
    out = compute.caveats([timed("a", 0.9, 100.0), timed("b", 0.8, 300.0)])
    assert len(out) == 1 and "3.0x" in out[0]


def test_sub_second_runs_are_jitter_not_a_budget_difference():
    assert compute.caveats([timed("a", 0.9, 0.09), timed("b", 0.8, 0.02)]) == []


def test_only_the_top_two_arms_are_compared():
    arms = [timed("a", 0.9, 100.0), timed("b", 0.8, 110.0), timed("c", 0.7, 900.0)]
    assert compute.caveats(arms) == []


def test_ratio_uses_time_per_run_so_replicate_counts_do_not_trip_it():
    """5 seeds cost 5x one seed by construction; per-run time is what matches."""
    pooled = timed("a", 0.9, 500.0, n_runs=5, wall_time_mean_s=100.0)
    single = timed("b", 0.8, 100.0)
    assert compute.caveats([pooled, single]) == []
    pooled_slow = timed("a", 0.9, 1500.0, n_runs=5, wall_time_mean_s=300.0)
    assert len(compute.caveats([pooled_slow, single])) == 1


def test_untimed_arms_are_named_only_when_another_arm_is_timed():
    one = [run_arm("a", 0.9), run_arm("b", 0.8)]
    assert compute.caveats(one) == []
    out = compute.caveats([timed("a", 0.9, 10.0), run_arm("b", 0.8), run_arm("c", 0.7)])
    assert len(out) == 1 and "compute not recorded for b, c" in out[0]


def test_a_zero_second_arm_is_an_infinite_ratio_not_a_crash():
    out = compute.caveats([timed("a", 0.9, 0.0), timed("b", 0.8, 5.0)])
    assert len(out) == 1 and "infx" in out[0]


def test_compare_carries_the_caveat_end_to_end(tmp_path):
    files = {
        "m_big": {"accuracy": 0.90, "wall_time_s": 600.0},
        "m_small": {"accuracy": 0.85, "wall_time_s": 200.0},
    }
    out = ledger.compare(scan_dir(tmp_path, files), "m", metric="accuracy")
    texts = [c for c in out["caveats"] if "wall time per run" in c]
    assert len(texts) == 1 and "3.0x" in texts[0] and "not compute" in texts[0]
    files["m_small"]["wall_time_s"] = 500.0
    out = ledger.compare(scan_dir(tmp_path / "x", files), "m", metric="accuracy")
    assert not any("wall time per run" in c for c in out["caveats"])


# --- real fixtures ---------------------------------------------------------------------


def test_the_committed_sacred_fixture_reports_its_real_durations():
    """examples/sacred was written by sacred 0.8.7; the numbers are start/stop
    times in the committed run.json files, not constructed here."""
    runs = generic.discover(ROOT / "examples" / "sacred")
    assert len(runs) == 4 and all(len(_wall_of(r)) == 1 for r in runs)
    for r in runs:
        run_json = json.loads((Path(r.source_path) / "run.json").read_text())
        want = compute.iso_duration(run_json["start_time"], run_json["stop_time"])
        assert _wall_of(r) == [want]


def test_the_committed_mlflow_fixture_reports_its_real_durations():
    runs = generic.discover(ROOT / "examples" / "flows" / "training")
    assert runs
    for r in runs:
        meta = (Path(r.source_path) / "meta.yaml").read_text()
        kv = dict(line.split(": ", 1) for line in meta.splitlines() if ": " in line)
        want = (int(kv["end_time"]) - int(kv["start_time"])) / 1000
        assert _wall_of(r) == [pytest.approx(want)]


def test_committed_fixtures_are_too_short_to_trigger_the_compute_caveat(tmp_path):
    """Measured: the sacred sweep runs 24-155 ms; the 2x rule alone would call
    that a mismatch, which is why MIN_WALL_S exists."""
    conn = get_db(tmp_path / "db.sqlite")
    ledger.scan(conn, ROOT / "examples" / "sacred")
    (fam,) = ledger.families(conn)
    out = ledger.compare(conn, fam["family"])
    assert any("wall_time_s" in a for a in out["arms"])
    assert not any("wall time per run" in c for c in out["caveats"])


# --- run record environment -------------------------------------------------------------


@pytest.fixture
def signer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv(keys.ENV_KEY, raising=False)
    return keys.resolve_key_source().signer()


def _record(tmp_path, signer):
    root = tmp_path / "work"
    root.mkdir(exist_ok=True)
    (root / "draft.md").write_text("WER is 5.3%.\n")
    rec = build.RunRecorder("t", root=root)
    rec.add_input("draft.md", "draft")
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json", force=True)
    return root, out


def _environment(out):
    import base64

    env = json.loads(out.read_text())
    return json.loads(base64.b64decode(env["payload"]))["predicate"]["environment"]


def test_the_record_verifies_and_carries_the_environment(tmp_path, signer, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    root, out = _record(tmp_path, signer)
    (root / "uv.lock").write_text("lock\n")
    root, out = _record(tmp_path, signer)
    assert verify.verify_file(out, root=root).ok
    env = _environment(out)
    assert env["python"].split()[1].count(".") == 2
    assert env["argv"] and all(isinstance(a, str) for a in env["argv"])
    assert env["uv_lock_sha256"] == verify.sha256_file(root / "uv.lock")
    assert env["gpus"] == []


def test_a_workspace_without_a_lockfile_records_null(tmp_path, signer, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    _, out = _record(tmp_path, signer)
    assert _environment(out)["uv_lock_sha256"] is None


def test_an_nvidia_smi_that_exists_is_read(tmp_path, signer, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    real = subprocess.run

    def fake(cmd, **kw):
        if cmd[0] != "/usr/bin/nvidia-smi":
            return real(cmd, **kw)  # the record's git probe is not under test
        assert cmd[1] == "--query-gpu=name"
        return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA A100\nNVIDIA A100\n\n")

    monkeypatch.setattr(subprocess, "run", fake)
    _, out = _record(tmp_path, signer)
    assert _environment(out)["gpus"] == ["NVIDIA A100", "NVIDIA A100"]


@pytest.mark.parametrize(
    "failure",
    [
        OSError("exec format error"),
        subprocess.TimeoutExpired("nvidia-smi", 5),
        subprocess.CalledProcessError(9, "nvidia-smi"),
    ],
)
def test_a_failing_gpu_probe_never_fails_the_record(tmp_path, signer, monkeypatch, failure):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    real = subprocess.run

    def boom(cmd, **kw):
        if cmd[0] != "/usr/bin/nvidia-smi":
            return real(cmd, **kw)
        raise failure

    monkeypatch.setattr(subprocess, "run", boom)
    root, out = _record(tmp_path, signer)
    assert verify.verify_file(out, root=root).ok
    assert _environment(out)["gpus"] == []


def test_a_long_command_line_is_capped(tmp_path, signer, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr("sys.argv", ["x" * 500] + [f"a{i}" for i in range(60)])
    _, out = _record(tmp_path, signer)
    argv = _environment(out)["argv"]
    assert len(argv) == build.MAX_ARGV
    assert len(argv[0]) == build.MAX_ARG_CHARS
