"""Roadmap 16: the W&B and Hydra adapters read the whole run, as MLflow's does.

The committed fixtures are real: examples/wandb/wandb (wandb 0.17.6, its
`.wandb` logs kept and scrubbed by generate.py through wandb's own writer)
and examples/hydra/multirun (hydra-core 1.3.5). Every field of their core
records is pinned here. The synthetic logs below are framed by a small
writer that is first shown to reproduce a committed log byte for byte, so
the writer is not the only witness to the format; where wandb is importable
(it is not a dev dependency), the decoder is also checked against wandb's
own reader and field numbers.

Spec: docs/superpowers/specs/2026-11-20-wandb-hydra-adapters-design.md.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import attestation_schema as schema
import pytest

from attestation import ledger
from attestation.db import get_db
from attestation.ledger_adapters import (
    IMPORT_NAMESPACE,
    hydra_store,
    import_project,
    trackers,
    wandb_log,
    wandb_store,
)

ROOT = Path(__file__).resolve().parents[1]
WANDB_EXAMPLE = ROOT / "examples" / "wandb"
HYDRA_EXAMPLE = ROOT / "examples" / "hydra"
LOGS = sorted((WANDB_EXAMPLE / "wandb").glob("*/*.wandb"))
SWEEP = "2026-08-28/23-45-43"


# ---- a minimal log writer (validated against a real log below) -------------------


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, payload: bytes | str) -> bytes:
    data = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def _int_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _item(key: str | list[str], value) -> bytes:
    path = _field(1, key) if isinstance(key, str) else b"".join(_field(2, k) for k in key)
    return path + _field(16, json.dumps(value))


def _updates(updates: dict, removes: tuple = ()) -> bytes:
    return b"".join(_field(1, _item(k, v)) for k, v in updates.items()) + b"".join(
        _field(2, _item(k, None)) for k in removes
    )


def run_record(run_id="abc123", project="proj", config=None, commit=None, remote=None) -> bytes:
    body = _field(1, run_id) + _field(3, project) + _field(8, "display")
    body += _field(4, _updates({**(config or {}), "_wandb": {}}))
    body += _field(17, _int_field(1, 1_795_000_000) + _int_field(2, 123_456_000))
    if commit or remote:
        body += _field(21, (_field(1, remote) if remote else b"") + _field(2, commit or ""))
    return _field(17, body)


def history_record(step: int, values: dict, ts: float = 1_795_000_001.5) -> bytes:
    items = {**values, "_step": step, "_timestamp": ts + step, "_runtime": 0.1 * step}
    return _field(2, b"".join(_field(1, _item(k, v)) for k, v in items.items()))


def summary_record(updates: dict, removes: tuple = ()) -> bytes:
    return _field(3, _updates(updates, removes))


def exit_record(code: int = 0) -> bytes:
    return _field(18, _int_field(1, code) if code else b"")


def frame(payloads: list[bytes]) -> bytes:
    """LevelDB log framing as wandb's DataStore writes it."""
    out = bytearray(struct.pack("<4sHB", b":W&B", 0xBEE1, 0))
    for payload in payloads:
        left, first = payload, True
        while True:
            space = 32768 - len(out) % 32768
            if space < 7:
                out += b"\0" * space
                continue
            chunk, left = left[: space - 7], left[space - 7 :]
            kind = (1 if not left else 2) if first else (4 if not left else 3)
            seed = zlib.crc32(bytes([kind])) & 0xFFFFFFFF
            crc = zlib.crc32(chunk, seed) & 0xFFFFFFFF
            out += struct.pack("<IHB", crc, len(chunk), kind) + chunk
            first = False
            if not left:
                break
    return bytes(out)


def wandb_dir(
    root: Path,
    payloads: list[bytes] | None,
    *,
    name="offline-run-20261120_000000-abc123",
    program="train.py",
    summary=None,
    config_yaml=None,
    raw: bytes | None = None,
) -> Path:
    run = root / "wandb" / name
    files = run / "files"
    files.mkdir(parents=True)
    (files / "wandb-metadata.json").write_text(
        json.dumps({"program": program, "startedAt": "2026-11-20T00:00:00.000001"})
    )
    if raw is not None or payloads is not None:
        (run / f"run-{name.rsplit('-', 1)[-1]}.wandb").write_bytes(
            raw if raw is not None else frame(payloads or [])
        )
    if summary is not None:
        (files / "wandb-summary.json").write_text(json.dumps(summary))
    if config_yaml is not None:
        (files / "config.yaml").write_text(config_yaml)
    return run


def a_run(*extra: bytes, exit_code: int | None = 0, summary=None) -> list[bytes]:
    out = [run_record(config={"lr": 0.1}), history_record(0, {"loss": 1.0})]
    out += [history_record(1, {"loss": 0.5}), *extra]
    out.append(summary_record(summary or {"loss": 0.5, "acc": 0.9, "_runtime": 2.5}))
    if exit_code is not None:
        out.append(exit_record(exit_code))
    return out


def record_of(root: Path) -> schema.RunRecord:
    result = import_project(root)
    assert result.refused == [], result.refused
    (imported,) = result.imported
    return imported.record


def run_caveats(record: schema.RunRecord) -> list[str]:
    return record.ext[IMPORT_NAMESPACE]["run_caveats"]


# ---- the format itself ---------------------------------------------------------


def test_the_test_writer_reproduces_a_committed_log_byte_for_byte():
    """The synthetic logs below are only as good as this writer: re-framing
    a real log's payloads must give back wandb's exact bytes."""
    for path in LOGS:
        data = path.read_bytes()
        payloads, truncated = wandb_log.payloads(data)
        assert not truncated and len(payloads) >= 10
        assert frame(payloads) == data


def test_a_record_split_across_blocks_is_reassembled():
    big = {"blob": "x" * 70_000}
    log = wandb_log.read_log_bytes(frame(a_run(summary_record(big))))
    assert log.summary["blob"] == big["blob"] and log.exit_code == 0


# ---- the committed W&B fixture, field by field -------------------------------------


def _wandb_imports():
    result = import_project(WANDB_EXAMPLE)
    assert result.refused == [] and result.lossy == []
    assert {i.adapter for i in result.imported} == {"wandb"}
    return result.imported


def test_the_committed_wandb_runs_map_every_field():
    imported = _wandb_imports()
    assert len(imported) == 4
    by_lr = {}
    for imp in imported:
        rec, run_dir = imp.record, Path(imp.source_path)
        log = wandb_log.read_log(next(run_dir.glob("*.wandb")))
        assert imp.name == f"generate/{run_dir.name.rsplit('-', 1)[-1]}"
        assert rec.run_id == log.run_id == run_dir.name.rsplit("-", 1)[-1]
        assert rec.family == "flows"
        assert rec.arm is None and rec.seeds is None and rec.dependencies is None
        assert rec.status == "complete"
        assert rec.started_at == schema.format_timestamp(log.started)
        assert rec.finished_at is None
        assert rec.producer.name == "attestation wandb-import"
        assert rec.config == {"lr": rec.config["lr"], "seed": 0, "dataset": "sklearn:breast_cancer"}
        by_lr[rec.config["lr"]] = rec
        assert rec.code.provenance == "inferred" and rec.code.repo is None
        assert rec.code.git_sha and len(rec.code.git_sha) == 40 and rec.code.dirty is None
        assert rec.data == []
        names = [m.name for m in rec.metrics]
        assert names == ["accuracy", "auc", "train_loss", "wall_time_s"]
        assert all(m.step is None and m.direction is None for m in rec.metrics)
        wall = rec.metrics[-1]
        assert wall.unit == "s" and wall.value == log.summary["_runtime"]
        assert rec.history.series == 1 and rec.history.points == 10
        ext = rec.ext[IMPORT_NAMESPACE]
        assert ext["adapter"] == "wandb" and ext["status"] == "recorded"
        assert ext["caveats"] == [trackers.WANDB_CAVEAT] and ext["run_caveats"] == []
    assert set(by_lr) == {0.001, 0.01, 0.1, 1.0}


def test_the_committed_wandb_curve_is_the_logged_train_loss(tmp_path):
    for found in trackers.WANDB.scan(WANDB_EXAMPLE):
        (series,) = found.series
        assert series.name == "train_loss" and series.split is None
        assert [p.step for p in series.points] == list(range(10))
        assert series.points[-1].value == found.summary["train_loss"]
        assert all(p.at and p.at.startswith("2026-") for p in series.points)
        record = trackers.WANDB.to_core_record(found)
        out = tmp_path / found.run_id
        assert record.history == schema.write_history(out, [series])


def test_the_log_agrees_with_the_materialised_files(tmp_path):
    """generate.py materialised files/ from the same log: where both say
    something, they say the same thing (the log is still the one read)."""
    for found in trackers.WANDB.scan(WANDB_EXAMPLE):
        files = Path(found.source_path) / "files"
        materialised = json.loads((files / "wandb-summary.json").read_text())
        assert wandb_store.numeric_leaves(materialised) == wandb_store.numeric_leaves(found.summary)
        assert wandb_store.files_config(files / "config.yaml") == found.raw_config


def test_the_committed_logs_hold_nothing_the_scrub_removed():
    for path in LOGS:
        data = path.read_bytes()
        assert b"/home/" not in data and b"github.com" not in data
        log = wandb_log.read_log(path)
        assert log.git_remote is None and log.project == "flows"


# ---- against wandb itself (skipped unless importable) ------------------------------


def test_the_decoder_agrees_with_wandbs_own_reader():
    pytest.importorskip("wandb")
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    for path in LOGS:
        ds = datastore.DataStore()
        ds.open_for_scan(str(path))
        theirs = []
        while (data := ds.scan_data()) is not None:
            theirs.append(data)
        assert wandb_log.payloads(path.read_bytes())[0] == theirs
        ours = wandb_log.read_log(path)
        runs = [r for r in map(pb.Record.FromString, theirs) if r.HasField("run")]
        assert ours.run_id == runs[0].run.run_id and ours.project == runs[0].run.project
        assert ours.git_commit == runs[0].run.git.commit
        assert ours.exit_code == next(
            r.exit.exit_code for r in map(pb.Record.FromString, theirs) if r.HasField("exit")
        )


def test_the_field_numbers_are_wandbs():
    pytest.importorskip("wandb")
    from wandb.proto import wandb_internal_pb2 as pb

    messages = {
        "Record": pb.Record,
        "RunRecord": pb.RunRecord,
        "HistoryRecord": pb.HistoryRecord,
        "HistoryStep": pb.HistoryStep,
        "Item": pb.SummaryItem,
        "UpdateRecord": pb.SummaryRecord,
        "RunExitRecord": pb.RunExitRecord,
        "GitRepoRecord": pb.GitRepoRecord,
    }
    for name, message in messages.items():
        numbers = {f.name: f.number for f in message.DESCRIPTOR.fields}
        for field_name, number in wandb_log.FIELDS[name].items():
            assert numbers[field_name] == number, (name, field_name)
    for item in (pb.HistoryItem, pb.ConfigItem):
        numbers = {f.name: f.number for f in item.DESCRIPTOR.fields}
        assert {k: numbers[k] for k in wandb_log.FIELDS["Item"]} == wandb_log.FIELDS["Item"]
    numbers = {f.name: f.number for f in pb.ConfigRecord.DESCRIPTOR.fields}
    assert numbers["update"] == 1 and numbers["remove"] == 2


# ---- synthetic W&B runs ------------------------------------------------------------


def test_a_nonzero_exit_is_failed_and_caveated_by_the_ledger(tmp_path):
    wandb_dir(tmp_path, a_run(exit_code=1))
    assert record_of(tmp_path).status == "failed"
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path.parent, project=tmp_path.name)
    (caveats,) = conn.execute("SELECT caveats_json FROM runs").fetchone()
    assert "has status failed" in caveats


def test_a_negative_exit_code_is_failed(tmp_path):
    wandb_dir(tmp_path, a_run(exit_code=-9))
    assert record_of(tmp_path).status == "failed"


def test_no_exit_record_is_unknown_not_skipped(tmp_path):
    """In flight or died: W&B cannot say which, so the run is imported with
    status null and says so -- skipping it would hide every crashed run."""
    wandb_dir(tmp_path, a_run(exit_code=None))
    record = record_of(tmp_path)
    assert record.status is None
    assert any("no exit record" in c for c in run_caveats(record))


def test_a_log_ending_mid_record_is_read_up_to_it(tmp_path):
    data = frame(a_run())
    wandb_dir(tmp_path, None, raw=data[:-3])
    record = record_of(tmp_path)
    assert record.status is None  # the exit record was the one cut off
    assert any("partial record" in c for c in run_caveats(record))
    assert {m.name for m in record.metrics} >= {"loss", "acc"}


@pytest.mark.parametrize(
    "damage",
    ["checksum", "header", "payload"],
)
def test_a_damaged_log_is_quarantined_not_half_read(tmp_path, damage):
    data = bytearray(frame(a_run()))
    if damage == "checksum":
        data[20] ^= 0xFF
    elif damage == "header":
        data[0:4] = b"XXXX"
    else:
        data = bytearray(frame([b"\x0b\x0c"]))  # wire type 3: a group
    wandb_dir(tmp_path, None, raw=bytes(data))
    result = import_project(tmp_path)
    assert result.imported == []
    (q,) = result.quarantined
    assert q.adapter == "wandb" and ".wandb" in q.reasons[0]


def test_a_non_finite_summary_value_is_kept_and_not_stored(tmp_path):
    wandb_dir(tmp_path, a_run(summary={"loss": float("nan"), "acc": 0.9}))
    (loss,) = [m for m in record_of(tmp_path).metrics if m.name == "loss"]
    assert math.isnan(loss.value)
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path.parent, project=tmp_path.name)
    stored = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert "loss" not in stored and "acc" in stored
    (caveats,) = conn.execute("SELECT caveats_json FROM runs").fetchone()
    assert "loss is NaN" in caveats


def test_summary_nesting_removes_media_and_case(tmp_path):
    extra = [
        summary_record({"Val": {"acc": 0.7, "_hidden": 1}, "img": {"_type": "image-file"}}),
        summary_record({"gone": 1.0}),
        summary_record({}, removes=("gone",)),
        summary_record({"text": "hello", "flag": True}),
    ]
    wandb_dir(tmp_path, a_run(*extra))
    names = [m.name for m in record_of(tmp_path).metrics]
    assert names == ["Val.acc", "acc", "loss", "wall_time_s"]


def test_config_updates_removes_and_bookkeeping(tmp_path):
    config = _field(5, _updates({"epochs": 3, "lr": 0.2, "_private": 1}, removes=("epochs",)))
    wandb_dir(tmp_path, a_run(config))
    assert record_of(tmp_path).config == {"lr": 0.2}


def test_history_series_from_rows(tmp_path):
    wandb_dir(tmp_path, a_run(history_record(2, {"loss": 0.25, "img": {"_type": "image"}})))
    (found,) = trackers.WANDB.scan(tmp_path)
    (loss,) = found.series
    assert [(p.step, p.value) for p in loss.points] == [(0, 1.0), (1, 0.5), (2, 0.25)]
    assert loss.points[0].at == "2026-11-18T11:06:41.500000Z"  # _timestamp 1795000001.5
    assert record_of(tmp_path).history.points == 3


def test_git_from_the_run_record_with_credentials_stripped(tmp_path):
    payloads = a_run()
    payloads[0] = run_record(commit="a" * 40, remote="https://user:tok@example.com/r.git")
    wandb_dir(tmp_path, payloads)
    code = record_of(tmp_path).code
    assert code.git_sha == "a" * 40 and code.provenance == "inferred"
    assert code.repo == "https://example.com/r.git"


def test_user_files_are_outputs_by_sha256_and_wandbs_own_are_not(tmp_path):
    run = wandb_dir(tmp_path, a_run())
    files = run / "files"
    (files / "model.pt").write_bytes(b"weights")
    (files / "media").mkdir()
    (files / "media" / "plot.png").write_bytes(b"png")
    (files / "code").mkdir()
    (files / "code" / "train.py").write_text("print()")
    for own in ("requirements.txt", "output.log", "diff.patch"):
        (files / own).write_text("x")
    data = record_of(tmp_path).data
    assert [(d.name, d.role) for d in data] == [
        ("files/media/plot.png", "output"),
        ("files/model.pt", "output"),
    ]
    assert data[1].sha256 == schema.sha256_file(files / "model.pt")


def test_a_directory_without_a_log_falls_back_to_files(tmp_path):
    """A synced directory, or the pre-2026-11-20 fixture's shape."""
    wandb_dir(
        tmp_path,
        None,
        summary={"auc": 0.9, "_runtime": 3.0},
        config_yaml="wandb_version: 1\nlr:\n  desc: null\n  value: 0.01\n_wandb:\n  value: {}\n",
    )
    record = record_of(tmp_path)
    assert record.family == "train" and record.status is None and record.history is None
    assert record.config == {"lr": 0.01}
    assert record.started_at == "2026-11-20T00:00:00.000001Z"
    assert [m.name for m in record.metrics] == ["auc", "wall_time_s"]
    assert any("no .wandb log" in c for c in run_caveats(record))


def test_two_logs_in_one_run_directory_are_refused(tmp_path):
    run = wandb_dir(tmp_path, a_run())
    (run / "run-other.wandb").write_bytes(frame(a_run()))
    assert [q.adapter for q in import_project(tmp_path).quarantined] == ["wandb"]


def test_latest_run_symlink_is_not_read_twice(tmp_path):
    run = wandb_dir(tmp_path, a_run())
    (tmp_path / "wandb" / "latest-run").symlink_to(run, target_is_directory=True)
    assert len(import_project(tmp_path).imported) == 1


def test_a_log_with_nothing_measured_is_not_a_run(tmp_path):
    wandb_dir(tmp_path, [run_record(), exit_record(0)])
    assert import_project(tmp_path).imported == []


# ---- the committed Hydra fixture, field by field -----------------------------------


def test_the_committed_hydra_arms_map_every_field():
    result = import_project(HYDRA_EXAMPLE)
    assert result.refused == [] and result.lossy == []
    arms = [i for i in result.imported if i.adapter == "hydra"]
    assert [i.name for i in arms] == ["train/0", "train/1", "train/2", "train/3"]
    lrs = {"0": "0.01", "1": "0.1", "2": "1", "3": "10"}
    for imp in arms:
        rec, n = imp.record, imp.name.split("/")[1]
        arm_dir = Path(imp.source_path)
        assert rec.run_id == f"{SWEEP}/{n}"
        assert rec.family == "train"
        assert rec.arm == f"lr={lrs[n]}" and rec.seeds is None
        assert rec.config == {"lr": float(lrs[n]) if "." in lrs[n] else int(lrs[n])}
        assert rec.status is None and rec.started_at is None and rec.finished_at is None
        assert rec.code.provenance == "unknown" and rec.history is None
        assert rec.dependencies is None
        expected = json.loads((arm_dir / "metrics.json").read_text())
        assert {m.name: m.value for m in rec.metrics} == expected
        assert all(m.unit is None for m in rec.metrics)  # no timing recorded
        assert [(d.name, d.role) for d in rec.data] == [("metrics.json", "output")]
        assert rec.data[0].sha256 == schema.sha256_file(arm_dir / "metrics.json")
        ext = rec.ext[IMPORT_NAMESPACE]
        assert ext["adapter"] == "hydra" and ext["caveats"] == [trackers.HYDRA_CAVEAT]
        assert ext["run_caveats"] == []


# ---- synthetic Hydra sweeps --------------------------------------------------------


def hydra_arm(
    root: Path,
    n: int,
    *,
    overrides: list[str] | None,
    config: str = "lr: 0.1\n",
    metrics: dict | None = None,
    job: str = "train",
    sweep: str = "2026-01-01/10-00-00",
    task_in_hydra_yaml: list[str] | None = None,
) -> Path:
    arm = root / "multirun" / sweep / str(n)
    hydra = arm / ".hydra"
    hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(config)
    task = "".join(f"    - {o}\n" for o in task_in_hydra_yaml or [])
    (hydra / "hydra.yaml").write_text(
        f"hydra:\n  overrides:\n    task:\n{task}  job:\n    name: {job}\n"
    )
    if overrides is not None:
        (hydra / "overrides.yaml").write_text("".join(f"- {o}\n" for o in overrides))
    (arm / "metrics.json").write_text(json.dumps(metrics or {"auc": 0.9}))
    return arm


@pytest.mark.parametrize(
    ("overrides", "arm", "seeds"),
    [
        (["model=b", "lr=0.1"], "lr=0.1,model=b", None),
        (["lr=0.1", "seed=3"], "lr=0.1", {"seed": 3}),
        (["+seed=4"], "train", {"seed": 4}),
        (["seed=abc"], "seed=abc", None),
        ([], None, None),
    ],
)
def test_overrides_declare_the_arm_and_seeds(overrides, arm, seeds):
    assert hydra_store.arm_and_seeds(overrides, "train") == (arm, seeds)


def test_seed_replicates_of_one_configuration_pool(tmp_path):
    """Hydra's `<job>/<n>` names carry no seed, so the name convention could
    never pool a Hydra seed sweep; the overrides can."""
    for n, (seed, auc) in enumerate([(1, 0.90), (2, 0.92), (3, 0.94)]):
        hydra_arm(
            tmp_path,
            n,
            overrides=["lr=0.1", f"seed={seed}"],
            config=f"lr: 0.1\nseed: {seed}\n",
            metrics={"auc": auc},
        )
    records = [i.record for i in import_project(tmp_path).imported]
    assert {r.arm for r in records} == {"lr=0.1"}
    assert sorted(r.seeds["seed"] for r in records) == [1, 2, 3]
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path.parent, project=tmp_path.name)
    out = ledger.compare(conn, "train", metric="auc")
    (pooled,) = out["arms"]
    assert pooled["n_runs"] == 3 and pooled["value"] == pytest.approx(0.92)


def test_missing_overrides_yaml_falls_back_to_hydra_yaml(tmp_path):
    hydra_arm(tmp_path, 0, overrides=None, task_in_hydra_yaml=["lr=0.5"])
    assert record_of(tmp_path).arm == "lr=0.5"


def test_the_whole_config_is_read_and_non_finite_values_are_strings(tmp_path):
    hydra_arm(tmp_path, 0, overrides=[], config="model:\n  depth: 4\n  dropout: .nan\nlr: 0.1\n")
    record = record_of(tmp_path)
    assert record.config == {"model": {"depth": 4, "dropout": "nan"}, "lr": 0.1}
    assert any("config.model.dropout" in c for c in run_caveats(record))


def test_an_unreadable_config_is_a_run_caveat(tmp_path):
    hydra_arm(tmp_path, 0, overrides=[], config="lr: [unclosed\n")
    record = record_of(tmp_path)
    assert record.config == {}
    assert any("not readable YAML" in c for c in run_caveats(record))


def test_outputs_exclude_hydras_own_directory(tmp_path):
    arm = hydra_arm(tmp_path, 0, overrides=["lr=0.1"])
    (arm / "ckpt").mkdir()
    (arm / "ckpt" / "model.pt").write_bytes(b"w")
    (arm / "train.log").write_text("log")
    names = [d.name for d in record_of(tmp_path).data]
    assert names == ["ckpt/model.pt", "metrics.json", "train.log"]


def test_the_readers_name_no_tracker_outside_the_adapters_package():
    """The modules live where tracker names may (tests/test_tracker_names.py)."""
    for module in (wandb_log, wandb_store, hydra_store):
        assert Path(module.__file__).parent.name == "ledger_adapters"
