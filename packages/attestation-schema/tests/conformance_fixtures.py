"""Builds the conformance golden fixtures under tests/data/conformance/.

Every fixture is made with the package's own writers, then (for a broken one)
damaged in exactly one way, so each breaks exactly one rule. The committed
tree is this function's output; `test_conformance.py` rebuilds it and asserts
it is byte-identical, so the fixtures cannot drift from the writers.

    python packages/attestation-schema/tests/conformance_fixtures.py   # rewrite

Spec: docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import attestation_schema as s
from attestation_schema import layout
from attestation_schema.canonical import canonicalize
from attestation_schema.jobs import JobSpec, JobState

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "conformance"

T0 = "2026-11-19T10:00:00.000000Z"
T1 = "2026-11-19T10:00:05.000000Z"
T2 = "2026-11-19T10:42:13.250000Z"
PRODUCER = s.Producer("conformance-fixture", "1.0")
CODE = s.CodeIdentity("0" * 40, False, "https://example.invalid/train.git", "1.0", "verified", None)

# Every rule but `spec` has a broken fixture of its own; `spec` breaks a valid
# record against a JobSpec other than the one it ran with (see SPEC_CONFLICT).
BROKEN_RULES = (
    "read",
    "canonical",
    "manifest",
    "history",
    "state",
    "terminal",
    "source",
    "credentials",
    "ext-namespace",
)

# What each broken fixture's report must say: (rule, file under the run dir,
# JSON path, a fragment of the message).
EXPECTED = {
    "read": ("attestation/run.json", "$.status", "expected one of"),
    "canonical": ("attestation/run.json", "$", "not the canonical form"),
    "manifest": ("attestation/digests.json", '$.files["steps/00-train/model.txt"]', "sha256 is"),
    "history": ("attestation/history.jsonl", "$", "the file changed"),
    "state": ("attestation/state.json", "$.history[1].state", "not an allowed transition"),
    "terminal": ("attestation/state.json", "$.state", "every terminal job has a record"),
    "source": ("attestation/run.json", "$.dependencies[0].source", "is not index"),
    "credentials": ("attestation/run.json", "$.config.tracking_uri", "URL with credentials"),
    "ext-namespace": ("attestation/run.json", "$.ext.MyTool", "namespace 'MyTool'"),
}


def spec_for(run_id: str = "job-complete", **override: object) -> JobSpec:
    """The JobSpec `valid/job-complete` ran with (or one differing by `override`)."""
    fields: dict = {
        "run_id": run_id,
        "family": "lr-sweep",
        "arm": "lr=1e-3",
        "config": {"lr": 0.001, "epochs": 3},
        "run_dir": ".",
        "ext": {"submitter": {"queue": "local"}},
    }
    fields.update(override)
    return JobSpec(**fields)  # type: ignore[arg-type]


SPEC_CONFLICT = spec_for(config={"lr": 0.01, "epochs": 3})


def _record(run_id: str, **override: object) -> s.RunRecord:
    fields: dict = {
        "run_id": run_id,
        "family": "lr-sweep",
        "arm": "lr=1e-3",
        "status": "complete",
        "started_at": T1,
        "finished_at": T2,
        "producer": PRODUCER,
        "code": CODE,
        "config": {"lr": 0.001, "epochs": 3},
        "data": [],
        "metrics": [s.Metric("loss", 0.25, "min", 2, "eval", None, {})],
        "history": None,
        "seeds": {"python": 0},
        "dependencies": [],
        "ext": {},
    }
    fields.update(override)
    return s.RunRecord(**fields)


def _minimal(rd: Path) -> None:
    s.write(_record("minimal", dependencies=None, seeds=None, config={"note": "bare"}), rd)


def _job_complete(rd: Path) -> None:
    spec = spec_for(run_dir=str(rd))
    s.init_state(rd, run_id=spec.run_id, backend="local", job_id="j-1", at=T0)
    s.transition(rd, JobState.RUNNING, at=T1)
    out = layout.step_dir(rd, 0, "train")
    out.mkdir(parents=True)
    (out / "model.txt").write_text("weights v1\n")
    log = layout.step_log(rd, 0, "train")
    log.parent.mkdir(parents=True)
    log.write_text("epoch 0 loss 0.9\nepoch 2 loss 0.25\n")
    curve = [
        s.Series("loss", "train", {}, [s.Point(0, 0.9, T1), s.Point(2, 0.3, None)]),
        s.Series("loss", "eval", {}, [s.Point(2, 0.25, None)]),
    ]
    history = s.write_history(rd, curve)
    manifest = layout.write_manifest(rd)
    deps = [
        s.Dependency("numpy", "2.1.0", None, None, "inferred", "importlib.metadata", "index:pypi"),
        s.Dependency("ferric", "0.1.0", "1" * 40, False, "verified", "ferric.__build__", "file"),
        s.Dependency("lib", None, None, None, "unknown", None, "git+https://example.invalid/l"),
        s.Dependency("python", "3.12.0", None, None, "verified", "sys.version", "system"),
    ]
    record = _record(
        spec.run_id,
        config=spec.config,
        data=[manifest],
        history=history,
        dependencies=deps,
        ext={**spec.ext, "train.py": {"epochs_run": 3}},
    )
    s.write(record, rd)
    s.settle(rd, spec, JobState.COMPLETE, producer=PRODUCER, ended_at=T2)


def _in_flight(rd: Path) -> None:
    s.init_state(rd, run_id="in-flight", backend="local", job_id="j-2", at=T0)
    s.transition(rd, JobState.RUNNING, at=T1)


def _rewrite_json(path: Path, change) -> None:
    obj = json.loads(path.read_bytes())
    change(obj)
    path.write_bytes(canonicalize(obj))


def _broken(rule: str, rd: Path) -> None:
    run_json = rd / layout.RUN_JSON
    if rule == "read":
        _minimal(rd)
        _rewrite_json(run_json, lambda o: o.update(status="running"))
    elif rule == "canonical":
        _minimal(rd)
        run_json.write_text(json.dumps(json.loads(run_json.read_bytes()), indent=2) + "\n")
    elif rule in ("manifest", "history", "state"):
        _job_complete(rd)
        if rule == "manifest":
            (layout.step_dir(rd, 0, "train") / "model.txt").write_text("weights v2\n")
        elif rule == "history":
            with (rd / layout.HISTORY_JSONL).open("ab") as fh:
                fh.write(b'{"at":null,"series":1,"step":3,"value":0.2}\n')
        else:
            _rewrite_json(
                rd / layout.STATE_JSON,
                lambda o: o["history"].__setitem__(1, {"state": "complete", "at": T1}),
            )
    elif rule == "terminal":
        _in_flight(rd)
        _rewrite_json(
            rd / layout.STATE_JSON,
            lambda o: (
                o.update(state="failed", updated_at=T2, message="killed"),
                o["history"].append({"state": "failed", "at": T2}),
            ),
        )
    elif rule == "source":
        _minimal(rd)
        _rewrite_json(
            run_json,
            lambda o: o.update(
                dependencies=[
                    {
                        "name": "numpy",
                        "version": "2.1.0",
                        "build_sha": None,
                        "dirty": None,
                        "provenance": "inferred",
                        "evidence": None,
                        "source": "pip",
                    }
                ]
            ),
        )
    elif rule == "credentials":
        tracking = "https://ci:hunter2@example.invalid/mlflow"
        s.write(_record("credentials", config={"tracking_uri": tracking}), rd)
    elif rule == "ext-namespace":
        _minimal(rd)
        _rewrite_json(run_json, lambda o: o.update(ext={"MyTool": {"x": 1}}))
    else:  # pragma: no cover
        raise ValueError(rule)


# ---- datasets (attestation.dataset/1, draft) -------------------------------------
#
# datasets/valid/minimal, datasets/broken/<rule> for every rule a dataset
# directory can break, and lineage/: the whole chain ferric build -> smeltery
# runs -> dataset -> training run. Spec:
# docs/superpowers/specs/2026-11-19-dataset-records-design.md.

DATASET_BROKEN_RULES = ("read", "canonical", "manifest", "credentials", "ext-namespace")

DATASET_EXPECTED = {
    "read": ("attestation/dataset.json", "$.license", "not an SPDX license expression"),
    "canonical": ("attestation/dataset.json", "$", "not the canonical form"),
    "manifest": ("attestation/digests.json", '$.files["data.csv"]', "sha256 is"),
    "credentials": ("attestation/dataset.json", "$.ext.hub.endpoint", "URL with credentials"),
    "ext-namespace": ("attestation/dataset.json", "$.ext.MyTool", "namespace 'MyTool'"),
}

EXPORT_CODE = s.CodeIdentity("2" * 40, False, None, "0.4.0", "inferred", "importlib.metadata")


def _dataset(dd: Path, files: dict[str, str], **override: object) -> s.DatasetRecord:
    """Write `files` into `dd`, then the manifest, then the record."""
    for name, text in files.items():
        (dd / name).parent.mkdir(parents=True, exist_ok=True)
        (dd / name).write_bytes(text.encode())
    content = s.write_dataset_manifest(dd)
    fields: dict = {
        "dataset_id": "toy",
        "name": None,
        "version": "1",
        "created_at": T2,
        "producer": s.Producer("conformance-fixture", "1.0"),
        "code": CODE,
        "content_sha256": content,
        "files": s.dataset_files(dd),
        "rows": 1,
        "columns": [s.Column("a", "int64", None), s.Column("b", "int64", None)],
        "splits": [],
        "source_runs": [],
        "license": None,
        "ext": {},
    }
    fields.update(override)
    record = s.DatasetRecord(**fields)
    s.write_dataset(record, dd)
    return record


def _dataset_minimal(dd: Path) -> None:
    _dataset(dd, {"data.csv": "a,b\n1,2\n"})


def _dataset_broken(rule: str, dd: Path) -> None:
    _dataset_minimal(dd)
    record = dd / layout.DATASET_JSON
    if rule == "read":
        _rewrite_json(record, lambda o: o.update(license="NOASSERTION"))
    elif rule == "canonical":
        record.write_text(json.dumps(json.loads(record.read_bytes()), indent=2) + "\n")
    elif rule == "manifest":
        (dd / "data.csv").write_text("a,b\n1,3\n")  # same size, other bytes
    elif rule == "credentials":
        endpoint = "https://ci:hunter2@example.invalid/hub"
        _rewrite_json(record, lambda o: o.update(ext={"hub": {"endpoint": endpoint}}))
    elif rule == "ext-namespace":
        _rewrite_json(record, lambda o: o.update(ext={"MyTool": {"x": 1}}))
    else:  # pragma: no cover
        raise ValueError(rule)


# The lineage chain, shaped like a real smeltery export (dataset_export.py):
# two smeltery runs, each a core record beside its legacy run-<digest12>.json,
# built on one verified ferric build; a dataset whose table carries the
# export's provenance columns; a training run naming the dataset by digest.
FERRIC_SHA = "f3" * 20
CAMPAIGN = "benzoic-rhf"
INPUT_DIGESTS = {
    "smeltery-a": "07693d07e8b9" + "1" * 52,
    "smeltery-b": "9c1e44a0b2d7" + "2" * 52,
}
LIGANDS = {"smeltery-a": ("parent", "4-F"), "smeltery-b": ("parent", "3-Cl")}
SCORES = {"parent": -12.5, "4-F": -13.25, "3-Cl": -11.75}
EXPORT_COLUMNS = (
    "run_id",
    "input_digest",
    "record_sha256",
    "campaign",
    "ligand",
    "target",
    "tier",
    "score",
    "score_unit",
    "ferric_version",
    "ferric_git_sha",
    "ferric_dirty",
    "ferric_provenance_level",
)
DATASET_ID = "smeltery-benzoic-ddE"


def _smeltery_legacy(name: str) -> dict:
    """smeltery's own run-<digest12>.json, with its self-hash: sha256 of
    json.dumps(sort_keys=True) without the key -- NOT the core record digest."""
    digest = INPUT_DIGESTS[name]
    legacy = {
        "campaign": CAMPAIGN,
        "input_digest": digest,
        "schema_version": 1,
        "ferric": {
            "commit": FERRIC_SHA,
            "dirty": False,
            "provenance": "VERIFIED",
            "version": "0.1.0",
        },
        "results": {"per_pose": {lig: {"dE_int": [SCORES[lig]]} for lig in LIGANDS[name]}},
    }
    legacy["record_sha256"] = hashlib.sha256(
        json.dumps(legacy, sort_keys=True).encode()
    ).hexdigest()
    return legacy


def _smeltery_run(rd: Path, name: str) -> s.RunRecord:
    legacy = _smeltery_legacy(name)
    run_id = INPUT_DIGESTS[name][:12]
    legacy_name = f"run-{run_id}.json"
    rd.mkdir(parents=True)
    (rd / legacy_name).write_text(json.dumps(legacy, sort_keys=True, indent=1) + "\n")
    stage = layout.step_dir(rd, 0, "dock") / "stage.json"
    stage.parent.mkdir(parents=True)
    stage.write_text('{"stage": "dock"}\n')
    manifest = layout.write_manifest(rd)
    legacy_sha = hashlib.sha256((rd / legacy_name).read_bytes()).hexdigest()
    analogues = [lig for lig in LIGANDS[name] if lig != "parent"]
    record = s.RunRecord(
        run_id=run_id,
        family=CAMPAIGN,
        arm=None,
        status="complete",
        started_at=T0,
        finished_at=T1,
        producer=s.Producer("smeltery", "0.4.0"),
        code=s.CodeIdentity("2" * 40, False, None, "0.4.0", "inferred", "editable checkout HEAD"),
        config={"campaign": CAMPAIGN, "keep": 1},
        data=[s.DataDigest(legacy_name, legacy_sha, "output"), manifest],
        metrics=[
            s.Metric(
                "paired_ddE.mean",
                SCORES[a] - SCORES["parent"],
                "min",
                None,
                None,
                "kcal/mol",
                {"analogue": a},
            )
            for a in analogues
        ],
        history=None,
        seeds={"docking": [7]},
        dependencies=[
            s.Dependency(
                "ferric", "0.1.0", FERRIC_SHA, False, "verified", "ferric.__build__", "file"
            ),
            s.Dependency("python", "3.11.15", None, None, "verified", "sys.version", "system"),
        ],
        ext={"smeltery": {"input_digest": INPUT_DIGESTS[name]}},
    )
    s.write(record, rd)
    return record


def _export_table(runs: dict[str, s.RunRecord]) -> str:
    """runs.csv: the export's rows (a stand-in for runs.parquet; the schema
    hashes bytes and never parses the table). run_id is input_digest[:12] and
    record_sha256 the LEGACY self-hash, exactly as dataset_export.py copies them."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXPORT_COLUMNS)
    for name in runs:
        legacy = _smeltery_legacy(name)
        for lig in LIGANDS[name]:
            w.writerow(
                [
                    legacy["input_digest"][:12],
                    legacy["input_digest"],
                    legacy["record_sha256"],
                    CAMPAIGN,
                    lig,
                    "receptor.pdb",
                    "xtb",
                    SCORES[lig],
                    "kcal/mol",
                    "0.1.0",
                    FERRIC_SHA,
                    "false",
                    "VERIFIED",
                ]
            )
    return buf.getvalue()


def _split(table: str) -> str:
    """smeltery's split file (dataset_splits.py format), naming the table by
    the sha256 of its bytes."""
    rows = list(csv.DictReader(io.StringIO(table)))
    ident = [[r["run_id"], r["ligand"], r["target"], r["tier"]] for r in rows]
    doc = {
        "format": "smeltery.split",
        "schema_version": 1,
        "kind": "scaffold",
        "algorithm_version": 1,
        "seed": 0,
        "parameters": {"fractions": {"train": 0.75, "test": 0.25}, "generic": False},
        "table": {
            "sha256": hashlib.sha256(table.encode()).hexdigest(),
            "rows": len(rows),
            "identity_columns": ["run_id", "ligand", "target", "tier"],
        },
        "partitions": {
            "train": sorted(i for i in ident if i[1] != "3-Cl"),
            "test": sorted(i for i in ident if i[1] == "3-Cl"),
        },
    }
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def _lineage(dest: Path) -> None:
    runs = {name: _smeltery_run(dest / name, name) for name in INPUT_DIGESTS}
    table = _export_table(runs)
    dd = dest / "dataset"
    dd.mkdir(parents=True)
    split_text = _split(table)
    ds = _dataset(
        dd,
        {"runs.csv": table, "split-scaffold.json": split_text},
        dataset_id=DATASET_ID,
        name="benzoic RHF paired ddE",
        version="1",
        producer=s.Producer("smeltery export", "0.4.0"),
        code=EXPORT_CODE,
        rows=len(table.splitlines()) - 1,
        columns=[s.Column(c, None, "kcal/mol" if c == "score" else None) for c in EXPORT_COLUMNS],
        splits=[
            s.Split(
                "scaffold",
                "split-scaffold.json",
                hashlib.sha256(split_text.encode()).hexdigest(),
                "scaffold",
                0,
            )
        ],
        source_runs=[s.SourceRun(r.run_id, r.family, s.record_digest(r)) for r in runs.values()],
        license="CC-BY-4.0",
        ext={"smeltery": {"schema_version": 1, "grain": "one row per (ligand, target, tier)"}},
    )
    td = dest / "training"
    s.write(
        s.RunRecord(
            run_id="train-1",
            family="ddE-model",
            arm="gnn",
            status="complete",
            started_at=T1,
            finished_at=T2,
            producer=PRODUCER,
            code=CODE,
            config={"model": "gnn", "split": "scaffold"},
            data=s.dataset_lineage_entries(ds, role="train", split="scaffold"),
            metrics=[s.Metric("rmse", 0.42, "min", None, "test", "kcal/mol", {})],
            history=None,
            seeds={"torch": 0},
            dependencies=[],
            ext={},
        ),
        td,
    )


def build(dest: Path) -> None:
    """Write every fixture under `dest` (`valid/<name>/`, `broken/<rule>/`,
    `datasets/{valid,broken}/`, `lineage/`)."""
    for name, make in (
        ("minimal", _minimal),
        ("job-complete", _job_complete),
        ("in-flight", _in_flight),
    ):
        make(dest / "valid" / name)
    for rule in BROKEN_RULES:
        _broken(rule, dest / "broken" / rule)
    _dataset_minimal(dest / "datasets" / "valid" / "minimal")
    for rule in DATASET_BROKEN_RULES:
        _dataset_broken(rule, dest / "datasets" / "broken" / rule)
    _lineage(dest / "lineage")


if __name__ == "__main__":
    if DATA.exists():
        shutil.rmtree(DATA)
    build(DATA)
    print(f"wrote {DATA}", file=sys.stderr)
