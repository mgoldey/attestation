"""Builds the conformance golden fixtures under tests/data/conformance/.

Every fixture is made with the package's own writers, then (for a broken one)
damaged in exactly one way, so each breaks exactly one rule. The committed
tree is this function's output; `test_conformance.py` rebuilds it and asserts
it is byte-identical, so the fixtures cannot drift from the writers.

    python packages/attestation-schema/tests/conformance_fixtures.py   # rewrite

Spec: docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

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


def build(dest: Path) -> None:
    """Write every fixture under `dest` (`valid/<name>/`, `broken/<rule>/`)."""
    for name, make in (
        ("minimal", _minimal),
        ("job-complete", _job_complete),
        ("in-flight", _in_flight),
    ):
        make(dest / "valid" / name)
    for rule in BROKEN_RULES:
        _broken(rule, dest / "broken" / rule)


if __name__ == "__main__":
    if DATA.exists():
        shutil.rmtree(DATA)
    build(DATA)
    print(f"wrote {DATA}", file=sys.stderr)
