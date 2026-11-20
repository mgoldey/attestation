"""The ledger reads only core records (roadmap 11).

Spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md.
A directory holding `attestation/run.json` is read by the `core` adapter with
no conversion; every legacy reader's output becomes a validated core record
before it reaches a table.
"""

import json
import shutil
from pathlib import Path

import attestation_schema as schema
import pytest
from attestation_schema import layout

from attestation import cli, ledger, mcp_server
from attestation.db import get_db
from attestation.ledger_adapters import (
    ADAPTERS,
    IMPORT_NAMESPACE,
    LEGACY_ADAPTERS,
    Imported,
    generic,
    import_project,
)
from attestation.ledger_adapters.core import MAX_DEPTH
from attestation.ledger_adapters.found import Found, FoundMetric
from attestation.ledger_adapters.trackers import found_to_core

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "examples" / "mlflow-records" / "records"


def _record(run_id="r1", **over) -> schema.RunRecord:
    base = {
        "schema": schema.SCHEMA,
        "run_id": run_id,
        "family": "sweep",
        "arm": None,
        "status": "complete",
        "started_at": "2026-11-19T10:00:00.000000Z",
        "finished_at": None,
        "producer": {"name": "test", "version": None},
        "code": {
            "git_sha": None,
            "dirty": None,
            "repo": None,
            "version": None,
            "provenance": "unknown",
            "evidence": None,
        },
        "config": {"lr": 0.1},
        "data": [],
        "metrics": [
            {
                "name": "wer",
                "value": 0.2,
                "direction": None,
                "step": None,
                "split": "test",
                "unit": None,
                "labels": {},
            }
        ],
        "history": None,
        "seeds": None,
        "dependencies": None,
        "ext": {},
    }
    base.update(over)
    return schema.from_dict(base)


def _metric(name, value, split=None, labels=None):
    return {
        "name": name,
        "value": value,
        "direction": None,
        "step": None,
        "split": split,
        "unit": None,
        "labels": labels or {},
    }


@pytest.fixture
def conn(tmp_path):
    c = get_db(tmp_path / "t.db")
    yield c
    c.close()


def _scan(conn, ws):
    return ledger.scan(conn, ws)


def test_the_committed_mlflow_records_are_runs(conn, tmp_path):
    """The MLflow exporter's records scan as runs with their own final
    values, family and start time -- no conversion, no MLflow reader."""
    ws = tmp_path / "ws"
    shutil.copytree(RECORDS, ws / "proj" / "records")
    out = _scan(conn, ws)
    assert out["scanned"] == {"proj": 4}

    for run_dir in sorted((ws / "proj" / "records").iterdir()):
        record = schema.read(run_dir)
        found = ledger.detail(conn, "proj", record.run_id)
        assert found["adapter"] == "core"
        assert found["family"] == record.family
        assert found["status"] == "complete"
        assert found["started"] == record.started_at
        assert found["config"] == record.config
        stored = {(m["metric"], m["step"]): m["value"] for m in found["metrics"]}
        assert stored == {(m.name, m.step): m.value for m in record.metrics}
        assert found["caveats"] == []


def test_a_run_that_did_not_complete_carries_a_caveat(conn, tmp_path):
    ws = tmp_path / "ws"
    schema.write(_record("ok"), ws / "p" / "runs" / "ok")
    schema.write(_record("bad", status="failed"), ws / "p" / "runs" / "bad")
    _scan(conn, ws)
    assert ledger.detail(conn, "p", "bad")["status"] == "failed"
    out = ledger.compare(conn, "sweep", metric="wer")
    assert any("bad has status failed" in c for c in out["caveats"]), out["caveats"]


def test_a_non_finite_value_is_not_stored_and_says_so(conn, tmp_path):
    ws = tmp_path / "ws"
    # "NaN" is the wire spelling of a diverged value (core spec, MLflow finding 3).
    rec = _record("a", metrics=[_metric("loss", "NaN"), _metric("wer", 0.3)])
    schema.write(rec, ws / "p" / "runs" / "a")
    _scan(conn, ws)
    assert [m["metric"] for m in ledger.detail(conn, "p", "a")["metrics"]] == ["wer"]
    out = ledger.compare(conn, "sweep", metric="wer")
    assert any("loss is NaN" in c for c in out["caveats"]), out["caveats"]


def test_labels_fold_into_split(conn, tmp_path):
    ws = tmp_path / "ws"
    metrics = [
        _metric("dde", 1.0, split="test", labels={"analogue": "a1"}),
        _metric("dde", 2.0, split="test", labels={"analogue": "a2"}),
        _metric("dde", 3.0, labels={"analogue": "a3"}),
    ]
    schema.write(_record("a", metrics=metrics), ws / "p" / "runs" / "a")
    _scan(conn, ws)
    splits = {m["split"] for m in ledger.detail(conn, "p", "a")["metrics"]}
    assert splits == {"test,analogue=a1", "test,analogue=a2", "analogue=a3"}
    assert ledger._split_rank("test,analogue=a1") == ledger._split_rank("test")


def test_re_executions_sharing_a_run_id_are_both_kept_and_both_suffixed(conn, tmp_path):
    """run_id is not unique; a name lookup must not silently pick one."""
    ws = tmp_path / "ws"
    one, two = _record("same"), _record("same", config={"lr": 0.2})
    schema.write(one, ws / "p" / "runs" / "x")
    schema.write(two, ws / "p" / "runs" / "y")
    _scan(conn, ws)
    names = {r["name"] for r in conn.execute("SELECT name FROM runs")}
    assert names == {
        f"same@{schema.record_digest(one)[:12]}",
        f"same@{schema.record_digest(two)[:12]}",
    }


def test_a_record_the_reader_refuses_is_a_diagnostic_not_a_crash(conn, tmp_path):
    ws = tmp_path / "ws"
    run = ws / "p" / "runs" / "broken"
    schema.write(_record("broken"), run)
    path = schema.record_path(run)
    path.write_text(path.read_text().replace('"complete"', '"running"'))
    schema.write(_record("fine"), ws / "p" / "runs" / "fine")
    out = _scan(conn, ws)
    assert out["scanned"] == {"p": 1}
    assert "1 record(s) refused" in out["diagnostics"]["p"]
    assert "status" in out["diagnostics"]["p"]


def test_a_run_in_flight_is_skipped(conn, tmp_path):
    ws = tmp_path / "ws"
    flight = ws / "p" / "runs" / "flying" / "attestation"
    flight.mkdir(parents=True)
    (flight / "state.json").write_text("{}")
    out = _scan(conn, ws)
    assert out["scanned"] == {}


def test_runs_deeper_than_the_bound_are_not_walked(tmp_path):
    deep = tmp_path / "p"
    for i in range(MAX_DEPTH + 1):
        deep = deep / f"d{i}"
    schema.write(_record("deep"), deep)
    assert import_project(tmp_path / "p").imported == []


def test_files_inside_a_run_directory_are_not_read_twice(conn, tmp_path):
    """A run directory under results/ is the core adapter's; its loose JSON
    outputs must not also become generic runs."""
    ws = tmp_path / "ws"
    run = ws / "p" / "results" / "run-a"
    run.mkdir(parents=True)
    (run / "scores.json").write_text(json.dumps({"wer": 0.9}))
    schema.write(_record("a"), run)
    (ws / "p" / "results" / "other.json").write_text(json.dumps({"wer": 0.5}))
    _scan(conn, ws)
    rows = {r["name"]: r["adapter"] for r in conn.execute("SELECT name, adapter FROM runs")}
    assert rows == {"a": "core", "other": "generic"}


def test_a_native_name_is_taken_before_the_legacy_readers_run(tmp_path):
    proj = tmp_path / "p"
    schema.write(_record("eval"), proj / "runs" / "eval")
    (proj / "results").mkdir()
    (proj / "results" / "eval.json").write_text(json.dumps({"wer": 0.3}))
    names = sorted(i.name for i in import_project(proj).imported)
    assert names == ["eval", "results/eval"]


def test_every_legacy_reader_emits_a_valid_core_record():
    """Each committed tracker fixture converts, and the record round-trips
    through the schema's own writer and reader byte for byte."""
    for example in (
        "wandb",
        "sacred",
        "dvc",
        "hydra",
        "workspace/speech-distill",
        "flows/training",
    ):
        result = import_project(ROOT / "examples" / example)
        assert result.imported, example
        assert result.refused == [], result.refused
        for imp in result.imported:
            ext = imp.record.ext[IMPORT_NAMESPACE]
            assert ext["adapter"] == imp.adapter
            assert schema.loads(schema.dumps(imp.record)) == imp.record


def test_an_importer_claims_completion_only_when_it_observed_it():
    """Core spec Producer finding 4.1, decided at the v1 freeze: a legacy
    artifact rarely shows how its run ended, and a spec shows no run at all,
    so the importer writes status null -- never a claimed `complete` -- and
    keeps the ledger's own word in ext. Sacred's reader keeps only runs Sacred
    marked COMPLETED, and W&B's reads the exit record the run's own log holds
    (roadmap 16): those two alone say `complete`."""
    statuses = {}
    for example in ("wandb", "sacred", "dvc", "hydra", "workspace/speech-distill"):
        for imp in import_project(ROOT / "examples" / example).imported:
            statuses.setdefault(imp.adapter, set()).add(imp.record.status)
    assert statuses.pop("sacred") == {"complete"}
    assert statuses.pop("wandb") == {"complete"}
    assert statuses and all(s == {None} for s in statuses.values()), statuses
    words = {
        imp.record.ext[IMPORT_NAMESPACE]["status"]
        for imp in import_project(ROOT / "examples" / "workspace/speech-distill").imported
    }
    assert "spec" in words or "recorded" in words


def test_a_native_record_without_a_status_is_unknown_and_caveated(conn, tmp_path):
    ws = tmp_path / "ws"
    schema.write(_record("ok"), ws / "p" / "runs" / "ok")
    schema.write(_record("seen", status=None), ws / "p" / "runs" / "seen")
    _scan(conn, ws)
    assert ledger.detail(conn, "p", "seen")["status"] == "unknown"
    out = ledger.compare(conn, "sweep", metric="wer")
    assert any("seen has no status" in c for c in out["caveats"]), out["caveats"]


def test_tracker_caveats_travel_in_the_record():
    result = import_project(ROOT / "examples" / "wandb")
    ext = result.imported[0].record.ext[IMPORT_NAMESPACE]
    assert ext["adapter"] == "wandb" and "final" in ext["caveats"][0]
    assert ext["run_caveats"] == []


def test_a_dvc_hash_mismatch_is_a_run_caveat(tmp_path):
    dst = tmp_path / "dvc"
    shutil.copytree(ROOT / "examples" / "dvc", dst)
    metric = next((dst / "metrics").glob("*.json"))
    metric.write_text('{"auc": 0.5}')
    caveats = [
        c
        for i in import_project(dst).imported
        for c in i.record.ext[IMPORT_NAMESPACE]["run_caveats"]
    ]
    assert len(caveats) == 1 and metric.name in caveats[0]


def test_a_tracker_start_time_is_the_core_spelling():
    imported = import_project(ROOT / "examples" / "sacred").imported
    assert all(i.record.started_at and i.record.started_at.endswith("Z") for i in imported)


def test_the_registry_order_is_the_legacy_discover_order():
    """`generic.discover` is built from the registry, so the pipeline and the
    function the tests and `attest record` read cannot drift apart."""
    assert ADAPTERS[0].name == "core"
    assert [a.name for a in ADAPTERS[1:]] == [a.name for a in LEGACY_ADAPTERS]
    assert [a.name for a in LEGACY_ADAPTERS] == [
        "generic",
        "wandb",
        "mlflow",
        "sacred",
        "dvc",
        "hydra",
    ]
    proj = ROOT / "examples" / "workspace" / "retrieval-ablation"
    assert [f.name for f in generic.discover(proj)] == [
        i.name for i in import_project(proj).imported
    ]


def test_a_nan_config_cell_becomes_a_string_not_a_refusal(tmp_path):
    proj = tmp_path / "p"
    (proj / "results").mkdir(parents=True)
    (proj / "results" / "sweep.csv").write_text("name,wer,note\na,0.1,nan\n")
    result = import_project(proj)
    assert result.refused == []
    assert result.imported[0].record.config["note"] == "nan"


# ---- conformance: quarantine, loudly; coerce nothing silently (roadmap 15) ----
# Spec: docs/superpowers/specs/2026-11-19-conformance-suite-design.md,
# Decisions 3 and 4. Each mutant is a record the bare reader ACCEPTS (the
# ledger imported it before): what makes these bite is that the core adapter
# now runs the whole conformance suite, not just `read`.

FIXTURES = ROOT / "packages" / "attestation-schema" / "tests" / "data" / "conformance"


def _qrecord(run_id: str, **over) -> schema.RunRecord:
    base = {
        "schema": schema.SCHEMA,
        "run_id": run_id,
        "family": "sweep",
        "arm": None,
        "status": "complete",
        "started_at": None,
        "finished_at": None,
        "producer": {"name": "test", "version": None},
        "code": {
            "git_sha": None,
            "dirty": None,
            "repo": None,
            "version": None,
            "provenance": "unknown",
            "evidence": None,
        },
        "config": {"lr": 0.1},
        "data": [],
        "metrics": [
            {
                "name": "wer",
                "value": 0.2,
                "direction": None,
                "step": None,
                "split": "test",
                "unit": None,
                "labels": {},
            }
        ],
        "history": None,
        "seeds": None,
        "dependencies": None,
        "ext": {},
    }
    base.update(over)
    return schema.from_dict(base)


def _non_canonical(rd: Path) -> None:
    p = rd / layout.RUN_JSON
    p.write_text(json.dumps(json.loads(p.read_text()), indent=2))


def _credential(rd: Path) -> None:
    (rd / layout.RUN_JSON).unlink()
    schema.write(_qrecord("bad", config={"uri": "https://ci:" + "hunter2@example.invalid/x"}), rd)


def _stray_sidecar(rd: Path) -> None:
    (rd / layout.HISTORY_JSONL).write_text('{"schema":"attestation.history/1"}\n')


def _terminal_disagrees(rd: Path) -> None:
    shutil.rmtree(rd)
    shutil.copytree(FIXTURES / "valid" / "job-complete", rd)
    p = rd / layout.RUN_JSON
    p.write_text(p.read_text().replace('"status":"complete"', '"status":"failed"'))


def _unbound_manifest(rd: Path) -> None:
    (rd / "out.txt").write_text("x")
    (rd / layout.DIGESTS_JSON).write_bytes(schema.canonical.canonicalize(layout.build_manifest(rd)))


MUTANTS = {
    "canonical": _non_canonical,
    "credentials": _credential,
    "history": _stray_sidecar,
    "terminal": _terminal_disagrees,
    "manifest": _unbound_manifest,
}


def _workspace(tmp_path: Path, mutant=None) -> Path:
    ws = tmp_path / "ws"
    schema.write(_qrecord("good"), ws / "p" / "runs" / "good")
    bad = ws / "p" / "runs" / "bad"
    schema.write(_qrecord("bad"), bad)
    if mutant is not None:
        mutant(bad)
    return ws


def test_the_unmutated_workspace_imports_both(conn, tmp_path):
    out = ledger.scan(conn, _workspace(tmp_path))
    assert out["scanned"] == {"p": 2} and out["quarantined"] == []


@pytest.mark.parametrize("rule", sorted(MUTANTS))
def test_a_record_the_reader_accepts_but_conformance_refuses_is_quarantined(conn, tmp_path, rule):
    ws = _workspace(tmp_path, MUTANTS[rule])
    bad = ws / "p" / "runs" / "bad"
    # The bare reader accepts it -- the old ledger imported it.
    if rule != "terminal":
        assert schema.read(bad).run_id == "bad"
    out = ledger.scan(conn, ws)
    assert out["scanned"] == {"p": 1}
    (q,) = out["quarantined"]
    assert q["path"] == str(bad) and q["project"] == "p" and q["adapter"] == "core"
    assert rule in q["rules"], q
    assert any(f"[{rule}]" in r for r in q["reasons"]), q
    names = {r["name"] for r in conn.execute("SELECT name FROM runs")}
    assert names == {"good"}, "a quarantined record never enters the ledger"
    assert conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0] == 1


def test_cli_prints_quarantined_and_exits_1_unless_no_strict(tmp_path, monkeypatch, capsys):
    ws = _workspace(tmp_path, _credential)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "cli.db"))
    assert cli.main(["runs", "scan", "--root", str(ws)]) == 1
    out = capsys.readouterr().out
    assert "QUARANTINED (1 record(s)" in out
    assert str(ws / "p" / "runs" / "bad") in out
    assert "attestation/run.json at $.config.uri: [credentials]" in out
    assert cli.main(["runs", "scan", "--root", str(ws), "--no-strict"]) == 0
    assert "QUARANTINED" in capsys.readouterr().out
    clean = _workspace(tmp_path / "clean")
    assert cli.main(["runs", "scan", "--root", str(clean)]) == 0
    assert "QUARANTINED" not in capsys.readouterr().out


def test_mcp_scan_returns_the_quarantine_and_leads_with_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    monkeypatch.setenv("RESEARCH_ROOT", str(_workspace(tmp_path, _non_canonical)))
    out = mcp_server._runs_scan_impl(confirm=True)
    assert out["ok"] is True
    assert out["message"].startswith("QUARANTINED 1 record(s)")
    assert out["quarantined"][0]["rules"] == ["canonical"]


def test_a_failed_record_carrying_the_importer_namespace_keeps_its_caveat():
    """ext.ledger-import's status word replaces the column's; it used to
    silence the did-not-complete caveat as well."""
    rec = _qrecord(
        "r",
        status="failed",
        ext={IMPORT_NAMESPACE: {"status": "recorded", "adapter": "generic"}},
    )
    row = ledger.row_from_core("p", Imported(rec, "r", "x", "core"))
    assert row.status == "recorded"
    assert any("has status failed" in c for c in row.run_caveats), row.run_caveats


def test_the_converters_lossy_steps_are_run_caveats():
    found = Found(
        project="p",
        name="r",
        source_path="x",
        started="not a time",
        config={"alpha": float("nan"), "nested": [1.0, float("inf")]},
        metrics=[FoundMetric("wer", 0.2), FoundMetric("wer", 0.3)],
    )
    rec = found_to_core(found, adapter="wandb", caveats=[], run_caveats=[], started_is_utc=True)
    caveats = rec.ext[IMPORT_NAMESPACE]["run_caveats"]
    assert any("config.alpha" in c and "config.nested[1]" in c for c in caveats), caveats
    assert any("duplicate metric" in c and "wer=0.3" in c for c in caveats), caveats
    assert any("'not a time'" in c for c in caveats), caveats
    assert [m.value for m in rec.metrics] == [0.2]
    clean = Found(project="p", name="r", source_path="x", metrics=[FoundMetric("wer", 0.2)])
    assert (
        found_to_core(clean, adapter="generic", caveats=[], run_caveats=[]).ext[IMPORT_NAMESPACE][
            "run_caveats"
        ]
        == []
    )


def test_non_finite_values_a_legacy_reader_skips_are_reported_lossy(
    conn, tmp_path, monkeypatch, capsys
):
    ws = tmp_path / "ws"
    results = ws / "proj" / "results"
    results.mkdir(parents=True)
    (results / "eval.json").write_text('{"wer": 0.2, "loss": NaN}')
    out = ledger.scan(conn, ws)
    assert out["scanned"] == {"proj": 1}
    (note,) = out["lossy"]["proj"]
    assert "generic reader skipped 1 non-finite value" in note and "loss" in note
    monkeypatch.setenv("RSS_DB", str(tmp_path / "cli.db"))
    assert cli.main(["runs", "scan", "--root", str(ws)]) == 0
    assert "LOSSY (proj):" in capsys.readouterr().out
