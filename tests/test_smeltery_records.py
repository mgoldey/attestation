"""smeltery's own runs, with the core record smeltery itself wrote, through
the ledger end to end (roadmap 13).

Spec: docs/superpowers/specs/2026-11-20-smeltery-adapter-design.md. The
fixture is smeltery's prebuilt benzoic-rhf and benzoic-pbe runs (one
campaign, two methods), converted by `smeltery record core`; regenerate it
with tests/data/smeltery/generate.sh, never by hand. Nothing here imports
smeltery: the record is the contract.
"""

import json
import shutil
from pathlib import Path

import attestation_schema as schema
import pytest
from attestation_schema import conformance

from attestation import claims, cli, ledger, mcp_server
from attestation.db import get_db

FIXTURE = Path(__file__).resolve().parent / "data" / "smeltery" / "workspace"
PROJECT = "mwe-benzoic"
RHF, PBE = "5f7df3db8c7b", "6a37df69178b"
FAMILY = "mwe-benzoic-cli"
FINDINGS = ["code inferred, dirty", "dependencies unknown: numpy, rdkit"]
CAVEAT_PREFIX = "provenance not verified for"


def _run_dirs() -> list[Path]:
    return sorted(p for p in (FIXTURE / PROJECT).iterdir() if p.is_dir())


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    shutil.copytree(FIXTURE, root)
    return root


@pytest.fixture
def scanned(ws, tmp_path, monkeypatch):
    # The records declare paired_ddE's direction "min"; a TOML that AGREES
    # keeps these tests on the declared path (no direction_source, no caveat).
    # test_the_records_own_direction_ranks_without_a_toml drops it.
    directions = tmp_path / "metric_direction.toml"
    directions.write_text('[metric_direction]\n"paired_ddE.mean" = "lower_is_better"\n')
    monkeypatch.setenv("LEDGER_METRIC_DIRECTION_FILE", str(directions))
    conn = get_db(tmp_path / "t.db")
    out = ledger.scan(conn, ws)
    yield conn, out
    conn.close()


def test_the_fixture_is_what_smeltery_wrote():
    """Both runs conform as committed (manifest, canonical bytes, digests),
    were written by smeltery's converter, and the scrub left no machine path."""
    for run_dir in _run_dirs():
        report = conformance.check_run_dir(run_dir)
        assert report.ok, report.violations
        record = schema.read(run_dir)
        assert record.producer.name == "smeltery record-core"
        assert record.family == FAMILY and "smeltery" in record.ext
        ferric = next(d for d in record.dependencies if d.name == "ferric")
        assert (ferric.provenance, ferric.dirty) == ("verified", False)
        assert ferric.evidence.startswith("VERIFIED from ferric.__build__")
        for f in run_dir.rglob("*"):
            if f.is_file():
                assert b"/home/" not in f.read_bytes(), f


def test_both_runs_import_through_the_core_adapter(scanned):
    conn, out = scanned
    assert out["scanned"] == {PROJECT: 2}
    assert not out.get("quarantined")
    for name in (RHF, PBE):
        found = ledger.detail(conn, PROJECT, name)
        assert (found["adapter"], found["family"], found["status"]) == ("core", FAMILY, "complete")
        # Labels (what a value is about: an analogue, a funnel stage) fold into split.
        splits = {m["split"] for m in found["metrics"] if m["metric"] == "paired_ddE.mean"}
        assert splits == {"analogue=4-F", "analogue=4-Cl"}


def test_detail_lists_what_the_record_leaves_unverified(scanned, tmp_path, monkeypatch):
    """A dirty editable checkout and two packages the legacy record never
    identified; ferric's clean, stamped build contributes nothing."""
    conn, _ = scanned
    for name in (RHF, PBE):
        assert ledger.detail(conn, PROJECT, name)["provenance"] == FINDINGS
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    out = mcp_server._runs_detail_impl(PROJECT, RHF)
    assert out["run"]["provenance"] == FINDINGS


def test_the_records_own_direction_ranks_without_a_toml(ws, tmp_path):
    """smeltery declares paired_ddE.mean direction min in both records, so
    the family ranks with no metric_direction.toml at all (record-declared-
    direction spec) -- and compare with no metric named picks it."""
    conn = get_db(tmp_path / "bare.db")
    ledger.scan(conn, ws)
    out = ledger.compare(conn, FAMILY)
    assert out["metric"] == "paired_ddE.mean"
    assert (out["direction"], out["direction_source"]) == ("lower_is_better", "records")
    assert out["winner"] == PBE
    conn.close()


def test_compare_carries_one_grouped_provenance_caveat(scanned):
    conn, _ = scanned
    out = ledger.compare(conn, FAMILY, metric="paired_ddE.mean")
    assert out["winner"] == PBE
    prov = [c for c in out["caveats"] if c.startswith(CAVEAT_PREFIX)]
    assert prov == [
        f"{CAVEAT_PREFIX} {RHF}, {PBE}: {'; '.join(FINDINGS)} -- the values stand, but which"
        " code and environment produced them is not established (runs.detail lists each run's)"
    ]


def _draft(tmp_path) -> Path:
    doc = tmp_path / "draft.md"
    rhf = schema.read(FIXTURE / PROJECT / "benzoic-rhf")
    (ddE,) = [
        m.value
        for m in rhf.metrics
        if m.name == "paired_ddE.mean" and m.labels == {"analogue": "4-F"}
    ]
    doc.write_text(
        f"RHF puts 4-F at {ddE:.3f} kcal/mol.\n"
        f"<!-- claim: {PROJECT}/{RHF} metric=paired_ddE.mean split=analogue=4-F"
        f" value={ddE:.3f} tol=0.001 -->\n"
        "PBE puts 4-F at 2.000 kcal/mol.\n"
        f"<!-- claim: {PROJECT}/{PBE} metric=paired_ddE.mean split=analogue=4-F"
        " value=2.0 tol=0.001 -->\n"
    )
    return doc


def test_unverified_provenance_is_a_caveat_in_claims_never_a_contradiction(scanned, tmp_path):
    """The mixed-case metric matches (it used to read `unsupported`), the
    agreeing claim is `supported`, the disagreeing one `contradicted` on its
    number, and both carry the run's provenance caveat."""
    conn, _ = scanned
    verdicts = claims.check(conn, _draft(tmp_path))["verdicts"]
    assert [v.verdict for v in verdicts] == ["supported", "contradicted"]
    for v, name in zip(verdicts, (RHF, PBE), strict=True):
        assert v.caveats == [
            f"{CAVEAT_PREFIX} {name}: {'; '.join(FINDINGS)} -- the values stand, but which"
            " code and environment produced them is not established (runs.detail lists each"
            " run's)"
        ]


def test_the_cli_prints_caveats_and_exits_on_the_contradiction_alone(scanned, tmp_path, capsys):
    doc = _draft(tmp_path)
    db = str(tmp_path / "t.db")
    assert cli.main(["claims", str(doc), "--db", db]) == 1
    out = capsys.readouterr().out
    assert out.count(f"caveat: {CAVEAT_PREFIX}") == 2
    assert "2 claim(s): 1 contradicted, 1 supported" in out

    doc.write_text(doc.read_text().split("PBE puts")[0])
    assert cli.main(["claims", str(doc), "--db", db]) == 0, "a caveat alone must not fail"


def test_claims_check_rows_carry_the_caveats(scanned, tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    out = mcp_server._claims_check_impl(str(_draft(tmp_path)))
    assert out["counts"] == {"supported": 1, "contradicted": 1}
    assert all(r["caveats"][0].startswith(CAVEAT_PREFIX) for r in out["claims"])
    json.dumps(out)  # the payload stays plain JSON
