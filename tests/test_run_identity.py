"""What a core record leaves unverified about its code and environment.

Spec: docs/superpowers/specs/2026-11-20-smeltery-adapter-design.md
(Decision 2). Every finding is read from the frozen schema's own fields, so
these records are built with `attestation_schema.from_dict`, no producer.
"""

import attestation_schema as schema
import pytest

from attestation import ledger
from attestation.db import get_db
from attestation.ledger_adapters import IMPORT_NAMESPACE, Imported
from attestation.run_identity import (
    MAX_NAMED_RUNS,
    provenance_caveats,
    provenance_findings,
)

SHA = "a" * 40

VERIFIED_CODE = {
    "git_sha": SHA,
    "dirty": False,
    "repo": None,
    "version": None,
    "provenance": "verified",
    "evidence": "test",
}


def _dep(name, provenance="verified", dirty=None, version="1.0"):
    return {
        "name": name,
        "version": None if provenance == "unknown" else version,
        "build_sha": None,
        "dirty": dirty,
        "provenance": provenance,
        "evidence": None if provenance == "unknown" else "test",
        "source": None,
    }


def _record(code=None, dependencies=(), ext=None, run_id="r1") -> schema.RunRecord:
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": run_id,
            "family": "sweep",
            "arm": None,
            "status": "complete",
            "started_at": None,
            "finished_at": None,
            "producer": {"name": "test", "version": None},
            "code": {**VERIFIED_CODE, **(code or {})},
            "config": {},
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
            "dependencies": None if dependencies is None else list(dependencies),
            "ext": ext or {},
        }
    )


def test_a_verified_clean_record_has_no_findings():
    """A caveat must be earned: nothing to say about a fully verified run."""
    assert provenance_findings(_record(dependencies=[_dep("numpy")])) == []
    # `dependencies: []` is "observed, nothing to name" -- not a gap.
    assert provenance_findings(_record(dependencies=[])) == []


@pytest.mark.parametrize(
    ("code", "want"),
    [
        ({"provenance": "inferred"}, "code inferred"),
        ({"provenance": "unknown", "git_sha": None, "dirty": None}, "code unknown"),
        ({"dirty": True}, "code dirty"),
        ({"provenance": "inferred", "dirty": True}, "code inferred, dirty"),
    ],
)
def test_each_code_gap_is_one_finding(code, want):
    assert provenance_findings(_record(code=code, dependencies=[])) == [want]


def test_unobserved_dirtiness_is_not_a_finding():
    """null is the schema's "not observed"; beside a verified identity it is
    not a second claim about the run."""
    rec = _record(code={"dirty": None}, dependencies=[_dep("numpy", dirty=None)])
    assert provenance_findings(rec) == []


def test_dependency_gaps_are_grouped_by_level_with_names_sorted():
    rec = _record(
        dependencies=[
            _dep("rdkit", "unknown"),
            _dep("numpy", "unknown"),
            _dep("scipy", "inferred"),
            # A stamped build is verified AND dirty: the stamp names the build
            # exactly, and the build came from uncommitted source.
            _dep("ferric", "verified", dirty=True),
            _dep("python"),
        ]
    )
    assert provenance_findings(rec) == [
        "dependencies inferred: scipy",
        "dependencies unknown: numpy, rdkit",
        "dependencies dirty: ferric",
    ]


def test_an_unobserved_environment_is_a_finding():
    assert provenance_findings(_record(dependencies=None)) == ["dependencies not recorded"]


def test_runs_sharing_a_gap_share_one_caveat_and_long_lists_are_capped():
    gap = ["code dirty"]
    names = [f"seed{i}" for i in range(MAX_NAMED_RUNS + 2)]
    out = provenance_caveats([(n, gap) for n in names] + [("clean", []), ("odd", ["code unknown"])])
    assert len(out) == 2
    assert out[0].startswith(f"provenance not verified for {', '.join(names[:MAX_NAMED_RUNS])}")
    assert " and 2 more: code dirty -- " in out[0]
    assert out[1].startswith("provenance not verified for odd: code unknown")
    assert provenance_caveats([("clean", [])]) == []


def _row(record, name="r1"):
    return ledger.row_from_core("p", Imported(record, name, "/nowhere", "core"))


def test_only_a_native_record_carries_findings():
    """An importer's code/dependencies are its converter's "did not say",
    the same for every run that reader produces -- a reader property, so no
    per-run provenance caveat."""
    gappy = {"provenance": "unknown", "git_sha": None, "dirty": None}
    assert _row(_record(code=gappy, dependencies=None)).provenance == [
        "code unknown",
        "dependencies not recorded",
    ]
    imported = _record(
        code=gappy, dependencies=None, ext={IMPORT_NAMESPACE: {"adapter": "generic"}}
    )
    assert _row(imported).provenance == []


def test_findings_survive_the_table_and_never_change_a_verdict(tmp_path):
    """Stored in runs.caveats_json; shown by detail, compare and the claim
    checker; the claim's verdict is the number's alone."""
    from attestation import claims

    ws = tmp_path / "ws"
    schema.write(
        _record(code={"dirty": True}, dependencies=None, run_id="dirty"),
        ws / "p" / "runs" / "dirty",
    )
    schema.write(_record(dependencies=[], run_id="clean"), ws / "p" / "runs" / "clean")
    conn = get_db(tmp_path / "t.db")
    try:
        ledger.scan(conn, ws)
        assert ledger.detail(conn, "p", "dirty")["provenance"] == [
            "code dirty",
            "dependencies not recorded",
        ]
        assert ledger.detail(conn, "p", "clean")["provenance"] == []
        caveats = ledger.compare(conn, "sweep", metric="wer")["caveats"]
        prov = [c for c in caveats if c.startswith("provenance not verified")]
        assert prov == [
            "provenance not verified for dirty: code dirty; dependencies not recorded -- the"
            " values stand, but which code and environment produced them is not established"
            " (runs.detail lists each run's)"
        ]

        doc = tmp_path / "doc.md"
        doc.write_text(
            "<!-- claim: p/dirty metric=wer value=0.2 -->\n"
            "<!-- claim: p/dirty metric=wer value=0.9 -->\n"
            "<!-- claim: p/clean metric=wer value=0.2 -->\n"
        )
        verdicts = claims.check(conn, doc)["verdicts"]
        assert [v.verdict for v in verdicts] == ["supported", "contradicted", "supported"]
        assert verdicts[0].caveats == prov and verdicts[1].caveats == prov
        assert verdicts[2].caveats == []
    finally:
        conn.close()
