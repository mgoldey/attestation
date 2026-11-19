"""`resolve_lineage`: ferric build -> smeltery run -> dataset -> training run, by digest.

The committed chain (tests/data/conformance/lineage/) resolves with no issue;
each issue kind is then produced by ONE mutation of it, against the
unmutated control, so a check that always or never fires is caught
(docs/measurement-lessons.md, 4). Spec:
docs/superpowers/specs/2026-11-19-dataset-records-design.md (Decision 6).
"""

from __future__ import annotations

import csv
import dataclasses
import json

import attestation_schema as s
import pytest
from attestation_schema.lineage import BROKEN, ISSUE_KINDS, UNVERIFIED
from conformance_fixtures import DATA, DATASET_ID, FERRIC_SHA

LINEAGE = DATA / "lineage"


def _chain() -> tuple[s.RunRecord, s.DatasetRecord, list[s.RunRecord]]:
    training = s.read(LINEAGE / "training")
    dataset = s.read_dataset(LINEAGE / "dataset")
    sources = [s.read(LINEAGE / "smeltery-a"), s.read(LINEAGE / "smeltery-b")]
    return training, dataset, sources


def _kinds(lineage: s.Lineage) -> list[tuple[str, str, str]]:
    return [(i.severity, i.kind, i.at) for i in lineage.issues]


def test_the_chain_resolves_end_to_end_by_digest():
    training, dataset, sources = _chain()
    lin = s.resolve_lineage(training, [dataset], sources)
    assert lin.verified and lin.ok, lin.format()
    assert [n.kind for n in lin.nodes] == ["run", "dataset", "run", "run"]
    assert lin.root == f"run:{s.record_digest(training)}"
    ds_key = f"dataset:{s.dataset_digest(dataset)}"
    into_run = [(e.source, e.via, e.at) for e in lin.upstream(lin.root)]
    assert into_run == [(ds_key, "content", "data[0]"), (ds_key, "split", "data[1]")]
    into_ds = [(e.source, e.via) for e in lin.upstream(ds_key)]
    assert into_ds == [(f"run:{s.record_digest(r)}", "source_run") for r in sources]
    json.dumps(lin.to_dict())  # plain JSON
    assert "lineage: verified" in lin.format()


def test_the_ferric_build_is_the_source_runs_verified_dependency_and_the_tables_column():
    """The first arrow: each smeltery run's own record carries ferric's build
    identity; the exported table copied ferric_git_sha from the same legacy
    record. A stdlib reader cannot read Parquet, so this cross-check is the
    producer's -- here it is made on the fixture's CSV stand-in."""
    training, dataset, sources = _chain()
    lin = s.resolve_lineage(training, [dataset], sources)
    builds = lin.dependencies("ferric")
    assert len(builds) == 2
    assert {(d.build_sha, d.provenance) for _, d in builds} == {(FERRIC_SHA, "verified")}
    by_run = {lin.node(k).record.run_id: d.build_sha for k, d in builds}
    rows = list(csv.DictReader((LINEAGE / "dataset" / "runs.csv").open()))
    assert {r["run_id"] for r in rows} == set(by_run)
    for row in rows:
        assert row["ferric_git_sha"] == by_run[row["run_id"]]


def test_the_exports_record_sha256_column_is_not_the_core_digest():
    """smeltery finding: the column is the legacy self-hash, which resolves
    nothing; source_runs must carry sha256(attestation/run.json)."""
    _, dataset, sources = _chain()
    rows = list(csv.DictReader((LINEAGE / "dataset" / "runs.csv").open()))
    legacy = {r["record_sha256"] for r in rows}
    core = {sr.record_sha256 for sr in dataset.source_runs or []}
    assert core == {s.record_digest(r) for r in sources}
    assert not legacy & core


def test_the_split_file_names_the_table_the_dataset_lists():
    _, dataset, _ = _chain()
    split = json.loads((LINEAGE / "dataset" / "split-scaffold.json").read_bytes())
    table = next(f for f in dataset.files if f.path == "runs.csv")
    assert split["table"]["sha256"] == table.sha256


def test_a_run_that_digested_the_table_links_by_file():
    training, dataset, sources = _chain()
    table = next(f for f in dataset.files if f.path == "runs.csv")
    run = dataclasses.replace(training, data=[s.DataDigest("runs.parquet", table.sha256, "train")])
    lin = s.resolve_lineage(run, [dataset], sources)
    assert lin.verified
    assert [(e.via, e.detail) for e in lin.upstream(lin.root)] == [
        ("file", "role train, file runs.csv")
    ]


def test_data_that_is_not_a_dataset_is_not_an_issue():
    training, dataset, sources = _chain()
    run = dataclasses.replace(
        training, data=[*training.data, s.DataDigest("config.yaml", "e" * 64, "input")]
    )
    run = dataclasses.replace(run, data=[*run.data, s.DataDigest("weights", None, "input")])
    assert s.resolve_lineage(run, [dataset], sources).verified


# ---- one mutation per issue kind --------------------------------------------------------------


def _mutant(kind: str):
    training, dataset, sources = _chain()
    a, b = sources
    srs = list(dataset.source_runs or [])
    if kind == "missing-source-run":
        return training, [dataset], [a]
    if kind == "source-run-digest-mismatch":
        rerun = dataclasses.replace(b, finished_at="2026-11-19T11:00:00.000000Z")
        return training, [dataset], [a, rerun]
    if kind == "source-run-identity-mismatch":
        srs[0] = dataclasses.replace(srs[0], run_id="someone-else")
        return training, [dataclasses.replace(dataset, source_runs=srs)], sources
    if kind == "dataset-digest-mismatch":
        data = [dataclasses.replace(training.data[0], sha256="d" * 64), training.data[1]]
        return dataclasses.replace(training, data=data), [dataset], sources
    if kind == "undigested-data":
        data = [dataclasses.replace(training.data[0], sha256=None), training.data[1]]
        return dataclasses.replace(training, data=data), [dataset], sources
    if kind == "undigested-source-run":
        srs[1] = dataclasses.replace(srs[1], record_sha256=None)
        return training, [dataclasses.replace(dataset, source_runs=srs)], sources
    if kind == "unrecorded-source-runs":
        return training, [dataclasses.replace(dataset, source_runs=None)], sources
    if kind == "ambiguous":
        relabelled = dataclasses.replace(dataset, version="1-relicensed", license="MIT")
        return training, [dataset, relabelled], sources
    raise ValueError(kind)


EXPECTED = {
    "missing-source-run": [(BROKEN, "missing-source-run", "source_runs[1]")],
    "source-run-digest-mismatch": [(BROKEN, "source-run-digest-mismatch", "source_runs[1]")],
    "source-run-identity-mismatch": [(BROKEN, "source-run-identity-mismatch", "source_runs[0]")],
    "dataset-digest-mismatch": [(BROKEN, "dataset-digest-mismatch", "data[0]")],
    "undigested-data": [(UNVERIFIED, "undigested-data", "data[0]")],
    "undigested-source-run": [(UNVERIFIED, "undigested-source-run", "source_runs[1]")],
    "unrecorded-source-runs": [(UNVERIFIED, "unrecorded-source-runs", "source_runs")],
    "ambiguous": [(UNVERIFIED, "ambiguous", "data[0]"), (UNVERIFIED, "ambiguous", "data[1]")],
}


def test_every_issue_kind_has_a_mutant():
    assert set(EXPECTED) == set(ISSUE_KINDS)


@pytest.mark.parametrize("kind", sorted(ISSUE_KINDS))
def test_each_mutation_gives_exactly_its_issue(kind):
    lin = s.resolve_lineage(*_mutant(kind))
    assert _kinds(lin) == EXPECTED[kind], lin.format()
    assert lin.ok == (ISSUE_KINDS[kind] == UNVERIFIED)
    assert not lin.verified
    assert ("BROKEN" in lin.format()) == (ISSUE_KINDS[kind] == BROKEN)


def test_an_identity_mismatch_still_links_by_digest():
    lin = s.resolve_lineage(*_mutant("source-run-identity-mismatch"))
    assert len([n for n in lin.nodes if n.kind == "run"]) == 3


def test_a_name_only_link_is_kept_and_several_are_all_kept():
    training, dataset, sources = _chain()
    a, b = sources
    srs = list(dataset.source_runs or [])
    srs[1] = dataclasses.replace(srs[1], record_sha256=None)
    rerun = dataclasses.replace(b, finished_at="2026-11-19T11:00:00.000000Z")
    lin = s.resolve_lineage(
        training, [dataclasses.replace(dataset, source_runs=srs)], [a, b, rerun]
    )
    assert [k for _, k, _ in _kinds(lin)] == ["undigested-source-run", "ambiguous"]
    assert len([n for n in lin.nodes if n.kind == "run"]) == 4  # training, a, b, rerun


def test_a_mismatch_names_the_versions_it_knows():
    lin = s.resolve_lineage(*_mutant("dataset-digest-mismatch"))
    (issue,) = lin.issues
    _, dataset, _ = _chain()
    assert DATASET_ID in issue.message and dataset.content_sha256 in issue.message
    # The split entry still links: one broken entry does not hide the other.
    assert [e.at for e in lin.upstream(lin.root)] == ["data[1]"]


def test_lineage_walks_upstream_through_several_datasets_and_terminates_on_a_cycle():
    training, dataset, sources = _chain()
    a, b = sources
    # smeltery-a itself read an earlier dataset, which names training by
    # name only (a cycle no digest could make).
    earlier = dataclasses.replace(
        s.read_dataset(DATA / "datasets" / "valid" / "minimal"),
        dataset_id="earlier",
        source_runs=[s.SourceRun(training.run_id, training.family, None)],
    )
    a2 = dataclasses.replace(
        a, data=[*a.data, s.DataDigest("earlier", earlier.content_sha256, "input")]
    )
    first = s.SourceRun(a2.run_id, a2.family, s.record_digest(a2))
    ds2 = dataclasses.replace(dataset, source_runs=[first, *(dataset.source_runs or [])[1:]])
    run = dataclasses.replace(training, data=s.dataset_lineage_entries(ds2, split="scaffold"))
    lin = s.resolve_lineage(run, [ds2, earlier], [a2, b, training])
    kinds = [k for _, k, _ in _kinds(lin)]
    assert "undigested-source-run" in kinds
    labels = [n.label for n in lin.nodes]
    assert "earlier@1" in labels and any(lab.endswith("07693d07e8b9") for lab in labels)
    text = lin.format()
    assert "(shown above)" in text and text.count("built on ferric") == 2


def test_the_root_need_not_be_among_the_runs():
    training, dataset, sources = _chain()
    assert s.resolve_lineage(training, [dataset], [training, *sources]).verified
    lin = s.resolve_lineage(training, [], [])
    assert lin.verified and [n.kind for n in lin.nodes] == ["run"]
