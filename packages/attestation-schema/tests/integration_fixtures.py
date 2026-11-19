"""Shared by the training-callback tests: a real dataset directory a toy model
trains on, the skip-or-require rule, and the checks every record must pass.

Stdlib only, so the bare 3.11 job can import it (the framework tests then
skip). Spec: docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import random
from pathlib import Path
from types import ModuleType

import attestation_schema as s
import pytest
from attestation_schema import conformance

DATASET_ID = "toy-linear"
SPLIT = "random"
ROWS = 32


def require(module: str) -> ModuleType:
    """Import a framework, or skip -- unless ATTESTATION_REQUIRE_INTEGRATIONS=1,
    where a missing framework fails: the CI job that installs the extras must
    not go green by skipping."""
    if os.environ.get("ATTESTATION_REQUIRE_INTEGRATIONS") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)


def make_dataset(root: Path) -> s.DatasetRecord:
    """`<root>/` as an `attestation.dataset/1` directory: a CSV of
    y = 2*x1 - x2 + 0.5 and a seeded random split file, written in the
    dataset layout's order (files -> manifest -> record)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    with (root / "data.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["x1", "x2", "y"])
        for _ in range(ROWS):
            x1, x2 = round(rng.random(), 4), round(rng.random(), 4)
            out.writerow([x1, x2, round(2 * x1 - x2 + 0.5, 4)])
    order = list(range(ROWS))
    random.Random(1).shuffle(order)
    split = {"train": sorted(order[:24]), "eval": sorted(order[24:])}
    (root / "split-random.json").write_text(json.dumps(split, sort_keys=True))
    content = s.write_dataset_manifest(root)
    files = s.dataset_files(root)
    split_sha = next(f.sha256 for f in files if f.path == "split-random.json")
    record = s.DatasetRecord(
        dataset_id=DATASET_ID,
        name="toy linear regression",
        version="1",
        created_at=None,
        producer=s.Producer("integration tests", None),
        code=s.UNKNOWN_CODE,
        content_sha256=content,
        files=files,
        rows=ROWS,
        columns=[s.Column(c, "float64", None) for c in ("x1", "x2", "y")],
        splits=[s.Split(SPLIT, "split-random.json", split_sha, "random", 1)],
        source_runs=[],
        license=None,
        ext={},
    )
    s.write_dataset(record, root)
    return record


def load_rows(root: Path, part: str) -> tuple[list[list[float]], list[float]]:
    """The rows of one split, read from the dataset directory's own bytes."""
    with (root / "data.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    idx = json.loads((root / "split-random.json").read_text())[part]
    xs = [[float(rows[i]["x1"]), float(rows[i]["x2"])] for i in idx]
    return xs, [float(rows[i]["y"]) for i in idx]


def assert_conforms_and_links(run_dir: Path, dataset: s.DatasetRecord) -> s.RunRecord:
    """The two properties every callback record must have: the conformance
    suite passes, and lineage reaches the dataset by digest with no issue."""
    report = conformance.check_run_dir(run_dir)
    assert report.ok, report.format()
    record = s.read(run_dir)
    lineage = s.resolve_lineage(record, [dataset])
    assert lineage.verified, lineage.format()
    via = sorted(e.via for e in lineage.edges)
    assert via == ["content", "split"], via
    return record


def assert_outputs_are_the_files(record: s.RunRecord, root: Path) -> list[s.DataDigest]:
    """Every `output` entry is the sha256 of a file that exists under `root`."""
    outputs = [d for d in record.data if d.role == "output"]
    assert outputs, "no checkpoint was bound"
    for d in outputs:
        assert d.sha256 == hashlib.sha256((root / d.name).read_bytes()).hexdigest(), d.name
    return outputs
