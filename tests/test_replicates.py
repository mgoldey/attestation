"""Seed replicates: pooled into one arm, with statistics that say their limits.

See docs/superpowers/specs/2026-11-19-replicate-statistics-design.md. The
acceptance list there is numbered; tests below name the item they cover.
Each guard was mutation-checked (measurement-lessons section 4): removing the
`replicates.group` call from `ledger._compare` fails the pooling, veto and
separability tests; making the single-run caveat unconditional fails
`test_a_fully_replicated_family_does_not_deny_replication`.
"""

import json
from pathlib import Path

import pytest

from attestation import ledger, replicates
from attestation.db import get_db
from attestation.ledger_adapters import generic, trackers

ROOT = Path(__file__).resolve().parent.parent


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def arm(name, value, *, split=None, n=100, step=None):
    """A per-run arm dict shaped like ledger._arms_for_run's."""
    return {
        "name": name,
        "status": "recorded",
        "value": value,
        "step": step,
        "split": split,
        "source_path": f"/r/{name}.json",
        "n": n,
    }


def ids(arms, *, run_id_suffix=False):
    """The `identities` `ledger._compare` hands `replicates.group`: each arm's
    run read the way the import reads a legacy name (`trackers.replicate_of`
    -> the record's `arm` and `seeds`), a nested `[key]` kept on the arm."""
    out = {}
    for a in arms:
        head, bracket, tail = a["name"].partition("[")
        found = trackers.replicate_of(head, run_id_suffix=run_id_suffix)
        if found is not None:
            out[a["name"]] = (found[0] + bracket + tail, {"seed": found[1]})
    return out


def group(arms, configs, **kw):
    return replicates.group(arms, configs, ids(arms, **kw))


def seeded_sweep(root: Path, arms: dict[str, list[float]], metric: str = "accuracy") -> None:
    """`results/<arm>_seed<i>.json` with a `seed` field, the commonest layout."""
    for name, values in arms.items():
        for i, value in enumerate(values, start=1):
            write(
                root / "proj" / "results" / f"{name}_seed{i}.json",
                json.dumps({"seed": i, metric: value, "n_records": 500}),
            )


def insert_run(conn, name, family, config, values: dict[str, float]):
    """A run row with an explicit config_json -- the generic JSON reader keeps
    only `seed` in config, so a config that differs elsewhere is built here.
    `arm`/`seeds_json` are what the import would have stored for the name."""
    found = trackers.replicate_of(name)
    cur = conn.execute(
        "INSERT INTO runs(project, name, family, status, source_path, config_json, arm,"
        " seeds_json) VALUES ('p', ?, ?, 'recorded', ?, ?, ?, ?)",
        (
            name,
            family,
            f"/tmp/{name}.json",
            json.dumps(config) if config else None,
            found[0] if found else None,
            json.dumps({"seed": found[1]}) if found else None,
        ),
    )
    for metric, value in values.items():
        conn.execute(
            "INSERT INTO run_metrics(run_id, metric, value) VALUES (?, ?, ?)",
            (cur.lastrowid, metric, value),
        )
    conn.commit()


# --- the seed token and the family ------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("kd_t4_seed1", ("kd_t4", 1)),
        ("kd_t4_seed_2", ("kd_t4", 2)),
        ("kd-t4-seed-3", ("kd-t4", 3)),
        ("kd_seed=4_t4", ("kd_t4", 4)),
        ("kd_t4/seed5", ("kd_t4", 5)),
        ("Model_SEED7", ("Model", 7)),
    ],
)
def test_a_seed_token_is_found_and_removed(name, expected):
    """Read once, at import, into the core record's `arm` and `seeds`."""
    assert trackers.replicate_of(name) == expected


@pytest.mark.parametrize("name", ["seed1", "kd_t4", "seedling_3", "kd_seeded", "a_seed1_seed2"])
def test_names_that_are_not_one_seeded_configuration(name):
    """No token, a bare seed (no configuration named), a word that merely
    starts with "seed", and two tokens (which one varies?) all refuse."""
    assert trackers.replicate_of(name) is None


def test_a_seed_token_never_changes_a_family():
    """The flat seeded layout used to make one family PER CONFIGURATION, so a
    sweep could not be compared at all; now it groups as the unseeded names
    would."""
    assert generic.family_of("clf_knn_seed3") == generic.family_of("clf_knn") == "clf"
    assert generic.family_of("fam_a_seed1") == generic.family_of("fam_b_seed2") == "fam"
    assert generic.family_of("lr_0.01_seed1") == "lr"
    # nothing left but one token: it was family `run` before and still is
    assert generic.family_of("run_seed1") == "run"
    # a bare seed stem keeps its old family (pinned elsewhere too)
    assert generic.family_of("seed_3") == "seed"


# --- statistics ---------------------------------------------------------------


def test_bootstrap_is_pinned_and_deterministic():
    """Acceptance 2. Exact floats: a change in the RNG draw, the resample
    count or the percentile rule moves them, and CI runs this on 3.12 and
    3.13."""
    values = [0.9796, 0.9833, 0.9796, 0.987, 0.9852]
    assert replicates.bootstrap_ci(values) == (0.98034, 0.98554)
    assert replicates.bootstrap_ci(values) == replicates.bootstrap_ci(list(values))
    assert replicates.bootstrap_ci([1.0, 2.0, 3.0]) == (1.0, 3.0)
    assert replicates.mean(values) == 0.9829399999999999
    assert replicates.sample_sd(values) == pytest.approx(0.0033178306165324114, rel=1e-12)


def test_bootstrap_refuses_no_values():
    with pytest.raises(ValueError):
        replicates.bootstrap_ci([])


def test_effect_size_needs_spread_on_both_sides():
    top = {"n_runs": 3, "mean": 1.0, "sd": 0.1}
    second = {"n_runs": 3, "mean": 0.8, "sd": 0.1}
    assert replicates.cohen_d(top, second) == 2.0
    assert replicates.cohen_d(top, {"value": 0.8}) is None
    assert replicates.cohen_d(top, {"n_runs": 3, "mean": 1.0, "sd": 0.0}) == 0.0
    flat = {"n_runs": 2, "mean": 1.0, "sd": 0.0}
    assert replicates.cohen_d(flat, flat) is None


# --- grouping (pure) ----------------------------------------------------------


def test_group_pools_replicates_and_keeps_the_contract_keys():
    arms = [arm(f"a_seed{i}", v) for i, v in enumerate([0.90, 0.92, 0.94], 1)]
    out, notes = group(arms, {})
    assert notes == []
    (pooled,) = out
    assert pooled["name"] == "a"
    assert pooled["n_runs"] == 3
    assert pooled["value"] == pooled["mean"] == pytest.approx(0.92)
    assert pooled["runs"] == ["a_seed1", "a_seed2", "a_seed3"]
    assert pooled["ci_low"] <= pooled["mean"] <= pooled["ci_high"]
    assert (pooled["min"], pooled["max"]) == (0.90, 0.94)


def test_a_single_seed_is_left_exactly_as_it_was():
    single = arm("a_seed1", 0.9)
    out, notes = group([single], {})
    assert out == [single] and notes == []


def test_runs_listed_per_arm_are_capped():
    arms = [arm(f"a_seed{i}", 0.9 + i / 1000) for i in range(20)]
    (pooled,), _ = group(arms, {})
    assert pooled["n_runs"] == 20
    assert len(pooled["runs"]) == replicates.MAX_RUN_NAMES


def test_two_runs_report_a_range_and_no_interval():
    (pooled,), _ = group([arm("a_seed1", 0.9), arm("a_seed2", 0.8)], {})
    assert pooled["n_runs"] == 2
    assert pooled["ci_low"] is None and pooled["ci_high"] is None
    assert (pooled["min"], pooled["max"]) == (0.8, 0.9)


def test_seed_splits_are_not_read_as_different_splits():
    """The generic reader records `_split_from(stem)` -- for `a_seed1` that
    is `seed1`. Read literally every replicate would differ and none pool."""
    arms = [arm(f"a_seed{i}", 0.9, split=f"seed{i}") for i in (1, 2, 3)]
    (pooled,), notes = group(arms, {})
    assert pooled["n_runs"] == 3 and pooled["split"] is None and notes == []


@pytest.mark.parametrize(
    ("arms", "configs", "reason"),
    [
        (
            [arm("a_seed1", 0.9), arm("a_seed2", 0.8)],
            {"a_seed1": '{"seed": 1, "lr": 0.1}', "a_seed2": '{"seed": 2, "lr": 0.2}'},
            "their configs differ in lr",
        ),
        ([arm("a_seed1", 0.9), arm("a-seed1", 0.8)], {}, "two of them claim the same seed"),
        (
            [arm("a_seed1", 0.9, split="test"), arm("a_seed2", 0.8, split="val")],
            {},
            "they report different splits",
        ),
        (
            [arm("a", 0.7), arm("a_seed1", 0.9), arm("a_seed2", 0.8)],
            {},
            "a run named a with no seed token also exists",
        ),
    ],
)
def test_a_veto_keeps_every_member_apart_and_says_why(arms, configs, reason):
    out, notes = group(arms, configs)
    assert sorted(a["name"] for a in out) == sorted(a["name"] for a in arms)
    assert not any(a.get("n_runs") for a in out)
    assert len(notes) == 1 and reason in notes[0], notes


def test_a_missing_config_does_not_veto():
    """Most JSON results carry only a seed; absence is not disagreement."""
    configs = {"a_seed1": '{"seed": 1, "lr": 0.1}', "a_seed2": None}
    (pooled,), notes = group([arm("a_seed1", 0.9), arm("a_seed2", 0.8)], configs)
    assert pooled["n_runs"] == 2 and notes == []


def test_config_only_lookalikes_are_reported_never_merged():
    """Hydra/wandb names carry no seed token. Config alone never merges --
    a plain result's config is `{"seed": ...}` only, so "equal apart from
    seed" would be vacuously true -- but a real config that differs ONLY in
    seed is pointed out."""
    arms = [arm("train/0", 0.9), arm("train/1", 0.8), arm("train/2", 0.7)]
    configs = {
        "train/0": {"lr": 0.1, "seed": 0},
        "train/1": {"lr": 0.1, "seed": 1},
        "train/2": {"lr": 1.0, "seed": 0},
    }
    out, notes = group(arms, configs)
    assert len(out) == 3
    assert len(notes) == 1 and notes[0].startswith("train/0, train/1 have identical configs")
    # seed-only configs are NOT lookalikes, or every JSON sweep would warn
    seed_only = {"x_a": {"seed": "1"}, "x_b": {"seed": "2"}}
    _, notes = group([arm("x_a", 0.9), arm("x_b", 0.8)], seed_only)
    assert notes == []


# --- through the ledger -------------------------------------------------------


def test_seeded_sweep_scans_to_one_arm_per_configuration(tmp_path):
    """Acceptance 1: fam_a_seed1..3 and fam_b_seed1..3 are TWO arms of three
    runs, not six rivals in two families."""
    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.91, 0.92], "fam_b": [0.80, 0.81, 0.82]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    assert [f["family"] for f in ledger.families(conn)] == ["fam"]

    result = ledger.compare(conn, "fam", metric="accuracy")

    assert [a["name"] for a in result["arms"]] == ["fam_a", "fam_b"]
    assert [a["n_runs"] for a in result["arms"]] == [3, 3]
    assert result["arms"][0]["runs"] == ["fam_a_seed1", "fam_a_seed2", "fam_a_seed3"]
    assert result["winner"] == "fam_a"
    assert result["arms"][0]["n"] == 500
    conn.close()


def test_different_config_is_not_merged_and_a_caveat_says_so(tmp_path):
    """Acceptance 3, through compare()."""
    conn = get_db(tmp_path / "t.db")
    insert_run(conn, "m_seed1", "m", {"seed": 1, "lr": 0.1}, {"accuracy": 0.9})
    insert_run(conn, "m_seed2", "m", {"seed": 2, "lr": 0.5}, {"accuracy": 0.8})

    result = ledger.compare(conn, "m", metric="accuracy")

    assert [a["name"] for a in result["arms"]] == ["m_seed1", "m_seed2"]
    assert not any("n_runs" in a for a in result["arms"])
    assert any("m_seed1, m_seed2 differ only by seed" in c and "lr" in c for c in result["caveats"])
    conn.close()


def test_overlapping_intervals_are_not_separable_and_both_are_named(tmp_path):
    """Acceptance 5, overlap."""
    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.80, 0.85], "fam_b": [0.88, 0.78, 0.83]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "fam", metric="accuracy")

    assert result["separable"] is False
    (overlap,) = [c for c in result["caveats"] if "intervals overlap" in c]
    a, b = result["arms"]
    assert f"fam_a [{a['ci_low']:.4g}, {a['ci_high']:.4g}]" in overlap
    assert f"fam_b [{b['ci_low']:.4g}, {b['ci_high']:.4g}]" in overlap
    assert any("not a significance test" in c for c in result["caveats"])
    conn.close()


def test_disjoint_intervals_are_separable(tmp_path):
    """Acceptance 5, disjoint. The 5%-gap rule of thumb is not printed
    beside an interval comparison that answers the same question."""
    seeded_sweep(tmp_path, {"fam_a": [0.980, 0.983, 0.981], "fam_b": [0.970, 0.968, 0.969]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "fam", metric="accuracy")

    assert result["separable"] is True
    assert result["effect_size"]["between"] == ["fam_a", "fam_b"]
    assert result["effect_size"]["cohen_d"] > 0
    assert not any("too close to call" in c for c in result["caveats"])
    assert not any("intervals overlap" in c for c in result["caveats"])
    conn.close()


def test_two_runs_per_arm_give_no_interval_and_cannot_tell(tmp_path):
    """Acceptance 5, n_runs == 2."""
    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.92], "fam_b": [0.80, 0.81]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "fam", metric="accuracy")

    assert [a["n_runs"] for a in result["arms"]] == [2, 2]
    assert all(a["ci_low"] is None for a in result["arms"])
    assert result["separable"] is None
    assert any("a CI from 2 runs is not a CI" in c for c in result["caveats"])
    conn.close()


def test_a_fully_replicated_family_does_not_deny_replication(tmp_path):
    """The single-run caveat used to be unconditional, and was false on every
    sweep that had seeds."""
    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.91, 0.92], "fam_b": [0.80, 0.81, 0.82]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "fam", metric="accuracy")

    assert not any("no seed replication" in c for c in result["caveats"]), result["caveats"]
    conn.close()


def test_mixed_and_unequal_families_say_which_arms_are_which(tmp_path):
    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.91, 0.92], "fam_b": [0.80, 0.81, 0.82, 0.83]})
    write(tmp_path / "proj" / "results" / "fam_c.json", json.dumps({"accuracy": 0.5}))
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "fam", metric="accuracy")

    caveats = result["caveats"]
    assert any(c.startswith("fam_c is a single run beside replicated arms") for c in caveats)
    assert any("replicate counts differ across arms (3-4 runs)" in c for c in caveats)
    assert not any(c == replicates.SINGLE_RUN_CAVEAT for c in caveats)
    conn.close()


def test_single_run_family_output_is_unchanged():
    """Acceptance 4: the example workspace's kdsweep (whose README and
    golden-path test pin the output) keeps exactly its keys and its
    single-run caveat. kdsweep_t4 (seed 42) and kdsweep_t4b (seed 7) carry
    no seed token in their names, so they are NOT pooled."""
    conn = get_db(":memory:")
    ledger.scan(conn, ROOT / "examples" / "workspace")

    result = ledger.compare(conn, "kdsweep", metric="wer")

    assert set(result) == {
        "family",
        "metric",
        "direction",
        "arms",
        "winner",
        "without_metric",
        "corpus",
        "caveats",
    }
    assert [a["name"] for a in result["arms"]] == [
        "kdsweep_t4",
        "kdsweep_t4b",
        "kdsweep_t2",
        "kdsweep_baseline",
    ]
    assert all(
        set(a) == {"name", "status", "value", "step", "split", "source_path", "n"}
        for a in result["arms"]
    )
    assert result["caveats"][-1] == replicates.SINGLE_RUN_CAVEAT


def test_cli_renders_replicates_and_leaves_single_runs_alone(tmp_path, capsys):
    from attestation import cli

    seeded_sweep(tmp_path, {"fam_a": [0.980, 0.983, 0.981], "fam_b": [0.970, 0.968]})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    cli._print_compare(ledger.compare(conn, "fam", metric="accuracy"))
    out = capsys.readouterr().out
    assert "mean of 3 runs, first " in out
    assert "fam_a: mean 0.9813 sd 0.0015 over 3 runs, 95% CI [" in out
    assert "fam_b: mean 0.9690 sd 0.0014 over 2 runs, range [0.9680, 0.9700], no interval" in out
    assert "top two separable: cannot tell, Cohen's d" in out

    single = get_db(":memory:")
    ledger.scan(single, ROOT / "examples" / "workspace")
    cli._print_compare(ledger.compare(single, "kdsweep", metric="wer"))
    out = capsys.readouterr().out
    assert "replicates:" not in out and "separable" not in out
    conn.close()


# --- tracker run ids (code review, 2026-11-19) --------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("clf_seed1/abc12345", ("clf", 1)),  # mlflow: <run_name>/<run id[:8]>
        ("clf_seed2/ffe12345", ("clf", 2)),
        ("train_seed3/a1b2c3d4", ("train", 3)),  # wandb: <program>/<short id>
    ],
)
def test_a_tracker_run_id_does_not_take_part_in_pooling(name, expected):
    """mlflow and wandb append a per-run id to the arm name; left in, every
    seed of `clf_seed<N>/<id>` was its own base and nothing pooled."""
    assert trackers.replicate_of(name, run_id_suffix=True) == expected


def test_only_a_reader_that_appends_a_run_id_has_it_dropped():
    """The run id is dropped because the reader SAYS it appends one
    (TrackerAdapter.run_id_suffix), not because the last segment looks like
    one; a plain result's `kd_seed1/eval` keeps its segment."""
    assert trackers.replicate_of("kd_seed1/eval") == ("kd/eval", 1)
    assert trackers.replicate_of("clf_seed1/abc12345") == ("clf/abc12345", 1)
    assert trackers.MLFLOW.run_id_suffix and trackers.WANDB.run_id_suffix
    assert not (trackers.GENERIC.run_id_suffix or trackers.SACRED.run_id_suffix)


def test_tracker_runs_named_by_seed_pool_into_one_arm():
    arms = [arm(f"clf_seed{i}/{i:08x}", 0.9 + i / 100) for i in range(1, 6)]
    out, notes = group(arms, {}, run_id_suffix=True)
    assert [a["name"] for a in out] == ["clf"]
    assert out[0]["n_runs"] == 5
    assert not notes


def test_a_pooled_arm_keeps_every_members_source_path():
    """prereg's post_hoc check needs every replicate's file, not the first."""
    arms = [arm(f"kd_seed{i}", 0.5) for i in range(1, 8)]
    out, _ = group(arms, {})
    assert len(out[0]["runs"]) == replicates.MAX_RUN_NAMES
    assert sorted(out[0]["source_paths"]) == sorted(f"/r/kd_seed{i}.json" for i in range(1, 8))


def test_the_mcp_payload_drops_the_uncapped_source_paths(tmp_path, monkeypatch):
    """`source_paths` serves prereg; a model reading runs.compare every turn
    gets the capped `runs` list instead."""
    from attestation.mcp import provenance as pv

    seeded_sweep(tmp_path, {"fam_a": [0.90, 0.91, 0.92], "fam_b": [0.80, 0.81, 0.82]})
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    full = ledger.compare(conn, "fam", metric="accuracy")["arms"][0]
    assert "source_paths" in full and full["members"] == full["runs"]
    conn.close()
    out = pv._compare(family="fam", metric="accuracy")
    assert out["ok"] is True
    assert all("source_paths" not in a and "members" not in a for a in out["arms"])


# --- schema-driven: the core record's arm and seeds (roadmap 17) ---------------


def _native(ws: Path, run_id: str, arm_name, seeds, value: float, config=None) -> None:
    """A native core record (what smeltery or the MLflow exporter writes) in
    its own run directory, read by the `core` adapter with no conversion."""
    import attestation_schema as schema

    metric = {"name": "accuracy", "value": value, "direction": None, "step": None}
    record = {
        "schema": schema.SCHEMA,
        "run_id": run_id,
        "family": "lr",
        "arm": arm_name,
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
        "config": config or {"lr": arm_name},
        "data": [],
        "metrics": [metric | {"split": None, "unit": None, "labels": {}}],
        "history": None,
        "seeds": seeds,
        "dependencies": None,
        "ext": {},
    }
    schema.write(schema.from_dict(record), ws / "p" / "runs" / run_id)


def test_native_records_pool_by_arm_and_seeds_whatever_their_names(tmp_path):
    """The run names carry no seed token at all: the record's own `arm` and
    `seeds` are what pool, so a producer that fills them needs no naming
    convention. Config keys the record names as seeds do not veto."""
    for i, v in enumerate([0.90, 0.91, 0.92]):
        _native(tmp_path, f"r{i}", "lr=0.1", {"np": i}, v, {"lr": 0.1, "np": i})
    for i, v in enumerate([0.80, 0.81, 0.82]):
        _native(tmp_path, f"s{i}", "lr=1", {"np": i}, v, {"lr": 1, "np": i})
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "lr", metric="accuracy")

    assert [(a["name"], a["n_runs"]) for a in result["arms"]] == [("lr=0.1", 3), ("lr=1", 3)]
    assert result["arms"][0]["runs"] == ["r0", "r1", "r2"]
    assert result["separable"] is True
    conn.close()


def test_records_without_seeds_or_arm_never_pool(tmp_path):
    """`seeds: null` says nothing about seeds, so two runs of one arm could be
    re-executions; and a seed with no arm names no configuration."""
    _native(tmp_path, "a0", "x", None, 0.9)
    _native(tmp_path, "a1", "x", None, 0.8)
    _native(tmp_path, "b0", None, {"seed": 1}, 0.7)
    _native(tmp_path, "b1", None, {"seed": 2}, 0.6)
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "lr", metric="accuracy")

    assert [a["name"] for a in result["arms"]] == ["a0", "a1", "b0", "b1"]
    assert not any(a.get("n_runs") for a in result["arms"])
    conn.close()


def test_a_legacy_seed_name_becomes_the_records_arm_and_seeds(tmp_path):
    """The import is the ONE place a name is read: the ledger stores the
    record's arm/seeds and pools from those columns."""
    seeded_sweep(tmp_path, {"fam_a": [0.9, 0.91]})
    from attestation.ledger_adapters import import_project

    imported = import_project(tmp_path / "proj").imported
    got = sorted((i.record.run_id, i.record.arm, i.record.seeds) for i in imported)
    assert got == [("fam_a_seed1", "fam_a", {"seed": 1}), ("fam_a_seed2", "fam_a", {"seed": 2})]
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    rows = conn.execute("SELECT arm, seeds_json FROM runs ORDER BY name").fetchall()
    assert [tuple(r) for r in rows] == [("fam_a", '{"seed": 1}'), ("fam_a", '{"seed": 2}')]
    conn.close()
