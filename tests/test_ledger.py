"""Ledger tests.

Fixtures are built here rather than copied from any real project: the adapter
reads *conventions* (`results/`, `logs/`, `configs/` holding JSON/YAML/TOML),
so the tests must exercise those conventions, not one person's directory
layout. The shapes below are the ones that recur across ML/science repos --
a metrics dict, a per-item eval dump, a nested benchmark table, a config with a
prose header.
"""

import json
from pathlib import Path

import pytest

from attestation import ledger
from attestation.db import get_db


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def workspace(tmp_path):
    """A workspace of two projects using different, equally common layouts."""
    sweep = tmp_path / "speech-model"
    write(
        sweep / "configs" / "dit_small_rope_crossattn.yaml",
        "# Phase 13: cross-attn only.\n"
        "# Single new variable vs phase 12: model.frame_cond false.\n\n"
        "data:\n  root: data/corpus\nmodel:\n  frame_cond: false\n",
    )
    write(
        sweep / "configs" / "dit_small_rope_melmask.yaml", "# Phase 12.\n\nmodel:\n  mask: 0.85\n"
    )
    # a per-item eval dump, keyed by training step in the filename
    write(
        sweep / "logs" / "eval_step_22000.json",
        json.dumps(
            [{"file": "a.wav", "wer": 1.0, "cer": 0.5}, {"file": "b.wav", "wer": 2.0, "cer": 1.5}]
        ),
    )
    write(
        sweep / "logs" / "eval_step_18000_cfg2.0.json",
        json.dumps([{"file": "a.wav", "wer": 0.4, "cer": 0.2}]),
    )

    physics = tmp_path / "physics-engine"
    write(
        physics / "examples" / "water-mp2.toml",
        '[method]\nkind = "mp2"\n\n[basis]\nname = "cc-pvdz"\n',
    )
    # a nested benchmark table: {case: {method: value}}
    write(
        physics / "results" / "benchmark_adz.json",
        json.dumps({"dimer|0.1": {"rhf": -152.06, "mp2": -152.47}, "dimer|0.2": {"rhf": -151.9}}),
    )
    write(
        physics / "results" / "benchmark_atz.json",
        json.dumps({"dimer|0.1": {"rhf": -152.11}}),
    )
    (tmp_path / "not-a-project").mkdir()
    return tmp_path


@pytest.fixture
def conn(tmp_path):
    c = get_db(tmp_path / "l.db")
    yield c
    c.close()


def test_scan_discovers_projects_without_knowing_their_names(conn, workspace):
    """No project is registered anywhere; every subdirectory is tried."""
    out = ledger.scan(conn, workspace)

    assert out["scanned"] == {"speech-model": 4, "physics-engine": 3}


def test_a_directory_with_no_runs_is_reported_not_silently_skipped(conn, workspace):
    """Silent success for work not done is the failure this codebase keeps
    hitting: "found nothing" must never look like "nothing was there"."""
    out = ledger.scan(conn, workspace)

    assert "not-a-project" in out["empty"]


def test_config_header_is_kept_verbatim(conn, workspace):
    """The prose header usually states the hypothesis and the variable changed.
    Interpreting it is a judgement this layer does not make."""
    ledger.scan(conn, workspace)

    run = ledger.detail(conn, "speech-model", "dit_small_rope_crossattn")

    assert "Single new variable" in run["notes"]
    assert run["status"] == "spec"
    assert run["metrics"] == [], "a config specifies a run; it carries no result"


def test_per_item_eval_dump_is_averaged_with_its_step(conn, workspace):
    ledger.scan(conn, workspace)

    run = ledger.detail(conn, "speech-model", "eval_step_22000")
    by_name = {m["metric"]: m for m in run["metrics"]}

    assert by_name["wer"]["value"] == pytest.approx(1.5)  # mean of 1.0 and 2.0
    assert by_name["wer"]["step"] == 22000
    assert by_name["n_records"]["value"] == 2


def test_variant_token_in_a_filename_becomes_a_split(conn, workspace):
    ledger.scan(conn, workspace)

    run = ledger.detail(conn, "speech-model", "eval_step_18000_cfg2.0")

    assert all(m["split"] == "cfg2.0" for m in run["metrics"])
    assert {m["step"] for m in run["metrics"]} == {18000}


def test_nested_benchmark_table_keeps_the_case_key(conn, workspace):
    """{case: {method: value}} -- the outer key must survive so a number can be
    traced back to what produced it."""
    ledger.scan(conn, workspace)

    run = ledger.detail(conn, "physics-engine", "benchmark_adz")
    rhf = [m for m in run["metrics"] if m["metric"] == "rhf"]

    assert {m["split"] for m in rhf} == {"dimer|0.1", "dimer|0.2"}
    assert any(m["value"] == pytest.approx(-152.06) for m in rhf)


def test_compare_ranks_lower_is_better_correctly(conn, workspace):
    """Ranking WER ascending is the whole point; descending would name the
    worst arm the winner."""
    ledger.scan(conn, workspace)

    result = ledger.compare(conn, "eval", metric="wer")
    values = [a["value"] for a in result["arms"] if a["value"] is not None]

    assert result["direction"] == "lower_is_better"
    assert values == sorted(values)
    assert result["winner"] == result["arms"][0]["name"]


def test_compare_refuses_a_metric_with_unknown_direction(conn, workspace):
    """Total energies are not comparable across systems -- a bigger molecule
    always has a lower energy. Guessing would rank confidently and wrongly."""
    ledger.scan(conn, workspace)

    with pytest.raises(ValueError, match="unknown direction"):
        ledger.compare(conn, "benchmark", metric="rhf")


def test_compare_lists_arms_that_lack_the_metric(conn, workspace):
    """A config arm with no eval attached is a finding, not a row to drop."""
    ledger.scan(conn, workspace)

    result = ledger.compare(conn, "dit-small-rope", metric="wer")

    assert result["winner"] is None
    assert sorted(result["without_metric"]) == [
        "dit_small_rope_crossattn",
        "dit_small_rope_melmask",
    ]


def test_compare_uses_best_step_not_last(conn, tmp_path):
    """A run that diverges late should be judged by its best checkpoint, not by
    wherever it happened to stop."""
    project = tmp_path / "p"
    write(project / "results" / "run_a_step_100.json", json.dumps({"wer": 0.5}))
    write(project / "results" / "run_a_step_200.json", json.dumps({"wer": 0.9}))
    conn2 = get_db(tmp_path / "x.db")
    ledger.scan(conn2, tmp_path)

    result = ledger.compare(conn2, "run-a", metric="wer")

    assert result["arms"][0]["value"] == pytest.approx(0.5)
    conn2.close()


def test_scan_is_idempotent(conn, workspace):
    ledger.scan(conn, workspace)
    first = ledger.list_runs(conn, limit=100)
    ledger.scan(conn, workspace)
    second = ledger.list_runs(conn, limit=100)

    assert [(r["project"], r["name"]) for r in first] == [(r["project"], r["name"]) for r in second]


def test_scan_reports_a_missing_root_rather_than_raising(conn, tmp_path):
    out = ledger.scan(conn, tmp_path / "nope")

    assert out["scanned"] == {}
    assert "no such directory" in out["message"]


def test_unparseable_json_yields_no_run(conn, tmp_path):
    """An unrecognised shape must produce no run rather than a wrong one."""
    write(tmp_path / "p" / "results" / "broken.json", "{not json")
    write(tmp_path / "p" / "results" / "strings.json", json.dumps({"note": "no numbers here"}))

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"] == {}
    assert "p" in out["empty"]


def test_compare_carries_provenance_for_every_arm(conn, workspace):
    """An auditor's first question is "from which file?". A ranking whose rows
    cannot be traced back to an artifact cannot be checked."""
    ledger.scan(conn, workspace)

    result = ledger.compare(conn, "eval", metric="wer")

    assert all(a["source_path"] for a in result["arms"])
    assert all(Path(a["source_path"]).exists() for a in result["arms"])


def test_compare_warns_when_every_arm_is_a_small_sample(conn, tmp_path):
    """n=1 and n=2 differences are noise. Printing four decimal places and
    naming a winner implies a confidence the numbers have not earned."""
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps([{"wer": 0.4}, {"wer": 0.5}]))
    write(project / "results" / "arm_b.json", json.dumps([{"wer": 0.9}]))
    c = get_db(tmp_path / "s.db")
    ledger.scan(c, tmp_path)

    result = ledger.compare(c, "arm", metric="wer")

    assert result["winner"] == "arm_a"
    assert any("likely noise" in w for w in result["caveats"])
    assert any("different sample sizes" in w for w in result["caveats"])
    c.close()


def test_compare_warns_when_the_top_two_are_nearly_tied(conn, tmp_path):
    project = tmp_path / "p"
    rows = [{"wer": 0.500} for _ in range(40)]
    write(project / "results" / "arm_a.json", json.dumps(rows))
    write(project / "results" / "arm_b.json", json.dumps([{"wer": 0.501} for _ in range(40)]))
    c = get_db(tmp_path / "t2.db")
    ledger.scan(c, tmp_path)

    result = ledger.compare(c, "arm", metric="wer")

    assert any("too close to call" in w for w in result["caveats"])
    c.close()


def test_a_healthy_comparison_still_flags_no_seed_replication(conn, tmp_path):
    """The warnings must be earned, not boilerplate -- a tool that always warns
    trains the reader to ignore it. But arm_a/arm_b here really are one run
    each: the schema records no seed replication (seed lives in `config`, not
    as a second run of the same configuration), so even a well-powered,
    clearly-separated comparison cannot rule out run-to-run variance and must
    say so -- that is the one caveat that is earned here, not boilerplate."""
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps([{"wer": 0.20} for _ in range(50)]))
    write(project / "results" / "arm_b.json", json.dumps([{"wer": 0.80} for _ in range(50)]))
    c = get_db(tmp_path / "t3.db")
    ledger.scan(c, tmp_path)

    result = ledger.compare(c, "arm", metric="wer")

    assert result["winner"] == "arm_a"
    assert result["caveats"] == [
        "each arm is a single run; no seed replication, so this ranking cannot"
        " separate configuration from run-to-run variance"
    ]
    c.close()


def test_a_lookup_table_is_not_read_as_metrics(conn, tmp_path):
    """Regression: a tokenizer vocab.json is a dict of 50,258 numeric values.
    Read as a metrics record it produced 50,258 "metrics", one of which was
    keyed `wer` and ranked as a WER of 1554. A shape we cannot interpret must
    yield no run, not a wrong one."""
    project = tmp_path / "p"
    vocab = {chr(33 + i % 90) + str(i): i for i in range(200)}
    vocab["wer"] = 1554
    write(project / "results" / "vocab.json", json.dumps(vocab))
    write(project / "results" / "eval.json", json.dumps({"wer": 0.3, "cer": 0.1}))

    ledger.scan(conn, tmp_path)

    names = {r["name"] for r in ledger.list_runs(conn, limit=50)}
    assert "vocab" not in names, "a 200-key numeric map is a lookup table"
    assert "eval" in names, "a small metrics record is still read"


def test_a_results_csv_becomes_one_run_per_row(conn, tmp_path):
    """CSV is as common as JSON for recorded results, and was missing at first:
    a 48-config sweep sat invisible in results.csv while the ledger reported the
    project had no results at all. Each labelled row is its own arm -- collapsing
    them to one mean makes "which config won" unanswerable."""
    project = tmp_path / "p"
    write(
        project / "results" / "sweep.csv",
        "config_name,axis,wer,latency_ms\n"
        "baseline,none,0.0433,19616.1\n"
        "stack_4,layer_drop,0.0527,17000.0\n"
        "stack_5,layer_drop,0.3618,16000.0\n",
    )

    ledger.scan(conn, tmp_path)

    names = {r["name"] for r in ledger.list_runs(conn, limit=50)}
    assert names == {"sweep/baseline", "sweep/stack_4", "sweep/stack_5"}

    result = ledger.compare(conn, "sweep", metric="wer")
    assert result["winner"] == "sweep/baseline"
    assert [a["value"] for a in result["arms"]] == pytest.approx([0.0433, 0.0527, 0.3618])


def test_a_csv_with_no_label_column_is_skipped(conn, tmp_path):
    """Without a column naming each row, there is no arm identity to record --
    and inventing one would fabricate structure the file does not have."""
    write(tmp_path / "p" / "results" / "raw.csv", "x,y\n1,2\n3,4\n")

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"] == {}
    assert "p" in out["empty"]


def test_nested_results_are_found_not_just_top_level(conn, tmp_path):
    """Research JSON nests. A real file put its headline correlation at
    summary.variants.raw.rho_pooled while bookkeeping counts sat at the top;
    flattening one level captured n_molecules and missed every actual result.
    The path is kept as `split` so a value stays traceable to where it sat."""
    write(
        tmp_path / "p" / "results" / "study.json",
        json.dumps(
            {
                "n_molecules": 489,
                "summary": {"variants": {"raw": {"rho_pooled": 0.7189}}},
            }
        ),
    )

    ledger.scan(conn, tmp_path)
    run = ledger.detail(conn, "p", "study")
    by_name = {m["metric"]: m for m in run["metrics"]}

    assert by_name["rho_pooled"]["value"] == pytest.approx(0.7189)
    assert by_name["rho_pooled"]["split"] == "summary.variants.raw"
    assert "n_molecules" in by_name, "top-level scalars are still recorded"


def test_nesting_is_bounded(conn, tmp_path):
    """A pathological document must not yield unbounded metrics."""
    node = {"value": 1.0}
    for _ in range(12):
        node = {"deeper": node}
    write(tmp_path / "p" / "results" / "deep.json", json.dumps(node))

    ledger.scan(conn, tmp_path)

    run = ledger.detail(conn, "p", "deep")
    assert run is None or len(run["metrics"]) == 0


def test_compare_resolves_a_prefixed_metric_via_its_stem(conn, tmp_path):
    """The generic adapter extracts metric names verbatim from artifacts,
    where they are overwhelmingly prefixed (`test_accuracy`, `train_loss`)
    rather than bare. Stripping a known affix and looking up the declared
    direction for the stem is still a declaration -- not a guess."""
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps({"test_accuracy": 0.90}))
    write(project / "results" / "arm_b.json", json.dumps({"test_accuracy": 0.94}))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "arm", metric="test_accuracy")

    assert result["direction"] == "higher_is_better"
    assert result["winner"] == "arm_b"


def test_compare_val_loss_still_resolves_identically(conn, tmp_path):
    """val_loss was already an explicit entry in METRIC_DIRECTION; stem
    matching must not change its behaviour."""
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps({"val_loss": 0.5}))
    write(project / "results" / "arm_b.json", json.dumps({"val_loss": 0.2}))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "arm", metric="val_loss")

    assert result["direction"] == "lower_is_better"
    assert result["winner"] == "arm_b"


def test_compare_still_refuses_a_genuinely_undirectional_stem(conn, workspace):
    """`rhf` carries no split/phase affix and is not declared anywhere --
    stem-matching must not invent a direction for it."""
    ledger.scan(conn, workspace)

    with pytest.raises(ValueError, match="unknown direction"):
        ledger.compare(conn, "benchmark", metric="rhf")


def test_metric_direction_error_names_the_override_file_not_a_python_dict(conn, workspace):
    ledger.scan(conn, workspace)

    with pytest.raises(ValueError, match="metric_direction.toml"):
        ledger.compare(conn, "benchmark", metric="rhf")


def test_compare_honors_a_toml_supplied_metric_direction(conn, tmp_path, monkeypatch):
    """The escape hatch for a metric name with no known affix: a user-owned
    TOML file, not editing installed package source."""
    override = tmp_path / "metric_direction.toml"
    override.write_text('[metric_direction]\nrho_pooled = "higher_is_better"\n')
    monkeypatch.setenv("LEDGER_METRIC_DIRECTION_FILE", str(override))

    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps({"rho_pooled": 0.5}))
    write(project / "results" / "arm_b.json", json.dumps({"rho_pooled": 0.9}))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "arm", metric="rho_pooled")

    assert result["direction"] == "higher_is_better"
    assert result["winner"] == "arm_b"


def test_caveats_flag_the_winner_specifically_when_it_is_the_small_arm(conn, tmp_path):
    """The maximum across arms gating this caveat let a winner measured on
    n=3 rank ahead of a runner-up measured on n=10000 without a word about the
    winner's own sample size -- only a range, unattributed."""
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps([{"wer": 0.05} for _ in range(3)]))
    write(project / "results" / "arm_b.json", json.dumps([{"wer": 0.20} for _ in range(10000)]))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "arm", metric="wer")

    assert result["winner"] == "arm_a"
    assert any("winning arm's value rests on n=3" in c for c in result["caveats"]), result[
        "caveats"
    ]


def test_caveats_flag_no_seed_replication_for_single_run_arms(conn, tmp_path):
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps([{"wer": 0.0500} for _ in range(5000)]))
    write(project / "results" / "arm_b.json", json.dumps([{"wer": 0.0530} for _ in range(5000)]))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "arm", metric="wer")

    assert result["winner"] == "arm_a"
    assert any("no seed replication" in c for c in result["caveats"])


# --- Diagnosing an empty scan -------------------------------------------------
# "0 run(s)" with no reason is this tool's worst failure mode: adoption cost is
# the stated design constraint, and a researcher whose ordinary layout is not
# recognised sees a successful-looking run that found nothing. Each test below
# is a layout that legitimately yields no runs; the scan must say WHY.


def test_empty_scan_names_the_unrecognised_result_dir(conn, tmp_path):
    """Results in a per-arm directory rather than results/ -- an ordinary layout."""
    project = tmp_path / "asr"
    write(project / "baseline" / "results.json", json.dumps({"wer": 0.05}))
    write(project / "bigger" / "results.json", json.dumps({"wer": 0.04}))

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"] == {}
    why = out["diagnostics"]["asr"]
    # names both where it looked and where the files actually are
    assert "results/" in why, why
    assert "baseline" in why, why


def test_empty_scan_reports_a_project_with_no_files_at_all(conn, tmp_path):
    (tmp_path / "blank").mkdir()

    out = ledger.scan(conn, tmp_path)

    assert "no files" in out["diagnostics"]["blank"].lower()


def test_empty_scan_reports_config_only_project(conn, tmp_path):
    """A spec with no result attached is a real state, not an error -- say so."""
    project = tmp_path / "specs"
    write(project / "configs" / "arm_a.json", json.dumps({"lr": 0.001}))

    out = ledger.scan(conn, tmp_path)

    why = out["diagnostics"]["specs"]
    assert "config" in why.lower(), why


def test_successful_scan_reports_no_diagnostics(conn, tmp_path):
    project = tmp_path / "p"
    write(project / "results" / "arm_a.json", json.dumps({"wer": 0.05}))

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"] == {"p": 1}
    assert out["diagnostics"] == {}


def test_compare_on_a_project_name_suggests_the_real_families(conn, tmp_path):
    """`compare <project>` is the intuitive first guess and finds nothing:
    families are derived from filename prefixes, not from the project. Dead-
    ending there with no hint is the same silent failure as an unexplained
    empty scan."""
    project = tmp_path / "asr"
    for arm in ("baseline", "bigger-lm", "more-data"):
        write(project / "results" / f"{arm}.json", json.dumps({"wer": 0.05}))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "asr", metric="wer")

    assert result["winner"] is None
    assert "bigger" in result["available_families"]
    assert "asr" in result["message"]


def test_compare_warns_when_a_family_has_only_one_arm(conn, tmp_path):
    """A one-arm family always 'wins'. Saying so without a caveat implies a
    comparison happened."""
    project = tmp_path / "p"
    write(project / "results" / "solo_a.json", json.dumps({"wer": 0.05}))
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "solo", metric="wer")

    assert result["winner"] == "solo_a"
    assert any("only one arm" in c for c in result["caveats"]), result["caveats"]


# --- A training log is not an eval dump ---------------------------------------


def test_jsonl_training_log_keeps_per_step_values(conn, tmp_path):
    """Rows carrying distinct `step`s are one run measured over time, not N
    independent samples. Averaging a descending loss curve reports a number
    the run never had -- and `_best_step` exists precisely to pick across
    steps, so the steps must survive ingestion."""
    project = tmp_path / "p"
    write(
        project / "results" / "run_a.jsonl",
        "\n".join(
            json.dumps(row)
            for row in (
                {"step": 1000, "val_loss": 2.55},
                {"step": 2000, "val_loss": 1.99},
                {"step": 3000, "val_loss": 1.71},
            )
        ),
    )
    ledger.scan(conn, tmp_path)

    rows = list(
        conn.execute("SELECT value, step FROM run_metrics WHERE metric = 'val_loss' ORDER BY step")
    )

    assert [r["step"] for r in rows] == [1000, 2000, 3000]
    assert [round(r["value"], 2) for r in rows] == [2.55, 1.99, 1.71]


def test_jsonl_training_log_is_not_counted_as_samples(conn, tmp_path):
    """Three checkpoints of one run are not n=3 samples; claiming so triggers
    a bogus 'differences at this size are likely noise' caveat."""
    project = tmp_path / "p"
    for arm, losses in (("run_a", (2.55, 1.99, 1.71)), ("run_b", (2.50, 1.90, 1.62))):
        write(
            project / "results" / f"{arm}.jsonl",
            "\n".join(
                json.dumps({"step": s, "val_loss": v})
                for s, v in zip((1000, 2000, 3000), losses, strict=True)
            ),
        )
    ledger.scan(conn, tmp_path)

    result = ledger.compare(conn, "run", metric="val_loss")

    assert result["winner"] == "run_b"
    # best-across-steps, not the mean of the curve
    assert round(result["arms"][0]["value"], 2) == 1.62
    assert not any("likely noise" in c for c in result["caveats"]), result["caveats"]


def test_per_sample_eval_dump_still_aggregates(conn, tmp_path):
    """The existing behaviour must survive: rows with no step are independent
    samples, and their mean plus n_records is exactly right."""
    project = tmp_path / "p"
    write(
        project / "results" / "eval_a.json",
        json.dumps([{"wer": 0.10}, {"wer": 0.20}, {"wer": 0.30}]),
    )
    ledger.scan(conn, tmp_path)

    rows = dict(
        (r["metric"], r["value"]) for r in conn.execute("SELECT metric, value FROM run_metrics")
    )

    assert round(rows["wer"], 4) == 0.2000
    assert rows["n_records"] == 3.0


def test_jsonl_survives_a_truncated_final_line(conn, tmp_path):
    """JSONL is line-delimited so it survives a partial write: a run killed
    mid-flush leaves a truncated last line, which is the commonest way these
    files end. Discarding every valid row because of it is silent data loss."""
    project = tmp_path / "p"
    write(
        project / "results" / "run_a.jsonl",
        '{"step": 1000, "val_loss": 2.5}\n{"step": 2000, "val_loss": 1.9}\n{"step": 3000, "val_l',
    )
    ledger.scan(conn, tmp_path)

    rows = list(
        conn.execute("SELECT value, step FROM run_metrics WHERE metric = 'val_loss' ORDER BY step")
    )

    assert [r["step"] for r in rows] == [1000, 2000]


def test_labelless_csv_diagnostic_names_the_missing_column(conn, tmp_path):
    """Refusing a CSV with no arm-identity column is correct -- there is nothing
    to name the runs after -- but the generic 'no metrics record' message
    misdescribes a file whose numeric columns are fine."""
    project = tmp_path / "p"
    write(project / "results" / "grid.csv", "wer,cer\n0.05,0.02\n0.04,0.01\n")

    out = ledger.scan(conn, tmp_path)

    why = out["diagnostics"]["p"]
    assert "config_name" in why, why


def test_nan_in_one_file_does_not_abort_the_scan(conn, tmp_path):
    """A degenerate statistic must not cost the other 33 result files.

    A t-test between two identical arms is NaN, and json.dump writes it as a
    bare `NaN` token that json.loads accepts. Real result sets contain these
    routinely. Before this, `statistics.pstdev` raised on the NaN and the
    traceback escaped `scan()`, so one such file took down every project in
    the workspace and the CLI exited non-zero having recorded nothing.
    """
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "degenerate.json").write_text(
        '[{"arm": "a", "t_stat": NaN, "acc": 0.9}, {"arm": "a", "t_stat": NaN, "acc": 0.7}]'
    )
    (results / "healthy.json").write_text(json.dumps({"acc": 0.81}))

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"].get("proj"), f"scan recorded nothing: {out}"
    names = {r["name"] for r in ledger.list_runs(conn, project="proj")}
    assert "healthy" in names, f"a NaN elsewhere hid an unrelated file: {names}"


def test_nan_metric_is_dropped_not_stored_as_nan(conn, tmp_path):
    """NaN must not reach the DB: it compares false to everything, so a NaN
    silently loses every ranking it appears in rather than being reported."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "r.json").write_text('{"acc": 0.9, "p_value": NaN}')

    ledger.scan(conn, tmp_path)

    rows = conn.execute("SELECT metric, value FROM run_metrics").fetchall()
    stored = {r["metric"]: r["value"] for r in rows}
    assert "acc" in stored
    assert all(v == v for v in stored.values()), f"NaN stored: {stored}"
