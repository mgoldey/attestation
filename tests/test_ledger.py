"""Ledger tests.

Fixtures are built here rather than copied from any real project: the adapter
reads *conventions* (`results/`, `logs/`, `configs/` holding JSON/YAML/TOML),
so the tests must exercise those conventions, not one person's directory
layout. The shapes below are the ones that recur across ML/science repos --
a metrics dict, a per-item eval dump, a nested benchmark table, a config with a
prose header.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from attestation import corpus, ledger
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


def test_config_ladder_prefers_env_then_workspace_then_home(tmp_path, monkeypatch):
    """`_config_ladder`'s one defining property, unguarded until now: env
    beats workspace beats `~/.hermes`. P3's round-two review ran a mutant
    that swapped the env-var branch below the workspace branch and it
    survived -- 156 targeted tests plus a repo-wide keyword sweep, all
    green -- because no test drove the function directly with all three
    rungs present at once. DB-free: `_config_ladder` takes no connection.
    """
    env_var = "LEDGER_TEST_LADDER_FILE"
    filename = "ladder-test.toml"
    monkeypatch.delenv(env_var, raising=False)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_file = workspace / filename
    workspace_file.write_text("# workspace copy\n")

    env_file = tmp_path / "env-copy.toml"
    env_file.write_text("# env copy\n")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(ledger.Path, "home", classmethod(lambda cls: fake_home))

    # Rung 1: env set, workspace file also present -- env must win.
    monkeypatch.setenv(env_var, str(env_file))
    assert ledger._config_ladder(env_var, filename, workspace) == env_file

    # Rung 2: env unset, workspace file present -- workspace must win.
    monkeypatch.delenv(env_var, raising=False)
    assert ledger._config_ladder(env_var, filename, workspace) == workspace_file

    # Rung 3: env unset, workspace file absent (or no workspace at all) --
    # falls back to ~/.hermes/<filename>.
    empty_workspace = tmp_path / "empty-workspace"
    empty_workspace.mkdir()
    assert (
        ledger._config_ladder(env_var, filename, empty_workspace)
        == fake_home / ".hermes" / filename
    )
    assert ledger._config_ladder(env_var, filename, None) == fake_home / ".hermes" / filename


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


def test_scan_reads_results_when_root_is_the_project_itself(conn, tmp_path):
    """Pointing --root at a repo must work, not just at a directory of repos.

    `scan()` treats every subdirectory of root as a project, so aiming it at a
    single repo makes `results/` its own project root and the adapter then
    looks for `results/results/`. That is the layout of essentially every ML
    repo, and it reported "0 runs" with a diagnostic blaming the user's
    directory names.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "run_a.json").write_text(json.dumps({"acc": 0.9}))

    out = ledger.scan(conn, tmp_path)

    assert out["scanned"], f"a repo-shaped root scanned nothing: {out}"
    names = {r["name"] for r in ledger.list_runs(conn)}
    assert "run_a" in names, names


def test_a_dir_consumed_by_the_root_scan_is_not_also_reported_empty(conn, tmp_path):
    """`results/` must not be listed as an empty project once the root read it.

    Scanning a repo root reads `results/*.json` as runs, while the per-project
    pass also visits `results/` and finds nothing under `results/results/`.
    Reporting the same directory as both a success and a failure tells the
    reader to go fix a layout that just worked.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "run_a.json").write_text(json.dumps({"acc": 0.9}))

    out = ledger.scan(conn, tmp_path)

    assert "results" not in out["empty"], out["empty"]
    assert "results" not in out["diagnostics"], out["diagnostics"]


@pytest.mark.parametrize(
    "metric",
    [
        "best_val_loss",
        "final_val_loss",
        "final_train_loss",
        "best_val_ppl",
        "val_perplexity",
        "eval_nll",
    ],
)
def test_standard_lm_metric_names_have_a_direction(metric):
    """The names a language-model training loop actually writes must resolve.

    `best_` and `final_` qualify a metric without changing which way is
    better, and they stack with a split prefix (`best_val_loss`). Stripping
    only one affix left the most standard names in the field undeclared, so a
    real repo's runs were found and then refused for comparison. Perplexity is
    exp(loss): lower is better by definition, not by guess.
    """
    assert ledger._metric_direction(metric, ledger.metric_directions()) == "lower_is_better"


def test_an_undeclared_metric_still_refuses():
    """Stripping more affixes must not turn the refusal into a guess."""
    directions = ledger.metric_directions()
    assert ledger._metric_direction("best_val_novel_score", directions) is None
    assert ledger._metric_direction("final_throughput", directions) is None


def test_corpus_detected_from_a_dataset_load_call(tmp_path):
    """The corpus identity is in the source, and AST reads it exactly.

    In ~/qc/scmoe not one of 34 result files records vocab_size or seq_len;
    the corpus exists only as `load_dataset("Salesforce/wikitext",
    "wikitext-2-raw-v1")` plus `get_encoding("gpt2")`. Those are literal
    arguments, so a parser recovers them exactly -- no inference required.
    """
    src = tmp_path / "data.py"
    src.write_text(
        "import tiktoken\n"
        "from datasets import load_dataset\n"
        "def load():\n"
        "    enc = tiktoken.get_encoding('gpt2')\n"
        "    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')\n"
        "    return TokenDataset(ds, seq_len=256)\n"
    )
    found = corpus.detect_in_source(src)
    assert found is not None
    assert found.source == "Salesforce/wikitext"
    assert found.config == "wikitext-2-raw-v1"
    assert found.tokenizer == "gpt2"


def test_corpus_detection_is_identical_across_repeated_reads():
    """Identity is the join key: two runs share a corpus only if the strings
    match exactly. A detector that returns 'WikiText-2' once and
    'WikiText-2 data loading and tokenization' the next time makes the guard
    silently fail, which is worse than not having it."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "data.py"
        src.write_text("from datasets import load_dataset\nds = load_dataset('a/b', 'c')\n")
        seen = {corpus.detect_in_source(src) for _ in range(5)}
    assert len(seen) == 1, seen


def test_corpus_detection_reports_nothing_rather_than_guessing(tmp_path):
    """A file that only *calls* a loader states no corpus. Returning None is
    correct; inventing a tokenizer from a function name is the failure this
    avoids."""
    src = tmp_path / "driver.py"
    src.write_text(
        "from scmoe.lm.data import load_wikitext2\n"
        "def main():\n    return load_wikitext2(seq_len=256, batch_size=8)\n"
    )
    found = corpus.detect_in_source(src)
    assert found is None or (found.source is None and found.tokenizer is None), found


def test_corpus_detection_survives_unparseable_source(tmp_path):
    """A syntax error in one file must not abort a scan."""
    src = tmp_path / "broken.py"
    src.write_text("def f(:\n  pass\n")
    assert corpus.detect_in_source(src) is None


def test_a_lookup_table_of_entities_yields_no_run(conn, tmp_path):
    """A name->value table is data, not a metrics record.

    ferric/benchmarks/gmtkn30/aconf_pyscf_energies.json maps 18 molecular
    conformers to computed energies. The ledger read each conformer name as a
    metric, so `H_ttt` and `P_gg` became rankable quantities and the family
    was polluted with 18 nonsense columns. MAX_METRIC_KEYS did not catch it:
    18 keys is well under the vocabulary threshold.

    The signature is that the keys name *entities* rather than *quantities* --
    no key contains a word any metric vocabulary would use.
    """
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "aconf_energies.json").write_text(
        json.dumps(
            {
                "B_G": -157.907,
                "B_T": -157.908,
                "H_g+t+g-": -236.272,
                "H_ggg": -236.272,
                "H_gtg": -236.272,
                "H_gtt": -236.273,
                "H_tgg": -236.273,
                "H_ttt": -236.274,
                "P_gg": -197.089,
            }
        )
    )
    ledger.scan(conn, tmp_path)

    metrics = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert not metrics & {"h_ttt", "p_gg", "b_g"}, f"conformers read as metrics: {metrics}"


def test_a_real_metrics_record_is_still_read(conn, tmp_path):
    """The guard must not cost genuine records. This is the shape it has to
    keep working on -- heterogeneous named quantities, which is what
    distinguishes it from a table of one quantity across many entities."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "run_a.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "n_params": 88815104,
                "n_layers": 6,
                "d_model": 512,
                "epochs": 10,
                "final_val_loss": 5.0968,
                "best_val_loss": 5.0946,
            }
        )
    )
    ledger.scan(conn, tmp_path)

    metrics = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert "best_val_loss" in metrics, metrics
    assert "n_params" in metrics, metrics


def test_a_short_entity_table_is_not_refused(conn, tmp_path):
    """Three keys is too few to infer anything. The guard applies only where
    there is enough evidence, so a small record with unusual names survives."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "small.json").write_text(json.dumps({"alpha": 0.1, "beta": 0.2}))
    ledger.scan(conn, tmp_path)

    metrics = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert metrics == {"alpha", "beta"}, metrics


def test_corpus_read_from_artifact_fields(conn, tmp_path):
    """Corpus fields already present in a result file become a corpus row."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "run_a.json").write_text(
        json.dumps({"dataset": "wikitext-2", "tokenizer": "gpt2", "acc": 0.9})
    )
    ledger.scan(conn, tmp_path)

    row = conn.execute(
        "SELECT c.name, c.tokenizer FROM runs r JOIN corpora c ON c.id = r.corpus_id"
    ).fetchone()
    assert row is not None, "no corpus linked to the run"
    assert row["name"] == "wikitext-2"
    assert row["tokenizer"] == "gpt2"
    # `dataset` names the corpus; it must not also be ranked as a measurement.
    metrics = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert "dataset" not in metrics and "tokenizer" not in metrics, metrics


def test_manifest_assigns_a_corpus_to_a_family(conn, tmp_path, monkeypatch):
    """A declaration is the only source that can state intent: "these arms
    were meant to share a corpus". It outranks what the artifacts happened
    to record."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    for tag in ("a", "b"):
        (results / f"lm_{tag}.json").write_text(json.dumps({"best_val_loss": 5.0}))
    manifest = tmp_path / "corpora.toml"
    manifest.write_text(
        "[corpus.wikitext2]\n"
        'source = "Salesforce/wikitext"\n'
        'tokenizer = "gpt2"\n'
        "seq_len = 256\n"
        "\n[assign]\n"
        'family.lm = "wikitext2"\n'
    )
    monkeypatch.setenv("LEDGER_CORPUS_FILE", str(manifest))
    ledger.scan(conn, tmp_path)

    rows = conn.execute(
        "SELECT c.name, c.source, c.seq_len FROM runs r JOIN corpora c ON c.id = r.corpus_id"
    ).fetchall()
    assert len(rows) == 2, rows
    assert {r["name"] for r in rows} == {"wikitext2"}
    assert rows[0]["source"] == "Salesforce/wikitext"
    assert rows[0]["seq_len"] == 256


def test_compare_reports_the_shared_corpus(conn, tmp_path, monkeypatch):
    """When arms agree, say so: the reader learns the comparison was checked
    rather than assumed."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    for tag, loss in (("a", 5.0), ("b", 5.2)):
        (results / f"lm_{tag}.json").write_text(
            json.dumps({"dataset": "wikitext-2", "best_val_loss": loss})
        )
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "lm")

    assert out.get("corpus") == "wikitext-2", out
    assert not any("corpus" in c and "differ" in c for c in out["caveats"]), out["caveats"]


def test_compare_caveats_when_arms_used_different_corpora(conn, tmp_path):
    """Loss is only comparable across runs that saw the same data. Ranking
    arms trained on different corpora is the error this feature exists for."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "lm_a.json").write_text(json.dumps({"dataset": "wikitext-2", "best_val_loss": 5.0}))
    (results / "lm_b.json").write_text(
        json.dumps({"dataset": "wikitext-103", "best_val_loss": 4.1})
    )
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "lm")

    joined = " ".join(out["caveats"])
    assert "corpus" in joined.lower(), out["caveats"]
    assert "wikitext-103" in joined, out["caveats"]


def test_no_corpus_caveat_when_no_arm_records_one(conn, tmp_path):
    """A caveat must be earned. When no arm names a corpus there is no
    inconsistency -- it is simply a ledger without corpus data, and warning on
    every comparison would make the caveats boilerplate, which is what trains
    a reader to ignore them."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    for tag, loss in (("a", 5.0), ("b", 5.2)):
        (results / f"lm_{tag}.json").write_text(json.dumps({"best_val_loss": loss}))
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "lm")

    assert not any("corpus" in c.lower() for c in out["caveats"]), out["caveats"]


def test_compare_caveats_when_only_some_arms_record_a_corpus(conn, tmp_path):
    """A *partial* record is the finding: one arm's data is known and the
    other's is not, so the ledger cannot confirm they match."""
    results = tmp_path / "proj" / "results"
    results.mkdir(parents=True)
    (results / "lm_a.json").write_text(json.dumps({"dataset": "wikitext-2", "best_val_loss": 5.0}))
    (results / "lm_b.json").write_text(json.dumps({"best_val_loss": 5.2}))
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "lm")

    joined = " ".join(out["caveats"]).lower()
    assert "corpus" in joined and "unverified" in joined, out["caveats"]
    assert "lm_b" in joined, out["caveats"]


def test_a_scan_that_fails_partway_leaves_no_torn_write(workspace, tmp_path, monkeypatch):
    """scan() must be all-or-nothing across every project it touches.

    The transaction is real and load-bearing: _link_corpora() creates corpora
    rows via corpus.upsert(), _replace_project() then writes runs whose
    corpus_id is a foreign key to a row from that same uncommitted transaction,
    and the single commit lands after the loop over ALL projects.

    Nothing proved that until now. An earlier design for this refactor proposed
    "repositories open and close per method call", which would have split those
    two writes across two connections and two transactions -- producing exactly
    the torn state this test forbids: runs pointing at corpora that were rolled
    back, or corpora orphaned by a failed run insert.

    Failure is injected in the SECOND project so the first has already written
    rows that must not survive.
    """
    conn = get_db(tmp_path / "t.db")
    projects = sorted(p.name for p in workspace.iterdir() if p.is_dir())
    assert len(projects) >= 2, "this test needs a multi-project workspace"

    real_replace = ledger._replace_project
    calls = []

    def exploding_replace(conn_, project, records, scanned_at):
        calls.append(project)
        if len(calls) > 1:
            raise RuntimeError("disk full partway through the scan")
        return real_replace(conn_, project, records, scanned_at)

    monkeypatch.setattr(ledger, "_replace_project", exploding_replace)

    with pytest.raises(RuntimeError, match="disk full"):
        ledger.scan(conn, workspace)

    conn.rollback()  # what a caller that caught the error would do

    assert conn.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"] == 0, (
        "the first project's runs survived a scan that failed on the second"
    )
    assert conn.execute("SELECT COUNT(*) n FROM run_metrics").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM corpora").fetchone()["n"] == 0, (
        "corpora created by _link_corpora outlived the runs that referenced them"
    )

    dangling = conn.execute(
        "SELECT COUNT(*) n FROM runs r"
        " WHERE r.corpus_id IS NOT NULL"
        "   AND NOT EXISTS (SELECT 1 FROM corpora c WHERE c.id = r.corpus_id)"
    ).fetchone()["n"]
    assert dangling == 0, "a run references a corpus row that does not exist"
    conn.close()


def test_a_successful_scan_leaves_no_dangling_corpus_reference(workspace, tmp_path):
    """The invariant the failure test checks, asserted on the happy path too --
    otherwise the test above would pass on a database that is simply empty."""
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, workspace)

    assert conn.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"] > 0
    dangling = conn.execute(
        "SELECT COUNT(*) n FROM runs r"
        " WHERE r.corpus_id IS NOT NULL"
        "   AND NOT EXISTS (SELECT 1 FROM corpora c WHERE c.id = r.corpus_id)"
    ).fetchone()["n"]
    assert dangling == 0
    conn.close()


def test_comparing_never_ranks_one_arm_on_train_against_another_on_test(tmp_path):
    """The failure this module's docstring says it exists to prevent.

    `_best_step` picks a run's best value across steps -- correct, since a run
    that diverges late should not be judged by where it ended up. But it took
    the best across SPLITS too. An arm reporting train loss 0.01 and test loss
    0.90 was ranked at 0.01 and beat an arm whose test loss was 0.50, so the
    ablation came out backwards with no caveat saying why.

    The corpus guard already refuses to compare arms that saw different data.
    Splits are the same class of mistake one level down.
    """
    write(
        tmp_path / "exp" / "results" / "exp_armA.json",
        '{"train": {"loss": 0.01}, "test": {"loss": 0.90}}',
    )
    write(tmp_path / "exp" / "results" / "exp_armB.json", '{"test": {"loss": 0.50}}')

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "exp", metric="loss")

    assert out["winner"] == "exp_armB", (
        "armA won on its TRAIN loss while being worse on test -- "
        f"arms: {[(a['name'], a.get('value'), a.get('split')) for a in out['arms']]}"
    )
    by_name = {a["name"]: a for a in out["arms"]}
    assert by_name["exp_armA"]["value"] == 0.90, "armA must be judged on test, not train"
    assert by_name["exp_armA"]["split"] == "test"
    conn.close()


def test_an_arm_with_only_a_train_split_is_flagged_not_silently_ranked(tmp_path):
    """If no arm has an evaluation split, ranking on train is all that is
    available -- but the reader has to be told that is what happened."""
    write(tmp_path / "exp" / "results" / "exp_armA.json", '{"train": {"loss": 0.01}}')
    write(tmp_path / "exp" / "results" / "exp_armB.json", '{"train": {"loss": 0.50}}')

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "exp", metric="loss")

    assert " ".join(out["caveats"]).find("train") >= 0, (
        f"ranking on a training split must be caveated; got {out['caveats']}"
    )
    conn.close()


def test_a_training_arm_beating_an_unlabelled_one_is_caveated(tmp_path):
    """The gap between the two split branches.

    `_split_rank(None)` returns len(_EVAL_SPLITS) and train returns one more,
    so an all-train check (`all(rank > len(_EVAL_SPLITS))`) misses a mix of
    train and unlabelled, and the different-splits check filters None out
    before counting. Neither fires, and an arm judged on training loss beats
    one judged on an unlabelled number in silence -- the same backwards
    ablation the split-awareness fix was written to prevent, one case over.
    """
    write(tmp_path / "ab" / "results" / "ab_lowlr.json", '{"train": {"loss": 0.01}}')
    write(tmp_path / "ab" / "results" / "ab_highlr.json", '{"loss": 0.50}')

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path)
    out = ledger.compare(conn, "ab", metric="loss")

    assert out["winner"] == "ab_lowlr"  # it does win; the point is being told why not to trust it
    caveats = " ".join(out["caveats"])
    assert "train" in caveats, f"a training-split arm won without a caveat: {out['caveats']}"
    conn.close()


def test_an_unrecognised_split_name_is_not_treated_as_unlabelled(tmp_path):
    """`_split_rank` maps every unknown string to the same rank as None, so
    `{"split": "holdout_b"}` and no split at all become one candidate pool and
    `_best_step` picks the extreme across both."""
    from attestation.ledger import _split_rank

    assert _split_rank("test") < _split_rank(None)
    assert _split_rank("train") > _split_rank(None)
    assert _split_rank("some_unknown_name") == _split_rank("another_unknown"), (
        "unknown splits should at least be consistent with each other"
    )


def _two_projects_one_family(tmp_path):
    """Two unrelated projects whose runs happen to share family and arm names.

    English and Mandarin ASR: different task, different data, no relationship.
    The names collide because `asr_baseline` is what everyone calls a baseline.
    """
    import json

    for project, corpus_name, base, big in (
        ("asr-english", "librispeech-100h", 2.4, 2.1),
        ("asr-mandarin", "aishell-1", 9.9, 9.1),
    ):
        results = tmp_path / project / "results"
        results.mkdir(parents=True)
        for tag, wer in (("asr_baseline", base), ("asr_biglm", big)):
            (results / f"{tag}.json").write_text(
                json.dumps({"tag": tag, "wer": wer, "corpus": corpus_name})
            )
    return tmp_path


def test_compare_does_not_pool_a_family_across_projects(tmp_path):
    """`compare` selected WHERE family = ? with no project filter.

    `families()` and `runs.list` both present (project, family) as the unit,
    and `compare` had no project parameter at all -- so a caller could not ask
    the question correctly. Two unrelated projects with a shared family name
    were pooled into one ranking and a winner was named across them.

    `claims.py` scopes by project AND run and has an `ambiguous` verdict for
    exactly this. `compare` is the one consumer that dropped project scope, and
    it is the one that produces a verdict.
    """
    from attestation import ledger
    from attestation.db import get_db

    root = _two_projects_one_family(tmp_path)
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, root)

    import pytest

    with pytest.raises(ValueError, match="2 projects"):
        ledger.compare(conn, "asr", metric="wer")

    # Naming the project is the way to ask the question, and it answers.
    scoped = ledger.compare(conn, "asr", metric="wer", project="asr-english")
    roots = {a["source_path"].split("/results/")[0].rsplit("/", 1)[-1] for a in scoped["arms"]}
    assert roots == {"asr-english"}
    assert scoped["winner"] == "asr_biglm"


def test_corpus_agreement_is_not_erased_by_a_name_collision(tmp_path):
    """`_corpus_agreement` keyed on run NAME, so a collision overwrote it.

    `named = {r["name"]: r.get("corpus") for r in runs}` -- when two projects
    share an arm name, the later row wins and the disagreement disappears. The
    four-arm ASR case reported `corpus: aishell-1`, and the CLI printed "all
    arms on aishell-1", which is false for half of them.

    That sentence is the one the docstring says earns a reader's trust: "All
    arms agree: name it, so the reader learns the comparison was checked rather
    than assumed." The guard failed CLOSED to a confident false positive rather
    than open to unknown.
    """
    from attestation import ledger
    from attestation.db import get_db

    root = _two_projects_one_family(tmp_path)
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, root)

    # Scoped to one project, the corpus claim is true and worth making.
    scoped = ledger.compare(conn, "asr", metric="wer", project="asr-english")
    assert scoped.get("corpus") == "librispeech-100h"

    # And a name collision WITHIN one project can no longer erase disagreement,
    # because two projects can no longer be pooled in the first place.
    import pytest

    with pytest.raises(ValueError):
        ledger.compare(conn, "asr", metric="wer")


def test_corpus_agreement_is_safe_called_directly_with_colliding_names():
    """The helper must not depend on its caller's discipline.

    `compare` now refuses to span projects, so the collision cannot reach here
    through that path. But `_corpus_agreement` keyed on run name alone while
    the schema says `UNIQUE (project, name)` -- the database already knew the
    identity was a pair. A helper that is only correct because of where it
    happens to be called from is one refactor away from being wrong again.
    """
    from attestation.ledger import _corpus_agreement

    arms = [
        {"project": "asr-english", "name": "asr_baseline", "corpus": "librispeech-100h"},
        {"project": "asr-mandarin", "name": "asr_baseline", "corpus": "aishell-1"},
    ]
    corpus, caveats = _corpus_agreement(arms, "wer")

    assert corpus is None, f"reported agreement on {corpus!r} across two corpora"
    assert caveats, "two corpora and no caveat"


def test_two_corpora_sharing_a_name_do_not_silently_merge(tmp_path):
    """`corpora.name` is globally unique, so round-15's defect survives one
    scope down -- inside a single project.

    Two arms each declared corpus `internal-eval` with different
    dataset_source (librispeech-clean vs librispeech-other). upsert keys on the
    name, keeps the first source, discards the second, and hands
    _corpus_agreement two identical strings -- which then reports checked
    agreement between arms that ran on different data. Round 15 fixed that
    function's key; the disagreement is erased UPSTREAM of it.

    upsert's docstring says "a declaration must never be silently replaced by a
    weaker value". A CONFLICTING value is the worse case and was ignored
    entirely.
    """
    from attestation import corpus, ledger
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    corpus.upsert(conn, {"name": "internal-eval", "source": "librispeech-clean"})
    corpus.upsert(conn, {"name": "internal-eval", "source": "librispeech-other"})
    conn.commit()

    rows = list(conn.execute("SELECT name, source FROM corpora"))
    assert any("CONTESTED" in (r["source"] or "") for r in rows), (
        f"two different corpora collapsed silently into {[dict(r) for r in rows]}"
    )

    # And the marker must reach a READER, not just the database: compare
    # cannot claim agreement on a corpus whose own definition is disputed.
    conn.execute(
        "INSERT INTO runs(project, name, family, status, source_path, corpus_id)"
        " VALUES ('p', 'a', 'f', 'recorded', '/tmp/a.json', 1)"
    )
    conn.execute(
        "INSERT INTO runs(project, name, family, status, source_path, corpus_id)"
        " VALUES ('p', 'b', 'f', 'recorded', '/tmp/b.json', 1)"
    )
    for run_id, value in ((1, 0.05), (2, 0.09)):
        conn.execute(
            "INSERT INTO run_metrics(run_id, metric, value) VALUES (?, 'wer', ?)",
            (run_id, value),
        )
    conn.commit()

    result = ledger.compare(conn, "f", metric="wer")
    assert result.get("corpus") is None, "vouched for a corpus whose definition is disputed"
    assert any("disputed" in c for c in result["caveats"]), result["caveats"]


def test_concurrent_first_scans_do_not_lose_the_corpus(tmp_path):
    """The fourth site of the same check-then-insert shape, and the worst.

    `corpus.upsert` reads `SELECT * FROM corpora WHERE name = ?` and then
    INSERTs. Concurrent first scans of one workspace -- the ordinary case when
    two projects declare the same corpus -- gave 7 of 8 callers an
    OperationalError and left ZERO rows: the others lost their data rather than
    merely their race.

    Only the first scan is exposed; once the row exists the UPDATE path is
    safe, which is why it hides behind a populated database.
    """
    import concurrent.futures

    from attestation import corpus
    from attestation.db import get_db

    db = tmp_path / "t.db"
    get_db(db).commit()

    def declare(_):
        try:
            corpus.upsert(get_db(db), {"name": "shared-corpus", "source": "librispeech"})
            return None
        except Exception as exc:  # noqa: BLE001 -- the point is what leaks out
            return f"{type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        errors = [e for e in pool.map(declare, range(8)) if e]

    assert not errors, f"{len(errors)} of 8 concurrent upserts failed: {errors[0]}"
    rows = get_db(db).execute("SELECT COUNT(*) FROM corpora").fetchone()[0]
    assert rows == 1, f"{rows} corpora rows for one name"


def test_the_example_workspace_ranks_with_no_user_configuration():
    """A new user's FIRST `runs compare` was a refusal.

    The shipped examples/workspace records ndcg_at_10, which was absent from a
    15-entry METRIC_DIRECTION table, so the flagship capability's first
    impression was "no metric with a known direction" -- correct by the tool's
    rules, and a wall. Measured on the real corpus at ~/qc: 17 of 44 multi-run
    families refused, and the refusals were ordinary retrieval and
    classification metrics, not edge cases.

    Honest scope: widening the table did NOT change the real corpus's yield
    (22 families ranked before and after). It fixes the fixture, which is what
    a new user meets. The real bottleneck there is cross-project ambiguity.
    """
    from attestation.ledger import METRIC_DIRECTION, _metric_direction, _metric_stem

    assert _metric_stem("ndcg_at_10") == "ndcg", "the @k cutoff suffix is not stripped"
    assert _metric_direction("ndcg_at_10", METRIC_DIRECTION) == "higher_is_better"
    # k is unbounded, so this is a pattern rather than a list.
    assert _metric_stem("recall_at_5") == "recall"
    assert _metric_stem("map_at_100") == "map"
    # Affixes still compose with it.
    assert _metric_stem("test_ndcg_at_10") == "ndcg"


def test_the_refusal_declines_to_guess_when_no_metric_stands_out():
    """The suggestion added to this refusal must never be confidently wrong --
    the refusal exists to stop confident wrong answers.

    Three versions were measured against ~/qc's real families, and each earlier
    one named a metric that would have misled: alphabetical order picked
    `n_records` (a row count), a bookkeeping blocklist picked
    `consecutive_detections` out of 43 names, and substring matching picked
    `nll_missing_rate` because it contains "nll" -- a missing-data rate where
    higher_is_better is exactly backwards.
    """
    from attestation.ledger import _likeliest_metric

    assert _likeliest_metric({"n_records": 4, "ndcg_at_10": 4}) == "ndcg_at_10"
    # Several plausible candidates: name shape cannot rank between them.
    assert _likeliest_metric({"test_acc": 3, "probe_auc_polarity": 3}) is None
    # Nothing resembling a known metric.
    assert _likeliest_metric({"consecutive_detections": 9, "window": 9}) is None
    assert _likeliest_metric({}) is None


def test_the_cross_project_refusal_ranks_the_projects_by_arm_count():
    """This refusal listed 13 project names alphabetically for one family and
    asked the reader to pick, giving them nothing to pick ON -- and alphabetical
    order put a dated backup directory second.

    Measured on a real corpus: family 'water' spans ferric plus twelve of its
    git worktrees. Arm counts are the one signal already in hand that separates
    the main line of work from a worktree that ran a subset, and ordering by
    them puts `ferric (13)` first.

    Collapsing worktrees onto their main checkout was tried and REVERTED: 58 of
    62 run names in those directories collide, because they are the same
    experiment re-run on different branches rather than arms of a sweep.
    Merging would have presented twelve copies of one run as comparable arms.
    """

    from attestation.db import get_db
    from attestation.ledger import compare

    conn = get_db(":memory:")
    for project, n in (("worktree-b", 2), ("main-line", 5), ("aaa-backup", 3)):
        for i in range(n):
            cur = conn.execute(
                "INSERT INTO runs(project, name, family, status, source_path)"
                " VALUES (?, ?, 'fam', 'recorded', ?)",
                (project, f"arm{i}", f"/tmp/{project}/arm{i}.json"),
            )
            conn.execute(
                "INSERT INTO run_metrics(run_id, metric, value) VALUES (?, 'accuracy', ?)",
                (cur.lastrowid, 0.5 + i / 100),
            )
    conn.commit()

    with pytest.raises(ValueError) as excinfo:
        compare(conn, "fam")
    message = str(excinfo.value)
    assert "main-line (5)" in message, f"arm counts are not reported: {message}"
    # Ordered by count: the busiest project leads, not the alphabetical first.
    assert message.index("main-line") < message.index("aaa-backup") < message.index("worktree-b"), (
        f"projects are not ordered by arm count: {message}"
    )


def test_a_family_with_no_metrics_is_not_told_to_declare_a_direction():
    """ "No metrics at all" is a different problem from "direction undeclared",
    and the second message sends the reader to a TOML file where there is
    nothing for them to write.

    Reached on a real corpus by following this tool's own advice: `runs compare
    water --project ferric` reported `found none` and then advised declaring
    one of them.
    """
    from attestation.db import get_db
    from attestation.ledger import compare

    conn = get_db(":memory:")
    for i in range(3):
        conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES ('p', ?, 'fam', 'recorded', ?)",
            (f"arm{i}", f"/tmp/p/arm{i}.json"),
        )
    conn.commit()

    with pytest.raises(ValueError) as excinfo:
        compare(conn, "fam")
    message = str(excinfo.value)
    assert "no recorded metrics" in message, message
    assert "metric_direction" not in message, (
        f"a family with no metrics is pointed at the direction file: {message}"
    )


def test_nested_splits_become_separate_arms_not_one_run_scored_by_its_best():
    """Real artifacts put the arms of an experiment INSIDE one file, under keys
    like `arms.Baseline` / `arms.Oracle_Post`. The ledger's unit is the file,
    and `_split_rank` gives every unrecognised nested key the same rank, so
    `_best_step` took the max across siblings.

    Measured on a real corpus: a run recording Baseline 0.000, Control_RAG
    0.644, Oracle_Post 0.988 and Treatment_Eigen 0.655 scored 0.988. Every arm
    was credited with its own oracle upper bound. On lm-eval-harness output the
    same bug ranked each model by its own easiest subtask, reordering the
    bottom half of the table.
    """
    from attestation.db import get_db
    from attestation.ledger import compare, nested_arms

    # The detector fires only on several unrecognised nested keys.
    assert sorted(
        nested_arms([{"split": "arms.A", "value": 1.0}, {"split": "arms.B", "value": 2.0}])
    ) == ["arms.A", "arms.B"]
    assert nested_arms([{"split": "test", "value": 1.0}, {"split": "train", "value": 2.0}]) == {}
    assert nested_arms([{"split": "arms.Only", "value": 1.0}]) == {}

    conn = get_db(":memory:")
    for run_name, scores in (
        ("seed1", {"arms.Baseline": 0.10, "arms.Treatment": 0.60, "arms.Oracle": 0.99}),
        ("seed2", {"arms.Baseline": 0.12, "arms.Treatment": 0.62, "arms.Oracle": 0.98}),
    ):
        cur = conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES ('p', ?, 'fam', 'recorded', ?)",
            (run_name, f"/tmp/{run_name}.json"),
        )
        for split, value in scores.items():
            conn.execute(
                "INSERT INTO run_metrics(run_id, metric, value, split)"
                " VALUES (?, 'accuracy', ?, ?)",
                (cur.lastrowid, value, split),
            )
    conn.commit()

    out = compare(conn, "fam")
    names = [a["name"] for a in out["arms"]]
    assert len(names) == 6, f"two runs of three arms should give six arms, got {names}"
    assert "seed1[arms.Baseline]" in names, names
    # The oracle is still top -- it is genuinely the highest number -- but it is
    # now VISIBLE as its own arm rather than silently standing in for the run.
    baseline = next(a for a in out["arms"] if a["name"] == "seed1[arms.Baseline]")
    assert baseline["value"] == 0.10, (
        "the baseline arm reports the oracle's value; siblings are still collapsed"
    )


def test_an_empty_ledger_says_so_rather_than_blaming_the_filenames():
    """`runs.compare` on a fresh install said "no run has one: families are
    derived from a shared filename prefix, so arms need names like
    `asr_baseline`" -- describing a naming problem when the real one is that
    nothing has been scanned.

    `runs.list` gets this right, but `runs.compare` is where a model lands
    first: both gemma4:e2b and gemma4:e4b dead-ended here, asking the user for
    a family name that already existed on disk. Patching only this message made
    e2b scan and complete the task.
    """
    from attestation.db import get_db
    from attestation.ledger import compare

    empty = get_db(":memory:")
    out = compare(empty, "kdsweep")
    assert "EMPTY" in out["message"], out["message"]
    assert "runs.scan" in out["message"], (
        f"the refusal does not name the call that fixes it: {out['message']!r}"
    )

    # A populated ledger whose runs genuinely have no families keeps the
    # naming explanation, which is correct THERE.
    populated = get_db(":memory:")
    populated.execute(
        "INSERT INTO runs(project, name, status, source_path)"
        " VALUES ('p', 'lonely', 'recorded', '/tmp/lonely.json')"
    )
    populated.commit()
    named = compare(populated, "kdsweep")
    assert "EMPTY" not in named["message"], named["message"]
    assert "filename prefix" in named["message"], named["message"]


def test_compare_picks_the_majority_metric_with_no_db():
    from attestation.ledger import _compare

    rows = [
        {
            "id": 1,
            "name": "asr_a",
            "family": "asr",
            "project": "p",
            "status": "recorded",
            "source_path": "/tmp/asr_a.json",
            "corpus_id": None,
            "adapter": None,
        },
        {
            "id": 2,
            "name": "asr_b",
            "family": "asr",
            "project": "p",
            "status": "recorded",
            "source_path": "/tmp/asr_b.json",
            "corpus_id": None,
            "adapter": None,
        },
        {
            "id": 3,
            "name": "asr_c",
            "family": "asr",
            "project": "p",
            "status": "recorded",
            "source_path": "/tmp/asr_c.json",
            "corpus_id": None,
            "adapter": None,
        },
    ]
    values = {
        1: [{"run_id": 1, "value": 0.30, "step": None, "split": None}],
        2: [{"run_id": 2, "value": 0.20, "step": None, "split": None}],
        3: [],
    }
    out = _compare(
        rows, values, {}, metric="wer", directions={"wer": "lower_is_better"}, family="asr"
    )
    assert out["metric"] == "wer" and out["direction"] == "lower_is_better"
    assert out["winner"] == "asr_b"
    assert out["without_metric"] == ["asr_c"]
    assert [a["name"] for a in out["arms"]] == ["asr_b", "asr_a", "asr_c"]
    assert isinstance(out["caveats"], list)


def test_collapse_to_last_keeps_the_last_row_per_metric_name():
    from attestation.ledger import collapse_to_last

    metrics = [
        {"metric": "loss", "step": 1, "value": 0.9},
        {"metric": "acc", "step": 1, "value": 0.5},
        {"metric": "loss", "step": 2, "value": 0.4},
    ]
    out = collapse_to_last(metrics)
    assert [m["metric"] for m in out] == ["loss", "acc"]  # first-appearance order
    assert out[0]["step"] == 2

    # The live worst case the docstring names: `split` carries the sweep
    # coordinate, so keying on (metric, split) does not collapse a series at
    # all -- each split value gets kept as its own "last" row. Two rows share
    # the metric name "loss" but differ in split; the function must collapse
    # to the single LAST row by name alone, regardless of split.
    split_metrics = [
        {"metric": "loss", "split": "train", "step": 1, "value": 0.9},
        {"metric": "loss", "split": "eval", "step": 2, "value": 0.4},
    ]
    split_out = collapse_to_last(split_metrics)
    assert len(split_out) == 1, "keying on (metric, split) would keep one row per split"
    assert split_out[0]["split"] == "eval" and split_out[0]["step"] == 2


def test_compare_skips_the_metric_count_query_when_a_metric_is_named(conn, workspace):
    """`_compare_rows` only counts how many arms report each metric name when
    the caller has not already named one -- that GROUP BY exists purely to
    pick a metric, so running it after the caller already picked one is a
    wasted round trip. The original, unsplit `compare()` guarded this query
    with `if metric is None:`; the split must keep the same guard rather than
    running it unconditionally now that it lives in a separate reader.
    """
    ledger.scan(conn, workspace)

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        ledger.compare(conn, "eval", metric="wer")
    finally:
        conn.set_trace_callback(None)
    assert not any("GROUP BY run_id, metric" in s for s in statements), statements

    statements.clear()
    conn.set_trace_callback(statements.append)
    try:
        ledger.compare(conn, "eval")
    finally:
        conn.set_trace_callback(None)
    assert any("GROUP BY run_id, metric" in s for s in statements), statements


def test_detail_reports_when_it_was_scanned(conn, workspace):
    """A citable run needs an "as of when", separate from the artifact's own
    `started` time -- a stale artifact re-scanned today has an old `started`
    and a new `scanned_at`, and collapsing the two would hide exactly the
    staleness `claims.py`'s `stale` verdict exists to catch."""
    before = datetime.now(UTC).isoformat(timespec="seconds")
    ledger.scan(conn, workspace)

    run = ledger.detail(conn, "speech-model", "dit_small_rope_crossattn")

    assert run["scanned_at"] >= before


def test_run_ids_survive_a_rescan_of_an_unchanged_project(conn, workspace):
    """`runs.id` is the only thing a reader can use to cite one run (Datasette
    row URL); it must not change under a re-scan that found the same
    project/name pairs, or the citation mechanism is broken by construction."""
    ledger.scan(conn, workspace)
    ids = {
        (r["project"], r["name"]): r["id"]
        for r in conn.execute("SELECT id, project, name FROM runs")
    }

    ledger.scan(conn, workspace)
    again = {
        (r["project"], r["name"]): r["id"]
        for r in conn.execute("SELECT id, project, name FROM runs")
    }

    assert ids == again


def test_a_run_that_disappears_from_disk_is_removed_on_rescan(conn, workspace):
    """`_replace_project`'s stale-id branch: a run whose artifact vanished
    from disk must vanish from `runs` (and its `run_metrics`) on the next
    scan, while a surviving run in the same project keeps its id -- the
    upsert must not turn "replace wholesale" into "only ever add"."""
    ledger.scan(conn, workspace)
    before = {
        r["name"]: r["id"]
        for r in conn.execute("SELECT id, name FROM runs WHERE project = ?", ("physics-engine",))
    }
    removed_id = before["benchmark_atz"]
    surviving_id = before["benchmark_adz"]

    (workspace / "physics-engine" / "results" / "benchmark_atz.json").unlink()
    ledger.scan(conn, workspace)

    after = {
        r["name"]: r["id"]
        for r in conn.execute("SELECT id, name FROM runs WHERE project = ?", ("physics-engine",))
    }
    assert "benchmark_atz" not in after
    assert after["benchmark_adz"] == surviving_id
    assert (
        conn.execute(
            "SELECT COUNT(*) c FROM run_metrics WHERE run_id = ?", (removed_id,)
        ).fetchone()["c"]
        == 0
    )


def test_caveats_do_not_confuse_nested_arm_keys_with_eval_splits():
    """`_arms_for_run` reuses the `split` key for two different meanings: a
    genuine eval split (`test`, `val`) and a synthetic nested-arm key
    (`arms.Treatment_Eigen`) when one file fans out into several arms. The
    "arms are judged on different splits" caveat must not fire on two arms
    that are only nested-arm siblings, not different levels of eval trust."""
    from attestation.ledger import _caveats

    scored = [
        {
            "name": "run[arms.A]",
            "value": 0.9,
            "split": "arms.A",
            "step": None,
            "n": None,
            "status": "ok",
            "source_path": "x",
        },
        {
            "name": "run[arms.B]",
            "value": 0.8,
            "split": "arms.B",
            "step": None,
            "n": None,
            "status": "ok",
            "source_path": "x",
        },
    ]
    assert not [c for c in _caveats(scored, "auc") if "different splits" in c]
