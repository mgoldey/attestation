"""Tests for the attestation-record eval scorer, model-free.

`evals/` is not a package; the modules there are scripts with an importable
core, same as test_tagging_eval.py's own sys.path insertion.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import record_eval as re_  # noqa: E402


def _scenario(**overrides) -> dict:
    base = {
        "id": "t",
        "family": "asr",
        "arms": ["asr_baseline", "asr_biglm"],
        "metric": "wer",
        "metrics": ["wer"],
        "values": {"asr_baseline": 0.12, "asr_biglm": 0.08},
        "direction": "lower_is_better",
        "built_in": True,
        "corpus": "librispeech",
    }
    base.update(overrides)
    return base


def _good_manifest(scenario: dict, *, direction_toml: str | None = None) -> dict:
    files = {}
    for arm in scenario["arms"]:
        files[f"results/{arm}.json"] = json.dumps({scenario["metric"]: scenario["values"][arm]})
    files[f"configs/{scenario['arms'][0]}.yaml"] = "model: base\n"
    if direction_toml:
        files["metric_direction.toml"] = direction_toml
    return {"files": files}


def test_a_good_manifest_passes_every_check(tmp_path):
    scenario = _scenario()
    manifest = _good_manifest(scenario)

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert result["pass"], result["errors"]
    assert all(result["checks"].values()), result["checks"]


def test_a_not_built_in_metric_without_a_direction_declaration_fails(tmp_path):
    scenario = _scenario(
        family="opt",
        arms=["opt_adam", "opt_sgd"],
        metric="regret_bound",
        metrics=["regret_bound"],
        values={"opt_adam": 0.03, "opt_sgd": 0.09},
        built_in=False,
    )
    manifest = _good_manifest(scenario)  # no metric_direction.toml written

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["direction_declared"]
    assert any("not declared" in e or "refused" in e for e in result["errors"])


def test_a_not_built_in_metric_with_a_declaration_passes(tmp_path):
    scenario = _scenario(
        family="opt",
        arms=["opt_adam", "opt_sgd"],
        metric="regret_bound",
        metrics=["regret_bound"],
        values={"opt_adam": 0.03, "opt_sgd": 0.09},
        built_in=False,
    )
    manifest = _good_manifest(
        scenario, direction_toml='[metric_direction]\nregret_bound = "lower_is_better"\n'
    )

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert result["pass"], result["errors"]


def test_arms_in_different_prefixes_fail_the_one_family_check(tmp_path):
    """Two arms that do not share a filename prefix never group into one
    family -- `family_of` returns something else (or None) for each, so
    `ledger.compare(family=...)` finds fewer than len(arms) runs in it."""
    scenario = _scenario(
        family="asr",
        arms=["asr_baseline", "totally_unrelated_name"],
        values={"asr_baseline": 0.12, "totally_unrelated_name": 0.08},
    )
    manifest = _good_manifest(scenario)

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["grouped_and_winner"]


def test_an_absolute_path_is_rejected(tmp_path):
    scenario = _scenario()
    manifest = {"files": {"/etc/passwd": "pwned"}}

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["manifest_parses"]
    assert any("absolute" in e for e in result["errors"])


def test_a_dotdot_path_is_rejected(tmp_path):
    scenario = _scenario()
    manifest = {"files": {"../../escape.json": "{}"}}

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["manifest_parses"]
    assert any("escapes" in e for e in result["errors"])


def test_a_malformed_manifest_is_reported_not_raised(tmp_path):
    scenario = _scenario()
    result = re_.score_one(scenario, {"not_files": {}}, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["manifest_parses"]


def test_a_result_misplaced_under_configs_is_dropped_and_fails_the_scan_count(tmp_path):
    """The generic adapter's own safety net: a JSON file under `configs/`
    that parses as numeric data is neither read as a spec (that branch is
    skipped for a parseable JSON) nor picked up by the results/ walk (which
    never looks in configs/) -- it is silently dropped. A manifest that puts
    an arm's actual result under `configs/` instead of `results/` therefore
    loses that arm entirely, and `scan_count` (checking exactly len(arms)
    runs were found) is what catches it."""
    scenario = _scenario()
    files = {
        f"results/{scenario['arms'][0]}.json": json.dumps(
            {scenario["metric"]: scenario["values"][scenario["arms"][0]]}
        ),
        # Misplaced: the second arm's result sits under configs/, not results/.
        f"configs/{scenario['arms'][1]}.json": json.dumps(
            {scenario["metric"]: scenario["values"][scenario["arms"][1]]}
        ),
    }
    manifest = {"files": files}

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["scan_count"]
    assert any("expected 2" in e for e in result["errors"])


def test_no_config_file_at_all_fails_the_config_check(tmp_path):
    scenario = _scenario()
    files = {}
    for arm in scenario["arms"]:
        files[f"results/{arm}.json"] = json.dumps({scenario["metric"]: scenario["values"][arm]})
    manifest = {"files": files}

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["checks"]["config_not_metric"]
    assert any("no config file" in e for e in result["errors"])


def test_the_wrong_winner_fails_the_grouped_and_winner_check(tmp_path):
    """The scorer must check against the SCENARIO's own values, not just
    whatever `compare` returns -- a manifest that swaps which arm has which
    value still scans fine and still groups fine, but names the wrong
    winner."""
    scenario = _scenario()  # asr_biglm (0.08) should win on lower_is_better
    files = {
        # Swapped: baseline now reports the better (lower) value.
        f"results/{scenario['arms'][0]}.json": json.dumps({scenario["metric"]: 0.02}),
        f"results/{scenario['arms'][1]}.json": json.dumps({scenario["metric"]: 0.30}),
        f"configs/{scenario['arms'][0]}.yaml": "model: base\n",
    }
    manifest = {"files": files}

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert not result["pass"]
    assert not result["checks"]["grouped_and_winner"]


# ---------------------------------------------------------------------------
# RED proof: the winner check.
#
# Mechanically mutated evals/record_eval.py's score_one, in place, then ran
# `uv run pytest tests/test_record_eval.py -k winner -q`, then reverted:
#
#     elif cmp.get("winner") != expected_winner:      # ORIGINAL
#     elif False:                                        # MUTANT (never disagrees)
#
# With the mutant in place:
#     test_the_wrong_winner_fails_the_grouped_and_winner_check FAILED
#     -- AssertionError: assert not True  (the swapped-value manifest, which
#        should fail because asr_baseline is reported as winning at 0.02
#        instead of asr_biglm, scored checks["grouped_and_winner"] = True)
#
# Reverted `elif False:` back to `elif cmp.get("winner") != expected_winner:`
# and reran: PASSED. This proves the `!=` comparison, not merely the
# grouping-count check beside it, is what test_the_wrong_winner_... depends
# on -- exactly the mutation-testing lesson (a guard that passes while
# protecting nothing).
# ---------------------------------------------------------------------------


def test_two_arms_of_a_three_arm_family_scenario_all_group(tmp_path):
    scenario = _scenario(
        family="gen",
        arms=["gen_greedy", "gen_beam", "gen_sample"],
        metric="coherence_index",
        metrics=["coherence_index"],
        values={"gen_greedy": 0.61, "gen_beam": 0.74, "gen_sample": 0.58},
        direction="higher_is_better",
        built_in=False,
    )
    manifest = _good_manifest(
        scenario, direction_toml='[metric_direction]\ncoherence_index = "higher_is_better"\n'
    )

    result = re_.score_one(scenario, manifest, workspace=tmp_path)

    assert result["pass"], result["errors"]


def test_load_cases_reads_the_committed_fixture():
    cases = re_.load_cases()
    assert len(cases) >= 10
    not_built_in = [c for c in cases if not c.get("built_in", True)]
    three_arm = [c for c in cases if len(c["arms"]) >= 3]
    assert len(not_built_in) >= 4
    assert len(three_arm) >= 2
    assert any(c.get("expect_fail") for c in cases)


@pytest.mark.parametrize("case_idx", range(len(re_.load_cases())))
def test_every_committed_case_scores_as_its_expect_fail_says(case_idx, tmp_path):
    case = re_.load_cases()[case_idx]
    result = re_.score_one(case, case["answer"], workspace=tmp_path / str(case_idx))
    if case.get("expect_fail"):
        assert not result["pass"], f"{case['id']} was supposed to fail but passed"
    else:
        assert result["pass"], f"{case['id']}: {result['errors']}"


# ---------------------------------------------------------------------------
# Sidecar round-trip: the coordinator's fix-round-1 requirement that every
# --live sample's raw answer and per-check result is examinable after the
# fact, since a live model varies between calls and a failing trial cannot
# be re-asked to reproduce what it originally wrote.
# ---------------------------------------------------------------------------

import run_record_eval as rre  # noqa: E402


def test_offline_does_not_write_a_sidecar_or_record(tmp_path, monkeypatch):
    monkeypatch.setattr(rre, "PROMPTS_DIR", tmp_path / "prompts")
    cases = rre.load_cases()

    # Exercise the actual --offline code path rather than reimplementing it:
    # run_offline() + no write_record/write_answers_sidecar call, matching
    # main()'s own `if args.live:` guard.
    result, samples = rre.run_offline(cases)
    assert result.overall == 1.0
    assert samples  # per-sample data was collected in-memory
    assert not (tmp_path / "prompts").exists()  # but never written to disk


def test_answers_sidecar_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(rre, "PROMPTS_DIR", tmp_path / "prompts")
    samples = [
        {
            "id": "case-a",
            "sample": 0,
            "answer": {"files": {"results/x.json": "{}"}},
            "checks": {"manifest_parses": True, "scan_count": False},
            "errors": ["scan found 1 run(s), expected 2"],
        },
        {
            "id": "case-a",
            "sample": 1,
            "answer": {"files": {"results/x.json": "{}", "results/y.json": "{}"}},
            "checks": {"manifest_parses": True, "scan_count": True},
            "errors": [],
        },
    ]

    path = rre.write_answers_sidecar(samples)
    round_tripped = json.loads(path.read_text())

    assert path.parent == tmp_path / "prompts"
    assert round_tripped == samples
    assert round_tripped[0]["answer"]["files"]["results/x.json"] == "{}"
    assert round_tripped[1]["checks"]["scan_count"] is True


def test_answers_sidecar_appends_across_calls_on_the_same_day(tmp_path, monkeypatch):
    """A second --live invocation the same day (e.g. record then annotate
    sharing the coordinator's dated .md) must not clobber the first run's
    samples."""
    monkeypatch.setattr(rre, "PROMPTS_DIR", tmp_path / "prompts")
    first = [{"id": "a", "sample": 0, "answer": "x", "checks": {}, "errors": []}]
    second = [{"id": "b", "sample": 0, "answer": "y", "checks": {}, "errors": []}]

    rre.write_answers_sidecar(first)
    path = rre.write_answers_sidecar(second)

    round_tripped = json.loads(path.read_text())
    assert [s["id"] for s in round_tripped] == ["a", "b"]


def test_write_record_reports_per_scenario_k_of_n_and_links_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(rre, "PROMPTS_DIR", tmp_path / "prompts")
    result = rre.EvalResult(
        per_case={"case-a": 0.5},
        runs={
            "case-a": [
                {"checks": {"manifest_parses": True}},
                {"checks": {"manifest_parses": False}},
            ]
        },
        latencies=[],
    )

    path = rre.write_record(result, "gemma4:e2b-it-q4_K_M", n_scenarios=1, repeat=2)
    text = path.read_text()

    assert "case-a" in text
    assert "1/2" in text  # k/N pass count for the scenario
    assert "write-side-" in text and ".answers.json" in text  # links the sidecar
