"""Tests for `attestation.record`: the pure planner + one I/O function
behind `attest runs record`.

Mirrors `test_ledger.py`'s own shape: fixtures are literal dicts and
strings, not a real project on disk, so `plan()`/`undeclared()`/
`merge_toml_table()` are exercised as pure functions and only `write()`
touches a filesystem.
"""

import sys
from pathlib import Path

import pytest

from attestation import record

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# plan(): the manifest, byte-exact
# ---------------------------------------------------------------------------


def test_plan_two_arm_manifest_is_byte_exact():
    manifest = record.plan(
        "asr",
        {"asr_baseline": {"wer": 0.12}, "asr_biglm": {"wer": 0.08}},
        corpus=None,
        directions=None,
        config=None,
        recorded_at="2026-09-01T00:00:00+00:00",
        known_directions={"wer": "lower_is_better"},
    )

    assert manifest == {
        "results/asr_asr_baseline.json": '{\n  "wer": 0.12\n}\n',
        "configs/asr_asr_baseline.yaml": (
            "family: asr\narm: asr_baseline\nrecorded_at: 2026-09-01T00:00:00+00:00\n"
        ),
        "results/asr_asr_biglm.json": '{\n  "wer": 0.08\n}\n',
        "configs/asr_asr_biglm.yaml": (
            "family: asr\narm: asr_biglm\nrecorded_at: 2026-09-01T00:00:00+00:00\n"
        ),
    }


def test_plan_includes_corpus_and_config_and_direction_files():
    manifest = record.plan(
        "opt",
        {"opt_adam": {"regret_bound": 0.03}},
        corpus="cifar10",
        directions={"regret_bound": "lower_is_better"},
        config={"lr": "0.01"},
        recorded_at="2026-09-01T00:00:00+00:00",
        known_directions={},
    )

    assert manifest["configs/opt_opt_adam.yaml"] == (
        "family: opt\narm: opt_adam\ncorpus: cifar10\n"
        "recorded_at: 2026-09-01T00:00:00+00:00\nlr: 0.01\n"
    )
    assert manifest["corpora.toml"] == (
        '[corpus.cifar10]\nsource = "cifar10"\n\n[assign.family]\nopt = "cifar10"\n'
    )
    assert manifest["metric_direction.toml"] == (
        '[metric_direction]\nregret_bound = "lower_is_better"\n'
    )


def test_plan_omits_a_direction_file_entry_already_known():
    """A `--direction wer=lower_is_better` for a metric already in
    `known_directions` is redundant -- harmless per the skill's own "when in
    doubt, declare it" advice, but `plan` should not write a
    metric_direction.toml entry that says nothing new."""
    manifest = record.plan(
        "asr",
        {"asr_baseline": {"wer": 0.12}},
        directions={"wer": "lower_is_better"},
        known_directions={"wer": "lower_is_better"},
        recorded_at="2026-09-01T00:00:00+00:00",
    )

    assert "metric_direction.toml" not in manifest


def test_plan_defaults_recorded_at_to_now_iso8601_utc():
    manifest = record.plan("f", {"a": {"m": 1.0}}, known_directions={"m": "higher_is_better"})
    config = manifest["configs/f_a.yaml"]
    line = next(line for line in config.splitlines() if line.startswith("recorded_at:"))
    stamp = line.split("recorded_at: ", 1)[1]
    assert stamp.endswith("+00:00") or stamp.endswith("Z")


# ---------------------------------------------------------------------------
# undeclared()
# ---------------------------------------------------------------------------


def test_undeclared_names_the_unknown_metric_and_nothing_else():
    arms = {"a": {"wer": 0.1, "novelty_rate": 0.3}, "b": {"wer": 0.2, "novelty_rate": 0.4}}
    known = {"wer": "lower_is_better"}

    assert record.undeclared(arms, known) == ["novelty_rate"]


def test_undeclared_is_empty_when_every_metric_is_known():
    arms = {"a": {"wer": 0.1}, "b": {"wer": 0.2}}
    known = {"wer": "lower_is_better"}

    assert record.undeclared(arms, known) == []


def test_undeclared_deduplicates_and_sorts():
    arms = {"a": {"z_metric": 1.0, "a_metric": 2.0}, "b": {"z_metric": 3.0}}

    assert record.undeclared(arms, {}) == ["a_metric", "z_metric"]


# ---------------------------------------------------------------------------
# validation: non-numeric value, bad metric name
# ---------------------------------------------------------------------------


def test_parse_metric_value_refuses_a_non_numeric_string():
    with pytest.raises(ValueError, match="not a number"):
        record.parse_metric_value("not-a-number")


def test_parse_metric_value_accepts_a_float_string():
    assert record.parse_metric_value("0.061") == 0.061


def test_validate_metric_name_refuses_a_bad_name():
    with pytest.raises(ValueError, match="invalid metric name"):
        record.validate_metric_name("has a space")


@pytest.mark.parametrize("name", ["wer", "novelty_rate", "F1", "ndcg_at_10"])
def test_validate_metric_name_accepts_word_characters(name):
    record.validate_metric_name(name)  # must not raise


def test_metric_name_grammar_matches_the_claim_parser():
    """`record.METRIC_NAME_RE` must accept exactly what `claims._FIELD_RE`'s
    key group accepts (`\\w+`), so a metric this command writes is never one
    the claim grammar then refuses to parse."""
    from attestation import claims

    for candidate in ("wer", "novelty_rate", "F1", "a1_b2"):
        claim_match = claims._FIELD_RE.match(f"{candidate}=1.0")
        assert claim_match is not None
        assert claim_match.group(1) == candidate
        assert record.METRIC_NAME_RE.fullmatch(candidate)


# ---------------------------------------------------------------------------
# write(): new files only, --force
# ---------------------------------------------------------------------------


def test_write_writes_every_manifest_file(tmp_path):
    manifest = {"results/f_a.json": '{"m": 1.0}\n', "configs/f_a.yaml": "family: f\n"}

    written = record.write(tmp_path, manifest, force=False)

    assert {p.relative_to(tmp_path).as_posix() for p in written} == set(manifest)
    assert (tmp_path / "results" / "f_a.json").read_text() == '{"m": 1.0}\n'
    assert (tmp_path / "configs" / "f_a.yaml").read_text() == "family: f\n"


def test_write_refuses_on_an_existing_file(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "f_a.json").write_text("old")
    manifest = {"results/f_a.json": "new", "configs/f_a.yaml": "family: f\n"}

    with pytest.raises(FileExistsError, match="results/f_a.json"):
        record.write(tmp_path, manifest, force=False)


def test_write_refuses_before_writing_anything(tmp_path):
    """The refused call must not leave a PARTIAL manifest on disk -- a
    second file existing must stop the first from being written too, even
    though dict iteration would reach it first."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "f_a.yaml").write_text("old")
    manifest = {"results/f_a.json": "new", "configs/f_a.yaml": "new-config"}

    with pytest.raises(FileExistsError):
        record.write(tmp_path, manifest, force=False)

    assert not (tmp_path / "results" / "f_a.json").exists()
    assert (tmp_path / "configs" / "f_a.yaml").read_text() == "old"


def test_write_with_force_overwrites(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "f_a.json").write_text("old")
    manifest = {"results/f_a.json": "new"}

    record.write(tmp_path, manifest, force=True)

    assert (tmp_path / "results" / "f_a.json").read_text() == "new"


# ---------------------------------------------------------------------------
# merge_toml_table(): literal text
# ---------------------------------------------------------------------------


def test_merge_toml_table_into_empty_file():
    out = record.merge_toml_table("", "metric_direction", {"wer": "lower_is_better"}, force=False)

    assert out == '[metric_direction]\nwer = "lower_is_better"\n'


def test_merge_toml_table_keeps_foreign_entries():
    existing = '[metric_direction]\naccuracy = "higher_is_better"\n'

    out = record.merge_toml_table(
        existing, "metric_direction", {"wer": "lower_is_better"}, force=False
    )

    assert 'accuracy = "higher_is_better"' in out
    assert 'wer = "lower_is_better"' in out


def test_merge_toml_table_refuses_to_clobber_a_differing_value_without_force():
    existing = '[metric_direction]\nwer = "higher_is_better"\n'

    with pytest.raises(ValueError, match="refusing to overwrite"):
        record.merge_toml_table(
            existing, "metric_direction", {"wer": "lower_is_better"}, force=False
        )


def test_merge_toml_table_force_overwrites_a_differing_value():
    existing = '[metric_direction]\nwer = "higher_is_better"\n'

    out = record.merge_toml_table(
        existing, "metric_direction", {"wer": "lower_is_better"}, force=True
    )

    import tomllib

    assert tomllib.loads(out)["metric_direction"]["wer"] == "lower_is_better"


def test_merge_toml_table_is_a_noop_for_an_identical_existing_value():
    existing = '[metric_direction]\nwer = "lower_is_better"\n'

    out = record.merge_toml_table(
        existing, "metric_direction", {"wer": "lower_is_better"}, force=False
    )

    import tomllib

    assert tomllib.loads(out)["metric_direction"]["wer"] == "lower_is_better"


def test_merge_toml_table_handles_a_dotted_nested_table():
    out = record.merge_toml_table("", "assign.family", {"asr": "librispeech"}, force=False)

    import tomllib

    doc = tomllib.loads(out)
    assert doc["assign"]["family"]["asr"] == "librispeech"


def test_merge_toml_table_merges_two_dotted_tables_in_the_same_file():
    """`corpora.toml` needs `[corpus.<name>]` and `[assign.family]` merged
    in sequence into the same growing text -- the shape `toml_tables`'s
    walk actually does."""
    out = record.merge_toml_table("", "corpus.librispeech", {"source": "librispeech"}, force=False)
    out = record.merge_toml_table(out, "assign.family", {"asr": "librispeech"}, force=False)

    import tomllib

    doc = tomllib.loads(out)
    assert doc["corpus"]["librispeech"]["source"] == "librispeech"
    assert doc["assign"]["family"]["asr"] == "librispeech"


def test_merge_toml_table_force_does_not_rewrite_a_foreign_tables_same_named_key():
    """CRITICAL 1 (final review, round 2): a foreign table declaring the
    same KEY NAME earlier in the file must be left byte-identical when
    --force is used to update a DIFFERENT table's entry for that key.

    The reviewer's exact reproduction: `[assign.run]` has its own `source =`
    line before `[corpus.librispeech]`'s. A table-blind text substitution
    (the old `_replace_existing_keys`, matching the first `^source = ...$`
    anywhere in the file) rewrites the FOREIGN `[assign.run]` entry and
    leaves the intended `[corpus.librispeech]` entry stale -- silently,
    exit 0. Merging on parsed TOML (not text) must update only the table
    actually named.
    """
    existing = (
        '[assign.run]\nsource = "run-level-keepme"\n\n[corpus.librispeech]\nsource = "old-source"\n'
    )

    out = record.merge_toml_table(
        existing, "corpus.librispeech", {"source": "new-source"}, force=True
    )

    import tomllib

    doc = tomllib.loads(out)
    assert doc["assign"]["run"]["source"] == "run-level-keepme", "foreign entry must survive"
    assert doc["corpus"]["librispeech"]["source"] == "new-source", "the intended entry must update"


def test_merge_toml_table_preserves_a_comment_line():
    existing = (
        '# a hand-written note about this file\n[metric_direction]\nwer = "lower_is_better"\n'
    )

    out = record.merge_toml_table(
        existing, "metric_direction", {"accuracy": "higher_is_better"}, force=False
    )

    assert "# a hand-written note about this file" in out
    import tomllib

    doc = tomllib.loads(out)
    assert doc["metric_direction"]["wer"] == "lower_is_better"
    assert doc["metric_direction"]["accuracy"] == "higher_is_better"


def test_merge_toml_table_preserves_untouched_table_ordering():
    existing = '[corpus.a]\nsource = "a"\n\n[corpus.b]\nsource = "b"\n\n[corpus.c]\nsource = "c"\n'

    out = record.merge_toml_table(existing, "corpus.b", {"config": "clean"}, force=False)

    import tomllib

    doc = tomllib.loads(out)
    assert list(doc["corpus"]) == ["a", "b", "c"], "table order must survive an unrelated merge"
    assert doc["corpus"]["b"]["config"] == "clean"
    assert doc["corpus"]["a"]["source"] == "a"
    assert doc["corpus"]["c"]["source"] == "c"


# ---------------------------------------------------------------------------
# the manifest scored by record_eval.score_one for every committed scenario
# ---------------------------------------------------------------------------


def test_plan_output_passes_the_eval_scorer_for_every_committed_scenario(tmp_path):
    """`plan()`'s own output -- not a hand-written answer -- must pass
    `record_eval.score_one` for every non-expect_fail scenario in
    `record_cases.json`, and correctly fail the expect_fail one. This is
    the deterministic, offline half of the spec's 11/11 acceptance: the
    other half (`evals/run_record_eval.py --command`) drives the real CLI
    subprocess: this one drives `plan()` directly.
    """
    sys.path.insert(0, str(ROOT / "evals"))
    from record_eval import load_cases, score_one

    from attestation import ledger

    for case in load_cases():
        known = dict(ledger.METRIC_DIRECTION)
        directions = {}
        # The one expect_fail scenario ("bait-missing-direction") stands for
        # what an invocation would produce that the CLI refuses BEFORE
        # calling plan() at all -- no --direction given, so no
        # metric_direction.toml. Mirroring that refusal here means not
        # declaring a direction for it, same as a real refused call never
        # reaches plan().
        if not case.get("built_in", True) and not case.get("expect_fail"):
            directions[case["metric"]] = case["direction"]
        # scenario["arms"] entries are already full stems (e.g. "asr_baseline",
        # matching record_cases.json's own hand-written results/asr_baseline.json
        # answers). plan() prefixes NAME with FAMILY itself
        # (results/<FAMILY>_<NAME>.json), so the bare arm name passed in here
        # must have that family prefix stripped first, or the stem doubles up
        # (asr_asr_baseline) and family_of groups it under "asr-asr" instead
        # of the scenario's own "asr".
        prefix = f"{case['family']}_"
        arms = {
            (arm[len(prefix) :] if arm.startswith(prefix) else arm): {
                case["metric"]: case["values"][arm]
            }
            for arm in case["arms"]
        }

        manifest = record.plan(
            case["family"],
            arms,
            corpus=case["corpus"],
            directions=directions,
            known_directions=known,
            recorded_at="2026-09-01T00:00:00+00:00",
        )

        result = score_one(case, {"files": manifest}, workspace=tmp_path / case["id"])
        if case.get("expect_fail"):
            assert not result["pass"], f"{case['id']} was supposed to fail but passed"
        else:
            assert result["pass"], f"{case['id']}: {result['errors']}"


# ---------------------------------------------------------------------------
# architecture: no sqlite3, no attestation.llm
# ---------------------------------------------------------------------------


def test_record_module_avoids_sqlite3_and_llm():
    import ast

    tree = ast.parse(Path(record.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)

    assert not any(n == "sqlite3" or n.startswith("sqlite3.") for n in names)
    assert not any("attestation.llm" in n for n in names)
