"""The reaction eval harness's model-free half: scoring, the case file.

`evals/` is not a package; the modules there are scripts with an importable
core. These tests exercise that core so a scoring change cannot silently
re-weight every past measurement, and so an eval that stops rendering
through `simulate.reaction_messages` fails loudly (measurement-lessons.md
section 4: a guard that passes with the protected thing removed guards
nothing).
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import reaction_eval as re_  # noqa: E402

from attestation import simulate  # noqa: E402

CASE = {
    "id": "c",
    "split": "dev",
    "persona": "ana",
    "interests": "protein folding",
    "title": "A new method for cryo-EM structure refinement",
    "summary": "S",
    "verdict": True,
    "note": "hand case for score_one",
}


# --- score_one ---------------------------------------------------------


def test_a_correct_verdict_with_reasoning_naming_a_title_word_scores_one():
    out = {"reasoning": "squarely about cryo-em refinement", "verdict": True, "confidence": 5}
    r = re_.score_one(CASE, out)
    assert r["score"] == 1.0 and r["errors"] == []
    assert r["verdict"] is True and r["confidence"] == 5


def test_a_wrong_verdict_fails_the_verdict_check():
    out = {"reasoning": "squarely about cryo-em refinement", "verdict": False, "confidence": 4}
    r = re_.score_one(CASE, out)
    assert r["score"] == 0.5
    assert any("verdict False" in e for e in r["errors"])


def test_empty_reasoning_fails_the_reasoning_check():
    out = {"reasoning": "", "verdict": True, "confidence": 4}
    with pytest.raises(Exception):
        # Reaction requires min_length=1, so an empty reasoning does not even
        # validate -- this is the "invalid payload" path, exercised below by
        # score_one directly rather than raised here.
        simulate.Reaction.model_validate(out)
    r = re_.score_one(CASE, out)
    assert r["score"] == 0.0
    assert any("validation failed" in e for e in r["errors"])


def test_reasoning_that_does_not_mention_the_item_fails_that_check():
    out = {"reasoning": "sounds interesting I guess", "verdict": True, "confidence": 3}
    r = re_.score_one(CASE, out)
    assert r["score"] == 0.5
    assert any("does not mention the item's title" in e for e in r["errors"])


def test_an_invalid_payload_scores_zero_with_a_validation_error_in_prose():
    r = re_.score_one(CASE, {"_transport_error": "boom"})
    assert r["score"] == 0.0
    assert "validation failed" in r["errors"][0]
    assert r["verdict"] is None and r["confidence"] is None


# --- score_verdicts (moved verbatim; same hand-computed matrix as flows) ---


def _reaction(item_id, verdict, confidence=5):
    return {"item_id": item_id, "verdict": verdict, "confidence": confidence}


def test_score_verdicts_on_a_hand_computed_matrix():
    labels = {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}
    reactions = [
        _reaction(1, True),
        _reaction(2, True),
        _reaction(3, False),  # tp tp fn
        _reaction(4, True),
        _reaction(5, False),
        _reaction(6, False),  # fp tn tn
    ]
    out = re_.score_verdicts(reactions, labels)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 2)
    assert out["precision"] == 2 / 3
    assert out["recall"] == 2 / 3
    assert out["n_scored"] == 6 and out["n_unsure"] == 0


def test_score_verdicts_reports_unsure_items_not_dropping_them_silently():
    labels = {1: True, 2: False, 3: True}
    out = re_.score_verdicts([_reaction(1, True), _reaction(2, False)], labels)
    assert out["n_scored"] == 2
    assert out["n_unsure"] == 1


def test_score_verdicts_auc_is_none_when_confidence_never_varies():
    labels = {1: True, 2: False}
    out = re_.score_verdicts([_reaction(1, True, 5), _reaction(2, True, 5)], labels)
    assert out["auc"] is None
    assert out["confidence_histogram"] == {5: 2}


def test_rank_auc_over_an_ordering():
    labels = {10: True, 11: True, 12: False, 13: False}
    assert re_.rank_auc([10, 11, 12, 13], labels) == 1.0
    assert re_.rank_auc([12, 13, 10, 11], labels) == 0.0
    assert re_.rank_auc([10, 11], {10: True, 11: True}) is None


# --- the renderer mutation test ---------------------------------------------


def test_evaluate_renders_through_the_production_prompt(monkeypatch):
    """If reaction_eval stopped calling simulate.reaction_messages, this must
    fail: measurement-lessons.md section 4."""
    seen = []

    def fake_chat(messages, schema):
        seen.append(messages)
        return {"reasoning": "matches the title word", "verdict": True, "confidence": 5}

    monkeypatch.setattr(
        simulate, "reaction_messages", lambda *a: [{"role": "user", "content": "MUTATED"}]
    )
    re_.evaluate(fake_chat, [CASE])
    assert seen[0] == [{"role": "user", "content": "MUTATED"}]


def test_react_to_item_renders_through_the_same_function(monkeypatch):
    seen = []

    def fake_chat(messages, schema):
        seen.append(messages)
        return {"reasoning": "matches", "verdict": True, "confidence": 5}

    monkeypatch.setattr(
        simulate, "reaction_messages", lambda *a: [{"role": "user", "content": "MUTATED"}]
    )
    simulate.react_to_item(fake_chat, "ana", "protein folding", "T", "S")
    assert seen[0] == [{"role": "user", "content": "MUTATED"}]


def test_evaluate_aggregates_and_records_latency():
    def fake(messages, schema):
        return {"reasoning": "this is about cryo-em refinement", "verdict": True, "confidence": 5}

    r = re_.evaluate(fake, [CASE, dict(CASE, id="d", verdict=False)], repeat=2)
    assert r.per_case == {"c": 1.0, "d": 0.5}
    assert r.overall == pytest.approx(0.75)
    assert len(r.latencies) == 4


def test_evaluate_survives_a_transport_error_on_one_case():
    def flaky(messages, schema):
        if "flaky" in messages[-1]["content"]:
            raise ConnectionError("down")
        return {"reasoning": "this is about cryo-em refinement", "verdict": True, "confidence": 5}

    cases = [CASE, dict(CASE, id="f", title="flaky item")]
    r = re_.evaluate(flaky, cases)
    assert r.per_case == {"c": 1.0, "f": 0.0}


# --- dspy readiness ----------------------------------------------------------


def test_dspy_fields_names_the_reaction_signature():
    inputs, outputs = re_.dspy_fields()
    assert inputs == ("persona", "interests", "title", "summary")
    assert outputs == ("reasoning", "verdict", "confidence")


def test_to_dspy_example_carries_inputs_and_the_case():
    ex = re_.to_dspy_example(CASE)
    assert ex["persona"] == CASE["persona"]
    assert ex["interests"] == CASE["interests"]
    assert ex["title"] == CASE["title"]
    assert ex["summary"] == CASE["summary"]
    assert ex["case"] == CASE


# --- the case file -----------------------------------------------------------


CASES = re_.load_cases()


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_at_least_forty_cases():
    assert len(CASES) >= 40


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_case_is_well_formed(case):
    assert case["split"] in re_.SPLITS
    assert isinstance(case["verdict"], bool)
    assert case["persona"].strip()
    assert case["interests"].strip()
    assert case["title"].strip()
    assert isinstance(case["summary"], str)
    assert case.get("note", "").strip(), "a case must say which failure it targets"


def test_both_splits_are_large_enough_to_mean_something():
    counts = {s: len(re_.load_cases(split=s)) for s in re_.SPLITS}
    assert counts["train"] >= 20 and counts["dev"] >= 20, counts


def test_both_classes_are_present_in_each_split():
    for split in re_.SPLITS:
        verdicts = {c["verdict"] for c in re_.load_cases(split=split)}
        assert verdicts == {True, False}, (split, verdicts)


def test_dev_holds_every_bait_and_adjacent_case():
    for c in CASES:
        if c["id"].startswith("bait-") or c["id"].startswith("adjacent-"):
            assert c["split"] == "dev", c["id"]


def test_the_four_hard_families_are_all_present():
    prefixes = {"adjacent-", "bait-", "terse-", "crossover-"}
    for prefix in prefixes:
        matches = [c for c in CASES if c["id"].startswith(prefix)]
        assert len(matches) >= 1, prefix


def test_load_cases_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        re_.load_cases(split="test")


def test_the_case_file_is_valid_json_with_a_stable_key_order():
    text = (ROOT / "evals" / "reaction_cases.json").read_text()
    assert json.loads(text) == CASES


def test_fixture_derived_cases_name_their_source_item():
    fixture_cases = [c for c in CASES if c["id"].startswith("flows-")]
    assert len(fixture_cases) == 80
    for c in fixture_cases:
        assert "from examples/flows fixture item" in c["note"]


def test_a_fixture_items_two_personas_share_a_split():
    by_item = {}
    for c in CASES:
        if not c["id"].startswith("flows-"):
            continue
        guid = c["note"].rsplit(" ", 1)[-1]
        by_item.setdefault(guid, set()).add(c["split"])
    for guid, splits in by_item.items():
        assert len(splits) == 1, (guid, splits)
