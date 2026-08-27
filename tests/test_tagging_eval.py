"""The eval harness's model-free half: scoring, the case file, the gate.

`evals/` is not a package; the modules there are scripts with an importable
core. These tests exercise that core so a scoring change cannot silently
re-weight every past measurement.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import tagging_eval as te  # noqa: E402

from attestation.features import CONTENT_TYPES, NON_TOPIC_TAGS, TagPrompt  # noqa: E402
from attestation.kg import canonical  # noqa: E402

CASE = {
    "id": "c",
    "split": "dev",
    "title": "T",
    "summary": "S",
    "content_type": "paper",
    "should_include_any": ["cryo-em"],
    "must_not": ["nature", "optimization"],
    "vocab_should_reuse": ["structural-biology"],
}


def test_a_perfect_answer_scores_one():
    out = {"content_type": "paper", "tags": ["cryo-em", "structural-biology"]}
    r = te.score_one(CASE, out)
    assert r["score"] == 1.0 and r["errors"] == []


def test_each_check_is_worth_one_point_and_names_its_failure():
    out = {"content_type": "blog", "tags": ["ribosome"]}
    r = te.score_one(CASE, out)
    assert r["score"] == 0.25  # only the must_not check passes
    joined = " ".join(r["errors"])
    assert "content_type 'blog', expected 'paper'" in joined
    assert "no expected topic" in joined
    assert "instead of reusing" in joined


def test_a_wrong_vocabulary_tag_is_reported_differently_from_a_non_topic_tag():
    """`optimization` on a CFT paper is not stripped by the validator -- it
    enters the graph. The message must say so, because the optimizer reads
    these messages as feedback."""
    r = te.score_one(CASE, {"content_type": "paper", "tags": ["cryo-em", "optimization"]})
    assert any("which this item is not about" in e for e in r["errors"])
    assert not any("slot wasted" in e for e in r["errors"])
    r2 = te.score_one(CASE, {"content_type": "paper", "tags": ["cryo-em", "nature"]})
    assert any("slot wasted" in e for e in r2["errors"])


def test_a_must_not_check_normalizes_case_and_spaces():
    r = te.score_one(CASE, {"content_type": "paper", "tags": ["cryo-em", " Optimization "]})
    assert any("which this item is not about" in e for e in r["errors"])


def test_an_unparseable_reply_scores_zero_not_a_crash():
    r = te.score_one(CASE, {"_transport_error": "boom"})
    assert r["score"] == 0.0 and "item lost" in r["errors"][0]
    r = te.score_one(CASE, "not even a dict")
    assert r["score"] == 0.0


def test_a_case_with_no_expected_topic_scores_only_type_and_traps():
    case = dict(CASE, should_include_any=[], vocab_should_reuse=[])
    r = te.score_one(case, {"content_type": "paper", "tags": ["anything"]})
    assert r["score"] == 1.0


# --- the case file ---------------------------------------------------------


CASES = te.load_cases()
TAG = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_case_is_well_formed(case):
    assert case["split"] in te.SPLITS
    assert case["content_type"] in CONTENT_TYPES
    assert case["title"].strip()
    assert isinstance(case["summary"], str)
    assert case.get("note", "").strip(), "a case must say which failure it targets"
    for key in ("should_include_any", "must_not", "vocab_should_reuse"):
        for tag in case.get(key) or []:
            assert TAG.match(tag), f"{key} tag {tag!r} is not a well-formed tag"
    # A reuse expectation the vocabulary cannot satisfy would fail every prompt.
    for tag in case.get("vocab_should_reuse") or []:
        assert tag in te.VOCAB, f"vocab_should_reuse names {tag!r}, not in VOCAB"
    # Only an item genuinely about nothing may leave should_include_any empty.
    if not case.get("should_include_any"):
        assert case["content_type"] == "other"


def test_both_splits_are_large_enough_to_mean_something():
    counts = {s: len(te.load_cases(split=s)) for s in te.SPLITS}
    assert counts["train"] >= 20 and counts["dev"] >= 20, counts


def test_the_original_ten_cases_stay_in_dev():
    """The 0.883 / 0.850 / 0.792 baseline in the spec was measured on these;
    keeping them out of training keeps that number comparable."""
    original = {
        "paper-lm",
        "survey-rag",
        "release-pytorch",
        "announcement-grant",
        "blog-debug",
        "trap-nature",
        "trap-retraction",
        "acronym-crispr",
        "vocab-reuse",
        "empty-ok",
    }
    dev = {c["id"] for c in te.load_cases(split="dev")}
    assert original <= dev


def test_the_bait_family_names_generic_vocabulary_tags():
    """The failure the live corpus shows: generic top-of-vocabulary tags on
    off-vocabulary items. Every bait case must forbid at least one."""
    baits = [c for c in CASES if c["id"].startswith("bait-")]
    assert len(baits) >= 8
    for c in baits:
        assert set(c["must_not"]) & set(te.VOCAB), c["id"]


def test_the_frozen_vocabulary_is_canonical_and_topical():
    assert len(te.VOCAB) == 40
    assert not set(te.VOCAB) & NON_TOPIC_TAGS
    assert [canonical(t) for t in te.VOCAB] == te.VOCAB
    assert len(set(te.VOCAB)) == 40


def test_load_cases_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        te.load_cases(split="test")


# --- evaluate() through a fake model ----------------------------------------


def test_evaluate_renders_through_the_production_prompt_and_aggregates():
    seen = []

    def fake(messages, schema):
        seen.append(messages)
        return {"content_type": "paper", "tags": ["cryo-em", "structural-biology"]}

    prompt = TagPrompt(instruction="INSTR")
    r = te.evaluate(fake, [CASE, dict(CASE, id="d", content_type="blog")], prompt, repeat=2)
    assert seen[0][0]["content"] == "INSTR"
    assert "Existing vocabulary: optimization, machine-learning" in seen[0][-1]["content"]
    assert r.per_case == {"c": 1.0, "d": 0.75}
    assert r.overall == pytest.approx(0.875)
    assert len(r.latencies) == 4


def test_evaluate_survives_a_transport_error_on_one_case():
    def flaky(messages, schema):
        if "T" in messages[-1]["content"] and "flaky" in messages[-1]["content"]:
            raise ConnectionError("down")
        return {"content_type": "paper", "tags": ["cryo-em", "structural-biology"]}

    cases = [CASE, dict(CASE, id="f", title="T flaky")]
    r = te.evaluate(flaky, cases, None)
    assert r.per_case == {"c": 1.0, "f": 0.0}


# --- the gate ----------------------------------------------------------------


BASE = {"gemma4:e2b": 0.883, "gemma4:e4b": 0.850, "hermes3:8b": 0.792}


def test_a_candidate_that_transfers_passes():
    g = te.gate(BASE, {"gemma4:e2b": 0.92, "gemma4:e4b": 0.90, "hermes3:8b": 0.85}, "gemma4:e2b")
    assert g.passed and g.reasons == ()


def test_a_better_headline_with_a_wider_spread_is_refused():
    """0.95/0.94/0.60 beats 0.883/0.850/0.792 on two models and is fitted to
    one -- the spec's worked example."""
    g = te.gate(BASE, {"gemma4:e2b": 0.95, "gemma4:e4b": 0.94, "hermes3:8b": 0.60}, "gemma4:e2b")
    assert not g.passed
    assert any("wider" in r for r in g.reasons)
    assert any("beats the baseline on 1 other" in r for r in g.reasons)


def test_not_beating_the_primary_model_is_reason_enough():
    g = te.gate(BASE, {"gemma4:e2b": 0.883, "gemma4:e4b": 0.90, "hermes3:8b": 0.85}, "gemma4:e2b")
    assert not g.passed and any("does not beat" in r for r in g.reasons)


def test_the_gate_only_compares_models_both_prompts_were_scored_on():
    g = te.gate(
        BASE, {"gemma4:e2b": 0.9, "gemma4:e4b": 0.9, "hermes3:8b": 0.85, "x": 0.1}, "gemma4:e2b"
    )
    assert g.passed


def test_the_gate_needs_the_primary_model_in_both():
    with pytest.raises(ValueError):
        te.gate(BASE, {"gemma4:e4b": 0.9}, "gemma4:e2b")


def test_the_spec_baseline_spread_is_the_number_recorded():
    assert te.spread(BASE) == pytest.approx(0.091)


def test_the_case_file_is_valid_json_with_a_stable_key_order():
    text = (ROOT / "evals" / "tagging_cases.json").read_text()
    assert json.loads(text) == CASES


# --- the optimizer's model-free parts (local only: dspy is not a CI dependency) ---


def test_transfer_matrix_markdown_carries_scores_spread_and_verdict():
    import transfer_matrix as tm

    scores = {
        "baseline": dict(BASE),
        "cand": {"gemma4:e2b": 0.9, "gemma4:e4b": 0.9, "hermes3:8b": 0.85},
    }
    verdicts = {"cand": te.gate(scores["baseline"], scores["cand"], "gemma4:e2b")}
    md = tm.render_markdown(list(BASE), scores, verdicts, "dev", 28, "gemma4:e2b", 2)
    assert "| baseline | 0.883 | 0.850 | 0.792 | 0.091 |" in md
    assert "| cand | 0.900 | 0.900 | 0.850 | 0.050 |" in md
    assert "**cand**: PASS" in md and "2 run(s)" in md


def test_the_optimizer_sends_the_production_prompt_and_scores_with_the_eval_metric():
    """The adapter must hand DSPy exactly what `attest tag` sends: the same
    system message, the same user turn. Otherwise GEPA tunes an instruction
    for a prompt the product never renders."""
    dspy = pytest.importorskip("dspy")
    import optimize_tagging as ot

    from attestation.features import DEFAULT_TAG_INSTRUCTION, tag_messages

    adapter = ot.build_adapter(dspy, te.VOCAB)
    program = ot.build_program(dspy)
    sig = program.signature
    assert sig.instructions == DEFAULT_TAG_INSTRUCTION
    rendered = adapter.format(sig, demos=[], inputs={"title": "T", "summary": "S"})
    assert rendered == tag_messages("T", "S", te.VOCAB)

    parsed = adapter.parse(sig, 'Sure: {"content_type": "paper", "tags": ["cryo-em"]} trailing')
    assert parsed == {"content_type": "paper", "tags": ["cryo-em"]}
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        adapter.parse(sig, "no object here")

    metric = ot.make_metric(dspy)
    gold = ot.to_examples(dspy, [CASE])[0]
    good = dspy.Prediction(content_type="paper", tags=["cryo-em", "structural-biology"])
    assert dict(metric(gold, good)) == {"score": 1.0, "feedback": "correct"}
    bad = dspy.Prediction(content_type="paper", tags=["cryo-em", "optimization"])
    out = metric(gold, bad)
    assert out.score < 1.0 and "which this item is not about" in out.feedback
    empty = dspy.Prediction()
    assert metric(gold, empty).score == 0.0


# transfer-<date>.json is a matrix's raw scores, not a prompt; everything else is.
ARTIFACTS = sorted(
    p for p in (ROOT / "evals" / "prompts").glob("*.json") if not p.name.startswith("transfer-")
)


@pytest.mark.parametrize("path", ARTIFACTS, ids=[p.name for p in ARTIFACTS])
def test_every_committed_prompt_artifact_loads(path):
    """An artifact that `load_tag_prompt` rejects is one `attest tag` would
    refuse at startup; committing it would ship a prompt nobody can use."""
    from attestation.features import load_tag_prompt

    prompt = load_tag_prompt(path)
    assert prompt.instruction.strip()
