"""The eval harness's model-free half: scoring, the case file, the renderer tie.

`evals/` is not a package; the modules there are scripts with an importable
core. These tests exercise that core so a scoring change cannot silently
re-weight every past measurement, and so the eval can never drift from the
prompt `feed.explain` actually sends.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "tests"))

import explanation_eval as ee  # noqa: E402
from conftest import seeded_db  # noqa: E402

from attestation.explain import explain, explanation_messages  # noqa: E402

TOPIC_CASE = {
    "id": "t",
    "split": "dev",
    "note": "n",
    "interests": "cryo-EM, protein structure",
    "title": "T",
    "summary": "S",
    "refuse": False,
    "must_mention_any": ["cryo-em", "protein structure"],
}

REFUSE_CASE = {
    "id": "r",
    "split": "dev",
    "note": "n",
    "interests": "cryo-EM, protein structure",
    "title": "T",
    "summary": "S",
    "refuse": True,
    "must_mention_any": [],
}


# --- REFUSAL must equal the prompt's own wording ----------------------------


def test_refusal_constant_is_read_from_the_production_prompt():
    """The constant and the prompt must not be able to drift independently."""
    content = explanation_messages("x", "y", "z")[0]["content"]
    assert ee.REFUSAL in content


# --- score_one: refusal cases -----------------------------------------------


def test_a_refusal_case_with_the_exact_refusal_scores_one():
    r = ee.score_one(REFUSE_CASE, {"text": ee.REFUSAL})
    assert r["score"] == 1.0 and r["errors"] == []


def test_a_refusal_case_with_a_manufactured_connection_fails():
    r = ee.score_one(REFUSE_CASE, {"text": "You share an interest in scientific methodology."})
    assert r["score"] == 0.0
    assert any("manufactured a connection" in e for e in r["errors"])


# --- score_one: topic cases --------------------------------------------------


def test_a_good_topic_answer_scores_one():
    r = ee.score_one(TOPIC_CASE, {"text": "You follow cryo-EM closely."})
    assert r["score"] == 1.0 and r["errors"] == []


def test_a_long_answer_with_a_preamble_fails_two_checks():
    """26 words opening 'You will find' -- the measured pre-fix failure."""
    long_answer = (
        "You will find that this item touches on cryo-EM and several adjacent"
        " areas of structural biology and imaging methodology that may be of"
        " some passing interest to your broader research programme overall."
    )
    assert len(long_answer.split()) >= 15
    r = ee.score_one(TOPIC_CASE, {"text": long_answer})
    assert r["score"] == pytest.approx(0.5)  # mentions + "you" pass; length + preamble fail
    joined = " ".join(r["errors"])
    assert "expected under 15" in joined
    assert "opens with a preamble" in joined


def test_a_topic_answer_mentioning_nothing_expected_fails():
    r = ee.score_one(TOPIC_CASE, {"text": "You might find this generally interesting."})
    assert any("mentions none of" in e for e in r["errors"])


def test_a_topic_answer_not_addressing_the_reader_fails():
    r = ee.score_one(TOPIC_CASE, {"text": "Relevant to cryo-EM work in general."})
    assert any("does not address the reader" in e for e in r["errors"])


def test_an_unparseable_reply_scores_zero_not_a_crash():
    r = ee.score_one(TOPIC_CASE, {"_transport_error": "boom"})
    assert r["score"] == 0.0 and "item lost" in r["errors"][0]
    r = ee.score_one(TOPIC_CASE, "not even a dict")
    assert r["score"] == 0.0


# --- the case file -----------------------------------------------------------


CASES = ee.load_cases()


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_at_least_forty_cases():
    assert len(CASES) >= 40


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_case_is_well_formed(case):
    assert case["split"] in ee.SPLITS
    assert case["note"].strip(), "a case must say which failure it targets"
    assert case["title"].strip()
    assert isinstance(case["summary"], str)
    assert isinstance(case["interests"], str) and case["interests"].strip()
    assert isinstance(case["refuse"], bool)
    if case["refuse"]:
        assert case.get("must_mention_any") == []
    else:
        assert case.get("must_mention_any"), "a topic case must name words to mention"


def test_both_splits_are_present_with_both_classes():
    for split in ee.SPLITS:
        cases = ee.load_cases(split=split)
        assert len(cases) >= 5, (split, len(cases))
        refuse_values = {c["refuse"] for c in cases}
        assert refuse_values == {True, False}, (split, refuse_values)


def test_dev_holds_at_least_half_the_refusals():
    refusals = [c for c in CASES if c["refuse"]]
    dev_refusals = [c for c in refusals if c["split"] == "dev"]
    assert len(dev_refusals) * 2 >= len(refusals), (len(dev_refusals), len(refusals))


def test_load_cases_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        ee.load_cases(split="test")


def test_the_case_file_is_valid_json_with_a_stable_key_order():
    text = (ROOT / "evals" / "explanation_cases.json").read_text()
    assert json.loads(text) == CASES


# --- evaluate() through a fake model ----------------------------------------


def test_evaluate_renders_through_the_production_prompt_and_aggregates():
    seen = []

    def fake(messages, schema):
        seen.append(messages)
        return {"text": "You follow cryo-em closely."}

    r = ee.evaluate(fake, [TOPIC_CASE, dict(TOPIC_CASE, id="t2")], repeat=2)
    assert seen[0][0]["content"] == explanation_messages("x", "y", "z")[0]["content"]
    assert (
        seen[0][-1]["content"]
        == explanation_messages(
            TOPIC_CASE["interests"], TOPIC_CASE["title"], TOPIC_CASE["summary"]
        )[-1]["content"]
    )
    assert r.per_case == {"t": 1.0, "t2": 1.0}
    assert r.overall == pytest.approx(1.0)
    assert len(r.latencies) == 4


def test_evaluate_survives_a_transport_error_on_one_case():
    def flaky(messages, schema):
        if "flaky" in messages[-1]["content"]:
            raise ConnectionError("down")
        return {"text": ee.REFUSAL}

    cases = [REFUSE_CASE, dict(REFUSE_CASE, id="f", title="flaky")]
    r = ee.evaluate(flaky, cases)
    assert r.per_case == {"r": 1.0, "f": 0.0}


def test_refusal_precision_recall():
    def fake(messages, schema):
        # refuses everything -- one false refusal, one correct refusal
        return {"text": ee.REFUSAL}

    cases = [TOPIC_CASE, REFUSE_CASE]
    r = ee.evaluate(fake, cases)
    rp = r.refusal_precision_recall(cases)
    assert rp == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0}


# --- the renderer mutation test: the eval must call the production builder --


def test_the_eval_renders_through_explanation_messages_not_a_private_copy(monkeypatch):
    """measurement-lessons.md sec.4: a guard that passes with the protected
    thing removed guards nothing. Mutating explanation_messages' system
    prompt must change what evaluate() sends."""
    import attestation.explain as explain_mod

    original = explain_mod.explanation_messages

    def mutated(profile, title, summary):
        msgs = original(profile, title, summary)
        msgs[0]["content"] = "MUTATED SYSTEM PROMPT"
        return msgs

    monkeypatch.setattr(ee, "explanation_messages", mutated)

    seen = []

    def fake(messages, schema):
        seen.append(messages)
        return {"text": ee.REFUSAL}

    ee.evaluate(fake, [REFUSE_CASE])
    assert seen[0][0]["content"] == "MUTATED SYSTEM PROMPT"


def test_explain_renders_through_explanation_messages(monkeypatch, tmp_path):
    """Drives the real public entry point (`explain.explain`) over a seeded
    tmp DB with a fake chat_fn, and asserts the graph's own message content
    matches explanation_messages' output for the same inputs -- so the
    production graph and the eval score the same prompt."""
    conn = seeded_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO items(feed_id, title, summary, content_hash)"
        " VALUES (NULL, 'A paper on cryo-EM', 'about protein structure', 'h1')"
    )
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    conn.commit()

    seen = []

    def chat_fn(messages, schema):
        seen.append(messages)
        return {"text": "You follow cryo-EM closely."}

    text = explain(conn, user_id=1, item_id=1, chat_fn=chat_fn)
    assert text == "You follow cryo-EM closely."

    row = conn.execute("SELECT interests FROM users WHERE id = 1").fetchone()
    expected = explanation_messages(
        row["interests"], "A paper on cryo-EM", "about protein structure"
    )
    assert seen[-1] == expected

    # And mutating the production renderer changes what the graph itself sends.
    import attestation.explain as explain_mod

    def mutated(profile, title, summary):
        return [{"role": "system", "content": "MUTATED"}, {"role": "user", "content": "u"}]

    monkeypatch.setattr(explain_mod, "explanation_messages", mutated)
    conn.execute("DELETE FROM explanations")
    conn.commit()
    seen.clear()
    explain(conn, user_id=1, item_id=1, chat_fn=chat_fn)
    assert seen[-1][0]["content"] == "MUTATED"


# --- dspy readiness without dspy --------------------------------------------


def test_dspy_fields_and_to_dspy_example():
    inputs, outputs = ee.dspy_fields()
    assert inputs == ("interests", "title", "summary")
    assert "refuse" in outputs
    example = ee.to_dspy_example(TOPIC_CASE)
    assert example["interests"] == TOPIC_CASE["interests"]
    assert example["title"] == TOPIC_CASE["title"]
    assert example["summary"] == TOPIC_CASE["summary"]
    assert example["case"] is TOPIC_CASE


# --- nothing under evals/ (other than optimize_tagging.py) imports dspy -----


def test_no_new_eval_module_imports_dspy():
    import ast

    tree = ast.parse((ROOT / "evals" / "explanation_eval.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {n.name for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(n == "dspy" or n.startswith("dspy.") for n in names)
