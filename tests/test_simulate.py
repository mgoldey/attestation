"""Simulated reader reactions.

The property that matters: these labels must be independent of the embedding.
`bootstrap_persona` labels by a linear threshold on the same vector the
classifier trains on, so evaluation over those rows is a tautology and
`evaluate_user` excludes them. Reactions here are produced by a chat model
reading text, which is what makes them worth training on -- and they carry
their own source so a future evaluation can make the same choice knowingly.
"""

import pytest

from attestation import simulate
from attestation.db import get_db
from attestation.rank import get_user


@pytest.fixture
def seeded(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'protein folding')")
    for i in range(4):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (f"Item {i}", f"Summary {i}", f"h{i}"),
        )
    conn.commit()
    return conn


def _items(conn):
    return conn.execute("SELECT id, title, summary FROM items ORDER BY id").fetchall()


def _chat(verdicts):
    """A chat_fn returning a scripted reaction per call."""
    seq = iter(verdicts)

    def chat_fn(messages, schema):
        verdict, strength = next(seq)
        return {
            "reasoning": "because of the subject matter",
            "verdict": verdict,
            "confidence": strength,
        }

    return chat_fn


def test_negatives_are_recorded_so_the_classifier_can_train(seeded):
    """The whole point. A single-class history keeps classifier_probs at None,
    so the ranker never learns -- which is the state of every real account."""
    uid = get_user(seeded, "ana")["id"]
    assert not simulate.classifier_would_train(seeded, uid)

    out = simulate.simulate_feedback(
        seeded, _chat([(True, 5), (False, 4), (True, 4), (False, 5)]), "ana", _items(seeded)
    )

    assert out["counts"]["useful"] == 2
    assert out["counts"]["not_useful"] == 2
    assert simulate.classifier_would_train(seeded, uid), "still single-class after simulation"


def test_rows_are_tagged_simulated_not_ui(seeded):
    """A simulated reader must never be mistaken for a person. bootstrap rows
    are excluded from evaluation for being tautological; these need the same
    ability to be told apart."""
    simulate.simulate_feedback(seeded, _chat([(True, 4)] * 4), "ana", _items(seeded))
    sources = {r["source"] for r in seeded.execute("SELECT DISTINCT source FROM clicks")}
    assert sources == {"simulated"}


def test_unsure_verdicts_are_dropped_not_recorded(seeded):
    """Indifference recorded as a verdict is noise the classifier must fit."""
    out = simulate.simulate_feedback(
        seeded, _chat([(True, 1), (False, 2), (True, 5), (False, 4)]), "ana", _items(seeded)
    )
    assert out["counts"]["skipped_unsure"] == 2
    assert seeded.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 2


def test_an_unusable_reply_costs_one_item_not_the_run(seeded):
    """And must never become a silent useful=False -- that would train the
    ranker on the model's formatting problems rather than its judgement."""
    calls = {"n": 0}

    def flaky(messages, schema):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"reasoning": "", "verdict": "maybe"}  # unparseable
        return {"reasoning": "fits my work", "verdict": True, "confidence": 4}

    out = simulate.simulate_feedback(seeded, flaky, "ana", _items(seeded))
    assert out["counts"]["failed"] == 1
    assert out["counts"]["useful"] == 3
    assert out["counts"]["not_useful"] == 0, "a parse failure must not become a negative"


def test_reactions_carry_their_reasoning(seeded):
    """A verdict with no reasoning is unreviewable, and these are training
    labels a human should be able to audit."""
    out = simulate.simulate_feedback(seeded, _chat([(False, 5)] * 4), "ana", _items(seeded))
    assert all(r["reasoning"] for r in out["reactions"])
    assert all(r["verdict"] is False for r in out["reactions"])


def test_unknown_user_is_refused(seeded):
    with pytest.raises(ValueError, match="unknown user"):
        simulate.simulate_feedback(seeded, _chat([(True, 4)]), "nobody", _items(seeded))


def test_reacting_twice_replaces_rather_than_duplicates(seeded):
    """record_click is INSERT OR REPLACE, so a re-run updates a verdict."""
    items = _items(seeded)
    simulate.simulate_feedback(seeded, _chat([(True, 4)] * 4), "ana", items)
    simulate.simulate_feedback(seeded, _chat([(False, 5)] * 4), "ana", items)
    rows = seeded.execute("SELECT useful FROM clicks").fetchall()
    assert len(rows) == 4
    assert all(r["useful"] == 0 for r in rows)
