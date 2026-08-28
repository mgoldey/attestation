"""The arithmetic behind the printed precision/recall/AUC, on a
hand-computed confusion matrix. Model-free."""

import importlib.util
from pathlib import Path

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _eval():
    spec = importlib.util.spec_from_file_location("flows_eval", FLOWS / "persona_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reaction(item_id, verdict, confidence=5):
    return {"item_id": item_id, "verdict": verdict, "confidence": confidence}


def test_precision_recall_on_a_hand_computed_matrix():
    m = _eval()
    labels = {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}
    reactions = [
        _reaction(1, True),
        _reaction(2, True),
        _reaction(3, False),  # tp tp fn
        _reaction(4, True),
        _reaction(5, False),
        _reaction(6, False),  # fp tn tn
    ]
    out = m.score_verdicts(reactions, labels)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 2)
    assert out["precision"] == 2 / 3
    assert out["recall"] == 2 / 3
    assert out["n_scored"] == 6 and out["n_unsure"] == 0


def test_unsure_items_are_reported_not_dropped_silently():
    m = _eval()
    labels = {1: True, 2: False, 3: True}
    out = m.score_verdicts([_reaction(1, True), _reaction(2, False)], labels)
    assert out["n_scored"] == 2
    assert out["n_unsure"] == 1  # item 3 never got a verdict


def test_auc_is_none_when_confidence_never_varies():
    m = _eval()
    labels = {1: True, 2: False}
    out = m.score_verdicts([_reaction(1, True, 5), _reaction(2, True, 5)], labels)
    assert out["auc"] is None
    assert out["confidence_histogram"] == {5: 2}


def test_auc_rewards_confident_correct_verdicts():
    m = _eval()
    labels = {1: True, 2: True, 3: False, 4: False}
    perfect = [
        _reaction(1, True, 5),
        _reaction(2, True, 4),
        _reaction(3, False, 4),
        _reaction(4, False, 5),
    ]
    assert m.score_verdicts(perfect, labels)["auc"] == 1.0


def test_rank_auc_over_an_ordering():
    m = _eval()
    labels = {10: True, 11: True, 12: False, 13: False}
    assert m.rank_auc([10, 11, 12, 13], labels) == 1.0
    assert m.rank_auc([12, 13, 10, 11], labels) == 0.0
    assert m.rank_auc([10, 11], {10: True, 11: True}) is None
