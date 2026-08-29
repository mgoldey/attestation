"""Scoring and acceptance logic for the reaction prompt, model-free.

Mirrors `tagging_eval.py`: `load_cases`, `score_one`, `evaluate` and `gate`
(the last imported, not copied, so there is one gate). Nothing here calls a
model, so all of it is unit-tested; `run_reaction_eval.py` beside it is the
only place a live model is touched.

Prompt rendering is NOT here: `attestation.simulate.reaction_messages` is
the one renderer, so every score in this directory is a score of the prompt
`feed.simulate_ratings` would actually send.

`score_verdicts` and `rank_auc` were moved here verbatim from
`examples/flows/persona_eval.py`, which now imports them from here: one
definition of the confusion-matrix/precision/recall/AUC arithmetic, used by
both the flow and this eval.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable

from attestation import simulate
from attestation.simulate import Reaction

# evals/ is not a package (see tagging_eval.py's own docstring): every script
# here puts this directory on sys.path and imports its modules top-level, so
# `gate`/`spread`/`EvalResult` -- the acceptance logic -- stay one definition
# rather than a copy per task.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tagging_eval import EvalResult, gate, spread  # re-exported

CASES_PATH = pathlib.Path(__file__).parent / "reaction_cases.json"
SPLITS = ("train", "dev")

_WORD = re.compile(r"[a-z0-9]+")


def load_cases(path: pathlib.Path = CASES_PATH, split: str | None = None) -> list[dict]:
    cases = json.loads(path.read_text())
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        cases = [c for c in cases if c["split"] == split]
    return cases


def _title_words(title: str) -> set[str]:
    return {w for w in _WORD.findall(title.lower()) if len(w) > 3}


def score_one(case: dict, out: dict) -> dict:
    """Score one response. Every check maps to a real failure, not a preference.

    Returns {"id", "score", "errors", "verdict", "confidence"}. `errors` is
    prose because a future optimizer feeds it back to a reflection model: a
    score alone says a candidate lost, the sentence says what it did.
    """
    result: dict = {"id": case["id"], "errors": []}
    try:
        parsed = Reaction.model_validate(out)
    except Exception as exc:  # noqa: BLE001 - a model returning an unparseable
        # shape IS the measurement here; crashing would lose the score for the
        # whole variant over one bad response.
        result["errors"].append(f"validation failed ({exc.__class__.__name__}) -- item lost")
        result["verdict"] = None
        result["confidence"] = None
        result["score"] = 0.0
        return result

    result["verdict"] = parsed.verdict
    result["confidence"] = parsed.confidence

    points, total = 0.0, 0.0

    total += 1
    if parsed.verdict == case["verdict"]:
        points += 1
    else:
        result["errors"].append(f"verdict {parsed.verdict!r}, expected {case['verdict']!r}")

    total += 1
    reasoning = parsed.reasoning.strip()
    title_words = _title_words(case["title"])
    reasoning_words = _title_words(reasoning)
    if not reasoning:
        result["errors"].append("reasoning is empty -- a verdict with no reason is unreviewable")
    elif not (title_words & reasoning_words):
        result["errors"].append(f"reasoning {reasoning!r} does not mention the item's title")
    else:
        points += 1

    result["score"] = points / total if total else 1.0
    return result


def score_verdicts(reactions: list[dict], labels: dict[int | str, bool]) -> dict:
    """Confusion matrix, precision, recall, and AUC of a signed confidence.

    Moved verbatim from `examples/flows/persona_eval.py` (same keys); that
    module now imports this function so there is one definition.

    Items in `labels` with no reaction were skipped as unsure by the model;
    they are counted in n_unsure and excluded from the matrix, never from
    the report.
    """
    by_item = {r["item_id"]: r for r in reactions if r["item_id"] in labels}
    tp = fp = fn = tn = 0
    scores, truth = [], []
    for item_id, label in labels.items():
        r = by_item.get(item_id)
        if r is None:
            continue
        verdict = bool(r["verdict"])
        tp += verdict and label
        fp += verdict and not label
        fn += (not verdict) and label
        tn += (not verdict) and not label
        scores.append(r["confidence"] if verdict else -r["confidence"])
        truth.append(label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    auc = None
    if len(set(scores)) > 1 and len(set(truth)) > 1:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(truth, scores))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_scored": len(scores),
        "n_unsure": len(labels) - len(scores),
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "confidence_histogram": dict(
            sorted(Counter(r["confidence"] for r in by_item.values()).items())
        ),
    }


def rank_auc(order: list[int | str], labels: dict[int | str, bool]) -> float | None:
    """AUC of a ranking: earlier = higher score. None on a single class.

    Moved verbatim from `examples/flows/persona_eval.py`.
    """
    truth = [labels[i] for i in order if i in labels]
    if len(set(truth)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    scores = [len(order) - pos for pos, i in enumerate(order) if i in labels]
    return float(roc_auc_score(truth, scores))


def evaluate(
    chat_json: Callable[[list[dict], dict], dict],
    cases: list[dict],
    *,
    repeat: int = 1,
    on_case: Callable[[dict, float, list[dict]], None] | None = None,
) -> EvalResult:
    """Score the reaction prompt on `cases` through the production renderer.

    `chat_json(messages, schema) -> dict` is ChatClient.chat_json or a fake.
    A transport error costs one case, not the run: the point is the aggregate.
    """
    schema = Reaction.model_json_schema()
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    latencies: list[float] = []
    all_tags: list[str] = []  # unused for reaction; EvalResult expects the field
    for case in cases:
        scored_runs = []
        for _ in range(repeat):
            # Through the module, not a bound name imported at load time, so
            # monkeypatching simulate.reaction_messages (the renderer
            # mutation test) actually changes what gets sent.
            messages = simulate.reaction_messages(
                case["persona"], case["interests"], case["title"], case["summary"]
            )
            t0 = time.perf_counter()
            try:
                out = chat_json(messages, schema)
            except Exception as exc:  # noqa: BLE001 - see docstring
                out = {"_transport_error": str(exc)}
            latencies.append(time.perf_counter() - t0)
            scored = score_one(case, out)
            scored_runs.append(scored)
        mean = statistics.mean(r["score"] for r in scored_runs)
        per_case[case["id"]] = mean
        runs[case["id"]] = scored_runs
        if on_case:
            on_case(case, mean, scored_runs)
    return EvalResult(per_case=per_case, runs=runs, latencies=latencies, tags=all_tags)


def dspy_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(input field names, output field names) for a DSPy signature."""
    return (("persona", "interests", "title", "summary"), ("reasoning", "verdict", "confidence"))


def to_dspy_example(case: dict) -> dict:
    """Inputs plus the case, ready for `dspy.Example(**to_dspy_example(c))`."""
    inputs, _ = dspy_fields()
    return {**{k: case[k] for k in inputs}, "case": case}


__all__ = [
    "CASES_PATH",
    "SPLITS",
    "load_cases",
    "score_one",
    "score_verdicts",
    "rank_auc",
    "evaluate",
    "dspy_fields",
    "to_dspy_example",
    "gate",
    "spread",
    "EvalResult",
]
