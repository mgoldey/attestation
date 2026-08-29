"""Scoring logic for the explanation prompt, model-free.

Mirrors tagging_eval.py: `evaluate()` renders every case through
`attestation.explain.explanation_messages`, the same function
`generate_explanation` calls, so a score here is a score of the prompt
`feed.explain` actually sends.

`EvalResult` and `gate` are imported from tagging_eval, not copied -- there
is one result shape and one acceptance rule, not one per task. No optimizer
exists for this prompt yet (see the task-corpora design doc); importing
`gate` now means one exists to call the day one lands.

Nothing here calls a model; run_explanation_eval.py is the only place a live
model is touched.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import time
from collections.abc import Callable

from tagging_eval import EvalResult, gate

from attestation.explain import Explanation, explanation_messages

__all__ = [
    "CASES_PATH",
    "SPLITS",
    "REFUSAL",
    "EvalResult",
    "gate",
    "load_cases",
    "score_one",
    "evaluate",
    "refusal_precision_recall",
    "dspy_fields",
    "to_dspy_example",
]

CASES_PATH = pathlib.Path(__file__).parent / "explanation_cases.json"
SPLITS = ("train", "dev")

# The mandated refusal string. Read from the production renderer's own system
# message rather than hard-coded a second time in explain.py, tests assert
# this equals explanation_messages("x", "y", "z")[0]["content"]'s wording --
# the prompt requires the model return these exact words when nothing shared.
REFUSAL = "Outside your stated interests."

_PREAMBLES = ("you will find", "this item")


def load_cases(path: pathlib.Path = CASES_PATH, split: str | None = None) -> list[dict]:
    cases = json.loads(path.read_text())
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        cases = [c for c in cases if c["split"] == split]
    return cases


def score_one(case: dict, out: dict) -> dict:
    """Score one response. Every check maps to wording the prompt mandates.

    Returns {"id", "score", "errors", "text"}. `errors` is prose because an
    optimizer would feed it back to a reflection model: a score alone says a
    candidate lost, the sentence says what it did.
    """
    result: dict = {"id": case["id"], "errors": []}
    try:
        parsed = Explanation.model_validate(out)
    except Exception as exc:  # noqa: BLE001 - a model returning an unparseable
        # shape IS the measurement here; crashing would lose the score for the
        # whole variant over one bad response.
        result["errors"].append(f"validation failed ({exc.__class__.__name__}) -- item lost")
        result["text"] = out.get("text") if isinstance(out, dict) else None
        result["score"] = 0.0
        return result

    text = parsed.text
    result["text"] = text

    if case["refuse"]:
        if text == REFUSAL:
            result["score"] = 1.0
        else:
            result["errors"].append(
                f"manufactured a connection instead of refusing -- expected the exact"
                f" {REFUSAL!r}, got {text!r}"
            )
            result["score"] = 0.0
        return result

    points, total = 0.0, 0.0
    words = text.split()

    total += 1
    wanted = case.get("must_mention_any") or []
    lowered = text.lower()
    if wanted and any(w.lower() in lowered for w in wanted):
        points += 1
    else:
        result["errors"].append(f"mentions none of {wanted} -- {text!r}")

    total += 1
    if len(words) < 15:
        points += 1
    else:
        result["errors"].append(f"{len(words)} words, expected under 15 -- {text!r}")

    total += 1
    if "you" in lowered.split() or "your" in lowered.split():
        points += 1
    else:
        result["errors"].append(f"does not address the reader as 'you' -- {text!r}")

    total += 1
    if not lowered.startswith(_PREAMBLES):
        points += 1
    else:
        result["errors"].append(f"opens with a preamble -- {text!r}")

    result["score"] = points / total if total else 1.0
    return result


def refusal_precision_recall(result: EvalResult, cases: list[dict]) -> dict:
    """Precision/recall of refusing when a case says `refuse: true`.

    "Refused" is read off the same exact-match the scorer uses (a text equal
    to REFUSAL), not off the case's expectation -- so a model that refuses a
    topic case it should have answered counts as a false refusal, not a free
    pass. A free function, not a method on `EvalResult`, because that class
    is imported from tagging_eval and shared across tasks.
    """
    by_id = {c["id"]: c for c in cases}
    tp = fp = fn = 0
    for case_id, runs in result.runs.items():
        case = by_id.get(case_id)
        if case is None:
            continue
        should = bool(case["refuse"])
        for r in runs:
            refused = r.get("text") == REFUSAL
            tp += refused and should
            fp += refused and not should
            fn += (not refused) and should
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def evaluate(
    chat_json: Callable[[list[dict], dict], dict],
    cases: list[dict],
    *,
    repeat: int = 1,
    on_case: Callable[[dict, float, list[dict]], None] | None = None,
) -> EvalResult:
    """Score `cases` through the production renderer.

    `chat_json(messages, schema) -> dict` is ChatClient.chat_json or a fake.
    A transport error costs one case, not the run: the point is the aggregate.
    """
    schema = Explanation.model_json_schema()
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    latencies: list[float] = []
    for case in cases:
        scored_runs = []
        for _ in range(repeat):
            messages = explanation_messages(case["interests"], case["title"], case["summary"])
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
    return EvalResult(per_case=per_case, runs=runs, latencies=latencies)


def dspy_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(input_names, output_names) so an optimizer can build a signature.

    Inputs match explanation_messages' parameters via the case's own key
    names (`interests` stands in for `profile`); outputs are the fields an
    optimizer's metric needs to score a prediction against a case.
    """
    return ("interests", "title", "summary"), ("refuse", "must_mention_any")


def to_dspy_example(case: dict) -> dict:
    """Inputs plus the case, ready for `dspy.Example(**to_dspy_example(c))`."""
    inputs, _ = dspy_fields()
    return {**{name: case[name] for name in inputs}, "case": case}
