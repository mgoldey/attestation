"""Scoring and acceptance logic for the tagging prompt, model-free.

Shared by run_tagging_eval.py (score one prompt), optimize_tagging.py (the
metric the optimizer climbs) and transfer_matrix.py (the acceptance gate).
Nothing here calls a model, so all of it is unit-tested; the scripts beside
it are the only places a live model is touched.

Prompt rendering is NOT here: `attestation.features.tag_messages` is the one
renderer, so every score in this directory is a score of the prompt that
`attest tag` would actually send.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import time
from collections.abc import Callable

from attestation.features import NON_TOPIC_TAGS, ItemTags, TagPrompt, tag_messages

CASES_PATH = pathlib.Path(__file__).parent / "tagging_cases.json"
SPLITS = ("train", "dev")

# The top of the LIVE vocabulary (tag_vocabulary(conn, 40) on 2026-08-27,
# 6488 items). Frozen here so the eval is reproducible, and taken from the
# real corpus rather than invented because the failure it exposes depends on
# the real shape: generic tags at the top -- optimization, machine-learning,
# representation-learning, evaluation-metrics -- that a model told to
# "strongly prefer the vocabulary" attaches to a physics paper about the
# modular bootstrap. An invented vocabulary of tidy topics never shows that.
VOCAB = [
    "optimization",
    "machine-learning",
    "large-language-models",
    "representation-learning",
    "evaluation-metrics",
    "reasoning",
    "reinforcement-learning",
    "agentic-workflows",
    "hugging-face",
    "transformers",
    "deep-learning",
    "time-series",
    "generative-ai",
    "inference",
    "natural-language-processing",
    "ai",
    "fine-tuning",
    "biology",
    "vision-language-models",
    "modeling",
    "forecasting",
    "diffusion-models",
    "agent-based-modeling",
    "transfer-learning",
    "safety",
    "graph-neural-networks",
    "multimodal",
    "causal-inference",
    "attention-mechanisms",
    "alignment",
    "quantum-chemistry",
    "security",
    "open-source",
    "computer-vision",
    "neural-networks",
    "robotics",
    "federated-learning",
    "hardware",
    "simulation",
    "physics",
]


def load_cases(path: pathlib.Path = CASES_PATH, split: str | None = None) -> list[dict]:
    cases = json.loads(path.read_text())
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        cases = [c for c in cases if c["split"] == split]
    return cases


def _normalize(tags: list) -> list[str]:
    return [str(t).strip().lower().replace(" ", "-") for t in tags]


def score_one(case: dict, out: dict) -> dict:
    """Score one response. Every check maps to a real failure, not a preference.

    Returns {"id", "score", "errors", "tags", "content_type"}. `errors` is
    prose because the optimizer feeds it back to the reflection model: a
    score alone says a candidate lost, the sentence says what it did.
    """
    result: dict = {"id": case["id"], "errors": []}
    try:
        parsed = ItemTags.model_validate(out)
    except Exception as exc:  # noqa: BLE001 - a model returning an unparseable
        # shape IS the measurement here; crashing would lose the score for the
        # whole variant over one bad response.
        result["errors"].append(f"validation failed ({exc.__class__.__name__}) -- item lost")
        result["tags"] = out.get("tags") if isinstance(out, dict) else None
        result["content_type"] = out.get("content_type") if isinstance(out, dict) else None
        result["score"] = 0.0
        return result

    tags = parsed.tags
    result["tags"] = tags
    result["content_type"] = parsed.content_type

    points, total = 0.0, 0.0

    total += 1
    if parsed.content_type == case["content_type"]:
        points += 1
    else:
        result["errors"].append(
            f"content_type {parsed.content_type!r}, expected {case['content_type']!r}"
        )

    # Two traps share a check but fail differently. A source name or post
    # type (NON_TOPIC_TAGS) is stripped by the validator: a wasted slot. A
    # plausible-but-wrong vocabulary tag -- `optimization` on a CFT paper --
    # is NOT stripped: it enters the graph and pulls the item into the wrong
    # cluster, which is the failure the live corpus actually shows.
    total += 1
    raw = _normalize(out.get("tags", []))
    non_topic = [t for t in raw if t in NON_TOPIC_TAGS]
    wrong = [t for t in raw if t in case.get("must_not", []) and t not in NON_TOPIC_TAGS]
    if non_topic:
        result["errors"].append(f"emitted non-topic tags {non_topic} (dropped, slot wasted)")
    if wrong:
        result["errors"].append(f"tagged {wrong}, which this item is not about")
    if not non_topic and not wrong:
        points += 1

    wanted = case.get("should_include_any") or []
    if wanted:
        total += 1
        if any(w in tags for w in wanted):
            points += 1
        else:
            result["errors"].append(f"no expected topic in {tags}; wanted any of {wanted}")

    reuse = case.get("vocab_should_reuse") or []
    if reuse:
        total += 1
        if any(r in tags for r in reuse):
            points += 1
        else:
            result["errors"].append(f"minted new tags {tags} instead of reusing {reuse}")

    result["score"] = points / total if total else 1.0
    return result


@dataclasses.dataclass
class EvalResult:
    per_case: dict[str, float]
    runs: dict[str, list[dict]]
    latencies: list[float]
    tags: list[str]

    @property
    def overall(self) -> float:
        return statistics.mean(self.per_case.values()) if self.per_case else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def singleton_rate(self) -> float:
        # kg.health's singleton_rate caught a model minting 85% one-off tags;
        # a prompt that invents a synonym per item fragments the graph.
        distinct = set(self.tags)
        singles = sum(1 for t in distinct if self.tags.count(t) == 1)
        return singles / max(len(distinct), 1)


def evaluate(
    chat_json: Callable[[list[dict], dict], dict],
    cases: list[dict],
    prompt: TagPrompt | None,
    *,
    vocab: list[str] = VOCAB,
    repeat: int = 1,
    on_case: Callable[[dict, float, list[dict]], None] | None = None,
) -> EvalResult:
    """Score `prompt` on `cases` through the production renderer.

    `chat_json(messages, schema) -> dict` is ChatClient.chat_json or a fake.
    A transport error costs one case, not the run: the point is the aggregate.
    """
    schema = ItemTags.model_json_schema()
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    latencies: list[float] = []
    all_tags: list[str] = []
    for case in cases:
        scored_runs = []
        for _ in range(repeat):
            messages = tag_messages(case["title"], case["summary"], vocab, prompt)
            t0 = time.perf_counter()
            try:
                out = chat_json(messages, schema)
            except Exception as exc:  # noqa: BLE001 - see docstring
                out = {"_transport_error": str(exc)}
            latencies.append(time.perf_counter() - t0)
            scored = score_one(case, out)
            scored_runs.append(scored)
            all_tags.extend(scored.get("tags") or [])
        mean = statistics.mean(r["score"] for r in scored_runs)
        per_case[case["id"]] = mean
        runs[case["id"]] = scored_runs
        if on_case:
            on_case(case, mean, scored_runs)
    return EvalResult(per_case=per_case, runs=runs, latencies=latencies, tags=all_tags)


@dataclasses.dataclass(frozen=True)
class Gate:
    """The acceptance test from the spec: transfer, not score."""

    passed: bool
    reasons: tuple[str, ...]
    spread: float
    baseline_spread: float


def spread(scores: dict[str, float]) -> float:
    return max(scores.values()) - min(scores.values()) if scores else 0.0


def gate(
    baseline: dict[str, float], candidate: dict[str, float], primary: str, *, min_others: int = 2
) -> Gate:
    """Decide whether `candidate` may ship, given per-model scores for both.

    1. Beats the baseline on `primary`, the model the optimizer ran on.
    2. Beats it on at least `min_others` other models.
    3. Its spread across models is no wider than the baseline's.

    Rule 3 is the one that makes this different from keeping the winner: a
    prompt scoring 0.95/0.94/0.60 has a better headline than 0.88/0.85/0.79
    and is disqualified, because it has been fitted to one model and the next
    backend swap silently degrades tagging with no error.
    """
    reasons: list[str] = []
    if primary not in baseline or primary not in candidate:
        raise ValueError(f"primary model {primary!r} missing from scores")
    models = sorted(set(baseline) & set(candidate))
    if candidate[primary] <= baseline[primary]:
        reasons.append(
            f"does not beat the baseline on {primary}"
            f" ({candidate[primary]:.3f} vs {baseline[primary]:.3f})"
        )
    others = [m for m in models if m != primary]
    won = [m for m in others if candidate[m] > baseline[m]]
    if len(won) < min_others:
        reasons.append(f"beats the baseline on {len(won)} other model(s) {won}; needs {min_others}")
    cand = {m: candidate[m] for m in models}
    base = {m: baseline[m] for m in models}
    s, bs = spread(cand), spread(base)
    if s > bs + 1e-9:
        reasons.append(f"spread across models {s:.3f} is wider than the baseline's {bs:.3f}")
    return Gate(passed=not reasons, reasons=tuple(reasons), spread=s, baseline_spread=bs)
