#!/usr/bin/env python
"""Score a tagging prompt against a live model.

Prompt quality here is not a matter of taste. The failure modes are recorded
in the code: `NON_TOPIC_TAGS` exists because `nature`, `science-feed` and
`retraction` were pulled into the `biology` cluster, making a feed look like a
research area; `kg.health`'s `singleton_rate` caught one model minting 85%
one-off tags. Each case below targets one of those.

Not a pytest test: it needs a live model, takes minutes, and is
non-deterministic. It is a measurement tool, run deliberately.

    uv run python evals/run_tagging_eval.py                    # current prompt
    uv run python evals/run_tagging_eval.py --variant strict   # a candidate
    uv run python evals/run_tagging_eval.py --repeat 3         # stability
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from attestation.features import NON_TOPIC_TAGS, ItemTags  # noqa: E402
from attestation.llm import ChatClient, chat_model  # noqa: E402

CASES = json.loads((pathlib.Path(__file__).parent / "tagging_cases.json").read_text())

# A shared vocabulary, as a real run would have. `vocab_should_reuse` cases
# check the model prefers these over minting a synonym.
VOCAB = [
    "machine-learning", "transformers", "attention", "language-models",
    "retrieval", "reinforcement-learning", "computer-vision", "biology",
    "chemistry", "structural-biology", "cryo-em", "genomics", "crispr",
    "materials-science", "superconductivity", "gpu-programming", "cuda",
    "distributed-training", "scaling-laws", "evaluation", "long-context",
]


def build_messages(variant: str, title: str, summary: str, vocab: list[str]) -> list[dict]:
    vocab_line = ", ".join(vocab) if vocab else "(none yet)"

    if variant == "current":
        system = (
            "You label science-feed items. Reply with JSON: content_type"
            " (one of paper, survey, announcement, release, blog, other)"
            " and tags: 1-4 short lowercase-hyphenated topic tags."
            " Strongly prefer tags from the existing vocabulary; invent a"
            " new tag only if nothing in it fits."
            " Tag the SUBJECT MATTER only. Never tag where the item came"
            " from (nature, arxiv, science-feed) or what kind of post it"
            " is (release, announcement, newsletter) -- the publication is"
            " already recorded, and the kind of post is content_type."
        )
    elif variant == "strict":
        # Same rules, but the negative constraint is shown rather than
        # described. A small model follows an example better than a rule.
        system = (
            "You label science-feed items. Reply with JSON: content_type"
            " (one of paper, survey, announcement, release, blog, other)"
            " and tags: 1-4 short lowercase-hyphenated topic tags naming the"
            " SUBJECT MATTER.\n"
            "Prefer a tag from the existing vocabulary over a new synonym.\n"
            "A tag names what the work is ABOUT, never where it appeared or"
            " what kind of post it is.\n"
            'Example: "Retraction: superconductivity in lutetium hydride"'
            ' published in Nature ->'
            ' {"content_type": "announcement",'
            ' "tags": ["superconductivity", "materials-science"]}\n'
            "  -- not `retraction` (that is content_type), not `nature`"
            " (that is the publication).\n"
            "If an item is genuinely about no subject, return its best single"
            " tag anyway; content_type carries the rest."
        )
    elif variant == "terse":
        # Shortest thing that could work -- is the long prompt earning its
        # tokens on a 2B model?
        system = (
            "Label this science item. JSON: content_type (paper|survey|"
            "announcement|release|blog|other) and tags: 1-4 lowercase-hyphenated"
            " subject-matter topics. Reuse the given vocabulary where it fits."
            " Never use the publication name or the post type as a tag."
        )
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Existing vocabulary: {vocab_line}\n\nTitle: {title}\nSummary: {summary}",
        },
    ]


def score_one(case: dict, out: dict) -> dict:
    """Score one response. Every check maps to a real failure, not a preference."""
    result = {"id": case["id"], "errors": []}
    try:
        parsed = ItemTags.model_validate(out)
    except Exception as exc:
        result["errors"].append(f"validation failed ({exc.__class__.__name__}) -- item lost")
        result["tags"] = out.get("tags")
        result["content_type"] = out.get("content_type")
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

    # The trap: a source name or post type used as a topic. The validator
    # strips NON_TOPIC_TAGS, so seeing one here means it was emitted and
    # silently dropped -- a wasted slot out of four.
    total += 1
    raw = [t.strip().lower().replace(" ", "-") for t in out.get("tags", [])]
    leaked = [t for t in raw if t in NON_TOPIC_TAGS or t in case.get("must_not", [])]
    if leaked:
        result["errors"].append(f"emitted non-topic tags {leaked} (dropped, slot wasted)")
    else:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current", choices=["current", "strict", "terse"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per case, for stability")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = ChatClient(model=args.model)
    model = args.model or chat_model()
    print(f"variant={args.variant}  model={model}  repeat={args.repeat}\n")

    schema = ItemTags.model_json_schema()
    per_case, latencies, all_tags = {}, [], []

    for case in CASES:
        runs = []
        for _ in range(args.repeat):
            messages = build_messages(args.variant, case["title"], case["summary"], VOCAB)
            t0 = time.perf_counter()
            try:
                out = client.chat_json(messages, schema)
            except Exception as exc:
                out = {"_transport_error": str(exc)}
            latencies.append(time.perf_counter() - t0)
            scored = score_one(case, out)
            runs.append(scored)
            all_tags.extend(scored.get("tags") or [])

        best = statistics.mean(r["score"] for r in runs)
        per_case[case["id"]] = best
        flag = "ok " if best == 1.0 else ("~  " if best >= 0.5 else "FAIL")
        print(f"  {flag} {case['id']:20s} {best:.2f}  {runs[-1].get('tags')}")
        for e in dict.fromkeys(e for r in runs for e in r["errors"]):
            print(f"         - {e}")

    overall = statistics.mean(per_case.values())
    singles = sum(1 for t in set(all_tags) if all_tags.count(t) == 1)
    print(f"\n  overall           {overall:.3f}")
    print(f"  median latency    {statistics.median(latencies):.2f}s")
    print(f"  distinct tags     {len(set(all_tags))}")
    # kg.health's singleton_rate caught a model minting 85% one-off tags;
    # a prompt that invents a synonym per item fragments the graph.
    print(f"  singleton rate    {100 * singles / max(len(set(all_tags)), 1):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
