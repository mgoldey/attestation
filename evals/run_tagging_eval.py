#!/usr/bin/env python
"""Score a tagging prompt against a live model.

Prompt quality here is not a matter of taste. The failure modes are recorded
in the cases: `NON_TOPIC_TAGS` exists because `nature`, `science-feed` and
`retraction` were pulled into the `biology` cluster; the `bait-*` family
exists because the live corpus tagged a conformal-field-theory paper
`deep-learning, machine-learning, optimization, representation-learning`.
Each case targets one such failure and says which in its `note`.

Not a pytest test: it needs a live model, takes minutes, and is
non-deterministic. It is a measurement tool, run deliberately.

    uv run python evals/run_tagging_eval.py                          # shipped default, dev split
    uv run python evals/run_tagging_eval.py --artifact evals/prompts/hand-written.json
    uv run python evals/run_tagging_eval.py --split all --repeat 3   # stability
    uv run python evals/run_tagging_eval.py --model hermes3:8b       # transfer

The prompt is rendered by `attestation.features.tag_messages`, the same
function `attest tag` uses, so the number here is the number in production.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tagging_eval import VOCAB, evaluate, load_cases

from attestation.features import load_tag_prompt
from attestation.llm import ChatClient, chat_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact",
        default=None,
        help="prompt artifact JSON; default = the shipped DEFAULT_TAG_INSTRUCTION",
    )
    ap.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per case, for stability")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    prompt = load_tag_prompt(args.artifact) if args.artifact else None
    cases = load_cases(split=None if args.split == "all" else args.split)
    client = ChatClient(model=args.model)
    model = args.model or chat_model()
    print(
        f"prompt={args.artifact or 'default'}  model={model}  split={args.split}"
        f"  cases={len(cases)}  repeat={args.repeat}\n"
    )

    def report(case, mean, runs):
        flag = "ok " if mean == 1.0 else ("~  " if mean >= 0.5 else "FAIL")
        print(f"  {flag} {case['id']:28s} {mean:.2f}  {runs[-1].get('tags')}")
        for e in dict.fromkeys(e for r in runs for e in r["errors"]):
            print(f"         - {e}")

    result = evaluate(
        client.chat_json, cases, prompt, vocab=VOCAB, repeat=args.repeat, on_case=report
    )
    print(f"\n  overall           {result.overall:.3f}")
    print(f"  median latency    {result.median_latency:.2f}s")
    print(f"  distinct tags     {len(set(result.tags))}")
    print(f"  singleton rate    {100 * result.singleton_rate:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
