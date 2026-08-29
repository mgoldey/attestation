#!/usr/bin/env python
"""Score the explanation prompt against a live model.

The refusal clause in explanation_messages is load-bearing: measured against
gemma4:e2b, dropping it let the model claim a termite-feed additive paper
shared "advanced topics like AI and machine learning" with an ML persona.
`bait-refuse-*` cases in the corpus are that failure, named.

Not a pytest test: it needs a live model, takes about a minute for the full
corpus, and is non-deterministic. It is a measurement tool, run deliberately.

    uv run python evals/run_explanation_eval.py                # dev split
    uv run python evals/run_explanation_eval.py --split all --repeat 2
    uv run python evals/run_explanation_eval.py --model hermes3:8b

The prompt is rendered by `attestation.explain.explanation_messages`, the
same function `generate_explanation` calls, so the number here is the number
`feed.explain` would actually produce.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from explanation_eval import evaluate, load_cases

from attestation.llm import ChatClient, chat_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per case, for stability")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cases = load_cases(split=None if args.split == "all" else args.split)
    client = ChatClient(model=args.model)
    model = args.model or chat_model()
    print(f"model={model}  split={args.split}  cases={len(cases)}  repeat={args.repeat}\n")

    def report(case, mean, runs):
        flag = "ok " if mean == 1.0 else ("~  " if mean >= 0.5 else "FAIL")
        print(f"  {flag} {case['id']:28s} {mean:.2f}  {runs[-1].get('text')!r}")
        for e in dict.fromkeys(e for r in runs for e in r["errors"]):
            print(f"         - {e}")

    result = evaluate(client.chat_json, cases, repeat=args.repeat, on_case=report)
    rp = result.refusal_precision_recall(cases)
    print(f"\n  model              {model}")
    print(f"  overall            {result.overall:.3f}")
    print(f"  refusal precision  {rp['precision']:.3f}  (tp={rp['tp']} fp={rp['fp']})")
    print(f"  refusal recall     {rp['recall']:.3f}  (tp={rp['tp']} fn={rp['fn']})")
    print(f"  median latency     {result.median_latency:.2f}s  model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
