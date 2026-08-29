#!/usr/bin/env python
"""Score the reaction prompt against a live model.

`simulate.py`'s `confidence` was measured INERT on gemma4:e2b: 45 live calls
returned confidence 4 or 5 every time, including on a content-free item, so
AUC over signed confidence was undefined for that run. This script prints
the confidence histogram beside the score on every run so that failure is
visible rather than assumed fixed.

Not a pytest test: it needs a live model, takes minutes, and is
non-deterministic. It is a measurement tool, run deliberately.

    uv run python evals/run_reaction_eval.py                      # dev split, live model
    uv run python evals/run_reaction_eval.py --split all --repeat 3
    uv run python evals/run_reaction_eval.py --offline             # no Ollama, no network
    uv run python evals/run_reaction_eval.py --model hermes3:8b    # transfer

The prompt is rendered by `attestation.simulate.reaction_messages`, the same
function `feed.simulate_ratings` uses, so the number here is the number in
production.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples" / "flows"))

from reaction_eval import evaluate, load_cases, score_verdicts

from attestation.llm import ChatClient, chat_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per case, for stability")
    ap.add_argument("--model", default=None)
    ap.add_argument("--offline", action="store_true", help="use the stub model server")
    args = ap.parse_args()

    server = None
    if args.offline:
        import stub_openai

        server, url = stub_openai.start()
        model = stub_openai.MODEL
        client = ChatClient(base_url=url, model=model)
    else:
        client = ChatClient(model=args.model)
        model = args.model or chat_model()

    cases = load_cases(split=None if args.split == "all" else args.split)
    print(f"model={model}  split={args.split}  cases={len(cases)}  repeat={args.repeat}\n")

    def report(case, mean, runs):
        flag = "ok " if mean == 1.0 else ("~  " if mean >= 0.5 else "FAIL")
        last = runs[-1]
        print(f"  {flag} {case['id']:28s} {mean:.2f}  verdict={last.get('verdict')}")
        for e in dict.fromkeys(e for r in runs for e in r["errors"]):
            print(f"         - {e}")

    try:
        result = evaluate(client.chat_json, cases, repeat=args.repeat, on_case=report)
    finally:
        if server:
            server.shutdown()

    # Precision/recall/AUC against the same labels the cases carry, using the
    # same function examples/flows/persona_eval.py prints -- one definition.
    labels = {c["id"]: c["verdict"] for c in cases}
    reactions = [
        {"item_id": cid, "verdict": run[-1]["verdict"], "confidence": run[-1]["confidence"]}
        for cid, run in result.runs.items()
        if run[-1]["verdict"] is not None
    ]
    verdicts = score_verdicts(reactions, labels)

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    print(f"\n  model={model}  overall            {result.overall:.3f}")
    print(f"  model={model}  precision          {fmt(verdicts['precision'])}")
    print(f"  model={model}  recall             {fmt(verdicts['recall'])}")
    print(f"  model={model}  auc                {fmt(verdicts['auc'])}")
    print(f"  model={model}  confidence histogram {verdicts['confidence_histogram']}")
    if verdicts["auc"] is None:
        print(f"  model={model}  <- confidence never varies: AUC undefined")
    print(f"  model={model}  median latency     {result.median_latency:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
