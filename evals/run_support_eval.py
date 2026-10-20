#!/usr/bin/env python
"""Score `cite.support` verdicts against hand-labelled citing sentences.

The sources are real arXiv abstracts and the sentences are written to hit
one failure each (`note` in evals/support_cases.json names it); the anchor is
`nequip-wrong-meaning`, a sentence that keeps the paper's "three orders of
magnitude" and changes what it measures.

Not a pytest test when a model is involved: it needs a live server and is
non-deterministic. `--judge none` IS deterministic and needs nothing; the
unit tests pin its numbers.

    uv run python evals/run_support_eval.py --judge none
    uv run python evals/run_support_eval.py                    # default chat model
    uv run python evals/run_support_eval.py --model hermes3:3b --embed

The judge prompt is rendered by `attestation.support.judge_messages`, the
same function `cite.support` calls, so the number here is the tool's number.
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from attestation import support

CASES = pathlib.Path(__file__).resolve().parent / "support_cases.json"


def load() -> tuple[dict, list[dict]]:
    """(key -> SourceText, cases)."""
    data = json.loads(CASES.read_text())
    sources = {
        s["key"]: support.SourceText(s["key"], s["abstract"], "abstract", s["title"])
        for s in data["sources"]
    }
    return sources, data["cases"]


def score(checker: support.Checker, cases: list[dict], on_case=None) -> dict:
    """Exact-verdict accuracy, plus the two errors that matter most for a gate:
    a contradicted sentence called supported, and a supported one called
    contradicted."""
    confusion: Counter = Counter()
    for case in cases:
        t0 = time.monotonic()
        r = checker.check(case["sentence"], case["key"])
        confusion[(case["label"], str(r.verdict))] += 1
        if on_case:
            on_case(case, r, time.monotonic() - t0)
    n = sum(confusion.values())
    right = sum(v for (gold, got), v in confusion.items() if gold == got)
    return {
        "n": n,
        "accuracy": right / n if n else 0.0,
        "contradiction_recall": _recall(confusion, "contradicted"),
        "false_supported": confusion[("contradicted", "supported")],
        "false_contradicted": confusion[("supported", "contradicted")],
        "confusion": confusion,
    }


def _recall(confusion: Counter, label: str) -> float:
    total = sum(v for (gold, _), v in confusion.items() if gold == label)
    return confusion[(label, label)] / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llm", choices=support.JUDGES)
    ap.add_argument("--model", default=None, help="chat model (default: the configured one)")
    ap.add_argument("--embed", action="store_true", help="rerank with the configured embedder")
    args = ap.parse_args()

    chat_fn, label = None, ""
    if args.judge != "none":
        from attestation.llm import ChatClient

        client = ChatClient(model=args.model)
        chat_fn, label = client.chat_json, f"llm:{client.model}"
    sources, cases = load()
    checker = support.Checker(
        judge=support.Judge(args.judge, chat_fn=chat_fn, label=label),
        retriever=support.Retriever(enabled=args.embed),
    )
    checker._sources = sources

    def report(case, r, dt):
        flag = "ok  " if r.verdict == case["label"] else "MISS"
        print(f"  {flag} {case['id']:26s} {r.verdict:20s} want {case['label']:20s} {dt:5.1f}s")
        if flag == "MISS":
            print(f"       {r.judge}: {r.reason[:140]}")

    print(f"judge={args.judge} {label}  retrieval={'embedding' if args.embed else 'lexical'}\n")
    out = score(checker, cases, on_case=report)
    print(f"\n  accuracy              {out['accuracy']:.3f}  ({out['n']} cases)")
    print(f"  contradiction recall  {out['contradiction_recall']:.3f}")
    print(f"  contradicted->supported  {out['false_supported']}")
    print(f"  supported->contradicted  {out['false_contradicted']}")
    for note in (checker.judge.note, checker.retriever.note):
        if note:
            print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
