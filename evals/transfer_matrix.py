#!/usr/bin/env python
"""Score prompt candidates across model families and apply the transfer gate.

    uv run python evals/transfer_matrix.py --artifact evals/prompts/tagging-2026-08-27.json
    uv run python evals/transfer_matrix.py --artifact a.json --artifact b.json \
        --models gemma4:e2b,hermes3:8b

Prints prompt x model -> dev score, the spread across models for each
prompt, and the gate verdict for every candidate against the hand-written
baseline (tagging_eval.gate). Writes the same as Markdown so the decision
is committed beside the artifact and readable later.

This is the acceptance test, and it is transfer rather than score on
purpose: this project's premise is that LLM_BASE_URL points anywhere, so a
prompt that wins on one model and collapses on another breaks the promise
silently. Models run sequentially -- Ollama serves one at a time on a
single GPU -- so the wall-clock is roughly prompts x models x cases x 2s.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tagging_eval import VOCAB, Gate, evaluate, gate, load_cases, spread

from attestation.features import TagPrompt, load_tag_prompt
from attestation.llm import ChatClient

DEFAULT_MODELS = "gemma4:e2b,gemma4:e4b,hermes3:8b"
PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


def render_markdown(
    models: list[str],
    scores: dict[str, dict[str, float]],
    verdicts: dict[str, Gate],
    split: str,
    n_cases: int,
    primary: str,
    repeat: int = 1,
) -> str:
    lines = [
        f"# Tagging prompt transfer matrix ({datetime.date.today().isoformat()})",
        "",
        f"Split `{split}`, {n_cases} cases, {repeat} run(s) each. Primary model `{primary}`.",
        "",
        "| prompt | " + " | ".join(models) + " | spread |",
        "|---|" + "---|" * (len(models) + 1),
    ]
    for name, row in scores.items():
        cells = " | ".join(f"{row[m]:.3f}" for m in models)
        lines.append(f"| {name} | {cells} | {spread(row):.3f} |")
    lines += ["", "## Gate", ""]
    for name, verdict in verdicts.items():
        lines.append(f"- **{name}**: {'PASS' if verdict.passed else 'FAIL'}")
        for reason in verdict.reasons:
            lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact", action="append", default=[], help="candidate artifact (repeatable)"
    )
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--primary", default=None, help="the optimizer's model (default: first)")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per case; production samples")
    ap.add_argument(
        "--out", default=None, help="Markdown path (default: evals/prompts/transfer-<date>.md)"
    )
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    primary = args.primary or models[0]
    cases = load_cases(split=None if args.split == "all" else args.split)
    prompts: dict[str, TagPrompt | None] = {"baseline": None}
    for path in args.artifact:
        prompts[pathlib.Path(path).stem] = load_tag_prompt(path)

    scores: dict[str, dict[str, float]] = {name: {} for name in prompts}
    for model in models:  # outer loop: one model swap per model, not per prompt
        client = ChatClient(model=model)
        for name, prompt in prompts.items():
            result = evaluate(client.chat_json, cases, prompt, vocab=VOCAB, repeat=args.repeat)
            scores[name][model] = result.overall
            print(
                f"  {model:14s} {name:28s} {result.overall:.3f}"
                f"  ({result.median_latency:.2f}s/case)"
            )

    verdicts = {
        name: gate(scores["baseline"], scores[name], primary)
        for name in prompts
        if name != "baseline"
    }
    md = render_markdown(models, scores, verdicts, args.split, len(cases), primary, args.repeat)
    print("\n" + md)
    out = (
        pathlib.Path(args.out)
        if args.out
        else PROMPTS_DIR / f"transfer-{datetime.date.today().isoformat()}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    # The matrix's raw numbers, for anyone who wants to re-derive the verdict.
    out.with_suffix(".json").write_text(
        json.dumps({"scores": scores, "split": args.split}, indent=2) + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
