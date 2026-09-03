#!/usr/bin/env python
"""Score skill-trigger candidates across model families and apply the
transfer gate. Mirrors transfer_matrix.py exactly -- see its docstring for
why this is transfer rather than a single-model score, and for the wall-clock
shape (models x candidates x cases, sequential, one GPU).

    uv run python evals/transfer_skill_triggers.py \
        --artifact evals/prompts/provenance-trigger-2026-09-03.json
    uv run python evals/transfer_skill_triggers.py --artifact a.json --models gemma4:e2b,hermes3:8b

The artifact names its own `skill`; only that skill's cases are scored, each
candidate against baseline (`section_text=None`, i.e. the shipped SKILL.md).
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from skill_trigger_eval import evaluate, load_cases
from tagging_eval import Gate, gate, spread

from attestation.llm import ChatClient

DEFAULT_MODELS = "gemma4:e2b,gemma4:e4b,hermes3:8b"
PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


def render_markdown(
    skill: str,
    models: list[str],
    scores: dict[str, dict[str, float]],
    verdicts: dict[str, Gate],
    split: str,
    n_cases: int,
    primary: str,
) -> str:
    lines = [
        f"# Skill trigger transfer matrix: {skill} ({datetime.date.today().isoformat()})",
        "",
        f"Split `{split}`, {n_cases} case(s). Primary model `{primary}`.",
        "",
        "| candidate | " + " | ".join(models) + " | spread |",
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
        "--artifact",
        action="append",
        default=[],
        required=True,
        help="candidate artifact (repeatable)",
    )
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--primary", default=None, help="the optimizer's model (default: first)")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    primary = args.primary or models[0]

    artifacts = [json.loads(pathlib.Path(p).read_text()) for p in args.artifact]
    skills = {a["skill"] for a in artifacts}
    if len(skills) != 1:
        raise SystemExit(f"all --artifact candidates must be for the same skill, got {skills}")
    skill = skills.pop()

    cases = [
        c
        for c in load_cases(split=None if args.split == "all" else args.split)
        if c["skill"] == skill
    ]
    candidates: dict[str, str | None] = {"baseline": None}
    for path, artifact in zip(args.artifact, artifacts, strict=True):
        candidates[pathlib.Path(path).stem] = artifact["instruction"]

    scores: dict[str, dict[str, float]] = {name: {} for name in candidates}
    for model in models:  # outer loop: one model swap per model, not per candidate
        client = ChatClient(model=model)
        for name, section in candidates.items():
            result = evaluate(client.chat_json, cases, section, skill=skill)
            scores[name][model] = result.overall
            print(
                f"  {model:14s} {name:28s} {result.overall:.3f}"
                f"  ({result.median_latency:.2f}s/case)"
            )

    verdicts = {
        name: gate(scores["baseline"], scores[name], primary)
        for name in candidates
        if name != "baseline"
    }
    md = render_markdown(skill, models, scores, verdicts, args.split, len(cases), primary)
    print("\n" + md)
    out = (
        pathlib.Path(args.out)
        if args.out
        else PROMPTS_DIR / f"transfer-{skill}-{datetime.date.today().isoformat()}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    out.with_suffix(".json").write_text(
        json.dumps({"skill": skill, "scores": scores, "split": args.split}, indent=2) + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
