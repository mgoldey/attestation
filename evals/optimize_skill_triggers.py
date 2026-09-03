#!/usr/bin/env python
"""Optimize one skill's trigger section with DSPy's GEPA, offline, ship data.

Mirrors optimize_tagging.py's exact pattern -- read that file's docstring
first, this one only calls out where a skill's trigger text differs from the
tagging instruction:

    uv run --group optimize python evals/optimize_skill_triggers.py --skill provenance
    uv run --group optimize python evals/optimize_skill_triggers.py --skill feed \
        --max-metric-calls 60

What makes this honest, the same three properties optimize_tagging.py names:

* The model sees the PRODUCTION prompt. `SkillTriggerAdapter` renders every
  DSPy call through `attestation.trigger_messages` (imported from
  skill_trigger_eval, not duplicated), so the instruction GEPA evolves is the
  system text a real session would see, byte for byte, and the metric is the
  same `score_one` the eval harness uses.
* Instruction-only, one skill at a time. GEPA mutates the named section's
  text for ONE skill; every other skill's cases in the shared case file are
  scored against THEIR OWN shipped default throughout (trigger_messages'
  `skill=`/`section_text=` parameters keep them separate), so a candidate for
  `provenance` is never accidentally credited with a `feed` case it did not
  touch.
* Train and dev never meet. GEPA sees the train split (for this skill) only;
  the artifact records dev scores measured afterwards through the real
  ChatClient, not through DSPy.

Per-skill case counts here are small (4-7 cases per skill, this being a new
eval built for one session rather than tagging's 51-case corpus grown over
weeks) -- `--max-metric-calls` is scaled down accordingly; see the module's
own default and the report this script prints for what a given run actually
cost.

Gate: the same `tagging_eval.gate()` a candidate must clear before shipping
-- not worse than baseline on the primary model AND better on >=2 other
models AND no wider spread. Reused, not reimplemented; see its docstring for
the amendment history and why "not worse" rather than "beats".
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from skill_trigger_eval import (
    SKILL_SECTIONS,
    EvalResult,
    default_section_text,
    evaluate,
    load_cases,
)

from attestation.llm import ChatClient, _first_json_object, base_url, chat_model

PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
DEFAULT_REFLECTION_MODEL = "gemma4:12b"  # matches optimize_tagging.py's choice and its reasoning


def _dspy():
    try:
        import dspy
    except ImportError as exc:
        raise SystemExit(
            "dspy is not installed. It lives in its own dependency group:\n"
            "    uv run --group optimize python evals/optimize_skill_triggers.py"
        ) from exc
    return dspy


def build_adapter(dspy, skill: str):
    """An adapter that renders exactly what a live session would see for
    `skill` -- see optimize_tagging.py's `build_adapter` for why this exists
    instead of a DSPy stock adapter: the instruction GEPA mutates must be the
    system message production sends, not that message wrapped in DSPy's own
    field markup."""

    class TriggerMessagesAdapter(dspy.Adapter):
        def format(self, signature, demos, inputs):
            # deliberately unused: instruction-only, see the module docstring.
            from skill_trigger_eval import trigger_messages

            return trigger_messages(skill, inputs["question"], signature.instructions)

        def parse(self, signature, completion):
            try:
                out = _first_json_object(completion)
            except ValueError as exc:
                raise dspy.utils.exceptions.AdapterParseError(
                    adapter_name="TriggerMessagesAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=str(exc),
                ) from exc
            return {"tool": out.get("tool"), "args": out.get("args") or {}}

    return TriggerMessagesAdapter()


def build_program(dspy, skill: str):
    default = default_section_text(skill)

    class DecideTool(dspy.Signature):
        __doc__ = default
        question: str = dspy.InputField()
        tool: str = dspy.OutputField(desc="namespaced tool name, or the string 'null'")
        args: dict = dspy.OutputField()

    return dspy.Predict(DecideTool)


def to_examples(dspy, cases: list[dict]) -> list:
    return [dspy.Example(question=c["question"], case=c).with_inputs("question") for c in cases]


def make_metric(dspy):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        from skill_trigger_eval import score_one

        tool = getattr(pred, "tool", None)
        if tool == "null":
            tool = None
        out = {"tool": tool, "args": getattr(pred, "args", None) or {}}
        scored = score_one(gold.case, out)
        feedback = "; ".join(scored["errors"]) or "correct"
        return dspy.Prediction(score=scored["score"], feedback=feedback)

    return metric


def local_lm(dspy, model: str, **kwargs):
    return dspy.LM(f"openai/{model}", api_base=base_url(), api_key="local", **kwargs)


def summarize(result: EvalResult) -> dict[str, Any]:
    return {
        "overall": round(result.overall, 4),
        "per_case": {k: round(v, 3) for k, v in result.per_case.items()},
        "median_latency_s": round(result.median_latency, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skill", required=True, choices=sorted(SKILL_SECTIONS))
    ap.add_argument(
        "--model", default=None, help="student model (default: the configured chat model)"
    )
    ap.add_argument("--reflection-model", default=DEFAULT_REFLECTION_MODEL)
    ap.add_argument("--max-metric-calls", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=None,
        help="artifact path (default: evals/prompts/<skill>-trigger-<date>.json)",
    )
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    dspy = _dspy()
    model = args.model or chat_model()
    today = datetime.date.today().isoformat()
    out_path = (
        pathlib.Path(args.out) if args.out else PROMPTS_DIR / f"{args.skill}-trigger-{today}.json"
    )

    all_cases = load_cases()
    train = [c for c in all_cases if c["skill"] == args.skill and c["split"] == "train"]
    dev = [c for c in all_cases if c["skill"] == args.skill and c["split"] == "dev"]
    if not train:
        raise SystemExit(f"no train cases for skill={args.skill!r}")

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
                "required": ["tool", "args"],
            },
            "strict": True,
        },
    }
    student_lm = local_lm(
        dspy,
        model,
        temperature=0.0,
        max_tokens=300,
        response_format=response_format,
        reasoning_effort="none",
        allowed_openai_params=["reasoning_effort"],
    )
    reflection_lm = local_lm(dspy, args.reflection_model, temperature=1.0, max_tokens=6000)
    dspy.configure(lm=student_lm, adapter=build_adapter(dspy, args.skill))

    program = build_program(dspy, args.skill)
    print(
        f"skill={args.skill}  student={model}  reflection={args.reflection_model}"
        f"  train={len(train)}  dev={len(dev)}  budget={args.max_metric_calls} metric calls\n"
    )

    optimizer = dspy.GEPA(
        metric=make_metric(dspy),
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=min(4, len(train)),
        add_format_failure_as_feedback=True,
        track_stats=True,
        seed=args.seed,
        log_dir=args.log_dir,
        num_threads=1,  # one local GPU; parallel calls just queue and time out
    )
    optimized = optimizer.compile(program, trainset=to_examples(dspy, train))
    instruction = optimized.signature.instructions
    default = default_section_text(args.skill)
    changed = instruction.strip() != default.strip()
    print("\n--- optimized instruction " + ("(unchanged)" if not changed else "") + "\n")
    print(instruction)

    # Score through the REAL client, not DSPy: these are the numbers the
    # artifact claims, so they come from the path a live session uses.
    client = ChatClient(model=model)
    print("\n--- scoring through ChatClient")
    scores = {}
    for label, section in (("baseline", None), ("candidate", instruction)):
        scores[label] = {
            split: summarize(evaluate(client.chat_json, cases, section, skill=args.skill))
            for split, cases in (("train", train), ("dev", dev))
            if cases
        }
        train_score = scores[label].get("train", {}).get("overall")
        dev_score = scores[label].get("dev", {}).get("overall")
        print(f"  {label:9s} train {train_score}  dev {dev_score}")

    artifact = {
        "skill": args.skill,
        "instruction": instruction,
        "default": default,
        "optimizer": "dspy.GEPA",
        "created": today,
        "student_model": model,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        "seed": args.seed,
        "train_ids": [c["id"] for c in train],
        "dev_ids": [c["id"] for c in dev],
        "scores": scores,
        "shipped": False,
        "note": (
            "Candidate only. Run evals/transfer_skill_triggers.py --artifact <this"
            " file> to decide whether it clears the transfer gate; `shipped` records"
            " that decision -- same division of labour as optimize_tagging.py vs"
            " transfer_matrix.py."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
