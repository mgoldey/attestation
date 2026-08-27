#!/usr/bin/env python
"""Optimize the tagging instruction with DSPy's GEPA, offline, and ship data.

The optimizer runs a model many times. That never happens inside `attest
tag`: this script is a deliberate command like run_tagging_eval.py beside
it, never imported by the library, and its output is a prompt ARTIFACT on
disk -- instruction text plus the scores that justified it -- which
`attest tag` loads via ATTEST_TAG_PROMPT only if someone chooses to.

    uv run --group optimize python evals/optimize_tagging.py
    uv run --group optimize python evals/optimize_tagging.py --reflection-model qwen3.5:27b
    uv run --group optimize python evals/optimize_tagging.py --max-metric-calls 150   # quick

What makes this honest rather than "run DSPy and keep the winner":

* The model sees the PRODUCTION prompt. `TagMessagesAdapter` renders every
  DSPy call through `attestation.features.tag_messages`, so the instruction
  GEPA evolves is the system message `attest tag` sends, byte for byte, and
  the metric is the same `score_one` the eval harness uses.
* Instruction-only. Demonstrations selected from the labelled cases and
  scored on those cases would be a tautology -- the shape of this repo's
  bootstrap-label problem -- and the corpus's existing tags were produced by
  the prompt being replaced, so they cannot serve as gold either. GEPA
  mutates text; demos stay out of scope until a held-out pool exists.
* Train and dev never meet. GEPA sees the train split only; the artifact
  records dev scores measured afterwards through the real ChatClient, not
  through DSPy.
* Shipping is a separate decision. This writes a candidate. Whether it may
  replace the current default is transfer_matrix.py's call: not worse than
  the baseline on the optimizer's model, better on two others, with no wider
  a spread across them than the baseline has.

Budget: `--max-metric-calls` bounds student-model calls. One full pass over
the train split costs len(train) calls (~5s each on gemma4:e2b with the
40-tag vocabulary), plus a few reflection calls to the larger model per
round; the default 300 is about 25 minutes, inside the spec's 30.
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

from tagging_eval import VOCAB, EvalResult, evaluate, load_cases, score_one

from attestation.features import DEFAULT_TAG_INSTRUCTION, ItemTags, TagPrompt, tag_messages
from attestation.llm import ChatClient, _first_json_object, base_url, chat_model

PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
DEFAULT_REFLECTION_MODEL = "gemma4:12b"  # 7.6GB: fits beside the 7.2GB student on a 23GB box


def _dspy():
    try:
        import dspy
    except ImportError as exc:
        raise SystemExit(
            "dspy is not installed. It lives in its own dependency group:\n"
            "    uv run --group optimize python evals/optimize_tagging.py"
        ) from exc
    return dspy


def build_adapter(dspy, vocab: list[str]):
    """An adapter that renders exactly what `attest tag` sends.

    DSPy's own adapters wrap inputs in their field markup; an instruction
    tuned under that markup would be tuned for a prompt the product never
    sends. This one hands `signature.instructions` -- the text GEPA mutates
    -- to the production renderer as the system message.
    """

    class TagMessagesAdapter(dspy.Adapter):
        def format(self, signature, demos, inputs):
            # deliberately unused: instruction-only, see the module docstring.
            prompt = TagPrompt(instruction=signature.instructions)
            return tag_messages(inputs["title"], inputs["summary"], vocab, prompt)

        def parse(self, signature, completion):
            # The same tolerance production has: the first JSON object in the
            # reply, whatever surrounds it. Anything less would score the
            # adapter, not the prompt.
            try:
                out = _first_json_object(completion)
            except ValueError as exc:
                raise dspy.utils.exceptions.AdapterParseError(
                    adapter_name="TagMessagesAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=str(exc),
                ) from exc
            return {"content_type": out.get("content_type"), "tags": out.get("tags")}

    return TagMessagesAdapter()


def build_program(dspy):
    class TagItem(dspy.Signature):
        __doc__ = DEFAULT_TAG_INSTRUCTION
        title: str = dspy.InputField()
        summary: str = dspy.InputField()
        content_type: str = dspy.OutputField()
        tags: list[str] = dspy.OutputField()

    return dspy.Predict(TagItem)


def to_examples(dspy, cases: list[dict]) -> list:
    return [
        dspy.Example(title=c["title"], summary=c["summary"], case=c).with_inputs("title", "summary")
        for c in cases
    ]


def make_metric(dspy):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        out = {
            "content_type": getattr(pred, "content_type", None),
            "tags": getattr(pred, "tags", None),
        }
        scored = score_one(gold.case, out if out["tags"] is not None else {})
        feedback = "; ".join(scored["errors"]) or "correct"
        return dspy.Prediction(score=scored["score"], feedback=feedback)

    return metric


def local_lm(dspy, model: str, **kwargs):
    # litellm's openai/ provider speaks to any OpenAI-compatible server; the
    # key is required by the client and ignored by Ollama. Nothing leaves the
    # machine: base_url() is the same LLM_BASE_URL every other call uses.
    return dspy.LM(f"openai/{model}", api_base=base_url(), api_key="local", **kwargs)


def summarize(result: EvalResult) -> dict[str, Any]:
    return {
        "overall": round(result.overall, 4),
        "per_case": {k: round(v, 3) for k, v in result.per_case.items()},
        "median_latency_s": round(result.median_latency, 2),
        "singleton_rate": round(result.singleton_rate, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", default=None, help="student model (default: the configured chat model)"
    )
    ap.add_argument("--reflection-model", default=DEFAULT_REFLECTION_MODEL)
    ap.add_argument("--max-metric-calls", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=None, help="artifact path (default: evals/prompts/tagging-<date>.json)"
    )
    ap.add_argument("--log-dir", default=None, help="GEPA's own run log")
    args = ap.parse_args()

    dspy = _dspy()
    model = args.model or chat_model()
    today = datetime.date.today().isoformat()
    out_path = pathlib.Path(args.out) if args.out else PROMPTS_DIR / f"tagging-{today}.json"

    train = load_cases(split="train")
    dev = load_cases(split="dev")
    schema = ItemTags.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }
    # reasoning_effort="none" is what ChatClient sends: with it on, gemma4:e2b
    # spent its whole token budget in reasoning_content and returned an empty
    # reply (measured here on the first smoke call). The optimizer must see
    # the model exactly as production does or it optimizes a different model.
    student_lm = local_lm(
        dspy,
        model,
        temperature=0.0,
        max_tokens=300,
        response_format=response_format,
        reasoning_effort="none",
        # litellm's generic openai provider drops unknown params unless told
        # they are allowed; Ollama understands this one (llm.py measured it).
        allowed_openai_params=["reasoning_effort"],
    )
    reflection_lm = local_lm(dspy, args.reflection_model, temperature=1.0, max_tokens=6000)
    dspy.configure(lm=student_lm, adapter=build_adapter(dspy, VOCAB))

    program = build_program(dspy)
    print(
        f"student={model}  reflection={args.reflection_model}  train={len(train)}"
        f"  dev={len(dev)}  budget={args.max_metric_calls} metric calls\n"
    )

    optimizer = dspy.GEPA(
        metric=make_metric(dspy),
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=4,
        add_format_failure_as_feedback=True,
        track_stats=True,
        seed=args.seed,
        log_dir=args.log_dir,
        num_threads=1,  # one local GPU; parallel calls just queue and time out
    )
    optimized = optimizer.compile(program, trainset=to_examples(dspy, train))
    instruction = optimized.signature.instructions
    changed = instruction.strip() != DEFAULT_TAG_INSTRUCTION.strip()
    print("\n--- optimized instruction " + ("(unchanged)" if not changed else "") + "\n")
    print(instruction)

    # Score through the REAL client, not DSPy: these are the numbers the
    # artifact claims, so they come from the path `attest tag` uses.
    client = ChatClient(model=model)
    candidate = TagPrompt(instruction=instruction)
    print("\n--- scoring through ChatClient")
    scores = {}
    for label, prompt in (("baseline", None), ("candidate", candidate)):
        scores[label] = {
            split: summarize(evaluate(client.chat_json, cases, prompt, vocab=VOCAB))
            for split, cases in (("train", train), ("dev", dev))
        }
        print(
            f"  {label:9s} train {scores[label]['train']['overall']:.3f}"
            f"  dev {scores[label]['dev']['overall']:.3f}"
        )

    artifact = {
        "instruction": instruction,
        "demos": [],
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
            "Candidate only. Run evals/transfer_matrix.py --artifact <this file> to decide"
            " whether it clears the transfer gate; `shipped` records that decision."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
