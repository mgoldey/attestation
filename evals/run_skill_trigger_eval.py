#!/usr/bin/env python
"""Score a skill's trigger section (its "Ask the router first" text, or the
write-side equivalent) against realistic user turns.

    uv run python evals/run_skill_trigger_eval.py --offline
    uv run python evals/run_skill_trigger_eval.py --live --skill provenance
    uv run python evals/run_skill_trigger_eval.py --live --skill feed --repeat 3
    uv run python evals/run_skill_trigger_eval.py --live --model hermes3:8b

`--offline` is model-free and CI-safe. It does NOT call `skill_trigger_eval`'s
`evaluate()` (that needs a model) -- instead it cross-checks every case whose
`skill` has a pure router (feed/provenance/knowledge/symbolic) against
`attestation.mcp.routing` directly: the case's `expect_tool` must be
`"<ns>.ask"` and the router itself must not silently decline in a way the
case doesn't expect (a router regression would be a routing.py bug, already
guarded by tests/test_ask_routing.py, but this catches a case file that
drifted out of sync with the real router). record/annotate/setup cases have
no pure function to check against, so --offline only checks their shape
(case fields present, valid JSON).

`--live` sends `skill_trigger_eval.trigger_messages` (the shipped section
text, or an `--artifact` candidate) to a real model via ChatClient and scores
the tool decision it returns. This is the number the optimizer's gate reads.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import statistics
import time

from skill_trigger_eval import (
    SKILL_SECTIONS,
    TOOL_SCHEMA,
    EvalResult,
    load_cases,
    score_one,
    trigger_messages,
)

from attestation.llm import ChatClient, chat_model
from attestation.mcp.routing import route_feed, route_kg, route_runs, route_sym

_ROUTERS = {
    "feed": route_feed,
    "provenance": route_runs,
    "knowledge": route_kg,
    "symbolic": route_sym,
}


def run_offline(cases: list[dict]) -> EvalResult:
    """Model-free shape + router-consistency check. Does not measure a model
    -- it measures that the case file's ground truth agrees with the real
    router and that every case is well-formed."""
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    for case in cases:
        errors = []
        skill = case["skill"]
        if skill not in SKILL_SECTIONS:
            errors.append(f"unknown skill {skill!r}")
        if "expect_tool" not in case:
            errors.append("missing expect_tool")
        expected = case.get("expect_tool")
        if skill in _ROUTERS and expected is not None and expected.endswith(".ask"):
            # The case claims this question reaches the router at all. What
            # the router does BENEATH .ask (which sub-tool, or a decline
            # asking to disambiguate) is not re-checked here -- that is
            # test_ask_routing's job -- only that this is a real, sensible
            # question the router module can accept without raising.
            try:
                if skill == "symbolic":
                    route_sym(case["question"], "")
                else:
                    _ROUTERS[skill](case["question"])
            except Exception as exc:  # noqa: BLE001 - a case whose router call crashes is a bad case
                errors.append(f"router raised on this question: {exc}")
        ok = not errors
        per_case[case["id"]] = 1.0 if ok else 0.0
        result = {"id": case["id"], "errors": errors, "pass": ok}
        runs[case["id"]] = [result]
        flag = "ok " if ok else "FAIL"
        print(f"  {flag} {case['id']:32s} skill={skill}")
        for e in errors:
            print(f"         - {e}")
    return EvalResult(per_case=per_case, runs=runs, latencies=[])


def run_live(
    cases: list[dict], model: str | None, repeat: int, skill: str | None, section_text: str | None
) -> EvalResult:
    """Score `cases` through a live model. `skill`/`section_text` (both or
    neither) point a single skill's cases at a candidate trigger section;
    every other case in the list still scores against ITS OWN shipped
    default -- see `trigger_messages`'s docstring on why."""
    client = ChatClient(model=model)
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    latencies: list[float] = []
    for case in cases:
        this_section = section_text if (skill is None or case["skill"] == skill) else None
        scored_runs = []
        for _ in range(repeat):
            messages = trigger_messages(case["skill"], case["question"], this_section)
            t0 = time.perf_counter()
            try:
                out = client.chat_json(messages, TOOL_SCHEMA)
            except Exception as exc:  # noqa: BLE001 - a transport error costs one trial
                out = {"_transport_error": str(exc)}
            latencies.append(time.perf_counter() - t0)
            scored_runs.append(score_one(case, out))
        mean = statistics.mean(r["score"] for r in scored_runs)
        per_case[case["id"]] = mean
        runs[case["id"]] = scored_runs
        flag = "ok " if mean == 1.0 else ("~  " if mean > 0 else "FAIL")
        print(f"  {flag} {case['id']:32s} {mean:.2f}  tool={scored_runs[-1].get('tool')!r}")
        for e in dict.fromkeys(e for r in scored_runs for e in r["errors"]):
            print(f"         - {e}")
    return EvalResult(per_case=per_case, runs=runs, latencies=latencies)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="model-free shape/router check")
    mode.add_argument("--live", action="store_true", help="score a live model")
    ap.add_argument(
        "--skill", default=None, choices=sorted(SKILL_SECTIONS), help="filter to one skill"
    )
    ap.add_argument("--split", default="all", choices=["train", "dev", "all"])
    ap.add_argument("--repeat", type=int, default=1, help="samples per case, --live only")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--artifact",
        default=None,
        help="a candidate trigger-section JSON artifact (evals/prompts/*.json);"
        " default = shipped SKILL.md",
    )
    args = ap.parse_args()

    cases = load_cases(split=None if args.split == "all" else args.split)
    if args.skill:
        cases = [c for c in cases if c["skill"] == args.skill]

    section_text = None
    if args.artifact:
        artifact = json.loads(pathlib.Path(args.artifact).read_text())
        section_text = artifact["instruction"]
        if not args.skill:
            raise SystemExit("--artifact requires --skill (a candidate is per-skill)")

    mode_name = "live" if args.live else "offline"
    model = args.model or chat_model()
    print(
        f"skill_trigger eval: mode={mode_name} skill={args.skill or 'all'} split={args.split}"
        f" cases={len(cases)}" + (f" model={model}" if args.live else "") + "\n"
    )

    if args.offline:
        result = run_offline(cases)
    else:
        result = run_live(cases, args.model, args.repeat, args.skill, section_text)

    n_trials = len(cases) if not args.live else len(cases) * args.repeat
    print(f"\n  overall  {result.overall:.3f}  ({n_trials} trial(s))")
    if args.live:
        print(f"  median latency  {result.median_latency:.2f}s")
    return 0 if result.overall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
