#!/usr/bin/env python
"""Score the attestation-record skill: what a model writes to disk after
running an experiment, checked by the REAL ledger reader.

    uv run python evals/run_record_eval.py --offline   # scorer only, no model
    uv run python evals/run_record_eval.py --live       # acceptance run

`--offline` scores the committed `record_cases.json` fixtures -- each case
carries a scenario and a hand-written, correct-by-construction `answer`
manifest, plus one deliberately-bad case marked `expect_fail: true` (proves
the scorer can fail things, not just pass them). This is what CI runs: no
model touched, `record_eval.score_one` exercised against known inputs.

`--live` sends the `attestation-record` SKILL.md body (read by path, not
imported -- the other half of this spec's work may not exist yet) plus each
scenario's prompt to a live model, parses its manifest, and scores it the
same way. Requires a model server at LLM_BASE_URL (default model
`gemma4:e2b-it-q4_K_M`, temp 0). Ollama's gemma4:e2b now thinks by default;
`llm.ChatClient.chat_json` already sends `reasoning_effort: "none"` on the
OpenAI-compatible endpoint for exactly this reason (see its docstring), so
this driver reuses it rather than hand-rolling a native `/api/chat` call.
Writes a dated record to `evals/prompts/write-side-YYYY-MM-DD.md`, mirroring
`examples/flows`'s RESULTS.md rule: the committed artifact is the number, and
only a live run may write it.
"""

import argparse
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from record_eval import load_cases, score_one
from tagging_eval import EvalResult

from attestation.llm import ChatClient

SKILL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "attestation"
    / "skills"
    / "attestation-record"
    / "SKILL.md"
)

# The manifest shape a --live trial must answer in, expressed as a JSON
# schema so ChatClient.chat_json can request it directly (it always requests
# structured output, never free text) -- {"files": {relpath: content}}.
MANIFEST_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
    },
    "required": ["files"],
}

PROMPT_TEMPLATE = (
    "You just ran an experiment named {family} with arms {arms}. The final"
    " {metric} values were: {values_str}. The corpus was {corpus}."
    " Following the skill above, write the files you would leave on disk to"
    ' record this. Answer with a JSON object {{"files": {{relpath: content}}}}'
    " -- paths relative to one project directory (e.g. results/..., configs/...)."
)


def scenario_prompt(scenario: dict) -> str:
    values_str = ", ".join(f"{arm}={scenario['values'][arm]}" for arm in scenario["arms"])
    return PROMPT_TEMPLATE.format(
        family=scenario["family"],
        arms=", ".join(scenario["arms"]),
        metric=scenario["metric"],
        values_str=values_str,
        corpus=scenario["corpus"],
    )


def run_offline(cases: list[dict]) -> EvalResult:
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    for case in cases:
        result = score_one(case, case["answer"])
        expect_fail = case.get("expect_fail", False)
        ok = (not result["pass"]) if expect_fail else result["pass"]
        per_case[case["id"]] = 1.0 if ok else 0.0
        runs[case["id"]] = [result]
        flag = "ok " if ok else "FAIL"
        print(f"  {flag} {case['id']:28s} pass={result['pass']} expect_fail={expect_fail}")
        for err in result["errors"]:
            print(f"         - {err}")
    return EvalResult(per_case=per_case, runs=runs, latencies=[])


def run_live(cases: list[dict], model: str | None) -> EvalResult:
    if not SKILL_PATH.is_file():
        print(f"warning: {SKILL_PATH} does not exist yet -- sending an empty system message")
        skill_body = ""
    else:
        skill_body = SKILL_PATH.read_text()

    client = ChatClient(model=model)
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    for case in cases:
        messages = [
            {"role": "system", "content": skill_body},
            {"role": "user", "content": scenario_prompt(case)},
        ]
        try:
            answer = client.chat_json(messages, MANIFEST_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - a transport failure costs one trial
            print(f"  FAIL {case['id']:28s} transport error: {exc}")
            per_case[case["id"]] = 0.0
            runs[case["id"]] = [{"errors": [str(exc)], "pass": False}]
            continue
        result = score_one(case, answer)
        per_case[case["id"]] = 1.0 if result["pass"] else 0.0
        runs[case["id"]] = [result]
        flag = "ok " if result["pass"] else "FAIL"
        print(f"  {flag} {case['id']:28s} checks={result['checks']}")
        for err in result["errors"]:
            print(f"         - {err}")
    return EvalResult(per_case=per_case, runs=runs, latencies=[])


def write_record(result: EvalResult, model: str, n: int) -> pathlib.Path:
    today = datetime.date.today().isoformat()
    path = pathlib.Path(__file__).parent / "prompts" / f"write-side-{today}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    per_check_rates: dict[str, list[bool]] = {}
    for case_runs in result.runs.values():
        for r in case_runs:
            for check, ok in r.get("checks", {}).items():
                per_check_rates.setdefault(check, []).append(ok)
    lines = [
        f"# Write-side skills acceptance -- {today}",
        "",
        f"record: model={model}, trials={n}, overall={result.overall:.3f}",
        "",
        "| check | pass rate |",
        "| --- | --- |",
    ]
    for check, oks in per_check_rates.items():
        rate = sum(oks) / len(oks) if oks else 0.0
        lines.append(f"| {check} | {rate:.2f} ({sum(oks)}/{len(oks)}) |")
    existing = path.read_text() if path.is_file() else ""
    path.write_text(existing + "\n".join(lines) + "\n\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="score committed fixtures, no model")
    mode.add_argument("--live", action="store_true", help="score a live model (acceptance run)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cases = load_cases()
    print(f"record eval: {len(cases)} case(s), mode={'live' if args.live else 'offline'}\n")

    if args.offline:
        result = run_offline(cases)
    else:
        result = run_live(cases, args.model)

    print(f"\n  overall  {result.overall:.3f}  ({len(cases)} trial(s))")

    if args.live:
        from attestation.llm import chat_model

        model = args.model or chat_model()
        path = write_record(result, model, len(cases))
        print(f"  wrote {path}")

    return 0 if result.overall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
