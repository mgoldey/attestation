#!/usr/bin/env python
"""Score the attestation-annotate skill: whether a model's results paragraph
is fully claim-covered, correct, and free of invented citations, checked by
the REAL `claims.coverage`/`claims.check`.

    uv run python evals/run_annotate_eval.py --offline   # scorer only, no model
    uv run python evals/run_annotate_eval.py --live       # acceptance run

`--offline` scores the committed `annotate_cases.json` fixtures -- each case
carries a `runs.detail`-shaped payload and a hand-written, correct-by-
construction `paragraph`, plus two deliberately-bad cases marked
`expect_fail: true` (an uncovered decimal, and a wrong value with an
invented `cite=`). This is what CI runs.

`--live` sends the `attestation-annotate` SKILL.md body plus each payload's
prompt to a live model and scores its paragraph the same way. Requires a
model server at LLM_BASE_URL (default model `gemma4:e2b-it-q4_K_M`, temp 0).

The paragraph is free text, not a JSON object, so this driver does not reuse
`llm.ChatClient.chat_json` (which always requests `response_format: json_schema`
and would force the prose itself into a JSON string). It talks to the same
OpenAI-compatible `/chat/completions` endpoint directly and carries over
`chat_json`'s exact fix for Ollama's now-on-by-default thinking: send
`reasoning_effort: "none"`, retry once without it on a 400 for a server that
rejects the field -- see `llm.ChatClient.chat_json`'s docstring for the
measurement (19.8s/~500 thinking tokens vs 10.5s with it off, on gemma4:e2b).
"""

import argparse
import datetime
import pathlib
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from annotate_eval import load_cases, score_one
from tagging_eval import EvalResult

from attestation.llm import _headers, base_url, chat_model

SKILL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "attestation"
    / "skills"
    / "attestation-annotate"
    / "SKILL.md"
)

PROMPT_TEMPLATE = (
    "Here is a runs.detail-shaped payload for project={project!r} run={run!r}:\n"
    "{metrics}\n\n"
    "Write a short Markdown paragraph about {topic}, following the skill above."
)


def scenario_prompt(payload: dict) -> str:
    metrics_lines = "\n".join(
        f"  - metric={m['metric']} value={m['value']} step={m.get('step')} split={m.get('split')}"
        for m in payload["metrics"]
    )
    return PROMPT_TEMPLATE.format(
        project=payload["project"],
        run=payload["run"],
        metrics=metrics_lines,
        topic=payload["topic"],
    )


def chat_text(client: httpx.Client, model: str, messages: list[dict]) -> str:
    """One free-text chat completion -- see the module docstring for why this
    does not go through `llm.ChatClient.chat_json` (schema-constrained JSON
    only) but keeps its thinking-off fix."""
    payload = {"model": model, "messages": messages, "reasoning_effort": "none"}
    resp = client.post("/chat/completions", json=payload)
    if resp.status_code == 400:
        payload.pop("reasoning_effort")
        resp = client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_offline(cases: list[dict]) -> EvalResult:
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    for case in cases:
        payload = {k: case[k] for k in ("id", "project", "run", "metrics", "topic")}
        result = score_one(payload, case["paragraph"])
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

    resolved_model = model or chat_model()
    client = httpx.Client(base_url=base_url(), timeout=120, headers=_headers(None))
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    for case in cases:
        payload = {k: case[k] for k in ("id", "project", "run", "metrics", "topic")}
        messages = [
            {"role": "system", "content": skill_body},
            {"role": "user", "content": scenario_prompt(payload)},
        ]
        try:
            paragraph = chat_text(client, resolved_model, messages)
        except Exception as exc:  # noqa: BLE001 - a transport failure costs one trial
            print(f"  FAIL {case['id']:28s} transport error: {exc}")
            per_case[case["id"]] = 0.0
            runs[case["id"]] = [{"errors": [str(exc)], "pass": False}]
            continue
        result = score_one(payload, paragraph)
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
        f"annotate: model={model}, trials={n}, overall={result.overall:.3f}",
        "",
        "| check | pass rate |",
        "| --- | --- |",
    ]
    for check, oks in per_check_rates.items():
        rate = sum(oks) / len(oks) if oks else 0.0
        lines.append(f"| {check} | {rate:.2f} ({sum(oks)}/{len(oks)}) |")
    existing = (
        path.read_text() if path.is_file() else f"# Write-side skills acceptance -- {today}\n\n"
    )
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
    print(f"annotate eval: {len(cases)} case(s), mode={'live' if args.live else 'offline'}\n")

    if args.offline:
        result = run_offline(cases)
    else:
        result = run_live(cases, args.model)

    print(f"\n  overall  {result.overall:.3f}  ({len(cases)} trial(s))")

    if args.live:
        model = args.model or chat_model()
        path = write_record(result, model, len(cases))
        print(f"  wrote {path}")

    return 0 if result.overall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
