#!/usr/bin/env python
"""Score the attestation-annotate skill: whether a model's results paragraph
is fully claim-covered, correct, and free of invented citations, checked by
the REAL `claims.coverage`/`claims.check`.

    uv run python evals/run_annotate_eval.py --offline          # scorer only, no model
    uv run python evals/run_annotate_eval.py --live             # acceptance run
    uv run python evals/run_annotate_eval.py --live --repeat 5  # 5 samples/scenario

`--offline` scores the committed `annotate_cases.json` fixtures -- each case
carries a `runs.detail`-shaped payload and a hand-written, correct-by-
construction `paragraph`, plus two deliberately-bad cases marked
`expect_fail: true` (an uncovered decimal, and a wrong value with an
invented `cite=`). This is what CI runs. It never writes under
`evals/prompts/` -- that is the live acceptance's job.

`--live` sends the `attestation-annotate` SKILL.md body plus each payload's
prompt to a live model and scores its paragraph the same way. Requires a
model server at LLM_BASE_URL (default model `gemma4:e2b-it-q4_K_M`, temp 0).

`--repeat N` (default 1) draws N samples per scenario -- see
run_record_eval.py's docstring for why a single live sample cannot answer
"does the skill work" (a re-ask can score differently on the same
scenario). The overall score is the mean over every sample; the written
record additionally reports each scenario's own k/N pass count, and every
sample's raw paragraph and per-check result go to a sidecar
`write-side-YYYY-MM-DD.answers.json` alongside the .md.

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
import json
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

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"

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


def run_offline(cases: list[dict]) -> tuple[EvalResult, list[dict]]:
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    samples: list[dict] = []
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
        samples.append(
            {
                "id": case["id"],
                "sample": 0,
                "answer": case["paragraph"],
                "checks": result.get("checks", {}),
                "errors": result.get("errors", []),
            }
        )
    return EvalResult(per_case=per_case, runs=runs, latencies=[]), samples


def run_live(cases: list[dict], model: str | None, repeat: int) -> tuple[EvalResult, list[dict]]:
    if not SKILL_PATH.is_file():
        print(f"warning: {SKILL_PATH} does not exist yet -- sending an empty system message")
        skill_body = ""
    else:
        skill_body = SKILL_PATH.read_text()

    resolved_model = model or chat_model()
    client = httpx.Client(base_url=base_url(), timeout=120, headers=_headers(None))
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    samples: list[dict] = []
    for case in cases:
        payload = {k: case[k] for k in ("id", "project", "run", "metrics", "topic")}
        messages = [
            {"role": "system", "content": skill_body},
            {"role": "user", "content": scenario_prompt(payload)},
        ]
        scored_runs = []
        for sample_idx in range(repeat):
            try:
                paragraph = chat_text(client, resolved_model, messages)
            except Exception as exc:  # noqa: BLE001 - a transport failure costs one sample
                print(f"  FAIL {case['id']:28s} [{sample_idx}] transport error: {exc}")
                result = {"errors": [str(exc)], "pass": False, "checks": {}}
                paragraph = None
            else:
                result = score_one(payload, paragraph)
            scored_runs.append(result)
            samples.append(
                {
                    "id": case["id"],
                    "sample": sample_idx,
                    "answer": paragraph,
                    "checks": result.get("checks", {}),
                    "errors": result.get("errors", []),
                }
            )
            flag = "ok " if result.get("pass") else "FAIL"
            print(f"  {flag} {case['id']:28s} [{sample_idx}] checks={result.get('checks', {})}")
            for err in result.get("errors", []):
                print(f"         - {err}")
        k = sum(1 for r in scored_runs if r.get("pass"))
        per_case[case["id"]] = k / repeat
        runs[case["id"]] = scored_runs
        print(f"       {case['id']:28s} {k}/{repeat} sample(s) passed")
    return EvalResult(per_case=per_case, runs=runs, latencies=[]), samples


def write_record(result: EvalResult, model: str, n_scenarios: int, repeat: int) -> pathlib.Path:
    today = datetime.date.today().isoformat()
    path = PROMPTS_DIR / f"write-side-{today}.md"
    answers_path = path.with_suffix("").with_suffix(".answers.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    per_check_rates: dict[str, list[bool]] = {}
    for case_runs in result.runs.values():
        for r in case_runs:
            for check, ok in r.get("checks", {}).items():
                per_check_rates.setdefault(check, []).append(ok)
    lines = [
        f"annotate: model={model}, scenarios={n_scenarios}, repeat={repeat},"
        f" trials={n_scenarios * repeat}, overall={result.overall:.3f}",
        f"raw answers and per-sample checks: {answers_path.name}",
        "",
        "| scenario | pass rate (k/N) |",
        "| --- | --- |",
    ]
    for case_id, score in result.per_case.items():
        k = round(score * repeat)
        lines.append(f"| {case_id} | {score:.2f} ({k}/{repeat}) |")
    lines += ["", "| check | pass rate |", "| --- | --- |"]
    for check, oks in per_check_rates.items():
        rate = sum(oks) / len(oks) if oks else 0.0
        lines.append(f"| {check} | {rate:.2f} ({sum(oks)}/{len(oks)}) |")
    existing = (
        path.read_text() if path.is_file() else f"# Write-side skills acceptance -- {today}\n\n"
    )
    path.write_text(existing + "\n".join(lines) + "\n\n")
    return path


def write_answers_sidecar(samples: list[dict]) -> pathlib.Path:
    """Every --live sample's raw paragraph and per-check result, so a
    failing trial can be examined after the fact instead of re-asked (which
    a varying live model answers differently the second time)."""
    today = datetime.date.today().isoformat()
    path = PROMPTS_DIR / f"write-side-{today}.answers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = json.loads(path.read_text()) if path.is_file() else []
    existing.extend(samples)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="score committed fixtures, no model")
    mode.add_argument("--live", action="store_true", help="score a live model (acceptance run)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat", type=int, default=1, help="samples per scenario, --live only")
    args = ap.parse_args()

    cases = load_cases()
    print(f"annotate eval: {len(cases)} case(s), mode={'live' if args.live else 'offline'}\n")

    if args.offline:
        result, _samples = run_offline(cases)
    else:
        result, samples = run_live(cases, args.model, args.repeat)

    n_trials = len(cases) if args.offline else len(cases) * args.repeat
    print(f"\n  overall  {result.overall:.3f}  ({n_trials} trial(s))")

    if args.live:
        model = args.model or chat_model()
        answers_path = write_answers_sidecar(samples)
        record_path = write_record(result, model, len(cases), args.repeat)
        print(f"  wrote {answers_path}")
        print(f"  wrote {record_path}")

    return 0 if result.overall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
