#!/usr/bin/env python
"""Score the attestation-record skill: what a model writes to disk after
running an experiment, checked by the REAL ledger reader.

    uv run python evals/run_record_eval.py --offline          # scorer only, no model
    uv run python evals/run_record_eval.py --command          # real `attest runs record` CLI
    uv run python evals/run_record_eval.py --live             # acceptance run
    uv run python evals/run_record_eval.py --live --repeat 5  # 5 samples/scenario

`--offline` scores the committed `record_cases.json` fixtures -- each case
carries a scenario and a hand-written, correct-by-construction `answer`
manifest, plus one deliberately-bad case marked `expect_fail: true` (proves
the scorer can fail things, not just pass them). This is what CI runs: no
model touched, `record_eval.score_one` exercised against known inputs. It
never writes under `evals/prompts/` -- that is the live acceptance's job.

`--command` is the CLI's own acceptance test: it builds each scenario's
`attest runs record --dry-run` argument list (`scenario_argv`), runs
`attestation.cli.main()` as a subprocess of THIS interpreter (`sys.
executable -c ...`, not a bare `attest` PATH lookup -- pinned so a stray
same-named binary elsewhere on a caller's PATH cannot be scored by
accident), parses the manifest it prints, and scores it with the same
`record_eval.score_one` -- no model, no hand-written answer, fully
deterministic. This proves the command a user actually types produces a
ledger-readable manifest, not just that `record.plan()` can.

`--live` sends the `attestation-record` SKILL.md body (read by path, not
imported -- the other half of this spec's work may not exist yet) plus each
scenario's prompt to a live model, parses its manifest, and scores it the
same way. Requires a model server at LLM_BASE_URL (default model
`gemma4:e2b-it-q4_K_M`, temp 0). Ollama's gemma4:e2b now thinks by default;
`llm.ChatClient.chat_json` already sends `reasoning_effort: "none"` on the
OpenAI-compatible endpoint for exactly this reason (see its docstring), so
this driver reuses it rather than hand-rolling a native `/api/chat` call.

`--repeat N` (default 1) draws N samples per scenario -- a live model varies
between calls, so a single sample cannot distinguish "the skill sometimes
fails this" from "this model, this once, failed this"; "N trials" then means
scenarios x repeat. The overall score is the mean over every sample, and the
written record additionally reports each scenario's own k/N pass count.

Writes a dated record to `evals/prompts/write-side-YYYY-MM-DD.md`, mirroring
`examples/flows`'s RESULTS.md rule: the committed artifact is the number, and
only a live run may write it. Every sample's raw answer, per-check result,
and errors are ALSO written to a sidecar `write-side-YYYY-MM-DD.answers.json`
-- the .md alone could not answer "what did the model write on the failing
sample", which is exactly what made a varying live run unexaminable after
the fact.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
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

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"

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

# The sandbox's real constraint, not skill guidance: the skill correctly
# tells a real agent to declare an undeclared metric's direction at
# ~/.hermes/metric_direction.toml, but this harness's scorer sandboxes every
# written path (record_eval.py rejects an absolute path outright) and points
# LEDGER_METRIC_DIRECTION_FILE at <project>/metric_direction.toml instead.
# Telling the model this is harness alignment -- where THIS SANDBOX reads the
# file from -- not a coaching hint about the check itself; the check still
# scores whatever the model actually writes.
SANDBOX_DIRECTION_NOTE = (
    " In this sandbox, write any [metric_direction] declaration to"
    " metric_direction.toml at the project root (the harness points"
    " LEDGER_METRIC_DIRECTION_FILE at it) instead of ~/.hermes/."
)

PROMPT_TEMPLATE = (
    "You just ran an experiment named {family} with arms {arms}. The final"
    " {metric} values were: {values_str}. The corpus was {corpus}."
    " Following the skill above, write the files you would leave on disk to"
    ' record this. Answer with a JSON object {{"files": {{relpath: content}}}}'
    " -- paths relative to one project directory (e.g. results/..., configs/...)."
    + SANDBOX_DIRECTION_NOTE
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


def run_offline(cases: list[dict]) -> tuple[EvalResult, list[dict]]:
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    samples: list[dict] = []
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
        samples.append(
            {
                "id": case["id"],
                "sample": 0,
                "answer": case["answer"],
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

    client = ChatClient(model=model)
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    samples: list[dict] = []
    for case in cases:
        messages = [
            {"role": "system", "content": skill_body},
            {"role": "user", "content": scenario_prompt(case)},
        ]
        scored_runs = []
        for sample_idx in range(repeat):
            try:
                answer = client.chat_json(messages, MANIFEST_SCHEMA)
            except Exception as exc:  # noqa: BLE001 - a transport failure costs one sample
                print(f"  FAIL {case['id']:28s} [{sample_idx}] transport error: {exc}")
                result = {"errors": [str(exc)], "pass": False, "checks": {}}
                answer = None
            else:
                result = score_one(case, answer)
            scored_runs.append(result)
            samples.append(
                {
                    "id": case["id"],
                    "sample": sample_idx,
                    "answer": answer,
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


def _bare_arm_name(family: str, arm: str) -> str:
    """`scenario["arms"]` entries are already full stems (`asr_baseline`,
    matching `record_cases.json`'s own hand-written `results/asr_baseline.
    json` answers), but `attest runs record FAMILY --arm NAME ...` writes
    `results/<FAMILY>_<NAME>.json` -- it prefixes NAME with FAMILY itself.
    Passing the full stem as NAME would double the prefix
    (`results/asr_asr_baseline.json`), which `family_of` groups under
    `asr-asr`, not the scenario's own `asr`. Stripping the family's leading
    `family_` here is what makes `--command` type the same NAME a human
    reading the skill's `kdsweep_t4`-style example would type.
    """
    prefix = f"{family}_"
    return arm[len(prefix) :] if arm.startswith(prefix) else arm


def scenario_argv(scenario: dict) -> list[str]:
    """The `attest runs record --dry-run` argument list for `scenario`,
    mirroring exactly what a human following the skill's now-first example
    would type: one `--arm NAME METRIC=VALUE` per arm, `--corpus`, and a
    `--direction METRIC=...` whenever the scenario's metric is not already
    in `ledger.METRIC_DIRECTION` (`built_in: false`) -- using the direction
    the scenario itself carries as ground truth. The one committed
    `expect_fail` scenario (`bait-missing-direction`) is `built_in: false`
    but deliberately gets NO `--direction`: it exists to prove the command
    itself refuses when a declaration is owed and withheld, so building its
    argv as if the caller had remembered would test nothing.
    """
    argv = ["runs", "record", scenario["family"], "--dry-run", "--corpus", scenario["corpus"]]
    for arm in scenario["arms"]:
        name = _bare_arm_name(scenario["family"], arm)
        pair = f"{scenario['metric']}={scenario['values'][arm]}"
        argv += ["--arm", name, pair]
    if not scenario.get("built_in", True) and not scenario.get("expect_fail"):
        argv += ["--direction", f"{scenario['metric']}={scenario['direction']}"]
    return argv


def run_command(cases: list[dict]) -> tuple[EvalResult, list[dict]]:
    """Score the REAL `attest runs record --dry-run` command, not the Python
    function underneath it -- the acceptance the spec names: build each
    scenario's argv, run the installed console script, parse the printed
    manifest, and score it with the identical `record_eval.score_one` the
    other two modes use. No model is involved and the result is
    deterministic, so this is what CI (and the coordinator's own gate) runs.

    Invoked as `sys.executable -c "import attestation.cli; ...main()"`
    rather than a bare `["attest", *argv]` PATH lookup (the final review's
    Important 4): a bare `attest` scores whatever `attest` happens to
    resolve to first on the CALLER's PATH, silently, which is not
    necessarily this checkout's own code -- the same pattern
    `tests/test_architecture.py`'s import-cost timing already uses to pin
    the CLI to THIS interpreter's installed `attestation` package.
    """
    per_case: dict[str, float] = {}
    runs: dict[str, list[dict]] = {}
    samples: list[dict] = []
    for case in cases:
        argv = scenario_argv(case)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import attestation.cli; sys.exit(attestation.cli.main())",
                *argv,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        expect_fail = case.get("expect_fail", False)
        if proc.returncode != 0:
            ok = expect_fail
            result = {
                "errors": [proc.stderr.strip() or proc.stdout.strip()],
                "pass": False,
                "checks": {},
            }
            answer = None
        else:
            try:
                answer = json.loads(proc.stdout)["files"]
            except (json.JSONDecodeError, KeyError) as exc:
                ok = False
                result = {
                    "errors": [f"could not parse --dry-run output: {exc}"],
                    "pass": False,
                    "checks": {},
                }
                answer = None
            else:
                result = score_one(case, {"files": answer})
                ok = (not result["pass"]) if expect_fail else result["pass"]
        per_case[case["id"]] = 1.0 if ok else 0.0
        runs[case["id"]] = [result]
        flag = "ok " if ok else "FAIL"
        print(f"  {flag} {case['id']:28s} ok={ok} expect_fail={expect_fail}")
        for err in result.get("errors", []):
            print(f"         - {err}")
        samples.append(
            {
                "id": case["id"],
                "sample": 0,
                "answer": answer,
                "checks": result.get("checks", {}),
                "errors": result.get("errors", []),
            }
        )
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
        f"# Write-side skills acceptance -- {today}",
        "",
        f"record: model={model}, scenarios={n_scenarios}, repeat={repeat},"
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
    existing = path.read_text() if path.is_file() else ""
    path.write_text(existing + "\n".join(lines) + "\n\n")
    return path


def write_answers_sidecar(samples: list[dict]) -> pathlib.Path:
    """Every --live sample's raw answer and per-check result, so a failing
    trial can be examined after the fact instead of re-asked (which a
    varying live model answers differently the second time)."""
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
    mode.add_argument(
        "--command", action="store_true", help="score the real `attest runs record` CLI"
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat", type=int, default=1, help="samples per scenario, --live only")
    args = ap.parse_args()

    cases = load_cases()
    mode_name = "live" if args.live else "command" if args.command else "offline"
    print(f"record eval: {len(cases)} case(s), mode={mode_name}\n")

    if args.offline:
        result, _samples = run_offline(cases)
    elif args.command:
        result, _samples = run_command(cases)
    else:
        result, samples = run_live(cases, args.model, args.repeat)

    n_trials = len(cases) if not args.live else len(cases) * args.repeat
    print(f"\n  overall  {result.overall:.3f}  ({n_trials} trial(s))")

    if args.live:
        from attestation.llm import chat_model

        model = args.model or chat_model()
        answers_path = write_answers_sidecar(samples)
        record_path = write_record(result, model, len(cases), args.repeat)
        print(f"  wrote {answers_path}")
        print(f"  wrote {record_path}")

    return 0 if result.overall == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
