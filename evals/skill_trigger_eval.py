"""Scoring for a skill's trigger section: does it get the model to call the
right tool (or correctly call none), model-free where the router already
settles it.

A trial is a realistic user turn against ONE skill's SKILL.md. The model
answers `{"tool": <namespaced tool name or null>, "args": {...}}`. `score_one`
checks the decision against ground truth:

  * For the four router skills (feed, provenance, knowledge, symbolic), ground
    truth for `expect_tool="<ns>.ask"` is NOT re-derived from `routing.py`
    here -- the router itself is pure, deterministic, and already covered by
    `tests/test_ask_routing.py`. What THIS eval measures is the step upstream
    of the router: whether the model, holding the skill's trigger text in
    context, reaches for `<ns>.ask` (or the correct specific tool / no tool at
    all for a case outside the surface) rather than inventing a call, picking
    a wrong specific tool, or refusing. That is the live gap: real `hermes
    chat` sessions call real tools (confirmed via session logs) but the small
    model driving the loop is not reliably choosing the RIGHT one.
  * For record/annotate/setup (no `.ask` router) ground truth is the tool (or
    null) the skill's own prose names for that scenario, cross-checked by
    hand against the skill body -- there is no pure function to defer to.

`expect_args_contains` is a light, sentinel-based check (never an exact-value
match, since a live model paraphrases): `"not_placeholder"` fails only the
literal string `"user"` (the measured failure mode), `"present"` only checks
the key exists with a non-empty value.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import time
from collections.abc import Callable

CASES_PATH = pathlib.Path(__file__).parent / "skill_trigger_cases.json"
SKILLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "attestation" / "skills"
SPLITS = ("train", "dev")

# skill short name -> (skill dir name, [section headers to include, in order])
# This is the OPTIMIZATION TARGET per skill: the text that governs "what tool
# do I reach for", not the whole SKILL.md (which has a hard size ceiling --
# tests/test_skill_files.py -- and most of the body is reference material for
# once a tool has already been picked).
SKILL_SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "feed": ("attestation-feed", ("## When NOT to use this", "## Ask the router first")),
    "provenance": (
        "attestation-provenance",
        ("## When NOT to use this", "## Ask the router first"),
    ),
    "knowledge": (
        "attestation-knowledge",
        ("## When NOT to use this", "## Ask the router first"),
    ),
    "symbolic": (
        "attestation-symbolic",
        ("## When NOT to use this", "## Ask the router first"),
    ),
    "record": ("attestation-record", ("## Run one command", "## When NOT to use this")),
    "annotate": (
        "attestation-annotate",
        ("## When NOT to use this", "## Put a claim beside every decimal"),
    ),
    "setup": ("attestation-setup", ()),  # frontmatter+intro only; see default_section_text
}

# The default instruction text per skill, extracted from the shipped SKILL.md
# at import time -- exactly DEFAULT_TAG_INSTRUCTION's role for tagging. GEPA
# mutates a copy of this string; run_skill_trigger_eval.py's --offline/--live
# and the optimizer both go through `trigger_messages`, so every score is a
# score of what a real session would actually see.


def _skill_md(skill: str) -> str:
    dir_name = SKILL_SECTIONS[skill][0]
    return (SKILLS_DIR / dir_name / "SKILL.md").read_text()


def _extract_sections(text: str, headers: tuple[str, ...]) -> str:
    """Concatenate named `## ` sections, each up to the next `## ` heading."""
    parts = []
    for header in headers:
        start = text.find(header)
        if start == -1:
            continue
        rest = text[start:]
        nxt = rest.find("\n## ", len(header))
        parts.append(rest if nxt == -1 else rest[:nxt])
    return "\n\n".join(parts)


def default_section_text(skill: str) -> str:
    """The as-shipped trigger text for `skill` -- the baseline every candidate
    is compared against."""
    text = _skill_md(skill)
    headers = SKILL_SECTIONS[skill][1]
    if not headers:
        # setup has no single named section; the first two `## ` blocks after
        # the intro are its whole trigger surface (see SKILL.md structure).
        first = text.index("\n## ")
        second_search = text.find("\n## ", first + 1)
        third = text.find("\n## ", second_search + 1) if second_search != -1 else -1
        end = third if third != -1 else len(text)
        return text[:end]
    return _extract_sections(text, headers)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": ["string", "null"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


def trigger_messages(skill: str, question: str, section_text: str | None = None) -> list[dict]:
    """The exact messages a live decision would see: the skill's trigger
    section as system context, the user's question, and a fixed instruction
    for the output shape (kept OUT of the optimized text, same reasoning as
    `tag_messages` keeping the vocabulary and JSON-shape instruction separate
    from the mutable instruction paragraph)."""
    body = section_text if section_text is not None else default_section_text(skill)
    system = (
        body + "\n\nYou are deciding which tool to call, not calling it. Answer with a JSON"
        ' object {"tool": "<namespaced tool name, or null if none of your tools'
        ' apply>", "args": {...}}. Use the exact tool names named above. If the'
        " question is outside what your tools cover, answer with tool: null."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _check_args(args: dict, expect: dict) -> list[str]:
    errors = []
    for key, sentinel in expect.items():
        val = args.get(key) if isinstance(args, dict) else None
        if sentinel == "not_placeholder":
            if not val or str(val).strip().lower() == "user":
                errors.append(f"args[{key!r}]={val!r} is missing or the literal placeholder")
        elif sentinel == "present":
            if val in (None, "", []):
                errors.append(f"args[{key!r}] is missing or empty, expected something present")
        else:
            if val != sentinel:
                errors.append(f"args[{key!r}]={val!r}, expected {sentinel!r}")
    return errors


def score_one(case: dict, out: dict) -> dict:
    """Score one trial. Returns {"id", "score", "errors", "tool"}."""
    result: dict = {"id": case["id"], "errors": []}
    tool = out.get("tool") if isinstance(out, dict) else None
    args = out.get("args") if isinstance(out, dict) else {}
    if not isinstance(args, dict):
        args = {}
    result["tool"] = tool

    points, total = 0.0, 0.0

    total += 1
    expected = case["expect_tool"]
    if tool == expected:
        points += 1
    else:
        result["errors"].append(f"called {tool!r}, expected {expected!r}")

    expect_args = case.get("expect_args_contains")
    if expect_args:
        total += 1
        arg_errors = _check_args(args, expect_args)
        if not arg_errors:
            points += 1
        else:
            result["errors"].extend(arg_errors)

    result["score"] = points / total if total else 1.0
    return result


def load_cases(path: pathlib.Path = CASES_PATH, split: str | None = None) -> list[dict]:
    cases = json.loads(path.read_text())
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        cases = [c for c in cases if c["split"] == split]
    return cases


def cases_for_skill(cases: list[dict], skill: str) -> list[dict]:
    return [c for c in cases if c["skill"] == skill]


@dataclasses.dataclass
class EvalResult:
    per_case: dict[str, float]
    runs: dict[str, list[dict]]
    latencies: list[float]
    tags: list[str] = dataclasses.field(default_factory=list)

    @property
    def overall(self) -> float:
        return statistics.mean(self.per_case.values()) if self.per_case else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


def evaluate(
    chat_json: Callable[[list[dict], dict], dict],
    cases: list[dict],
    section_text: str | None = None,
    *,
    skill: str | None = None,
    repeat: int = 1,
) -> EvalResult:
    """Score `section_text` (or each case's own skill's default) on `cases`
    through the real renderer. When `section_text` is given, `skill` selects
    which cases it applies to; cases for other skills are scored against
    THEIR OWN shipped default (so a mixed-skill case list still scores
    sensibly when only one skill's candidate is being evaluated)."""
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
                out = chat_json(messages, TOOL_SCHEMA)
            except Exception as exc:  # noqa: BLE001 - a transport error costs one trial
                out = {"_transport_error": str(exc)}
            latencies.append(time.perf_counter() - t0)
            scored_runs.append(score_one(case, out))
        mean = statistics.mean(r["score"] for r in scored_runs)
        per_case[case["id"]] = mean
        runs[case["id"]] = scored_runs
    return EvalResult(per_case=per_case, runs=runs, latencies=latencies)
