# Skill optimization with live rollouts: feed and knowledge through agentopt

**Date:** 2026-09-05
**Status:** spec 3 of 3 from the 2026-09-05 brainstorm; the user chose
"knowledge + feed, more cases, ship only what passes the existing gate" and
asked for `~/agent-loop-optimizer` (agentopt) to be the measurement. Executed
autonomously overnight; the numbers in §5 are what was measured, and an
empty §5 row means the run did not finish before morning.
**Depends on:** agentopt at `3193e35` (its fixtures for the four
`ATTEST_TOOLS` router servers, `--fixture`, `--skill`, `--bank`,
`--results-dir`, `rescore`), `evals/skill_trigger_cases.json` (upstream case
format, read unchanged), the two library specs of the same date (the
knowledge skill now names `cite.sync` and `cite.related`).

## Problem

`evals/optimize_skill_triggers.py` scores one cheap model call per case and
compares tool names as strings. On 2026-09-03 it improved feed and symbolic
on that proxy and both candidates failed the transfer gate. It cannot see the
failures that matter on the deployed surface: gemma4:e2b clarifying instead
of calling the router, calling `runs.ask` without the family it was told,
sending "what are my main areas" to Hermes's built-in `memory` tool, or
receiving a router's `options` reply and giving up. agentopt runs the real
Hermes against the real MCP server in an isolated home and scores the
transcript with model-free predicates, which is the measurement this repo's
`docs/measurement-lessons.md` keeps saying to take.

## Decisions

- **agentopt is the measurement; attestation supplies cases, seeds and the
  fixture database.** No agentopt code changes from this side; anything it
  needs is a message to its session. Runs use `--bank` and `--results-dir`
  under this checkout's `.venv/agentopt/`, one bank per skill, so nothing
  lands in the optimizer's tree.
- **The deployed surface is what is scored.** Fixtures
  `attestation-feed` / `attestation-knowledge` expose the router pair only,
  exactly as the Discord gateway serves them since 2026-09-04. A case
  expecting a tool the collapsed surface hides is a wrong case, not a wrong
  skill.
- **Every case names a measured failure or a real reader phrasing.** The 15
  added cases (8 feed, 7 knowledge) are the 2026-09-04 Discord refusals, the
  agentopt 2026-09-05 findings, and molecular-AI phrasings a chemistry reader
  would type. Two are negatives (`expect_tool: null`) where the right answer
  is a hand-off to another skill.
- **Hand edits to the seed are measured too.** Three sentences were added
  to the skills for the three measured failures (§3) before any optimizer
  run; the baseline row in §5 is the seed WITH them, and the prior seed is
  kept as `seed-raw` by agentopt so the edit's effect is a number.
- **Acceptance is unchanged.** `tagging_eval.gate()`: not worse on the
  primary, better on at least two other models, no wider spread. agentopt's
  own acceptance (paired LCB > 0 at α = 0.2 on the dev split, k ≥ 5) is the
  live-rollout form of the first rule; the transfer models are
  `gemma4:e4b` and `qwen3.5:9b`. A candidate ships only when both hold.

## 1. Cases

Added to `evals/skill_trigger_cases.json`, upstream format, ids
`feed-ranked-today-molecular`, `feed-daily-feed`, `feed-new-on-mlips`,
`feed-equivariant-search`, `feed-rate-dft-noise`, `feed-add-chemph`,
`feed-why-catalysis`, `feed-not-provenance`, `knowledge-main-areas`,
`knowledge-connect-forcefields-catalysis`, `knowledge-between-md-drug`,
`knowledge-concepts-protein`, `knowledge-clusters`, `knowledge-central-hub`,
`knowledge-not-summarise`. Splits: 8 train / 7 dev, negatives on dev. The
offline validator (`evals/run_skill_trigger_eval.py --offline`) checks each
router case against `attestation.mcp.routing` before any model sees it.

## 2. The fixture database

`ATTEST_DB` points at a copy of the live database (9,401 items, real clicks)
taken 2026-09-05 03:22 into `.venv/agentopt/seed.db`. A copy, not the live
file: agentopt refuses to launch an attestation fixture without `ATTEST_DB`,
and a rollout that routes to `feed.rate` writes a click.

## 3. Seed edits (before optimizing)

- `attestation-knowledge`: "my main research areas / what do I read about"
  is `kg.ask`, never a memory or notes tool; the graph is the reader's
  record.
- `attestation-feed` and `attestation-knowledge`: a router reply with
  `ok: false` and `options` means ask the reader which, or re-ask the router
  with the question reworded for the option that plainly fits ("today's
  ranked feed" is `feed.list`); never call the router with empty arguments
  and never say you lack a tool.
- `attestation-feed`: a question about experiment arms, sweeps or a draft's
  numbers is `attestation-provenance`'s; hand off rather than searching the
  feed for a sweep name.

## 4. Runs

From this checkout, `ATTESTATION_DIR` at this worktree so the fixtures
launch this branch's `attest-mcp`:

```
agentopt run       --skill feed      --fixture attestation-feed      --k 3   # baseline, both seeds
agentopt run       --skill knowledge --fixture attestation-knowledge --k 3
agentopt calibrate --skill feed      --fixture attestation-feed      --k 3   # proxy trust
agentopt optimize  --skill knowledge --fixture attestation-knowledge --reflection-lm ollama_chat/qwen3.5:9b --max-live-rollouts 120
agentopt report / export
```

Order: knowledge first for `optimize` because its case set is smallest and
its measured failure (the `memory` tool) is the clearest; feed's optimize
runs if the night allows. Transfer on `gemma4:e4b` and `qwen3.5:9b` with
`agentopt run --model <m>` against the exported candidate.

## 5. Measured

Filled in as runs finish; see `.venv/agentopt/results/` for agentopt's own
dated reports and `evals/prompts/` for any exported candidate.

| skill | candidate | k | cases | mean | calls gate | notes |
|---|---|---|---|---|---|---|
| feed | seed-raw (2026-09-05 agentopt, before this spec) | 1 | 7 | 0.24 | 2/7 | peer's smoke |
| feed | seed (same) | 1 | 7 | 0.48 | 4/7 | |
| knowledge | seed (same) | 1 | 4 | 0.42 | 2/4 | main-areas went to `memory` |

## What this spec does not decide

Whether `kg.ask` gets a route to `cite.related`; symbolic's expression
parser (a router gap agentopt found: `integral(x^2, x, 0, 1)` and `sin^2(x)`
fail to parse); provenance's `family` argument, which is a case that must
keep failing until the provenance skill fixes it.
