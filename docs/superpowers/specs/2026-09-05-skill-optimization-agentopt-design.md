# Skill optimization with live rollouts: feed and knowledge through agentopt

**Date:** 2026-09-05
**Status:** spec 3 of 3 from the 2026-09-05 brainstorm; the user chose
"knowledge + feed, more cases, ship only what passes the existing gate" and
asked for `~/agent-loop-optimizer` (agentopt) to be the measurement. Executed
autonomously overnight; the numbers in §5 are what was measured, rescored
after review round 1 against agentopt's decline-aware scorer. **Result:
measured, shipped nothing** (§6).
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
  added cases (9 feed, 6 knowledge; `knowledge-main-areas` was already on
  main) are the 2026-09-04 Discord refusals, the agentopt 2026-09-05
  findings, molecular-AI phrasings a chemistry reader would type, and --
  after review round 1 -- one ambiguity case (`no_clarify: false`) so the
  options rule can regress in the other direction and be seen. Two are
  negatives (`expect_tool: null`) where the right answer is a hand-off to
  another skill.
- **Hand edits to the seed were NOT measured, and this spec first said they
  were.** Three sentences were added to the skills for the three measured
  failures (§3) before any optimizer run. agentopt's `seed-raw` differs from
  `seed` only in tool-name canonicalization (dotted vs registered names);
  both banked seeds contain the edits byte for byte (review round 1). The
  seed-raw/seed gap in §5 is therefore a canonicalization measurement.
  Measuring the edits needs a run on the pre-edit text, which has not
  happened; the edits stay because they name concrete values, but they are
  unmeasured prose.
- **Acceptance is unchanged, and was not met.** `tagging_eval.gate()`: not
  worse on the primary, better on at least two other models, no wider
  spread. agentopt's acceptance race runs on TRAIN minibatches (k 3..8);
  its pre-registered dev check (paired LCB > 0 at α = 0.2, k ≥ 5) is a
  separate step that never ran here (dev k = 3). The transfer models are
  `gemma4:e4b` and `qwen3.5:9b`; no rollout on either exists in the banks.
  A candidate ships only when both hold; none did.

## 1. Cases

Added to `evals/skill_trigger_cases.json`, upstream format, ids
`feed-ranked-today-molecular`, `feed-daily-feed`, `feed-new-on-mlips`,
`feed-equivariant-search`, `feed-rate-dft-noise`, `feed-add-chemph`,
`feed-why-catalysis`, `feed-not-provenance`, `knowledge-main-areas`,
`knowledge-connect-forcefields-catalysis`, `knowledge-between-md-drug`,
`knowledge-concepts-protein`, `knowledge-clusters`, `knowledge-central-hub`,
`knowledge-not-summarise`, plus `feed-ambiguous-followup` from review
round 1 (31 -> 46 ids). Splits: 6 train / 9 dev, negatives on dev. The
offline validator (`evals/run_skill_trigger_eval.py --offline`) checks each
router case against `attestation.mcp.routing` before any model sees it --
that the router does not RAISE, not that it routes right: on the current
router "which topic is the hub" and "list the concepts I have about
proteins" both reach `kg.concepts` with no prefix, and "find me the papers
behind the 'diffusion-models' concept" too; those are `routing.py` gaps a
perfect rollout would still score 1.0 on. `knowledge-vs-feed-search`
expected `feed.search` until review round 1, a tool the collapsed knowledge
surface does not serve: it scored 0 on all seven candidates by construction
and pinned one sixth of every dev mean below. It now expects `kg.ask`.

## 2. The fixture database

`ATTEST_DB` points at a copy of the live database (9,407 items, real clicks)
taken 2026-09-05 03:22 into `.venv/agentopt/seed.db`. A copy, not the live
file: agentopt refuses to launch an attestation fixture without `ATTEST_DB`,
and a rollout that routes to `feed.rate` writes a click -- and the copy now
holds autocreated `me` and `user` personas, because the fixture's Hermes
home names no persona and the model passed a placeholder on the daily-feed
cases, the un-prompted path measurement-lessons already records as 0/9
versus 9/9 with the persona named. The fixture measures a path the deployed
gateway does not take; a `persona` field in agentopt's fixture is an open
question on its side.

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

Overnight 2026-09-05 on gemma4:e2b through the real Hermes and the real
`attest-mcp` (this worktree), rescored 2026-09-05 afternoon against agentopt
main after its scorer stopped crediting a relayed router decline as an
answer (review round 1 found `feed-ranked-today-molecular` scoring 1.0 on
"Which one would you like me to do?", the exact failure the case was
written for). Banks and agentopt's dated reports are under
`.venv/agentopt/`; CIs are agentopt's at α = 0.2.

| skill | candidate | k | cases | mean (CI) | calls gate | notes |
|---|---|---|---|---|---|---|
| feed | seed-raw `e3411564` | 1 | 15 | 0.51 [0.37, 0.66] n=15 | 9/15 | 0.56 before rescore; six positives never called `feed.ask` (rate-implicit, explain, capability, digest-lately, rate-dft-noise, add-chemph); `user='user'` on daily-feed |
| feed | seed `fd6804ce` | 1 | 15 | 0.51 [0.37, 0.64] n=15 | 10/15 | paired vs seed-raw: +0.00 with a ±0.12 interval -- not a number at k = 1; six of fifteen cases move by 0.33 or more between the two |
| feed | calibrate / optimize | -- | -- | -- | -- | did not run |
| knowledge | seed-raw `12e08030` | 3 | 10 | 0.51 [0.41, 0.61] n=30 (dev 0.48 n=18) | 7/10 | main-areas 0/3, central-hub 0/3, vs-feed-search 0/3 (tool hidden, see §1) |
| knowledge | seed `1dbdede5` | 3 (train topped to 8 by races) | 10 | 0.44 [0.37, 0.52] n=50 (dev 0.43 n=18) | 7/10 | paired vs seed-raw: −0.07, LCB −0.18; main-areas 0/8 -- the edit did not move the case it named |
| knowledge | `826e5e38` (`kg.tools` description 367 -> 3,803 chars) | 3 dev / 8 train | 9 | dev 0.61 n=18; train 0.94 n=24 | 8/9 | accepted by GEPA on a 3-case TRAIN race; dev k = 3 < 5; contains invented statistics ("measured to be correct 26/26 times"); no second model |
| knowledge | `9dbc7d8c` (`kg.ask` + `kg.tools` descriptions) | 8 | 3 train | 0.89 n=24 | 3/3 | never scored on dev; race vs `826e` undecided |
| both | transfer gemma4:e4b, qwen3.5:9b | -- | -- | -- | -- | not run: the candidate cannot ship (below), so the runs were cancelled |

Budget: 150 live rollouts against `--max-live-rollouts 120` (agentopt
overshoots to finish a race); knowledge optimize 04:32-07:15, feed baseline
07:15-07:38.

## 6. Result: measured, shipped nothing

- **The one accepted candidate changes only `kg.tools`' tool description**
  -- ten times longer, carrying statistics the reflection model invented
  and a "Failure Prevention Checklist" -- and its exported `SKILL.md` is
  byte-identical to the seed. The gain, real or not, lives in a string the
  MCP server serves, not in the skill; a candidate whose gain is in a tool
  description is telling us the shipped description is the weak part, and
  the fix belongs in `mcp/knowledge.py`'s docstring under its own
  measurement, not in `SKILL.md`. Invented numbers in a description loaded
  into every Hermes turn are disqualifying on their own; agentopt has
  logged the proposer-side rule (no numbers absent from the reflective
  dataset) on its side.
- **What the scorer cannot see, listed so the next run does not trust it:**
  a negative case (`feed-not-provenance`, `knowledge-not-summarise`) scores
  1.0 on `calls_none` alone, so a correct hand-off and the pseudo-call
  `session_search(query="kdsweep sweep WER")` printed as prose score the
  same (rollout ids relayed to agentopt; a `mentions_any` predicate or a
  judged one is an open question there); the fixture names no persona (§2);
  the offline validator checks that the router does not raise, not that it
  routes right (§1).
- **What changed on this branch because of the measurement:** the
  unsatisfiable dev case now expects a tool the surface serves; the
  main-areas sentence in the knowledge skill is the short form that names
  the values ("what do I read about most" is `kg.ask`; "I do not have
  access to your reading" is never true with `kg.ask` in the tool list)
  rather than a narrative about a memory tool the transcripts do not show;
  and `feed-ambiguous-followup` exists so the options rule ("when one
  option plainly fits, re-ask the router") has a case that fails when the
  model re-asks instead of asking.
- **The honest one sentence.** On gemma4:e2b through the real path, the
  hand edits were not measured; the feed baseline is 0.51 either way with
  a ±0.12 paired interval and no calibration; the knowledge seed scored
  0.44 against 0.51 for the same text under dotted names (not significant
  at α = 0.2); and GEPA's single accepted candidate is +0.18 on six dev
  cases at k = 3 (one of them unsatisfiable), below the pre-registered
  k ≥ 5, on one model, with fabricated text -- so `tagging_eval.gate()`
  cannot be evaluated from anything on disk and nothing meets the bar.

## What this spec does not decide

Whether `kg.ask` gets a route to `cite.related`; the three `route_kg` gaps
§1 names (hub -> `kg.central`, "concepts about X" -> `kg.concepts(prefix)`,
"papers behind a concept" -> `feed.search` or a `kg.ask` answer that names
it); symbolic's expression parser (a router gap agentopt found:
`integral(x^2, x, 0, 1)` and `sin^2(x)` fail to parse); provenance's
`family` argument, which is a case that must keep failing until the
provenance skill fixes it; a measured rewrite of `kg.tools`' description
(the finding in §6); a pre-edit run that would measure the §3 sentences;
and whether the harness should name a persona (agentopt's call).
