# Skill optimization with live rollouts: feed and knowledge through agentopt

**Date:** 2026-09-05
**Status:** spec 3 of 3 from the 2026-09-05 brainstorm; the user chose
"knowledge + feed, more cases, ship only what passes the existing gate" and
asked for `~/agent-loop-optimizer` (agentopt) to be the measurement. Executed
autonomously overnight; the numbers in §5 are what was measured, rescored
after review round 1 against agentopt's decline-aware scorer. **Result:
measured, shipped nothing** (§6).
**Depends on:** agentopt at `1cf0b28` (its fixtures for the four
`ATTEST_TOOLS` router servers, `--fixture`, `--skill`, `--bank`,
`--results-dir`, `rescore`, `run --candidate`), `evals/skill_trigger_cases.json`
(upstream case format plus an agentopt-only `expect` key on one case), the
two library specs of the same date (the
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
  *Measured 2026-09-06 (§5, "hand edit"):* the pre-edit text (the Sep 3
  install) and this branch's text at k = 5 on the same bank differ by less
  than the pre-edit text differs from its own repeat, and the main-areas
  sentence moved its case 0/5 -> 0/5. The edits stay because they name
  concrete values a reader can check, not because they moved a number.
- **Acceptance is unchanged, and was not met.** `tagging_eval.gate()`: not
  worse on the primary, better on at least two other models, no wider
  spread. agentopt's acceptance race runs on TRAIN minibatches (k 3..8);
  its pre-registered dev check (paired LCB > 0 at α = 0.2, k ≥ 5) is a
  separate step that never ran here (dev k = 3). The transfer models are
  `gemma4:e4b` and `qwen3.5:9b`; the only rollouts on either are seven
  seed-raw rollouts on gemma4:e4b from a transfer run cancelled at 13:33
  (`bank-knowledge-transfer`). A candidate ships only when both hold; none
  did.

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
surface does not serve: it scored 0 on the three candidates that ran it (both
seeds and `826e5e38`) by construction and pinned one sixth of every dev mean
below. It now expects `kg.ask`, which is not neutral: the seeds' rollouts on
it called nothing or a browser tool (still 0), while `826e5e38` already
called `kg.ask` in two of three, so the change raises the candidate's dev
mean, not the seeds'. A 1.0 on it today is a concept list for a "papers
behind" question -- satisfiable, honest, weak. `route_kg("what are my main
research areas")` declines with options, so the knowledge skill now leads
its example list with "what do I read about most", which routes.

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
| feed | seed `fd6804ce` | 1 | 15 | 0.51 [0.37, 0.64] n=15 | 10/15 | paired vs seed-raw: +0.00 with a ±0.10 interval -- not a number at k = 1; four of fifteen cases move by 0.33 or more between the two |
| feed | calibrate / optimize | -- | -- | -- | -- | did not run |
| knowledge | seed-raw `12e08030` | 3 | 10 | 0.51 [0.41, 0.61] n=30 (dev 0.48 n=18) | 7/10 | main-areas 0/3, central-hub 0/3, vs-feed-search 0/3 (tool hidden, see §1) |
| knowledge | seed `1dbdede5` | 3 (train topped to 8 by races) | 10 | 0.44 [0.37, 0.52] n=50 (dev 0.43 n=18) | 7/10 | paired vs seed-raw: −0.07, LCB −0.18; main-areas 0/8 -- the edit did not move the case it named |
| knowledge | `826e5e38` (`kg.tools` description 367 -> 3,803 chars) | 3 dev / 8 train | 9 | dev 0.61 n=18; train 0.94 n=24 | 8/9 | accepted by GEPA on a 3-case train minibatch (subsample score 0.5625 -> 2.92; every paired race in the log was undecided); dev paired vs seed +0.185, LCB +0.084 at α = 0.2, but k = 3 < 5; its text carries "26/26" and "0% of the time" -- the seed's own `kg.tools` line (1 in 26 vs 26 in 26) garbled into a correctness claim; no second model |
| knowledge | `9dbc7d8c` (`kg.ask` + `kg.tools` descriptions) | 8 | 3 train | 0.89 n=24 | 3/3 | never scored on dev; race vs `826e` undecided |
| both | transfer gemma4:e4b, qwen3.5:9b | 3 | -- | -- | -- | cancelled at 13:33 after seven seed-raw rollouts on gemma4:e4b (`bank-knowledge-transfer`: connect-topics 3/3, main-areas 0/3); nothing on the candidate, nothing on qwen3.5:9b; the candidate cannot ship (below) |

"calls gate" is cases with at least one rollout passing the gate predicate
(`calls_any` / `calls_none`); strict all-rollouts is 5/10, 3/10 and 6/9.

Budget: 150 live rollouts against `--max-live-rollouts 120` (agentopt
overshoots to finish a race); knowledge optimize 04:32-07:15, feed baseline
07:15-07:38.

### The hand edit, measured (2026-09-06)

agentopt's noise-floor step (its spec §11) on the knowledge skill, run by its
session: three texts of `attestation-knowledge/SKILL.md` at k = 5 over the
same 10 cases in one bank (`knowledge-proof2`), this worktree's `attest-mcp`
(schema 8) on a seeded copy of the database, model unloaded before each arm,
fixture health 0 errored calls and 0 rollouts lost before the agent answered
(a first attempt served the copy from main's schema-6 server, failed inside
every graph call, and scored 0.67 on the partial predicates -- retracted,
quarantined, and the reason every agentopt report now carries a
fixture-health line). Texts are named by sha256 prefix:

| text | sha256 | what it is |
|---|---|---|
| installed | `63460cea` | the Sep 3 `attest install` copy: pre-edit |
| main | `aab7da3d` | adds the "dotted names are for you to read" paragraph |
| worktree | `a5a16e0c` | adds the References rewrite (library specs) and the two §3 sentences |

| pair | paired mean | LCB | UCB | verdict |
|---|---|---|---|---|
| installed vs its own repeat (the noise floor) | +0.027 | −0.041 | +0.094 | spread > 0.3 on 1/10 cases (not-todays-feed, exactly 0.30) |
| worktree − installed ("everything since Sep 3") | −0.005 | −0.067 | +0.057 | undecided; smaller than the floor |
| worktree − installed's repeat | +0.022 | −0.046 | +0.090 | undecided |
| main − installed (dotted-names paragraph alone) | -- | -- | -- | not run: the main arm was stopped at the user's call after 34 of 100 rollouts, because the line above is already inside the noise band and splitting it further was not worth the GPU time; the 34 rollouts stay in the bank |
| worktree − main (References rewrite + the §3 sentences) | -- | -- | -- | not run, same reason |

Per case, worktree / installed at n = 5 / 10: connect-topics 0.40 / 0.33,
main-areas **0.00 / 0.00**, vs-feed-search 0.00 / 0.00, not-todays-feed
0.70 / 1.00, connect-forcefields-catalysis 0.80 / 0.75, between-md-drug
0.73 / 0.67, concepts-protein 0.87 / 0.80, clusters 0.80 / 0.80, central-hub
**0.00 / 0.00**, not-summarise 1.00 / 1.00 (`calls_none` alone; a
`mentions_any` check follows in the next rescore). The row the §3 edit
exists for, main-areas, is 0/5 under the edited text: the sentence did not
move gemma4:e2b off the built-in memory tool at all. Two cases are
deterministic zeros on every text (main-areas, central-hub) -- failures to
fix in the router or the description, not noise to optimize through.

### Feed, two proposers, two students (2026-09-08)

Run by the agentopt session after its scorer, fixture and proposer fixes
(a relayed decline is not an answer; the fixture names a persona; a
proposal carrying a number absent from the reflective data scores 0 and
spends no rollouts): the installed feed `SKILL.md` as seed, the
`attestation-feed` fixture with the `demo-reader` persona, this branch's
16 feed cases (8 train / 8 dev) on the schema-8 seeded copy, `qwen3.5:4b`
as the reflection model for both proposers, every proposal raced against
the seed over all 8 train cases at 5+5 rollouts (paired LCB, α = 0.2),
then scored on dev. Reports: agentopt `results/2026-09-08-feed-gepa-vs-
skillopt.md` and `results/2026-09-08-feed-qwen9b-train-wins-dev-losses.md`.

| student | seed dev (n=40) | proposal | size | train vs seed | dev vs seed |
|---|---|---|---|---|---|
| gemma4:e2b | 0.39 [0.30, 0.47] | GEPA trigger rewrite | 264 -> 2,665 chars | −0.167 [−0.322, −0.011], **rejected** | -- |
| gemma4:e2b | | GEPA body rewrite (shorter by 45%) | 11,678 -> 6,472 | +0.042 [−0.005, +0.089], undecided | -- (digest-lately 0.60 -> 0.87, nothing else) |
| gemma4:e2b | | GEPA examples | 0 -> 2,444 | −0.015, undecided | -- |
| gemma4:e2b | | SkillOpt examples edit | 0 -> 808 | +0.015 [−0.031, +0.061], undecided | -- (explain 0.00 -> 0.13, search-topic 1.00 -> 0.87) |
| qwen3.5:9b | 0.41 [0.33, 0.49] | GEPA trigger rewrite | 264 -> 3,335 | +0.26, LCB +0.17, accepted | 0.00 [−0.07, +0.07] |
| qwen3.5:9b | | GEPA body rewrite | 11,678 -> 5,714 | +0.13, LCB +0.03, accepted | −0.09, UCB −0.01, **worse** |
| qwen3.5:9b | | SkillOpt trigger, three appended rules | 264 -> 1,460 | +0.22, LCB +0.13, accepted | −0.13 [−0.27, +0.01]; not-provenance 0.80 -> 0.20 |

Fixture health clean on both (0 errored fixture calls); on the 9B, 39 of
358 rollouts timed out at 300 s and score 0, unevenly across candidates,
and excluding them changes no verdict. The persona fixture fixed the
placeholder-user failure (persona-name 0.80 with `user="demo-reader"` on
every call, against 0/9 unprompted in this repo's own measurement). Five
of sixteen cases are 0.00 on every gemma rollout (capability,
rate-implicit, ambiguous-followup, explain, rate-dft-noise); two of them
(capability, rate-implicit) moved for no candidate on either student.

What the two students say together: on gemma4:e2b nothing is decided in
either direction except that a tenfold-longer trigger is worse, and the
only near-win *shortened* the body; on qwen3.5:9b every accepted
candidate wins the eight train questions with a closed interval and is
flat or worse on the eight it never saw, and the one whose text can be
read ("feed content queries always require tool use") breaks the hand-off
case by construction. That is overfitting to eight questions, not a lack
of search power, and it is the train/dev gap the pre-registered dev bar
exists to catch. The pre-registered bar (dev k ≥ 5, paired LCB > 0) is
unmet by every candidate on both students.

**Caveat on every number above (found 2026-09-09).** All of these rollouts
ran Hermes with its full local toolset -- terminal, file read/write, code
execution, outbound curl -- and the fixture's `cwd` was the checkout's
`examples/workspace`. On the qwen3.5:9b feed run that was 215 terminal,
162 search, 87 read, 12 write and 9 execute calls plus 16 curls over 358
rollouts, with writes into this worktree (a staged edit to
`feed_candidates.toml`) and nine files in the user's home. The scores are
what they are, but "called `feed.ask`" was measured on an agent that could
also grep the source and curl arXiv, which the deployed Discord gateway
cannot. agentopt now disables every toolset except `skills` and `clarify`,
copies the workspace fresh per rollout, and reports WORKSPACE WRITES on
the fixture-health line; anything cited from these banks as a decision
input should be rerun under that guard.

## 6. Result: measured, shipped nothing

- **The one accepted candidate changes only `kg.tools`' tool description**
  -- ten times longer, carrying numbers misattributed from the seed's own
  measurement line into a correctness claim and a "Failure Prevention
  Checklist" -- and its exported `SKILL.md` carries the seed's body and
  trigger unchanged (the export reshapes frontmatter, an agentopt artefact,
  not a candidate edit). The gain, real or not, lives in a string the
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
- **The honest one sentence, after the overnight runs (2026-09-05).** On
  gemma4:e2b through the real path, the hand edits were not measured; the
  feed baseline is 0.51 either way with a ±0.10 paired interval and no
  calibration; the knowledge seed scored 0.44 against 0.51 for the same
  text under dotted names (not significant at α = 0.2); and GEPA's single
  accepted candidate is +0.18 on six dev cases at k = 3 (paired LCB +0.08
  at α = 0.2 -- a positive signal, blocked by the pre-registered k ≥ 5,
  with one of the six unsatisfiable), on one model, with a description
  that misstates the seed's numbers -- so `tagging_eval.gate()` cannot be
  evaluated from anything on disk and nothing meets the bar, which is not
  the same as nothing being there.
- **The honest one sentence, after the proof runs (2026-09-06 to 08).**
  With the scorer, the fixture and the proposer fixed and the model
  unloaded between arms: the knowledge hand edit sits inside the noise
  band and left its target case at 0/5; on feed, gemma4:e2b decides
  nothing except that a tenfold-longer trigger is worse, and qwen3.5:9b
  accepts three candidates on the train questions that are flat or worse
  on the held-out ones -- so what the skill text can move on these
  students, at this case count, is smaller than the noise floor or does
  not transfer, and the cases that stay at zero on every text (capability,
  rate-implicit, main-areas, central-hub) need a router phrase, a
  description or a persona line, not a skill rewrite. Nothing ships; the
  optimizer, the harness and the fixture-health line are what this spec
  produced.

## What this spec does not decide

Whether `kg.ask` gets a route to `cite.related`; the four `route_kg` gaps
§1 names (hub -> `kg.central`, "concepts about X" -> `kg.concepts(prefix)`,
"papers behind a concept" -> `feed.search` or a `kg.ask` answer that names
it, "my main research areas" -> `kg.central` instead of a decline);
symbolic's expression parser (a router gap agentopt found:
`integral(x^2, x, 0, 1)` and `sin^2(x)` fail to parse); provenance's
`family` argument, which is a case that must keep failing until the
provenance skill fixes it; a measured rewrite of `kg.tools`' description
(the finding in §6); a pre-edit run that would measure the §3 sentences;
and whether the harness should name a persona (agentopt's call).
