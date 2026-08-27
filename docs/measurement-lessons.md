# What measurement kept overturning

**Date:** 2026-08-24
**Status:** living. Append findings; do not rewrite history.
**Scope:** cross-cutting method. Individual findings live in their own specs —
this file records what they have in common.

This repo's design decisions have been reversed by measurement often enough
that the pattern is now the most valuable thing to write down. Four subsystems
were built or rebuilt on results that contradicted a confident prior:

- the swarm (`docs/superpowers/specs/2026-08-22-swarm-refutation.md`)
- the 0.964 AUC that turned out to classify provenance (commit `bc37a5b`)
- the ledger fixture that overstated yield (3 real comparisons in 1045 runs)
- the agent-side prompt (2026-08-24, below)

The through-line: **a number is about the artifact it was taken from, not
about the system.** Every failure below is one of confusing the two.

---

## 1. Architecture: a model choosing at runtime is a stage that can be wrong

Measured on `gemma4:e2b-it-q4_K_M`, 15 turns × 3 runs:

| architecture | correct | latency |
|---|---|---|
| Routed — deterministic dispatch | **13/15** | 1.3s |
| Flat — one namespace | 8/15 | 1.3s |
| Swarm — supervisor LLM + subagent LLM | 7.3/15 | 2.8s |

The swarm did worse than doing nothing, at twice the latency. Two 80%-accurate
stages in series make a 64%-accurate pipeline, and a namespace miss is
unrecoverable — competence inside the wrong agent cannot save the turn.

The decomposition was not the problem. Four per-domain agent surfaces shipped;
what was rejected is a *model* selecting between them at runtime.

> Separate agents help when a **person** chooses which to talk to.
> They hurt when a **model** chooses at runtime.

Full argument, including what survives for multi-step orchestration:
`docs/superpowers/specs/2026-08-22-swarm-refutation.md`.

---

## 2. Prompt tuning: length is zero-sum on a small model

Every intuition here was wrong at least once. Measured on `gemma4:e2b`,
temp 0, against the live 46-tool MCP surface.

**Longer descriptions make routing worse.** Lengthening the skill description
for `research-provenance` to assert priority dropped it 12/16 → 9/16. The
extra words added keyword surface for competitors to match, and it lost cases
the short version won.

**A worked example outweighs the rule beside it.** A paragraph explaining that
Slack needs `<url|title>`, placed directly above a Markdown example, produced
5/5 items and **0/5** correct links across three runs — the model copied the
example and ignored the prose. Only changing the example moved the result.

**Fixing one case breaks another.** Adding "call a listing tool to discover
arguments you lack" repaired `runs_compare` and broke two feed cases. Net zero.

**Instructions must name concrete values, not describe them.** With no persona
named, the model passed `user: "user"` on **9 of 9** feed calls — the literal
placeholder. Because `@tool(autocreate_user=True)` creates unknown names rather
than refusing, every such call risked an empty persona that ranks badly
forever. Naming the persona in the prompt: 9/9 correct.

**Restricting the tool surface is not automatically a win.** `ATTEST_TOOLS=feed`
(22 tools) scored *worse* than the full 46 tools (7/12 vs 9/12) before the system
prompt was fixed. The failures were the model declining to call anything, not
choosing wrongly — a framing problem, not a surface-size problem.

Net effect of the final agent-side prompt, read from the live file:

| | before | after |
|---|---|---|
| correct persona passed | 0/9 | **9/9** |
| tool selection | 12/24 | **24/24** |
| Telegram links clickable | 0/5 | **5/5** |

**Rule:** re-measure after every prompt edit, including edits that only add.
There is no monotonic "more guidance is better" on a 2B model.

---

## 3. The recurring bug: measuring the artifact instead of the system

Four measurements in a single session (2026-08-24) were confidently wrong.
Each was repeatable, deterministic, and about the wrong thing.

| what was measured | what it was taken to mean | actual |
|---|---|---|
| a transcript header | reader is on Slack | Telegram; the "fix" would have broken every link |
| a hand-built 8-skill index | collisions cost 21/24 → 8/24 | on the real 70-skill tree: 4/16 either way |
| skill-name selection | "routing is 25%" | the feed uses MCP tools directly and never selects a skill |
| grading `feed_list` as the only right answer | 12/24 | `feed_digest` is also correct; correct refusals were scored as failures → 21/24 |

Each was cheap to check against the real thing: `~/.hermes/config.yaml` names
the platform and the enabled MCP servers; the skill tree can be walked; the
Telegram formatter can be imported and called on a candidate string. **Modelling
was faster than looking, and wrong every time.**

**Before reporting a measurement, answer three questions:**

1. Is this the path the system actually takes? (An enabled MCP server means the
   model never looks for a skill.)
2. Is this index/fixture/dump the live one? (A prompt dump is evidence about
   that dump. One sampled here predated the MCP wiring entirely.)
3. Is a "failure" here actually correct behaviour? (Declining to call
   `runs_claims_check` when the question carries no file path is right.)

---

## 4. Guards: a passing test is evidence of nothing

Every regression guard written on 2026-08-24 passed on first run. Three
protected nothing:

- **Wrong scope.** A "tools returning a url must say to show it" test keyed off
  `outputSchema`; only the four `.ask` routers declare one, so it inspected
  four tools and skipped `feed.list` and `feed.search` — the two from the bug
  report. Deleting either instruction still passed.
- **Co-occurrence, not adjacency.** `"url" in text and "show" in text` passed
  with the whole instruction removed: every docstring already says "Returns …
  url" in passing.
- **Prose vs machinery.** Scanning the serialized schema matched a field
  literally named `url` (`"title": "Url"`) beside an unrelated verb in one
  unsplittable JSON blob.

A fourth was *inverted* — it required Slack syntax, which measurement showed
breaks Telegram. Both directions are now guarded.

**Rule:** delete or weaken the exact thing a guard protects and confirm it
FAILS, then restore. Mutate each protected site separately — a partial mutation
that leaves matching text behind is a bad mutant, not a good guard. Where a
rule has two failure directions, mutate both.

This is the repo's stated failure mode ("tests that pass against the bug they
were written to catch") reproduced three times in one day by the person who
wrote that line.

---

## 5. Where the bug usually is not

The reported symptom pointed at the wrong layer in every case this session:

- "You didn't give links" — not a data bug. All 5499 items carry a url, and the
  five in question had real arXiv addresses in the payload the model received.
  It rendered `item_id` instead.
- "The links weren't clickable" — not a rendering bug. The gateway converts
  Markdown to MarkdownV2 correctly; there were simply no links to convert.
- "Routing is 25%" — not a routing bug. The eval was wrong.

The one genuine live defect (a junk persona created on every feed query)
surfaced only while testing something else.

**Rule:** reproduce the symptom against the live payload before theorising
about the layer that produced it.

---

## 5. An optimizer memorizes; only a held-out model tells you it happened

DSPy GEPA on the tagging prompt (2026-08-27; full record in
`docs/superpowers/specs/2026-08-23-dspy-prompt-optimization-design.md`).

**The train number lied in the usual way.** 23 train cases, 300 metric calls:
0.790 → 0.902 inside the optimizer. Through the production client on the
28 held-out dev cases: 0.819 → 0.824. The instruction it wrote is 8× longer
and quotes tags from specific train items back as "rules".

**Transfer told the truth the primary model could not.** Scored on three
models, twice per case: the candidate *tied* the model it was tuned on
(0.807 / 0.807) and beat the two it never saw by +0.110 and +0.086. The
pre-registered gate failed it on the tie, and stays as written. The finding
underneath is the hypothesis the spec started from: the hand-written prompt
is fitted to gemma4:e2b's idiom, and a longer instruction e2b cannot exploit
(§2: length is zero-sum on a 2B model) is exactly what larger models can.

**A tie at 56 samples is not a result either way.** Re-running until the
primary wins is the same tautology as selecting demonstrations from the
scoring set. The next step is more cases, decided before the next run.

**Rule:** decide the acceptance bar before the run, score on models the
optimizer never saw, and treat the optimizer's own number as the artifact it
is.
