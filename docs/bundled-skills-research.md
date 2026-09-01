# Which skills attestation should bundle

**Date:** 2026-08-30
**Status:** research; steps 1 and 2 of the order of work were implemented
the same day (uncommitted at time of writing): `install.py` syncs every
bundled skill into `~/.hermes/skills/` and each `~/.hermes/profiles/*/skills/`
that exists, respects a `SKILL.md.<anything>` disable rename, and disables a
leftover `research-provenance` copy by renaming its `SKILL.md`; the monolith
is split into `attestation-{setup,feed,provenance,knowledge,symbolic}` with
`tests/test_skill_files.py` enforcing the verb-first descriptions and the
surface-only tool rule, and `tests/test_install_skills.py` the installer
behaviour. The routing measurement (step 2's acceptance) was run 2026-09-01
and ACCEPTS the split — see "The routing measurement" below; B1/B2 are
unbuilt. Every remaining recommendation names the measurement
that would accept or reject it.
**Method:** local only. The repo's skill, installer, tests, emitted agents
and docs were read directly; the 73 `SKILL.md` files under
`~/.hermes/skills/` (and the `research` profile's mirror) were surveyed for
overlap. No web lookup.

## What is bundled today, and what it costs

One skill, `src/attestation/skills/research-provenance/`, ships in the wheel
and is copied byte-for-byte to `~/.hermes/skills/research-provenance/` by
`attest install` (`install.py:step_skill_copy`, keyed on the single
`SKILL_NAME` constant). It is **39,513 bytes** and covers all four agent
surfaces at once — feed, provenance, knowledge, symbolic — plus setup,
`uvx` install, the configuration contract, the `ATTEST_TOOLS` mechanism, the
HTTP fallback path, presentation rules for Telegram, verdict extraction from
ordinary conversation, and the reload/staleness procedure.

Three measured facts from `CLAUDE.md` and `measurement-lessons.md` frame
what a skill is worth here:

- **The index is cheap; the body is not.** 68 skills cost ~7 KB in the
  prompt (~70 B each); the body loads only on invoke. Tool schemas cost 85 KB
  *every turn*. So bundling more skills costs almost nothing per turn — but a
  39.5 KB body is roughly 10K tokens landing in a 2B model's context the
  moment the skill is invoked, on a model where "length is zero-sum".
- **Routing to attestation goes through MCP tools, not skill selection**
  (88% on the live surface; the feed never selects a skill). A skill's job
  is therefore not to get the agent to attestation — the tools do that — but
  to carry the *judgment* a tool description cannot: sequences, refusals,
  presentation, what to record, and what to produce.
- **Skill descriptions collide.** With `arxiv`, `blogwatcher`,
  `weights-and-biases` and `research-paper-writing` listed beside the feed
  tools, routing fell 6/6 → 3/6; two of those are now on disk as
  `SKILL.md.disabled-collides-with-attestation`. Any skill this project adds
  competes in the same 70-entry index, including with its own siblings.

Contracts a bundled skill must already satisfy (`tests/test_skill_files.py`):
it invokes only declared console scripts, never `hermes install`, documents
every live namespace, teaches every `.ask` router by name, and its
presentation example stays Markdown.

## The routing measurement (2026-09-01): the split accepted

Run as specified above: the real skill index — the 70 enabled `SKILL.md`
files under `~/.hermes/skills/` (nested dirs included; the two
`.disabled-collides-with-attestation` renames and `.retired`
science-recommendations excluded, as Hermes excludes them) — with the
monolith's installed description (before) versus the same index with the
five split entries in its place (after, 74 entries). 56 attestation
questions (12 per surface + 8 setup), each carrying the arguments a real
user would supply, plus 10 control questions owned by other live skills.
`gemma4:e2b-it-q4_K_M`, temperature 0, one run (deterministic for a fixed
prompt). The model sees the alphabetised `name: description` index in a
system prompt and answers with one skill name or `none`.

**What the number is and is not.** This is skill-INDEX selection on a
fixed harness — a before/after delta, not Hermes end to end, and not tool
routing: the live feed path selects among MCP tools directly (measured
88%, `measurement-lessons.md` §3) and never consults the index. The index
governs which *judgment body* loads, which is exactly what the split
changed. One harness note that did not exist in August: Ollama now runs
this model with thinking ON by default, and a thinking reply spends its
whole token budget in `reasoning` with empty content — the harness sets
`think: false` on the native `/api/chat` endpoint. Any future comparison
against the Aug-24 numbers must account for that.

| surface | before: attestation-hit | after: attestation-hit | after: exact sibling |
|---|---|---|---|
| feed | 6/12 | 12/12 | 9/12 |
| provenance | 12/12 | 12/12 | 11/12 |
| knowledge | 9/12 | 11/12 | 10/12 |
| symbolic | 7/12 | 12/12 | 12/12 |
| setup | 3/8 | 8/8 | 6/8 |
| **total** | **37/56 (66%)** | **55/56 (98%)** | **48/56 (86%)** |

Controls: attestation stole **zero** of the ten control questions in
either condition — five sibling entries do not grab other skills'
traffic. (Control own-skill was 8/10 before, 7/10 after; every miss was a
defensible neighbour with no attestation involvement — `ocr-and-documents`
for PDF table extraction, `claude-design` for a slide deck,
`ascii-video` for an animation — index noise, not a split effect.)

**Where the monolith was losing.** The before-misses are the collision
problem in the flesh, and they cluster exactly where the monolith's
description says least: symbolic questions went to `python-debugpy`,
`claude-code`, `codex` and a hallucinated `simplify_code`; setup questions
to `computer-use`, `touchdesigner-mcp`, `hermes-agent`; five questions
across surfaces to `grounded-citations`. The one after-miss is
"search my bibliography for papers by Hinton" → `arxiv`, a defensible
neighbour for a bibliography-search phrasing.

**The split's own cost is sibling confusion, all of it inside
attestation:** eight wrong-sibling picks (feed↔knowledge on
persona/interests/suggest-sources phrasings; a staleness question to
feed; citation-key resolution to provenance; two setup questions with
feed words in them to feed). A wrong sibling still lands the agent in
attestation with a body that names the right surface's tools one
cross-reference away — the failure the fallback ("two skills") was held
for did not appear, so the fallback is not taken.

**Verdict:** the split is accepted — +18 questions routed to attestation
with zero control theft. The 66% baseline also revises this doc's framing
upward: the monolith was not merely oversized, it was losing a third of
attestation's own questions to better-worded neighbours.

### Two findings from the survey, before any recommendation

1. **The installed copy is stale.** `~/.hermes/skills/research-provenance/
   SKILL.md` is 38,926 bytes against the repo's 39,513. `attest install
   --check` reports this as `skill_copy BROKEN`; running `attest install`
   fixes it. Not fixed here — it is a live-machine change.
2. **The `research` profile runs a skill from before `cite.*` existed.**
   `~/.hermes/profiles/research/skills/research-provenance/SKILL.md` is
   24,928 bytes and its description has no Zotero/`.bib` clause. `install.py`
   syncs `~/.hermes/skills/` only; `~/.hermes/profiles/*/skills/` is a
   hand-made mirror nothing refreshes. The profile that exists *for*
   attestation work is the one reading the oldest instructions. Whatever is
   bundled, the installer (or `attest emit`'s drift check, which already
   knows about profiles' config) should cover profile skill trees too.

## The neighbourhood: installed skills that touch the same ground

| Installed skill | Size | Relation to attestation |
|---|---|---|
| `research/research-paper-writing` | 103,656 B | Full NeurIPS pipeline: lit review, experiments, stats, LaTeX, BibTeX. Overlaps everything, owns none of it. The natural *caller* of `runs.claims_check` / `cite.check`. |
| `research/grounded-citations` | 11,241 B | Web-source grounding: numbered `[n]` citations, a ledger script owns the url→n map, verbatim-quote verification. About *prose sourced from the web*, not numbers vs runs. Defers BibTeX to paper-writing. |
| `research/llm-wiki` | 20,129 B | Karpathy-style interlinked markdown knowledge base; agent summarises and cross-references, human curates. A consumer of `feed.read` + `kg.*`, not a competitor. |
| `note-taking/obsidian` | 2,935 B | Vault file access. Same relation as llm-wiki. |
| `research/arxiv` | 10,085 B | Live arXiv API search via curl. **Network.** Active in the main tree, disabled in the research profile. Collides with `feed.search` on description. |
| `research/blogwatcher` | disabled | RSS monitoring CLI. Disabled for collision with `feed.*`. |
| `mlops/evaluation/weights-and-biases` | disabled (main) / active (profile) | Instrumented tracking, sweeps, registry. Disabled for collision with `runs.*`; the ledger *reads* W&B directories instead. |
| `mlops/evaluation/evaluating-llms-harness` | 12,204 B | Runs lm-eval-harness. Produces exactly the result files `runs.scan` wants to read — if they land in a recognised directory. |
| `productivity/ocr-and-documents`, `productivity/pdf` | 5-7 KB | PDF text extraction. Supporting cast for reading a paper the feed surfaced. |
| `software-development/hermes-agent-skill-authoring` | 11,357 B | The house style for `SKILL.md` frontmatter and tags; anything bundled should follow it. |

There is no standalone SymPy, Zotero/BibTeX, MLflow, Sacred, DVC, Hydra or
TensorBoard skill anywhere in the tree. Symbolic and citation resolution
exist only inside `research-provenance`.

## The principle for what earns a bundled skill

`docs/hermes-agent-plugin-research.md` settled this once: a skill tells the
*model* something; a tool gives it a capability. Keep the skill only for
"workflow/judgment that doesn't fit a tool description". Applied to this
project, a skill earns its place when it teaches one of four things the tool
surface cannot say about itself:

1. **A sequence across tools** — `runs.list` before `runs.compare`; all
   three linters on a manuscript; `kg.concepts` before a tag filter.
2. **How to *produce* what the tools *read*** — the ledger reads artifacts
   on disk and the claim checker reads HTML comments; nothing today teaches
   an agent to leave either behind in a readable shape.
3. **Presentation and verdict extraction** — Markdown links, list every
   item, read `ranking_quality`, treat "not my area" as `useful=0`.
4. **When not to** — `unproven` is not a disproof, a refused comparison is
   the answer, an empty search is the relevance floor working.

And one rule for the index: **each bundled skill's description must lead
with a verb no other description in the tree uses** — record, annotate,
rank, derive — because the measured collisions were between descriptions
that named the same *topic* (arXiv, feeds, tracking), and a topic-named
sibling would collide with our own skills exactly as `blogwatcher` did.

## Recommendations

### A. Split the monolith into per-surface skills (restructure, not addition)

Four skills mirroring `AGENT_SURFACES` and the four emitted
`.claude/agents/attestation-*.md` files, plus one thin operational skill:

| Skill | Leads with | Body target | What moves into it |
|---|---|---|---|
| `attestation-feed` | *rank* / *read* | ≤10 KB | "What should I read", search, feedback + verdict extraction, presentation rules, `ranking_quality`, harvest/simulate, sources |
| `attestation-provenance` | *verify* | ≤8 KB | "Which arm won", the three linters, the five verdicts, tracker caveats, direction refusals |
| `attestation-knowledge` | *connect* | ≤5 KB | `kg.*` sequences, alias/vocabulary rule, hand-off to llm-wiki/obsidian for notes |
| `attestation-symbolic` | *derive* | ≤4 KB | `sym.*`, `unproven` is not disproof, sandbox limits, `sym.derivation` is genuine only for integrals |
| `attestation-setup` | *install* | ≤6 KB | `setup.sh`, `uvx`, configuration contract, `ATTEST_TOOLS`, reload and staleness, HTTP fallback |

Why: the invoked body drops from ~10K tokens to 1-2.5K on the surface the
session is actually using; each description is disjoint by verb; the split
matches the surfaces the config, the agents and the tests already know
about; and a `research`-profile session with only `attestation-feed`
enabled is not handed 30 KB about tools it cannot see.

What has to change to allow it: `install.py`'s `SKILL_NAME` becomes a list
and `step_skill_copy` iterates; `test_skill_files.py`'s "documents every
namespace" and "teaches every router" become assertions over the *union* of
bundled skills, with a per-skill assertion that a skill documents only the
namespaces on its surface (a model that reads about `runs.*` in the feed
skill will try to call it); the presentation-example guard applies to the
feed skill. The `hermes-agent-skill-authoring` skill's frontmatter
conventions apply.

Risk to measure: five sibling entries in the index instead of one. The
memory record says a small mock index makes routing artificially easy, so
the measurement is on the real 70-entry tree: 12+ questions per surface,
gemma4:e2b at temp 0, tool selection before and after. If the split costs
routing, the fallback is two skills (feed; everything else) rather than
reverting to one.

### B. Bundle two write-side skills that no tool can replace

These are the genuinely new ones. Both teach an agent to *produce* the
inputs attestation's read-only tools consume, which is where the ledger
guide's own adoption argument ("reads what is already there") stops short:
an agent running experiments on the reader's behalf can be told to leave
what is readable.

**B1. `attestation-record` — leads with *record*.** When this agent runs an
experiment or evaluation (through `claude-code`, `codex`,
`evaluating-llms-harness`, or directly), it writes:

- final values, one JSON per arm, into a recognised results directory
  (`results/`, `metrics/`, `eval/`, …), arms sharing a filename prefix so
  `family_of` groups them;
- the config beside it in `configs/` as provenance, never as a metric;
- a `[metric_direction]` entry in `~/.hermes/metric_direction.toml` for any
  metric the built-in table does not know, *before* the first
  `runs.compare` — because compare refuses undeclared directions on
  purpose, and the agent that produced the metric is the one that knows;
- a corpus declaration when the driver script does not make it detectable;
- for Hydra sweeps, `hydra.job.chdir=true`, or every arm overwrites one
  file (measured, `measurement-lessons.md` §6).

It ends with `runs.scan(confirm=true)` and `runs.compare` so the run enters
the ledger in the same session it was made. It does not collide with the
disabled W&B skill: that one instruments; this one leaves files.

**B2. `attestation-annotate` — leads with *annotate*.** When this agent
writes or edits prose that states a result, it puts
`<!-- claim: <run> metric=<m> value=<v> tol=<t> -->` beside each decimal,
adds `cite=<key>` only after `cite.lookup(key)` resolves (never a key it
composed), runs `runs.claims_coverage` before handing the draft back and
reports the uncovered decimals, and treats `contradicted` as "fix the
document or the run, and say which". It explicitly hands citation *style*
and BibTeX to `research-paper-writing` and web-sourced prose to
`grounded-citations` — the description must not lead with "citation", or
it lands in the same collision those two would have with each other.
`examples/citations/` is already the worked run.

Measured acceptance for both: on a fresh sandbox project, an agent given the
skill and asked to run a two-arm comparison (B1) or to write a results
paragraph from `runs.detail` output (B2) produces artifacts that
`runs.scan` reads to the right number of runs, and claims that
`runs.claims_check` returns as `supported` — scored over ≥10 trials, the
way the tagging and explanation prompts are scored, not by one
demonstration.

### C. Fold, do not add

- **Onboarding by rating six items** (from
  `recommendation-literature-review.md` §12) belongs *inside*
  `attestation-feed`, not as a skill of its own — it is the first move of
  the feed conversation, not a separate task.
- **A measurement-discipline skill** ("a number is about the artifact it was
  taken from"; the three questions; interleaving over A/B for one reader)
  is the repo's most transferable idea and would lead with a verb nothing
  else uses (*measure*). It is not attestation-specific, though, and its
  natural home is the `research` profile's `SOUL.md` or a standalone skill
  outside this package. Recorded as a candidate, not bundled.

### D. Do not bundle

| Candidate | Why not |
|---|---|
| arXiv / web search | Network. The offline guarantee has one exception (`citations.WebReader`, armed at construction) and it is not this. `feed.source_preview` covers "what does this feed carry" without leaving the machine. |
| Tracker instrumentation (W&B, MLflow, DVC…) | The ledger's stated design is to read finished artifacts, not to add discipline. B1 is the write-side complement that respects that. |
| Notes / wiki / Obsidian | `llm-wiki` and `obsidian` exist and are consumers of `feed.read` + `kg.*`; the knowledge skill names them as the hand-off. |
| Paper-writing pipeline | 103 KB already installed; attestation supplies the linters it should call (B2 says how). |
| Web-source citation grounding | `grounded-citations` exists and is about a different object (URLs in prose) from `cite.check` (keys against Zotero/`.bib`). |
| Symbolic math as its own product | `attestation-symbolic` (A) is enough; SymPy needs no second skill. |

## Order of work, and how each step is measured

1. **Sync the profile skill tree** — extend `step_skill_copy` (or
   `attest emit`'s drift check) to `~/.hermes/profiles/*/skills/`. Measured
   by `attest install --check` reporting the profile copy's staleness, which
   today it cannot see.
2. **Split (A)** — five skills, tests generalised to the union, installer
   iterating. Measured by routing on the real index before/after, and by
   body size per surface.
3. **`attestation-record` (B1)** — with a sandbox-project acceptance test.
4. **`attestation-annotate` (B2)** — with the claims acceptance test;
   `examples/citations/` extended to show the write side.
5. Re-run the description-collision measurement with all seven entries in
   the tree, since the last one was taken with one.

## What this does not claim

- No routing number here is predicted. Every prior prompt edit in this repo
  moved routing in a direction someone did not expect.
- The split (A) is a restructure with a measurable downside; if the index
  measurement says one entry routes better than five, the bodies still
  shrink by splitting *content* across files that one index entry points
  at, which Hermes supports via `related_skills`.
- Nothing here touches the tool surface, `AGENT_SURFACES`, or the offline
  guarantee.
