# Docs for collaborators: a front door that leads with the answer

**Date:** 2026-08-29
**Status:** implemented 2026-08-29 (d6bbe54, a2e5515, fa49ebe), with deviations below.
**Depends on:** package docs (`2026-08-29-package-docs-design.md`), golden
paths (`2026-08-28-golden-paths-design.md`).

## Situation, complication, answer

External collaborators are joining. What they will open first is the
README, which today is 778 lines in twelve sections and is, at once, the
pitch, the quickstart, the install guide, a 297-line hermes-agent
integration manual, the ledger manual, the claims manual, the ranking
explainer and the test guide. The docs site built yesterday mirrors it
(Home includes the README whole) and its Guides section holds four
research narratives, not guides. A newcomer cannot tell in a minute what
this is for, what to run, or where the manual for their question lives.

The rule this repo already lives by — a documented fact is a tested fact —
does not have to change. The *shape* does: lead with the answer, support
it with a few mutually exclusive things the tool does, and push every
manual one click down under a name a reader would guess.

## Design

### The README leads with the answer (Minto)

Target: under 200 lines, in this order.

1. **The answer, in one paragraph.** attestation makes research provenance
   auditable and fully local: it reads the experiment runs you already
   have on disk (results files, W&B/MLflow/Sacred/DVC/Hydra directories),
   checks the numbers in your drafts against them, keeps a personalised
   science feed with a knowledge graph, and does symbolic derivations —
   all exposed to agents as MCP tools, with nothing leaving the machine.
   One sentence on who it is for.
2. **Try it in 60 seconds** — the existing block, byte-for-byte (pinned by
   `tests/test_examples.py::test_the_readme_quickstart_runs_without_a_model_server`).
3. **What it does** — four MECE items, one short paragraph each with the
   one number or rule that earns its keep, and a link to its guide:
   the experiment ledger; verifiable claims and citations; the feed and
   the knowledge graph; symbolic derivations. Agents are how you use all
   four, not a fifth thing: a fifth short item "Use it from an agent"
   points at the agents guide and the repo's own skill
   (`src/attestation/skills/research-provenance/SKILL.md`).
4. **Install** — six lines (uv, models optional, `attest install --check`)
   and a link to the install guide.
5. **Golden paths** — the existing grouped list and catalogue link.
6. **Documentation** — the site (`uv run --group docs mkdocs serve`), the
   guides by name, the CLI and API reference, design records, measurement
   lessons, CONTRIBUTING, CHANGELOG.
7. **Licence** — one line.

Everything else moves, verbatim where it can, into guides.

### Guides under `docs/guides/`, one per question a collaborator has

| page | question it answers | moved from |
|---|---|---|
| `install.md` | how do I set it up, with or without a model server? | README "Installation" (135 lines) |
| `agents.md` | how does an agent use this? | README "Launching alongside hermes-agent" (297), `attest emit`, `ATTEST_TOOLS` surfaces; links out to the skill |
| `ledger.md` | will it read my runs, and how does it rank them? | README "The experiment ledger", "Browsing the ledger"; the five tracker conventions with caveats |
| `claims-and-citations.md` | can it check my draft? | README "Verifiable claims"; the citation lint |
| `feed.md` | how does the feed decide what to show me? | README "Feed ranking", "How ranking works"; click provenance |
| `evals.md` | how are the prompts measured? | README "Prompt evals and the optimizer"; the three corpora |
| `testing.md` | how do I know it still works? | README "Tests"; the gates; the CI jobs |

Each guide opens with a one-sentence answer, then the moved material,
lightly edited so it reads standalone (no "as above"). A `docs/concepts.md`
glossary defines the words a newcomer meets in the first ten minutes:
run, family, arm, spec, claim and its five verdict kinds, corpus, persona,
click provenance, surface, golden path, tracker convention.

### The site's nav follows the same order

Start here (README) → Concepts → Install → Guides (the seven) → Golden
paths → Reference (CLI, API) → Design records → Measurement lessons →
Contributing → Changelog. The four research narratives currently under
"Guides" move to a "Notes" group, labelled as what they are.

### What is tested

- README under 200 lines, and every `docs/` link in it resolves
  (`tests/test_docs_site.py`).
- Every `docs/guides/*.md` is in the nav and opens with a one-sentence
  answer paragraph before its first `##` (a test reads the file).
- `mkdocs build --strict` stays green — every moved link is checked by the
  build; the `docs` CI job runs it.
- The quickstart test is unchanged and still passes.
- `CONTRIBUTING.md` gains a "Where things live" table (directories →
  purpose → the test that guards them).

## Not in scope

- Rewriting the content of the moved sections beyond what standalone
  reading needs. The manuals are correct; they are in the wrong place.
- Deploying the site.
- Touching `CLAUDE.md`'s agent-facing map beyond the docs index.

## Success criteria

- A reader who stops after the first paragraph knows what the tool is,
  who it is for, and that it runs locally; after "What it does" they know
  which guide to open.
- README ≤ 200 lines; seven guides + concepts exist and are in the nav;
  the strict build and the whole suite are green; no README fact was
  dropped — each moved section appears, verbatim or lightly edited, in
  exactly one guide (a reviewer diffs the moved text).

## Deviations and findings

**The README's links into `docs/` are real markdown links, and the site
home includes the README through a hook, not a bare snippet.** A link
written for the repository root (`[..](docs/guides/x.md)`) resolves to
`docs/docs/...` once `docs/` is the serving root, and `pymdownx.snippets`
cannot rewrite what it includes; a `docs/docs` symlink recurses forever.
`docs/_hooks.py` performs the one include for `index.md` and drops the
`docs/` prefix as it reads — it must do the include itself because
`on_page_markdown` runs BEFORE snippets expand, so a hook that only
rewrites sees the one-line directive. The first cut used backtick paths
instead, which built clean but were not clickable on GitHub and, being
plain text, were invisible to `test_the_readme_is_a_front_door`'s link
check: 17 references a typo could have broken with every gate green.

**Two `tests/test_architecture.py` tests followed the facts they guard.**
They asserted the 46-tool table and the `feed.list` cap were present in
`README.md`; that content now lives in `docs/guides/agents.md`, so the
tests read that file with the same live-surface comparison.

**Install has its own nav entry beside Guides.** The spec's "Install →
Guides (the seven)" counts `install.md` among the seven; it is a top-level
entry, because a newcomer looks for "Install" in a nav, and the README's
flat guide list is left as is.

**`docs/architecture/research-profile.md` was orphaned** from any nav and is
now the fourth entry in Notes. README went from 778 lines to 148; the
cold-reader review found one dropped sentence (which golden-path group
needs a model server), restored in the fix round.
