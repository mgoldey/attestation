# Docs for Collaborators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A README under 200 lines that leads with the answer and links to seven guides under `docs/guides/` plus a concepts page, a site nav in the same order, and tests that keep it so — with no README fact dropped.

**Architecture:** One implementer owns every file in this plan (README, `docs/guides/*`, `docs/concepts.md`, `mkdocs.yml`, `CONTRIBUTING.md`, `tests/test_docs_site.py`, `CLAUDE.md` index) so the moves are atomic; content is moved verbatim first, then edited only for standalone reading. A second, separate agent reviews as a cold reader.

**Tech Stack:** Markdown, mkdocs-material (existing `docs` group), pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-docs-for-collaborators-design.md`

## Global Constraints

- The README's "Try it in 60 seconds" fenced blocks are byte-identical before and after (`tests/test_examples.py::test_the_readme_quickstart_runs_without_a_model_server` and a `git diff` of that section).
- Every README section that moves lands in exactly one guide; the moved text is verbatim except edits needed to read standalone (no "as above", no "this section"). Record each move as `README § → docs/guides/<page>` in the report so the reviewer can diff.
- README ≤ 200 lines after the move; every relative link in it resolves; `mkdocs build --strict` exit 0 with no WARNING.
- Guides open with a one-sentence answer paragraph before the first `##`.
- The site nav order is the spec's; the four research narratives move to a "Notes" group.
- `CLAUDE.md`: only the docs index changes (new `docs/guides` dir, `docs/concepts.md`), plus the docs-index test must pass.
- Line length rules do not apply to Markdown (excluded from ruff); gates after `git add`; commit by pathspec; message style: plain sentence, blank line, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: The move

**Files:**
- Create: `docs/guides/install.md`, `docs/guides/agents.md`, `docs/guides/ledger.md`, `docs/guides/claims-and-citations.md`, `docs/guides/feed.md`, `docs/guides/evals.md`, `docs/guides/testing.md`, `docs/concepts.md`
- Modify: `README.md`, `mkdocs.yml` (nav + `not_in_nav` unchanged), `CONTRIBUTING.md` ("Where things live" table), `tests/test_docs_site.py`, `CLAUDE.md` (docs index)

- [ ] **Step 1: Failing tests** in `tests/test_docs_site.py`: `test_the_readme_is_a_front_door` (≤ 200 lines; every `](docs/...)`/`](CONTRIBUTING.md)`/`](examples/...)` link target exists; the string "Try it in 60 seconds" present); `test_every_guide_is_in_the_nav_and_leads_with_an_answer` (for each `docs/guides/*.md`: its path appears in `mkdocs.yml`'s nav; the first non-heading, non-blank line is a paragraph that ends with a period and comes before the first `## `); `test_the_concepts_page_defines_the_first_ten_minutes` (each of: run, family, arm, spec, claim, verdict, corpus, persona, provenance, surface, golden path, convention appears as a `**term**` or `### term` in `docs/concepts.md`). Run: all three fail.
- [ ] **Step 2: Move sections** exactly per the spec's table. Use the README's current section boundaries (`## Installation` 135 lines → install.md; `## Launching alongside hermes-agent` 297 → agents.md, adding `attest emit`/`ATTEST_TOOLS` from `docs/superpowers/specs/2026-08-22-agent-surfaces-design.md`'s summary and a link to `src/attestation/skills/research-provenance/SKILL.md` and to `examples/agents/`; `## The experiment ledger` + `## Browsing the ledger` → ledger.md (keep the tracker table and `family_of` rule); `## Verifiable claims` → claims-and-citations.md (plus the CLI/MCP citation lint, `examples/citations/`); `## Feed ranking` + `## How ranking works` → feed.md; `### Prompt evals and the optimizer` → evals.md (plus the three corpora and `examples/prompt-evals/`); `## Tests` → testing.md (plus the CI jobs `gates`, `wheel-smoke`, `flows`, `docs site`)). `## Commands` becomes a link to `docs/reference/cli.md`. Each guide: title, one-sentence answer, moved text.
- [ ] **Step 3: Rewrite the README** in the spec's seven-part order; the answer paragraph must mention: local/nothing leaves the machine; reads runs from files you already have (name the five tracker layouts); checks claims in drafts; feed + knowledge graph; symbolic derivations; MCP tools for agents. Keep "Golden paths" as is. "Documentation" lists the guides by name.
- [ ] **Step 4: `docs/concepts.md`** — one paragraph per term, each pointing at the guide or module where it lives; no new claims (every definition traceable to a spec or docstring — cite it inline).
- [ ] **Step 5: `mkdocs.yml` nav** per the spec; `docs/index.md` keeps including README. `CONTRIBUTING.md` gets "## Where things live": a table of `src/attestation/`, `src/attestation/mcp/`, `src/attestation/ledger_adapters/`, `evals/`, `examples/`, `docs/guides/`, `docs/superpowers/`, `tests/`, `scripts/` → purpose → guarding test.
- [ ] **Step 6:** `uv run --frozen pytest tests/test_docs_site.py tests/test_examples.py tests/test_architecture.py tests/test_golden_paths.py -q`; `uv run --group docs mkdocs build --strict`; gates; commit by pathspec.

### Task 2: Cold-reader review (a reviewer, not an implementer)

A fresh agent reads `README.md` top to bottom as a newcomer with no repo context, then opens each guide; reports where it got lost, every fact that moved and lost meaning, every link that leads somewhere unexpected, and diffs the moved sections against `git show <base>:README.md` to confirm nothing was dropped. Findings go to a fix round on Task 1's implementer.

## Self-review
Spec coverage: README shape (T1 s3), guides + concepts (s2, s4), nav + Notes group (s5), tests (s1), CONTRIBUTING map (s5), quickstart preserved (constraint), cold-reader check (T2). No placeholders: files, section boundaries with line counts, test assertions and commands are named.
