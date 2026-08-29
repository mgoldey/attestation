# The feed and the knowledge graph

How does the feed decide what to show me? Cosine similarity against your
interests profile until you have clicked anything, then a blend with two
click-trained terms that phases in as evidence accumulates — plus click
provenance, so only trustworthy feedback trains the ranker.

## Feed ranking

The original core, still here. An agent orchestrator for personalized feed
recommendations coordinating three layers with distinct reliability contracts:
deterministic ingest (fetch → dedup → embed), a per-user learnable ranking core
(profile embedding + click-trained classifier), and a LangGraph explain agent
that says *why* items rank where they do — lazily, cached, never blocking the
feed. The LLM is a swappable OpenAI-compatible backend, not the point.

It runs two ways, and they share one database:

1. **Standalone** — a web UI at `http://127.0.0.1:8899` with ✓/✗ feedback
   buttons that retrain the ranker on every click.
2. **Alongside [hermes-agent](https://github.com/NousResearch/hermes-agent)** —
   as an MCP stdio server exposing `feed.*` tools as native tool calls (see
   the [agents guide](agents.md) for the full table), plus an optional
   `research-provenance` skill for setup automation and fallback.

## How ranking works

- 0 clicks: cosine similarity between item embeddings (embeddinggemma, 256-dim)
  and your `interests` profile text.
- With clicks, two click-driven terms join the blend (weight
  `w = n_clicks / (n_clicks + 5)`, averaged when both are active):
  - a per-user logistic regression over item embeddings (guard: this term
    only participates once your clicks contain both classes);
  - a feature-preference term — Laplace-smoothed like/dislike ratios per
    LLM-extracted topic tag, content type, and source feed (see
    `uv run attest tag`). This term works from click one, including for
    users who have only ever downvoted: two ✗ on items sharing a tag demote
    every item carrying that tag on the next render.
- Visible movement by click 3-4; tag-level demotion is visible immediately.

## Click provenance

Every recorded click stores its provenance, and provenance decides what a row
may be used for:

| source | what it is |
|---|---|
| `ui` | you pressed a button on the web page |
| `agent` | an MCP `feed.rate` call, usually the agent reading your reply |
| `implicit` | you asked why an item ranked; engagement, counted as a weak positive |
| `simulated` | a local model reacting to the text as the persona would |
| `bootstrap` | synthetic persona seeding |

`bootstrap` labels are a linear threshold on the same embedding the ranker's
classifier consumes, so scoring against them is a tautology and
`evaluate_user` excludes them. The other four are trainable, and
`feed.persona_status` breaks the counts down by source so you can see how much
of a persona's history is yours.

Explicit feedback is scarce by nature — this project's own database held 68
web clicks and 2 agent clicks across 5,167 items before the two synthetic
channels were added, every one of them positive, which is why the click
classifier had never fired for a real account.

The knowledge graph that sits beside the feed — concepts, centrality,
communities — is derived from the same tagging pass; see the
[agents guide](agents.md) for how it is built and the full `kg.*` tool table.
