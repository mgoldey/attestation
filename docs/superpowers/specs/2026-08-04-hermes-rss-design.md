# hermes-rss — Personalized RSS Ranking Agent (Design)

Date: 2026-08-04
Status: Approved design, pre-implementation
Reviewed by: Fable design review (3 blockers + 11 findings folded in)
Revision (2026-08-04, mid-build): **"Hermes" names the orchestrator** — the CLI +
LangGraph agent system that coordinates ingest → rank → explain — NOT the LLM.
The chat model is a swappable backend: `HERMES_CHAT_MODEL` env var, default
`gemma4:12b` (local), `hermes3:8b` optional. All hermes3:8b references below
read as "the configured chat model".

## Purpose

A spike with two jobs:

1. **Daily-use tool**: triage Matt's real reading (arXiv, ML/AI blogs, general
   news) into one ranked feed, personalized by y/n "useful" clicks.
2. **Demo** for a founder conversation about personalized science
   recommendations. Two demo moments, in order of robustness:
   - **Persona switch** (most robust — zero clicks, zero LLM): the same feed
     ranks differently for different user identities, driven by profile
     embeddings alone. Build and rehearse this first.
   - **Live learning**: click ✓/✗ a handful of times and watch the feed
     visibly reorder by click 3–4.

Non-goals: production hardening, deployment, real multi-tenant auth,
DPO/preference fine-tuning (that is the "what this grows into" talking point,
not the build).

## Constraints

- Local only. No cloud services, no Docker, no Postgres.
- Hardware: 2× GTX 1080 (8GB VRAM each), 23GB RAM, Ollama installed.
- Stack must showcase LangGraph + pydantic (part of what the spike advertises).
- Timeframe: days.

## Architecture

```
feeds ──ingest──▶ SQLite (sqlite-vec) ──rank──▶ FastAPI ──▶ single-page UI
                     ▲                            │ ✓/✗ clicks
                     └────── clicks ◀─────────────┘
                                                  │ lazy, async
                     LangGraph "explain" graph ◀──┘
                     (hermes3:8b via Ollama)
```

Three layers with distinct reliability contracts:

| Layer | Contract |
|-------|----------|
| Ingest + ranking | Deterministic, no LLM, never blocks, milliseconds |
| Agent (explain) | Lazy, async, cached; failure degrades to no-explanation |
| UI | Renders ranked list immediately; explanations stream in |

This split is itself the design opinion: deterministic pipeline where
reliability matters, LLM agent where flexibility matters, pydantic schemas as
the contracts between them.

## Project shape

- New standalone repo `~/hermes-rss`. Python 3.12, `uv`, `ruff`, `pytest`.
- Deps: `feedparser`, `sqlite-vec`, `httpx`, `scikit-learn`, `langgraph`,
  `pydantic`, `fastapi`, `uvicorn`.
- CLI entry point `hermes` (argparse or typer) with subcommands: `ingest`,
  `serve`, `eval`, `warmup`, `bootstrap-persona`.

## Data model (single file: `hermes.db`)

SQLite with `PRAGMA journal_mode=WAL` and `busy_timeout` set; the FastAPI app
uses a single writer connection (ingest may run concurrently via cron).

- `users` — id, name, `interests` (free text; the cold-start profile).
- `feeds` — id, url, title, last_fetched.
- `items` — id, feed_id, guid, title, url, summary, published, content_hash.
  - Dedup: RSS `guid`/entry-id primarily; content_hash as fallback (arXiv
    v1→v2 abstract edits change the hash but not the guid).
- `item_vectors` — sqlite-vec virtual table, 256-dim.
- `clicks` — user_id, item_id, `useful` bool, clicked_at.
- `explanations` — user_id, item_id, text, created_at (cache; never
  invalidated by re-ranking alone).

Seeded users: Matt (real), plus personas `bench-chemist` and `ml-engineer`
with hand-written interests text.

Known limitation (accepted): sqlite-vec is narrative more than necessity at
this scale — a numpy full scan would do. It costs nothing and mirrors the
pgvector patterns used in production.

## Ingest pipeline (deterministic, no LLM)

`hermes ingest`:

1. Fetch each feed with feedparser. Per-feed failures are logged and skipped —
   never fatal.
2. Dedup (guid, then content hash).
3. Clean text: strip arXiv boilerplate ("Announce Type: … Abstract:") before
   embedding.
4. Embed title + summary with embeddinggemma via Ollama, using its
   **document** task prefix; truncate 768→256 (Matryoshka) and **re-normalize
   after truncation**.
5. Store item + vector.

Idempotent and cron-able. No enrich step: the original design had a per-item
LLM "enrich" graph; review cut it — arXiv abstracts are already good text, and
300 items × 5–10s of 8B inference per daily dump would take an hour.

Feed list config: `feeds.toml` checked into the repo (url + title per feed);
`hermes ingest` syncs it into the `feeds` table. Start deliberately **diverse** (chemistry + ML + general
science + general news) so persona contrast and click-driven movement are
visually legible. A homogeneous all-cs.LG corpus makes re-ranking look like
noise.

## Ranking core

Per-user score over items from the last N days (default 14) or unseen items —
never the all-time pile.

- **Profile score**: cosine similarity between item vector and the embedded
  `interests` text (embedded with the **query** task prefix).
- **Classifier score**: per-user `LogisticRegression` over item embeddings,
  trained on that user's clicks, `class_weight='balanced'`, strong L2.
  Retrained on every click (milliseconds at this scale).
- **Guard**: if the user's clicks contain fewer than 2 classes (all-yes or
  all-no — always true for the first few clicks), skip the classifier entirely
  and use profile score alone. Never let sklearn see a single-class fit.
- **Blend by rank, not raw score** (cosine and LR probability have
  incompatible scales): compute each item's rank under both scores, combine as
  `w * classifier_rank + (1 - w) * profile_rank`, where `w = n / (n + 5)` and
  `n` = number of clicks. Smooth ramp: visible movement by click 3–4, no
  cliff at any click count.

`hermes eval`: leave-last-N-out per user, reports AUC. Kept because it's
cheap and demonstrates eval-first habits, but at ~15 clicks the number is
noise — do not present it as evidence.

`hermes bootstrap-persona <name>`: auto-label ~30 items as pseudo-clicks by
similarity to the persona's interests text (top decile → yes, bottom → no),
so personas can also demonstrate classifier behavior. The persona-switch demo
works without this (profile similarity alone); this is optional garnish.

## Agent layer (LangGraph + hermes3:8b)

One graph: **explain**. Pydantic-schema state throughout.

- Input: user_id + item_id.
- Node 1 — profile synthesis: summarize the user's clicked-useful titles into
  a short natural-language profile.
- Node 2 — explanation: one sentence, "why this is ranked here for you,"
  grounded in actual clicked titles (or the interests text pre-clicks).
- Output validated against a pydantic schema; one retry on validation
  failure, then fall back to no-explanation. **Ranking never blocks on the
  LLM.**

Serving policy:

- Explanations generated **lazily, top-20 items only, asynchronously after
  the list renders**, and cached in the `explanations` table keyed by
  (user_id, item_id). Re-ranking alone never invalidates the cache.
- Ollama: hermes3:8b (Q4_K_M, 4.7GB) with `num_ctx=8192`, `keep_alive=-1`,
  `OLLAMA_MAX_LOADED_MODELS=2` — the 8B pinned to one GPU, embeddinggemma on
  the other; both fit and serve concurrently.
- `hermes warmup`: load both models and run a token through each. Run before
  any demo — cold load is 10–20s of dead air otherwise.

## Web UI

FastAPI serving one page, no build step (htmx or vanilla JS):

- User switcher (the persona-switch demo moment).
- Ranked list: title, source, score, explanation (streams in when cached).
- ✓/✗ buttons → POST /clicks → retrain → re-rank → partial page update.

## Testing

- pytest; canned RSS XML fixtures (including an arXiv-style feed with
  boilerplate and a v1→v2 duplicate pair).
- Unit tests: dedup (guid + hash paths), single-class guard, rank-blend math
  (including w ramp values), recency window.
- LangGraph nodes tested against a stubbed Ollama client, including the
  schema-validation-failure → retry → fallback path.
- No live-network tests.

## Demo runbook (write as DEMO.md during implementation)

1. `hermes warmup` beforehand; verify both models resident.
2. Open UI as `bench-chemist`, then switch to `ml-engineer` — same feed,
   different order (zero clicks, zero LLM — most robust moment, lead with it).
3. Switch to Matt, click ✓/✗ through ~8 items on a rehearsed sequence —
   reorder visible by click 3–4.
4. Explanations fill in as garnish; never wait on them.
5. Talking points: rank-blend cold-start ramp, eval-first habit (and why AUC
   at n=15 is honest noise), what this grows into at scale (DPO/ORPO
   preference modeling, real rerankers, pgvector).
