# Demo runbook

Rehearse once end-to-end before the call.

## Before (10 min prior)

1. `export OLLAMA_MAX_LOADED_MODELS=2` (in the shell that starts ollama serve,
   or systemd override) — prevents the embedder evicting the chat model.
2. `uv run attest warmup` — both models resident; re-run until it returns fast.
3. `uv run attest ingest` — fresh items.
4. `uv run attest serve` — open http://127.0.0.1:8899, confirm list renders.
5. Reset demo state if needed: delete matt's clicks
   ```bash
   uv run python -c "from attestation.db import get_db, resolve_db_path; c = get_db(resolve_db_path(None)); c.execute(\"DELETE FROM clicks WHERE user_id=(SELECT id FROM users WHERE name='matt')\"); c.commit()"
   ```
   (uses the same `resolve_db_path()` precedence as the CLI, so it hits
   whichever db `attest serve` is actually using -- see `src/attestation/db.py`.)

## The show (in this order)

0. **The front door — Hermes Agent** (Nous Research's open agent framework, with
   our engine packaged as a skill; see Task 9 / `src/attestation/skills/research-provenance/`):
   Launch with the exact invocation below, then use the rehearsed prompt — do
   NOT ad-lib vague phrasing (see the tool-calling caveats below).

   Launch. Two settings must already be in place or the model reports that
   the tools do not exist — see README "Required for local models":
   `OLLAMA_CONTEXT_LENGTH=32768` in the Ollama systemd unit, and
   `tools.tool_search.enabled: off` in `~/.hermes/config.yaml`.

   ```bash
   cd /home/matt/attestation && hermes chat \
     --provider local-ollama --model gemma4:e2b-it-q4_K_M
   ```

   (`hermes3:8b` was the old recommendation. It emitted a malformed tag on
   ~40% of items and loops on the tool bridge; gemma4:e2b calls the tools
   cleanly. The `reasoning_overrides` workaround below is hermes3-only.)

   Rehearsed prompt (verified to reliably trigger a real terminal tool call —
   do not substitute a vaguer phrasing like "what's in my feed?", which
   reliably produced a hallucinated text answer with zero tool calls in
   testing). Pin `"background": false` in the arguments — without it the 8B
   sometimes backgrounds the curl and never sees the output. Also avoid
   over-constraining the reply format ("your entire reply must be ONLY...") —
   in testing that reliably flipped the model into hallucinating a sample list
   with zero tool calls:

   > Use your terminal tool with EXACTLY these arguments:
   > `{"command": "curl -s --max-time 10 'http://127.0.0.1:8899/list?user=matt'", "background": false}`.
   > Wait for the output, then
   > summarize the titles you see in the output.

   The agent fires a real `terminal` tool call, hits our local API, and
   presents the ranked feed. Then, to mark an item useful, again be
   literal rather than vague — say which endpoint/command to run, e.g.:

   > Call the terminal tool with command set to: `curl -s --max-time 10 -X
   > POST "http://127.0.0.1:8899/clicks" -d user=matt -d item_id=<ID> -d
   > useful=1` (substitute `<ID>` for the item you want to mark).

   That's the founder's product shape live: Hermes Agent orchestrating a
   domain recommendation engine.

   **Tool-calling pre-flight (both required, or this act fails):**
   - `OLLAMA_CONTEXT_LENGTH=32768` set when Ollama starts (the <24GB-VRAM
     default silently truncates tool schemas — verify with `ollama ps`;
     this box has no passwordless sudo for the systemd override, so if
     unset, the demo still works with the model's smaller loaded context
     since this skill's schema is small — but pin it for headroom if you can:
     `sudo systemctl edit ollama` → `Environment="OLLAMA_CONTEXT_LENGTH=32768"`).
   - **Reasoning must be off for `hermes3:8b`** — hermes-agent's default
     `agent.reasoning_effort: medium` sends a thinking parameter that
     Ollama rejects for this model with `HTTP 400: "hermes3:8b" does not
     support thinking`, which aborts every tool-calling turn immediately.
     Persisted fix already applied in `~/.hermes/config.yaml`:
     ```yaml
     agent:
       reasoning_overrides:
         hermes3:8b: none
     ```
     If running against a `~/.hermes/config.yaml` that doesn't have this
     yet, pass `--reasoning none` on the command line as a fallback.
   - Driver model: `hermes3:8b`, not `gemma4:12b` — `gemma4:12b` partially
     CPU-offloads on this box's 2x GTX 1080 8GB and took ~94s/response in
     testing (see Smoke test log below); `hermes3:8b` is both fast (~5-15s)
     and on the docs' known-good tool-calling list.
   - **Prompt discipline**: hermes3:8b's tool-calling is real (confirmed via
     verbose `-v` logs showing actual `Tool call:` entries and executed
     results, not text) but prompt-sensitive. Vague natural questions
     reliably produced either a hallucinated plain-text answer with zero
     tool calls, or malformed tool-call arguments. Explicit, literal
     prompts (name the tool, state the exact command) reliably worked —
     use the rehearsed phrasing above, don't improvise on the fly.
1. **Persona switch** (zero clicks, zero LLM): open the web UI as `bench-chemist`,
   then `ml-engineer`. Same feed, different order, driven by profile embeddings alone.
2. **Live learning**: switch to `matt`, click ✓ on 3-4 on-topic items and ✗ on
   3-4 off-topic ones on a rehearsed sequence. Reorder is visible by click 3-4
   (blend weight w = n/(n+5)).
3. **Explanations as garnish**: point at the italics filling in asynchronously —
   the LangGraph explain graph calling a local model on a GTX 1080, cached per
   (user, item), and the feed never waits on it. The orchestrator's reliability
   contract, visible.

## Talking points

- Deterministic core / LLM garnish split = reliability contracts per layer.
- Cold start: interests text -> profile embedding; ramps smoothly into the
  classifier as clicks arrive. No cliff.
- `attest eval` exists and reports honest noise at n=15 — eval-first habit,
  not decorative metrics.
- What this grows into at scale: preference-optimization post-training
  (DPO/ORPO), learned rerankers, pgvector. Same shapes, bigger substrate.

## Failure modes

- Explanations blank → Ollama down or model evicted; feed still works. Say so:
  that's the reliability contract doing its job.
- Feed empty → recency window; run `uv run attest ingest`.

## Smoke test log (2026-08-04)

Live end-to-end smoke test run against real feeds and a real local Ollama
instance. See `.superpowers/sdd/2026-08-04-attestation/task-8-report.md` for
the full command-by-command transcript. Summary:

- Chat model settled on for demo use: **`CHAT_MODEL=hermes3:8b`**.
  `gemma4:12b` (the default) is functional on this box's 2x GTX 1080 8GB but
  took ~94s for a single `/explanation` call (partial CPU offload) — over the
  60s "painfully slow" threshold. Switching to `hermes3:8b` brought explain
  latency down to ~5.7s on a warm model. For the "explanations as garnish"
  demo beat, export `CHAT_MODEL=hermes3:8b` before `attest warmup` and
  `attest serve`.
- `uv run attest ingest` on this run: `{'added': 889, 'skipped': 616,
  'failed_feeds': 0}` against all 7 configured feeds — no feed failures.
- Persona switch (`bench-chemist` vs `matt`) confirmed to produce different
  orderings from profile-embedding cosine similarity alone (verified via
  `/list?user=...` item ordering).
- A real click (`useful=1`) via `POST /clicks` returned 200 and the item was
  removed from both the re-rendered `/list` fragment and a subsequent fresh
  `/list` fetch.
- `GET /explanation` returned a one-sentence explanation via the real
  structured-output path through Ollama (see model/latency note above).
- `bootstrap-persona` writes 30 pseudo-clicks per persona by default
  (`bench-chemist`: 30, `ml-engineer`: 30).

## Hermes Agent (Act 0) smoke test log (2026-08-04, fix round 1)

Live smoke test of Act 0 through Hermes Agent v0.20.0, `hermes3:8b` via
Ollama custom endpoint. Full transcripts and divergences in
`.superpowers/sdd/2026-08-04-attestation/task-9-report.md`. Summary:

- First finding: `hermes-agent`'s default `agent.reasoning_effort: medium`
  crashes every tool-calling turn against `hermes3:8b` on Ollama with
  `HTTP 400: "hermes3:8b" does not support thinking`. Fixed by persisting
  `agent.reasoning_overrides: {hermes3:8b: none}` in `~/.hermes/config.yaml`
  (a per-model override, doesn't touch other providers/models). Re-verified
  with a `hermes chat` invocation carrying **no** `--reasoning` flag: agent
  init succeeded, a real `terminal` tool call fired
  (`curl ... /list?user=matt`), and the response returned actual feed HTML
  with no HTTP 400 — confirmed via `-v` debug logs.
- Second finding: vague prompts ("what's in my feed today?", "list your
  skills") are unreliable — the model frequently answers from imagination
  with zero tool calls, or emits tool calls with fabricated argument names.
  Explicit, literal prompts (name the tool, state the exact command)
  reliably triggered real tool calls for both the feed fetch (`GET /list`)
  and the click (`POST /clicks`, DB-verified `useful=1` persisted). Use the
  rehearsed phrasing in Act 0 above, not ad-libbed phrasing.
