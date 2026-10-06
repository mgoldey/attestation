# NVIDIA NIM embeddings

**Date:** 2026-10-06
**Status:** approved by request ("we need to add embedding support based on
nvidia nim", 2026-10-06); measured before written.
**Depends on:** the LLM client extraction (`2026-08-05-llm-client-extraction-design.md`:
`EmbeddingClient`, `EMBED_BASE_URL`, `EMBED_API_KEY`), the embedding-model
guard (migration 010, `db._ensure_vec_tables`), and the hosted-models install
step.

## Why

The AgentMarkit research assistant embeds on its own CPU through a local
Ollama serving `embeddinggemma`. On 2026-10-06 that agent told its owner
personalized ranking was unavailable: the machine had ~647 MB free, it had
pulled `phi3:mini` (~5 GB) trying to fix things itself, and ingest failed with
`KeyError: 'data'` -- `EmbeddingClient` indexing a 200 response that was not
OpenAI-shaped. A hosted embedder takes the model off a 4 GB machine
entirely. NVIDIA NIM speaks the OpenAI `/v1/embeddings` shape, so
`EMBED_BASE_URL=https://integrate.api.nvidia.com/v1` *nearly* works today.
"Nearly" is the trap this spec is about.

## Measured (2026-10-06, this account's key)

Seven embedding models are listed at `/v1/models`; five answer 404 ("not
found for account"). The two that answer:

| model | no `input_type` | `dimensions=256` | input > 8192 tokens |
|---|---|---|---|
| `nvidia/llama-nemotron-embed-vl-1b-v2` | **400** "input_type parameter is required for asymmetric models" | 200 | 400 unless `truncate: END` |
| `nvidia/nemotron-3-embed-1b` | **200** | 400 ("must be one of 2048") | 400 unless `truncate: END` |

Relevance AUC on `evals/reaction_cases.json` (100 labelled persona x item
cases; "hard" = the 20 hand-written adjacent/bait/terse/crossover cases),
interests as query, title + summary as document, cosine after attestation's
own `truncate_normalize`:

| configuration | all | hard |
|---|---|---|
| nemotron-3, `input_type`, raw text, 1024 | 0.994 | 0.979 |
| nemotron-3, `input_type`, raw text, 768 | 0.989 | 0.979 |
| nemotron-3, `input_type`, raw text, 256 | 0.962 | 0.938 |
| nemotron-3, **no** `input_type`, raw text, 256 | **0.625** | 0.667 |
| nemotron-3, **no** `input_type`, gemma prefixes, 256 | 0.740 | 0.573 |
| vl-1b-v2, `input_type`, raw text, 2048 | 0.958 | 0.875 |
| vl-1b-v2, `input_type`, raw text, 256 | 0.903 | 0.688 |

The failure this design exists to prevent is the fourth row: `nemotron-3`
**accepts** a request without `input_type`, answers 200 with a 2048-wide
vector, and ranks barely better than chance. A plain OpenAI-compatible
hookup passes every install check and produces a feed that is quietly bad --
the "worked" recorded in memory on 2026-09-11 was exactly that.

## Decisions

1. **`input_type` is sent when the embedder is NIM.** `llm.embed_input_types()`
   reads `EMBED_INPUT_TYPE` (`1/true/on`, `0/false/off`); unset, it is on
   exactly when `EMBED_BASE_URL`'s host ends in `api.nvidia.com`. A
   self-hosted NIM on another host sets `EMBED_INPUT_TYPE=1`. Read once, at
   `EmbeddingClient` construction, like every other client setting.
2. **When on, every request also sends `truncate: "END"`.** Without it a long
   abstract or a full-text window is a 400 that abandons the whole ingest pass.
3. **When on, `Embedder` sends raw text, not the gemma prompt templates.**
   `DOC_PROMPT`/`QUERY_PROMPT` are embeddinggemma's task prefixes; NIM's
   asymmetry is `input_type=passage|query` instead. Documents are
   `"{title}\n{text}"`. The prefixes measurably did not help either NIM model.
4. **A response with no `data` raises a `ValueError` naming the server and
   quoting the body**, instead of `KeyError: 'data'`, which an agent cannot act on.
5. **No default model change.** `EMBED_MODEL` stays `embeddinggemma` for the
   local path. The documented NIM configuration is
   `EMBED_MODEL=nvidia/nemotron-3-embed-1b`, `EMBED_DIMS=1024` (the AUC plateau;
   4 KB a vector), `EMBED_API_KEY=<nvapi key>`.
6. **`attest reembed` switches an existing database to a new embedder in
   place.** Today a model or width change is refused with "re-ingest into a
   fresh database", which on an installed machine throws away personas,
   clicks and the library. `reembed` drops the two vec0 tables and their
   `embedding_model` rows, lets `get_db` recreate them at the configured
   width and model, then re-embeds every item (batched, `EMBED_BATCH_SIZE`)
   and every reference (the library's existing missing-vector pass). Nothing
   relational is touched. It refuses to start unless the configured embedder
   answers, so a dead endpoint never leaves an empty index behind.
7. **The install probe sends `input_type`** (`query`) through the same client,
   so `attest install` checks the request ingest will actually make.

## Not in scope

- AgentMarkit's install writing the NIM configuration and where the key comes
  from (the owner's own `nvapi-` key or AgentMarkit's). That is a product
  decision on the AgentMarkit side; this spec makes attestation ready for either.
- Choosing the chat model. Only embeddings move.
