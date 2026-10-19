# Cite support: does the cited paper say what the sentence says?

**Date:** 2026-10-17
**Status:** implemented on `feat/cite-support` (roadmap step 4), not yet reviewed.
**Depends on:** the library store (`library.lookup_row`, `reference_fulltext`),
`embed.Embedder`, `llm.default_chat_fn`, `citations.Resolver`.

## The gap

`cite.check` says a key resolves. Its payload says, in every reply, that this is
not "the paper backs this", because a 2B model read it that way. Nothing answered
the real question. The failure that makes it worth answering is the one a reader
does not notice:

    draft:  NequIP outperforms invariant models by up to three orders of magnitude
    source: NequIP outperforms existing models with up to three orders of
            magnitude fewer training data

The number is right. What it measures is not.

## Design

`support.py` (orchestration, verdicts) and `support_text.py` (pure text: sentence
spans, citation syntax, quantities). Kept out of `claims.py`, whose verdicts are
deterministic by contract and which another branch is extending for LaTeX.

1. **Citing sentences.** Markdown: Pandoc `[@a; @b]` and `@key`, plus a claim
   annotation's `cite=` attached to the sentence ending just before it (the
   `examples/citations/DRAFT.md` convention). LaTeX: `\cite`-family commands with
   optional notes. Code fences and comments are blanked with offsets kept, so
   line numbers stay true.
2. **Source text.** Library row by identity/DOI/arXiv id/bib key (a DataCite
   arXiv DOI is unwrapped; a key only a `.bib` knows is retried by its DOI or
   arXiv id). Best text: `reference_fulltext` (abstract prepended when the body
   lacks it) > abstract > title. The level is reported on every result.
3. **Retrieval.** Source sentences (offset spans) ranked by content-word overlap,
   top 12 reranked by embedding similarity when the embedder answers. A dead
   embedder is tried once per run.
4. **Judge.** The top 3 passages go to the local chat model with a schema that
   asks for passage, quote, verdict, reason in that order (evidence before
   verdict). Labels are four short words with one-line definitions. The prompt is
   rendered by one function (`judge_messages`) shared with the eval.
5. **Deterministic checks**, run whether or not a model answers:
   - *quantities* -- numbers normalised to value + coarse unit (`%`, `x` with
     orders of magnitude as powers of ten, a hyphen suffix like `four-body`, or
     bare). For each number in the sentence, every source occurrence is checked
     for the phrase that follows it (stopping at scope words such as "on",
     "over", "compared") and for names the sentence pairs with it that the source
     pairs with a different number. Same number, different phrase or name:
     `meaning`. Same unit, different number: `value`. Either makes the verdict
     `contradicted` **and overrules the model**.
   - *gaps* -- a number or name in the sentence that the source never states
     caps `supported` at `partially_supported`.
   - *negation* -- a negating word whose clause in the best passage holds a
     third of the sentence's content words, when the sentence negates nothing,
     caps `supported` at `partially_supported`. Measured: gemma4:e2b answered
     SUPPORTS for "networks require massive training sets" against "challenging
     the belief that networks require massive training sets".
   - *title only* caps `supported` at `partially_supported`.
   Without a model, the verdict is overlap thresholds (0.6 / 0.34) plus the above.
6. **Quotes** are spans of the source, or the model's quote located in the source
   (exact, then case/whitespace-insensitive, returning the source's characters),
   else the passage the model pointed at. `quote_verified` is the result of an
   `in` check, never assumed.

Verdicts: `supported`, `partially_supported`, `contradicted`, `not_found`,
`source_unavailable`.

## Surfaces

- `attest support PATH | --sentence S --cite KEY [--judge auto|llm|none]
  [--fail-on contradicted,...] [--json] [--limit N]`. Exits 1 when a verdict in
  `--fail-on` occurs (default `contradicted`, like `attest claims`).
- `cite.support(path | sentence+cite, judge)` on the `provenance` surface (beside
  `cite.check`, same argument) and `knowledge` (all `cite.*`). At most 20
  citations judged per call, 4 rows returned worst first (every row field is capped;
  the worst case is measured under the 7000-char ceiling); tool count 50 -> 51. Its own
  module, `mcp/citation_support.py`: `citation.py` is at its size cap.

## Router-only sessions

A deployed Hermes session serves only `runs.ask`/`kg.ask` and the disclosure
tool, so `mcp/routing_support.py` gives both routers a rule checked before their
own tables: a support/say/back-up phrase plus a word naming a cited source
(paper, source, citation, reference, sentence), or a quoted sentence plus a
named key; never when the question mentions runs, the ledger or arms. `path=`
checks a draft; otherwise the sentence and key come out of the question, and a
question carrying neither is asked back. The skills name the routers, not
`cite.support`, so the hidden-tool ratchet does not move.

## Measured (2026-10-17, evals/support_cases.json: 6 real arXiv abstracts, 20 sentences)

| judge | exact verdict | contradicted recall | contradicted called supported | supported called contradicted |
|---|---|---|---|---|
| deterministic | 15/20 | 4/7 | 0 | 0 |
| gemma4:e2b, no guards | 16/20 | 4/7 | 2 | 0 |
| gemma4:e2b + guards + prompt (2 runs) | 17/20, 17/20 | 5/7, 5/7 | 0, 0 | 0, 0 |

The motivating case is `contradicted` in every row. n=20, few runs:
read these as "the guards remove the dangerous error", not as an accuracy figure.

## Not done

- No reranker model: the judge reads 3 passages; a cross-encoder would need a new
  dependency or server.
- Abstract-level `not_found` is weak evidence; nothing fetches full text on demand.
- Multi-key sentences are judged per key, never "supported jointly".
